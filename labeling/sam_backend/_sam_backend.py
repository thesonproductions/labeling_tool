# -*- coding: utf-8 -*-
"""
=============================================================================
Label Studio ML Backend - Segment Anything Model (SAM / MobileSAM)
Tích hợp Interactive Segmentation với cơ chế Box Prompt
=============================================================================
Quy trình hoạt động:
1. Người dùng vẽ Bounding Box (RectangleLabels) trên giao diện Label Studio.
2. Label Studio gửi tọa độ Box Prompt và thông tin ảnh tới ML Backend qua context.
3. ML Backend chuyển đổi tọa độ phần trăm (%) sang pixel thực tế và truyền vào SAM.
4. SAM suy luận và xuất ra Binary Mask khớp với đường bao đối tượng.
5. OpenCV trích xuất contour, xấp xỉ đa giác (approxPolyDP), chuẩn hóa về 0-100%
   và trả về danh sách các điểm PolygonLabels hiển thị tức thì trên web.
6. Hỗ trợ cơ chế LRU Cache lưu trữ image_embedding theo từng task_id, giúp các
   lần vẽ tiếp theo trên cùng một ảnh phản hồi tức thì (< 30ms) mà không cần
   chạy lại Image Encoder nặng nề.
=============================================================================
"""

import os
import sys
import time
import logging
import argparse
import traceback
from collections import OrderedDict
from typing import List, Dict, Optional, Tuple, Any

import cv2
import numpy as np
import requests
import torch

# Thư viện Label Studio ML
from label_studio_ml.model import LabelStudioMLBase

# Cấu hình Logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger("SAMBackend")

# Trọng số mặc định và URL tải tự động nếu thiếu
MOBILESAM_DEFAULT_WEIGHTS_URL = "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
MOBILESAM_BACKUP_WEIGHTS_URL = "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"


class EmbeddingLRUCache:
    """
    Bộ nhớ đệm LRU lưu trữ Image Embeddings của SAM theo từng task_id/image.
    Giúp phản hồi dưới 30ms khi vẽ liên tiếp nhiều Box Prompt trên cùng một ảnh.
    """
    def __init__(self, capacity: int = 10):
        self.capacity = capacity
        self.cache: OrderedDict[str, Dict[str, Any]] = OrderedDict()

    def get(self, key: str) -> Optional[Dict[str, Any]]:
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Dict[str, Any]):
        if key in self.cache:
            self.cache.move_to_end(key)
        self.cache[key] = value
        if len(self.cache) > self.capacity:
            oldest_key, _ = self.cache.popitem(last=False)
            logger.debug(f"[CACHE EVICT] Đã giải phóng cache task cũ: {oldest_key}")

    def clear(self):
        self.cache.clear()


class SAMBackend(LabelStudioMLBase):
    """
    ML Backend tích hợp MobileSAM cho Label Studio Interactive Segmentation.
    """

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_t",
        device: Optional[str] = None,
        cache_size: int = 10,
        approx_epsilon: float = 0.002,
        min_contour_area: float = 15.0,
        **kwargs
    ):
        super(SAMBackend, self).__init__(**kwargs)
        # KHẮC PHỤC LỖI: Đảm bảo model_dir luôn là chuỗi đường dẫn hợp lệ, không bao giờ là None
        if not getattr(self, "model_dir", None):
            self.model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models_cache")
        os.makedirs(self.model_dir, exist_ok=True)
        
        self.model_version = f"sam-{model_type}"
        self.approx_epsilon = float(os.getenv("APPROX_EPSILON", approx_epsilon))
        self.min_contour_area = float(os.getenv("MIN_CONTOUR_AREA", min_contour_area))

        # 1. Tự động nhận diện thiết bị tính toán (GPU / CPU)
        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        logger.info(f"[DEVICE] Thiết bị tính toán được chọn: {self.device}")

        # 2. Định vị file trọng số weights/mobile_sam.pt
        weights_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weights")
        os.makedirs(weights_dir, exist_ok=True)

        if checkpoint is None:
            checkpoint = os.getenv("CHECKPOINT_PATH")
        if not checkpoint:
            checkpoint = os.path.join(weights_dir, "mobile_sam.pt")

        self.checkpoint_path = checkpoint
        self.model_type = os.getenv("MODEL_TYPE", model_type)

        # 3. Nạp mô hình MobileSAM
        self.predictor = self._load_sam_predictor(self.checkpoint_path, self.model_type, self.device)

        # 4. Khởi tạo LRU Cache
        self.embedding_cache = EmbeddingLRUCache(capacity=cache_size)
        self.current_cache_key: Optional[str] = None
        self._processed_box_ids: Dict[Any, set] = {} #FIX

        # 5. Phân tích cấu hình thẻ nhãn ban đầu
        self._parse_config_tags()

    def setup(self):
        """Được gọi tự động mỗi khi Label Studio kết nối hoặc cập nhật XML mới."""
        super(SAMBackend, self).setup()
        self._parse_config_tags()
        logger.info("[SETUP SYNC] Đã đồng bộ cấu hình XML mới nhất từ Label Studio thành công!")

    def _ensure_checkpoint_exists(self, checkpoint_path: str):
        """Tự động tải file trọng số nếu chưa có."""
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000:
            return

        logger.info(f"[WEIGHTS] Trọng số chưa tồn tại tại {checkpoint_path}. Bắt đầu tải...")
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        urls = [MOBILESAM_DEFAULT_WEIGHTS_URL, MOBILESAM_BACKUP_WEIGHTS_URL]
        downloaded = False

        for url in urls:
            try:
                logger.info(f"[DOWNLOAD] Tải từ nguồn: {url}")
                req = requests.get(url, stream=True, timeout=60, headers={"User-Agent": "Mozilla/5.0"})
                req.raise_for_status()

                with open(checkpoint_path, "wb") as f:
                    for chunk in req.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)

                if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000:
                    logger.info("[WEIGHTS] Tải trọng số MobileSAM hoàn tất!")
                    downloaded = True
                    break
            except Exception as e:
                logger.warning(f"[DOWNLOAD FAILED] Nguồn {url} lỗi: {e}")

        if not downloaded:
            raise RuntimeError(f"Không thể tải trọng số MobileSAM. Vui lòng tải thủ công vào: {checkpoint_path}")

    def _load_sam_predictor(self, checkpoint: str, model_type: str, device: str):
        """Khởi tạo và nạp trọng số MobileSAM / SAM vào SamPredictor."""
        self._ensure_checkpoint_exists(checkpoint)

        try:
            from mobile_sam import sam_model_registry, SamPredictor
            logger.info(f"[MODEL] Đang nạp MobileSAM ('{model_type}') từ '{checkpoint}'...")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            sam.eval()
            predictor = SamPredictor(sam)
            logger.info("[MODEL] Khởi tạo MobileSAM thành công!")
            return predictor
        except ImportError:
            logger.warning("[MODEL] Gói mobile_sam chưa có, thử dùng segment_anything...")
        except Exception as e:
            logger.warning(f"[MODEL] Nạp mobile_sam lỗi: {e}, thử dùng segment_anything...")

        try:
            from segment_anything import sam_model_registry, SamPredictor
            if model_type == "vit_t":
                model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            sam.eval()
            predictor = SamPredictor(sam)
            logger.info("[MODEL] Khởi tạo SAM tiêu chuẩn thành công!")
            return predictor
        except Exception as e:
            raise RuntimeError(f"Không thể nạp mô hình SAM: {e}")

    def _parse_config_tags(self):
        """Tự động phân tích các thẻ XML (Image, RectangleLabels, PolygonLabels)."""
        self.image_tag_name = os.getenv("IMAGE_TO_NAME", "image")
        self.polygon_tag_name = os.getenv("POLYGON_FROM_NAME", "polygon")
        self.bbox_tag_name = os.getenv("BBOX_FROM_NAME", "label")
        self.image_value_key = "image"

        if hasattr(self, "parsed_label_config") and self.parsed_label_config:
            for tag_name, tag_info in self.parsed_label_config.items():
                tag_type = tag_info.get("type", "").lower()
                if tag_type == "image":
                    self.image_tag_name = tag_name
                    self.image_value_key = tag_info.get("value", "image")
                elif tag_type == "polygonlabels":
                    self.polygon_tag_name = tag_name
                elif tag_type in ["rectanglelabels", "rectangle"]:
                    self.bbox_tag_name = tag_name

        logger.info(
            f"[CONFIG TAGS] Image='{self.image_tag_name}', "
            f"BBox Tag='{self.bbox_tag_name}', Polygon Tag='{self.polygon_tag_name}'"
        )

    def _get_auth_token(self) -> Optional[str]:
        """Tự động tìm kiếm API Token của Label Studio."""
        token = getattr(self, "access_token", None) or os.getenv("LABEL_STUDIO_API_KEY")
        if token:
            return token

        # Tìm trong các file token.txt dự phòng
        candidate_tokens = [
            os.path.join(os.path.dirname(__file__), "token.txt"),
            os.path.join(os.path.dirname(__file__), "..", "token.txt"),
            os.path.join(os.path.dirname(__file__), "..", "..", "token.txt"),
            os.path.join(os.getcwd(), "token.txt"),
            os.path.join(os.getcwd(), "labeling", "token.txt"),
        ]
        for path in candidate_tokens:
            if os.path.exists(path):
                try:
                    with open(path, "r", encoding="utf-8") as f:
                        t = f.read().strip()
                        if t:
                            return t
                except Exception:
                    pass
        return None

    def _resolve_image_rgb(self, image_url: str, task: Dict[str, Any]) -> Tuple[np.ndarray, str]:
        """
        Nạp ảnh thông minh: Thử quét ổ đĩa cục bộ, nếu là URL thì tự động tải kèm Token.
        """
        # 1. Đường dẫn tệp tồn tại trực tiếp trên ổ đĩa
        if os.path.exists(image_url):
            img = cv2.imread(image_url)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), image_url

        # 2. Tìm trong thư mục upload cục bộ của Label Studio
        candidate_roots = [
            os.path.join(os.getcwd(), "labeling", "data", "media", "upload"),
            os.path.join(os.getcwd(), "labeling", "data", "media"),
            os.path.join(os.getcwd(), "data", "media", "upload"),
            os.path.join(os.path.dirname(__file__), "..", "data", "media", "upload"),
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "label-studio", "label-studio", "media", "upload"),
            os.path.join(os.path.expanduser("~"), ".local", "share", "label-studio", "media", "upload"),
        ]

        clean_subpath = image_url
        if "/data/upload/" in image_url:
            clean_subpath = image_url.split("/data/upload/")[-1]
        elif "/data/" in image_url:
            clean_subpath = image_url.split("/data/")[-1]

        for root in candidate_roots:
            full_path = os.path.normpath(os.path.join(root, clean_subpath))
            if os.path.exists(full_path):
                img = cv2.imread(full_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), full_path

        # 3. Thử qua get_local_path của thư viện gốc
        try:
            local_path = self.get_local_path(image_url, task_id=task.get("id"))
            if local_path and os.path.exists(local_path):
                img = cv2.imread(local_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), local_path
        except Exception:
            pass

        # 4. Tải qua HTTP/HTTPS từ chính server Label Studio
        token = self._get_auth_token()
        headers = {"Authorization": f"Token {token}"} if token else {}

        possible_hosts = [
            getattr(self, "hostname", None),
            os.getenv("LABEL_STUDIO_HOST"),
            "http://localhost:19090",
            "http://127.0.0.1:19090",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        target_urls = []
        if image_url.startswith("http://") or image_url.startswith("https://"):
            target_urls.append(image_url)
        elif image_url.startswith("/"):
            for h in possible_hosts:
                if h:
                    target_urls.append(f"{h.rstrip('/')}{image_url}")

        for target in target_urls:
            try:
                resp = requests.get(target, headers=headers, timeout=12)
                if resp.status_code == 200:
                    nparr = np.frombuffer(resp.content, np.uint8)
                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if img is not None:
                        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), target
            except Exception:
                continue

        raise FileNotFoundError(f"Không thể tìm thấy hoặc tải ảnh từ: {image_url}")

    def _sync_image_embedding(self, task_id: Any, image_key: str, image_rgb: np.ndarray):
        """Đồng bộ Image Embedding qua LRU Cache."""
        cache_key = f"{task_id}_{image_key}"

        if self.current_cache_key == cache_key and getattr(self.predictor, "is_image_set", False):
            return

        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            for attr_name, attr_val in cached.items():
                if attr_val is not None:
                    setattr(self.predictor, attr_name, attr_val)
            self.predictor.is_image_set = True
            self.current_cache_key = cache_key
            logger.info(f"[CACHE HIT] Tái sử dụng Image Embedding cho task {task_id}")
            return

        logger.info(f"[CACHE MISS] Đang tạo Image Embedding mới cho task {task_id}...")
        t0 = time.time()
        with torch.no_grad():
            self.predictor.set_image(image_rgb)

        cache_state = {
            "features": self.predictor.features,
            "orig_h": getattr(self.predictor, "orig_h", None),
            "orig_w": getattr(self.predictor, "orig_w", None),
            "input_h": getattr(self.predictor, "input_h", None),
            "input_w": getattr(self.predictor, "input_w", None),
            "original_size": getattr(self.predictor, "original_size", None),
            "input_size": getattr(self.predictor, "input_size", None),
        }
        self.embedding_cache.put(cache_key, cache_state)
        self.current_cache_key = cache_key
        logger.info(f"[EMBEDDING DONE] Hoàn thành sau {time.time() - t0:.3f}s")

    def _mask_to_polygons(
        self,
        binary_mask: np.ndarray,
        img_width: int,
        img_height: int,
        label_name: str,
        score: float
    ) -> List[Dict[str, Any]]:
        """Chuyển mask sang đa giác Polygon chuẩn hóa [0, 100]% cho Label Studio."""
        contours, _ = cv2.findContours(
            binary_mask.astype(np.uint8),
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )
        # 1. Làm sạch nhãn một lần duy nhất trước khi lặp qua các đường viền
        if isinstance(label_name, (list, tuple)):
            clean_label = str(label_name[0]) if len(label_name) > 0 else "Object"
        else:
            clean_label = str(label_name) if label_name else "Object"

        results = []
        for contour in contours:
            if cv2.contourArea(contour) < self.min_contour_area:
                continue

            epsilon = self.approx_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            pts = approx.reshape(-1, 2)

            if len(pts) < 3:
                continue

            points_pct = []
            for px, py in pts:
                norm_x = round(float(px) / float(img_width) * 100.0, 2)
                norm_y = round(float(py) / float(img_height) * 100.0, 2)
                norm_x = max(0.0, min(100.0, norm_x))
                norm_y = max(0.0, min(100.0, norm_y))
                points_pct.append([norm_x, norm_y])

            results.append({
                "from_name": self.polygon_tag_name,
                "to_name": self.image_tag_name,
                "type": "polygonlabels",
                "value": {
                    "points": points_pct,
                    "polygonlabels": [clean_label]
                },
                "score": round(float(score), 4)
            })

        return results

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """Bắt sự kiện vẽ Bounding Box và chỉ trả về viền Polygon cho chiếc hộp mới nhất."""
        if context is None:
            try:
                from flask import request
                if request and request.is_json:
                    req_json = request.get_json(silent=True) or {}
                    context = req_json.get("context") or (req_json.get("params") or {}).get("context")
            except Exception:
                pass

        if not context or not context.get("result"):
            logger.info("[PREDICT] Không có thao tác vẽ box nào được gửi kèm.")
            return [{"result": [], "model_version": self.model_version}]

        # 1. Lọc toàn bộ Bounding Box có trong context
        all_boxes = [
            item for item in context.get("result", [])
            if item.get("type") in ["rectanglelabels", "rectangle"] and "value" in item
        ]

        if not all_boxes or not tasks:
            logger.info("[PREDICT] Không tìm thấy Bounding Box hợp lệ.")
            return [{"result": [], "model_version": self.model_version}]

        task = tasks[0]
        task_id = task.get("id", "task_0")

        # 2. CHỈ LẤY CHIẾC HỘP CUỐI CÙNG (Hộp mới nhất vừa vẽ)
        # Tuyệt đối không lặp qua các hộp cũ để tránh biến các Bounding Box cũ thành Segment!
        target_box = all_boxes[-1]
        box_prompts = [target_box]

        logger.info(f"[PREDICT] Bỏ qua {len(all_boxes) - 1} box cũ, chỉ chạy SAM cho box mới nhất ID: {target_box.get('id')}")

        image_url = task.get("data", {}).get(self.image_value_key) or task.get("data", {}).get("image")
        if not image_url:
            logger.error(f"[PREDICT] Task {task_id} thiếu link ảnh!")
            return [{"result": [], "model_version": self.model_version}]

        # 3. Nạp ảnh RGB
        try:
            image_rgb, image_ref = self._resolve_image_rgb(image_url, task)
        except Exception as e:
            logger.error(f"[PREDICT ERROR] Không thể mở ảnh: {e}")
            logger.error(traceback.format_exc())
            return [{"result": [], "model_version": self.model_version}]

        img_height, img_width = image_rgb.shape[:2]

        # 4. Đồng bộ Image Embedding (LRU Cache)
        self._sync_image_embedding(task_id, image_ref, image_rgb)

        polygon_results = []
        scores_list = []

        # 5. Suy luận duy nhất cho chiếc hộp mới này
        for box_item in box_prompts:
            val = box_item["value"]
            x_pct, y_pct = val["x"], val["y"]
            w_pct, h_pct = val["width"], val["height"]

            labels = val.get("rectanglelabels") or val.get("labels") or ["Object"]
            label_name = labels[0] if labels else "Object"

            x_min = (x_pct / 100.0) * img_width
            y_min = (y_pct / 100.0) * img_height
            x_max = ((x_pct + w_pct) / 100.0) * img_width
            y_max = ((y_pct + h_pct) / 100.0) * img_height

            x_min = max(0.0, min(float(img_width - 1), x_min))
            y_min = max(0.0, min(float(img_height - 1), y_min))
            x_max = max(x_min + 1.0, min(float(img_width), x_max))
            y_max = max(y_min + 1.0, min(float(img_height), y_max))

            box_prompt_np = np.array([x_min, y_min, x_max, y_max])
            logger.info(f"[SAM INFERENCE] Đang xử lý '{label_name}': [{x_min:.1f}, {y_min:.1f}, {x_max:.1f}, {y_max:.1f}]")

            t0 = time.time()
            with torch.no_grad():
                masks, scores, _ = self.predictor.predict(
                    box=box_prompt_np,
                    multimask_output=False
                )
            logger.info(f"[SAM SUCCESS] Suy luận xong trong {(time.time() - t0)*1000:.1f}ms - Score: {scores[0]:.4f}")

            binary_mask = masks[0] > 0.0
            scores_list.append(float(scores[0]))

            polys = self._mask_to_polygons(
                binary_mask=binary_mask,
                img_width=img_width,
                img_height=img_height,
                label_name=label_name,
                score=float(scores[0])
            )
            polygon_results.extend(polys)

        avg_score = float(np.mean(scores_list)) if scores_list else 1.0

        return [{
            "result": polygon_results,
            "score": round(avg_score, 4),
            "model_version": self.model_version
        }]


# =============================================================================
# TÍCH HỢP NGẦM DYNAMIC LABELS (TỰ ĐỘNG THÊM NHÃN MỚI CHO CẢ BOX & SAM POLYGON)
# =============================================================================
import threading
import xml.etree.ElementTree as ET
from flask import request, jsonify

_xml_lock = threading.Lock()
LS_URL = os.getenv("LABEL_STUDIO_URL", "http://127.0.0.1:19090")

# Các nhãn tạm sẽ bị thay bằng tên loài người dùng gõ
SENTINELS = {"__dynamic__", "+ Nhập nhãn mới...", "_poly", "Object"}
TEXT_FIELDS = {"species", "dynamic_class"}


def _get_ls_token():
    token = os.getenv("LABEL_STUDIO_API_KEY") or os.getenv("LABEL_STUDIO_TOKEN", "")
    if token:
        return token.strip()
    candidate_paths = [
        os.path.join(os.path.dirname(__file__), "token.txt"),
        os.path.join(os.path.dirname(__file__), "..", "token.txt"),
        os.path.join(os.path.dirname(__file__), "..", "dynamic_labels", "token.txt"),
        os.path.join(os.getcwd(), "token.txt"),
        os.path.join(os.getcwd(), "labeling", "token.txt"),
    ]
    for p in candidate_paths:
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    t = f.read().strip()
                    if t:
                        return t
            except Exception:
                pass
    return ""


def _get_ls_headers():
    token = _get_ls_token()
    if not token:
        return {"Content-Type": "application/json"}
    # Token dạng JWT refresh -> đổi sang access token
    try:
        r = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": token}, timeout=5)
        if r.ok and "access" in r.json():
            return {"Authorization": f"Bearer {r.json()['access']}",
                    "Content-Type": "application/json"}
    except Exception:
        pass
    # Token cũ dạng legacy
    return {"Authorization": f"Token {token}", "Content-Type": "application/json"}

# FIX
UNIFORM_COLOR = "#27ae60" 

def _sync_rect_colors_to_polygons(root_xml: ET.Element) -> bool:
    """Tất cả nhãn của RectangleLabels và PolygonLabels đều dùng chung UNIFORM_COLOR."""
    changed = False

    for rect_tag in root_xml.iter("RectangleLabels"):
        for lbl in rect_tag.findall("Label"):
            if lbl.get("background") != UNIFORM_COLOR:
                lbl.set("background", UNIFORM_COLOR)
                changed = True

    for poly_tag in root_xml.iter("PolygonLabels"):
        existing_vals = {lbl.get("value"): lbl for lbl in poly_tag.findall("Label")}
        for rect_tag in root_xml.iter("RectangleLabels"):
            for lbl in rect_tag.findall("Label"):
                val = lbl.get("value")
                if val in existing_vals:
                    if existing_vals[val].get("background") != UNIFORM_COLOR:
                        existing_vals[val].set("background", UNIFORM_COLOR)
                        changed = True
                else:
                    new_lbl = ET.SubElement(poly_tag, "Label")
                    new_lbl.set("value", val)
                    new_lbl.set("background", UNIFORM_COLOR)
                    changed = True

    return changed


def _is_box_overlapping_polygon(box_r: Dict[str, Any], poly_list: List[Dict[str, Any]]) -> bool:
    """Xác định hộp mồi của SAM: dựa vào trùng ID hoặc tâm con vật nằm trong hộp."""
    box_id = box_r.get("id")
    val = box_r.get("value", {})
    bx1 = val.get("x", 0.0)
    by1 = val.get("y", 0.0)
    bx2 = bx1 + val.get("width", 0.0)
    by2 = by1 + val.get("height", 0.0)

    for p in poly_list:
        if box_id and p.get("id") and box_id == p.get("id"):
            return True

        pts = p.get("value", {}).get("points", [])
        if not pts:
            continue
        xs = [pt[0] for pt in pts]
        ys = [pt[1] for pt in pts]
        px1, py1, px2, py2 = min(xs), min(ys), max(xs), max(ys)

        cx = (px1 + px2) / 2.0
        cy = (py1 + py2) / 2.0

        if (bx1 <= cx <= bx2) and (by1 <= cy <= by2):
            return True

    return False

def _sync_dynamic_label(payload: Dict[str, Any]):
    action = payload.get("action")
    if action not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
        return

    annotation = payload.get("annotation", {})
    project_id = (payload.get("project") or {}).get("id") or annotation.get("project")
    ann_id = annotation.get("id")
    results = annotation.get("result", [])
    if not project_id or not ann_id or not results:
        return

    # 1. Gom tên loài theo id vùng: {region_id: "buom"} (nếu người dùng có gõ chữ)
    name_by_region = {}
    for r in results:
        if r.get("from_name") in TEXT_FIELDS and r.get("type") == "textarea":
            texts = r.get("value", {}).get("text", [])
            if isinstance(texts, str):
                texts = [texts]
            if texts and texts[0].strip():
                region_id = r.get("parentID") or r.get("parent_id") or r.get("id")
                name_by_region[region_id] = texts[0].strip()

    # Kiểm tra xem ảnh này đã có đa giác SAM Polygon hay chưa
    has_polygons = any(r.get("type") == "polygonlabels" for r in results)

    # NẾU KHÔNG CÓ TÊN MỚI VÀ CŨNG KHÔNG CẦN DỌN BOX -> Thoát (chống lặp vô hạn)
    if not name_by_region and not has_polygons:
        return

    if not _get_ls_token():
        logger.warning("[DYNAMIC LABEL] Chưa có token! Đặt LABEL_STUDIO_API_KEY hoặc file token.txt")
        return

    headers = _get_ls_headers()

    # 2. Thêm nút nhãn mới vào XML nếu có gõ tên mới
    new_classes = set(name_by_region.values()) if name_by_region else set()
    if new_classes:
        logger.info(f"[DYNAMIC LABEL] Nhãn mới: {new_classes} (Project {project_id})")
        with _xml_lock:
            try:
                proj_res = requests.get(f"{LS_URL}/api/projects/{project_id}/", headers=headers, timeout=10)
                if proj_res.status_code == 200:
                    root = ET.fromstring(proj_res.json().get("label_config", ""))
                    changed = False
                    for tag in ("RectangleLabels", "PolygonLabels"):
                        for parent in root.iter(tag):
                            existing = {lbl.get("value") for lbl in parent.findall("Label")}
                            for cls in new_classes:
                                if cls not in existing:
                                    # ET.SubElement(parent, "Label").set("value", cls)
                                    # changed = True
                                    # Tạo thẻ Label mới kèm màu cố định UNIFORM_COLOR
                                    new_lbl = ET.SubElement(parent, "Label")
                                    new_lbl.set("value", cls)
                                    new_lbl.set("background", UNIFORM_COLOR)  # <-- Bổ sung dòng này
                                    changed = True
                    if changed:
                        patch = requests.patch(
                            f"{LS_URL}/api/projects/{project_id}/",
                            headers=headers,
                            json={"label_config": ET.tostring(root, encoding="unicode")},
                            timeout=10,
                        )
                        if patch.status_code in (200, 201):
                            logger.info("[DYNAMIC LABEL] Đã thêm nút nhãn vào XML!")
            except Exception as e:
                logger.error(f"[DYNAMIC LABEL ERROR] Lỗi cập nhật XML: {e}")

    # 3. DỌN BOX MỒI + KHỬ NHÃN TRÙNG LẶP
    only_name = next(iter(new_classes)) if len(new_classes) == 1 else None
    poly_results = [r for r in results if r.get("type") == "polygonlabels"]
    new_results = []
    changed = False

    for r in results:
        if r.get("from_name") in TEXT_FIELDS:
            changed = True
            continue  # Bỏ ô text tạm

        t = r.get("type")

        # Xóa Box mồi nếu trùng với Polygon; BB độc lập (không có Polygon) thì GIỮ NGUYÊN
        if t in ("rectanglelabels", "rectangle") and _is_box_overlapping_polygon(r, poly_results):
            changed = True
            continue

        if t in ("rectanglelabels", "polygonlabels"):
            val = r.setdefault("value", {})

            # Cập nhật nhãn mới nếu người dùng có gõ
            if name_by_region:
                vals = val.get(t, [])
                target = name_by_region.get(r.get("id"))
                if not target and only_name and any(v in SENTINELS for v in vals):
                    target = only_name
                if target and val.get(t) != [target]:
                    val[t] = [target]
                    changed = True

            # KHỬ TRÙNG LẶP: Ép ['con_trung', 'con_trung'] -> ['con_trung']
            cur = val.get(t, [])
            uniq = list(dict.fromkeys(cur))
            if len(uniq) != len(cur):
                val[t] = uniq
                changed = True

        new_results.append(r)

    # 4. GỬI PATCH KHI CÓ BẤT KỲ THAY ĐỔI NÀO (Xóa box hoặc khử nhãn lặp)
    if changed:
        try:
            res = requests.patch(
                f"{LS_URL}/api/annotations/{ann_id}/",
                headers=headers,
                json={"result": new_results},
                timeout=10
            )
            if res.status_code in (200, 201):
                logger.info(f"[DYNAMIC LABEL] Đã dọn sạch Box mồi và nhãn trùng cho Annotation {ann_id}!")
        except Exception as e:
            logger.error(f"[DYNAMIC LABEL ERROR] Lỗi cập nhật Annotation: {e}")

# ==============================================================================
# MODULE TỰ ĐỘNG CẤU HÌNH DỰ ÁN MỚI (AUTO PROVISIONING)
# ==============================================================================
import time
import threading
import requests
import xml.etree.ElementTree as ET

LS_HOST = os.getenv("LABEL_STUDIO_URL", "http://127.0.0.1:19090")
AUTO_PROVISION_POLL = int(os.getenv("AUTO_PROVISION_POLL", 3))
CONFIG_FILE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "label_config.xml")

_provisioned_projects = set()

def _load_blueprint_xml():
    """Đọc file default_config.xml và kiểm tra cú pháp an toàn."""
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.warning(f"[AUTO PROVISION] Không tìm thấy file {CONFIG_FILE_PATH}")
        return None
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        ET.fromstring(content)
        return content
    except Exception as e:
        logger.error(f"[AUTO PROVISION] Lỗi cú pháp XML trong default_config.xml: {e}")
        return None

def _run_provision_worker():
    """Quét ngầm liên tục để phát hiện dự án mới và tự động nạp cấu hình."""
    logger.info(f"[AUTO PROVISION] Đã kích hoạt quét ngầm (Chu kỳ: {AUTO_PROVISION_POLL}s)")
    headers = _get_ls_headers()

    # Nạp danh sách các project cũ để không cấu hình đè
    try:
        init_res = requests.get(f"{LS_HOST}/api/projects/?page_size=200", headers=headers, timeout=5)
        if init_res.status_code == 200:
            for p in init_res.json().get("results", []):
                _provisioned_projects.add(p["id"])
    except Exception:
        pass

    while True:
        try:
            time.sleep(AUTO_PROVISION_POLL)
            res = requests.get(f"{LS_HOST}/api/projects/?page_size=50", headers=headers, timeout=5)
            if res.status_code != 200:
                continue

            projects = res.json().get("results", [])
            for proj in projects:
                p_id = proj["id"]
                label_config = (proj.get("label_config") or "").strip()
                is_empty_config = label_config in ["", "<View></View>", "<View>\n</View>"]
                # Nếu là dự án mới tạo hoặc đang để trống
                if p_id not in _provisioned_projects or is_empty_config:
                    time.sleep(1) # Chờ 1 giây để database ổn định

                    # 1. Ghi đè XML nếu dự án đang để trống
                    if is_empty_config:
                        blueprint = _load_blueprint_xml()
                        if blueprint:
                            patch_res = requests.patch(
                                f"{LS_HOST}/api/projects/{p_id}/",
                                headers=headers,
                                json={"label_config": blueprint},
                                timeout=10
                            )
                            if patch_res.status_code == 200:
                                print(f"\n>>> [AUTO PROVISION] Đã tự nạp XML cho Project {p_id}!", flush=True)

                    # 2. Tự kết nối SAM 9090 nếu chưa có
                    try:
                        ml_check = requests.get(f"{LS_HOST}/api/ml/?project={p_id}", headers=headers, timeout=5)
                        has_sam = False
                        if ml_check.status_code == 200:
                            has_sam = any("9090" in str(item.get("url")) for item in ml_check.json())
                        
                        if not has_sam:
                            requests.post(
                                f"{LS_HOST}/api/ml/",
                                headers=headers,
                                json={
                                    "project": p_id,
                                    "url": "http://127.0.0.1:9090",
                                    "title": "MobileSAM Backend",
                                    "is_interactive": True
                                },
                                timeout=10
                            )
                            print(f">>> [AUTO PROVISION] Đã kết nối MobileSAM 9090 cho Project {p_id}!", flush=True)
                    except Exception as e:
                        logger.warning(f"[AUTO PROVISION ML ERROR]: {e}")

                    # 3. Tự kết nối Webhook nhãn động nếu chưa có
                    try:
                        wh_check = requests.get(f"{LS_HOST}/api/webhooks/?project={p_id}", headers=headers, timeout=5)
                        has_wh = False
                        if wh_check.status_code == 200:
                            has_wh = any("dynamic-webhook" in str(item.get("url")) for item in wh_check.json())

                        if not has_wh:
                            requests.post(
                                f"{LS_HOST}/api/webhooks/",
                                headers=headers,
                                json={
                                    "project": p_id,
                                    "url": "http://127.0.0.1:9090/api/dynamic-webhook",
                                    "send_payload": True,
                                    "send_for_all_actions": True,
                                    "is_active": True
                                },
                                timeout=10
                            )
                            print(f">>> [AUTO PROVISION] Đã kích hoạt Webhook nhãn động cho Project {p_id}!\n", flush=True)
                    except Exception as e:
                        logger.warning(f"[AUTO PROVISION WEBHOOK ERROR]: {e}")

                    _provisioned_projects.add(p_id)

        except Exception as e:
            pass

def start_auto_provision():
    """Khởi động luồng chạy ngầm của auto_provision."""
    t = threading.Thread(target=_run_provision_worker, daemon=True)
    t.start()

# =============================================================================
# Khởi chạy ML Backend Server độc lập (Cổng 9090 kiêm luôn Webhook)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Label Studio ML Backend - MobileSAM Interactive Segmentation")
    parser.add_argument("-p", "--port", type=int, default=9090, help="Cổng server (mặc định 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ host (mặc định 0.0.0.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file .pt của MobileSAM")
    parser.add_argument("--model-type", type=str, default="vit_t", choices=["vit_t", "vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--cache-size", type=int, default=10, help="Dung lượng bộ nhớ đệm LRU")
    parser.add_argument("--debug", action="store_true", help="Bật chế độ debug")

    args = parser.parse_args()

    logger.info("=" * 65)
    logger.info("  LABEL STUDIO ML BACKEND - MOBILESAM + DYNAMIC LABELS (9090)  ")
    logger.info("=" * 65)

    from label_studio_ml.api import init_app

    app = init_app(
        model_class=SAMBackend,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        cache_size=args.cache_size
    )

    # Đăng ký nhận webhook đồng bộ nhãn động ngay trên cổng 9090
    @app.route("/api/dynamic-webhook", methods=["POST"])
    def handle_ls_webhook():
        data = request.get_json(silent=True) or {}
        threading.Thread(target=_sync_dynamic_label, args=(data,), daemon=True).start()
        return jsonify({"status": "ok"}), 200
    
    # Kích hoạt tính năng Auto Provision chạy ngầm
    start_auto_provision()

    print(f"\n[SERVER READY] MobileSAM + Dynamic Labels đang chạy tại http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)
