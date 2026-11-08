# -*- coding: utf-8 -*-
"""
=============================================================================
Label Studio ML Backend - Segment Anything Model (SAM / MobileSAM)
Tích hợp:
  1. Interactive Segmentation (Kéo Box tới đâu bọc viền tới đó < 30ms)
  2. Batch Auto-Segmentation (Tự động bọc viền hàng loạt từ JSON Bounding Box
     qua CLI --batch-input, qua Webhook TASKS_CREATED khi Import, hoặc API;
     Web chỉ hiển thị Polygon nhưng Output JSON giữ nguyên cả BB + Polygon)
  3. Dynamic Labels & Color Sync (Tự động thêm nhãn & đồng bộ màu BB == SAM)
=============================================================================
"""

import argparse
import base64
import copy
import json
import logging
import math
import os
import re
import threading
import time
import traceback
import urllib.parse
import uuid
import xml.etree.ElementTree as ET
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2
import numpy as np
import requests
import torch
from flask import jsonify, request
from label_studio_ml.model import LabelStudioMLBase

# Cấu hình Logging chi tiết
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("SAMBackend")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LS_URL = os.getenv("LABEL_STUDIO_URL") or os.getenv("LABEL_STUDIO_HOST") or "http://127.0.0.1:19090"

MOBILESAM_DEFAULT_WEIGHTS_URL = (
    "https://raw.githubusercontent.com/ChaoningZhang/MobileSAM/master/weights/mobile_sam.pt"
)
MOBILESAM_BACKUP_WEIGHTS_URL = (
    "https://huggingface.co/dhkim2810/MobileSAM/resolve/main/mobile_sam.pt"
)

UUID_PREFIX_RE = re.compile(r"^[0-9a-fA-F]{8}-(.+)$")


# =============================================================================
# CÁC HÀM HÌNH HỌC HỖ TRỢ KHỚP BOX <-> POLYGON & KHỬ TRÙNG LẶP
# =============================================================================
def _read_image_unicode(path: str) -> Optional[np.ndarray]:
    """Đọc ảnh an toàn trên Windows kể cả đường dẫn có dấu Tiếng Việt."""
    try:
        buf = np.fromfile(path, dtype=np.uint8)
        if buf.size == 0:
            return None
        return cv2.imdecode(buf, cv2.IMREAD_COLOR)
    except Exception:
        return None


def _get_rect_bbox(box_r: Dict[str, Any]) -> Tuple[float, float, float, float]:
    val = box_r.get("value") if isinstance(box_r.get("value"), dict) else box_r
    x1 = float(val.get("x", 0.0))
    y1 = float(val.get("y", 0.0))
    x2 = x1 + float(val.get("width", 0.0))
    y2 = y1 + float(val.get("height", 0.0))
    return min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)


def _get_poly_bbox(poly_r: Dict[str, Any]) -> Optional[Tuple[float, float, float, float]]:
    pts = poly_r.get("value", {}).get("points", [])
    if not pts or len(pts) < 3:
        return None
    xs = [float(pt[0]) for pt in pts]
    ys = [float(pt[1]) for pt in pts]
    return min(xs), min(ys), max(xs), max(ys)


def _bbox_metrics(
    b1: Tuple[float, float, float, float],
    b2: Tuple[float, float, float, float],
) -> Tuple[float, float]:
    """Trả về (iou, ratio_b2_inside_b1)."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter_w = max(0.0, ix2 - ix1)
    inter_h = max(0.0, iy2 - iy1)
    inter_area = inter_w * inter_h
    if inter_area <= 0.0:
        return 0.0, 0.0

    area1 = max(1e-6, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    area2 = max(1e-6, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union_area = area1 + area2 - inter_area
    return inter_area / union_area, inter_area / area2


def _find_best_matching_polygon(
    box_r: Dict[str, Any],
    poly_list: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Tìm đa giác SAM được sinh ra từ chiếc hộp mồi box_r (nếu có)."""
    box_id = box_r.get("id")
    rect_bb = _get_rect_bbox(box_r)

    best_poly = None
    best_iou = 0.0

    for p in poly_list:
        p_meta_box_id = (p.get("meta") or {}).get("bbox_id")
        if box_id and (box_id == p.get("id") or box_id == p_meta_box_id):
            return p

        poly_bb = _get_poly_bbox(p)
        if poly_bb is None:
            continue

        iou, poly_in_box = _bbox_metrics(rect_bb, poly_bb)
        cx = (poly_bb[0] + poly_bb[2]) / 2.0
        cy = (poly_bb[1] + poly_bb[3]) / 2.0
        center_in_box = (rect_bb[0] <= cx <= rect_bb[2]) and (rect_bb[1] <= cy <= rect_bb[3])

        if (poly_in_box >= 0.75 and iou >= 0.20) or (center_in_box and iou >= 0.25):
            if iou > best_iou:
                best_iou = iou
                best_poly = p

    return best_poly


def _ls_box_val_to_xyxy_px(val: Dict[str, Any], img_w: int, img_h: int) -> Optional[np.ndarray]:
    """Chuyển tọa độ % của Label Studio (hỗ trợ cả box xoay) sang pixel [x1, y1, x2, y2]."""
    try:
        x_pct = float(val["x"])
        y_pct = float(val["y"])
        w_pct = float(val["width"])
        h_pct = float(val["height"])
    except (KeyError, TypeError, ValueError):
        return None

    rotation = float(val.get("rotation", 0.0) or 0.0)
    x0 = (x_pct / 100.0) * img_w
    y0 = (y_pct / 100.0) * img_h
    bw = (w_pct / 100.0) * img_w
    bh = (h_pct / 100.0) * img_h

    if abs(rotation) % 360.0 < 1e-3:
        x_min, y_min, x_max, y_max = x0, y0, x0 + bw, y0 + bh
    else:
        rad = math.radians(rotation)
        cos_a, sin_a = math.cos(rad), math.sin(rad)
        corners = [(0.0, 0.0), (bw, 0.0), (bw, bh), (0.0, bh)]
        xs = [x0 + dx * cos_a - dy * sin_a for dx, dy in corners]
        ys = [y0 + dx * sin_a + dy * cos_a for dx, dy in corners]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

    x_min = max(0.0, min(float(img_w - 1), x_min))
    y_min = max(0.0, min(float(img_h - 1), y_min))
    x_max = max(0.0, min(float(img_w), x_max))
    y_max = max(0.0, min(float(img_h), y_max))

    if (x_max - x_min) < 1.0 or (y_max - y_min) < 1.0:
        return None
    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def _fallback_box_polygon(
    box_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    label_name: str,
    from_name: str,
    to_name: str,
) -> Dict[str, Any]:
    """Dự phòng tạo đa giác 4 đỉnh ôm hộp nếu đối tượng quá bé khiến mask SAM rỗng (không làm mất mẫu)."""
    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    pts_pct = [
        [round(x1 / img_w * 100.0, 2), round(y1 / img_h * 100.0, 2)],
        [round(x2 / img_w * 100.0, 2), round(y1 / img_h * 100.0, 2)],
        [round(x2 / img_w * 100.0, 2), round(y2 / img_h * 100.0, 2)],
        [round(x1 / img_w * 100.0, 2), round(y2 / img_h * 100.0, 2)],
    ]
    return {
        "id": uuid.uuid4().hex[:10],
        "from_name": from_name,
        "to_name": to_name,
        "type": "polygonlabels",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "points": pts_pct,
            "polygonlabels": [label_name],
            "closed": True,
        },
        "score": 0.5,
    }


def _build_standard_bbox_region(
    box_xyxy: np.ndarray,
    img_w: int,
    img_h: int,
    label_name: str,
    from_name: str,
    to_name: str,
    raw_item: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Chuẩn hóa Bounding Box về đúng định dạng rectanglelabels của Label Studio để bảo toàn trong Output JSON."""
    if (
        isinstance(raw_item, dict)
        and raw_item.get("type") in ("rectanglelabels", "rectangle")
        and isinstance(raw_item.get("value"), dict)
    ):
        val = copy.deepcopy(raw_item["value"])
        val["rectanglelabels"] = [label_name]
        val.setdefault("rotation", 0)
        return {
            "id": raw_item.get("id") or f"box_{uuid.uuid4().hex[:8]}",
            "from_name": raw_item.get("from_name") or from_name,
            "to_name": raw_item.get("to_name") or to_name,
            "type": "rectanglelabels",
            "original_width": raw_item.get("original_width") or img_w,
            "original_height": raw_item.get("original_height") or img_h,
            "image_rotation": raw_item.get("image_rotation", 0),
            "value": val,
        }

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    raw_id = raw_item.get("id") if isinstance(raw_item, dict) else None
    return {
        "id": raw_id or f"box_{uuid.uuid4().hex[:8]}",
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "value": {
            "x": round(x1 / img_w * 100.0, 4),
            "y": round(y1 / img_h * 100.0, 4),
            "width": round((x2 - x1) / img_w * 100.0, 4),
            "height": round((y2 - y1) / img_h * 100.0, 4),
            "rotation": 0,
            "rectanglelabels": [label_name],
        },
    }


def _merge_bbox_lists(
    existing_boxes: List[Dict[str, Any]],
    new_boxes: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], bool]:
    """Gộp danh sách Bounding Box mà không làm mất BB gốc và không tạo bản sao trùng lặp."""
    merged: List[Dict[str, Any]] = [copy.deepcopy(b) for b in existing_boxes if isinstance(b, dict)]
    changed = False

    for nb in new_boxes:
        if not isinstance(nb, dict):
            continue
        nb_id = nb.get("id")
        nb_bb = _get_rect_bbox(nb)
        matched_idx = None

        for idx, eb in enumerate(merged):
            if nb_id and eb.get("id") == nb_id:
                matched_idx = idx
                break
            iou, _ = _bbox_metrics(nb_bb, _get_rect_bbox(eb))
            if iou >= 0.95:
                matched_idx = idx
                break

        if matched_idx is not None:
            if merged[matched_idx].get("value") != nb.get("value"):
                merged[matched_idx] = copy.deepcopy(nb)
                changed = True
        else:
            merged.append(copy.deepcopy(nb))
            changed = True

    return merged, changed


# =============================================================================
# BỘ NHỚ ĐỆM LRU EMBEDDING & LỚP ML BACKEND CHÍNH (SINGLETON MODEL)
# =============================================================================
class EmbeddingLRUCache:
    """Bộ nhớ đệm LRU lưu trữ Image Embeddings của SAM theo từng task_id/image."""

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
    """ML Backend tích hợp MobileSAM cho cả Interactive Segmentation & Batch JSON Segmentation."""

    _shared_init_lock = threading.Lock()
    _shared_predict_lock = threading.Lock()
    _shared_predictor: Optional[Any] = None
    _shared_embedding_cache: Optional[EmbeddingLRUCache] = None
    _shared_current_cache_key: Optional[str] = None
    _shared_processed_boxes: OrderedDict[Any, Set[Tuple[Any, ...]]] = OrderedDict()
    _runtime_opts: Dict[str, Any] = {}

    def __init__(
        self,
        checkpoint: Optional[str] = None,
        model_type: str = "vit_t",
        device: Optional[str] = None,
        cache_size: int = 10,
        approx_epsilon: float = 0.0015,
        min_contour_area: float = 4.0,
        **kwargs,
    ):
        super(SAMBackend, self).__init__(**kwargs)
        if not getattr(self, "model_dir", None):
            self.model_dir = os.path.join(BASE_DIR, "models_cache")
        os.makedirs(self.model_dir, exist_ok=True)

        self.model_version = f"sam-{model_type}"
        self.approx_epsilon = float(os.getenv("APPROX_EPSILON", approx_epsilon))
        self.min_contour_area = float(os.getenv("MIN_CONTOUR_AREA", min_contour_area))
        self.keep_largest_contour = os.getenv("KEEP_LARGEST_CONTOUR", "1") != "0"

        if device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        else:
            self.device = device

        weights_dir = os.path.join(BASE_DIR, "weights")
        os.makedirs(weights_dir, exist_ok=True)

        if checkpoint is None:
            checkpoint = os.getenv("CHECKPOINT_PATH")
        if not checkpoint:
            checkpoint = os.path.join(weights_dir, "mobile_sam.pt")

        self.checkpoint_path = checkpoint
        self.model_type = os.getenv("MODEL_TYPE", model_type)

        # Nạp MobileSAM duy nhất 1 lần (Singleton) để tiết kiệm VRAM cho cả Interactive và Batch
        with SAMBackend._shared_init_lock:
            if SAMBackend._shared_predictor is None:
                logger.info(f"[DEVICE] Thiết bị tính toán được chọn: {self.device}")
                SAMBackend._shared_predictor = self._load_sam_predictor(
                    self.checkpoint_path, self.model_type, self.device
                )
                SAMBackend._shared_embedding_cache = EmbeddingLRUCache(capacity=cache_size)
                SAMBackend._runtime_opts = {
                    "checkpoint": self.checkpoint_path,
                    "model_type": self.model_type,
                    "device": self.device,
                    "cache_size": cache_size,
                }

        self._predict_lock = SAMBackend._shared_predict_lock
        self.predictor = SAMBackend._shared_predictor
        self.embedding_cache = SAMBackend._shared_embedding_cache
        self._processed_boxes = SAMBackend._shared_processed_boxes

        self._parse_config_tags()

    @classmethod
    def get_shared_instance(cls) -> "SAMBackend":
        """Lấy instance phục vụ xử lý Batch ngầm mà không phải nạp lại trọng số MobileSAM."""
        opts = cls._runtime_opts
        return cls(
            checkpoint=opts.get("checkpoint"),
            model_type=opts.get("model_type", "vit_t"),
            device=opts.get("device"),
            cache_size=opts.get("cache_size", 10),
        )

    def setup(self):
        """Được gọi tự động mỗi khi Label Studio kết nối hoặc cập nhật XML mới."""
        super(SAMBackend, self).setup()
        self._parse_config_tags()
        logger.info("[SETUP SYNC] Đã đồng bộ cấu hình XML mới nhất từ Label Studio thành công!")

    @staticmethod
    def _box_signature(box_item: Dict[str, Any]) -> Tuple[Any, ...]:
        val = box_item.get("value", {})
        return (
            box_item.get("id") or "no_id",
            round(float(val.get("x", 0.0)), 1),
            round(float(val.get("y", 0.0)), 1),
            round(float(val.get("width", 0.0)), 1),
            round(float(val.get("height", 0.0)), 1),
        )

    def _is_box_already_segmented(
        self,
        task_id: Any,
        box_item: Dict[str, Any],
        existing_polys: List[Dict[str, Any]],
    ) -> bool:
        sig = self._box_signature(box_item)
        task_sigs = self._processed_boxes.get(task_id, set())

        if sig in task_sigs:
            if not existing_polys or _find_best_matching_polygon(box_item, existing_polys) is not None:
                return True
            task_sigs.discard(sig)

        return False

    def _mark_box_segmented(self, task_id: Any, box_item: Dict[str, Any]):
        if task_id not in self._processed_boxes:
            self._processed_boxes[task_id] = set()
        self._processed_boxes.move_to_end(task_id)
        self._processed_boxes[task_id].add(self._box_signature(box_item))
        if len(self._processed_boxes) > 100:
            self._processed_boxes.popitem(last=False)

    def _ensure_checkpoint_exists(self, checkpoint_path: str):
        """Tự động tải file trọng số nếu chưa có."""
        if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 1000:
            return

        logger.info(f"[WEIGHTS] Trọng số chưa tồn tại tại {checkpoint_path}. Bắt đầu tải...")
        os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

        downloaded = False
        for url in (MOBILESAM_DEFAULT_WEIGHTS_URL, MOBILESAM_BACKUP_WEIGHTS_URL):
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
            from mobile_sam import SamPredictor, sam_model_registry

            logger.info(f"[MODEL] Đang nạp MobileSAM ('{model_type}') từ '{checkpoint}'...")
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            sam.eval()
            return SamPredictor(sam)
        except ImportError:
            logger.warning("[MODEL] Gói mobile_sam chưa có, thử dùng segment_anything...")
        except Exception as e:
            logger.warning(f"[MODEL] Nạp mobile_sam lỗi: {e}, thử dùng segment_anything...")

        try:
            from segment_anything import SamPredictor, sam_model_registry

            if model_type == "vit_t":
                model_type = "vit_b"
            sam = sam_model_registry[model_type](checkpoint=checkpoint)
            sam.to(device=device)
            sam.eval()
            return SamPredictor(sam)
        except Exception as e:
            raise RuntimeError(f"Không thể nạp mô hình SAM: {e}")

    def _parse_config_tags(self):
        """Tự động phân tích các thẻ XML (từ Label Studio hoặc dự phòng từ file label_config.xml)."""
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
                elif tag_type in ("rectanglelabels", "rectangle"):
                    self.bbox_tag_name = tag_name
        else:
            xml_path = os.path.join(BASE_DIR, "label_config.xml")
            if os.path.isfile(xml_path):
                try:
                    root = ET.parse(xml_path).getroot()
                    img_el = root.find(".//Image")
                    if img_el is not None and img_el.get("name"):
                        self.image_tag_name = img_el.get("name", "image")
                        self.image_value_key = (img_el.get("value") or "$image").lstrip("$")
                    poly_el = root.find(".//PolygonLabels")
                    if poly_el is not None and poly_el.get("name"):
                        self.polygon_tag_name = poly_el.get("name", "polygon")
                    rect_el = root.find(".//RectangleLabels")
                    if rect_el is not None and rect_el.get("name"):
                        self.bbox_tag_name = rect_el.get("name", "label")
                except Exception:
                    pass

    def _resolve_image_rgb(
        self,
        image_url: str,
        task: Dict[str, Any],
        extra_search_dirs: Optional[List[str]] = None,
    ) -> Tuple[np.ndarray, str]:
        """Nạp ảnh thông minh: Hỗ trợ Base64, thư mục chỉ định, ổ đĩa cục bộ (Unicode Windows) hoặc HTTP."""
        if image_url.startswith("data:image/") and ";base64," in image_url:
            b64_data = image_url.split(";base64,", 1)[1]
            buf = np.frombuffer(base64.b64decode(b64_data), dtype=np.uint8)
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), f"b64_task_{task.get('id', 0)}"

        decoded_url = urllib.parse.unquote(image_url.strip())

        # 1. Đường dẫn trực tiếp hoặc tương đối trong extra_search_dirs
        direct_candidates = [decoded_url]
        parsed = urllib.parse.urlparse(decoded_url)
        path_only = parsed.path if parsed.scheme in ("http", "https") else decoded_url
        file_name = os.path.basename(path_only)

        if extra_search_dirs:
            for d in extra_search_dirs:
                if not d:
                    continue
                direct_candidates.append(os.path.join(d, path_only.lstrip("/\\")))
                if file_name:
                    direct_candidates.append(os.path.join(d, file_name))
                    m = UUID_PREFIX_RE.match(file_name)
                    if m:
                        direct_candidates.append(os.path.join(d, m.group(1)))

        for cand in direct_candidates:
            if cand and os.path.isfile(cand):
                img = _read_image_unicode(cand)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), os.path.abspath(cand)

        # 2. Xử lý /data/local-files/?d=... (Hỗ trợ cả Windows và Linux/macOS)
        if "/data/local-files/" in parsed.path and parsed.query:
            d_param = urllib.parse.parse_qs(parsed.query).get("d", [""])[0]
            if d_param:
                d_clean = urllib.parse.unquote(d_param).replace("\\", os.sep).replace("/", os.sep)
                default_root = (os.environ.get("SystemDrive", "C:") + "\\") if os.name == "nt" else "/"
                doc_root = os.environ.get("LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT", "") or default_root
                local_cands = [d_clean, os.path.join(doc_root, d_clean.lstrip("/\\"))]
                if extra_search_dirs:
                    local_cands.extend(os.path.join(d, d_clean.lstrip("/\\")) for d in extra_search_dirs if d)
                for candidate in local_cands:
                    if candidate and os.path.isfile(candidate):
                        img = _read_image_unicode(candidate)
                        if img is not None:
                            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), os.path.abspath(candidate)

        # 3. Tìm trong các thư mục upload media của Label Studio
        candidate_roots = [
            os.path.join(BASE_DIR, "..", "data", "media", "upload"),
            os.path.join(BASE_DIR, "..", "data", "media"),
            os.path.join(os.getcwd(), "labeling", "data", "media", "upload"),
            os.path.join(os.getcwd(), "labeling", "data", "media"),
            os.path.join(os.getcwd(), "data", "media", "upload"),
            os.path.join(os.getcwd(), "data", "media"),
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "label-studio", "label-studio", "media", "upload"),
            os.path.join(os.path.expanduser("~"), "AppData", "Local", "label-studio", "label-studio", "media"),
            os.path.join(os.path.expanduser("~"), ".local", "share", "label-studio", "media", "upload"),
            os.path.join(os.path.expanduser("~"), ".local", "share", "label-studio", "media"),
        ]

        clean_subpath = parsed.path
        if "/data/upload/" in clean_subpath:
            clean_subpath = clean_subpath.split("/data/upload/")[-1]
        elif "/data/" in clean_subpath:
            clean_subpath = clean_subpath.split("/data/")[-1]

        for root in candidate_roots:
            full_path = os.path.normpath(os.path.join(root, clean_subpath.lstrip("/\\")))
            if os.path.isfile(full_path):
                img = _read_image_unicode(full_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), full_path

        # 4. Thử qua get_local_path của LabelStudioMLBase
        try:
            local_path = self.get_local_path(image_url, task_id=task.get("id"))
            if local_path and os.path.isfile(local_path):
                img = _read_image_unicode(local_path)
                if img is not None:
                    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB), local_path
        except Exception:
            pass

        # 5. Tải qua HTTP/HTTPS từ server Label Studio
        headers = _get_ls_headers()
        headers.pop("Content-Type", None)

        possible_hosts = [
            getattr(self, "hostname", None),
            LS_URL,
            "http://localhost:19090",
            "http://127.0.0.1:19090",
            "http://localhost:8080",
            "http://127.0.0.1:8080",
        ]

        target_urls = []
        if image_url.startswith(("http://", "https://")):
            target_urls.append(image_url)
        elif image_url.startswith("/"):
            for h in dict.fromkeys(possible_hosts):
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
        """Đồng bộ Image Embedding qua LRU Cache (gọi bên trong self._predict_lock)."""
        cache_key = f"{task_id}_{image_key}"

        if SAMBackend._shared_current_cache_key == cache_key and getattr(self.predictor, "is_image_set", False):
            return

        cached = self.embedding_cache.get(cache_key)
        if cached is not None:
            for attr_name, attr_val in cached.items():
                if attr_val is not None:
                    setattr(self.predictor, attr_name, attr_val)
            self.predictor.is_image_set = True
            SAMBackend._shared_current_cache_key = cache_key
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
        SAMBackend._shared_current_cache_key = cache_key
        logger.info(f"[EMBEDDING DONE] Hoàn thành sau {time.time() - t0:.3f}s")

    def _mask_to_polygons(
        self,
        binary_mask: np.ndarray,
        img_width: int,
        img_height: int,
        label_name: str,
        score: float,
        from_name: Optional[str] = None,
        to_name: Optional[str] = None,
        include_id: bool = True,
    ) -> List[Dict[str, Any]]:
        """Chuyển mask sang đa giác Polygon chuẩn hóa [0, 100]% (chống đẻ mảnh nhiễu lặp nhãn)."""
        poly_from = from_name or self.polygon_tag_name
        img_to = to_name or self.image_tag_name

        mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
        if np.count_nonzero(mask_u8) > 64:
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [(cv2.contourArea(c), c) for c in contours]
        valid_contours = [(area, c) for area, c in valid_contours if area >= self.min_contour_area]
        if not valid_contours:
            return []

        valid_contours.sort(key=lambda pair: pair[0], reverse=True)
        if self.keep_largest_contour:
            selected_contours = [valid_contours[0][1]]
        else:
            max_area = valid_contours[0][0]
            selected_contours = [c for area, c in valid_contours if area >= max_area * 0.2]

        if isinstance(label_name, (list, tuple)):
            clean_label = str(label_name[0]) if len(label_name) > 0 else "Object"
        else:
            clean_label = str(label_name) if label_name else "Object"

        results = []
        for contour in selected_contours:
            epsilon = self.approx_epsilon * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            pts = approx.reshape(-1, 2)

            if len(pts) < 3:
                raw_pts = contour.reshape(-1, 2)
                if len(raw_pts) < 3:
                    continue
                pts = raw_pts

            points_pct = []
            for px, py in pts:
                norm_x = max(0.0, min(100.0, round(float(px) / float(img_width) * 100.0, 2)))
                norm_y = max(0.0, min(100.0, round(float(py) / float(img_height) * 100.0, 2)))
                points_pct.append([norm_x, norm_y])

            item: Dict[str, Any] = {
                "from_name": poly_from,
                "to_name": img_to,
                "type": "polygonlabels",
                "original_width": img_width,
                "original_height": img_height,
                "image_rotation": 0,
                "value": {
                    "points": points_pct,
                    "polygonlabels": [clean_label],
                    "closed": True,
                },
                "score": round(float(score), 4),
            }
            if include_id:
                item["id"] = uuid.uuid4().hex[:10]
            results.append(item)

        return results

    # =========================================================================
    # TÍNH NĂNG BATCH: TỰ ĐỘNG BỌC VIỀN HÀNG LOẠT TỪ MỌI BOUNDING BOX TRONG TASK
    # =========================================================================
    def _extract_task_image_url(self, task: Dict[str, Any]) -> str:
        data = task.get("data") or {}
        if not isinstance(data, dict):
            return ""
        for key in (self.image_value_key, "image", "img", "image_url", "url", "file_upload"):
            val = data.get(key)
            if isinstance(val, str) and val.strip():
                return val.strip()
        img_exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")
        for val in data.values():
            if isinstance(val, str) and val.lower().split("?")[0].endswith(img_exts):
                return val.strip()
        return ""

    @staticmethod
    def extract_box_source_from_task(
        task: Dict[str, Any],
    ) -> Tuple[str, Optional[Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Trích xuất danh sách Bounding Box có sẵn từ task.
        Trả về: (source_type, source_item_id, box_regions, other_non_box_regions)
        - source_type: 'coco_px' | 'annotations' | 'predictions' | 'data_boxes' | 'none'
        """
        # 1. Trường hợp dữ liệu pixel từ COCO JSON đã chuẩn hóa tạm
        if "_coco_boxes" in task:
            return "coco_px", None, task.get("_coco_boxes") or [], []

        existing_non_boxes: List[Dict[str, Any]] = []

        # 2. Ưu tiên kiểm tra annotations trước, sau đó tới predictions
        for source_key in ("annotations", "completions", "predictions"):
            items = task.get(source_key) or []
            if not isinstance(items, list):
                continue
            for item in reversed(items):
                if not isinstance(item, dict) or item.get("was_cancelled"):
                    continue
                results = item.get("result") or []
                boxes = [
                    r for r in results
                    if isinstance(r, dict) and r.get("type") in ("rectanglelabels", "rectangle")
                ]
                others = [
                    r for r in results
                    if isinstance(r, dict) and r.get("type") not in ("rectanglelabels", "rectangle")
                ]
                if others and not existing_non_boxes:
                    existing_non_boxes = others
                if boxes:
                    norm_source = "annotations" if source_key in ("annotations", "completions") else "predictions"
                    return norm_source, item.get("id"), boxes, others

        # 3. Hỗ trợ thêm JSON đơn giản chứa danh sách hộp ngay trong data.boxes / data.bboxes
        data = task.get("data") or {}
        if isinstance(data, dict):
            raw_boxes = data.get("boxes") or data.get("bboxes")
            if isinstance(raw_boxes, list) and raw_boxes:
                return "data_boxes", None, raw_boxes, existing_non_boxes

        return "none", None, [], existing_non_boxes

    def segment_task_boxes(
        self,
        task: Dict[str, Any],
        polygon_from_name: Optional[str] = None,
        extra_search_dirs: Optional[List[str]] = None,
        only_unsegmented_boxes: bool = True,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], float, Set[str], str, Optional[Any]]:
        """
        Chạy MobileSAM bọc viền toàn bộ Bounding Box của 1 ảnh (tính Image Embedding đúng 1 lần/ảnh).
        Trả về: (polygon_results, preserved_other_regions, preserved_bboxes, avg_score, discovered_labels, source_type, source_id)
        - polygon_results: dùng hiển thị trên Web (mỗi polygon có đính kèm meta.bbox)
        - preserved_bboxes: danh sách rectanglelabels chuẩn để lưu vào Output JSON (meta.bboxes) mà không hiện đè lên Web
        """
        poly_tag = polygon_from_name or self.polygon_tag_name
        bbox_tag = self.bbox_tag_name
        source_type, source_id, all_box_items, other_regions = self.extract_box_source_from_task(task)
        if not all_box_items:
            return [], other_regions, [], 0.0, set(), "none", None

        existing_polys = [r for r in other_regions if r.get("type") == "polygonlabels"]
        box_items = all_box_items
        if only_unsegmented_boxes and existing_polys:
            box_items = [
                b for b in all_box_items
                if _find_best_matching_polygon(b, existing_polys) is None
            ]
            if not box_items:
                return [], other_regions, [], 0.0, set(), "already_segmented", source_id

        image_url = self._extract_task_image_url(task)
        if not image_url:
            raise ValueError(f"Task {task.get('id', '?')} không có đường dẫn ảnh trong trường 'data'.")

        image_rgb, image_ref = self._resolve_image_rgb(
            image_url, task, extra_search_dirs=extra_search_dirs
        )
        img_h, img_w = image_rgb.shape[:2]

        def _parse_single_box(item: Any) -> Optional[Tuple[np.ndarray, str, str, Dict[str, Any]]]:
            if source_type == "coco_px":
                if not isinstance(item, dict) or "bbox_px" not in item:
                    return None
                x, y, w, h = [float(v) for v in item["bbox_px"]]
                x1 = max(0.0, min(float(img_w - 1), x))
                y1 = max(0.0, min(float(img_h - 1), y))
                x2 = max(0.0, min(float(img_w), x + w))
                y2 = max(0.0, min(float(img_h), y + h))
                if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                    return None
                box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                lbl = str(item.get("label") or "Object")
                to_name = self.image_tag_name
                std_box = _build_standard_bbox_region(box_np, img_w, img_h, lbl, bbox_tag, to_name, item)
                return box_np, lbl, to_name, std_box

            if source_type == "data_boxes":
                if not isinstance(item, dict):
                    return None
                lbl = str(item.get("label") or item.get("class") or "Object")
                to_name = self.image_tag_name
                if all(k in item for k in ("x1", "y1", "x2", "y2")):
                    x1 = max(0.0, min(float(img_w - 1), float(item["x1"])))
                    y1 = max(0.0, min(float(img_h - 1), float(item["y1"])))
                    x2 = max(0.0, min(float(img_w), float(item["x2"])))
                    y2 = max(0.0, min(float(img_h), float(item["y2"])))
                    if (x2 - x1) < 1.0 or (y2 - y1) < 1.0:
                        return None
                    box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
                else:
                    box_np = _ls_box_val_to_xyxy_px(item, img_w, img_h)
                    if box_np is None:
                        return None
                std_box = _build_standard_bbox_region(box_np, img_w, img_h, lbl, bbox_tag, to_name, item)
                return box_np, lbl, to_name, std_box

            if not isinstance(item, dict):
                return None
            val = item.get("value") or {}
            box_np = _ls_box_val_to_xyxy_px(val, img_w, img_h)
            if box_np is None:
                return None
            labels = val.get("rectanglelabels") or val.get("labels") or ["Object"]
            lbl = str(labels[0]) if isinstance(labels, list) and labels else "Object"
            to_name = str(item.get("to_name") or self.image_tag_name)
            std_box = _build_standard_bbox_region(box_np, img_w, img_h, lbl, bbox_tag, to_name, item)
            return box_np, lbl, to_name, std_box

        # Bảo toàn toàn bộ danh sách Bounding Box gốc của ảnh
        preserved_bboxes: List[Dict[str, Any]] = []
        for raw_b in all_box_items:
            parsed_b = _parse_single_box(raw_b)
            if parsed_b is not None:
                preserved_bboxes.append(parsed_b[3])

        # Chuẩn bị danh sách (box_xyxy, label_name, to_name, std_box_region) cần chạy SAM
        prompts: List[Tuple[np.ndarray, str, str, Dict[str, Any]]] = []
        discovered_labels: Set[str] = set()
        for item in box_items:
            parsed_p = _parse_single_box(item)
            if parsed_p is not None:
                discovered_labels.add(parsed_p[1])
                prompts.append(parsed_p)

        if not prompts:
            return [], other_regions, preserved_bboxes, 0.0, set(), "none", source_id

        task_id = task.get("id", f"batch_{hash(image_ref)}")
        polygon_results: List[Dict[str, Any]] = []
        scores_list: List[float] = []

        # Khóa GPU trong phạm vi 1 ảnh để không làm đơ thao tác Interactive vẽ tay trên Web
        with self._predict_lock:
            self._sync_image_embedding(task_id, image_ref, image_rgb)
            with torch.no_grad():
                for box_np, lbl, to_name, std_box in prompts:
                    masks, scores, _ = self.predictor.predict(
                        box=box_np,
                        multimask_output=True,
                    )
                    best_idx = int(np.argmax(scores))
                    best_mask = masks[best_idx]
                    best_score = float(scores[best_idx])
                    scores_list.append(best_score)

                    polys = self._mask_to_polygons(
                        binary_mask=best_mask,
                        img_width=img_w,
                        img_height=img_h,
                        label_name=lbl,
                        score=best_score,
                        from_name=poly_tag,
                        to_name=to_name,
                        include_id=True,
                    )
                    if not polys:
                        # Không bỏ sót con sâu quá bé trên đĩa nếu mask nhỏ hơn ngưỡng
                        polys = [_fallback_box_polygon(box_np, img_w, img_h, lbl, poly_tag, to_name)]

                    for p in polys:
                        p["meta"] = {
                            "bbox_id": std_box["id"],
                            "bbox": copy.deepcopy(std_box["value"]),
                        }
                    polygon_results.extend(polys)

        avg_score = round(sum(scores_list) / len(scores_list), 4) if scores_list else 0.0
        return (
            polygon_results,
            other_regions,
            preserved_bboxes,
            avg_score,
            discovered_labels,
            source_type,
            source_id,
        )

    # =========================================================================
    # PREDICT: TỰ ĐỘNG NHẬN DIỆN INTERACTIVE (1 HỘP) HOẶC BATCH (NHIỀU HỘP)
    # =========================================================================
    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> List[Dict]:
        """
        - Khi có `context.result` (người dùng kéo Box trực tiếp trên Web): chạy Interactive Mode (< 30ms).
        - Khi không có `context.result` (mở ảnh mới Import hoặc Retrieve Predictions): chạy Batch Mode.
        """
        if context is None:
            try:
                if request and request.is_json:
                    req_json = request.get_json(silent=True) or {}
                    context = req_json.get("context") or (req_json.get("params") or {}).get("context")
            except Exception:
                pass

        if not tasks:
            return []

        ctx_results = (context or {}).get("result") or []

        # --- NHÁNH 1: BATCH MODE (Khi mở ảnh vừa Import hoặc gọi Retrieve Predictions) ---
        if not ctx_results:
            batch_predictions: List[Dict[str, Any]] = []
            for task in tasks:
                try:
                    tid = task.get("id")
                    # Nếu payload task từ Web chưa kèm chi tiết annotations.result thì lấy từ API
                    anns = task.get("annotations")
                    if tid and (not isinstance(anns, list) or (anns and not isinstance(anns[0], dict))):
                        try:
                            t_res = requests.get(
                                f"{LS_URL}/api/tasks/{tid}/",
                                headers=_get_ls_headers(),
                                timeout=8,
                            )
                            if t_res.status_code == 200:
                                task = t_res.json()
                        except Exception:
                            pass

                    polys, others, preserved_bboxes, avg_score, found_labels, src_type, src_id = (
                        self.segment_task_boxes(task, only_unsegmented_boxes=True)
                    )
                    combined = others + polys if polys else []
                    logger.info(
                        f"[BATCH PREDICT] Task {tid}: Sinh {len(polys)} đa giác (nguồn: {src_type})"
                    )

                    # Nếu ảnh được Import kèm sẵn Annotation (tab QU chứa Box), tự động nâng cấp thẳng tab QU sang Polygon!
                    if polys and tid:
                        proj_id = task.get("project")
                        threading.Thread(
                            target=_upgrade_imported_task_on_ls,
                            args=(tid, proj_id, combined, preserved_bboxes, avg_score, found_labels, src_type, src_id),
                            daemon=True,
                        ).start()

                    batch_predictions.append({
                        "result": combined,
                        "score": avg_score,
                        "model_version": self.model_version,
                    })
                except Exception as e:
                    logger.error(f"[BATCH PREDICT ERROR] Task {task.get('id')}: {e}")
                    batch_predictions.append({"result": [], "model_version": self.model_version})
            return batch_predictions

        # --- NHÁNH 2: INTERACTIVE MODE (Người dùng đang kéo Box bằng chuột trên giao diện) ---
        all_boxes = [
            item for item in ctx_results
            if item.get("type") in ("rectanglelabels", "rectangle") and "value" in item
        ]
        existing_polys = [
            item for item in ctx_results
            if item.get("type") == "polygonlabels" and "value" in item
        ]

        if not all_boxes:
            return [{"result": [], "model_version": self.model_version}]

        task = tasks[0]
        task_id = task.get("id", "task_0")

        unprocessed_boxes = [
            b for b in all_boxes
            if not self._is_box_already_segmented(task_id, b, existing_polys)
        ]

        if not unprocessed_boxes:
            logger.info("[PREDICT SKIP] Không có Box mới cần cắt (tránh sinh lặp nhãn Polygon).")
            return [{"result": [], "model_version": self.model_version}]

        target_box = unprocessed_boxes[-1]
        logger.info(f"[PREDICT] Chạy SAM duy nhất cho Box mới ID: {target_box.get('id')}")

        image_url = self._extract_task_image_url(task)
        if not image_url:
            logger.error(f"[PREDICT] Task {task_id} thiếu link ảnh!")
            return [{"result": [], "model_version": self.model_version}]

        try:
            image_rgb, image_ref = self._resolve_image_rgb(image_url, task)
        except Exception as e:
            logger.error(f"[PREDICT ERROR] Không thể mở ảnh: {e}")
            logger.error(traceback.format_exc())
            return [{"result": [], "model_version": self.model_version}]

        img_height, img_width = image_rgb.shape[:2]
        val = target_box["value"]
        labels = val.get("rectanglelabels") or val.get("labels") or ["Object"]
        label_name = labels[0] if labels else "Object"

        box_prompt_np = _ls_box_val_to_xyxy_px(val, img_width, img_height)
        if box_prompt_np is None:
            return [{"result": [], "model_version": self.model_version}]

        with self._predict_lock:
            self._sync_image_embedding(task_id, image_ref, image_rgb)
            t0 = time.time()
            with torch.no_grad():
                masks, scores, _ = self.predictor.predict(
                    box=box_prompt_np,
                    multimask_output=True,
                )
            best_idx = int(np.argmax(scores))
            best_mask = masks[best_idx]
            best_score = float(scores[best_idx])

        logger.info(
            f"[SAM SUCCESS] '{label_name}' xong trong {(time.time() - t0) * 1000:.1f}ms - Score: {best_score:.4f}"
        )

        polygon_results = self._mask_to_polygons(
            binary_mask=best_mask,
            img_width=img_width,
            img_height=img_height,
            label_name=label_name,
            score=best_score,
            include_id=False,
        )

        if polygon_results:
            self._mark_box_segmented(task_id, target_box)

        return [{
            "result": polygon_results,
            "score": round(best_score, 4),
            "model_version": self.model_version,
        }]


# =============================================================================
# TÍCH HỢP NGẦM DYNAMIC LABELS, ĐỒNG BỘ MÀU & BATCH WEBHOOK KHI IMPORT JSON
# =============================================================================
_xml_lock = threading.Lock()
_ann_sync_lock = threading.Lock()
_batch_task_lock = threading.Lock()
_active_ann_ids: Set[Any] = set()
_active_batch_tasks: Set[Any] = set()
_jwt_cache: Dict[str, Any] = {"raw": "", "access": "", "expires_at": 0.0}

SENTINELS = {"__dynamic__", "+ Nhập nhãn mới...", "_poly", "Object"}
TEXT_FIELDS = {"species", "dynamic_class"}
UNIFORM_COLOR = "#27ae60"


def _get_ls_token() -> str:
    token = os.getenv("LABEL_STUDIO_API_KEY") or os.getenv("LABEL_STUDIO_TOKEN", "")
    if token:
        return token.strip()
    candidate_paths = [
        os.path.join(BASE_DIR, "token.txt"),
        os.path.join(BASE_DIR, "..", "token.txt"),
        os.path.join(BASE_DIR, "..", "dynamic_labels", "token.txt"),
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

    # Fallback tự động 100% cho máy mới clone từ Git (không cần tạo file token.txt thủ công):
    # Tự động bật cho phép Legacy Token trong jwt_auth_jwtsettings và lấy key từ authtoken_token
    db_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "label_studio.sqlite3"))
    if os.path.isfile(db_path):
        try:
            import sqlite3
            conn = sqlite3.connect(db_path, timeout=5)
            try:
                cur = conn.execute("UPDATE jwt_auth_jwtsettings SET legacy_api_tokens_enabled = 1 WHERE legacy_api_tokens_enabled = 0")
                if cur.rowcount > 0:
                    conn.commit()
            except Exception:
                pass
            row = conn.execute("SELECT key FROM authtoken_token ORDER BY created DESC LIMIT 1").fetchone()
            conn.close()
            if row and row[0]:
                return str(row[0]).strip()
        except Exception:
            pass
    return ""


def _get_ls_headers() -> Dict[str, str]:
    """Lấy Header xác thực Label Studio (có cache JWT Access Token 4 phút)."""
    token = _get_ls_token()
    if not token:
        return {"Content-Type": "application/json"}

    now = time.time()
    if _jwt_cache["raw"] == token and _jwt_cache["access"] and now < _jwt_cache["expires_at"]:
        return {
            "Authorization": f"Bearer {_jwt_cache['access']}",
            "Content-Type": "application/json",
        }

    try:
        r = requests.post(f"{LS_URL}/api/token/refresh", json={"refresh": token}, timeout=5)
        if r.ok and "access" in r.json():
            access_tok = r.json()["access"]
            _jwt_cache["raw"] = token
            _jwt_cache["access"] = access_tok
            _jwt_cache["expires_at"] = now + 240.0
            return {
                "Authorization": f"Bearer {access_tok}",
                "Content-Type": "application/json",
            }
    except Exception:
        pass

    return {"Authorization": f"Token {token}", "Content-Type": "application/json"}


def _persist_bboxes_to_task_meta(
    task_id: Any,
    bboxes: List[Dict[str, Any]],
    headers: Dict[str, str],
    current_meta: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Lưu danh sách Bounding Box chuẩn vào task.meta.bboxes để Output JSON luôn có đủ BB mà Web không bị vẽ đè."""
    # if not task_id or not bboxes:
    return bboxes

    meta_obj: Dict[str, Any] = copy.deepcopy(current_meta) if isinstance(current_meta, dict) else {}
    if current_meta is None:
        try:
            t_res = requests.get(f"{LS_URL}/api/tasks/{task_id}/", headers=headers, timeout=10)
            if t_res.status_code == 200:
                fetched_meta = t_res.json().get("meta")
                if isinstance(fetched_meta, dict):
                    meta_obj = fetched_meta
        except Exception:
            pass

    existing_boxes = meta_obj.get("bboxes") if isinstance(meta_obj.get("bboxes"), list) else []
    merged_boxes, changed = _merge_bbox_lists(existing_boxes, bboxes)
    if changed or "bboxes" not in meta_obj:
        meta_obj["bboxes"] = merged_boxes
        try:
            requests.patch(
                f"{LS_URL}/api/tasks/{task_id}/",
                headers=headers,
                json={"meta": meta_obj},
                timeout=10,
            )
        except Exception as e:
            logger.warning(f"[META BBOX SAVE] Không thể lưu meta.bboxes cho task {task_id}: {e}")
    return merged_boxes


def _upgrade_imported_task_on_ls(
    tid: Any,
    project_id: Any,
    merged_results: List[Dict[str, Any]],
    preserved_bboxes: List[Dict[str, Any]],
    avg_score: float,
    found_labels: Set[str],
    src_type: str,
    src_id: Any,
):
    """Cập nhật trực tiếp tab Annotation (QU) sang Polygon và lưu BB vào meta.bboxes."""
    if not _get_ls_token():
        return
    headers = _get_ls_headers()
    if found_labels and project_id:
        _ensure_labels_in_project_xml(project_id, found_labels)
    if preserved_bboxes:
        _persist_bboxes_to_task_meta(tid, preserved_bboxes, headers)

    if src_type == "annotations" and src_id:
        with _ann_sync_lock:
            _active_ann_ids.add(src_id)
        try:
            resp = requests.patch(
                f"{LS_URL}/api/annotations/{src_id}/",
                headers=headers,
                json={"result": merged_results},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                logger.info(
                    f"[AUTO-UPGRADE] Task {tid}: Đã chuyển tab Annotation {src_id} sang Polygon & giữ {len(preserved_bboxes)} BB!"
                )
                # Dọn các tab Prediction thừa trên cùng task để không bị chia 2 tab
                time.sleep(0.5)
                t_res = requests.get(f"{LS_URL}/api/tasks/{tid}/", headers=headers, timeout=8)
                if t_res.status_code == 200:
                    for pred in t_res.json().get("predictions") or []:
                        pid = pred.get("id") if isinstance(pred, dict) else pred
                        if pid:
                            requests.delete(f"{LS_URL}/api/predictions/{pid}/", headers=headers, timeout=5)
        except Exception as e:
            logger.warning(f"[AUTO-UPGRADE ERROR] Task {tid}: {e}")
        finally:
            with _ann_sync_lock:
                _active_ann_ids.discard(src_id)


def _sync_rect_colors_to_polygons(root_xml: ET.Element) -> bool:
    """Ép màu của từng nhãn bên SAM (PolygonLabels) giống hệt màu bên BB (RectangleLabels)."""
    changed = False
    rect_colors = {}
    for rect_tag in root_xml.iter("RectangleLabels"):
        for lbl in rect_tag.findall("Label"):
            val = lbl.get("value")
            color = lbl.get("background") or UNIFORM_COLOR
            if not lbl.get("background"):
                lbl.set("background", color)
                changed = True
            if val:
                rect_colors[val] = color

    for poly_tag in root_xml.iter("PolygonLabels"):
        existing_vals = {lbl.get("value"): lbl for lbl in poly_tag.findall("Label")}
        for val, color in rect_colors.items():
            if val in existing_vals:
                if existing_vals[val].get("background") != color:
                    existing_vals[val].set("background", color)
                    changed = True
            else:
                new_lbl = ET.SubElement(poly_tag, "Label")
                new_lbl.set("value", val)
                new_lbl.set("background", color)
                changed = True

    return changed


def _ensure_labels_in_project_xml(project_id: Any, class_names: Set[str]) -> Optional[Tuple[str, str]]:
    """Đảm bảo mọi tên nhãn từ file JSON đều có mặt trong XML của Project và trả về (polygon_tag, image_tag)."""
    clean_classes = {c.strip() for c in class_names if c and c.strip()}
    headers = _get_ls_headers()
    poly_tag_name = "polygon"
    img_tag_name = "image"
    db_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "label_studio.sqlite3"))

    with _xml_lock:
        try:
            xml_str = ""
            used_sqlite = False
            try:
                proj_res = requests.get(f"{LS_URL}/api/projects/{project_id}/", headers=headers, timeout=5)
                if proj_res.status_code == 200:
                    xml_str = proj_res.json().get("label_config", "")
            except Exception:
                pass

            if not xml_str.strip() and os.path.isfile(db_path):
                import sqlite3
                conn = sqlite3.connect(db_path, timeout=10)
                row = conn.execute("SELECT label_config FROM project WHERE id = ?", (project_id,)).fetchone()
                conn.close()
                if row and row[0]:
                    xml_str = row[0]
                    used_sqlite = True

            if not xml_str.strip():
                return None

            root = ET.fromstring(xml_str)
            img_el = root.find(".//Image")
            if img_el is not None and img_el.get("name"):
                img_tag_name = img_el.get("name", "image")
            poly_el = root.find(".//PolygonLabels")
            if poly_el is not None and poly_el.get("name"):
                poly_tag_name = poly_el.get("name", "polygon")

            xml_changed = False
            for tag in ("RectangleLabels", "PolygonLabels"):
                for parent in root.iter(tag):
                    existing = {lbl.get("value") for lbl in parent.findall("Label")}
                    for cls in clean_classes:
                        if cls not in existing:
                            new_lbl = ET.SubElement(parent, "Label")
                            new_lbl.set("value", cls)
                            new_lbl.set("background", UNIFORM_COLOR)
                            xml_changed = True

            if _sync_rect_colors_to_polygons(root):
                xml_changed = True

            if xml_changed:
                new_xml_str = ET.tostring(root, encoding="unicode")
                patched_ok = False
                if not used_sqlite:
                    try:
                        patch = requests.patch(
                            f"{LS_URL}/api/projects/{project_id}/",
                            headers=headers,
                            json={"label_config": new_xml_str},
                            timeout=10,
                        )
                        if patch.status_code in (200, 201):
                            patched_ok = True
                    except Exception:
                        pass
                if not patched_ok and os.path.isfile(db_path):
                    import sqlite3
                    conn = sqlite3.connect(db_path, timeout=10)
                    conn.execute("UPDATE project SET label_config = ? WHERE id = ?", (new_xml_str, project_id))
                    conn.commit()
                    conn.close()
                    patched_ok = True
                if patched_ok:
                    logger.info(f"[XML SYNC] Đã bổ sung nhãn {clean_classes} & đồng bộ màu cho Project {project_id}")
        except Exception as e:
            logger.error(f"[XML SYNC ERROR] Project {project_id}: {e}")

    return poly_tag_name, img_tag_name


def _deduplicate_polygons(regions: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], bool]:
    """Khử 2 đối tượng Polygon bị sinh chồng khít lên nhau (IoU khung bao >= 0.85)."""
    cleaned: List[Dict[str, Any]] = []
    kept_polys: List[Tuple[Tuple[float, float, float, float], List[str], int]] = []
    changed = False

    for reg in regions:
        if reg.get("type") != "polygonlabels":
            cleaned.append(reg)
            continue

        poly_bb = _get_poly_bbox(reg)
        labels = reg.get("value", {}).get("polygonlabels", [])
        if poly_bb is None:
            cleaned.append(reg)
            continue

        is_dup = False
        for idx, (k_bb, k_labels, cleaned_pos) in enumerate(kept_polys):
            iou, _ = _bbox_metrics(poly_bb, k_bb)
            if iou >= 0.85:
                same_label = (labels == k_labels)
                one_is_sentinel = any(v in SENTINELS for v in labels) or any(v in SENTINELS for v in k_labels)
                if same_label or one_is_sentinel:
                    if any(v in SENTINELS for v in k_labels) and not any(v in SENTINELS for v in labels):
                        cleaned[cleaned_pos] = reg
                        kept_polys[idx] = (poly_bb, labels, cleaned_pos)
                    is_dup = True
                    changed = True
                    break

        if not is_dup:
            kept_polys.append((poly_bb, labels, len(cleaned)))
            cleaned.append(reg)

    return cleaned, changed

_project_batch_locks_guard = threading.Lock()
_project_batch_locks: Dict[Any, threading.Lock] = {}


def _get_project_batch_lock(project_id: Any) -> threading.Lock:
    with _project_batch_locks_guard:
        if project_id not in _project_batch_locks:
            _project_batch_locks[project_id] = threading.Lock()
        return _project_batch_locks[project_id]


def _reconcile_project_tasks_dual_mode(project_id: int, headers: Dict[str, str]) -> List[int]:
    """
    Hỗ trợ đồng thời cả 2 cơ chế cho người dùng (Xử lý trực tiếp SQLite siêu tốc < 10ms, chống lặp tab Gmail 100%):
    - Cơ chế 1 (Người không chuyên): Upload 50 ảnh lên Web (/data/upload/...) trước -> Import JSON sau.
      => Tự động chuyển nhãn từ task JSON sang task ảnh Upload, xóa task JSON tạm, giữ đúng 1 tab Gmail duy nhất.
    - Cơ chế 2 (Người chuyên): Dùng Cloud Storage (Local files) -> Import JSON.
      => Nếu người dùng lỡ bấm 'Sync Storage' sinh ra task rỗng ($undefined$), tự động xóa sạch task rỗng.
    """
    import sqlite3

    db_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "label_studio.sqlite3"))
    if not os.path.isfile(db_path):
        return []

    merged_real_task_ids: List[int] = []
    try:
        conn = sqlite3.connect(db_path, timeout=15)
        cur = conn.cursor()

        # 1. Xóa ngay các task rỗng "$undefined$" (do bấm nhầm Sync Storage ở Cơ chế 2)
        cur.execute(
            "DELETE FROM task WHERE project_id = ? AND data LIKE '%$undefined$%' AND total_annotations = 0",
            (project_id,),
        )
        if cur.rowcount > 0:
            logger.info(f"[DUAL-MODE SYNC] Đã dọn {cur.rowcount} task rỗng ($undefined$) trong Project {project_id}.")

        # 2. Đọc toàn bộ task của Project để phân loại Ảnh Upload vs Task JSON tạm
        cur.execute("SELECT id, data, meta FROM task WHERE project_id = ?", (project_id,))
        rows = cur.fetchall()

        uploaded_by_name: Dict[str, Tuple[int, str]] = {}
        json_tasks: List[Tuple[int, str, str]] = []

        for tid, data_str, meta_str in rows:
            try:
                data = json.loads(data_str) if data_str else {}
            except Exception:
                data = {}
            img_val = str(data.get("image") or "")
            if "/data/upload/" in img_val or img_val.startswith("upload/"):
                base = os.path.basename(img_val)
                m = UUID_PREFIX_RE.match(base)
                clean_name = m.group(1) if m else base
                uploaded_by_name[clean_name] = (tid, meta_str or "{}")
            else:
                json_tasks.append((tid, img_val, meta_str or "{}"))

        # 3. Nếu có cả Ảnh Upload và Task JSON tạm -> Chuyển nhãn sang Ảnh Upload & xóa Task JSON tạm
        if uploaded_by_name and json_tasks:
            for jt_id, img_val, jt_meta_str in json_tasks:
                clean_val = urllib.parse.unquote(img_val).replace("\\", "/")
                base_name = os.path.basename(clean_val)
                target_info = uploaded_by_name.get(base_name)
                if not target_info:
                    continue
                target_id, target_meta_str = target_info
                if target_id == jt_id:
                    continue

                # Kiểm tra xem task JSON tạm có chứa annotation không
                cur.execute("SELECT id FROM task_completion WHERE task_id = ?", (jt_id,))
                jt_anns = cur.fetchall()
                if jt_anns:
                    # Chuyển trực tiếp liên kết annotation từ jt_id sang target_id
                    cur.execute(
                        "UPDATE task_completion SET task_id = ? WHERE task_id = ?",
                        (target_id, jt_id),
                    )
                    # Gộp meta.bboxes nếu có
                    try:
                        jm = json.loads(jt_meta_str) if jt_meta_str else {}
                        if isinstance(jm, dict) and jm.get("bboxes"):
                            tm = json.loads(target_meta_str) if target_meta_str else {}
                            if not isinstance(tm, dict):
                                tm = {}
                            tm["bboxes"] = jm["bboxes"]
                            cur.execute("UPDATE task SET meta = ? WHERE id = ?", (json.dumps(tm), target_id))
                    except Exception:
                        pass

                    merged_real_task_ids.append(target_id)

                # Xóa các bảng phụ liên quan (nếu có) và xóa task JSON tạm
                cur.execute("DELETE FROM prediction WHERE task_id = ?", (jt_id,))
                cur.execute("DELETE FROM task WHERE id = ?", (jt_id,))

            if merged_real_task_ids:
                logger.info(
                    f"[DUAL-MODE SYNC] Đã ghép nhãn siêu tốc cho {len(merged_real_task_ids)} ảnh Upload & xóa task JSON tạm!"
                )

        # 4. KHỬ TRÙNG LẶP TAB GMAIL: Đảm bảo mỗi ảnh trong Project chỉ có ĐÚNG 1 tab Annotation duy nhất!
        cur.execute("SELECT id FROM task WHERE project_id = ?", (project_id,))
        all_project_tids = [r[0] for r in cur.fetchall()]
        dedup_tabs_removed = 0
        for tid in all_project_tids:
            cur.execute(
                "SELECT id, result FROM task_completion WHERE task_id = ? ORDER BY id ASC",
                (tid,),
            )
            t_anns = cur.fetchall()
            if len(t_anns) > 1:
                # Ưu tiên giữ lại tab đã có polygonlabels, nếu chưa có thì giữ tab đầu tiên
                keep_id = t_anns[0][0]
                for aid, res_str in t_anns:
                    if res_str and '"polygonlabels"' in res_str:
                        keep_id = aid
                        break
                cur.execute(
                    "DELETE FROM task_completion WHERE task_id = ? AND id != ?",
                    (tid, keep_id),
                )
                dedup_tabs_removed += cur.rowcount

            # Đồng bộ lại bộ đếm total_annotations và is_labeled chuẩn xác trên bảng task
            cur.execute("SELECT COUNT(*) FROM task_completion WHERE task_id = ?", (tid,))
            cnt = cur.fetchone()[0]
            cur.execute(
                "UPDATE task SET total_annotations = ?, is_labeled = ? WHERE id = ?",
                (cnt, 1 if cnt > 0 else 0, tid),
            )

        if dedup_tabs_removed > 0:
            logger.info(f"[DUAL-MODE SYNC] Đã xóa {dedup_tabs_removed} tab Gmail bị trùng lặp trong Project {project_id}.")

        conn.commit()
        conn.close()
    except Exception as e:
        logger.warning(f"[DUAL-MODE SYNC] Lỗi xử lý SQLite project {project_id}: {e}")

    return merged_real_task_ids


def _run_batch_on_project_tasks(project_id: Any, task_ids: Optional[List[Any]] = None) -> Dict[str, int]:
    """
    Tự động quét các task vừa Import (hoặc toàn bộ project_id) có chứa Bounding Box,
    chạy MobileSAM cắt viền hàng loạt:
      - Web chỉ hiển thị Polygon (gọn sạch, không đè khung vuông).
      - Toàn bộ Bounding Box gốc được bảo toàn vào task.meta.bboxes & poly.meta.bbox cho Output JSON.
    """
    db_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "label_studio.sqlite3"))
    if not os.path.isfile(db_path) and not _get_ls_token():
        logger.warning("[BATCH IMPORT] Thiếu API Token (LABEL_STUDIO_API_KEY hoặc token.txt)!")
        return {"processed": 0, "polygons": 0}

    # Khóa theo từng Project để tuyệt đối không chạy chồng chéo nhiều luồng làm sinh lặp tab Gmail
    proj_lock = _get_project_batch_lock(project_id)
    if not proj_lock.acquire(blocking=False):
        logger.info(f"[BATCH IMPORT] Project {project_id} đang được xử lý ở luồng khác, bỏ qua luồng trùng.")
        return {"processed": 0, "polygons": 0}

    try:
        headers = _get_ls_headers()
        backend = SAMBackend.get_shared_instance()

        # Đọc tên thẻ PolygonLabels thực tế của project
        tag_info = _ensure_labels_in_project_xml(project_id, set())
        poly_tag = tag_info[0] if tag_info else backend.polygon_tag_name

        # Tự động đồng bộ Dual-Mode (Ghép nhãn vào ảnh Upload, xóa task rỗng & khử trùng lặp tab Gmail)
        _reconcile_project_tasks_dual_mode(project_id, headers)

        # Thu thập danh sách task đầy đủ (kèm chi tiết annotations.result)
        full_tasks: List[Dict[str, Any]] = []
        if os.path.isfile(db_path):
            try:
                import sqlite3
                with sqlite3.connect(db_path, timeout=15) as conn:
                    cur = conn.cursor()
                    cur.execute("SELECT id, data, meta FROM task WHERE project_id = ? ORDER BY id ASC", (project_id,))
                    t_rows = cur.fetchall()
                    for tid, d_str, m_str in t_rows:
                        if task_ids and len(task_ids) == 1 and tid not in task_ids:
                            continue
                        cur.execute(
                            "SELECT id, result, was_cancelled FROM task_completion WHERE task_id = ? ORDER BY id ASC",
                            (tid,),
                        )
                        anns = []
                        for aid, res_str, was_canc in cur.fetchall():
                            try:
                                r_list = json.loads(res_str) if res_str else []
                            except Exception:
                                r_list = []
                            anns.append({"id": aid, "result": r_list, "was_cancelled": bool(was_canc)})
                        full_tasks.append({
                            "id": tid,
                            "project": project_id,
                            "data": json.loads(d_str) if d_str else {},
                            "meta": json.loads(m_str) if m_str else {},
                            "annotations": anns,
                            "predictions": [],
                        })
                logger.info(f"[BATCH IMPORT] Đã nạp siêu tốc {len(full_tasks)} tasks trực tiếp từ SQLite!")
            except Exception as e:
                logger.warning(f"[BATCH IMPORT] Lỗi đọc SQLite, chuyển sang HTTP: {e}")

        if not full_tasks:
            if task_ids and len(task_ids) == 1:
                session = requests.Session()
                for tid in task_ids:
                    try:
                        r = session.get(f"{LS_URL}/api/tasks/{tid}/", headers=headers, timeout=30)
                        if r.status_code == 200:
                            full_tasks.append(r.json())
                    except Exception as e:
                        logger.warning(f"[BATCH IMPORT] Lỗi đọc task {tid}: {e}")
            else:
                try:
                    r = requests.get(
                        f"{LS_URL}/api/projects/{project_id}/export?exportType=JSON",
                        headers=headers,
                        timeout=45,
                    )
                    if r.status_code == 200:
                        full_tasks = r.json()
                        logger.info(f"[BATCH IMPORT] Đã lấy trọn gói {len(full_tasks)} tasks trong 1 request duy nhất!")
                    else:
                        logger.warning(f"[BATCH IMPORT] Export trả về status {r.status_code}: {r.text[:200]}")
                except Exception as e:
                    logger.error(f"[BATCH IMPORT] Lỗi khi lấy trọn gói project {project_id}: {e}")

        processed_count = 0
        total_polys = 0
        all_new_labels: Set[str] = set()

        for task in full_tasks:
            tid = task.get("id")
            if not tid:
                continue

            with _batch_task_lock:
                if tid in _active_batch_tasks:
                    continue
                _active_batch_tasks.add(tid)

            try:
                polys, others, preserved_bboxes, avg_score, found_labels, src_type, src_id = backend.segment_task_boxes(
                    task,
                    polygon_from_name=poly_tag,
                    only_unsegmented_boxes=True,
                )
                if not polys:
                    continue

                all_new_labels.update(found_labels)
                merged_results = others + polys  # Chỉ chứa Polygon để trên Web chỉ hiện Poly

                # Lưu lại đúng nơi chứa Box ban đầu để mở Web ra là thấy ngay viền đa giác
                if src_type == "annotations" and src_id:
                    db_path = os.path.abspath(os.path.join(BASE_DIR, "..", "data", "label_studio.sqlite3"))
                    saved_sqlite = False
                    if os.path.isfile(db_path):
                        try:
                            import sqlite3
                            with sqlite3.connect(db_path, timeout=15) as conn:
                                cur = conn.cursor()
                                cur.execute(
                                    "UPDATE task_completion SET result = ?, result_count = ? WHERE id = ?",
                                    (json.dumps(merged_results), len(merged_results), src_id),
                                )
                                if preserved_bboxes:
                                    cur_meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
                                    cur_meta["bboxes"] = preserved_bboxes
                                    cur.execute(
                                        "UPDATE task SET meta = ? WHERE id = ?",
                                        (json.dumps(cur_meta), tid),
                                    )
                                cur.execute("DELETE FROM prediction WHERE task_id = ?", (tid,))
                                conn.commit()
                                saved_sqlite = True
                        except Exception as sqle:
                            logger.warning(f"[BATCH AUTO-SEG] Fallback HTTP do SQLite bận: {sqle}")

                    if saved_sqlite:
                        processed_count += 1
                        total_polys += len(polys)
                        logger.info(
                            f"[BATCH AUTO-SEG] Task {tid}: Web hiện {len(polys)} Poly & Output giữ đủ {len(preserved_bboxes)} BB"
                        )
                    else:
                        if preserved_bboxes:
                            _persist_bboxes_to_task_meta(
                                tid,
                                preserved_bboxes,
                                headers,
                                current_meta=task.get("meta") if isinstance(task.get("meta"), dict) else None,
                            )
                        with _ann_sync_lock:
                            _active_ann_ids.add(src_id)
                        try:
                            resp = requests.patch(
                                f"{LS_URL}/api/annotations/{src_id}/",
                                headers=headers,
                                json={"result": merged_results},
                                timeout=15,
                            )
                            if resp.status_code in (200, 201):
                                processed_count += 1
                                total_polys += len(polys)
                                logger.info(
                                    f"[BATCH AUTO-SEG] Task {tid}: Web hiện {len(polys)} Poly & Output giữ đủ {len(preserved_bboxes)} BB"
                                )
                        finally:
                            with _ann_sync_lock:
                                _active_ann_ids.discard(src_id)
                else:
                    # Nguồn từ predictions hoặc data.boxes -> lưu Prediction mới và dọn Prediction Box cũ
                    updated_pred = False
                    if src_type == "predictions" and src_id:
                        p_res = requests.patch(
                            f"{LS_URL}/api/predictions/{src_id}/",
                            headers=headers,
                            json={
                                "result": merged_results,
                                "score": avg_score,
                                "model_version": f"{backend.model_version}-batch",
                            },
                            timeout=15,
                        )
                        updated_pred = p_res.status_code in (200, 201)
                        if not updated_pred:
                            requests.delete(f"{LS_URL}/api/predictions/{src_id}/", headers=headers, timeout=10)

                    if not updated_pred:
                        c_res = requests.post(
                            f"{LS_URL}/api/predictions/",
                            headers=headers,
                            json={
                                "task": tid,
                                "project": project_id,
                                "result": merged_results,
                                "score": avg_score,
                                "model_version": f"{backend.model_version}-batch",
                            },
                            timeout=15,
                        )
                        updated_pred = c_res.status_code in (200, 201)

                    if updated_pred:
                        processed_count += 1
                        total_polys += len(polys)
                        logger.info(
                            f"[BATCH AUTO-SEG] Task {tid}: Đã nạp {len(polys)} Poly chờ Review & giữ {len(preserved_bboxes)} BB!"
                        )
            except Exception as e:
                logger.error(f"[BATCH AUTO-SEG ERROR] Task {tid}: {e}")
            finally:
                with _batch_task_lock:
                    _active_batch_tasks.discard(tid)

        if all_new_labels:
            _ensure_labels_in_project_xml(project_id, all_new_labels)

        if processed_count > 0:
            logger.info(
                f"[BATCH HOÀN TẤT] Project {project_id}: Đã tự động bọc viền {processed_count} ảnh ({total_polys} đa giác)!"
            )
        return {"processed": processed_count, "polygons": total_polys}
    finally:
        proj_lock.release()


def _sync_dynamic_label(payload: Dict[str, Any]):
    action = payload.get("action")

    # 1. TỰ ĐỘNG BỌC VIỀN HÀNG LOẠT NGAY KHI BẤM IMPORT FILE JSON BOUNDING BOX LÊN WEB
    if action == "TASKS_CREATED":
        project_id = (payload.get("project") or {}).get("id")
        raw_tasks = payload.get("tasks") or []
        if not project_id and raw_tasks and isinstance(raw_tasks[0], dict):
            project_id = raw_tasks[0].get("project")
        if project_id:
            task_ids = [t["id"] for t in raw_tasks if isinstance(t, dict) and "id" in t]
            time.sleep(1.0)  # Chờ Label Studio ghi xong toàn bộ annotations/predictions vừa Import vào DB
            _run_batch_on_project_tasks(project_id, task_ids=task_ids or None)
        return

    if action not in ("ANNOTATION_CREATED", "ANNOTATION_UPDATED"):
        return

    annotation = payload.get("annotation", {})
    project_id = (payload.get("project") or {}).get("id") or annotation.get("project")
    task_id = annotation.get("task") or (payload.get("task") or {}).get("id")
    ann_id = annotation.get("id")
    results = annotation.get("result", [])
    if not project_id or not ann_id or not results:
        return

    # Nếu Annotation có các Bounding Box chưa được cắt thành Polygon (ví dụ vừa bấm Update trên ảnh mới Import)
    has_polygons = any(r.get("type") == "polygonlabels" for r in results)
    # has_boxes = any(r.get("type") in ("rectanglelabels", "rectangle") for r in results)
    # if has_boxes and not has_polygons and task_id:
    #     _run_batch_on_project_tasks(project_id, task_ids=[task_id])
    #     return

    with _ann_sync_lock:
        if ann_id in _active_ann_ids:
            return
        _active_ann_ids.add(ann_id)

    try:
        name_by_region: Dict[Any, str] = {}
        for r in results:
            if r.get("from_name") in TEXT_FIELDS and r.get("type") == "textarea":
                texts = r.get("value", {}).get("text", [])
                if isinstance(texts, str):
                    texts = [texts]
                if texts and texts[0].strip():
                    region_id = r.get("parentID") or r.get("parent_id") or r.get("id")
                    name_by_region[region_id] = texts[0].strip()

        if not name_by_region and not has_polygons:
            return

        if not _get_ls_token():
            logger.warning("[DYNAMIC LABEL] Chưa có token! Đặt LABEL_STUDIO_API_KEY hoặc file token.txt")
            return

        headers = _get_ls_headers()

        new_classes = set(name_by_region.values()) if name_by_region else set()
        if new_classes:
            _ensure_labels_in_project_xml(project_id, new_classes)

        # Đọc danh sách BB gốc đã lưu trong task.meta.bboxes (nếu có)
        task_meta: Dict[str, Any] = {}
        saved_meta_bboxes: List[Dict[str, Any]] = []
        if task_id:
            try:
                t_res = requests.get(f"{LS_URL}/api/tasks/{task_id}/", headers=headers, timeout=10)
                if t_res.status_code == 200:
                    raw_meta = t_res.json().get("meta")
                    if isinstance(raw_meta, dict):
                        task_meta = raw_meta
                        if isinstance(task_meta.get("bboxes"), list):
                            saved_meta_bboxes = task_meta["bboxes"]
            except Exception:
                pass

        poly_results = [r for r in results if r.get("type") == "polygonlabels"]
        matched_box_ids: Set[int] = set()
        poly_target_label: Dict[int, str] = {}
        poly_prompt_box: Dict[int, Dict[str, Any]] = {}

        for r in results:
            if r.get("type") in ("rectanglelabels", "rectangle"):
                matched_poly = _find_best_matching_polygon(r, poly_results)
                if matched_poly is not None:
                    matched_box_ids.add(id(r))
                    box_id = r.get("id")
                    if box_id in name_by_region:
                        poly_target_label[id(matched_poly)] = name_by_region[box_id]
                    else:
                        box_lbls = r.get("value", {}).get("rectanglelabels") or []
                        if box_lbls and box_lbls[0] not in SENTINELS:
                            poly_target_label[id(matched_poly)] = box_lbls[0]

                    # Giữ lại thông tin tọa độ của hộp mồi thực tế trước khi xóa hộp khỏi giao diện Web
                    r_val = r.get("value") or {}
                    has_real_coords = (
                        "x" in r_val
                        and "y" in r_val
                        and float(r_val.get("width", 0.0)) > 0.0
                        and float(r_val.get("height", 0.0)) > 0.0
                        and "points" not in r_val
                    )
                    if has_real_coords and (
                        id(matched_poly) not in poly_prompt_box or box_id != matched_poly.get("id")
                    ):
                        poly_prompt_box[id(matched_poly)] = copy.deepcopy(r)

        only_name = next(iter(new_classes)) if len(new_classes) == 1 else None
        new_results: List[Dict[str, Any]] = []
        changed = False

        for r in results:
            if r.get("from_name") in TEXT_FIELDS:
                changed = True
                continue

            if id(r) in matched_box_ids:
                changed = True
                continue

            t = r.get("type")
            if t in ("rectanglelabels", "polygonlabels"):
                val = r.setdefault("value", {})
                cur_vals = val.get(t, [])

                target = name_by_region.get(r.get("id")) or poly_target_label.get(id(r))
                if not target and only_name and any(v in SENTINELS for v in cur_vals):
                    target = only_name

                if target and cur_vals != [target]:
                    val[t] = [target]
                    cur_vals = [target]
                    changed = True

                uniq = list(dict.fromkeys(cur_vals))
                if len(uniq) != len(cur_vals):
                    val[t] = uniq
                    changed = True

            new_results.append(r)

        new_results, dedup_changed = _deduplicate_polygons(new_results)
        if dedup_changed:
            changed = True

        # Đảm bảo mọi Polygon sau khi Submit/Update vẫn luôn giữ cặp dữ liệu BB tương ứng trong Output JSON
        bboxes_to_persist: List[Dict[str, Any]] = []
        for r in new_results:
            if r.get("type") != "polygonlabels":
                continue

            p_labels = r.get("value", {}).get("polygonlabels") or ["Object"]
            p_lbl = p_labels[0] if p_labels else "Object"
            cur_p_meta = r.get("meta") if isinstance(r.get("meta"), dict) else {}
            bbox_val = copy.deepcopy(cur_p_meta.get("bbox")) if isinstance(cur_p_meta.get("bbox"), dict) else None
            bbox_id = cur_p_meta.get("bbox_id")

            # 1. Nếu vừa kéo Box mồi mới trên Web -> lấy tọa độ Box mồi đó
            prompt_b = poly_prompt_box.get(id(r))
            if bbox_val is None and prompt_b is not None:
                p_val = copy.deepcopy(prompt_b.get("value") or {})
                p_val["rectanglelabels"] = [p_lbl]
                p_val.setdefault("rotation", 0)
                bbox_val = p_val
                bbox_id = prompt_b.get("id") or f"box_{uuid.uuid4().hex[:8]}"

            # 2. Nếu là Polygon từ file BB JSON đã Import -> khôi phục từ task.meta.bboxes nếu thiếu
            if bbox_val is None and saved_meta_bboxes:
                for sb in saved_meta_bboxes:
                    if _find_best_matching_polygon(sb, [r]) is not None:
                        bbox_val = copy.deepcopy(sb.get("value") or {})
                        bbox_id = sb.get("id")
                        break

            # 3. Nếu vẽ Polygon thuần bằng tay -> tự tính khung bao ngoại tiếp Polygon
            if bbox_val is None:
                p_bb = _get_poly_bbox(r)
                if p_bb is not None:
                    bbox_val = {
                        "x": round(p_bb[0], 4),
                        "y": round(p_bb[1], 4),
                        "width": round(p_bb[2] - p_bb[0], 4),
                        "height": round(p_bb[3] - p_bb[1], 4),
                        "rotation": 0,
                        "rectanglelabels": [p_lbl],
                    }
                    bbox_id = f"box_{uuid.uuid4().hex[:8]}"

            if bbox_val is not None:
                bbox_val["rectanglelabels"] = [p_lbl]
                bbox_id = bbox_id or f"box_{uuid.uuid4().hex[:8]}"
                new_meta = dict(cur_p_meta)
                new_meta["bbox_id"] = bbox_id
                new_meta["bbox"] = bbox_val
                if r.get("meta") != new_meta:
                    r["meta"] = new_meta
                    changed = True

                bboxes_to_persist.append({
                    "id": bbox_id,
                    "from_name": "label",
                    "to_name": r.get("to_name") or "image",
                    "type": "rectanglelabels",
                    "original_width": r.get("original_width"),
                    "original_height": r.get("original_height"),
                    "image_rotation": r.get("image_rotation", 0),
                    "value": copy.deepcopy(bbox_val),
                })

        if task_id and bboxes_to_persist:
            _persist_bboxes_to_task_meta(task_id, bboxes_to_persist, headers, current_meta=task_meta)

        if changed:
            res = requests.patch(
                f"{LS_URL}/api/annotations/{ann_id}/",
                headers=headers,
                json={"result": new_results},
                timeout=10,
            )
            if res.status_code in (200, 201):
                logger.info(
                    f"[DYNAMIC LABEL] Đã giữ sạch Web (chỉ hiện Poly) & bảo toàn BB trong Output cho Annotation {ann_id}!"
                )
    except Exception as e:
        logger.error(f"[DYNAMIC LABEL ERROR] Lỗi xử lý Annotation {ann_id}: {e}")
    finally:
        with _ann_sync_lock:
            _active_ann_ids.discard(ann_id)


# ==============================================================================
# MODULE CHUYỂN ĐỔI FILE JSON OFFLINE (CLI: --batch-input -> --batch-output)
# ==============================================================================
def _normalize_batch_input_json(raw_data: Any) -> Tuple[List[Dict[str, Any]], str]:
    """Nhận diện tự động Label Studio JSON hoặc COCO JSON (xuất từ YOLO/Roboflow)."""
    if isinstance(raw_data, list):
        return [t for t in raw_data if isinstance(t, dict)], "label_studio"

    if isinstance(raw_data, dict):
        if "images" in raw_data and "annotations" in raw_data:
            cat_map = {
                c["id"]: str(c.get("name", "Object"))
                for c in raw_data.get("categories", [])
                if isinstance(c, dict) and "id" in c
            }
            anns_by_img: Dict[Any, List[Dict[str, Any]]] = {}
            for ann in raw_data.get("annotations", []):
                if isinstance(ann, dict) and "image_id" in ann:
                    anns_by_img.setdefault(ann["image_id"], []).append(ann)

            tasks: List[Dict[str, Any]] = []
            for img_info in raw_data.get("images", []):
                if not isinstance(img_info, dict):
                    continue
                img_id = img_info.get("id")
                coco_boxes = [
                    {
                        "bbox_px": [float(v) for v in ann["bbox"]],
                        "label": cat_map.get(ann.get("category_id"), "Object"),
                    }
                    for ann in anns_by_img.get(img_id, [])
                    if isinstance(ann.get("bbox"), (list, tuple)) and len(ann["bbox"]) == 4
                ]
                tasks.append({
                    "data": {"image": img_info.get("file_name", "")},
                    "_coco_boxes": coco_boxes,
                })
            return tasks, "coco"

        if "data" in raw_data:
            return [raw_data], "label_studio"

    raise ValueError("File JSON đầu vào không đúng chuẩn Label Studio JSON hoặc COCO JSON!")


def run_cli_batch_file(
    input_json_path: str,
    output_json_path: str,
    backend: SAMBackend,
    image_dir: Optional[str] = None,
    polygon_tag: Optional[str] = None,
    save_to: str = "predictions",
) -> None:
    """Đọc file JSON chứa Bounding Box -> Chạy MobileSAM hàng loạt -> Xuất JSON có cả BB (meta.bboxes) + Polygon."""
    in_abs = os.path.abspath(input_json_path)
    if not os.path.isfile(in_abs):
        raise FileNotFoundError(f"Không tìm thấy file JSON đầu vào: {in_abs}")

    with open(in_abs, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    tasks, fmt = _normalize_batch_input_json(raw_data)
    logger.info(f"[CLI BATCH] Đã nạp {len(tasks)} ảnh (Định dạng: {fmt.upper()})")

    search_dirs = [os.path.dirname(in_abs)]
    if image_dir:
        search_dirs.insert(0, os.path.abspath(image_dir))

    processed_imgs = 0
    total_polys = 0

    for idx, task in enumerate(tasks, 1):
        try:
            polys, others, preserved_bboxes, avg_score, _, _, _ = backend.segment_task_boxes(
                task,
                polygon_from_name=polygon_tag,
                extra_search_dirs=search_dirs,
                only_unsegmented_boxes=False,
            )
        except Exception as e:
            logger.warning(f"[{idx}/{len(tasks)}] [BỎ QUA] {e}")
            task.pop("_coco_boxes", None)
            continue

        task.pop("_coco_boxes", None)
        if not polys:
            logger.warning(f"[{idx}/{len(tasks)}] [BỎ QUA] Ảnh không có Bounding Box nào.")
            continue

        # Bảo toàn nguyên vẹn Bounding Box gốc vào meta.bboxes của từng task trong Output JSON
        task_meta = task.get("meta") if isinstance(task.get("meta"), dict) else {}
        existing_meta_boxes = task_meta.get("bboxes") if isinstance(task_meta.get("bboxes"), list) else []
        merged_boxes, _ = _merge_bbox_lists(existing_meta_boxes, preserved_bboxes)
        task_meta["bboxes"] = merged_boxes
        task["meta"] = task_meta

        merged = others + polys  # Chỉ đưa Polygon vào result để khi mở trên Web chỉ hiện Poly
        if save_to == "annotations":
            task["annotations"] = [{"result": merged, "score": avg_score}]
        else:
            task["predictions"] = [{
                "model_version": f"{backend.model_version}-batch",
                "score": avg_score,
                "result": merged,
            }]
            task.pop("annotations", None)
            task.pop("completions", None)

        processed_imgs += 1
        total_polys += len(polys)
        img_name = os.path.basename(backend._extract_task_image_url(task))
        logger.info(
            f"[{idx}/{len(tasks)}] Đã bọc viền {len(polys)} Poly & giữ {len(merged_boxes)} BB cho '{img_name}' (IoU TB: {avg_score:.3f})"
        )

    out_abs = os.path.abspath(output_json_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    logger.info(
        f"[CLI BATCH HOÀN TẤT] Đã xử lý {processed_imgs}/{len(tasks)} ảnh, tạo {total_polys} đa giác (kèm BB) -> '{out_abs}'"
    )


# ==============================================================================
# MODULE TỰ ĐỘNG CẤU HÌNH DỰ ÁN MỚI (AUTO PROVISIONING)
# ==============================================================================
AUTO_PROVISION_POLL = int(os.getenv("AUTO_PROVISION_POLL", 3))
CONFIG_FILE_PATH = os.path.join(BASE_DIR, "label_config.xml")
_provisioned_projects: Set[Any] = set()
_project_task_signatures: Dict[Any, Tuple[int, int]] = {}
REQUIRED_WEBHOOK_ACTIONS = ["TASKS_CREATED", "PROJECT_UPDATED", "ANNOTATION_CREATED", "ANNOTATION_UPDATED"]


def _load_blueprint_xml() -> Optional[str]:
    if not os.path.exists(CONFIG_FILE_PATH):
        logger.warning(f"[AUTO PROVISION] Không tìm thấy file {CONFIG_FILE_PATH}")
        return None
    try:
        with open(CONFIG_FILE_PATH, "r", encoding="utf-8") as f:
            content = f.read()
        ET.fromstring(content)
        return content
    except Exception as e:
        logger.error(f"[AUTO PROVISION] Lỗi cú pháp XML trong label_config.xml: {e}")
        return None


def _ensure_project_webhook(p_id: Any, headers: Dict[str, str]):
    """Đảm bảo Project có Webhook lắng nghe cả TASKS_CREATED, PROJECT_UPDATED & ANNOTATION_*."""
    wh_check = requests.get(f"{LS_URL}/api/webhooks/?project={p_id}", headers=headers, timeout=5)
    existing_wh = None
    if wh_check.status_code == 200:
        for item in wh_check.json():
            if "dynamic-webhook" in str(item.get("url")):
                existing_wh = item
                break

    if existing_wh is None:
        requests.post(
            f"{LS_URL}/api/webhooks/",
            headers=headers,
            json={
                "project": p_id,
                "url": "http://127.0.0.1:9090/api/dynamic-webhook",
                "send_payload": True,
                "send_for_all_actions": False,
                "actions": REQUIRED_WEBHOOK_ACTIONS,
                "is_active": True,
            },
            timeout=10,
        )
        logger.info(f"[AUTO PROVISION] Đã kích hoạt Webhook (Import Batch + Nhãn động) cho Project {p_id}!")
    else:
        cur_actions = set(existing_wh.get("actions") or [])
        if not existing_wh.get("send_for_all_actions") and not set(REQUIRED_WEBHOOK_ACTIONS).issubset(cur_actions):
            wh_id = existing_wh.get("id")
            if wh_id:
                requests.patch(
                    f"{LS_URL}/api/webhooks/{wh_id}/",
                    headers=headers,
                    json={
                        "actions": list(cur_actions.union(REQUIRED_WEBHOOK_ACTIONS)),
                        "is_active": True,
                        "send_payload": True,
                    },
                    timeout=10,
                )
                logger.info(f"[AUTO PROVISION] Đã nâng cấp Webhook Project {p_id} hỗ trợ thêm Import Batch!")


def _run_provision_worker():
    """Quét ngầm liên tục để phát hiện dự án mới, ảnh mới Import, đồng bộ màu và gắn kết nối ML/Webhook."""
    logger.info(f"[AUTO PROVISION] Đã kích hoạt quét ngầm (Chu kỳ: {AUTO_PROVISION_POLL}s)")

    while True:
        try:
            headers = _get_ls_headers()
            res = requests.get(f"{LS_URL}/api/projects/?page_size=50", headers=headers, timeout=5)
            if res.status_code == 200:
                projects = res.json().get("results", [])
                for proj in projects:
                    p_id = proj["id"]
                    label_config = (proj.get("label_config") or "").strip()
                    is_empty_config = label_config in ("", "<View></View>", "<View>\n</View>")

                    if not is_empty_config:
                        try:
                            root_xml = ET.fromstring(label_config)
                            if _sync_rect_colors_to_polygons(root_xml):
                                sync_res = requests.patch(
                                    f"{LS_URL}/api/projects/{p_id}/",
                                    headers=headers,
                                    json={"label_config": ET.tostring(root_xml, encoding="unicode")},
                                    timeout=10,
                                )
                                if sync_res.status_code in (200, 201):
                                    logger.info(f"[COLOR SYNC] Đã đồng bộ màu cho Project {p_id}")
                        except Exception as e:
                            logger.warning(f"[COLOR SYNC] Lỗi project {p_id}: {e}")

                    if p_id not in _provisioned_projects or is_empty_config:
                        if is_empty_config:
                            blueprint = _load_blueprint_xml()
                            if blueprint:
                                patch_res = requests.patch(
                                    f"{LS_URL}/api/projects/{p_id}/",
                                    headers=headers,
                                    json={"label_config": blueprint},
                                    timeout=10,
                                )
                                if patch_res.status_code == 200:
                                    logger.info(f"[AUTO PROVISION] Đã tự nạp XML cho Project {p_id}!")

                        try:
                            ml_check = requests.get(f"{LS_URL}/api/ml/?project={p_id}", headers=headers, timeout=5)
                            has_sam = ml_check.status_code == 200 and any(
                                "9090" in str(item.get("url")) for item in ml_check.json()
                            )
                            if not has_sam:
                                requests.post(
                                    f"{LS_URL}/api/ml/",
                                    headers=headers,
                                    json={
                                        "project": p_id,
                                        "url": "http://127.0.0.1:9090",
                                        "title": "MobileSAM Backend",
                                        "is_interactive": True,
                                    },
                                    timeout=10,
                                )
                                logger.info(f"[AUTO PROVISION] Đã kết nối MobileSAM 9090 cho Project {p_id}!")
                        except Exception as e:
                            logger.warning(f"[AUTO PROVISION ML ERROR]: {e}")

                        try:
                            _ensure_project_webhook(p_id, headers)
                        except Exception as e:
                            logger.warning(f"[AUTO PROVISION WEBHOOK ERROR]: {e}")

                        _provisioned_projects.add(p_id)

                    # Tự động phát hiện khi Project có Task mới Import (hoặc vừa khởi động server) để cắt SAM ngay
                    task_num = int(proj.get("task_number") or 0)
                    prev_sig = _project_task_signatures.get(p_id)
                    prev_task_num = prev_sig[0] if isinstance(prev_sig, tuple) else (prev_sig or 0)
                    if task_num > 0 and task_num != prev_task_num:
                        _project_task_signatures[p_id] = (task_num, 0)
                        threading.Thread(
                            target=_run_batch_on_project_tasks,
                            args=(p_id, None),
                            daemon=True,
                        ).start()
                        
        except Exception as e:
            logger.warning(f"[AUTO PROVISION LOOP]: {e}")

        time.sleep(AUTO_PROVISION_POLL)


def start_auto_provision():
    t = threading.Thread(target=_run_provision_worker, daemon=True)
    t.start()


# =============================================================================
# KHỞI CHẠY SERVER (CỔNG 9090) HOẶC CHẠY CHẾ ĐỘ CLI BATCH (--batch-input)
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Label Studio ML Backend - MobileSAM (Interactive + Batch JSON Auto-Segmentation)"
    )
    parser.add_argument("-p", "--port", type=int, default=9090, help="Cổng server (mặc định 9090)")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Địa chỉ host (mặc định 0.0.0.0)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Đường dẫn file .pt của MobileSAM")
    parser.add_argument("--model-type", type=str, default="vit_t", choices=["vit_t", "vit_b", "vit_l", "vit_h"])
    parser.add_argument("--device", type=str, default=None, choices=["cuda", "mps", "cpu"])
    parser.add_argument("--cache-size", type=int, default=10, help="Dung lượng bộ nhớ đệm LRU")
    parser.add_argument("--debug", action="store_true", help="Bật chế độ debug")

    # Nhóm tham số Batch CLI (Chuyển đổi trực tiếp từ file JSON Box -> file JSON Polygon)
    parser.add_argument("--batch-input", type=str, default=None, help="Đường dẫn file JSON chứa Bounding Box đầu vào")
    parser.add_argument("--batch-output", type=str, default="tasks_with_polygons.json", help="File JSON Polygon đầu ra")
    parser.add_argument("--image-dir", type=str, default=None, help="Thư mục ảnh gốc (tùy chọn khi chạy --batch-input)")
    parser.add_argument("--polygon-tag", type=str, default=None, help="Tên thẻ PolygonLabels (mặc định tự đọc từ XML)")
    parser.add_argument(
        "--batch-save-to",
        choices=["predictions", "annotations"],
        default="predictions",
        help="Đóng gói vào 'predictions' (mở Web review rồi bấm Submit) hoặc 'annotations' (hoàn tất luôn)",
    )

    args = parser.parse_args()

    # Nạp trước Singleton MobileSAM 1 lần duy nhất
    shared_backend = SAMBackend(
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        cache_size=args.cache_size,
    )

    # Nếu truyền --batch-input -> Chạy chuyển đổi file JSON hàng loạt rồi thoát
    if args.batch_input:
        run_cli_batch_file(
            input_json_path=args.batch_input,
            output_json_path=args.batch_output,
            backend=shared_backend,
            image_dir=args.image_dir,
            polygon_tag=args.polygon_tag,
            save_to=args.batch_save_to,
        )
        raise SystemExit(0)

    logger.info("=" * 72)
    logger.info("  LABEL STUDIO ML BACKEND - MOBILESAM (INTERACTIVE + BATCH + DYNAMIC)  ")
    logger.info("=" * 72)

    from label_studio_ml.api import init_app

    app = init_app(
        model_class=SAMBackend,
        checkpoint=args.checkpoint,
        model_type=args.model_type,
        device=args.device,
        cache_size=args.cache_size,
    )

    @app.route("/api/dynamic-webhook", methods=["POST"])
    def handle_ls_webhook():
        data = request.get_json(silent=True) or {}
        threading.Thread(target=_sync_dynamic_label, args=(data,), daemon=True).start()
        return jsonify({"status": "ok"}), 200

    @app.route("/api/batch-segment", methods=["POST"])
    def handle_manual_batch_api():
        """API kích hoạt bọc viền hàng loạt theo project_id hoặc danh sách task_ids."""
        body = request.get_json(silent=True) or {}
        project_id = body.get("project_id")
        task_ids = body.get("task_ids")
        if not project_id:
            return jsonify({"error": "Thiếu project_id"}), 400
        threading.Thread(
            target=_run_batch_on_project_tasks,
            args=(project_id, task_ids),
            daemon=True,
        ).start()
        return jsonify({"status": "started", "project_id": project_id}), 202

    @app.route("/api/export-merged", methods=["GET"])
    def handle_export_merged_bbox_and_poly():
        """Tùy chọn: Xuất JSON gộp phẳng cả rectanglelabels (từ meta.bboxes) + polygonlabels vào chung mảng result."""
        project_id = request.args.get("project_id")
        if not project_id:
            return jsonify({"error": "Thiếu tham số ?project_id=..."}), 400
        headers = _get_ls_headers()
        exp_res = requests.get(
            f"{LS_URL}/api/projects/{project_id}/export",
            params={"exportType": "JSON"},
            headers=headers,
            timeout=30,
        )
        if exp_res.status_code != 200:
            return jsonify({"error": exp_res.text}), exp_res.status_code

        exported_tasks = exp_res.json()
        for t in exported_tasks:
            meta_boxes = (t.get("meta") or {}).get("bboxes") or []
            if not meta_boxes:
                continue
            for ann in t.get("annotations") or []:
                res_list = ann.get("result") or []
                existing_box_ids = {r.get("id") for r in res_list if r.get("type") == "rectanglelabels"}
                extra_boxes = [b for b in meta_boxes if b.get("id") not in existing_box_ids]
                ann["result"] = extra_boxes + res_list

        return jsonify(exported_tasks), 200

    start_auto_provision()

    print(f"\n[SERVER READY] MobileSAM (Interactive + Auto-Batch Import) đang chạy tại http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, debug=args.debug)