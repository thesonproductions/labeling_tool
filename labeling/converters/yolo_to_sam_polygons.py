# -*- coding: utf-8 -*-
"""
Công cụ Offline Batch: Chuyển đổi toàn bộ thư mục YOLO (image/ + label/*.txt + class.txt)
thành file JSON Label Studio ĐÃ BỌC VIỀN ĐA GIÁC (POLYGON) bằng MobileSAM.
- Chạy 100% offline nội bộ (tận dụng GPU/CPU).
- Không cần bật server Label Studio hay ML Backend.
- File JSON sinh ra Import vào Label Studio là có sẵn viền đa giác ôm sát từng con côn trùng!
"""

import argparse
import base64
import json
import os
import sys
import time
import uuid
import warnings
from typing import Any, Dict, List, Optional, Tuple

# Thiết lập UTF-8 cho Windows Console tránh lỗi UnicodeEncodeError
if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Tắt cảnh báo không cần thiết từ timm / torch
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch

try:
    from mobile_sam import sam_model_registry, SamPredictor
except ImportError:
    print("[LỖI] Chưa cài đặt thư viện 'mobile_sam'. Hãy chạy: pip install git+https://github.com/ChaoningZhang/MobileSAM.git")
    sys.exit(1)


# Đường dẫn file weights mặc định
DEFAULT_WEIGHTS = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "sam_backend", "weights", "mobile_sam.pt")
)


# 1. ĐỌC FILE CLASS.TXT
def read_classes(class_txt_path: str) -> List[str]:
    """Đọc file class.txt và trả về danh sách nhãn theo index dòng."""
    if not os.path.isfile(class_txt_path):
        raise FileNotFoundError(f"Không tìm thấy file nhãn: {class_txt_path}")
    with open(class_txt_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f if line.strip()]
    if not classes:
        raise ValueError(f"File nhãn rỗng: {class_txt_path}")
    return classes


# 2. TỰ ĐỘNG TÌM KIẾM CÁC THƯ MỤC TRONG THƯ MỤC YOLO
def discover_yolo_paths(yolo_dir: str) -> Tuple[str, str, str]:
    """Tự động nhận diện (image_dir, label_dir, class_txt_path)."""
    yolo_dir = os.path.abspath(yolo_dir)

    image_dir = ""
    for candidate in ("image", "images", "imgs"):
        p = os.path.join(yolo_dir, candidate)
        if os.path.isdir(p):
            image_dir = p
            break
    if not image_dir:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'image' trong: {yolo_dir}")

    label_dir = ""
    for candidate in ("label", "labels", "anns"):
        p = os.path.join(yolo_dir, candidate)
        if os.path.isdir(p):
            label_dir = p
            break
    if not label_dir:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'label' trong: {yolo_dir}")

    class_txt_path = ""
    for candidate in ("class.txt", "classes.txt", "obj.names"):
        p = os.path.join(yolo_dir, candidate)
        if os.path.isfile(p):
            class_txt_path = p
            break
    if not class_txt_path:
        raise FileNotFoundError(f"Không tìm thấy file 'class.txt' trong: {yolo_dir}")

    return image_dir, label_dir, class_txt_path


# 3. CHUYỂN MASK NHỊ PHÂN CỦA SAM THÀNH POLYGON CHUẨN % CỦA LABEL STUDIO
def mask_to_polygon_points(
    binary_mask: np.ndarray,
    img_w: int,
    img_h: int,
    approx_epsilon: float = 0.0015,
    min_contour_area: float = 6.0,
) -> Optional[List[List[float]]]:
    """Trích xuất đường viền từ Mask và làm mịn thành danh sách điểm đa giác [0..100%]."""
    mask_u8 = (binary_mask > 0).astype(np.uint8) * 255
    if np.count_nonzero(mask_u8) > 64:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [(cv2.contourArea(c), c) for c in contours if cv2.contourArea(c) >= min_contour_area]
    if not valid_contours:
        return None

    # Lấy contour lớn nhất
    valid_contours.sort(key=lambda pair: pair[0], reverse=True)
    best_c = valid_contours[0][1]

    epsilon = approx_epsilon * cv2.arcLength(best_c, True)
    approx = cv2.approxPolyDP(best_c, epsilon, True)
    pts = approx.reshape(-1, 2)
    if len(pts) < 3:
        pts = best_c.reshape(-1, 2)
        if len(pts) < 3:
            return None

    points_pct: List[List[float]] = []
    for px, py in pts:
        norm_x = max(0.0, min(100.0, round(float(px) / float(img_w) * 100.0, 3)))
        norm_y = max(0.0, min(100.0, round(float(py) / float(img_h) * 100.0, 3)))
        points_pct.append([norm_x, norm_y])

    return points_pct


# 4. CHUYỂN TOÀN BỘ THƯ MỤC YOLO SANG JSON CHỨA SẴN POLYGON
def convert_yolo_to_polygons(
    yolo_dir: str,
    output_json_path: str,
    weights_path: str = DEFAULT_WEIGHTS,
    embed_base64: bool = True,
    device: Optional[str] = None,
) -> str:
    image_dir, label_dir, class_txt_path = discover_yolo_paths(yolo_dir)
    classes = read_classes(class_txt_path)
    print(f"\n[1/4] Đã tìm thấy thư mục YOLO:")
    print(f"      - Ảnh: {image_dir}")
    print(f"      - Nhãn: {label_dir}")
    print(f"      - Danh sách ({len(classes)} nhãn): {classes[:5]}...")

    # Thiết bị tính toán: Tự động nhận diện CUDA (NVIDIA) -> MPS (Apple Silicon) -> CPU
    if not device:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    device_info = device.upper()
    if device == "cuda":
        try:
            device_info = f"CUDA (GPU: {torch.cuda.get_device_name(0)})"
        except Exception:
            device_info = "CUDA (NVIDIA GPU)"
    elif device == "mps":
        device_info = "MPS (Apple Silicon GPU)"

    print(f"\n[2/4] Nạp MobileSAM lên thiết bị: {device_info}...")
    if not os.path.isfile(weights_path):
        raise FileNotFoundError(f"Không tìm thấy file trọng số MobileSAM tại: {weights_path}")

    sam = sam_model_registry["vit_t"](checkpoint=weights_path)
    sam.to(device=device)
    predictor = SamPredictor(sam)
    print(f"      -> Nạp MobileSAM thành công!")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    )
    total_imgs = len(image_files)
    print(f"\n[3/4] Bắt đầu tự động bọc viền {total_imgs} ảnh bằng MobileSAM...")

    tasks: List[Dict[str, Any]] = []
    total_polygons = 0
    start_all = time.time()

    for idx, img_name in enumerate(image_files, 1):
        t0 = time.time()
        img_path = os.path.join(image_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        txt_path = os.path.join(label_dir, f"{stem}.txt")

        # Đọc ảnh an toàn với đường dẫn tiếng Việt trên Windows
        buf = np.fromfile(img_path, dtype=np.uint8)
        if buf.size == 0:
            continue
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            continue
        img_h, img_w = img.shape[:2]

        # Đọc tọa độ Bounding Box từ file YOLO .txt
        raw_boxes: List[Tuple[str, float, float, float, float]] = []
        if os.path.isfile(txt_path):
            with open(txt_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        try:
                            cid = int(float(parts[0]))
                            xc, yc, bw, bh = [float(p) for p in parts[1:5]]
                            lbl = classes[cid] if 0 <= cid < len(classes) else f"class_{cid}"
                            raw_boxes.append((lbl, xc, yc, bw, bh))
                        except Exception:
                            continue

        # Nạp ảnh vào SAM (Image Embedding tính đúng 1 lần/ảnh)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        predictor.set_image(img_rgb)

        polygon_regions: List[Dict[str, Any]] = []
        preserved_bboxes: List[Dict[str, Any]] = []

        for lbl, xc, yc, bw, bh in raw_boxes:
            # Tọa độ pixel góc trên-trái và dưới-phải cho SAM
            x1 = max(0.0, (xc - bw / 2.0) * img_w)
            y1 = max(0.0, (yc - bh / 2.0) * img_h)
            x2 = min(float(img_w), (xc + bw / 2.0) * img_w)
            y2 = min(float(img_h), (yc + bh / 2.0) * img_h)

            # Tọa độ % cho Label Studio Bounding Box
            x_pct = max(0.0, min(100.0, (xc - bw / 2.0) * 100.0))
            y_pct = max(0.0, min(100.0, (yc - bh / 2.0) * 100.0))
            w_pct = max(0.0, min(100.0 - x_pct, bw * 100.0))
            h_pct = max(0.0, min(100.0 - y_pct, bh * 100.0))

            box_id = f"box_{uuid.uuid4().hex[:8]}"
            poly_id = f"poly_{uuid.uuid4().hex[:8]}"

            # Lưu BBox gốc để bảo toàn dữ liệu
            bbox_region = {
                "id": box_id,
                "from_name": "label",
                "to_name": "image",
                "type": "rectanglelabels",
                "original_width": img_w,
                "original_height": img_h,
                "value": {
                    "x": round(x_pct, 4),
                    "y": round(y_pct, 4),
                    "width": round(w_pct, 4),
                    "height": round(h_pct, 4),
                    "rotation": 0,
                    "rectanglelabels": [lbl],
                },
            }
            preserved_bboxes.append(bbox_region)

            # MobileSAM dự đoán Mask dựa trên Bounding Box Prompt
            box_np = np.array([x1, y1, x2, y2], dtype=np.float32)
            try:
                masks, scores, _ = predictor.predict(box=box_np, multimask_output=False)
                if masks is not None and len(masks) > 0:
                    pts = mask_to_polygon_points(masks[0], img_w, img_h)
                    if pts and len(pts) >= 3:
                        poly_region = {
                            "id": poly_id,
                            "from_name": "label",
                            "to_name": "image",
                            "type": "polygonlabels",
                            "original_width": img_w,
                            "original_height": img_h,
                            "score": round(float(scores[0]), 4) if scores is not None and len(scores) > 0 else 0.95,
                            "value": {
                                "points": pts,
                                "polygonlabels": [lbl],
                            },
                            "meta": {
                                "source_box_id": box_id,
                                "bbox": [round(x_pct, 2), round(y_pct, 2), round(w_pct, 2), round(h_pct, 2)],
                            },
                        }
                        polygon_regions.append(poly_region)
                        total_polygons += 1
            except Exception as e:
                pass

        # Xử lý trường đường dẫn ảnh
        if embed_base64:
            ext = os.path.splitext(img_name)[1].lower().lstrip(".")
            mime = "jpeg" if ext in ("jpg", "jpeg") else ext
            with open(img_path, "rb") as bf:
                b64_str = base64.b64encode(bf.read()).decode("utf-8")
            image_field = f"data:image/{mime};base64,{b64_str}"
        else:
            # Dùng đường dẫn tương đối từ Users/... chuẩn Label Studio Local Files
            abs_p = os.path.abspath(img_path).replace("\\", "/")
            if abs_p[1:3] == ":/":
                rel_drive = abs_p[3:]
            else:
                rel_drive = abs_p.lstrip("/")
            image_field = f"/data/local-files/?d={rel_drive}"

        task_item: Dict[str, Any] = {
            "data": {
                "image": image_field,
            },
            # Đưa thẳng Polygon vào annotations để khi mở Web là hiển thị sẵn 100%
            "annotations": [
                {
                    "result": polygon_regions,
                }
            ],
            # Lưu trữ toàn bộ Bounding Box gốc vào meta.bboxes
            "meta": {
                "bboxes": preserved_bboxes,
            },
        }
        tasks.append(task_item)

        elapsed = time.time() - t0
        print(f"      [{idx:02d}/{total_imgs:02d}] {img_name:<20} -> {len(polygon_regions):2d} đa giác ({elapsed:.2f}s)")

    # Lưu file JSON kết quả
    out_abs = os.path.abspath(output_json_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)
    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    total_time = time.time() - start_all
    print(f"\n[4/4] [HOÀN TẤT XUẤT SẮC!]")
    print(f"      - Đã bọc viền: {total_polygons} đa giác trên {len(tasks)} bức ảnh")
    print(f"      - Thời gian: {total_time:.2f}s (Trung bình {total_time/max(1, len(tasks)):.2f}s/ảnh)")
    print(f"      - File JSON xuất ra: {out_abs}")
    return out_abs


# CLI & GUI ENTRY POINT
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Offline Batch: Tự động dùng MobileSAM bọc viền đa giác (Polygon) toàn bộ thư mục YOLO sang Label Studio JSON"
    )
    parser.add_argument(
        "--yolo-dir",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục YOLO (chứa image, label, class.txt)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Đường dẫn file JSON đầu ra",
    )
    parser.add_argument(
        "--no-base64",
        action="store_true",
        help="Không nhúng ảnh Base64 vào JSON (dùng đường dẫn local-files)",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=DEFAULT_WEIGHTS,
        help="Đường dẫn tới file trọng số mobile_sam.pt",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Thiết bị chạy AI ('cuda' nếu có GPU NVIDIA, 'mps' cho Mac, 'cpu' nếu dùng CPU)",
    )

    args = parser.parse_args()
    yolo_dir = args.yolo_dir

    # Nếu người dùng double click vào file BAT -> Hiện popup chọn thư mục bằng chuột
    is_gui = False
    if not yolo_dir:
        is_gui = True

        # Bật chế độ hiển thị sắc nét (High-DPI) trên Windows để cửa sổ không bị mờ
        if os.name == "nt":
            try:
                import ctypes
                # 2 = PROCESS_PER_MONITOR_DPI_AWARE, 1 = PROCESS_SYSTEM_DPI_AWARE
                ctypes.windll.shcore.SetProcessDpiAwareness(2)
            except Exception:
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except Exception:
                    pass

        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        yolo_dir = filedialog.askdirectory(
            title="Chọn thư mục YOLO (chứa thư mục image, label và file class.txt)"
        )
        if not yolo_dir:
            print("[HỦY] Bạn chưa chọn thư mục nào.")
            sys.exit(0)

    out_file = args.output or os.path.join(yolo_dir, "yolo_polygons_label_studio.json")

    try:
        saved_path = convert_yolo_to_polygons(
            yolo_dir=yolo_dir,
            output_json_path=out_file,
            weights_path=args.weights,
            embed_base64=not args.no_base64,
            device=args.device,
        )

        if is_gui:
            from tkinter import messagebox
            messagebox.showinfo(
                "MobileSAM Bọc Viền Thành Công!",
                f"Đã tự động bọc viền toàn bộ Bounding Box thành Polygon!\n\n"
                f"File JSON kết quả đã lưu tại:\n{saved_path}\n\n"
                f"Bây giờ bạn chỉ cần vào Label Studio -> Tạo Project -> Bấm Import file JSON này là có sẵn toàn bộ viền đa giác!",
            )
            if os.name == "nt":
                os.startfile(os.path.dirname(saved_path))
    except Exception as e:
        if is_gui:
            from tkinter import messagebox
            messagebox.showerror("Lỗi bọc viền MobileSAM", str(e))
        print(f"\n[LỖI] {e}")
        sys.exit(1)
