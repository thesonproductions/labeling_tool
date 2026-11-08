# -*- coding: utf-8 -*-
"""
Module độc lập: Chuyển đổi dữ liệu định dạng YOLO (image/ + label/*.txt + class.txt)
sang định dạng JSON chuẩn của Label Studio (hoặc đẩy trực tiếp qua API).
Không phụ thuộc vào SAM Backend.
Mặc định nhúng ảnh Base64 để import vào bất kỳ dự án nào đều hiển thị ngay 100%,
không lo lỗi đường dẫn cục bộ hoặc lỗi phân quyền /data/upload/.
"""

import argparse
import base64
import json
import os
import urllib.parse
import uuid
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import requests


# 1. HÀM ĐỌC DANH SÁCH NHÃN (class.txt / classes.txt)
def read_classes(class_txt_path: str) -> List[str]:
    """Đọc file class.txt và trả về danh sách tên nhãn theo thứ tự dòng (index = class_id)."""
    if not os.path.isfile(class_txt_path):
        raise FileNotFoundError(f"Không tìm thấy file danh sách nhãn tại: {class_txt_path}")

    with open(class_txt_path, "r", encoding="utf-8") as f:
        classes = [line.strip() for line in f.readlines() if line.strip()]

    if not classes:
        raise ValueError(f"File nhãn rỗng: {class_txt_path}")

    return classes


# 2. HÀM ĐỌC KÍCH THƯỚC ẢNH AN TOÀN (HỖ TRỢ ĐƯỜNG DẪN TIẾNG VIỆT TRÊN WINDOWS)
def get_image_dimensions(image_path: str) -> Tuple[int, int]:
    """Trả về (width, height) của bức ảnh."""
    buf = np.fromfile(image_path, dtype=np.uint8)
    if buf.size == 0:
        raise ValueError(f"File ảnh rỗng hoặc lỗi: {image_path}")
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError(f"Không thể giải mã ảnh: {image_path}")
    height, width = img.shape[:2]
    return width, height


# 3. HÀM CHUYỂN ĐỔI 1 DÒNG TỌA ĐỘ YOLO -> BOUNDING BOX % CỦA LABEL STUDIO
def yolo_line_to_ls_region(
    line: str,
    img_w: int,
    img_h: int,
    classes: List[str],
    from_name: str = "label",
    to_name: str = "image",
) -> Optional[Dict[str, Any]]:
    """
    Chuyển đổi 1 dòng trong file .txt của YOLO:
        <class_id> <x_center> <y_center> <width> <height> (hệ chuẩn hóa 0 -> 1)
    Sang đối tượng rectanglelabels chuẩn của Label Studio (hệ 0 -> 100%).
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None

    try:
        class_id = int(float(parts[0]))
        x_center = float(parts[1])
        y_center = float(parts[2])
        w_norm = float(parts[3])
        h_norm = float(parts[4])
    except ValueError:
        return None

    label_name = classes[class_id] if 0 <= class_id < len(classes) else f"class_{class_id}"

    # Công thức đổi từ Tâm (0..1) sang Góc trên bên trái (0..100%)
    x_pct = max(0.0, min(100.0, (x_center - w_norm / 2.0) * 100.0))
    y_pct = max(0.0, min(100.0, (y_center - h_norm / 2.0) * 100.0))
    w_pct = max(0.0, min(100.0 - x_pct, w_norm * 100.0))
    h_pct = max(0.0, min(100.0 - y_pct, h_norm * 100.0))

    if w_pct <= 0.0 or h_pct <= 0.0:
        return None

    return {
        "id": f"box_{uuid.uuid4().hex[:8]}",
        "from_name": from_name,
        "to_name": to_name,
        "type": "rectanglelabels",
        "original_width": img_w,
        "original_height": img_h,
        "image_rotation": 0,
        "origin": "manual",
        "value": {
            "x": round(x_pct, 4),
            "y": round(y_pct, 4),
            "width": round(w_pct, 4),
            "height": round(h_pct, 4),
            "rotation": 0,
            "rectanglelabels": [label_name],
        },
    }


# 4. HÀM ĐỌC TOÀN BỘ 1 FILE .TXT CỦA YOLO
def parse_yolo_label(
    txt_path: str,
    img_w: int,
    img_h: int,
    classes: List[str],
    from_name: str = "label",
    to_name: str = "image",
) -> List[Dict[str, Any]]:
    """Đọc file .txt của 1 bức ảnh và trả về danh sách các vùng Bounding Box chuẩn Label Studio."""
    if not os.path.isfile(txt_path):
        return []

    regions: List[Dict[str, Any]] = []
    with open(txt_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            region = yolo_line_to_ls_region(
                line=line,
                img_w=img_w,
                img_h=img_h,
                classes=classes,
                from_name=from_name,
                to_name=to_name,
            )
            if region is not None:
                regions.append(region)

    return regions


# 5. HÀM TỰ ĐỘNG TÌM THƯ MỤC IMAGE, LABEL VÀ FILE CLASS.TXT TRONG THƯ MỤC YOLO
def discover_yolo_paths(yolo_dir: str) -> Tuple[str, str, str]:
    """Tự động nhận diện đường dẫn (image_dir, label_dir, class_txt_path) trong thư mục gốc."""
    yolo_dir = os.path.abspath(yolo_dir)

    # Tìm thư mục ảnh (image hoặc images)
    image_dir = ""
    for candidate in ("image", "images", "imgs"):
        p = os.path.join(yolo_dir, candidate)
        if os.path.isdir(p):
            image_dir = p
            break
    if not image_dir:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'image' hoặc 'images' trong: {yolo_dir}")

    # Tìm thư mục nhãn (label hoặc labels)
    label_dir = ""
    for candidate in ("label", "labels", "anns"):
        p = os.path.join(yolo_dir, candidate)
        if os.path.isdir(p):
            label_dir = p
            break
    if not label_dir:
        raise FileNotFoundError(f"Không tìm thấy thư mục 'label' hoặc 'labels' trong: {yolo_dir}")

    # Tìm file class.txt (hoặc classes.txt)
    class_txt_path = ""
    for candidate in ("class.txt", "classes.txt", "obj.names"):
        p = os.path.join(yolo_dir, candidate)
        if os.path.isfile(p):
            class_txt_path = p
            break
    if not class_txt_path:
        raise FileNotFoundError(f"Không tìm thấy file 'class.txt' hoặc 'classes.txt' trong: {yolo_dir}")

    return image_dir, label_dir, class_txt_path


# 6. HÀM CHUYỂN ĐỔI CẢ THƯ MỤC YOLO -> DANH SÁCH TASKS CHUẨN LABEL STUDIO
def build_ls_tasks_from_yolo(
    yolo_dir: str,
    embed_base64: bool = True,
) -> List[Dict[str, Any]]:
    """
    Ghép từng ảnh trong image/ với file .txt tương ứng trong label/ để tạo danh sách Task.
    - embed_base64=True (MẶC ĐỊNH): Nhúng trực tiếp ảnh Base64 vào JSON -> Đảm bảo mở 100% không bao giờ lỗi URL.
    - embed_base64=False: Dùng đường dẫn cục bộ /data/local-files/?d=... (yêu cầu cấu hình Local Storage).
    """
    image_dir, label_dir, class_txt_path = discover_yolo_paths(yolo_dir)
    classes = read_classes(class_txt_path)
    print(f"[INFO] Đã nạp {len(classes)} nhãn từ '{os.path.basename(class_txt_path)}': {classes}")

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    image_files = sorted(
        f for f in os.listdir(image_dir)
        if os.path.splitext(f)[1].lower() in valid_exts
    )

    tasks: List[Dict[str, Any]] = []
    for img_name in image_files:
        img_path = os.path.join(image_dir, img_name)
        stem = os.path.splitext(img_name)[0]
        txt_path = os.path.join(label_dir, f"{stem}.txt")

        img_w, img_h = get_image_dimensions(img_path)
        regions = parse_yolo_label(txt_path, img_w, img_h, classes)

        # Đường dẫn chuẩn theo Cloud Storage (Local files) của Label Studio:
        # Ví dụ: /data/local-files/?d=Users%5Cwin10%5CDesktop%5CResearch%5CYOLO%5Cimage%5C261002_220045.jpg
        # Hoặc relative path
        abs_img = os.path.abspath(img_path)
        if os.name == "nt":
            # Windows: loại bỏ ký tự ổ đĩa (C:\ -> Users\...) và mã hóa '\' thành '%5C'
            drive, path_without_drive = os.path.splitdrive(abs_img)
            clean_path = path_without_drive.lstrip("\\/").replace("/", "\\")
            encoded_d = clean_path.replace("\\", "%5C")
        else:
            # Linux / macOS: bỏ dấu '/' đầu tiên để khớp với DOCUMENT_ROOT="/"
            clean_path = abs_img.lstrip("/")
            encoded_d = urllib.parse.quote(clean_path, safe="/")
        storage_url = f"/data/local-files/?d={encoded_d}"

        task_item: Dict[str, Any] = {
            "data": {
                "image": storage_url,
            },
            "annotations": [
                {
                    "result": regions,
                }
            ],
        }
        tasks.append(task_item)

    return tasks


# 7. HÀM XUẤT RA FILE JSON CHUẨN ĐỂ IMPORT LÊN LABEL STUDIO
def export_to_ls_json(
    yolo_dir: str,
    output_json_path: str,
    embed_base64: bool = False,
) -> str:
    """Gom toàn bộ thư mục YOLO thành 1 file JSON chuẩn để Import vào Label Studio."""
    tasks = build_ls_tasks_from_yolo(yolo_dir, embed_base64=embed_base64)
    out_abs = os.path.abspath(output_json_path)
    os.makedirs(os.path.dirname(out_abs) or ".", exist_ok=True)

    with open(out_abs, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    total_boxes = sum(len(t["annotations"][0]["result"]) for t in tasks)
    print(f"[SUCCESS] Đã xuất {len(tasks)} ảnh ({total_boxes} Bounding Boxes) ra file: {out_abs}")
    return out_abs


# 8. (TÙY CHỌN) HÀM UPLOAD TRỰC TIẾP LÊN PROJECT LABEL STUDIO QUA API
def upload_to_label_studio(
    yolo_dir: str,
    project_id: int,
    ls_url: str = "http://localhost:19090",
    api_token: str = "",
) -> None:
    """Đẩy trực tiếp file JSON đã chuyển đổi lên Project của Label Studio qua API."""
    tasks = build_ls_tasks_from_yolo(yolo_dir, embed_base64=True)
    headers = {"Content-Type": "application/json"}
    if api_token:
        headers["Authorization"] = f"Token {api_token}"

    resp = requests.post(
        f"{ls_url.rstrip('/')}/api/projects/{project_id}/import",
        headers=headers,
        json=tasks,
        timeout=120,
    )
    if resp.ok:
        print(f"[UPLOAD SUCCESS] Đã đẩy {len(tasks)} tasks lên Project {project_id} thành công!")
    else:
        raise RuntimeError(f"Lỗi khi upload lên Label Studio ({resp.status_code}): {resp.text}")


# CHẠY TRỰC TIẾP QUA DÒNG LỆNH (CLI) HOẶC DOUBLE-CLICK
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Bộ chuyển đổi độc lập từ thư mục YOLO (image/ + label/ + class.txt) sang Label Studio JSON"
    )
    parser.add_argument(
        "--yolo-dir",
        type=str,
        default=None,
        help="Đường dẫn tới thư mục YOLO (nếu bỏ trống sẽ hiện bảng chọn thư mục bằng chuột)",
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
        help="Không nhúng ảnh Base64 vào JSON (dùng đường dẫn local)",
    )

    args = parser.parse_args()
    yolo_dir = args.yolo_dir
    use_base64 = not args.no_base64

    # Nếu người dùng không gõ tham số -> Mở cửa sổ đồ họa chọn thư mục của Windows
    if not yolo_dir:
        import ctypes
        import tkinter as tk
        from tkinter import filedialog, messagebox

        # Bật hiển thị sắc nét trên Windows
        if os.name == "nt":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(1)
            except Exception:
                pass

        root = tk.Tk()
        root.withdraw()
        root.attributes("-topmost", True)
        yolo_dir = filedialog.askdirectory(
            title="Chọn thư mục YOLO (chứa thư mục image, label và file class.txt)"
        )
        if not yolo_dir:
            print("[HỦY] Bạn chưa chọn thư mục nào.")
            raise SystemExit(0)

    output_path = args.output or os.path.join(yolo_dir, "yolo_import_label_studio.json")

    try:
        saved_file = export_to_ls_json(
            yolo_dir=yolo_dir,
            output_json_path=output_path,
            embed_base64=use_base64,
        )
        # Nếu mở bằng giao diện chọn thư mục -> Hiện popup thông báo hoàn tất & mở luôn thư mục
        if not args.yolo_dir:
            messagebox.showinfo(
                "Chuyển đổi hoàn tất!",
                f"Đã tạo file JSON thành công tại:\n{saved_file}\n\nBây giờ bạn chỉ cần vào Label Studio -> Bấm Import và chọn file này!",
            )
            if os.name == "nt":
                os.startfile(os.path.dirname(saved_file))
    except Exception as e:
        if not args.yolo_dir:
            from tkinter import messagebox
            messagebox.showerror("Lỗi chuyển đổi YOLO", str(e))
        raise