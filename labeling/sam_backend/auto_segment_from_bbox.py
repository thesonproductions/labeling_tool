import os
import json
import cv2
import numpy as np
import torch
from mobile_sam import sam_model_registry, SamPredictor

# ==============================================================================
# 1. CẤU HÌNH THÔNG SỐ VÀ ĐƯỜNG DẪN
# ==============================================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Đường dẫn file checkpoint MobileSAM
CHECKPOINT = os.path.join(BASE_DIR, "weights", "mobile_sam.pt")
if not os.path.exists(CHECKPOINT):
    # Dự phòng nếu script để ở thư mục gốc
    CHECKPOINT = os.path.join(BASE_DIR, "labeling", "sam_backend", "weights", "mobile_sam.pt")

MODEL_TYPE = "vit_t"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Tên file JSON đầu vào và file JSON đầu ra
INPUT_JSON = "tasks_input.json"          # File JSON chứa sẵn Bounding Box của bạn
OUTPUT_JSON = "tasks_with_polygons.json" # File JSON kết quả sau khi SAM bọc viền

# ==============================================================================
# 2. KHỞI TẠO MÔ HÌNH MOBILESAM
# ==============================================================================
print(f"[*] Đang nạp MobileSAM lên thiết bị: {DEVICE}...")
if not os.path.exists(CHECKPOINT):
    raise FileNotFoundError(f"Không tìm thấy file checkpoint tại: {CHECKPOINT}")

sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT)
sam.to(device=DEVICE)
sam.eval()
predictor = SamPredictor(sam)


def mask_to_polygon(binary_mask, img_w, img_h, label_name, approx_epsilon=0.002, min_area=15.0):
    """Chuyển mask nhị phân thành các điểm đa giác polygonlabels chuẩn % của Label Studio."""
    contours, _ = cv2.findContours(
        binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    polys = []
    clean_label = str(label_name[0]) if isinstance(label_name, (list, tuple)) and label_name else str(label_name or "Object")

    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
        eps = approx_epsilon * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, eps, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < 3:
            continue

        points_pct = [
            [round(float(px) / img_w * 100.0, 2), round(float(py) / img_h * 100.0, 2)]
            for px, py in pts
        ]
        polys.append({
            "from_name": "polygon",
            "to_name": "image",
            "type": "polygonlabels",
            "value": {
                "points": points_pct,
                "polygonlabels": [clean_label]
            }
        })
    return polys


def main():
    if not os.path.exists(INPUT_JSON):
        print(f"[!] Không tìm thấy file JSON đầu vào: {INPUT_JSON}")
        print("Vui lòng đặt file JSON chứa Bounding Box cùng thư mục hoặc sửa biến INPUT_JSON.")
        return

    with open(INPUT_JSON, "r", encoding="utf-8") as f:
        tasks = json.load(f)

    print(f"[*] Bắt đầu xử lý {len(tasks)} tasks...")

    for idx, task in enumerate(tasks, 1):
        image_path = task.get("data", {}).get("image", "")

        # Xử lý nếu đường dẫn ảnh là URL cục bộ của Label Studio
        if image_path.startswith("/data/"):
            # Trỏ về thư mục upload media của Label Studio
            possible_path = os.path.join(os.getcwd(), "labeling", "data", "media", image_path.replace("/data/", ""))
            if os.path.exists(possible_path):
                image_path = possible_path

        if not os.path.exists(image_path):
            print(f"[{idx}/{len(tasks)}] [BỎ QUA] Không tìm thấy ảnh: {image_path}")
            continue

        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"[{idx}/{len(tasks)}] [LỖI] Không thể đọc ảnh: {image_path}")
            continue

        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_h, img_w = img_rgb.shape[:2]

        # Nạp ảnh vào SAM để tính Image Embedding một lần duy nhất cho toàn bộ ảnh
        with torch.no_grad():
            predictor.set_image(img_rgb)

        polygon_results = []
        # Lấy danh sách box từ annotations hoặc predictions
        boxes_source = task.get("predictions", []) or task.get("annotations", [])

        for item in boxes_source:
            for res in item.get("result", []):
                if res.get("type") in ("rectanglelabels", "rectangle"):
                    val = res["value"]
                    # Đổi tọa độ từ % sang pixel thực tế
                    x_min = (val["x"] / 100.0) * img_w
                    y_min = (val["y"] / 100.0) * img_h
                    x_max = ((val["x"] + val["width"]) / 100.0) * img_w
                    y_max = ((val["y"] + val["height"]) / 100.0) * img_h

                    labels = val.get("rectanglelabels") or val.get("labels") or ["Object"]
                    label_name = labels[0] if labels else "Object"

                    box_np = np.array([x_min, y_min, x_max, y_max])

                    with torch.no_grad():
                        masks, scores, _ = predictor.predict(box=box_np, multimask_output=False)

                    polys = mask_to_polygon(masks[0] > 0.0, img_w, img_h, label_name)
                    polygon_results.extend(polys)

        # Đóng gói viền đa giác vào task (dưới dạng predictions có sẵn)
        task["predictions"] = [{
            "model_version": "mobilesam_auto_batch",
            "result": polygon_results
        }]
        print(f"[{idx}/{len(tasks)}] Đã sinh xong {len(polygon_results)} đa giác cho ảnh: {os.path.basename(image_path)}")

    # Ghi ra file JSON mới
    with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
        json.dump(tasks, f, ensure_ascii=False, indent=2)

    print(f"\n[✓] THÀNH CÔNG! Đã xuất file kết quả tại: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()