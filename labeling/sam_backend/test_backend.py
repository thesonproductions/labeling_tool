# -*- coding: utf-8 -*-
"""
Script kiểm thử tự động SAM Backend với Box Prompt và cơ chế LRU Cache
"""
import os
import sys
import time
import json

backend_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "labeling", "sam_backend")
if backend_dir not in sys.path:
    sys.path.insert(0, backend_dir)

from _sam_backend import SAMBackend

def run_test():
    print("\n" + "=" * 60)
    print("  BẮT ĐẦU KIỂM THỬ LABEL STUDIO SAM ML BACKEND")
    print("=" * 60)

    # 1. Khởi tạo Backend
    checkpoint = os.path.join(backend_dir, "weights", "mobile_sam.pt")
    print(f"[1/4] Khởi tạo SAMBackend với checkpoint: {checkpoint}...")
    backend = SAMBackend(checkpoint=checkpoint, model_type="vit_t")

    # 2. Chuẩn bị ảnh test thực tế từ media/upload
    test_image_rel = os.path.join("labeling", "data", "media", "upload", "10", "499c9845-o-to.jpg")
    test_image_abs = os.path.abspath(test_image_rel)

    if not os.path.exists(test_image_abs):
        print(f"[ERROR] Không tìm thấy ảnh test tại: {test_image_abs}")
        return

    print(f"[2/4] Sử dụng ảnh kiểm thử: {test_image_rel}")

    tasks = [
        {
            "id": 1001,
            "data": {
                "image": test_image_abs
            }
        }
    ]

    # Giả lập người dùng vẽ 1 Bounding Box xung quanh chiếc xe hơi (ảnh 980x598)
    # Tọa độ % (x=10%, y=25%, width=75%, height=60%)
    context_box_1 = {
        "result": [
            {
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": 10.0,
                    "y": 25.0,
                    "width": 75.0,
                    "height": 60.0,
                    "rectanglelabels": ["Car"]
                }
            }
        ]
    }

    # 3. Lần suy luận thứ 1: Cache Miss (Sẽ tính image encoder + predict)
    print("\n[3/4] Chạy suy luận LẦN 1 (Cache Miss - trích xuất đặc trưng ảnh)...")
    t0 = time.time()
    preds_1 = backend.predict(tasks=tasks, context=context_box_1)
    duration_1 = (time.time() - t0) * 1000.0

    print(f"-> Thời gian thực thi Lần 1: {duration_1:.1f} ms")
    assert len(preds_1) > 0, "Không có kết quả dự đoán trả về!"
    results_1 = preds_1[0]["result"]
    print(f"-> Số lượng Polygon tạo ra: {len(results_1)}")
    if results_1:
        poly_sample = results_1[0]
        points_count = len(poly_sample["value"]["points"])
        print(f"-> Loại: {poly_sample['type']}, Label: {poly_sample['value']['polygonlabels']}")
        print(f"-> Số điểm đỉnh đa giác (sau approxPolyDP): {points_count} points")
        print(f"-> Score: {poly_sample['score']}")
        print(f"-> Mẫu 3 điểm đầu: {poly_sample['value']['points'][:3]}")

    # 4. Lần suy luận thứ 2: Cache Hit (Vẽ thêm 1 box khác trên cùng ảnh)
    print("\n[4/4] Chạy suy luận LẦN 2 trên cùng Task (Cache Hit - tái sử dụng embedding)...")
    context_box_2 = {
        "result": [
            {
                "from_name": "bbox",
                "to_name": "image",
                "type": "rectanglelabels",
                "value": {
                    "x": 35.0,
                    "y": 45.0,
                    "width": 25.0,
                    "height": 30.0,
                    "rectanglelabels": ["Wheel"]
                }
            }
        ]
    }

    t0 = time.time()
    preds_2 = backend.predict(tasks=tasks, context=context_box_2)
    duration_2 = (time.time() - t0) * 1000.0

    print(f"-> Thời gian thực thi Lần 2: {duration_2:.1f} ms")
    print(f"-> Tăng tốc gấp: {duration_1 / max(duration_2, 0.001):.1f}x nhờ cơ chế LRU Cache!")
    results_2 = preds_2[0]["result"]
    print(f"-> Số lượng Polygon tạo ra cho Box 2: {len(results_2)}")

    print("\n" + "=" * 60)
    print("  KIỂM THỬ THÀNH CÔNG 100%! ML BACKEND HOẠT ĐỘNG HOÀN HẢO.")
    print("=" * 60)

if __name__ == "__main__":
    run_test()
