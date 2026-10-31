# Labeling Tool

Công cụ gán nhãn ảnh dựa trên Label Studio, bổ sung khả năng tạo class mới ngay khi gán nhãn bounding box.

## Cấu trúc

- `labeling/label-studio-src/`: mã nguồn Label Studio.
- `labeling/dynamic_labels/dynamic_labels.py`: webhook xử lý nhãn động.
- `labeling/start_labelstudio.sh`: script khởi chạy trên Linux/Conda.
- `labeling/data/`: database, ảnh upload và dữ liệu export (lưu cục bộ).

## Chạy

Yêu cầu: Python, Label Studio và thư viện `requests`.

Từ thư mục gốc, chạy Label Studio:

```bash
label-studio start --port 19090 --no-browser --data-dir ./labeling/data
```

Trong terminal khác, chạy webhook:

```bash
python labeling/dynamic_labels/dynamic_labels.py
```

- Giao diện: http://localhost:19090
- Webhook health check: http://127.0.0.1:19091/health
- Lưu Label Studio API token vào `labeling/dynamic_labels/token.txt`.
- Trong Label Studio, tạo webhook trỏ đến `http://127.0.0.1:19091/` và bật các sự kiện `ANNOTATION_CREATED`, `ANNOTATION_UPDATED`. Hai dịch vụ cần chạy trên cùng máy để dùng địa chỉ này.

## Nhãn động

Labeling config cần có `RectangleLabels` với `name="label"` và `TextArea` với `name="dynamic_class"`.

Khi một bounding box mang nhãn tạm `__dynamic__` và có tên class mới trong trường `dynamic_class`, webhook sẽ thêm class vào cấu hình project và cập nhật annotation tương ứng.

**Lưu ý:** `start_labelstudio.sh` đang sử dụng đường dẫn `--data-dir` cố định trên máy Linux; chỉnh lại đường dẫn nếu chạy script này ở môi trường khác.
