# Test tính năng Label Studio

Thêm file theo tên `test_<tinh_nang>.py` ở thư mục này. Đặt tên hàm bắt đầu
bằng `test_`. Khi mở hoặc cập nhật PR vào `main`, job **Labeling tests** tự
thu thập các test trong thư mục này và các thư mục con.

Dùng fixture chung trong `tests/conftest.py`, chẳng hạn `business_client` để
test API bằng tài khoản đã đăng nhập. Dùng `@pytest.mark.django_db` khi test
cần database. Tạo dữ liệu thử trong test và dùng `tmp_path` cho file tạm.
Các test này chạy trên SQLite. Mock dịch vụ ngoài khi test logic ứng dụng.

Ví dụ cho tính năng import nhãn, kiểm tra file hợp lệ tạo đúng annotation,
file sai định dạng trả lỗi phù hợp, và dữ liệu cũ vẫn còn nguyên sau lỗi.
Với phân quyền, kiểm tra cả quyền được cấp và request trực tiếp đến project
mà tài khoản không có quyền.

Chạy riêng một file từ thư mục gốc repository:

```bash
cd labeling/label-studio-src
poetry install --only main,test
cd label_studio
PYTHONPATH="$PWD/.." DJANGO_DB=sqlite poetry run python -m pytest tests/feature_checks/test_import_labels.py
```

Thay tên ví dụ bằng file test bạn vừa viết. Khi giao PR, đính kèm các trường
hợp đã kiểm tra. Test này không cần token của hệ thống pilot.
