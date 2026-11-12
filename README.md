# Labeling Tool

Công cụ gán nhãn ảnh thông minh dựa trên **Label Studio** kết hợp **MobileSAM AI Backend**, hỗ trợ tự động chuyển đổi Bounding Box sang đa giác (Polygon), tự động thêm nhãn mới (Dynamic Labels) và xuất/nhập chuẩn YOLO.

## Cấu trúc dự án

- `labeling/sam_backend/`: MobileSAM ML Backend (tự động cắt viền Polygon `< 30ms`, đồng bộ nhãn động, hỗ trợ Dual-Mode Import).
- `labeling/converters/`: Công cụ chuyển đổi dữ liệu YOLO ⇄ Label Studio JSON và tự động vá lỗi Export YOLO (`patch_sdk_converter.py`).
- `labeling/dynamic_labels/dynamic_labels.py`: Webhook xử lý nhãn động độc lập.
- `labeling/start_labelstudio.bat` / `.sh`: Script khởi chạy tự động trên Windows và Linux.
- `labeling/data/`: Database, ảnh upload và dữ liệu cấu hình cục bộ.

---

### 🔑 Cấu hình API Token cho lần chạy đầu tiên (Chỉ làm 1 lần - Có thể làm hoặc không ):

Khi mới tải dự án từ Git về, bạn cần cấp Token để SAM Backend giao tiếp được với Label Studio:

1. Mở Web Label Studio (`http://localhost:19090`), đăng nhập tài khoản của bạn.
2. Bấm vào biểu tượng **Avatar góc trên bên phải** ➔ chọn **Account & Settings**.
3. Chọn mục **Access Token** (hoặc **Personal Access Token**) ➔ Copy chuỗi Token của bạn.
4. Tạo một file tên là **`token.txt`** nằm trong thư mục **`labeling/sam_backend/token.txt`** (hoặc `labeling/token.txt`), dán mã Token vừa copy vào đó rồi lưu lại.

> 🔑 **API Token:** Hệ thống đã tích hợp cơ chế tự động trích xuất Token từ database nội bộ ngay sau khi đăng nhập. Bạn không cần tạo file `token.txt` thủ công (chỉ cần tạo `token.txt` nếu bạn dùng database ngoài hoặc môi trường mạng đặc biệt).

## 🚀 Hướng Dẫn Sử Dụng Nhanh (Hỗ Trợ Cả Windows & Linux)

### 1. Khởi động công cụ

- **Trên Windows (Chỉ cần click đúp chuột):**
  1. Chạy `labeling/start_labelstudio.bat` (Mở giao diện Web tại `http://localhost:19090`)
  2. Chạy `labeling/sam_backend/start_sam_backend.bat` (Mở MobileSAM AI Backend tại cổng `9090`)

- **Trên Linux / macOS:**
  ```bash
  bash labeling/start_labelstudio.sh
  # Mở thêm 1 terminal khác:
  bash labeling/sam_backend/start_sam_backend.sh
  ```

> ⚡ **Kết nối AI Tự Động (Auto-Provisioning):**
> MobileSAM Backend chạy ngầm sẽ **tự động kết nối 100%** vào bất kỳ Project mới nào bạn tạo trên Web (kèm đồng bộ bảng màu và kích hoạt chế độ Interactive). Bạn không cần phải thao tác cấu hình thủ công!
>
> 💡 **Lưu ý kiểm tra (Nếu cần):**
> Trường hợp bạn lỡ tạo Project _trước_ khi khởi động file SAM Backend:
>
> - Hãy vào Project ➔ **Settings** ➔ **Model** để kiểm tra (thấy thẻ MobileSAM báo trạng thái xanh `Connected` là xong).
> - Nếu chưa thấy, chỉ cần bấm **Connect Model** ➔ điền URL `http://localhost:9090`, tích chọn **Interactive preannotations** rồi bấm **Save**.

---

### 2. Chuẩn bị dữ liệu (Tạo file JSON từ thư mục YOLO có sẵn)

1. Chuẩn bị thư mục dữ liệu YOLO gồm: thư mục `images/` (hoặc `image/`), `labels/` (hoặc `label/`) và file `class.txt` (hoặc `classes.txt`).
2. Chạy công cụ chuyển đổi:
   - **Windows:** Click đúp `labeling/converters/convert_yolo.bat`
   - **Linux:** Chạy `bash labeling/converters/convert_yolo.sh`.
3. Công cụ sẽ xuất ra file **`yolo_import_label_studio.json`** siêu nhẹ (chỉ vài chục KB).

---

### 3. Nạp dữ liệu lên Web Label Studio (Hỗ trợ song song 2 Cơ Chế)

Bạn có thể chọn **1 trong 2 cách** dưới đây tùy theo nhu cầu:

#### 🟢 Cách 1: Dành cho người mới / Tập ảnh vừa & nhỏ (Không cần cài Cloud Storage)

1. Ngoài giao diện chính của Project, bấm **Import** ➔ Kéo thả trực tiếp **các file ảnh** (`.jpg`, `.png`) vào trước ➔ Bấm **Import**.
2. Tiếp tục bấm **Import** lần 2 ➔ Kéo thả file **`yolo_import_label_studio.json`** vào ➔ Bấm **Import**.
3. ⚡ **Tự động hoàn toàn:** Hệ thống sẽ tự động ghép tọa độ nhãn từ file JSON vào đúng các ảnh vừa tải lên, tự xóa các dòng JSON tạm, giữ đúng 1 tab tài khoản duy nhất và kích hoạt MobileSAM cắt viền Polygon ôm sát vật thể! _(Sau vài giây, bấm **F5** để xem kết quả)._

#### 🔵 Cách 2: Dành cho người chuyên / Tập dữ liệu lớn hàng ngàn ảnh (Dùng Local Storage)

- **Bước 1 (Cấp quyền đọc thư mục ảnh trên ổ cứng):**
  - Mở Project ➔ **Settings** ➔ **Cloud Storage** ➔ **Add Source Storage**.
  - **Bước 1 (Select Provider):** Chọn **Local files**.
  - **Bước 2 (Configure Connection):** Tại **Absolute local path**, điền đường dẫn thư mục ảnh (Ví dụ: `C:\Users\ten_may\Desktop\dataset\images`) ➔ Bấm **Next**.
  - **Bước 3 (Import Settings & Preview):**
    👉 Tại ô **Import Method**, chuyển từ `Tasks` sang chọn **`Files`** _(Automatically creates a task for each storage object (e.g. JPG, MP3, TXT))_.
    _(Có thể bấm "Load Preview" để thấy trước danh sách ảnh)_ ➔ Bấm **Next**.
  - **Bước 4 (Review & Confirm):** Bấm Save Storage .
  - **Bước 5:** Ra ngoài giao diện chính, bấm Import ➔ Kéo thả file yolo_import_label_studio.json vào. (Ảnh từ ổ cứng và nhãn sẽ tự động hiển thị cùng lúc!).

  > ⚠️ **LƯU Ý:** **KHÔNG cần bấm nút "Sync Storage"** nếu bạn dùng kèm file JSON nhãn (hệ thống sẽ tự đọc ảnh qua file JSON ở Bước 2 để tránh trùng lặp).

### 4. Xuất dữ liệu (Export về chuẩn YOLO Segmentation)

- Khi hoàn thiện gán nhãn, bấm nút **Export ➔ YOLO** trên Web.
- Nhờ cơ chế tự động của `patch_sdk_converter.py`, dù bạn nạp ảnh theo **Cách 1** hay **Cách 2**, file `.zip` tải về đều có đầy đủ:
  - 📁 `images/`: Đầy đủ toàn bộ file ảnh gốc.
  - 📁 `labels/`: Các file `.txt` khớp đúng 100% tên với ảnh gốc (đã tự động lọc bỏ mã hash `030ef8d4-` và không bao giờ bị lỗi `[Errno 22] Invalid argument` trên Windows).
  - 📄 `classes.txt`: Danh sách class chuẩn.

---

## 💡 Cách Sử Dụng SAM Trực Tiếp Trên Giao Diện Web (Interactive Labeling)

Nếu bạn muốn gán nhãn thêm các đối tượng mới hoặc chỉnh sửa trực tiếp trên Web:

1. Mở bất kỳ ảnh nào trên giao diện Label Studio.
2. Chọn nhãn cần gán ở danh sách nhãn .
3. Chọn công cụ **Rectangle (Hình chữ nhật)** hoặc nhấn phím tắt `R`.
4. **Kéo 1 khung chữ nhật bao quanh đối tượng:**
   - ⚡ MobileSAM Backend chạy ngầm (`< 30ms`) sẽ tự động phát hiện vật thể bên trong và bọc viền đa giác (**Polygon**) ôm khít đối tượng ngay lập tức!
   - Không cần phải click chấm từng điểm thủ công bằng tay.

---

## 🛠️ Xử Lý Lỗi Thường Gặp (Troubleshooting)

### 1. Trạng thái Cloud Storage báo đỏ `Status: Failed` hoặc lỗi đọc file

- **Nguyên nhân:** Label Studio chưa được mở khóa quyền đọc ổ đĩa cục bộ, hoặc bấm nhầm `Sync Storage` ở chế độ JSON.
- **Cách xử lý:** Đảm bảo bạn khởi động Label Studio bằng file `start_labelstudio.bat` (Windows) hoặc `start_labelstudio.sh` (Linux). Khi dùng Cloud Storage kèm file JSON, chỉ cần bấm **Save Storage**, không bấm **Sync Storage**.

### 2. Vừa Import xong chưa thấy Polygon hiện ngay

- **Nguyên nhân:** Khi bạn nạp 50–100 ảnh cùng lúc, MobileSAM chạy ngầm ở Background mất khoảng vài chục giây để cắt viền toàn bộ ảnh.
- **Cách xử lý:** Chờ khoảng 10–30 giây (có thể xem tiến độ trên cửa sổ Terminal của SAM Backend) rồi nhấn **F5 (Tải lại trang Web)**, toàn bộ nhãn đa giác sẽ hiển thị sạch đẹp.

---

## ⌨️ Phím Tắt Nhanh Hữu Ích Trên Web

- `1`, `2`, `3`...: Chọn nhanh class tương ứng trong danh sách nhãn.
- `R`: Chọn công cụ Bounding Box (kết hợp SAM để tự động vẽ viền đa giác).
- `Ctrl + Enter`: Lưu (Submit / Update) nhãn hiện tại.
- `Ctrl + Z`: Hoàn tác thao tác vừa vẽ.
