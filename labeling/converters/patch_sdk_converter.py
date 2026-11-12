"""
Tự động vá nhẹ `label_studio_sdk/converter/converter.py` trên mọi máy tính chạy dự án.
Đảm bảo khi người dùng bấm nút Export -> YOLO trên giao diện Web của Label Studio:
1. Với ảnh từ Cloud Storage Local files (?d=...):
   - Tự động giải mã đường dẫn Windows, copy ảnh vào thư mục `images/` trong file .zip.
   - Trích xuất tên file gốc sạch sẽ cho file `.txt` (chống lỗi [Errno 22] Invalid argument trên Windows).
2. Với ảnh kéo thả trực tiếp (/data/upload/<project_id>/<hash>-<tên_ảnh>):
   - Tự động tìm ảnh trong kho upload của Label Studio, cắt bỏ mã hash 8 ký tự ở đầu.
   - Copy ảnh gốc vào `images/` và đặt tên file `.txt` khớp 100% với tên ảnh gốc.
"""

import os
import sys


PATCH_MARKER = "# Ho tro dac biet cho Local Files va Direct Upload tren Windows:"

OLD_BLOCK_1 = """            for candidate in reversed(image_paths):
                try:
                    if os.path.exists(candidate):"""

NEW_BLOCK_1 = """            for candidate in reversed(image_paths):
                try:
                    # Ho tro dac biet cho Local Files va Direct Upload tren Windows:
                    if "?d=" in candidate:
                        import urllib.parse, shutil
                        _d_val = urllib.parse.parse_qs(urllib.parse.urlparse(candidate).query).get("d", [""])[0]
                        if _d_val:
                            if os.name == "nt":
                                _clean = _d_val.replace("/", "\\\\").lstrip("\\\\")
                                _root = os.environ.get("SystemDrive", "C:") + "\\\\"
                            else:
                                _clean = _d_val.replace("\\\\", "/").lstrip("/")
                                _root = "/"
                            _candidate_real = os.path.join(_root, _clean)
                            if os.path.exists(_candidate_real):
                                _target_img = os.path.join(output_image_dir, os.path.basename(_candidate_real))
                                shutil.copy(_candidate_real, _target_img)
                                resolved_image_path = os.path.relpath(_target_img, output_dir)
                                break
                    elif "/data/upload/" in candidate:
                        import re, shutil
                        _rel_upload = candidate.split("/data/upload/")[-1].replace("/", os.sep)
                        _dyn_roots = [
                            self.upload_dir,
                            os.path.join(self.project_dir, "media", "upload") if getattr(self, "project_dir", None) else None,
                            os.path.abspath(os.path.join(os.getcwd(), "data", "media", "upload")),
                        ]
                        _upload_roots = [r for r in _dyn_roots if r]
                        _found_upload = None
                        for _ur in _upload_roots:
                            _cand_up = os.path.join(_ur, _rel_upload)
                            if os.path.exists(_cand_up):
                                _found_upload = _cand_up
                                break
                            # Neu self.upload_dir da tro toi thu muc con project_id
                            _cand_up2 = os.path.join(_ur, os.path.basename(_rel_upload))
                            if os.path.exists(_cand_up2):
                                _found_upload = _cand_up2
                                break
                        if _found_upload:
                            _clean_img_name = re.sub(r"^[0-9a-fA-F]{8}-", "", os.path.basename(_found_upload))
                            _target_img = os.path.join(output_image_dir, _clean_img_name)
                            shutil.copy(_found_upload, _target_img)
                            resolved_image_path = os.path.relpath(_target_img, output_dir)
                            break

                    if os.path.exists(candidate):"""

OLD_BLOCK_2 = """            # identify label file path
            filename = os.path.splitext(os.path.basename(image_path))[0]"""

NEW_BLOCK_2 = """            # identify label file path
            if "?d=" in image_path:
                import urllib.parse
                _d_val = urllib.parse.parse_qs(urllib.parse.urlparse(image_path).query).get("d", [""])[0]
                if _d_val:
                    filename = os.path.splitext(os.path.basename(_d_val.replace("\\\\", "/")))[0]
                else:
                    filename = os.path.splitext(os.path.basename(image_path))[0]
            else:
                import re
                filename = re.sub(r"^[0-9a-fA-F]{8}-", "", os.path.splitext(os.path.basename(image_path))[0])"""


def ensure_sdk_converter_patched() -> None:
    try:
        import label_studio_sdk.converter.converter as conv_mod
        conv_path = os.path.abspath(conv_mod.__file__)
    except Exception:
        return

    if not os.path.isfile(conv_path):
        return

    try:
        with open(conv_path, "r", encoding="utf-8") as f:
            content = f.read()

        if PATCH_MARKER in content:
            return

        updated = content
        if OLD_BLOCK_1 in updated:
            updated = updated.replace(OLD_BLOCK_1, NEW_BLOCK_1, 1)
        if OLD_BLOCK_2 in updated:
            updated = updated.replace(OLD_BLOCK_2, NEW_BLOCK_2, 1)

        if updated != content:
            with open(conv_path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"[AUTO-PATCH] Da kich hoat ho tro Dual-Mode YOLO Export trong: {conv_path}")
    except Exception as e:
        print(f"[AUTO-PATCH] Bo qua cap nhat converter.py: {e}", file=sys.stderr)


def ensure_all_patches() -> None:
    ensure_sdk_converter_patched()
    try:
        here = os.path.dirname(os.path.abspath(__file__))
        if here not in sys.path:
            sys.path.insert(0, here)
        from patch_label_filter import ensure_label_filter_patched

        ensure_label_filter_patched()
    except Exception as e:
        print(f"[AUTO-PATCH] Bo qua cap nhat Label Filter UI: {e}", file=sys.stderr)


if __name__ == "__main__":
    ensure_all_patches()
