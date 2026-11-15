"""
Tự động tiêm tiện ích Lọc hiển thị theo nhãn/loài (Checkbox Filter UI)
vào giao diện Web của Label Studio (`templates/base.html`) trên mọi máy tính.

- Hoạt động đồng bộ cả Box (Rectangle) và Segment (Polygon) thông qua `window.Htx`.
- Không sửa mã nguồn cốt lõi hay làm mất dữ liệu khi Submit / Export YOLO.
"""

import importlib.util
import os
import re
import sys

START_MARKER = "<!-- LABEL_FILTER_UI_START -->"
END_MARKER = "<!-- LABEL_FILTER_UI_END -->"


def _get_base_html_path() -> str:
    spec = importlib.util.find_spec("label_studio")
    if spec and spec.submodule_search_locations:
        ls_dir = os.path.abspath(spec.submodule_search_locations[0])
        return os.path.join(ls_dir, "templates", "base.html")
    import label_studio

    ls_dir = os.path.dirname(os.path.abspath(label_studio.__file__))
    return os.path.join(ls_dir, "templates", "base.html")


def ensure_label_filter_patched() -> None:
    here = os.path.dirname(os.path.abspath(__file__))
    js_path = os.path.join(here, "label_filter_ui.js")
    if not os.path.isfile(js_path):
        return

    try:
        base_html_path = _get_base_html_path()
    except Exception:
        return

    if not os.path.isfile(base_html_path):
        return

    try:
        with open(js_path, "r", encoding="utf-8") as f:
            js_code = f.read().strip()

        # Đảm bảo không có chuỗi đóng thẻ script trần bên trong JS
        js_code = js_code.replace("</script>", "<\\/script>")

        inject_block = (
            f"{START_MARKER}\n"
            f'<script nonce="{{{{request.csp_nonce}}}}">\n'
            f"{js_code}\n"
            f"</script>\n"
            f"{END_MARKER}"
        )

        with open(base_html_path, "r", encoding="utf-8") as f:
            content = f.read()

        pattern = re.compile(
            re.escape(START_MARKER) + r".*?" + re.escape(END_MARKER),
            re.DOTALL,
        )

        if pattern.search(content):
            updated = pattern.sub(lambda _: inject_block, content, count=1)
        elif "</body>" in content:
            updated = content.replace("</body>", f"{inject_block}\n</body>", 1)
        else:
            updated = content + "\n" + inject_block

        if updated != content:
            with open(base_html_path, "w", encoding="utf-8") as f:
                f.write(updated)
            print(f"[AUTO-PATCH] Da kich hoat bo loc nhan (Label Filter UI) trong: {base_html_path}")
    except Exception as e:
        print(f"[AUTO-PATCH] Bo qua cap nhat Label Filter UI: {e}", file=sys.stderr)


if __name__ == "__main__":
    ensure_label_filter_patched()
