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
IMPORT_START_MARKER = "<!-- IMPORT_FOLDER_UI_START -->"
IMPORT_END_MARKER = "<!-- IMPORT_FOLDER_UI_END -->"


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
    static_js_path = os.path.normpath(
        os.path.join(
            here,
            "..",
            "label-studio-src",
            "label_studio",
            "core",
            "static",
            "js",
            "label_filter_ui.js",
        )
    )
    js_path = static_js_path if os.path.isfile(static_js_path) else os.path.join(here, "label_filter_ui.js")
    if not os.path.isfile(js_path):
        return

    try:
        base_html_path = _get_base_html_path()
    except Exception:
        return

    if not os.path.isfile(base_html_path):
        return

    try:
        with open(base_html_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Neu base.html da nhung truc tiep qua static tag (Docker / label-studio-src) thi khong tiem trung lap
        if not ("label_filter_ui.js" in content and START_MARKER not in content):
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

    # Dong bo them nut Import Folder UI cho moi truong local (site-packages) neu chua co san tag static
    try:
        import_js_path = os.path.normpath(
            os.path.join(
                here,
                "..",
                "label-studio-src",
                "label_studio",
                "core",
                "static",
                "js",
                "import_folder_ui.js",
            )
        )
        if not os.path.isfile(import_js_path):
            return
        with open(base_html_path, "r", encoding="utf-8") as f:
            content2 = f.read()
        if "import_folder_ui.js" in content2 and IMPORT_START_MARKER not in content2:
            return
        with open(import_js_path, "r", encoding="utf-8") as f:
            import_js = f.read().strip().replace("</script>", "<\\/script>")
        import_block = (
            f"{IMPORT_START_MARKER}\n"
            f'<script nonce="{{{{request.csp_nonce}}}}">\n'
            f"{import_js}\n"
            f"</script>\n"
            f"{IMPORT_END_MARKER}"
        )
        imp_pat = re.compile(
            re.escape(IMPORT_START_MARKER) + r".*?" + re.escape(IMPORT_END_MARKER),
            re.DOTALL,
        )
        if imp_pat.search(content2):
            updated2 = imp_pat.sub(lambda _: import_block, content2, count=1)
        elif "</body>" in content2:
            updated2 = content2.replace("</body>", f"{import_block}\n</body>", 1)
        else:
            updated2 = content2 + "\n" + import_block
        if updated2 != content2:
            with open(base_html_path, "w", encoding="utf-8") as f:
                f.write(updated2)
            print(f"[AUTO-PATCH] Da kich hoat Import Folder UI trong: {base_html_path}")
    except Exception as e:
        print(f"[AUTO-PATCH] Bo qua cap nhat Import Folder UI: {e}", file=sys.stderr)

    # Dong bo them backend API /api/projects/<pk>/import-folder cho moi truong local (site-packages)
    try:
        ls_dir = os.path.dirname(os.path.dirname(base_html_path))
        src_folder_api = os.path.normpath(
            os.path.join(here, "..", "label-studio-src", "label_studio", "data_import", "folder_import.py")
        )
        dst_data_import = os.path.join(ls_dir, "data_import")
        dst_folder_api = os.path.join(dst_data_import, "folder_import.py")
        dst_urls = os.path.join(dst_data_import, "urls.py")

        if os.path.isfile(src_folder_api) and os.path.isdir(dst_data_import):
            same_file = os.path.isfile(dst_folder_api) and os.path.samefile(src_folder_api, dst_folder_api)
            if not same_file:
                with open(src_folder_api, "r", encoding="utf-8") as f:
                    src_code = f.read()
                old_code = ""
                if os.path.isfile(dst_folder_api):
                    with open(dst_folder_api, "r", encoding="utf-8") as f:
                        old_code = f.read()
                if src_code != old_code:
                    with open(dst_folder_api, "w", encoding="utf-8") as f:
                        f.write(src_code)
                    print(f"[AUTO-PATCH] Da dong bo folder_import.py vao: {dst_folder_api}")

        if os.path.isfile(dst_urls):
            with open(dst_urls, "r", encoding="utf-8") as f:
                urls_content = f.read()
            if "project-import-folder" not in urls_content:
                urls_updated = urls_content
                if "from . import api\n" in urls_updated:
                    urls_updated = urls_updated.replace("from . import api\n", "from . import api, folder_import\n", 1)
                elif "from . import api\r\n" in urls_updated:
                    urls_updated = urls_updated.replace("from . import api\r\n", "from . import api, folder_import\r\n", 1)
                target_route = "path('<int:pk>/import', api.ImportAPI.as_view(), name='project-import'),"
                new_route = (
                    target_route
                    + "\n    path('<int:pk>/import-folder', folder_import.ImportFolderAPI.as_view(), name='project-import-folder'),"
                )
                if target_route in urls_updated:
                    urls_updated = urls_updated.replace(target_route, new_route, 1)
                if urls_updated != urls_content:
                    with open(dst_urls, "w", encoding="utf-8") as f:
                        f.write(urls_updated)
                    print(f"[AUTO-PATCH] Da kich hoat API /import-folder trong: {dst_urls}")
    except Exception as e:
        print(f"[AUTO-PATCH] Bo qua cap nhat API /import-folder: {e}", file=sys.stderr)


if __name__ == "__main__":
    ensure_label_filter_patched()
