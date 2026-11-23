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
THUMB_START_MARKER = "<!-- DM_THUMBNAIL_OPT_START -->"
THUMB_END_MARKER = "<!-- DM_THUMBNAIL_OPT_END -->"

DM_THUMBNAIL_JS = r"""(function () {
  if (window.__lsDmThumbOptLoaded) return;
  window.__lsDmThumbOptLoaded = true;

  function toThumbUrl(src) {
    if (typeof src !== "string" || src.indexOf("/data/upload/") === -1 || /[?&]thumb=/.test(src)) {
      return src;
    }
    return src + (src.indexOf("?") === -1 ? "?" : "&") + "thumb=1&w=240";
  }

  function isDataManagerThumbImg(img) {
    if (!img || img.tagName !== "IMG") return false;
    if (img.closest && img.closest(".lsf-image, [class*='lsf-image'], .konvajs-content, [class*='main-view__annotation']")) {
      return false;
    }
    if (img.getAttribute("alt") === "Data") return true;
    if (img.style && img.style.maxWidth === "100px") return true;
    if (img.closest && img.closest("[class*='annotation-preview']")) return true;
    return false;
  }

  function optimizeImg(img) {
    if (!isDataManagerThumbImg(img)) return;
    var cur = img.getAttribute("src") || "";
    if (cur.indexOf("/data/upload/") === -1 || /[?&]thumb=/.test(cur)) return;
    img.setAttribute("loading", "lazy");
    img.setAttribute("decoding", "async");
    img.setAttribute("src", toThumbUrl(cur));
  }

  function scanNode(node) {
    if (!node || node.nodeType !== 1) return;
    if (node.tagName === "IMG") {
      optimizeImg(node);
      return;
    }
    if (node.querySelectorAll) {
      var imgs = node.querySelectorAll("img[src*='/data/upload/']");
      for (var i = 0; i < imgs.length; i++) {
        optimizeImg(imgs[i]);
      }
    }
  }

  var observer = new MutationObserver(function (mutations) {
    for (var i = 0; i < mutations.length; i++) {
      var m = mutations[i];
      if (m.type === "attributes" && m.target && m.target.tagName === "IMG") {
        optimizeImg(m.target);
      } else if (m.addedNodes && m.addedNodes.length) {
        for (var j = 0; j < m.addedNodes.length; j++) {
          scanNode(m.addedNodes[j]);
        }
      }
    }
  });

  function start() {
    if (document.body) {
      scanNode(document.body);
      observer.observe(document.body, {
        childList: true,
        subtree: true,
        attributes: true,
        attributeFilter: ["src"]
      });
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", start);
  } else {
    start();
  }
})();"""

UPLOAD_CACHE_MARKER = "_UPLOAD_CACHE_CONTROL = 'private, max-age=86400'"

OLD_UPLOADED_FILE_GET = """        if not file_upload.has_permission(request.user):
            return Response(status=status.HTTP_403_FORBIDDEN)

        file = file_upload.file
        if file.storage.exists(file.name):
            content_type, encoding = mimetypes.guess_type(str(file.name))
            content_type = content_type or 'application/octet-stream'
            return RangedFileResponse(request, file.open(mode='rb'), content_type=content_type)

        return Response(status=status.HTTP_404_NOT_FOUND)"""

NEW_UPLOADED_FILE_GET = """        if file_upload is None:
            return Response(status=status.HTTP_404_NOT_FOUND)

        if not file_upload.has_permission(request.user):
            return Response(status=status.HTTP_403_FORBIDDEN)

        file = file_upload.file
        if not file.storage.exists(file.name):
            return Response(status=status.HTTP_404_NOT_FOUND)

        import hashlib
        import os
        import threading
        from pathlib import Path
        from django.http import HttpResponseNotModified

        _UPLOAD_CACHE_CONTROL = 'private, max-age=86400'
        content_type, _ = mimetypes.guess_type(str(file.name))
        content_type = content_type or 'application/octet-stream'

        is_local = isinstance(file.storage, FileSystemStorage)
        if is_local:
            st = os.stat(file.path)
            size, mtime_ns = st.st_size, st.st_mtime_ns
        else:
            size, mtime_ns = getattr(file, 'size', 0), 0

        cache_key = hashlib.sha256(f'{file.name}:{size}:{mtime_ns}'.encode('utf-8')).hexdigest()[:24]

        def _etag_matches(header_val, etag_val):
            if not header_val:
                return False
            for part in header_val.split(','):
                tok = part.strip()
                if tok == '*':
                    return True
                if tok.startswith('W/'):
                    tok = tok[2:].strip()
                if tok == etag_val:
                    return True
            return False

        wants_thumb = str(request.GET.get('thumb', '')).strip().lower() in ('1', 'true', 'yes')
        can_thumb = (
            wants_thumb
            and is_local
            and content_type.startswith('image/')
            and content_type not in ('image/svg+xml', 'image/gif')
        )

        if can_thumb:
            try:
                max_dim = max(64, min(512, int(request.GET.get('w', 240))))
            except (TypeError, ValueError):
                max_dim = 240
            thumb_etag = f'"{cache_key}-thumb-{max_dim}"'
            if _etag_matches(request.META.get('HTTP_IF_NONE_MATCH', ''), thumb_etag):
                not_mod = HttpResponseNotModified()
                not_mod['ETag'] = thumb_etag
                not_mod['Cache-Control'] = _UPLOAD_CACHE_CONTROL
                return not_mod

            thumb_dir = Path(settings.MEDIA_ROOT) / '.thumbs'
            thumb_path = thumb_dir / f'{cache_key}_{max_dim}.jpg'
            ready = False
            try:
                ready = thumb_path.is_file() and thumb_path.stat().st_size > 0
            except OSError:
                ready = False

            if not ready:
                tmp_path = None
                try:
                    from PIL import Image, ImageOps
                    thumb_dir.mkdir(parents=True, exist_ok=True)
                    with Image.open(file.path) as im:
                        im = ImageOps.exif_transpose(im)
                        if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                            rgba = im.convert('RGBA')
                            bg = Image.new('RGB', rgba.size, (255, 255, 255))
                            bg.paste(rgba, mask=rgba.split()[-1])
                            im = bg
                        elif im.mode != 'RGB':
                            im = im.convert('RGB')
                        im.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
                        tmp_path = thumb_dir / f'.{cache_key}_{max_dim}.{os.getpid()}.{threading.get_ident()}.tmp.jpg'
                        im.save(tmp_path, format='JPEG', quality=80, optimize=True)
                        os.replace(tmp_path, thumb_path)
                        tmp_path = None
                    ready = True
                except Exception:
                    if tmp_path is not None:
                        try:
                            tmp_path.unlink(missing_ok=True)
                        except OSError:
                            pass

            if ready:
                response = RangedFileResponse(request, open(thumb_path, 'rb'), content_type='image/jpeg')
                response['ETag'] = thumb_etag
                response['Cache-Control'] = _UPLOAD_CACHE_CONTROL
                return response

        full_etag = f'"{cache_key}-full"'
        if _etag_matches(request.META.get('HTTP_IF_NONE_MATCH', ''), full_etag):
            not_mod = HttpResponseNotModified()
            not_mod['ETag'] = full_etag
            not_mod['Cache-Control'] = _UPLOAD_CACHE_CONTROL
            return not_mod

        response = RangedFileResponse(request, file.open(mode='rb'), content_type=content_type)
        response['ETag'] = full_etag
        response['Cache-Control'] = _UPLOAD_CACHE_CONTROL
        return response"""


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

    # Dong bo them toi uu Thumbnail Data Manager cho moi truong local prebuilt (Windows/Linux site-packages)
    try:
        with open(base_html_path, "r", encoding="utf-8") as f:
            content3 = f.read()
        # Bo qua tren Docker / label-studio-src vi ImageCell.jsx da duoc build truc tiep
        if not ("import_folder_ui.js" in content3 and IMPORT_START_MARKER not in content3):
            thumb_block = (
                f"{THUMB_START_MARKER}\n"
                f'<script nonce="{{{{request.csp_nonce}}}}">\n'
                f"{DM_THUMBNAIL_JS}\n"
                f"</script>\n"
                f"{THUMB_END_MARKER}"
            )
            thumb_pat = re.compile(
                re.escape(THUMB_START_MARKER) + r".*?" + re.escape(THUMB_END_MARKER),
                re.DOTALL,
            )
            if thumb_pat.search(content3):
                updated3 = thumb_pat.sub(lambda _: thumb_block, content3, count=1)
            elif "</body>" in content3:
                updated3 = content3.replace("</body>", f"{thumb_block}\n</body>", 1)
            else:
                updated3 = content3 + "\n" + thumb_block
            if updated3 != content3:
                with open(base_html_path, "w", encoding="utf-8") as f:
                    f.write(updated3)
                print(f"[AUTO-PATCH] Da kich hoat toi uu Thumbnail Data Manager trong: {base_html_path}")
    except Exception as e:
        print(f"[AUTO-PATCH] Bo qua cap nhat Thumbnail Data Manager UI: {e}", file=sys.stderr)

    # Dong bo them backend API /api/projects/<pk>/import-folder va thumbnail/cache cho moi truong local (site-packages)
    try:
        ls_dir = os.path.dirname(os.path.dirname(base_html_path))
        src_folder_api = os.path.normpath(
            os.path.join(here, "..", "label-studio-src", "label_studio", "data_import", "folder_import.py")
        )
        dst_data_import = os.path.join(ls_dir, "data_import")
        dst_folder_api = os.path.join(dst_data_import, "folder_import.py")
        dst_urls = os.path.join(dst_data_import, "urls.py")
        dst_api = os.path.join(dst_data_import, "api.py")

        if os.path.isfile(dst_api):
            with open(dst_api, "r", encoding="utf-8") as f:
                api_content = f.read()
            if UPLOAD_CACHE_MARKER not in api_content:
                api_norm = api_content.replace("\r\n", "\n")
                if OLD_UPLOADED_FILE_GET in api_norm:
                    api_updated = api_norm.replace(OLD_UPLOADED_FILE_GET, NEW_UPLOADED_FILE_GET, 1)
                    if "\r\n" in api_content:
                        api_updated = api_updated.replace("\n", "\r\n")
                    with open(dst_api, "w", encoding="utf-8") as f:
                        f.write(api_updated)
                    print(f"[AUTO-PATCH] Da kich hoat Thumbnail & HTTP Cache trong: {dst_api}")

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
