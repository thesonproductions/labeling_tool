from pathlib import Path


def _repo_root() -> Path:
    # .../labeling/label-studio-src/label_studio/tests/feature_checks/test_label_filter.py
    return Path(__file__).resolve().parents[4]


def test_label_filter_static_asset_and_base_template_wired():
    repo_root = _repo_root()
    static_js = repo_root / 'label-studio-src' / 'label_studio' / 'core' / 'static' / 'js' / 'label_filter_ui.js'
    converter_js = repo_root / 'converters' / 'label_filter_ui.js'
    base_html = repo_root / 'label-studio-src' / 'label_studio' / 'templates' / 'base.html'

    assert static_js.is_file(), f'Missing static asset: {static_js}'
    assert converter_js.is_file(), f'Missing converter asset: {converter_js}'
    assert static_js.read_text(encoding='utf-8') == converter_js.read_text(encoding='utf-8'), (
        'core/static/js/label_filter_ui.js and converters/label_filter_ui.js must stay identical'
    )

    html = base_html.read_text(encoding='utf-8')
    assert "static 'js/label_filter_ui.js'" in html
    assert 'nonce="{{request.csp_nonce}}"' in html
    assert 'LABEL_FILTER_UI_START' not in html

    js_text = static_js.read_text(encoding='utf-8')
    assert 'white-space: normal;' in js_text
    assert 'word-break: break-word;' in js_text
    assert 'display: none;' in js_text


def test_patch_label_filter_skips_duplicate_when_static_tag_present(tmp_path, monkeypatch):
    import sys

    repo_root = _repo_root()
    converters_dir = str(repo_root / 'converters')
    if converters_dir not in sys.path:
        sys.path.insert(0, converters_dir)

    import patch_label_filter

    fake_base = tmp_path / 'base.html'
    original = (
        '<html><body>\n'
        '<script src="{% static \'js/import_folder_ui.js\' %}" nonce="{{request.csp_nonce}}"></script>\n'
        '<script src="{% static \'js/label_filter_ui.js\' %}" nonce="{{request.csp_nonce}}"></script>\n'
        '</body></html>\n'
    )
    fake_base.write_text(original, encoding='utf-8')

    monkeypatch.setattr(patch_label_filter, '_get_base_html_path', lambda: str(fake_base))
    patch_label_filter.ensure_label_filter_patched()

    assert fake_base.read_text(encoding='utf-8') == original
