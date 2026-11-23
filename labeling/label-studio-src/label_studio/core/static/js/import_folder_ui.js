(function () {
  "use strict";

  if (window.__lsImportFolderMounted) {
    return;
  }
  window.__lsImportFolderMounted = true;

  var IMAGE_EXTS = {
    ".jpg": true,
    ".jpeg": true,
    ".png": true,
    ".webp": true,
    ".bmp": true,
    ".tif": true,
    ".tiff": true
  };

  var BATCH_SIZE_IMAGES = 25;
  var BATCH_MAX_BYTES = 50 * 1024 * 1024; // 50 MB/batch (well below Nginx 200m & Django 250m)
  var BATCH_SIZE_LABELS = 80; // <= 81 files incl. classes.txt (safely below Django DATA_UPLOAD_MAX_NUMBER_FILES=100)
  var MAX_RETRIES = 2;
  var MAX_PREVIEW_ROWS = 100;

  var styleEl = null;
  var triggerBtn = null;
  var modalBackdrop = null;
  var activeController = null;
  var cancelRequested = false;
  var isUploading = false;
  var isScanning = false;
  var parsedSelection = null;
  var hiddenDirInput = null;
  var onSelectionCallback = null;

  function getProjectRouteInfo() {
    var pathname = String(window.location.pathname || "");
    var m = pathname.match(/^(.*?)\/projects\/(\d+)\/data\/?$/);
    if (!m) {
      return null;
    }
    var prefix = m[1] ? m[1].replace(/\/+$/, "") : "";
    return {
      projectId: parseInt(m[2], 10),
      basePrefix: prefix
    };
  }

  function getProjectIdFromUrl() {
    var info = getProjectRouteInfo();
    return info ? info.projectId : null;
  }

  function buildImageBatches(images, maxCount, maxBytes) {
    var batches = [];
    var current = [];
    var currentBytes = 0;
    for (var i = 0; i < images.length; i++) {
      var f = images[i];
      var fSize = Number(f && f.size) || 0;
      if (current.length > 0 && (current.length >= maxCount || currentBytes + fSize > maxBytes)) {
        batches.push(current);
        current = [];
        currentBytes = 0;
      }
      current.push(f);
      currentBytes += fSize;
    }
    if (current.length > 0) {
      batches.push(current);
    }
    return batches;
  }

  function isEditorActive() {
    var search = String(window.location.search || "");
    if (/[?&]task=\d+/.test(search)) {
      return true;
    }
    var canvasDom = document.querySelector(".lsf-main-view, [class*='main-view']");
    return Boolean(canvasDom && canvasDom.offsetParent !== null);
  }

  function getCsrfToken() {
    var match = document.cookie.match(/(?:^|;\s*)csrftoken=([^;]+)/);
    if (match) {
      return decodeURIComponent(match[1]);
    }
    var inp = document.querySelector("input[name='csrfmiddlewaretoken']");
    return inp ? inp.value : "";
  }

  function getExt(name) {
    var idx = name.lastIndexOf(".");
    return idx >= 0 ? name.slice(idx).toLowerCase() : "";
  }

  function getStem(name) {
    var idx = name.lastIndexOf(".");
    return idx >= 0 ? name.slice(0, idx) : name;
  }

  function sanitizeFolderName(name) {
    var s = String(name || "").trim().replace(/[\\/:*?"<>|]+/g, "_").replace(/\.\.+/g, "_");
    return s || "imported_folder";
  }

  function formatNumber(n) {
    try {
      return Number(n).toLocaleString("en-US");
    } catch (e) {
      return String(n);
    }
  }

  function formatBytes(bytes) {
    var b = Number(bytes || 0);
    if (b < 1024) return b + " B";
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + " KB";
    return (b / (1024 * 1024)).toFixed(1) + " MB";
  }

  function getRelativePath(file) {
    return String(file.__relPath || file.webkitRelativePath || file.name || "").replace(/\\/g, "/");
  }

  function ensureStyles() {
    if (styleEl && document.head && document.head.contains(styleEl)) {
      return;
    }
    styleEl = document.createElement("style");
    styleEl.id = "ls-import-folder-styles";
    styleEl.textContent = [
      "#ls-import-folder-btn.ls-if-toolbar-fallback {",
      "  display: inline-flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "  height: 32px;",
      "  padding: 0 12px;",
      "  border-radius: 4px;",
      "  border: 1px solid var(--color-neutral-border, #d1d5db);",
      "  background: var(--color-neutral-surface, #ffffff);",
      "  color: var(--color-neutral-content, #1f2937);",
      "  font-family: inherit;",
      "  font-size: 13px;",
      "  font-weight: 500;",
      "  line-height: 1;",
      "  white-space: nowrap;",
      "  cursor: pointer;",
      "  transition: background-color 0.15s ease, border-color 0.15s ease;",
      "}",
      "#ls-import-folder-btn.ls-if-toolbar-fallback:hover {",
      "  background: var(--color-neutral-surface-hover, #f3f4f6);",
      "}",
      "[data-color-scheme='dark'] #ls-import-folder-btn.ls-if-toolbar-fallback {",
      "  border-color: var(--color-neutral-border, #3f3f46);",
      "  background: var(--color-neutral-surface, #1f2023);",
      "  color: var(--color-neutral-content, #f3f4f6);",
      "}",
      "[data-color-scheme='dark'] #ls-import-folder-btn.ls-if-toolbar-fallback:hover {",
      "  background: var(--color-neutral-surface-hover, #2b2d31);",
      "}",
      "#ls-import-folder-btn { white-space: nowrap; flex-shrink: 0; }",

      /* Native-matched Label Studio Fullscreen Modal (Modal.scss --fullscreen-size-reduction: 80px) */
      "#ls-if-backdrop {",
      "  position: fixed;",
      "  inset: 0;",
      "  z-index: 10050;",
      "  background: rgba(var(--color-neutral-shadow-raw, 0 0 0) / 50%);",
      "  backdrop-filter: blur(2px);",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "  padding: 40px;",
      "  box-sizing: border-box;",
      "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Inter, Arial, sans-serif;",
      "}",
      "#ls-if-modal {",
      "  width: calc(100vw - 80px);",
      "  height: calc(100vh - 80px);",
      "  max-width: 100%;",
      "  max-height: 100%;",
      "  display: flex;",
      "  flex-direction: column;",
      "  border-radius: var(--modal-border-radius, 8px);",
      "  background: var(--color-neutral-background, #ffffff);",
      "  color: var(--color-neutral-content, #1f2937);",
      "  border: 1px solid var(--color-neutral-border, #dcdfe4);",
      "  box-shadow: 0 16px 36px rgba(0, 0, 0, 0.42);",
      "  overflow: hidden;",
      "}",
      "[data-color-scheme='dark'] #ls-if-modal {",
      "  background: var(--color-neutral-background, #1d1d1f);",
      "  color: var(--color-neutral-content, #f3f4f6);",
      "  border-color: var(--color-neutral-border, #343538);",
      "}",

      /* Top Header matching Import Data (Modal.scss: padding 1rem 2rem) */
      ".ls-if-header {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  min-height: 40px;",
      "  padding: 16px 32px;",
      "  border-bottom: 1px solid var(--color-neutral-border, rgba(128, 128, 128, 0.22));",
      "  flex-shrink: 0;",
      "}",
      ".ls-if-title {",
      "  font-size: 16px;",
      "  font-weight: 500;",
      "  letter-spacing: -0.01em;",
      "}",
      ".ls-if-header-actions {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 12px;",
      "}",

      /* Buttons matching Label Studio Import Data */
      ".ls-if-btn {",
      "  display: inline-flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "  gap: 8px;",
      "  height: 36px;",
      "  padding: 0 16px;",
      "  border-radius: 4px;",
      "  border: 1px solid var(--color-neutral-border, #d1d5db);",
      "  background: transparent;",
      "  color: inherit;",
      "  font-family: inherit;",
      "  font-size: 14px;",
      "  font-weight: 500;",
      "  cursor: pointer;",
      "  transition: background-color 0.14s ease, border-color 0.14s ease, opacity 0.14s ease;",
      "}",
      ".ls-if-btn-select {",
      "  height: 40px;",
      "  padding: 0 18px;",
      "  font-size: 14px;",
      "}",
      ".ls-if-btn:disabled { opacity: 0.45; cursor: not-allowed; }",
      ".ls-if-btn-cancel {",
      "  border-color: rgba(224, 92, 74, 0.55);",
      "  color: #dc5b4a;",
      "}",
      "[data-color-scheme='dark'] .ls-if-btn-cancel {",
      "  border-color: rgba(235, 111, 93, 0.55);",
      "  color: #ee7b6b;",
      "}",
      ".ls-if-btn-cancel:hover:not(:disabled) {",
      "  background: rgba(224, 92, 74, 0.1);",
      "}",
      ".ls-if-btn-primary {",
      "  background: var(--color-primary-surface, #5b74e3);",
      "  border-color: var(--color-primary-surface, #5b74e3);",
      "  color: #ffffff;",
      "}",
      ".ls-if-btn-primary:hover:not(:disabled) {",
      "  background: #4d66d6;",
      "  border-color: #4d66d6;",
      "}",
      ".ls-if-btn-accent {",
      "  border-color: rgba(91, 116, 227, 0.55);",
      "  color: #4a63d4;",
      "  background: rgba(91, 116, 227, 0.05);",
      "}",
      "[data-color-scheme='dark'] .ls-if-btn-accent {",
      "  border-color: rgba(114, 137, 245, 0.55);",
      "  color: #889ff8;",
      "  background: rgba(91, 116, 227, 0.08);",
      "}",
      ".ls-if-btn-accent:hover:not(:disabled) {",
      "  background: rgba(91, 116, 227, 0.15);",
      "}",
      ".ls-if-btn-ghost {",
      "  border-color: transparent;",
      "  opacity: 0.78;",
      "  padding: 0 10px;",
      "}",
      ".ls-if-btn-ghost:hover:not(:disabled) {",
      "  opacity: 1;",
      "  background: rgba(128, 128, 128, 0.12);",
      "}",

      /* Action Strip below header (Import.scss: padding 24px 32px) */
      ".ls-if-toolbar {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  gap: 14px;",
      "  padding: 24px 32px;",
      "  flex-shrink: 0;",
      "}",
      ".ls-if-toolbar-left {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 14px;",
      "  min-width: 0;",
      "}",
      ".ls-if-toolbar-hint {",
      "  font-size: 14px;",
      "  opacity: 0.7;",
      "  overflow: hidden;",
      "  text-overflow: ellipsis;",
      "  white-space: nowrap;",
      "}",

      /* Body (Import.scss: padding 0 32px 32px) */
      ".ls-if-body {",
      "  padding: 0 32px 32px;",
      "  overflow-y: auto;",
      "  display: flex;",
      "  flex-direction: column;",
      "  gap: 16px;",
      "  flex: 1;",
      "}",

      /* Dropzone matching Label Studio Import Data (Import.scss) */
      ".ls-if-dropzone {",
      "  flex: 1;",
      "  min-height: 360px;",
      "  border-radius: var(--corner-radius-small, 4px);",
      "  border: 1px solid var(--color-primary-border-subtler, #d5dff8);",
      "  background: var(--color-primary-background, #f4f7ff);",
      "  display: flex;",
      "  flex-direction: column;",
      "  align-items: center;",
      "  justify-content: center;",
      "  padding: 36px 24px;",
      "  text-align: center;",
      "  cursor: pointer;",
      "  transition: border-color 0.15s ease, background-color 0.15s ease, box-shadow 0.15s ease;",
      "}",
      "[data-color-scheme='dark'] .ls-if-dropzone {",
      "  border-color: var(--color-primary-border-subtler, rgba(91, 116, 227, 0.28));",
      "  background: var(--color-primary-background, #0d1426);",
      "}",
      ".ls-if-dropzone:hover {",
      "  border-color: rgba(91, 116, 227, 0.7);",
      "}",
      ".ls-if-dropzone.is-dragover {",
      "  border-color: #5b74e3;",
      "  box-shadow: inset 0 0 0 1px #5b74e3;",
      "  background: #eaf0ff;",
      "}",
      "[data-color-scheme='dark'] .ls-if-dropzone.is-dragover {",
      "  background: #131d38;",
      "}",
      ".ls-if-dz-icon {",
      "  color: var(--color-primary-icon, #5b74e3);",
      "  margin-bottom: 24px;",
      "}",
      ".ls-if-dz-heading {",
      "  font-size: 24px;",
      "  font-weight: 600;",
      "  line-height: 32px;",
      "  margin-bottom: 28px;",
      "  color: var(--color-neutral-content, inherit);",
      "}",
      ".ls-if-dz-table {",
      "  display: grid;",
      "  grid-template-columns: auto 1fr;",
      "  column-gap: 30px;",
      "  row-gap: 4px;",
      "  text-align: left;",
      "  font-size: 14px;",
      "  line-height: 24px;",
      "  max-width: 620px;",
      "}",
      ".ls-if-dz-key { font-weight: 600; color: var(--color-neutral-content, inherit); white-space: nowrap; }",
      ".ls-if-dz-val { opacity: 0.76; }",

      /* Review 2-column workspace */
      ".ls-if-workspace {",
      "  display: grid;",
      "  grid-template-columns: minmax(0, 1.35fr) minmax(260px, 0.85fr);",
      "  gap: 16px;",
      "  align-items: start;",
      "}",
      "@media (max-width: 760px) {",
      "  .ls-if-workspace { grid-template-columns: 1fr; }",
      "}",
      ".ls-if-stats {",
      "  display: grid;",
      "  grid-template-columns: repeat(3, 1fr);",
      "  gap: 10px;",
      "  margin-bottom: 12px;",
      "}",
      ".ls-if-stat {",
      "  padding: 12px 14px;",
      "  border-radius: 6px;",
      "  border: 1px solid var(--color-neutral-border, rgba(128, 128, 128, 0.24));",
      "  background: rgba(128, 128, 128, 0.05);",
      "}",
      ".ls-if-stat-label { font-size: 12.5px; opacity: 0.72; margin-bottom: 4px; }",
      ".ls-if-stat-value { font-size: 20px; font-weight: 600; font-variant-numeric: tabular-nums; }",
      ".ls-if-stat-value.is-accent { color: #5b74e3; }",
      "[data-color-scheme='dark'] .ls-if-stat-value.is-accent { color: #889ff8; }",

      /* File Preview Table */
      ".ls-if-table-wrap {",
      "  border-radius: 6px;",
      "  border: 1px solid var(--color-neutral-border, rgba(128, 128, 128, 0.24));",
      "  overflow: hidden;",
      "}",
      ".ls-if-table-head {",
      "  display: grid;",
      "  grid-template-columns: 1fr 86px 80px;",
      "  padding: 9px 14px;",
      "  font-size: 12.5px;",
      "  font-weight: 600;",
      "  opacity: 0.75;",
      "  background: rgba(128, 128, 128, 0.08);",
      "  border-bottom: 1px solid var(--color-neutral-border, rgba(128, 128, 128, 0.2));",
      "}",
      ".ls-if-table-body {",
      "  max-height: calc(100vh - 380px);",
      "  min-height: 220px;",
      "  overflow-y: auto;",
      "}",
      ".ls-if-table-row {",
      "  display: grid;",
      "  grid-template-columns: 1fr 86px 80px;",
      "  align-items: center;",
      "  padding: 7px 14px;",
      "  font-size: 13.5px;",
      "  border-bottom: 1px solid rgba(128, 128, 128, 0.1);",
      "}",
      ".ls-if-table-row:last-child { border-bottom: none; }",
      ".ls-if-file-name {",
      "  overflow: hidden;",
      "  text-overflow: ellipsis;",
      "  white-space: nowrap;",
      "  padding-right: 8px;",
      "}",
      ".ls-if-file-size { font-size: 12.5px; opacity: 0.72; font-variant-numeric: tabular-nums; }",
      ".ls-if-tag {",
      "  display: inline-flex;",
      "  align-items: center;",
      "  justify-content: center;",
      "  padding: 2px 8px;",
      "  border-radius: 999px;",
      "  font-size: 11.5px;",
      "  font-weight: 500;",
      "  width: fit-content;",
      "  background: rgba(128, 128, 128, 0.14);",
      "  opacity: 0.85;",
      "}",
      ".ls-if-tag.is-yolo {",
      "  background: rgba(91, 116, 227, 0.18);",
      "  color: #4a63d4;",
      "  opacity: 1;",
      "}",
      "[data-color-scheme='dark'] .ls-if-tag.is-yolo { color: #9ab0ff; }",
      ".ls-if-table-more {",
      "  padding: 8px 14px;",
      "  font-size: 13px;",
      "  opacity: 0.72;",
      "  text-align: center;",
      "  background: rgba(128, 128, 128, 0.05);",
      "}",

      /* Settings side panel */
      ".ls-if-settings {",
      "  padding: 16px 18px;",
      "  border-radius: 6px;",
      "  border: 1px solid var(--color-neutral-border, rgba(128, 128, 128, 0.24));",
      "  background: rgba(128, 128, 128, 0.04);",
      "  display: flex;",
      "  flex-direction: column;",
      "  gap: 16px;",
      "}",
      ".ls-if-field { display: flex; flex-direction: column; gap: 7px; }",
      ".ls-if-label { font-size: 13.5px; font-weight: 600; }",
      ".ls-if-muted { font-size: 12.5px; opacity: 0.7; line-height: 1.45; word-break: break-all; }",
      ".ls-if-input, .ls-if-select {",
      "  width: 100%;",
      "  height: 38px;",
      "  padding: 0 12px;",
      "  border-radius: 4px;",
      "  border: 1px solid var(--color-neutral-border, rgba(128, 128, 128, 0.35));",
      "  background: var(--color-neutral-surface, transparent);",
      "  color: inherit;",
      "  font-family: inherit;",
      "  font-size: 14px;",
      "  box-sizing: border-box;",
      "}",
      ".ls-if-input:focus, .ls-if-select:focus {",
      "  outline: none;",
      "  border-color: #5b74e3;",
      "}",
      "[data-color-scheme='dark'] .ls-if-input, [data-color-scheme='dark'] .ls-if-select {",
      "  background: #141518;",
      "  color: #f3f4f6;",
      "}",
      ".ls-if-check {",
      "  display: flex;",
      "  align-items: flex-start;",
      "  gap: 8px;",
      "  font-size: 13.5px;",
      "  cursor: pointer;",
      "  line-height: 1.45;",
      "}",
      ".ls-if-check input[type='checkbox'] {",
      "  margin-top: 3px;",
      "  accent-color: #5b74e3;",
      "  cursor: pointer;",
      "}",

      /* Progress Dock */
      ".ls-if-progress-dock {",
      "  padding: 14px 18px;",
      "  border-radius: 6px;",
      "  border: 1px solid rgba(91, 116, 227, 0.35);",
      "  background: rgba(91, 116, 227, 0.06);",
      "  display: flex;",
      "  flex-direction: column;",
      "  gap: 8px;",
      "}",
      ".ls-if-progress-top {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  gap: 10px;",
      "  font-size: 13.5px;",
      "  font-weight: 500;",
      "}",
      ".ls-if-progress-wrap {",
      "  width: 100%;",
      "  height: 7px;",
      "  border-radius: 99px;",
      "  background: rgba(128, 128, 128, 0.22);",
      "  overflow: hidden;",
      "}",
      ".ls-if-progress-bar {",
      "  height: 100%;",
      "  width: 0%;",
      "  background: #5b74e3;",
      "  transition: width 0.2s ease;",
      "}",
      ".ls-if-status { font-size: 12.5px; line-height: 1.5; white-space: pre-line; opacity: 0.9; }"
    ].join("\n");
    document.head.appendChild(styleEl);
  }

  function analyzeSelectedFiles(fileList) {
    var files = Array.prototype.slice.call(fileList || []);
    if (!files.length) {
      return null;
    }

    var firstRel = getRelativePath(files[0]);
    var rootFolder = firstRel.indexOf("/") >= 0 ? firstRel.split("/")[0] : "imported_folder";

    var validImages = [];
    var zeroByteImages = [];
    var labelFiles = [];
    var labelStemSet = {};
    var classesFile = null;
    var nonImageSkipped = 0;

    // Detect if there is an images/ subfolder (YOLO layout)
    var hasImagesSubdir = files.some(function (f) {
      var rel = getRelativePath(f).toLowerCase();
      return /(?:^|\/)images\/[^/]+$/.test(rel);
    });

    for (var i = 0; i < files.length; i++) {
      var f = files[i];
      var rel = getRelativePath(f);
      var relLower = rel.toLowerCase();
      var baseLower = String(f.name || "").toLowerCase();
      var ext = getExt(baseLower);

      if (baseLower === "classes.txt" || baseLower === "class.txt") {
        if (!classesFile) {
          classesFile = f;
        }
        continue;
      }

      if (ext === ".txt" && /(?:^|\/)labels\/[^/]+\.txt$/.test(relLower)) {
        labelFiles.push(f);
        labelStemSet[getStem(baseLower)] = true;
        continue;
      }

      if (IMAGE_EXTS[ext]) {
        // If YOLO images/ subdir exists, only import images inside images/ to avoid duplicate thumbs
        if (hasImagesSubdir && !/(?:^|\/)images\/[^/]+$/.test(relLower)) {
          nonImageSkipped += 1;
          continue;
        }
        if (f.size <= 0) {
          zeroByteImages.push(f.name);
        } else {
          validImages.push(f);
        }
      } else {
        nonImageSkipped += 1;
      }
    }

    validImages.sort(function (a, b) {
      return getRelativePath(a).localeCompare(getRelativePath(b));
    });

    // Disambiguate duplicate basenames across nested subfolders (e.g. setA/img.jpg & setB/img.jpg)
    var nameCounts = {};
    for (var nc = 0; nc < validImages.length; nc++) {
      var bn = String(validImages[nc].name || "").toLowerCase();
      nameCounts[bn] = (nameCounts[bn] || 0) + 1;
    }
    var usedUploadNames = {};
    for (var ui = 0; ui < validImages.length; ui++) {
      var imgFile = validImages[ui];
      var origName = String(imgFile.name || "");
      var candidate = origName;
      if (nameCounts[origName.toLowerCase()] > 1) {
        var relParts = getRelativePath(imgFile).split("/").filter(Boolean);
        // Drop leading root folder and trailing filename to extract subfolder chain
        var subParts = relParts.length > 2 ? relParts.slice(1, -1) : [];
        if (subParts.length > 0) {
          var prefix = sanitizeFolderName(subParts.join("_"));
          candidate = prefix + "__" + origName;
        }
      }
      var finalName = candidate;
      var suffix = 2;
      while (usedUploadNames[finalName.toLowerCase()]) {
        var extPart = getExt(candidate);
        var stemPart = getStem(candidate);
        finalName = stemPart + "_" + suffix + extPart;
        suffix += 1;
      }
      usedUploadNames[finalName.toLowerCase()] = true;
      try {
        imgFile.__uploadName = finalName;
      } catch (e) {}
    }

    return {
      rootFolder: sanitizeFolderName(rootFolder),
      validImages: validImages,
      zeroByteImages: zeroByteImages,
      labelFiles: labelFiles,
      labelStemSet: labelStemSet,
      classesFile: classesFile,
      nonImageSkipped: nonImageSkipped,
      isYoloFolder: Boolean(labelFiles.length > 0 || classesFile)
    };
  }

  function readAllDirectoryEntries(dirReader) {
    return new Promise(function (resolve, reject) {
      var allEntries = [];
      function readBatch() {
        dirReader.readEntries(function (entries) {
          if (!entries || !entries.length) {
            resolve(allEntries);
            return;
          }
          for (var i = 0; i < entries.length; i++) {
            allEntries.push(entries[i]);
          }
          readBatch();
        }, reject);
      }
      readBatch();
    });
  }

  function fileFromEntry(fileEntry) {
    return new Promise(function (resolve) {
      fileEntry.file(function (file) {
        try {
          file.__relPath = String(fileEntry.fullPath || file.name || "").replace(/^\/+/, "");
        } catch (e) {}
        resolve(file);
      }, function () {
        resolve(null);
      });
    });
  }

  async function collectDroppedFiles(dataTransfer, onProgress) {
    var collected = [];
    var items = dataTransfer && dataTransfer.items ? Array.prototype.slice.call(dataTransfer.items) : [];
    var entries = [];

    for (var i = 0; i < items.length; i++) {
      if (typeof items[i].webkitGetAsEntry === "function") {
        var entry = items[i].webkitGetAsEntry();
        if (entry) {
          entries.push(entry);
        }
      }
    }

    if (!entries.length) {
      return Array.prototype.slice.call((dataTransfer && dataTransfer.files) || []);
    }

    var queue = entries.slice();
    var qIdx = 0;
    while (qIdx < queue.length) {
      var current = queue[qIdx++];
      if (current.isFile) {
        var f = await fileFromEntry(current);
        if (f) {
          collected.push(f);
          if (onProgress && collected.length % 100 === 0) {
            onProgress(collected.length);
          }
        }
      } else if (current.isDirectory) {
        try {
          var children = await readAllDirectoryEntries(current.createReader());
          for (var c = 0; c < children.length; c++) {
            queue.push(children[c]);
          }
        } catch (e) {}
      }
    }
    return collected;
  }

  async function postWithRetry(url, formData, retries) {
    var lastErr = null;
    for (var attempt = 0; attempt <= retries; attempt++) {
      if (cancelRequested) {
        throw new Error("CANCELLED");
      }
      try {
        activeController = typeof AbortController !== "undefined" ? new AbortController() : null;
        var resp = await fetch(url, {
          method: "POST",
          body: formData,
          credentials: "same-origin",
          headers: {
            "X-CSRFToken": getCsrfToken()
          },
          signal: activeController ? activeController.signal : undefined
        });
        return resp;
      } catch (err) {
        if (cancelRequested || (err && err.name === "AbortError")) {
          throw new Error("CANCELLED");
        }
        lastErr = err;
        if (attempt < retries) {
          await new Promise(function (resolve) { setTimeout(resolve, 1000 * (attempt + 1)); });
        }
      }
    }
    throw lastErr || new Error("Network error");
  }

  function closeModal() {
    if (isUploading || isScanning) {
      return;
    }
    if (modalBackdrop && modalBackdrop.parentNode) {
      modalBackdrop.parentNode.removeChild(modalBackdrop);
    }
    modalBackdrop = null;
    parsedSelection = null;
    onSelectionCallback = null;
  }

  function ensureHiddenDirInput() {
    if (hiddenDirInput && document.body && document.body.contains(hiddenDirInput)) {
      return hiddenDirInput;
    }
    hiddenDirInput = document.createElement("input");
    hiddenDirInput.type = "file";
    hiddenDirInput.id = "ls-if-global-dir-input";
    hiddenDirInput.setAttribute("webkitdirectory", "");
    hiddenDirInput.setAttribute("directory", "");
    hiddenDirInput.multiple = true;
    hiddenDirInput.style.display = "none";
    hiddenDirInput.addEventListener("change", function () {
      if (!hiddenDirInput.files || !hiddenDirInput.files.length) {
        return;
      }
      var info = analyzeSelectedFiles(hiddenDirInput.files);
      if (info && typeof onSelectionCallback === "function") {
        onSelectionCallback(info);
      }
    });
    document.body.appendChild(hiddenDirInput);
    return hiddenDirInput;
  }

  function triggerFolderPicker() {
    if (isUploading || isScanning) {
      return;
    }
    var inp = ensureHiddenDirInput();
    inp.value = "";
    inp.click();
  }

  function openModal(projectId) {
    ensureStyles();
    if (modalBackdrop) {
      closeModal();
    }

    var folderSvgSmall =
      "<svg width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='2' stroke-linecap='round' stroke-linejoin='round'>" +
      "<path d='M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z'/>" +
      "<path d='M12 11v6'/><path d='m9 14 3-3 3 3'/>" +
      "</svg>";

    var folderSvgLarge =
      "<svg class='ls-if-dz-icon' width='64' height='64' viewBox='0 0 24 24' fill='none' stroke='currentColor' stroke-width='1.7' stroke-linecap='round' stroke-linejoin='round'>" +
      "<path d='M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z'/>" +
      "<path d='M12 11v6'/><path d='m9 14 3-3 3 3'/>" +
      "</svg>";

    modalBackdrop = document.createElement("div");
    modalBackdrop.id = "ls-if-backdrop";
    modalBackdrop.innerHTML = [
      "<div id='ls-if-modal' role='dialog' aria-modal='true' aria-labelledby='ls-if-modal-title'>",
      "  <div class='ls-if-header'>",
      "    <span class='ls-if-title' id='ls-if-modal-title'>Import Folder</span>",
      "    <div class='ls-if-header-actions'>",
      "      <button type='button' class='ls-if-btn ls-if-btn-cancel' id='ls-if-cancel-btn'>Cancel</button>",
      "      <button type='button' class='ls-if-btn ls-if-btn-primary' id='ls-if-start-btn' disabled>Import</button>",
      "      <button type='button' class='ls-if-btn ls-if-btn-primary' id='ls-if-reload-btn' style='display:none;'>View Tasks</button>",
      "    </div>",
      "  </div>",
      "  <div class='ls-if-toolbar'>",
      "    <div class='ls-if-toolbar-left'>",
      "      <button type='button' class='ls-if-btn ls-if-btn-accent ls-if-btn-select' id='ls-if-pick-btn'>" + folderSvgSmall + "<span>Select Folder</span></button>",
      "      <span class='ls-if-toolbar-hint' id='ls-if-toolbar-hint'>Drag &amp; drop a folder below to skip browser upload prompt</span>",
      "    </div>",
      "    <button type='button' class='ls-if-btn ls-if-btn-ghost' id='ls-if-clear-btn' style='display:none;'>Clear</button>",
      "  </div>",
      "  <div class='ls-if-body'>",
      "    <div class='ls-if-dropzone' id='ls-if-dropzone' tabindex='0'>",
      "      " + folderSvgLarge,
      "      <div class='ls-if-dz-heading'>Drag &amp; drop a folder here<br/>or click to browse</div>",
      "      <div class='ls-if-dz-table'>",
      "        <span class='ls-if-dz-key'>Images</span>",
      "        <span class='ls-if-dz-val'>jpg, jpeg, png, webp, bmp, tif, tiff</span>",
      "        <span class='ls-if-dz-key'>YOLO Dataset</span>",
      "        <span class='ls-if-dz-val'>images/ + labels/*.txt + classes.txt (auto-detected)</span>",
      "        <span class='ls-if-dz-key'>Large Folders</span>",
      "        <span class='ls-if-dz-val'>2,000–3,000+ images (batched upload with safe resume)</span>",
      "        <span class='ls-if-dz-key'>Tip</span>",
      "        <span class='ls-if-dz-val'>Drag &amp; drop avoids the browser confirmation dialog</span>",
      "      </div>",
      "    </div>",
      "    <div class='ls-if-workspace' id='ls-if-workspace' style='display:none;'>",
      "      <div>",
      "        <div class='ls-if-stats'>",
      "          <div class='ls-if-stat'>",
      "            <div class='ls-if-stat-label'>Valid Images</div>",
      "            <div class='ls-if-stat-value is-accent' id='ls-if-stat-images'>0</div>",
      "          </div>",
      "          <div class='ls-if-stat'>",
      "            <div class='ls-if-stat-label'>YOLO Labels</div>",
      "            <div class='ls-if-stat-value' id='ls-if-stat-labels'>0</div>",
      "          </div>",
      "          <div class='ls-if-stat'>",
      "            <div class='ls-if-stat-label'>Skipped</div>",
      "            <div class='ls-if-stat-value' id='ls-if-stat-skipped'>0</div>",
      "          </div>",
      "        </div>",
      "        <div class='ls-if-table-wrap'>",
      "          <div class='ls-if-table-head'><span>File</span><span>Size</span><span>Type</span></div>",
      "          <div class='ls-if-table-body' id='ls-if-file-list'></div>",
      "        </div>",
      "      </div>",
      "      <div class='ls-if-settings'>",
      "        <div class='ls-if-field'>",
      "          <label class='ls-if-label' for='ls-if-folder-name'>Server Folder Name</label>",
      "          <input type='text' id='ls-if-folder-name' class='ls-if-input' />",
      "          <div class='ls-if-muted' id='ls-if-path-hint'></div>",
      "        </div>",
      "        <div class='ls-if-field' id='ls-if-yolo-wrap' style='display:none;'>",
      "          <label class='ls-if-label' for='ls-if-yolo-mode'>YOLO Labels Mode</label>",
      "          <select id='ls-if-yolo-mode' class='ls-if-select'>",
      "            <option value='annotations' selected>Create Annotations (ready for SAM)</option>",
      "            <option value='predictions'>Create Predictions</option>",
      "            <option value='none'>Images only (ignore labels)</option>",
      "          </select>",
      "          <div class='ls-if-muted' id='ls-if-classes-hint'></div>",
      "        </div>",
      "        <label class='ls-if-check'>",
      "          <input type='checkbox' id='ls-if-skip-existing' checked />",
      "          <span>Skip images already uploaded to this folder (safe resume)</span>",
      "        </label>",
      "      </div>",
      "    </div>",
      "    <div class='ls-if-progress-dock' id='ls-if-progress-section' style='display:none;'>",
      "      <div class='ls-if-progress-top'>",
      "        <span id='ls-if-progress-label'>Preparing upload...</span>",
      "        <span id='ls-if-progress-pct'>0%</span>",
      "      </div>",
      "      <div class='ls-if-progress-wrap'><div class='ls-if-progress-bar' id='ls-if-progress-bar'></div></div>",
      "      <div class='ls-if-status' id='ls-if-status'></div>",
      "    </div>",
      "  </div>",
      "</div>"
    ].join("");

    document.body.appendChild(modalBackdrop);

    var cancelBtn = modalBackdrop.querySelector("#ls-if-cancel-btn");
    var startBtn = modalBackdrop.querySelector("#ls-if-start-btn");
    var reloadBtn = modalBackdrop.querySelector("#ls-if-reload-btn");
    var pickBtn = modalBackdrop.querySelector("#ls-if-pick-btn");
    var clearBtn = modalBackdrop.querySelector("#ls-if-clear-btn");
    var toolbarHint = modalBackdrop.querySelector("#ls-if-toolbar-hint");
    var dropzoneEl = modalBackdrop.querySelector("#ls-if-dropzone");
    var workspaceEl = modalBackdrop.querySelector("#ls-if-workspace");
    var statImagesEl = modalBackdrop.querySelector("#ls-if-stat-images");
    var statLabelsEl = modalBackdrop.querySelector("#ls-if-stat-labels");
    var statSkippedEl = modalBackdrop.querySelector("#ls-if-stat-skipped");
    var fileListEl = modalBackdrop.querySelector("#ls-if-file-list");
    var folderNameInput = modalBackdrop.querySelector("#ls-if-folder-name");
    var pathHintEl = modalBackdrop.querySelector("#ls-if-path-hint");
    var skipExistingCb = modalBackdrop.querySelector("#ls-if-skip-existing");
    var yoloWrap = modalBackdrop.querySelector("#ls-if-yolo-wrap");
    var yoloModeSelect = modalBackdrop.querySelector("#ls-if-yolo-mode");
    var classesHintEl = modalBackdrop.querySelector("#ls-if-classes-hint");
    var progressSection = modalBackdrop.querySelector("#ls-if-progress-section");
    var progressLabelEl = modalBackdrop.querySelector("#ls-if-progress-label");
    var progressPctEl = modalBackdrop.querySelector("#ls-if-progress-pct");
    var progressBar = modalBackdrop.querySelector("#ls-if-progress-bar");
    var statusEl = modalBackdrop.querySelector("#ls-if-status");

    function updatePathHint() {
      var safeName = sanitizeFolderName(folderNameInput.value);
      pathHintEl.textContent = "upload/" + projectId + "/folders/" + safeName + "/";
    }

    function setProgress(pct, label) {
      var clamped = Math.max(0, Math.min(100, Math.round(pct)));
      progressBar.style.width = clamped + "%";
      progressPctEl.textContent = clamped + "%";
      if (label) {
        progressLabelEl.textContent = label;
      }
    }

    function renderFilePreview(info) {
      fileListEl.innerHTML = "";
      var total = info.validImages.length;
      if (!total) {
        var emptyRow = document.createElement("div");
        emptyRow.className = "ls-if-table-more";
        emptyRow.textContent = "No valid image files found in this folder.";
        fileListEl.appendChild(emptyRow);
        return;
      }

      var limit = Math.min(total, MAX_PREVIEW_ROWS);
      for (var i = 0; i < limit; i++) {
        var f = info.validImages[i];
        var row = document.createElement("div");
        row.className = "ls-if-table-row";

        var displayName = String(f.__uploadName || f.name || "");
        var nameSpan = document.createElement("span");
        nameSpan.className = "ls-if-file-name";
        nameSpan.textContent = displayName;
        nameSpan.title = getRelativePath(f);

        var sizeSpan = document.createElement("span");
        sizeSpan.className = "ls-if-file-size";
        sizeSpan.textContent = formatBytes(f.size);

        var hasLabel = Boolean(info.labelStemSet && info.labelStemSet[getStem(String(f.name || "").toLowerCase())]);
        var tagSpan = document.createElement("span");
        tagSpan.className = "ls-if-tag" + (hasLabel ? " is-yolo" : "");
        tagSpan.textContent = hasLabel ? "YOLO" : "Image";

        row.appendChild(nameSpan);
        row.appendChild(sizeSpan);
        row.appendChild(tagSpan);
        fileListEl.appendChild(row);
      }

      if (total > limit) {
        var moreRow = document.createElement("div");
        moreRow.className = "ls-if-table-more";
        moreRow.textContent = "... and " + formatNumber(total - limit) + " more images";
        fileListEl.appendChild(moreRow);
      }
    }

    function applySelectionInfo(info) {
      parsedSelection = info;
      reloadBtn.style.display = "none";
      startBtn.style.display = "inline-flex";
      progressSection.style.display = "none";
      setProgress(0, "Preparing upload...");
      statusEl.textContent = "";

      if (!info) {
        dropzoneEl.style.display = "flex";
        workspaceEl.style.display = "none";
        clearBtn.style.display = "none";
        toolbarHint.textContent = "Drag & drop a folder below to skip browser upload prompt";
        startBtn.disabled = true;
        return;
      }

      dropzoneEl.style.display = "none";
      workspaceEl.style.display = "grid";
      clearBtn.style.display = "inline-flex";

      var plannedBatches = buildImageBatches(info.validImages, BATCH_SIZE_IMAGES, BATCH_MAX_BYTES);
      var totalBatches = plannedBatches.length;
      toolbarHint.textContent =
        "Folder: " + info.rootFolder + " · " + formatNumber(info.validImages.length) +
        " images (" + formatNumber(totalBatches) + " batch" + (totalBatches === 1 ? "" : "es") + ")";

      statImagesEl.textContent = formatNumber(info.validImages.length);
      statLabelsEl.textContent = formatNumber(info.labelFiles.length);
      statSkippedEl.textContent = formatNumber(info.zeroByteImages.length + info.nonImageSkipped);

      renderFilePreview(info);

      folderNameInput.value = info.rootFolder;
      updatePathHint();

      yoloWrap.style.display = info.isYoloFolder ? "flex" : "none";
      if (info.isYoloFolder) {
        classesHintEl.textContent = info.classesFile
          ? "Detected " + info.classesFile.name + " and " + formatNumber(info.labelFiles.length) + " label files."
          : "No classes.txt found — will map class IDs using existing project labels.";
      }

      startBtn.disabled = info.validImages.length === 0;
    }

    onSelectionCallback = applySelectionInfo;

    // Drag & drop folder support (no browser confirmation dialog!)
    var dragDepth = 0;
    function preventDrag(ev) {
      ev.preventDefault();
      ev.stopPropagation();
    }
    dropzoneEl.addEventListener("dragenter", function (ev) {
      preventDrag(ev);
      if (isUploading || isScanning) return;
      dragDepth += 1;
      dropzoneEl.classList.add("is-dragover");
    });
    dropzoneEl.addEventListener("dragover", function (ev) {
      preventDrag(ev);
      if (ev.dataTransfer) {
        ev.dataTransfer.dropEffect = "copy";
      }
    });
    dropzoneEl.addEventListener("dragleave", function (ev) {
      preventDrag(ev);
      dragDepth = Math.max(0, dragDepth - 1);
      if (dragDepth === 0) {
        dropzoneEl.classList.remove("is-dragover");
      }
    });
    dropzoneEl.addEventListener("drop", async function (ev) {
      preventDrag(ev);
      dragDepth = 0;
      dropzoneEl.classList.remove("is-dragover");
      if (isUploading || isScanning) return;

      isScanning = true;
      pickBtn.disabled = true;
      progressSection.style.display = "flex";
      setProgress(15, "Scanning dropped folder...");
      statusEl.textContent = "Reading files from folder...";

      try {
        var droppedFiles = await collectDroppedFiles(ev.dataTransfer, function (count) {
          statusEl.textContent = "Scanning folder... (" + formatNumber(count) + " files found)";
        });
        var info = analyzeSelectedFiles(droppedFiles);
        isScanning = false;
        pickBtn.disabled = false;
        applySelectionInfo(info);
      } catch (err) {
        isScanning = false;
        pickBtn.disabled = false;
        progressLabelEl.textContent = "Scan failed";
        statusEl.textContent = "Could not read dropped folder: " + (err && err.message ? err.message : String(err));
      }
    });

    dropzoneEl.addEventListener("click", triggerFolderPicker);
    dropzoneEl.addEventListener("keydown", function (ev) {
      if (ev.key === "Enter" || ev.key === " ") {
        ev.preventDefault();
        triggerFolderPicker();
      }
    });

    function onEscKey(ev) {
      if (ev.key === "Escape" && modalBackdrop) {
        cleanupAndClose();
      }
    }
    document.addEventListener("keydown", onEscKey);

    function cleanupAndClose() {
      if (isUploading || isScanning) {
        return;
      }
      document.removeEventListener("keydown", onEscKey);
      closeModal();
    }

    modalBackdrop.addEventListener("click", function (ev) {
      if (ev.target === modalBackdrop) {
        cleanupAndClose();
      }
    });

    cancelBtn.addEventListener("click", function () {
      if (isUploading) {
        cancelRequested = true;
        if (activeController) {
          try { activeController.abort(); } catch (e) {}
        }
        cancelBtn.disabled = true;
        cancelBtn.textContent = "Stopping...";
      } else {
        cleanupAndClose();
      }
    });
    clearBtn.addEventListener("click", function () {
      if (!isUploading && !isScanning) {
        applySelectionInfo(null);
      }
    });
    reloadBtn.addEventListener("click", function () {
      window.location.reload();
    });
    folderNameInput.addEventListener("input", updatePathHint);
    pickBtn.addEventListener("click", triggerFolderPicker);

    startBtn.addEventListener("click", async function () {
      if (!parsedSelection || !parsedSelection.validImages.length || isUploading || isScanning) {
        return;
      }
      isUploading = true;
      cancelRequested = false;
      pickBtn.disabled = true;
      clearBtn.disabled = true;
      folderNameInput.disabled = true;
      skipExistingCb.disabled = true;
      yoloModeSelect.disabled = true;
      startBtn.disabled = true;
      cancelBtn.textContent = "Stop";
      progressSection.style.display = "flex";
      setProgress(0, "Starting import...");
      statusEl.textContent = "";

      var folderName = sanitizeFolderName(folderNameInput.value);
      var skipExisting = skipExistingCb.checked;
      var yoloMode = parsedSelection.isYoloFolder ? yoloModeSelect.value : "none";
      var routeInfo = getProjectRouteInfo();
      var basePrefix = routeInfo ? routeInfo.basePrefix : "";
      var folderEndpoint = basePrefix + "/api/projects/" + projectId + "/import-folder";
      var fallbackEndpoint = basePrefix + "/api/projects/" + projectId + "/import";
      var useFallback = false;

      var totalImages = parsedSelection.validImages.length;
      var totalCreated = 0;
      var totalAnnotations = 0;
      var totalPredictions = 0;
      var totalSkippedExisting = 0;
      var totalSkippedInvalid = parsedSelection.zeroByteImages.length;
      var failedBatches = 0;

      try {
        // Step 1: Upload YOLO metadata (classes.txt + labels/*.txt) first if present
        if (parsedSelection.isYoloFolder && yoloMode !== "none") {
          var labelList = parsedSelection.labelFiles;
          var metaBatches = Math.max(1, Math.ceil(labelList.length / BATCH_SIZE_LABELS));
          for (var mb = 0; mb < metaBatches; mb++) {
            setProgress(0, "Uploading YOLO labels (" + (mb + 1) + "/" + metaBatches + ")...");
            var metaFd = new FormData();
            metaFd.append("mode", "yolo_meta");
            metaFd.append("folder_name", folderName);
            if (mb === 0 && parsedSelection.classesFile) {
              metaFd.append("classes", parsedSelection.classesFile, parsedSelection.classesFile.name);
            }
            var lSlice = labelList.slice(mb * BATCH_SIZE_LABELS, (mb + 1) * BATCH_SIZE_LABELS);
            for (var li = 0; li < lSlice.length; li++) {
              metaFd.append("labels", lSlice[li], lSlice[li].name);
            }
            var metaResp = await postWithRetry(folderEndpoint, metaFd, MAX_RETRIES);
            if (metaResp.status === 404) {
              useFallback = true;
              break;
            }
            if (!metaResp.ok) {
              var errTxt = await metaResp.text();
              throw new Error("YOLO metadata upload failed (HTTP " + metaResp.status + "): " + errTxt.slice(0, 200));
            }
          }
        }

        // Step 2: Upload images in sequential count- and byte-capped batches
        var imageBatches = buildImageBatches(parsedSelection.validImages, BATCH_SIZE_IMAGES, BATCH_MAX_BYTES);
        var totalBatches = imageBatches.length;
        var processedCount = 0;
        for (var b = 0; b < totalBatches; b++) {
          if (cancelRequested) {
            throw new Error("CANCELLED");
          }
          var batch = imageBatches[b];
          setProgress(
            (processedCount / totalImages) * 100,
            "Uploading batch " + (b + 1) + " / " + totalBatches +
              " (" + formatNumber(Math.min(processedCount + batch.length, totalImages)) +
              " / " + formatNumber(totalImages) + " images)"
          );

          var fd = new FormData();
          var targetUrl = useFallback ? fallbackEndpoint : folderEndpoint;
          if (!useFallback) {
            fd.append("mode", "images");
            fd.append("folder_name", folderName);
            fd.append("skip_existing", skipExisting ? "true" : "false");
            fd.append("yolo_mode", yoloMode);
            for (var bi = 0; bi < batch.length; bi++) {
              fd.append("files", batch[bi], String(batch[bi].__uploadName || batch[bi].name));
            }
          } else {
            for (var fi = 0; fi < batch.length; fi++) {
              fd.append("file_" + fi, batch[fi], String(batch[fi].__uploadName || batch[fi].name));
            }
          }

          var resp = await postWithRetry(targetUrl, fd, MAX_RETRIES);
          if (!useFallback && resp.status === 404) {
            useFallback = true;
            b -= 1;
            continue;
          }

          if (resp.ok) {
            var data = await resp.json();
            totalCreated += Number(data.task_count || 0);
            totalAnnotations += Number(data.annotations_created || 0);
            totalPredictions += Number(data.predictions_created || 0);
            totalSkippedExisting += Number(data.skipped_existing || 0);
            if (Array.isArray(data.skipped_invalid)) {
              totalSkippedInvalid += data.skipped_invalid.length;
            }
          } else if (resp.status === 401 || resp.status === 403) {
            throw new Error("Permission denied for importing into this project (HTTP " + resp.status + ").");
          } else {
            failedBatches += 1;
          }

          processedCount += batch.length;
          setProgress((processedCount / totalImages) * 100);
        }

        setProgress(100, "Import completed");
        var summaryLines = [
          "Created: " + formatNumber(totalCreated) + " tasks" +
            "  ·  Skipped existing: " + formatNumber(totalSkippedExisting) +
            "  ·  Invalid/0-byte: " + formatNumber(totalSkippedInvalid)
        ];
        if (totalAnnotations > 0) {
          summaryLines.push("YOLO Annotations attached: " + formatNumber(totalAnnotations) + " images");
        }
        if (totalPredictions > 0) {
          summaryLines.push("YOLO Predictions attached: " + formatNumber(totalPredictions) + " images");
        }
        if (failedBatches > 0) {
          summaryLines.push("Warning: " + failedBatches + " batch(es) failed — re-select this folder with 'Skip existing' enabled to resume.");
        }
        if (!useFallback) {
          summaryLines.push("Server directory: upload/" + projectId + "/folders/" + folderName + "/");
        }
        statusEl.textContent = summaryLines.join("\n");
        startBtn.style.display = "none";
        reloadBtn.style.display = "inline-flex";
      } catch (err) {
        if (err && err.message === "CANCELLED") {
          progressLabelEl.textContent = "Import stopped";
          statusEl.textContent =
            "Stopped early. Created " + formatNumber(totalCreated) +
            " tasks. Re-import the same folder with 'Skip existing' checked to resume.";
          if (totalCreated > 0) {
            startBtn.style.display = "none";
            reloadBtn.style.display = "inline-flex";
          }
        } else {
          progressLabelEl.textContent = "Import failed";
          statusEl.textContent = err && err.message ? err.message : String(err);
        }
      } finally {
        isUploading = false;
        activeController = null;
        pickBtn.disabled = false;
        clearBtn.disabled = false;
        folderNameInput.disabled = false;
        skipExistingCb.disabled = false;
        yoloModeSelect.disabled = false;
        cancelBtn.disabled = false;
        cancelBtn.textContent = "Close";
        startBtn.disabled = !parsedSelection || !parsedSelection.validImages.length;
      }
    });
  }

  function findToolbarAnchor() {
    var refreshBtn = document.querySelector('button[aria-label="Refresh data"]');
    if (refreshBtn && refreshBtn.parentElement) {
      var group = refreshBtn.parentElement;
      var styleSource =
        group.querySelector('button[data-testid="dm-import-button"]') ||
        group.querySelector('button[data-testid="dm-export-button"]') ||
        refreshBtn;
      return {
        container: group,
        anchor: refreshBtn,
        insertPosition: "afterend",
        styleSource: styleSource
      };
    }

    var exportBtn = document.querySelector('button[data-testid="dm-export-button"]');
    if (exportBtn && exportBtn.parentElement) {
      var importBtn = exportBtn.parentElement.querySelector('button[data-testid="dm-import-button"]');
      return {
        container: exportBtn.parentElement,
        anchor: importBtn || exportBtn,
        insertPosition: "beforebegin",
        styleSource: importBtn || exportBtn
      };
    }
    return null;
  }

  function applyNativeButtonStyle(btn, styleSource) {
    if (styleSource && styleSource.className) {
      btn.className = styleSource.className;
      btn.style.minWidth = "auto";
      btn.style.width = "auto";
    } else {
      btn.className = "ls-if-toolbar-fallback";
    }
  }

  function syncButtonVisibility() {
    var projectId = getProjectIdFromUrl();
    if (!projectId || isEditorActive()) {
      if (triggerBtn && triggerBtn.parentNode) {
        triggerBtn.style.display = "none";
      }
      return;
    }

    var slot = findToolbarAnchor();
    if (!slot || !slot.container || !slot.anchor) {
      if (triggerBtn && triggerBtn.parentNode) {
        triggerBtn.style.display = "none";
      }
      return;
    }

    ensureStyles();
    if (!triggerBtn) {
      triggerBtn = document.createElement("button");
      triggerBtn.type = "button";
      triggerBtn.id = "ls-import-folder-btn";
      triggerBtn.setAttribute("data-testid", "dm-import-folder-button");
      triggerBtn.setAttribute("aria-label", "Import folder button");
      triggerBtn.textContent = "Import Folder";
      triggerBtn.title = "Import an entire image or YOLO folder (2,000–3,000+ images)";
      triggerBtn.addEventListener("click", function () {
        var pid = getProjectIdFromUrl();
        if (pid) {
          openModal(pid);
        }
      });
    }

    applyNativeButtonStyle(triggerBtn, slot.styleSource);
    if (triggerBtn.parentNode !== slot.container) {
      if (slot.insertPosition === "afterend") {
        slot.container.insertBefore(triggerBtn, slot.anchor.nextSibling);
      } else {
        slot.container.insertBefore(triggerBtn, slot.anchor);
      }
    }
    triggerBtn.style.display = "";
  }

  setInterval(syncButtonVisibility, 400);
})();
