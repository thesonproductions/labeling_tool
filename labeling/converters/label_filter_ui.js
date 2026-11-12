(function () {
  "use strict";

  if (window.__lsLabelFilterMounted) {
    return;
  }
  window.__lsLabelFilterMounted = true;

  var SENTINEL_LABELS = {
    "__dynamic__": true,
    "+ Nhập nhãn mới...": true,
    "_poly": true
  };

  var STORAGE_POS_KEY = "ls_label_filter_pos_v1";
  var STORAGE_COLLAPSED_KEY = "ls_label_filter_collapsed_v1";

  var state = {
    mode: "all", // "all" | "custom"
    selected: new Set(),
    collapsed: localStorage.getItem(STORAGE_COLLAPSED_KEY) === "1",
    lastSignature: "",
    lastTaskKey: "",
    knownLabels: [], // [{ name, color, count }]
    totalCount: 0,
    dragging: false,
    dragOffsetX: 0,
    dragOffsetY: 0,
    positionedOnce: false
  };

  var rootEl = null;
  var styleEl = null;

  function ensureStyles() {
    if (styleEl && document.head && document.head.contains(styleEl)) {
      return;
    }
    styleEl = document.createElement("style");
    styleEl.id = "ls-label-filter-styles";
    styleEl.textContent = [
      "#ls-label-filter-panel {",
      "  position: fixed;",
      "  z-index: 9998;",
      "  width: 268px;",
      "  border-radius: 10px;",
      "  font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Arial, sans-serif;",
      "  font-size: 13px;",
      "  box-shadow: 0 8px 24px rgba(0, 0, 0, 0.22);",
      "  border: 1px solid rgba(0, 0, 0, 0.14);",
      "  background: #ffffff;",
      "  color: #1f2937;",
      "  user-select: none;",
      "  display: none;",
      "  overflow: hidden;",
      "}",
      "[data-color-scheme='dark'] #ls-label-filter-panel {",
      "  background: #1e2028;",
      "  color: #e5e7eb;",
      "  border-color: rgba(255, 255, 255, 0.14);",
      "  box-shadow: 0 8px 28px rgba(0, 0, 0, 0.55);",
      "}",
      "#ls-label-filter-panel.is-collapsed {",
      "  width: auto;",
      "  min-width: 170px;",
      "}",
      "#ls-label-filter-header {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  gap: 8px;",
      "  padding: 8px 10px;",
      "  cursor: move;",
      "  background: #f3f4f6;",
      "  border-bottom: 1px solid rgba(0, 0, 0, 0.08);",
      "  font-weight: 600;",
      "}",
      "[data-color-scheme='dark'] #ls-label-filter-header {",
      "  background: #272a34;",
      "  border-bottom-color: rgba(255, 255, 255, 0.08);",
      "}",
      "#ls-label-filter-panel.is-collapsed #ls-label-filter-header {",
      "  border-bottom: none;",
      "}",
      ".lsf-flt-title {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 6px;",
      "  font-size: 12.5px;",
      "  white-space: nowrap;",
      "}",
      ".lsf-flt-badge {",
      "  font-size: 11px;",
      "  font-weight: 600;",
      "  padding: 2px 7px;",
      "  border-radius: 999px;",
      "  background: #e5e7eb;",
      "  color: #374151;",
      "}",
      "[data-color-scheme='dark'] .lsf-flt-badge {",
      "  background: #374151;",
      "  color: #e5e7eb;",
      "}",
      ".lsf-flt-badge.is-filtered {",
      "  background: #16a34a;",
      "  color: #ffffff;",
      "}",
      ".lsf-flt-toggle-btn {",
      "  border: none;",
      "  background: transparent;",
      "  color: inherit;",
      "  cursor: pointer;",
      "  font-size: 14px;",
      "  line-height: 1;",
      "  padding: 2px 6px;",
      "  border-radius: 4px;",
      "  opacity: 0.75;",
      "}",
      ".lsf-flt-toggle-btn:hover {",
      "  opacity: 1;",
      "  background: rgba(128, 128, 128, 0.18);",
      "}",
      "#ls-label-filter-body {",
      "  padding: 8px 10px;",
      "}",
      "#ls-label-filter-panel.is-collapsed #ls-label-filter-body {",
      "  display: none;",
      "}",
      ".lsf-flt-all-row {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  padding: 5px 6px;",
      "  border-radius: 6px;",
      "  font-weight: 600;",
      "  cursor: pointer;",
      "  background: rgba(39, 174, 96, 0.08);",
      "}",
      ".lsf-flt-all-row:hover {",
      "  background: rgba(39, 174, 96, 0.16);",
      "}",
      ".lsf-flt-divider {",
      "  height: 1px;",
      "  margin: 7px 0;",
      "  background: rgba(128, 128, 128, 0.22);",
      "}",
      ".lsf-flt-list {",
      "  max-height: 250px;",
      "  overflow-y: auto;",
      "  display: flex;",
      "  flex-direction: column;",
      "  gap: 3px;",
      "}",
      ".lsf-flt-row {",
      "  display: flex;",
      "  align-items: center;",
      "  justify-content: space-between;",
      "  gap: 6px;",
      "  padding: 4px 6px;",
      "  border-radius: 6px;",
      "  cursor: pointer;",
      "}",
      ".lsf-flt-row:hover {",
      "  background: rgba(128, 128, 128, 0.13);",
      "}",
      ".lsf-flt-row.is-zero {",
      "  opacity: 0.52;",
      "}",
      ".lsf-flt-left {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 7px;",
      "  min-width: 0;",
      "  flex: 1;",
      "}",
      ".lsf-flt-left input[type='checkbox'] {",
      "  width: 14px;",
      "  height: 14px;",
      "  margin: 0;",
      "  cursor: pointer;",
      "  accent-color: #27ae60;",
      "  flex-shrink: 0;",
      "}",
      ".lsf-flt-dot {",
      "  width: 9px;",
      "  height: 9px;",
      "  border-radius: 50%;",
      "  flex-shrink: 0;",
      "  border: 1px solid rgba(0,0,0,0.2);",
      "}",
      ".lsf-flt-name {",
      "  overflow: hidden;",
      "  text-overflow: ellipsis;",
      "  white-space: nowrap;",
      "  font-size: 12.5px;",
      "}",
      ".lsf-flt-right {",
      "  display: flex;",
      "  align-items: center;",
      "  gap: 5px;",
      "  flex-shrink: 0;",
      "}",
      ".lsf-flt-count {",
      "  font-size: 11px;",
      "  opacity: 0.72;",
      "  font-variant-numeric: tabular-nums;",
      "}",
      ".lsf-flt-solo-btn {",
      "  border: 1px solid rgba(39, 174, 96, 0.45);",
      "  background: transparent;",
      "  color: #27ae60;",
      "  font-size: 10.5px;",
      "  font-weight: 600;",
      "  padding: 1px 5px;",
      "  border-radius: 4px;",
      "  cursor: pointer;",
      "  opacity: 0;",
      "  transition: opacity 0.12s ease, background 0.12s ease;",
      "}",
      ".lsf-flt-row:hover .lsf-flt-solo-btn,",
      ".lsf-flt-solo-btn:focus {",
      "  opacity: 1;",
      "}",
      ".lsf-flt-solo-btn:hover {",
      "  background: #27ae60;",
      "  color: #ffffff;",
      "}",
      ".lsf-flt-empty {",
      "  font-size: 12px;",
      "  opacity: 0.68;",
      "  padding: 6px 4px;",
      "  text-align: center;",
      "}"
    ].join("\n");
    document.head.appendChild(styleEl);
  }

  function clampPosition(left, top) {
    var w = (rootEl && rootEl.offsetWidth) || 268;
    var h = (rootEl && rootEl.offsetHeight) || 120;
    var maxLeft = Math.max(8, window.innerWidth - w - 8);
    var maxTop = Math.max(8, window.innerHeight - Math.min(h, 80) - 8);
    return {
      left: Math.min(Math.max(8, left), maxLeft),
      top: Math.min(Math.max(8, top), maxTop)
    };
  }

  function applyInitialPosition() {
    if (!rootEl || state.positionedOnce) {
      return;
    }
    state.positionedOnce = true;
    try {
      var saved = JSON.parse(localStorage.getItem(STORAGE_POS_KEY) || "null");
      if (saved && typeof saved.left === "number" && typeof saved.top === "number") {
        var clamped = clampPosition(saved.left, saved.top);
        rootEl.style.left = clamped.left + "px";
        rootEl.style.top = clamped.top + "px";
        return;
      }
    } catch (e) {}

    // Đặt gọn ở góc phải vùng ảnh, né cột Outliner bên phải
    var mainView = document.querySelector(".lsf-main-view__annotation, [class*='main-view__annotation']");
    var defaultLeft = window.innerWidth - 580;
    var defaultTop = 108;
    if (mainView) {
      var rect = mainView.getBoundingClientRect();
      if (rect && rect.width > 300) {
        defaultLeft = rect.right - 284;
        defaultTop = Math.max(68, rect.top + 12);
      }
    }
    var pos = clampPosition(defaultLeft, defaultTop);
    rootEl.style.left = pos.left + "px";
    rootEl.style.top = pos.top + "px";
  }

  function ensurePanelDom() {
    ensureStyles();
    if (rootEl && document.body && document.body.contains(rootEl)) {
      return;
    }
    rootEl = document.createElement("div");
    rootEl.id = "ls-label-filter-panel";
    if (state.collapsed) {
      rootEl.classList.add("is-collapsed");
    }

    rootEl.innerHTML = [
      "<div id='ls-label-filter-header' title='Kéo để di chuyển vị trí bảng lọc'>",
      "  <div class='lsf-flt-title'>",
      "    <span>Lọc hiển thị</span>",
      "    <span class='lsf-flt-badge' id='ls-flt-badge'>All</span>",
      "  </div>",
      "  <button type='button' class='lsf-flt-toggle-btn' id='ls-flt-collapse-btn' title='Thu gọn / Mở rộng'></button>",
      "</div>",
      "<div id='ls-label-filter-body'>",
      "  <label class='lsf-flt-all-row'>",
      "    <span class='lsf-flt-left'>",
      "      <input type='checkbox' id='ls-flt-all-cb' checked />",
      "      <span>All (Tất cả loài)</span>",
      "    </span>",
      "    <span class='lsf-flt-count' id='ls-flt-total-count'>0</span>",
      "  </label>",
      "  <div class='lsf-flt-divider'></div>",
      "  <div class='lsf-flt-list' id='ls-flt-list'></div>",
      "</div>"
    ].join("");

    document.body.appendChild(rootEl);

    var headerEl = rootEl.querySelector("#ls-label-filter-header");
    var collapseBtn = rootEl.querySelector("#ls-flt-collapse-btn");
    var allCb = rootEl.querySelector("#ls-flt-all-cb");

    function updateCollapseBtnText() {
      collapseBtn.textContent = state.collapsed ? "+" : "−";
    }
    updateCollapseBtnText();

    collapseBtn.addEventListener("click", function (ev) {
      ev.stopPropagation();
      state.collapsed = !state.collapsed;
      rootEl.classList.toggle("is-collapsed", state.collapsed);
      updateCollapseBtnText();
      try {
        localStorage.setItem(STORAGE_COLLAPSED_KEY, state.collapsed ? "1" : "0");
      } catch (e) {}
    });

    allCb.addEventListener("change", function () {
      if (allCb.checked) {
        state.mode = "all";
        state.selected = new Set(
          state.knownLabels.map(function (item) { return item.name; })
        );
      } else {
        state.mode = "custom";
        state.selected = new Set();
      }
      renderCheckboxList();
      applyVisibilityFilter(true);
    });

    // Hỗ trợ kéo thả bảng lọc đến vị trí thuận mắt
    headerEl.addEventListener("mousedown", function (ev) {
      if (ev.target === collapseBtn || ev.button !== 0) {
        return;
      }
      state.dragging = true;
      var rect = rootEl.getBoundingClientRect();
      state.dragOffsetX = ev.clientX - rect.left;
      state.dragOffsetY = ev.clientY - rect.top;
      ev.preventDefault();
    });

    window.addEventListener("mousemove", function (ev) {
      if (!state.dragging || !rootEl) {
        return;
      }
      var pos = clampPosition(ev.clientX - state.dragOffsetX, ev.clientY - state.dragOffsetY);
      rootEl.style.left = pos.left + "px";
      rootEl.style.top = pos.top + "px";
    });

    window.addEventListener("mouseup", function () {
      if (!state.dragging || !rootEl) {
        return;
      }
      state.dragging = false;
      try {
        var rect = rootEl.getBoundingClientRect();
        localStorage.setItem(STORAGE_POS_KEY, JSON.stringify({ left: rect.left, top: rect.top }));
      } catch (e) {}
    });
  }

  function getActiveEditorContext() {
    var htx = window.Htx;
    if (!htx || !htx.annotationStore) {
      return null;
    }
    var ann = htx.annotationStore.selectedHistory || htx.annotationStore.selected;
    if (!ann || !ann.regionStore) {
      return null;
    }
    // Đảm bảo đang thực sự ở màn hình gán nhãn (không phải màn Data Manager hay Settings)
    var canvasDom = document.querySelector(".lsf-main-view, [class*='main-view']");
    if (!canvasDom || canvasDom.offsetParent === null) {
      return null;
    }
    return { htx: htx, ann: ann, regStore: ann.regionStore };
  }

  function getRegionLabels(region) {
    if (!region) {
      return [];
    }
    try {
      if (region.labels && typeof region.labels.length === "number") {
        return Array.from(region.labels);
      }
      if (region.labeling && region.labeling.mainValue) {
        return Array.from(region.labeling.mainValue);
      }
    } catch (e) {}
    return [];
  }

  function getRegionColor(region) {
    try {
      if (region.labeling && region.labeling.selectedLabels && region.labeling.selectedLabels[0]) {
        var bg = region.labeling.selectedLabels[0].background;
        if (bg) return bg;
      }
      if (typeof region.getOneColor === "function") {
        var c = region.getOneColor();
        if (c) return c;
      }
    } catch (e) {}
    return "#27ae60";
  }

  function isRectRegion(region) {
    var t = String((region && region.type) || "").toLowerCase();
    return t.indexOf("rect") !== -1;
  }

  function isPolyRegion(region) {
    var t = String((region && region.type) || "").toLowerCase();
    return t.indexOf("poly") !== -1;
  }

  function collectLabelsInfo(ctx) {
    var orderedNames = [];
    var metaMap = {}; // name -> { name, color, rects, polys, others }

    function ensureMeta(name, color) {
      if (!name || SENTINEL_LABELS[name]) {
        return null;
      }
      if (!metaMap[name]) {
        metaMap[name] = {
          name: name,
          color: color || "#27ae60",
          rects: 0,
          polys: 0,
          others: 0
        };
        orderedNames.push(name);
      } else if (color && metaMap[name].color === "#27ae60") {
        metaMap[name].color = color;
      }
      return metaMap[name];
    }

    // 1. Lấy các nhãn khai báo trong cấu hình Project (RectangleLabels / PolygonLabels)
    try {
      if (ctx.ann.names && typeof ctx.ann.names.values === "function") {
        var tags = Array.from(ctx.ann.names.values());
        for (var i = 0; i < tags.length; i++) {
          var tag = tags[i];
          var tagType = String((tag && tag.type) || "").toLowerCase();
          if (tagType.indexOf("labels") === -1) {
            continue;
          }
          var children = tag.children ? Array.from(tag.children) : [];
          for (var j = 0; j < children.length; j++) {
            var ch = children[j];
            if (ch && ch.value) {
              ensureMeta(String(ch.value), ch.background);
            }
          }
        }
      }
    } catch (e) {}

    // 2. Quét toàn bộ region hiện có trên ảnh (kể cả Box và Polygon)
    var regions = [];
    try {
      regions = Array.from(ctx.regStore.regions || []);
    } catch (e) {}

    var sigParts = [];
    for (var rIdx = 0; rIdx < regions.length; rIdx++) {
      var r = regions[rIdx];
      var lbls = getRegionLabels(r);
      var col = getRegionColor(r);
      var rIsRect = isRectRegion(r);
      var rIsPoly = isPolyRegion(r);
      sigParts.push((r.id || rIdx) + ":" + lbls.join(","));

      for (var lIdx = 0; lIdx < lbls.length; lIdx++) {
        var meta = ensureMeta(String(lbls[lIdx]), col);
        if (!meta) continue;
        if (rIsRect) meta.rects += 1;
        else if (rIsPoly) meta.polys += 1;
        else meta.others += 1;
      }
    }

    // Ẩn nhãn mặc định "Object" nếu dự án đã có nhãn loài thực tế và "Object" không có vùng nào
    var hasNonObject = orderedNames.some(function (n) { return n !== "Object"; });
    var result = [];
    var totalObjects = 0;
    for (var k = 0; k < orderedNames.length; k++) {
      var name = orderedNames[k];
      var item = metaMap[name];
      var count = (item.rects > 0 || item.polys > 0)
        ? Math.max(item.rects, item.polys)
        : item.others;
      if (name === "Object" && hasNonObject && count === 0) {
        continue;
      }
      totalObjects += count;
      result.push({
        name: name,
        color: item.color,
        count: count
      });
    }

    var taskKey = "";
    try {
      taskKey = String((ctx.htx.task && ctx.htx.task.id) || "") + "::" + String(ctx.ann.id || "");
    } catch (e) {}

    return {
      labels: result,
      totalCount: totalObjects,
      taskKey: taskKey,
      signature: taskKey + "|" + orderedNames.join(",") + "|" + regions.length + "|" + sigParts.join(";")
    };
  }

  function renderCheckboxList() {
    if (!rootEl) return;
    var listEl = rootEl.querySelector("#ls-flt-list");
    var allCb = rootEl.querySelector("#ls-flt-all-cb");
    var totalEl = rootEl.querySelector("#ls-flt-total-count");
    var badgeEl = rootEl.querySelector("#ls-flt-badge");
    if (!listEl || !allCb || !totalEl || !badgeEl) return;

    var items = state.knownLabels;
    totalEl.textContent = String(state.totalCount);

    var isAll = state.mode === "all";
    allCb.checked = isAll;

    if (isAll) {
      badgeEl.textContent = "All (" + state.totalCount + ")";
      badgeEl.classList.remove("is-filtered");
    } else {
      var activeNames = items.filter(function (it) { return state.selected.has(it.name); });
      if (activeNames.length === 1) {
        badgeEl.textContent = activeNames[0].name + " (" + activeNames[0].count + ")";
      } else {
        badgeEl.textContent = activeNames.length + "/" + items.length + " loài";
      }
      badgeEl.classList.add("is-filtered");
    }

    listEl.innerHTML = "";
    if (!items.length) {
      var empty = document.createElement("div");
      empty.className = "lsf-flt-empty";
      empty.textContent = "Chưa có nhãn loài nào trong dự án";
      listEl.appendChild(empty);
      return;
    }

    items.forEach(function (item) {
      var row = document.createElement("div");
      row.className = "lsf-flt-row" + (item.count === 0 ? " is-zero" : "");

      var left = document.createElement("label");
      left.className = "lsf-flt-left";

      var cb = document.createElement("input");
      cb.type = "checkbox";
      cb.checked = isAll || state.selected.has(item.name);

      var dot = document.createElement("span");
      dot.className = "lsf-flt-dot";
      dot.style.backgroundColor = item.color;

      var nameSpan = document.createElement("span");
      nameSpan.className = "lsf-flt-name";
      nameSpan.textContent = item.name;
      nameSpan.title = item.name;

      left.appendChild(cb);
      left.appendChild(dot);
      left.appendChild(nameSpan);

      var right = document.createElement("div");
      right.className = "lsf-flt-right";

      var soloBtn = document.createElement("button");
      soloBtn.type = "button";
      soloBtn.className = "lsf-flt-solo-btn";
      soloBtn.textContent = "Chỉ xem";
      soloBtn.title = "Chỉ hiển thị riêng " + item.name;

      var countSpan = document.createElement("span");
      countSpan.className = "lsf-flt-count";
      countSpan.textContent = "(" + item.count + ")";

      right.appendChild(soloBtn);
      right.appendChild(countSpan);

      row.appendChild(left);
      row.appendChild(right);
      listEl.appendChild(row);

      cb.addEventListener("change", function () {
        if (state.mode === "all") {
          // Chuyển từ All sang Custom với tất cả loài khác vẫn đang bật, chỉ bỏ loài vừa uncheck
          state.mode = "custom";
          state.selected = new Set(
            state.knownLabels.map(function (it) { return it.name; })
          );
        }

        if (cb.checked) {
          state.selected.add(item.name);
        } else {
          state.selected.delete(item.name);
        }

        // Nếu mọi loài đều được tích lại -> tự quay về trạng thái All
        if (
          state.knownLabels.length > 0 &&
          state.knownLabels.every(function (it) { return state.selected.has(it.name); })
        ) {
          state.mode = "all";
        }

        renderCheckboxList();
        applyVisibilityFilter(true);
      });

      soloBtn.addEventListener("click", function (ev) {
        ev.preventDefault();
        ev.stopPropagation();
        state.mode = "custom";
        state.selected = new Set([item.name]);
        renderCheckboxList();
        applyVisibilityFilter(true);
      });
    });
  }

  function applyVisibilityFilter(forceApply) {
    var ctx = getActiveEditorContext();
    if (!ctx) return;

    // Ở chế độ All mặc định: chỉ chủ động bật lại các region nếu người dùng vừa chuyển từ Custom -> All
    if (state.mode === "all" && !forceApply) {
      return;
    }

    var regions = [];
    try {
      regions = Array.from(ctx.regStore.regions || []);
    } catch (e) {
      return;
    }

    for (var i = 0; i < regions.length; i++) {
      var r = regions[i];
      if (!r || r.incomplete || typeof r.toggleHidden !== "function") {
        continue;
      }

      var shouldHide = false;
      if (state.mode !== "all") {
        var lbls = getRegionLabels(r);
        var matched = false;
        for (var j = 0; j < lbls.length; j++) {
          if (state.selected.has(String(lbls[j]))) {
            matched = true;
            break;
          }
        }
        shouldHide = !matched;
      }

      if (Boolean(r.hidden) !== shouldHide) {
        try {
          r.toggleHidden(null, true);
        } catch (e) {}
      }
    }
  }

  function tick() {
    var ctx = getActiveEditorContext();
    if (!ctx) {
      if (rootEl) {
        rootEl.style.display = "none";
      }
      return;
    }

    ensurePanelDom();
    rootEl.style.display = "block";
    applyInitialPosition();

    var info = collectLabelsInfo(ctx);
    if (info.signature !== state.lastSignature) {
      var isNewTask = info.taskKey !== state.lastTaskKey;
      state.lastSignature = info.signature;
      state.lastTaskKey = info.taskKey;
      state.knownLabels = info.labels;
      state.totalCount = info.totalCount;

      if (state.mode === "all") {
        state.selected = new Set(
          info.labels.map(function (it) { return it.name; })
        );
      }

      renderCheckboxList();
      // Khi vừa mở task mới ở chế độ lọc, hoặc có region mới xuất hiện, ép lại bộ lọc hiển thị
      if (state.mode !== "all" || isNewTask) {
        applyVisibilityFilter(state.mode !== "all");
      }
    }
  }

  window.addEventListener("resize", function () {
    if (!rootEl || rootEl.style.display === "none") return;
    var rect = rootEl.getBoundingClientRect();
    var pos = clampPosition(rect.left, rect.top);
    rootEl.style.left = pos.left + "px";
    rootEl.style.top = pos.top + "px";
  });

  setInterval(tick, 300);
})();
