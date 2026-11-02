// The Event2Reward annotation page, served by `event2reward annotate serve`.
// Successful and failed videos are annotated in separate modes. Saved
// annotations go to the project folder through the server, which checks them
// (src/event2reward/annotation/schema.py); drafts stay in the browser.
(() => {
  "use strict";

  const LANG = localStorage.getItem("e2r-annotation:language") || "en";
  const S = window.STRINGS[LANG];
  const COLORS = {
    success: "clean", progress_stall: "hesitation", regression_and_recovery: "regression",
    multiple_events: "mixed", no_further_progress: "hesitation", unrecovered_regression: "failure", uncertain: "uncertain",
  };
  const MODE_LABELS = {
    success: ["success", "progress_stall", "regression_and_recovery", "multiple_events", "uncertain"],
    failure: ["no_further_progress", "unrecovered_regression", "uncertain"],
  };

  const ids = [
    "brandSubtitle", "progressLabel", "progressNumbers", "progressFill", "languageButton", "helpButton",
    "filteredCount", "modeSwitch", "taskFilter", "statusFilter", "searchInput", "queueList",
    "taskBadge", "episodeTitle", "bucketBadge", "episodeMeta", "previousButton", "nextButton", "episodePosition",
    "viewSwitcher", "viewButtons", "video", "videoEmptyState", "frameReadout", "frameTotal",
    "timeline", "timelineProgress", "timelineRegions", "timelineMarkers", "timelinePlayhead",
    "timelineMidpoint", "timelineEndpoint", "jumpBackButton", "stepBackButton", "playButton",
    "stepForwardButton", "jumpForwardButton", "frameInput", "fpsInput", "speedSelect",
    "curveCanvas", "curveMessage", "annotationHeading", "saveState", "labelCards", "pointEditor",
    "endpointEditor", "validationMessage", "saveButton", "clearButton",
    "helpDialog", "closeHelpButton", "helpContent", "themeButton",
  ];
  const elements = Object.fromEntries(ids.map((id) => [id, document.getElementById(id)]));

  let queue = [];
  let filteredQueue = [];
  let annotations = {};
  let storageKey = "e2r-annotation:drafts";
  let preferences = loadJson("e2r-annotation:preferences", { mode: "success", task: "all", status: "all", speed: "1" });
  let currentRecord = null;
  let currentFrame = 0;
  let frameCount = 2;
  let messageTimer = null;

  // ---------- helpers ----------
  function loadJson(key, fallback) {
    try {
      const raw = localStorage.getItem(key);
      return raw ? { ...fallback, ...JSON.parse(raw) } : fallback;
    } catch {
      return fallback;
    }
  }
  function saveJson(key, value) {
    try { localStorage.setItem(key, JSON.stringify(value)); } catch { /* drafts only */ }
  }
  const clamp = (value, minimum, maximum) => Math.min(Math.max(value, minimum), maximum);
  const asInteger = (value) => (value === null || value === undefined || value === "" ? null
    : Number.isInteger(Number(value)) ? Number(value) : null);
  const labelTitle = (label) => S.labels[label]?.[0] || label;
  const lastFrame = () => Math.max(frameCount - 1, 1);
  const finalFrame = (annotation) => asInteger(annotation?.final_frame) ?? frameCount - 1;
  const escapeHtml = (text) => String(text).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

  // ---------- project and annotations ----------
  function buildQueue(project) {
    const tasks = [...new Set(project.videos.map((v) => v.task))].sort();
    return project.videos.map((v) => ({
      uid: v.uid, task: v.task, task_index: tasks.indexOf(v.task), instruction: v.instruction, kind: v.mode,
      source: v.source || v.uid.split("/").pop(), split: v.split, video: v.video, views: v.views || {},
      times: v.times, frame_count: v.times.length, duration_seconds: v.times[v.times.length - 1],
      fps: (v.times.length - 1) / v.times[v.times.length - 1],
    })).sort((a, b) => a.task_index - b.task_index || a.duration_seconds - b.duration_seconds || a.uid.localeCompare(b.uid));
  }

  // The page edits these fields; the project stores event2reward.annotation.v1.
  function fromSaved(saved, record) {
    const a = blankAnnotation(record, saved.label);
    const events = saved.events || [];
    if (saved.label === "progress_stall") [a.stall_a, a.stall_r] = [events[0].a_frame, events[0].r_frame];
    if (saved.label === "regression_and_recovery") [a.regression_a, a.regression_r] = [events[0].a_frame, events[0].r_frame];
    if (saved.label === "multiple_events") a.event_list = events.map((e) => ({ type: e.type, a: e.a_frame, r: e.r_frame }));
    if (saved.label === "no_further_progress" || saved.label === "unrecovered_regression") {
      a.failure_a = events[0].a_frame;
      a.failure_loss = events[0].loss_frame ?? null;
    }
    a.final_frame = saved.final_frame;
    a.review_status = "approved";
    a.updated_at = saved.saved_at || "";
    return a;
  }

  function toSaved(a, record) {
    const S_ = finalFrame(a);
    const events = [];
    if (a.label === "progress_stall") events.push({ type: "progress_stall", a_frame: a.stall_a, r_frame: a.stall_r });
    if (a.label === "regression_and_recovery") events.push({ type: "regression_and_recovery", a_frame: a.regression_a, r_frame: a.regression_r });
    if (a.label === "multiple_events") a.event_list.forEach((e) => events.push({ type: e.type, a_frame: e.a, r_frame: e.r }));
    if (a.label === "no_further_progress") events.push({ type: "no_further_progress", a_frame: a.failure_a });
    if (a.label === "unrecovered_regression") {
      const event = { type: "unrecovered_regression", a_frame: a.failure_a };
      if (asInteger(a.failure_loss) !== null) event.loss_frame = a.failure_loss;
      events.push(event);
    }
    const saved = { label: a.label, events, final_frame: S_ };
    return saved;
  }

  function blankAnnotation(record, label = "") {
    return {
      label, review_status: "draft",
      stall_a: null, stall_r: null,
      regression_a: null, regression_r: null,
      failure_a: null, failure_loss: null, event_list: [],
      final_frame: record.frame_count - 1, updated_at: new Date().toISOString(),
    };
  }

  // Successful videos default to clean success, but count only once saved.
  const defaultLabelForRecord = (record = currentRecord) => (record?.kind === "success" ? "success" : "");

  function annotationFor(record = currentRecord, create = false) {
    if (!record) return null;
    if (!annotations[record.uid] && create) annotations[record.uid] = blankAnnotation(record, defaultLabelForRecord(record));
    return annotations[record.uid] || null;
  }

  function persistDrafts() {
    const drafts = Object.fromEntries(Object.entries(annotations).filter(([, a]) => a.review_status === "draft"));
    saveJson(storageKey, drafts);
  }

  function updateAnnotation(mutator, { render = true } = {}) {
    const annotation = annotationFor(currentRecord, true);
    mutator(annotation);
    annotation.review_status = "draft";
    annotation.updated_at = new Date().toISOString();
    persistDrafts();
    if (render) {
      renderAnnotationPanel();
      renderTimeline();
      drawCurve();
      renderQueueList(false);
      renderProgress();
    } else {
      renderSaveState();
      renderProgress();
    }
  }

  const annotationStatus = (record) => annotations[record.uid]?.review_status || "pending";
  const isEventAnnotation = (annotation) => annotation?.label && !["success", "uncertain"].includes(annotation.label);

  // ---------- filters and queue ----------
  function initializeFilters() {
    const tasks = [...new Set(queue.map((record) => record.task))];
    elements.taskFilter.innerHTML = `<option value="all">${S.all_tasks}</option>` +
      tasks.map((task) => `<option value="${escapeHtml(task)}">${escapeHtml(task)}</option>`).join("");
    if (!queue.some((record) => record.kind === preferences.mode)) preferences.mode = queue[0]?.kind || "success";
    elements.taskFilter.value = tasks.includes(preferences.task) ? preferences.task : "all";
    elements.statusFilter.value = preferences.status || "all";
    elements.speedSelect.value = preferences.speed || "1";
  }

  function applyFilters({ preserveCurrent = true } = {}) {
    const task = elements.taskFilter.value;
    const status = elements.statusFilter.value;
    const search = elements.searchInput.value.trim().toLowerCase();
    preferences = { ...preferences, task, status };
    saveJson("e2r-annotation:preferences", preferences);
    elements.modeSwitch.querySelectorAll("button").forEach((b) => b.classList.toggle("selected", b.dataset.mode === preferences.mode));

    filteredQueue = queue.filter((record) => {
      if (record.kind !== preferences.mode) return false;
      if (task !== "all" && record.task !== task) return false;
      const annotation = annotations[record.uid];
      if (status === "event") {
        if (!isEventAnnotation(annotation)) return false;
      } else if (status !== "all" && annotationStatus(record) !== status) {
        return false;
      }
      if (search) {
        const haystack = [record.uid, record.task, record.instruction, labelTitle(annotation?.label)].join(" ").toLowerCase();
        if (!haystack.includes(search)) return false;
      }
      return true;
    });

    if (!filteredQueue.length) {
      currentRecord = null;
      renderQueueList();
      renderEmptyEpisode();
      renderProgress();
      return;
    }
    const keep = preserveCurrent && currentRecord && filteredQueue.some((record) => record.uid === currentRecord.uid);
    const target = keep ? currentRecord : filteredQueue.find((record) => annotationStatus(record) === "pending") || filteredQueue[0];
    renderQueueList();
    loadEpisode(target, { scrollQueue: false });
    renderProgress();
  }

  function renderQueueList(scrollCurrent = true) {
    const previousScrollTop = elements.queueList.scrollTop;
    elements.queueList.textContent = "";
    elements.filteredCount.textContent = `${filteredQueue.length} ${S.count}`.trim();
    if (!filteredQueue.length) {
      const empty = document.createElement("div");
      empty.className = "queue-empty";
      empty.textContent = S.empty_queue;
      elements.queueList.appendChild(empty);
      return;
    }
    const fragment = document.createDocumentFragment();
    filteredQueue.forEach((record, index) => {
      const annotation = annotations[record.uid];
      const status = annotation?.review_status || "pending";
      const active = currentRecord && record.uid === currentRecord.uid;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `queue-item${active ? " active" : ""}`;
      button.dataset.key = record.uid;
      button.setAttribute("role", "option");
      button.setAttribute("aria-selected", active ? "true" : "false");
      const indexElement = document.createElement("span");
      indexElement.className = "queue-index";
      indexElement.textContent = String(index + 1);
      const main = document.createElement("span");
      main.className = "queue-main";
      const title = document.createElement("strong");
      title.textContent = `${record.task} · ${record.source}`;
      const meta = document.createElement("span");
      const labelText = annotation?.label ? S.labels[annotation.label]?.[1] || annotation.label : statusText(status);
      meta.textContent = `${record.duration_seconds.toFixed(1)}s · ${labelText}`;
      main.append(title, meta);
      const dot = document.createElement("span");
      dot.className = `queue-status ${status}`;
      dot.title = statusText(status);
      button.append(indexElement, main, dot);
      button.addEventListener("click", () => loadEpisode(record));
      fragment.appendChild(button);
    });
    elements.queueList.appendChild(fragment);
    if (scrollCurrent && currentRecord) {
      elements.queueList.querySelector(`[data-key="${CSS.escape(currentRecord.uid)}"]`)?.scrollIntoView({ block: "nearest" });
    } else {
      elements.queueList.scrollTop = previousScrollTop;
    }
  }

  function statusText(status) {
    return { pending: S.status_pending, draft: S.status_draft, approved: S.status_approved }[status] || status;
  }

  function renderEmptyEpisode() {
    elements.taskBadge.textContent = "";
    elements.episodeTitle.textContent = "";
    elements.bucketBadge.hidden = true;
    elements.episodeMeta.textContent = S.empty_queue;
    elements.episodePosition.textContent = "0 / 0";
    elements.labelCards.textContent = "";
    elements.pointEditor.textContent = "";
    elements.endpointEditor.hidden = true;
    elements.video.removeAttribute("src");
  }

  // ---------- one video ----------
  function loadEpisode(record, { scrollQueue = true } = {}) {
    if (!record) return;
    currentRecord = record;
    currentFrame = 0;
    frameCount = Math.max(2, record.frame_count);
    localStorage.setItem(`${storageKey}:last`, record.uid);
    elements.fpsInput.value = record.fps.toFixed(record.fps % 1 ? 1 : 0);
    elements.frameInput.max = String(frameCount - 1);
    elements.frameInput.value = "0";
    elements.video.pause();
    elements.playButton.textContent = "▶";

    elements.taskBadge.textContent = record.task.toUpperCase();
    elements.episodeTitle.textContent = record.source;
    elements.bucketBadge.hidden = record.split === "train";
    elements.bucketBadge.textContent = `${S.split} ${record.split}`;
    elements.episodeMeta.textContent = [
      record.instruction,
      `${S.duration} ${record.duration_seconds.toFixed(1)}s`,
      record.kind === "success" ? S.source_success : S.source_failure,
      `${frameCount} ${S.frames}`,
    ].join(" · ");

    const position = filteredQueue.findIndex((item) => item.uid === record.uid);
    elements.episodePosition.textContent = `${Math.max(position + 1, 1)} / ${filteredQueue.length}`;
    elements.previousButton.disabled = position <= 0;
    elements.nextButton.disabled = position < 0 || position >= filteredQueue.length - 1;

    const views = { primary: record.video, ...record.views };
    elements.viewSwitcher.hidden = Object.keys(views).length < 2;
    elements.viewButtons.innerHTML = Object.keys(views).map((view) =>
      `<button type="button" data-view="${view}">${S.views[view] || view}</button>`).join("");
    setVideoView("primary", { preserveFrame: false });

    renderFrame();
    renderAnnotationPanel();
    renderTimeline();
    drawCurve();
    renderQueueList(scrollQueue);
    clearMessage();
  }

  function setVideoView(view, { preserveFrame = true } = {}) {
    if (!currentRecord) return;
    const views = { primary: currentRecord.video, ...currentRecord.views };
    const target = views[view] ? view : "primary";
    const frame = preserveFrame ? currentFrame : 0;
    elements.video.pause();
    elements.viewButtons.querySelectorAll("button").forEach((b) => b.classList.toggle("selected", b.dataset.view === target));
    elements.videoEmptyState.hidden = true;
    elements.video.src = "/" + views[target].split("/").map(encodeURIComponent).join("/");
    elements.video.load();
    currentFrame = frame;
  }

  // ---------- annotation panel ----------
  function renderSaveState() {
    const status = currentRecord ? annotationStatus(currentRecord) : "pending";
    elements.saveState.className = "save-state";
    if (status === "draft") elements.saveState.classList.add("draft");
    if (status === "approved") elements.saveState.classList.add("approved");
    elements.saveState.textContent = statusText(status);
  }

  function renderAnnotationPanel() {
    if (!currentRecord) return;
    const annotation = annotationFor(currentRecord, false);
    const selected = annotation?.label || defaultLabelForRecord(currentRecord);
    const display = annotation || (selected ? blankAnnotation(currentRecord, selected) : null);
    elements.annotationHeading.textContent = currentRecord.kind === "failure" ? S.heading_failure : S.heading_success;
    elements.labelCards.innerHTML = MODE_LABELS[currentRecord.kind].map((label, index) => `
      <button class="label-card${label === selected ? " selected" : ""}" type="button" data-label="${label}" data-key="${index + 1}" aria-pressed="${label === selected}">
        <span class="label-index">${index + 1}</span>
        <span class="label-copy"><strong>${S.labels[label][0]}</strong><small>${S.labels[label][2]}</small></span>
        <span class="label-dot ${COLORS[label]}"></span>
      </button>`).join("");
    renderSaveState();
    renderPointEditor(display);
    renderEndpointEditor(display);
  }

  function pointItem(field, label, shortcut, value, eventIndex = null) {
    const eventAttribute = eventIndex === null ? "" : ` data-event-index="${eventIndex}"`;
    return `
      <div class="point-item">
        <div class="point-item-top"><span>${label}</span><strong>${value === null || value === undefined ? "—" : `F${value}`}</strong></div>
        <button class="point-button" type="button" data-point-field="${field}"${eventAttribute}>${S.set_current} · ${shortcut}</button>
      </div>`;
  }

  function renderPointEditor(annotation) {
    if (!annotation?.label) {
      elements.pointEditor.className = "point-editor empty-editor";
      elements.pointEditor.innerHTML = `<div class="empty-editor-copy"><span class="empty-icon">⌁</span>
        <strong>${S.choose_label_title}</strong><p>${S.choose_label_body}</p></div>`;
      return;
    }
    elements.pointEditor.className = "point-editor";
    if (annotation.label === "success" || annotation.label === "uncertain") {
      const [title, body] = annotation.label === "success" ? [S.clean_title, S.clean_body] : [S.uncertain_title, S.uncertain_body];
      elements.pointEditor.innerHTML = `<div class="clean-editor"><strong>${title}</strong><p>${body}</p></div>`;
      return;
    }
    if (annotation.label === "multiple_events") {
      renderMixedEditor(annotation);
      return;
    }
    const definitions = {
      progress_stall: [S.hes_title, S.hes_body, [["stall_a", S.hes_m, "A"], ["stall_r", S.hes_n, "R"]]],
      regression_and_recovery: [S.reg_title, S.reg_body, [["regression_a", S.reg_m, "A"], ["regression_r", S.reg_r, "R"]]],
      no_further_progress: [S.nfp_title, S.nfp_body, [["failure_a", S.nfp_a, "A"]]],
      unrecovered_regression: [S.pl_title, S.pl_body, [["failure_a", S.pl_a, "A"], ["failure_loss", S.loss_l, "L"]]],
    };
    const [title, description, points] = definitions[annotation.label];
    const items = points.map(([field, label, key]) => pointItem(field, label, key, annotation[field])).join("");
    elements.pointEditor.innerHTML = `
      <div class="point-editor-heading">
        <div><strong>${title}</strong><p>${description}</p></div>
        <span class="current-frame-chip">${S.current} F${currentFrame}</span>
      </div>
      <div class="point-grid ${points.length === 1 ? "single" : ""}">${items}</div>`;
  }

  function renderMixedEditor(annotation) {
    const events = annotation.event_list || [];
    const list = events.length
      ? events.map((event, index) => {
          const hesitation = event.type === "progress_stall";
          return `
            <article class="mixed-event">
              <div class="mixed-event-header">
                <label><strong>${index + 1}.</strong>
                  <select data-event-type="${index}">
                    <option value="progress_stall"${hesitation ? " selected" : ""}>${S.hesitation_event}</option>
                    <option value="regression_and_recovery"${hesitation ? "" : " selected"}>${S.regression_event}</option>
                  </select>
                </label>
                <button class="event-remove-button" type="button" data-remove-event="${index}">×</button>
              </div>
              <div class="point-grid">
                ${pointItem("a", hesitation ? S.mixed_m_h : S.mixed_m_r, "A", event.a, index)}
                ${pointItem("r", hesitation ? S.mixed_n_h : S.mixed_n_r, "R", event.r, index)}
              </div>
            </article>`;
        }).join("")
      : `<div class="empty-editor-copy"><strong>${S.mixed_empty_title}</strong><p>${S.mixed_empty_body}</p></div>`;
    elements.pointEditor.innerHTML = `
      <div class="point-editor-heading">
        <div><strong>${S.mixed_title}</strong><p>${S.mixed_body}</p></div>
        <span class="current-frame-chip">${S.current} F${currentFrame}</span>
      </div>
      <div class="mixed-toolbar">
        <select id="mixedTypeSelect">
          <option value="progress_stall">${S.hesitation_event}</option>
          <option value="regression_and_recovery">${S.regression_event}</option>
        </select>
        <button id="mixedAddButton" class="mixed-add-button" type="button">${S.add_event}</button>
      </div>
      <div class="mixed-event-list">${list}</div>`;
  }

  // S, the last frame by default: the last frame used, and the completion frame of a success.
  function renderEndpointEditor(annotation) {
    const show = annotation?.label && annotation.label !== "uncertain";
    elements.endpointEditor.hidden = !show;
    if (!show) return;
    elements.endpointEditor.className = "endpoint-editor single";
    elements.endpointEditor.innerHTML = pointItem("final_frame", S.final, "S", finalFrame(annotation)) +
      `<p class="endpoint-note">${currentRecord.kind === "success" ? S.endpoint_note : S.endpoint_note_failure}</p>`;
  }

  function selectLabel(label) {
    if (!currentRecord || !MODE_LABELS[currentRecord.kind].includes(label)) return;
    updateAnnotation((annotation) => {
      const keep = { final_frame: annotation.final_frame };
      Object.assign(annotation, blankAnnotation(currentRecord, label), keep);
      // Clicking an event label records the current frame as its first point.
      if (label === "progress_stall") annotation.stall_a = currentFrame;
      if (label === "regression_and_recovery") annotation.regression_a = currentFrame;
      if (label === "multiple_events") annotation.event_list = [{ type: "progress_stall", a: currentFrame, r: null }];
      if (label === "no_further_progress" || label === "unrecovered_regression") annotation.failure_a = currentFrame;
    });
    clearMessage();
    if (!["success", "uncertain"].includes(label)) showMessage(S.msg_first_point(labelTitle(label), currentFrame), "success", 1200);
  }

  function setPoint(field, eventIndex = null) {
    updateAnnotation((annotation) => {
      if (eventIndex !== null) {
        const event = annotation.event_list?.[eventIndex];
        if (event) event[field] = currentFrame;
      } else {
        annotation[field] = currentFrame;
      }
    });
  }

  function addMixedEvent() {
    const type = document.getElementById("mixedTypeSelect")?.value || "progress_stall";
    updateAnnotation((annotation) => { annotation.event_list.push({ type, a: currentFrame, r: null }); });
  }

  // ---------- frames, video and timeline ----------
  function frameAt(seconds) {
    const times = currentRecord.times;
    let lo = 0, hi = times.length - 1;
    while (lo < hi) {
      const mid = (lo + hi + 1) >> 1;
      if (times[mid] <= seconds + 1e-4) lo = mid; else hi = mid - 1;
    }
    return lo;
  }

  function renderFrame() {
    const last = lastFrame();
    const percent = clamp(currentFrame / last, 0, 1) * 100;
    elements.frameReadout.textContent = String(currentFrame);
    elements.frameTotal.textContent = `/ ${last}`;
    elements.frameInput.value = String(currentFrame);
    elements.timelineProgress.style.width = `${percent}%`;
    elements.timelinePlayhead.style.left = `calc(${percent}% - 1px)`;
    elements.timelineMidpoint.textContent = `F${Math.round(last / 2)}`;
    elements.timelineEndpoint.textContent = `F${last}`;
    elements.timeline.setAttribute("aria-valuemax", String(last));
    elements.timeline.setAttribute("aria-valuenow", String(currentFrame));
    const chip = elements.pointEditor.querySelector(".current-frame-chip");
    if (chip) chip.textContent = `${S.current} F${currentFrame}`;
    drawCurve();
  }

  function applyVideoSeek(frame) {
    const times = currentRecord.times;
    // Midway between frame and frame + 1, so the browser shows this frame.
    const target = frame < times.length - 1 ? (times[frame] + times[frame + 1]) / 2 : times[frame] + 0.001;
    if (Math.abs(elements.video.currentTime - target) > 0.0001) elements.video.currentTime = target;
  }

  function seekFrame(frame) {
    if (!currentRecord) return;
    cancelTimelineSeek();
    currentFrame = clamp(Math.round(Number(frame) || 0), 0, lastFrame());
    applyVideoSeek(currentFrame);
    renderFrame();
  }

  function stepFrame(delta) {
    elements.video.pause();
    seekFrame(currentFrame + delta);
  }

  function togglePlayback() {
    if (elements.video.paused) elements.video.play()?.catch?.(() => showMessage(S.msg_play_failed, "error"));
    else elements.video.pause();
  }

  function renderTimeline() {
    elements.timelineMarkers.textContent = "";
    elements.timelineRegions.textContent = "";
    const annotation = annotationFor(currentRecord, false);
    if (!annotation?.label) return;
    const markers = [];
    const regions = [];
    const interval = (start, end, className, startLabel, endLabel) => {
      if (asInteger(start) !== null) markers.push({ frame: start, className, label: startLabel });
      if (asInteger(end) !== null) markers.push({ frame: end, className, label: endLabel });
      if (asInteger(start) !== null && asInteger(end) !== null) regions.push({ start, end, className });
    };
    if (annotation.label === "progress_stall") interval(annotation.stall_a, annotation.stall_r, "hesitation", S.hes_m, S.hes_n);
    if (annotation.label === "regression_and_recovery") interval(annotation.regression_a, annotation.regression_r, "regression", S.reg_m, S.reg_r);
    if (annotation.label === "multiple_events") {
      annotation.event_list.forEach((event, index) => {
        const h = event.type === "progress_stall";
        interval(event.a, event.r, "mixed", `${index + 1}. ${h ? S.mixed_m_h : S.mixed_m_r}`, `${index + 1}. ${h ? S.mixed_n_h : S.mixed_n_r}`);
      });
    }
    if (annotation.label === "no_further_progress" || annotation.label === "unrecovered_regression") {
      const peak = asInteger(annotation.failure_a);
      if (peak !== null) {
        markers.push({ frame: peak, className: annotation.label === "unrecovered_regression" ? "failure" : "hesitation",
                       label: annotation.label === "unrecovered_regression" ? S.pl_a : S.nfp_a });
        regions.push({ start: peak, end: finalFrame(annotation), className: annotation.label === "unrecovered_regression" ? "mixed" : "hesitation" });
      }
      if (asInteger(annotation.failure_loss) !== null) markers.push({ frame: annotation.failure_loss, className: "failure", label: S.loss_l });
    }
    if (finalFrame(annotation) !== frameCount - 1) markers.push({ frame: finalFrame(annotation), className: "failure", label: S.final });

    const last = lastFrame();
    markers.forEach((marker) => {
      const frame = asInteger(marker.frame);
      if (frame === null) return;
      const button = document.createElement("button");
      button.type = "button";
      button.className = `timeline-marker ${marker.className}`;
      button.style.left = `${clamp(frame / last, 0, 1) * 100}%`;
      button.title = `${marker.label} · F${frame}`;
      button.addEventListener("click", (event) => { event.stopPropagation(); seekFrame(frame); });
      elements.timelineMarkers.appendChild(button);
    });
    regions.forEach((region) => {
      const start = asInteger(region.start), end = asInteger(region.end);
      if (start === null || end === null || end <= start) return;
      const element = document.createElement("div");
      element.className = `timeline-region ${region.className}`;
      element.style.left = `${clamp(start / last, 0, 1) * 100}%`;
      element.style.width = `${clamp((end - start) / last, 0, 1) * 100}%`;
      elements.timelineRegions.appendChild(element);
    });
  }

  // ---------- rule preview: what training learns from the annotation ----------
  function curvePoints(annotation) {
    const last = frameCount - 1;
    const times = currentRecord.times;
    const end = finalFrame(annotation);
    if (!annotation?.label) {
      const message = currentRecord.kind === "failure" ? S.curve_pending_failure : S.curve_pending_success;
      return { points: [[0, 0], [last, 1]], incomplete: true, message };
    }
    if (annotation.label === "uncertain") return { points: [], incomplete: true, message: S.curve_uncertain };

    if (currentRecord.kind === "failure") {
      const peak = asInteger(annotation.failure_a);
      if (peak === null) return { points: [[0, 0], [last, 1]], incomplete: true, message: S.curve_need_points };
      const value = 0.6;
      if (annotation.label === "no_further_progress") {
        return { schematic: [[0, 0], [peak, value]], bound: [[peak, value], [end, value]], anchors: [[peak, value]],
                 message: S.curve_nfp(peak) };
      }
      const loss = asInteger(annotation.failure_loss) ?? Math.min(end, frameAt(times[peak] + 0.5));
      return { schematic: [[0, 0], [peak, value]], bound: [[peak, value], [end, value]], anchors: [[peak, value]],
               dropFrom: [loss, value], end, message: S.curve_pl(peak, loss) };
    }

    // Successful videos: time progress, with hesitations held level and their time removed.
    const C = end;
    const stalls = [];
    let hasRegression = false;
    const add = (type, a, b) => {
      if (asInteger(a) === null || asInteger(b) === null || b <= a) return false;
      stalls.push({ type, a, b });
      if (type === "regression_and_recovery") hasRegression = true;
      return true;
    };
    let complete = true;
    if (annotation.label === "progress_stall") complete = add("progress_stall", annotation.stall_a, annotation.stall_r);
    if (annotation.label === "regression_and_recovery") complete = add("regression_and_recovery", annotation.regression_a, annotation.regression_r);
    if (annotation.label === "multiple_events") {
      complete = annotation.event_list.length >= 2;
      annotation.event_list.forEach((e) => { complete = add(e.type, e.a, e.r) && complete; });
    }
    const active = (t) => t - stalls.reduce((sum, s) => sum + Math.max(0, Math.min(t, times[s.b]) - times[s.a]), 0);
    const total = Math.max(1e-6, active(times[C]));
    const progress = (f) => clamp(active(Math.min(times[f], times[C])) / total, 0, 1);
    const points = [];
    for (let f = 0; f <= end; f += Math.max(1, Math.floor(end / 400))) points.push([f, progress(f)]);
    points.push([end, progress(end)]);
    const masked = stalls.filter((s) => s.type === "regression_and_recovery").map((s) => [s.a, s.b, progress(s.a)]);
    const anchors = stalls.flatMap((s) => [[s.a, progress(s.a)], [s.b, progress(s.b)]]);
    let message = S.curve_clean;
    if (!complete) message = S.curve_need_points;
    else if (annotation.label === "progress_stall") message = S.curve_hes(stalls[0].a, stalls[0].b, progress(stalls[0].a).toFixed(3));
    else if (annotation.label === "regression_and_recovery") message = S.curve_reg(stalls[0].a, stalls[0].b);
    else if (annotation.label === "multiple_events") message = hasRegression ? S.curve_mixed_reg : S.curve_mixed_hes;
    // A success with a regression gives relations only: the time curve is shown as a guide.
    return { points, masked, anchors, incomplete: !complete, guideOnly: hasRegression, message };
  }

  function drawCurve() {
    const canvas = elements.curveCanvas;
    const rect = canvas.getBoundingClientRect();
    if (!currentRecord || !rect.width || !rect.height) return;
    const ratio = window.devicePixelRatio || 1;
    const width = Math.round(rect.width * ratio), height = Math.round(rect.height * ratio);
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);

    const annotation = annotationFor(currentRecord, false) || blankAnnotation(currentRecord, defaultLabelForRecord());
    const result = curvePoints(annotation);
    const css = getComputedStyle(document.documentElement);
    const token = (name) => css.getPropertyValue(name).trim();
    const C = { line: token("--accent"), guide: token("--curve-guide"), bound: token("--purple"), drop: token("--red"),
                now: token("--muted-strong"), mask: `color-mix(in srgb, ${token("--purple")} 14%, transparent)` };
    const last = lastFrame();
    const x = (frame) => (frame / last) * rect.width;
    const y = (progress) => rect.height - clamp(progress, 0, 1) * rect.height;
    const line = (points, color, dash = []) => {
      if (!points?.length) return;
      context.strokeStyle = color;
      context.setLineDash(dash);
      context.beginPath();
      points.forEach(([f, p], i) => (i ? context.lineTo(x(f), y(p)) : context.moveTo(x(f), y(p))));
      context.stroke();
      context.setLineDash([]);
    };

    (result.masked || []).forEach(([start, end, value]) => {
      context.fillStyle = C.mask;
      context.fillRect(x(start), 0, Math.max(1, x(end) - x(start)), rect.height);
      context.lineWidth = 1;
      line([[start, value], [end, value]], C.bound, [4, 4]);
    });
    context.lineWidth = 2;
    if (result.points?.length) {
      const color = result.incomplete || result.guideOnly ? C.guide : C.line;
      const dash = result.incomplete || result.guideOnly ? [5, 5] : [];
      if (result.masked?.length) {
        // No absolute target inside a regression.
        let segment = [];
        result.points.forEach((point) => {
          const inside = result.masked.some(([a, b]) => point[0] > a && point[0] < b);
          if (inside) { line(segment, color, dash); segment = []; } else segment.push(point);
        });
        line(segment, color, dash);
      } else {
        line(result.points, color, dash);
      }
    }
    if (result.schematic) line(result.schematic, C.guide, [5, 5]);
    if (result.bound) { context.lineWidth = 1.5; line(result.bound, C.bound, [4, 4]); }
    if (result.dropFrom) {
      context.lineWidth = 2;
      const [loss, value] = result.dropFrom;
      line([[result.anchors[0][0], value], [loss, value * 0.55], [result.end, value * 0.55]], C.drop, [5, 4]);
    }
    if (!result.incomplete) {
      context.fillStyle = C.line;
      (result.anchors || []).forEach(([frame, progress]) => {
        context.beginPath();
        context.arc(x(frame), y(progress), 3.3, 0, Math.PI * 2);
        context.fill();
      });
    }
    context.strokeStyle = C.now;
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x(currentFrame), 0);
    context.lineTo(x(currentFrame), rect.height);
    context.stroke();
    elements.curveMessage.textContent = result.message;
  }

  // ---------- checks and saving ----------
  function validateAnnotation(annotation) {
    if (!annotation?.label) return S.err_label;
    const last = frameCount - 1;
    const end = finalFrame(annotation);
    const validFrame = (value) => Number.isInteger(value) && value >= 0 && value <= last;
    const validInterval = (start, stop, name) => {
      if (!validFrame(start) || !validFrame(stop)) return S.err_range(name, last);
      if (stop <= start) return S.err_order(name);
      return "";
    };
    if (currentRecord.kind === "success" && annotation.label !== "uncertain") {
      const C = end;
      if (C <= 0) return S.err_s;
      const ends = [];
      if (annotation.label === "progress_stall") {
        const error = validInterval(annotation.stall_a, annotation.stall_r, S.hes_name);
        if (error) return error;
        ends.push(annotation.stall_r);
      }
      if (annotation.label === "regression_and_recovery") {
        const error = validInterval(annotation.regression_a, annotation.regression_r, S.reg_name);
        if (error) return error;
        if (annotation.regression_a === 0 || annotation.regression_r === last) return S.err_inside;
        ends.push(annotation.regression_r);
      }
      if (annotation.label === "multiple_events") {
        if (annotation.event_list.length < 2) return S.err_mixed_count;
        let previousEnd = -1;
        for (let index = 0; index < annotation.event_list.length; index += 1) {
          const event = annotation.event_list[index];
          const error = validInterval(event.a, event.r, S.event_name(index + 1));
          if (error) return error;
          if (event.a <= previousEnd) return S.err_mixed_order;
          previousEnd = event.r;
          ends.push(event.r);
        }
      }
      if (ends.some((frame) => frame > C)) return S.err_after_c;
    }
    if (annotation.label === "no_further_progress" || annotation.label === "unrecovered_regression") {
      const peak = annotation.failure_a;
      if (!validFrame(peak) || peak === 0 || peak >= end) return S.err_failure_point(end + 1);
      const loss = asInteger(annotation.failure_loss);
      if (annotation.label === "unrecovered_regression" && loss !== null && (loss <= peak || loss > end)) return S.err_loss;
    }
    return "";
  }

  async function saveCurrentAndAdvance() {
    if (!currentRecord) return;
    const annotation = annotationFor(currentRecord, true);
    const error = validateAnnotation(annotation);
    if (error) { showMessage(error, "error", 0); return; }
    const response = await fetch("/api/annotation", {
      method: "POST", headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ uid: currentRecord.uid, annotation: toSaved(annotation, currentRecord) }),
    });
    const result = await response.json();
    if (!response.ok) { showMessage(result.error, "error", 0); return; }
    annotations[currentRecord.uid] = fromSaved(result, currentRecord);
    persistDrafts();
    renderAnnotationPanel();
    renderQueueList(false);
    renderProgress();
    showMessage(S.msg_saved, "success", 900);
    window.setTimeout(goToNextPending, 180);
  }

  // Saved annotations stay in the project; clearing drops the draft and reloads the saved one.
  async function clearCurrentAnnotation() {
    if (!currentRecord || !annotations[currentRecord.uid] || !window.confirm(S.msg_confirm_clear)) return;
    delete annotations[currentRecord.uid];
    persistDrafts();
    const project = await (await fetch("/api/project")).json();
    const saved = project.annotations[currentRecord.uid];
    if (saved) annotations[currentRecord.uid] = fromSaved(saved, currentRecord);
    renderAnnotationPanel();
    renderTimeline();
    drawCurve();
    renderQueueList(false);
    renderProgress();
    showMessage(S.msg_cleared, "success", 1100);
  }

  function renderProgress() {
    const rows = queue.filter((record) => record.kind === preferences.mode);
    const completed = rows.filter((record) => annotationStatus(record) === "approved").length;
    elements.progressLabel.textContent = preferences.mode === "failure" ? S.progress_failure : S.progress_success;
    elements.progressNumbers.textContent = `${completed} / ${rows.length}`;
    elements.progressFill.style.width = `${rows.length ? (100 * completed) / rows.length : 0}%`;
  }

  function goToOffset(delta) {
    if (!currentRecord || !filteredQueue.length) return;
    const index = filteredQueue.findIndex((record) => record.uid === currentRecord.uid);
    const target = clamp(index + delta, 0, filteredQueue.length - 1);
    if (target !== index) loadEpisode(filteredQueue[target]);
  }

  function goToNextPending() {
    if (!filteredQueue.length) return;
    const index = filteredQueue.findIndex((record) => record.uid === currentRecord?.uid);
    for (let offset = 1; offset <= filteredQueue.length; offset += 1) {
      const candidate = filteredQueue[(index + offset) % filteredQueue.length];
      if (annotationStatus(candidate) !== "approved") { loadEpisode(candidate); return; }
    }
    if (index < filteredQueue.length - 1) loadEpisode(filteredQueue[index + 1]);
  }

  function showMessage(message, kind = "error", timeout = 2400) {
    if (messageTimer) window.clearTimeout(messageTimer);
    elements.validationMessage.hidden = false;
    elements.validationMessage.className = `validation-message${kind === "success" ? " success" : ""}`;
    elements.validationMessage.textContent = message;
    if (timeout > 0) messageTimer = window.setTimeout(clearMessage, timeout);
  }

  function clearMessage() {
    if (messageTimer) window.clearTimeout(messageTimer);
    messageTimer = null;
    elements.validationMessage.hidden = true;
    elements.validationMessage.textContent = "";
  }

  // ---------- timeline dragging: the frame follows at once, video seeks every 80 ms ----------
  let timelineDragging = false;
  let timelinePointerId = null;
  let pendingTimelineVideoFrame = null;
  let timelineVideoSeekTimer = null;
  let lastTimelineVideoSeekAt = 0;
  const TIMELINE_VIDEO_SEEK_INTERVAL_MS = 80;

  function cancelTimelineSeek() {
    if (timelineVideoSeekTimer !== null) window.clearTimeout(timelineVideoSeekTimer);
    timelineVideoSeekTimer = null;
    pendingTimelineVideoFrame = null;
    timelineDragging = false;
    timelinePointerId = null;
    elements.timeline.classList.remove("dragging");
  }

  function flushTimelineVideoSeek() {
    if (timelineVideoSeekTimer !== null) window.clearTimeout(timelineVideoSeekTimer);
    timelineVideoSeekTimer = null;
    if (pendingTimelineVideoFrame === null) return;
    const frame = pendingTimelineVideoFrame;
    pendingTimelineVideoFrame = null;
    applyVideoSeek(frame);
    lastTimelineVideoSeekAt = performance.now();
  }

  function scheduleTimelineVideoSeek(frame) {
    pendingTimelineVideoFrame = frame;
    if (timelineVideoSeekTimer !== null) return;
    const delay = Math.max(0, TIMELINE_VIDEO_SEEK_INTERVAL_MS - (performance.now() - lastTimelineVideoSeekAt));
    timelineVideoSeekTimer = window.setTimeout(() => { timelineVideoSeekTimer = null; flushTimelineVideoSeek(); }, delay);
  }

  function timelineSeekFromPointer(event, final = false) {
    const rect = elements.timeline.getBoundingClientRect();
    currentFrame = Math.round(clamp((event.clientX - rect.left) / rect.width, 0, 1) * (frameCount - 1));
    renderFrame();
    pendingTimelineVideoFrame = currentFrame;
    if (final) flushTimelineVideoSeek(); else scheduleTimelineVideoSeek(currentFrame);
  }

  function finishTimelineDrag(event, usePointerPosition = true) {
    if (!timelineDragging || (timelinePointerId !== null && event.pointerId !== timelinePointerId)) return;
    if (usePointerPosition) timelineSeekFromPointer(event, true); else flushTimelineVideoSeek();
    const pointerId = timelinePointerId;
    timelineDragging = false;
    timelinePointerId = null;
    elements.timeline.classList.remove("dragging");
    if (pointerId !== null && elements.timeline.hasPointerCapture(pointerId)) elements.timeline.releasePointerCapture(pointerId);
  }

  // ---------- events ----------
  function bindEvents() {
    elements.modeSwitch.addEventListener("click", (event) => {
      const button = event.target.closest("[data-mode]");
      if (!button || button.dataset.mode === preferences.mode) return;
      preferences.mode = button.dataset.mode;
      applyFilters({ preserveCurrent: false });
    });
    [elements.taskFilter, elements.statusFilter].forEach((select) => select.addEventListener("change", () => applyFilters({ preserveCurrent: false })));
    elements.searchInput.addEventListener("input", () => applyFilters());
    elements.previousButton.addEventListener("click", () => goToOffset(-1));
    elements.nextButton.addEventListener("click", () => goToOffset(1));
    elements.jumpBackButton.addEventListener("click", () => stepFrame(-5));
    elements.stepBackButton.addEventListener("click", () => stepFrame(-1));
    elements.playButton.addEventListener("click", togglePlayback);
    elements.stepForwardButton.addEventListener("click", () => stepFrame(1));
    elements.jumpForwardButton.addEventListener("click", () => stepFrame(5));
    elements.frameInput.addEventListener("change", () => seekFrame(elements.frameInput.value));
    elements.speedSelect.addEventListener("change", () => {
      elements.video.playbackRate = Number(elements.speedSelect.value);
      preferences.speed = elements.speedSelect.value;
      saveJson("e2r-annotation:preferences", preferences);
    });
    elements.viewButtons.addEventListener("click", (event) => {
      const button = event.target.closest("[data-view]");
      if (button) setVideoView(button.dataset.view);
    });

    elements.video.addEventListener("loadedmetadata", () => {
      elements.videoEmptyState.hidden = true;
      elements.video.playbackRate = Number(elements.speedSelect.value);
      seekFrame(currentFrame);
    });
    elements.video.addEventListener("error", () => { if (elements.video.getAttribute("src")) elements.videoEmptyState.hidden = false; });
    elements.video.addEventListener("play", () => { elements.playButton.textContent = "❚❚"; });
    elements.video.addEventListener("pause", () => { elements.playButton.textContent = "▶"; });
    elements.video.addEventListener("timeupdate", () => {
      if (currentRecord && !timelineDragging && !elements.video.seeking && !elements.video.paused) {
        currentFrame = frameAt(elements.video.currentTime);
        renderFrame();
      }
    });
    elements.video.addEventListener("ended", () => seekFrame(frameCount - 1));

    elements.timeline.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest(".timeline-marker") || !currentRecord) return;
      event.preventDefault();
      elements.video.pause();
      timelineDragging = true;
      timelinePointerId = event.pointerId;
      elements.timeline.classList.add("dragging");
      elements.timeline.setPointerCapture(event.pointerId);
      timelineSeekFromPointer(event);
    });
    elements.timeline.addEventListener("pointermove", (event) => {
      if (!timelineDragging || event.pointerId !== timelinePointerId) return;
      event.preventDefault();
      timelineSeekFromPointer(event);
    });
    elements.timeline.addEventListener("pointerup", (event) => finishTimelineDrag(event, true));
    elements.timeline.addEventListener("pointercancel", (event) => finishTimelineDrag(event, false));

    elements.labelCards.addEventListener("click", (event) => {
      const card = event.target.closest(".label-card");
      if (card) selectLabel(card.dataset.label);
    });
    const pointClick = (event) => {
      const pointButton = event.target.closest("[data-point-field]");
      if (pointButton) {
        const eventIndex = pointButton.dataset.eventIndex === undefined ? null : Number(pointButton.dataset.eventIndex);
        setPoint(pointButton.dataset.pointField, eventIndex);
        return;
      }
      const removeButton = event.target.closest("[data-remove-event]");
      if (removeButton) { updateAnnotation((a) => { a.event_list.splice(Number(removeButton.dataset.removeEvent), 1); }); return; }
      if (event.target.closest("#mixedAddButton")) addMixedEvent();
    };
    elements.pointEditor.addEventListener("click", pointClick);
    elements.endpointEditor.addEventListener("click", pointClick);
    elements.pointEditor.addEventListener("change", (event) => {
      const selector = event.target.closest("[data-event-type]");
      if (selector) updateAnnotation((a) => { a.event_list[Number(selector.dataset.eventType)].type = selector.value; });
    });

    elements.saveButton.addEventListener("click", saveCurrentAndAdvance);
    elements.clearButton.addEventListener("click", clearCurrentAnnotation);
    elements.helpButton.addEventListener("click", () => elements.helpDialog.showModal());
    elements.closeHelpButton.addEventListener("click", () => elements.helpDialog.close());
    elements.helpDialog.addEventListener("click", (event) => { if (event.target === elements.helpDialog) elements.helpDialog.close(); });
    elements.themeButton.addEventListener("click", () => {
      const theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
      document.documentElement.dataset.theme = theme;
      localStorage.setItem("e2r-annotation:theme", theme);
      renderThemeButton();
      drawCurve();
    });
    elements.languageButton.addEventListener("click", () => {
      localStorage.setItem("e2r-annotation:language", LANG === "zh" ? "en" : "zh");
      location.reload();
    });
    window.addEventListener("resize", drawCurve);
    document.addEventListener("keydown", handleKeyboard);
  }

  function handleKeyboard(event) {
    const typing = ["input", "textarea", "select"].includes(document.activeElement?.tagName?.toLowerCase());
    if ((event.metaKey || event.ctrlKey) && event.key === "Enter") { event.preventDefault(); saveCurrentAndAdvance(); return; }
    if (typing || elements.helpDialog.open || !currentRecord) return;
    const key = event.key.toLowerCase();
    if (event.key === "Enter") {
      event.preventDefault();
      if (!event.repeat) saveCurrentAndAdvance();
    } else if (event.key === " ") {
      event.preventDefault();
      togglePlayback();
    } else if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
      event.preventDefault();
      stepFrame((event.key === "ArrowLeft" ? -1 : 1) * (event.shiftKey ? 5 : 1));
    } else if (key === "j") {
      goToOffset(-1);
    } else if (key === "k") {
      goToOffset(1);
    } else if (/^[1-5]$/.test(event.key)) {
      const label = MODE_LABELS[currentRecord.kind][Number(event.key) - 1];
      if (label) selectLabel(label);
    } else if (["a", "r", "l", "s", "m", "n"].includes(key)) {
      setShortcutPoint(key);
    }
  }

  function setShortcutPoint(key) {
    const annotation = annotationFor(currentRecord, false) || (key === "s" ? annotationFor(currentRecord, true) : null);
    if (!annotation?.label) return;
    if (key === "s") { setPoint("final_frame"); return; }
    // A and R set the two points of an event; M and N also work.
    if (key === "m") key = "a";
    if (key === "n") key = "r";
    const first = key === "a";
    if (annotation.label === "progress_stall" && key !== "l") setPoint(first ? "stall_a" : "stall_r");
    else if (annotation.label === "regression_and_recovery" && key !== "l") setPoint(first ? "regression_a" : "regression_r");
    else if (annotation.label === "multiple_events" && annotation.event_list.length && key !== "l") {
      setPoint(first ? "a" : "r", annotation.event_list.length - 1);
    } else if ((annotation.label === "no_further_progress" || annotation.label === "unrecovered_regression") && first) setPoint("failure_a");
    else if (annotation.label === "unrecovered_regression" && key === "l") setPoint("failure_loss");
  }

  // ---------- start ----------
  function renderThemeButton() {
    const dark = document.documentElement.dataset.theme === "dark";
    elements.themeButton.textContent = dark ? "☀" : "☾";
    elements.themeButton.title = dark ? S.theme_light : S.theme_dark;
    elements.themeButton.setAttribute("aria-label", elements.themeButton.title);
  }

  function translatePage() {
    document.documentElement.lang = LANG === "zh" ? "zh-CN" : "en";
    elements.brandSubtitle.textContent = S.subtitle;
    elements.languageButton.textContent = LANG === "zh" ? "English" : "中文";
    document.querySelectorAll("[data-i18n]").forEach((el) => { el.textContent = S[el.dataset.i18n]; });
    document.querySelectorAll("[data-i18n-placeholder]").forEach((el) => { el.placeholder = S[el.dataset.i18nPlaceholder]; });
    elements.helpContent.innerHTML = S.help_html;
    renderThemeButton();
  }

  async function start() {
    translatePage();
    const project = await (await fetch("/api/project")).json();
    queue = buildQueue(project);
    storageKey = `e2r-annotation:${project.name || "project"}:drafts`;
    const drafts = loadJson(storageKey, {});
    for (const record of queue) {
      const saved = project.annotations[record.uid];
      const draft = drafts[record.uid];
      if (draft && (!saved || draft.updated_at > (saved.saved_at || ""))) annotations[record.uid] = draft;
      else if (saved) annotations[record.uid] = fromSaved(saved, record);
    }
    initializeFilters();
    bindEvents();
    applyFilters({ preserveCurrent: false });
    const lastUid = localStorage.getItem(`${storageKey}:last`);
    const last = filteredQueue.find((record) => record.uid === lastUid);
    if (last) loadEpisode(last);
  }

  start();
})();
