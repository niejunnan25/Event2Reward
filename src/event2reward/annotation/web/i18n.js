// Interface text of the annotation page, in Chinese and English. The labels and points are the paper's.
window.STRINGS = {
  zh: {
    subtitle: "标注 · 机器人任务进度",
    help: "标注说明", export: "导出标注", queue: "视频队列", theme_light: "切换到浅色", theme_dark: "切换到深色",
    mode_success: "成功轨迹", mode_failure: "失败轨迹",
    progress_success: "成功轨迹进度", progress_failure: "失败轨迹进度",
    task: "任务", status: "状态", all_tasks: "全部任务",
    status_all: "全部状态", status_pending: "待标注", status_draft: "草稿", status_approved: "已完成", status_event: "仅含事件",
    search: "搜索视频或任务…", count: "条", empty_queue: "当前筛选条件下没有视频。请调整任务或状态筛选。",
    video_missing: "视频未能载入", video_missing_hint: "请确认视频仍在项目的 media/ 文件夹中。",
    jump_frame: "跳转帧", speed: "速度",
    curve_title: "进度曲线预览", curve_legend_rule: "训练目标", curve_legend_now: "当前帧",
    axis_progress: "进度", axis_start: "起点", axis_time: "时间 / 帧", axis_end: "终点",
    heading_success: "成功轨迹的事件", heading_failure: "失败轨迹的事件",
    save: "保存并下一条", clear: "清除当前标注", help_title: "标注说明与快捷键",
    duration: "时长", frames: "帧", split: "划分", source_success: "成功", source_failure: "失败",
    views: { primary: "主视角", wrist: "腕部视角", right_wrist: "右腕视角" },
    labels: {
      success: ["成功", "成功", "没有停滞和回退；看完直接回车"],
      progress_stall: ["进度停滞", "停滞", "a：停滞开始；r：进度恢复推进"],
      regression_and_recovery: ["回退并恢复", "回退恢复", "a：进度丢失前的状态；r：首次恢复到该状态"],
      multiple_events: ["多个事件", "多事件", "两段及以上进度停滞或回退并恢复"],
      no_further_progress: ["无进一步进展", "无进展", "a：最后一个有实际进展的状态"],
      unrecovered_regression: ["未恢复的回退", "未恢复回退", "a：进度丢失前的最后状态"],
      uncertain: ["不确定", "不确定", "证据不足；不进入训练"],
    },
    choose_label_title: "先选择一种事件", choose_label_body: "然后在视频中定位帧，设置这个事件需要的点。",
    clean_title: "✓ 不需要标点", clean_body: "训练使用从 F0 到 S 的时间进度。",
    uncertain_title: "这条视频不进入训练", uncertain_body: "证据不足以确定事件或边界，不需要标点。",
    current: "当前", set_current: "设为当前帧",
    hes_title: "进度停滞", hes_body: "a 是停滞开始、不再有进展的第一帧；r 是进度恢复推进的第一帧。",
    hes_m: "停滞开始 a", hes_n: "恢复推进 r",
    reg_title: "回退并恢复", reg_body: "a 是进度丢失前的最后一帧；r 是首次恢复到 a 时状态的那一帧，不是谷底，也不是重新开始动的时刻。",
    reg_m: "丢失前 a", reg_r: "恢复到同等状态 r",
    nfp_title: "无进一步进展", nfp_body: "a 是最后一个有实际进展的状态；之后直到 S 不再有进展。",
    nfp_a: "最后有进展 a",
    pl_title: "未恢复的回退", pl_body: "a 是进度丢失前的最后状态；l（可选）是能看到进度丢失的一帧。之后直到 S 没有恢复。",
    pl_a: "丢失前 a", loss_l: "可见丢失 l（可选）",
    mixed_title: "多个事件", mixed_body: "按发生顺序记录两段及以上进度停滞或回退并恢复。",
    mixed_empty_title: "尚未添加事件", mixed_empty_body: "至少需要两段事件。",
    hesitation_event: "进度停滞", regression_event: "回退并恢复", add_event: "＋ 在当前帧添加事件",
    mixed_m_h: "停滞开始 a", mixed_n_h: "恢复推进 r", mixed_m_r: "丢失前 a", mixed_n_r: "恢复到同等状态 r",
    final: "最终采样点 S",
    endpoint_note: "S 默认是最后一帧，也是任务完成帧：进度在 S 到达 1，S 之后不参加训练。任务提前完成、结尾有空等时，把 S 前移到完成的那一帧。",
    endpoint_note_failure: "S 默认是最后一帧；结尾有空等时把 S 前移，S 之后不参加训练。",
    msg_first_point: (title, f) => `已选择${title}，a 已设为当前帧 F${f}。`,
    msg_saved: "已保存。", msg_cleared: "当前标注已清除。", msg_confirm_clear: "确定清除这条视频的标注吗？",
    msg_play_failed: "视频无法播放，请检查文件或浏览器格式支持。",
    err_label: "请选择一种事件。",
    err_range: (name, last) => `${name}的 a、r 必须位于 F0–F${last}。`,
    err_order: (name) => `${name}的 r 必须晚于 a。`,
    err_inside: "回退并恢复的 a、r 应位于视频内部。",
    err_after_c: "事件必须在最终采样点 S 之前结束。",
    err_failure_point: (last) => `a 必须位于 F1–F${last - 1}，且早于 S。`,
    err_loss: "可见丢失 l 必须晚于 a，且不晚于 S。",
    err_mixed_count: "多个事件至少需要两段事件。",
    err_mixed_order: "事件必须按时间排列，前后两段不能重叠或共用同一帧。",
    err_s: "最终采样点 S 必须晚于 F0。",
    hes_name: "进度停滞", reg_name: "回退并恢复", event_name: (i) => `第 ${i} 段事件`,
    curve_pending_success: "请选择一种事件，并设置它需要的点。",
    curve_pending_failure: "请选择一种事件，并设置 a。",
    curve_clean: "成功：F0 到 S 的进度随时间均匀增长。",
    curve_uncertain: "不确定：这条视频不进入训练。",
    curve_hes: (a, r, v) => `进度停滞：F${a}–F${r} 进度保持 ${v}，停滞时间不计入进度。`,
    curve_reg: (a, r) => `回退并恢复：F${a} 与 F${r} 进度相同，中间各帧只要求低于两端；这条视频不提供绝对进度标签。`,
    curve_mixed_hes: "多个进度停滞：每段停滞进度保持不变，停滞时间不计入进度。",
    curve_mixed_reg: "含回退并恢复：回退中间各帧只要求低于两端；这条视频不提供绝对进度标签。",
    curve_nfp: (a) => `F${a} 之后各帧的进度不能高于 F${a}。`,
    curve_pl: (a, l) => `F${l} 起各帧的进度必须低于 F${a}。`,
    curve_need_points: "请设置这个事件需要的点。",
    help_html: `
      <section><h3>只标事件的点，不填进度值</h3>
        <p>成功和失败视频分开标，用左上角的“成功轨迹 / 失败轨迹”切换。同一任务的视频按时长从短到长排列：先看几条紧凑的执行，再判断较长视频里的停滞和回退。</p></section>
      <div class="guide-grid">
        <article><strong>成功</strong><span>0 个点</span><p>默认选中。看完直接按 Enter 保存并进入下一条；保存后才算标完。</p></article>
        <article><strong>进度停滞</strong><span>a、r</span><p><b>a</b>：停滞开始；<b>r</b>：进度恢复推进。</p></article>
        <article><strong>回退并恢复</strong><span>a、r</span><p><b>a</b>：进度丢失前的状态；<b>r</b>：首次恢复到该状态，不是谷底，也不是重新开始动的时刻。</p></article>
        <article><strong>多个事件</strong><span>每段 a、r</span><p>两段及以上进度停滞或回退并恢复，按时间顺序。</p></article>
        <article><strong>无进一步进展</strong><span>a</span><p>失败视频：拖到最后一个有实际进展的状态，点这个事件，当前帧就是 a。</p></article>
        <article><strong>未恢复的回退</strong><span>a，可选 l</span><p>失败视频：a 是进度丢失前的最后状态；可再按 L 标出能看到丢失的一帧。</p></article>
        <article><strong>不确定</strong><span>0 个点</span><p>证据不足以确定事件或边界；不进入训练。</p></article>
      </div>
      <section><h3>快捷键</h3><dl class="shortcut-list">
        <div><dt>空格</dt><dd>播放 / 暂停</dd></div>
        <div><dt>← / →</dt><dd>前后 1 帧</dd></div>
        <div><dt>Shift + ← / →</dt><dd>前后 5 帧</dd></div>
        <div><dt>J / K</dt><dd>上一条 / 下一条</dd></div>
        <div><dt>1–5</dt><dd>选择事件；点事件时当前帧自动记为 a</dd></div>
        <div><dt>A / R</dt><dd>把当前帧设为 a / r</dd></div>
        <div><dt>L</dt><dd>可见丢失 l（未恢复的回退）</dd></div>
        <div><dt>S</dt><dd>最终采样点（也是成功视频的完成帧）</dd></div>
        <div><dt>Enter</dt><dd>保存并进入下一条</dd></div>
      </dl></section>
      <section class="warning-note"><strong>注意</strong>
        <p>标注保存在项目文件夹的 annotations.json 中。“导出标注”下载 JSONL，用 event2reward convert annotations 转成训练数据。任务完成后的停留不算停滞；只判断录像里看得到的事实。</p></section>`,
  },
  en: {
    subtitle: "annotation · robot task progress",
    help: "Guide", export: "Export", queue: "Video queue", theme_light: "Switch to light", theme_dark: "Switch to dark",
    mode_success: "Successful", mode_failure: "Failed",
    progress_success: "Successful videos", progress_failure: "Failed videos",
    task: "Task", status: "Status", all_tasks: "All tasks",
    status_all: "All", status_pending: "To do", status_draft: "Draft", status_approved: "Done", status_event: "With events",
    search: "Search videos or tasks…", count: "", empty_queue: "No videos match the filters.",
    video_missing: "The video did not load", video_missing_hint: "Check that it is still in the project's media/ folder.",
    jump_frame: "Frame", speed: "Speed",
    curve_title: "Progress preview", curve_legend_rule: "Training target", curve_legend_now: "Current frame",
    axis_progress: "progress", axis_start: "start", axis_time: "time / frame", axis_end: "end",
    heading_success: "Events of a successful video", heading_failure: "Events of a failed video",
    save: "Save and next", clear: "Clear this annotation", help_title: "How to annotate",
    duration: "duration", frames: "frames", split: "split", source_success: "successful", source_failure: "failed",
    views: { primary: "Main view", wrist: "Wrist view", right_wrist: "Right wrist" },
    labels: {
      success: ["Success", "Success", "No stall or regression; watch, then press Enter"],
      progress_stall: ["Progress stall", "Stall", "a: the stall begins; r: progress resumes"],
      regression_and_recovery: ["Regression and recovery", "Regression", "a: the last state before the loss; r: the first state where it is restored"],
      multiple_events: ["Multiple events", "Multiple", "Two or more progress stalls or regressions and recoveries"],
      no_further_progress: ["No further progress", "No progress", "a: the last state with real progress"],
      unrecovered_regression: ["Unrecovered regression", "Unrecovered", "a: the last state before the loss"],
      uncertain: ["Uncertain", "Uncertain", "Not enough evidence; kept out of training"],
    },
    choose_label_title: "Choose an event first", choose_label_body: "Then find the frames in the video and set the points the event needs.",
    clean_title: "✓ No points needed", clean_body: "Training uses time progress from F0 to S.",
    uncertain_title: "Kept out of training", uncertain_body: "The event or a boundary cannot be told reliably; no points needed.",
    current: "now", set_current: "Set to current",
    hes_title: "Progress stall", hes_body: "a: the first frame without progress. r: the first frame where progress resumes.",
    hes_m: "Stall begins a", hes_n: "Progress resumes r",
    reg_title: "Regression and recovery", reg_body: "a: the last frame before progress is lost. r: the first frame where the state of a is restored; not the lowest point, and not where motion restarts.",
    reg_m: "Before the loss a", reg_r: "Restored r",
    nfp_title: "No further progress", nfp_body: "a: the last state with real progress; nothing is gained after it until S.",
    nfp_a: "Last progress a",
    pl_title: "Unrecovered regression", pl_body: "a: the last state before the loss. l (optional): the frame where the loss is visible. It is not restored until S.",
    pl_a: "Before the loss a", loss_l: "Loss visible l (optional)",
    mixed_title: "Multiple events", mixed_body: "Record two or more progress stalls or regressions and recoveries in time order.",
    mixed_empty_title: "No events yet", mixed_empty_body: "Add at least two events.",
    hesitation_event: "Progress stall", regression_event: "Regression and recovery", add_event: "＋ Add event at this frame",
    mixed_m_h: "Stall begins a", mixed_n_h: "Progress resumes r", mixed_m_r: "Before the loss a", mixed_n_r: "Restored r",
    final: "Last frame used S",
    endpoint_note: "S defaults to the last frame and is the completion frame: progress reaches 1 at S, and later frames are not used. If the task is done before the video ends, move S to that frame.",
    endpoint_note_failure: "S defaults to the last frame; move it earlier to drop idle frames at the end.",
    msg_first_point: (title, f) => `${title}: a is set to the current frame F${f}.`,
    msg_saved: "Saved.", msg_cleared: "Annotation cleared.", msg_confirm_clear: "Clear the annotation of this video?",
    msg_play_failed: "The video cannot be played in this browser.",
    err_label: "Choose an event.",
    err_range: (name, last) => `a and r of the ${name} must lie in F0–F${last}.`,
    err_order: (name) => `r of the ${name} must come after a.`,
    err_inside: "a and r of a regression and recovery must lie inside the video.",
    err_after_c: "Events must end by S.",
    err_failure_point: (last) => `a must lie in F1–F${last - 1}, before S.`,
    err_loss: "The loss frame l must come after a and by S.",
    err_mixed_count: "Multiple events needs at least two events.",
    err_mixed_order: "Events must be in time order and must not overlap or share a frame.",
    err_s: "S must come after F0.",
    hes_name: "progress stall", reg_name: "regression and recovery", event_name: (i) => `event ${i}`,
    curve_pending_success: "Choose an event, and set the points it needs.",
    curve_pending_failure: "Choose an event, and set a.",
    curve_clean: "Success: progress grows evenly with time from F0 to S.",
    curve_uncertain: "Uncertain: this video is kept out of training.",
    curve_hes: (a, r, v) => `Progress stall: progress stays at ${v} from F${a} to F${r}; that time does not count as progress.`,
    curve_reg: (a, r) => `Regression and recovery: F${a} and F${r} have the same progress, and the frames between must stay below it. This video gives no absolute progress targets.`,
    curve_mixed_hes: "Progress stalls: progress stays level in each; that time does not count as progress.",
    curve_mixed_reg: "With a regression and recovery, the frames inside it must stay below both ends; this video gives no absolute progress targets.",
    curve_nfp: (a) => `After F${a}, no frame may have higher progress than F${a}.`,
    curve_pl: (a, l) => `From F${l} on, every frame must have lower progress than F${a}.`,
    curve_need_points: "Set the points this event needs.",
    help_html: `
      <section><h3>Mark events, not progress values</h3>
        <p>Successful and failed videos are annotated separately; switch with the two buttons at the top left. Within a task, videos are listed from short to long: watch a few compact executions first, then judge the stalls and regressions in longer ones.</p></section>
      <div class="guide-grid">
        <article><strong>Success</strong><span>no points</span><p>Selected by default. Watch the video and press Enter to save and go on; a video counts once saved.</p></article>
        <article><strong>Progress stall</strong><span>a, r</span><p><b>a</b>: the stall begins; <b>r</b>: progress resumes.</p></article>
        <article><strong>Regression and recovery</strong><span>a, r</span><p><b>a</b>: the last state before the loss; <b>r</b>: the first state where it is restored, not the lowest point and not where motion restarts.</p></article>
        <article><strong>Multiple events</strong><span>a, r each</span><p>Two or more progress stalls or regressions and recoveries, in time order.</p></article>
        <article><strong>No further progress</strong><span>a</span><p>Failed video: move to the last state with real progress and click the event; the current frame becomes a.</p></article>
        <article><strong>Unrecovered regression</strong><span>a, optional l</span><p>Failed video: a is the last state before the loss; press L where the loss becomes visible.</p></article>
        <article><strong>Uncertain</strong><span>no points</span><p>The event or a boundary cannot be told reliably; kept out of training.</p></article>
      </div>
      <section><h3>Keys</h3><dl class="shortcut-list">
        <div><dt>Space</dt><dd>Play / pause</dd></div>
        <div><dt>← / →</dt><dd>One frame</dd></div>
        <div><dt>Shift + ← / →</dt><dd>Five frames</dd></div>
        <div><dt>J / K</dt><dd>Previous / next video</dd></div>
        <div><dt>1–5</dt><dd>Choose an event; the current frame becomes its a</dd></div>
        <div><dt>A / R</dt><dd>Set a / r to the current frame</dd></div>
        <div><dt>L</dt><dd>Loss visible l (unrecovered regression)</dd></div>
        <div><dt>S</dt><dd>Last frame used (the completion frame of a success)</dd></div>
        <div><dt>Enter</dt><dd>Save and go to the next video</dd></div>
      </dl></section>
      <section class="warning-note"><strong>Note</strong>
        <p>Annotations are saved in the project's annotations.json. Export downloads them as JSONL for event2reward convert annotations. Waiting after the task is done is not a stall; judge only what the recording shows.</p></section>`,
  },
};
