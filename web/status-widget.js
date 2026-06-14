// Live status widget: fixed bottom-right, polls /api/system-status every
// 2.5s. Shows render queue (in-progress + pending count), CPU%, mem%,
// disk free. Click to expand and see currently-rendering stem(s).
(function() {
  const css = `
    #sw-root {
      position: fixed; right: 12px; bottom: 12px; z-index: 100;
      background: rgba(22,27,34,0.92); color: #e6edf3;
      border: 1px solid #2a313c; border-radius: 8px;
      font: 11px/1.4 ui-monospace, Menlo, monospace;
      padding: 8px 12px; min-width: 200px;
      box-shadow: 0 4px 16px rgba(0,0,0,0.4);
      backdrop-filter: blur(6px);
      cursor: pointer;
      transition: transform 0.1s;
    }
    #sw-root:hover { transform: translateY(-1px); border-color: #58a6ff; }
    #sw-root .sw-row { display: flex; justify-content: space-between; gap: 12px; }
    #sw-root .sw-label { color: #8b949e; }
    #sw-root .sw-value { color: #e6edf3; font-weight: 600; }
    #sw-root .sw-bar {
      height: 3px; background: #0e1116; border-radius: 2px; overflow: hidden;
      margin-top: 2px;
    }
    #sw-root .sw-bar > span {
      display: block; height: 100%; background: #58a6ff;
      transition: width 0.3s, background 0.3s;
    }
    #sw-root .sw-bar.warn > span { background: #d29922; }
    #sw-root .sw-bar.over > span { background: #d1242f; }
    #sw-root.expanded .sw-detail { display: block; }
    #sw-root .sw-detail {
      display: none; margin-top: 8px; padding-top: 8px;
      border-top: 1px solid #2a313c; max-height: 200px; overflow-y: auto;
    }
    #sw-root .sw-detail .row { display: flex; justify-content: space-between; gap: 8px; padding: 1px 0; }
    #sw-root .sw-detail .stem { color: #58a6ff; }
    #sw-root .sw-spin {
      display: inline-block; width: 8px; height: 8px;
      border-radius: 50%; background: #2ea043;
      animation: sw-pulse 1.2s ease-in-out infinite;
      margin-right: 6px;
    }
    @keyframes sw-pulse {
      0%, 100% { transform: scale(0.6); opacity: 0.4; }
      50% { transform: scale(1); opacity: 1; }
    }
  `;
  const style = document.createElement("style");
  style.textContent = css;
  document.head.appendChild(style);

  const root = document.createElement("div");
  root.id = "sw-root";
  root.innerHTML = `
    <div class="sw-row">
      <span class="sw-label" id="sw-render-label">render</span>
      <span class="sw-value" id="sw-render-value">—</span>
    </div>
    <div class="sw-row" id="sw-eta-row" style="display:none;">
      <span class="sw-label">eta</span>
      <span class="sw-value" id="sw-eta">—</span>
    </div>
    <div class="sw-row">
      <span class="sw-label">cpu</span>
      <span class="sw-value" id="sw-cpu">—</span>
    </div>
    <div class="sw-bar" id="sw-cpu-bar"><span style="width:0"></span></div>
    <div class="sw-row" style="margin-top:4px;">
      <span class="sw-label">mem</span>
      <span class="sw-value" id="sw-mem">—</span>
    </div>
    <div class="sw-bar" id="sw-mem-bar"><span style="width:0"></span></div>
    <div class="sw-row" style="margin-top:4px;">
      <span class="sw-label">disk free</span>
      <span class="sw-value" id="sw-disk">—</span>
    </div>
    <div class="sw-detail" id="sw-detail"></div>
  `;
  document.body.appendChild(root);

  root.onclick = () => root.classList.toggle("expanded");

  function setBar(barId, percent) {
    const bar = document.getElementById(barId);
    bar.firstElementChild.style.width = `${Math.min(100, percent)}%`;
    bar.classList.toggle("warn", percent >= 75 && percent < 90);
    bar.classList.toggle("over", percent >= 90);
  }

  function fmtDuration(s) {
    if (s == null || !isFinite(s) || s < 0) return "—";
    if (s < 60) return `${Math.round(s)}s`;
    const m = Math.floor(s / 60);
    const sec = Math.round(s % 60);
    if (m < 60) return sec ? `${m}m ${sec}s` : `${m}m`;
    const h = Math.floor(m / 60);
    const mm = m % 60;
    return mm ? `${h}h ${mm}m` : `${h}h`;
  }

  async function tick() {
    try {
      const s = await fetch("/api/system-status").then(r => r.json());
      const renderLbl = document.getElementById("sw-render-label");
      const renderVal = document.getElementById("sw-render-value");
      if (s.rendering.length) {
        renderLbl.innerHTML = '<span class="sw-spin"></span>rendering';
        renderVal.textContent = `${s.pending_count} queued`;
      } else if (s.pending_count) {
        renderLbl.textContent = "queued";
        renderVal.textContent = `${s.pending_count}`;
      } else {
        renderLbl.textContent = "render";
        renderVal.textContent = "idle";
      }

      const etaRow = document.getElementById("sw-eta-row");
      const etaVal = document.getElementById("sw-eta");
      const busy = s.rendering.length || s.pending_count;
      if (busy && s.eta_s != null) {
        etaRow.style.display = "";
        const avg = s.avg_render_s != null ? `~${fmtDuration(s.avg_render_s)}/clip` : "";
        etaVal.textContent = avg ? `${fmtDuration(s.eta_s)} (${avg})` : fmtDuration(s.eta_s);
      } else if (busy && s.samples === 0) {
        etaRow.style.display = "";
        etaVal.textContent = "calibrating…";
      } else {
        etaRow.style.display = "none";
      }

      document.getElementById("sw-cpu").textContent = `${s.cpu_percent.toFixed(0)}%`;
      setBar("sw-cpu-bar", s.cpu_percent);
      document.getElementById("sw-mem").textContent = `${s.mem_percent.toFixed(0)}%`;
      setBar("sw-mem-bar", s.mem_percent);
      const free = s.disk_free_gb;
      document.getElementById("sw-disk").textContent =
        free == null ? "—" : `${free.toFixed(1)} GB`;

      // Detail panel: show currently-rendering stems and the next few queued.
      const detail = document.getElementById("sw-detail");
      const lines = [];
      for (const stem of s.rendering) {
        lines.push(`<div class="row"><span class="stem">▶ ${stem}</span><span>rendering</span></div>`);
      }
      for (const stem of s.pending.slice(0, 8)) {
        lines.push(`<div class="row"><span class="stem">${stem}</span><span style="color:#8b949e;">pending</span></div>`);
      }
      if (s.pending.length > 8) {
        lines.push(`<div class="row"><span style="color:#8b949e;">+${s.pending.length - 8} more</span></div>`);
      }
      if (s.errored.length) {
        for (const stem of s.errored) {
          lines.push(`<div class="row"><span class="stem">${stem}</span><span style="color:#d1242f;">error</span></div>`);
        }
      }
      if (s.done_count) {
        lines.push(`<div class="row"><span style="color:#8b949e;">${s.done_count} done this session</span></div>`);
      }
      detail.innerHTML = lines.join("") || `<div class="row"><span style="color:#8b949e;">queue empty</span></div>`;
    } catch {}
  }
  tick();
  setInterval(tick, 2500);
})();
