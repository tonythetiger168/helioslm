/* hsh web UI client. Plain browser JavaScript, no dependencies. */
"use strict";

const $ = (id) => document.getElementById(id);
const S = {
  app: null,          // /api/state
  sid: null,          // open session id
  events: [],         // its log
  seen: new Set(),    // event seqs already held
  status: "idle",
  live: { reasoning: "", text: "", start: 0 },
  questions: [],      // open question / approval frames
  meta: null,
  permission: "auto",
  es: null,
  open: new Set(),    // expanded tool cards / thinking blocks
  draft: { preset: null, provider: null, model: null, permission: "auto", plan: false },
  skills: [],
  slashSel: 0,
  settingsTab: "general",
};

const ICON = {
  chev: '<svg viewBox="0 0 20 20"><path d="M8 5l5 5-5 5"/></svg>',
  caret: '<svg class="caret" viewBox="0 0 20 20"><path d="M6 8l4 4 4-4"/></svg>',
  check: '<svg class="check" viewBox="0 0 20 20"><path d="M5 10.5l3.2 3L15 6.5"/></svg>',
  folder: '<svg viewBox="0 0 20 20"><path d="M3 6.5A1.5 1.5 0 0 1 4.5 5h3.2l1.6 1.6h6.2A1.5 1.5 0 0 1 17 8.1v6.4A1.5 1.5 0 0 1 15.5 16h-11A1.5 1.5 0 0 1 3 14.5z"/></svg>',
  shield: '<svg viewBox="0 0 20 20"><path d="M10 3l6 2.2v4.3c0 3.6-2.5 6.3-6 7.5-3.5-1.2-6-3.9-6-7.5V5.2z"/><path d="M7.5 10l1.8 1.8L12.8 8.3"/></svg>',
  lock: '<svg viewBox="0 0 20 20"><rect x="5" y="9" width="10" height="8" rx="2"/><path d="M7 9V7a3 3 0 0 1 6 0v2"/></svg>',
  hand: '<svg viewBox="0 0 20 20"><path d="M7 10V5.5a1.2 1.2 0 0 1 2.4 0V9m0-4.5V4.2a1.2 1.2 0 0 1 2.4 0V9m0-3.3a1.2 1.2 0 0 1 2.4 0V12a5 5 0 0 1-5 5h-.4a5 5 0 0 1-4-2l-2-2.8a1.2 1.2 0 0 1 1.9-1.4L7 12"/></svg>',
  layers: '<svg viewBox="0 0 20 20"><path d="M10 3l7 3.8-7 3.8-7-3.8z"/><path d="M3 10.2l7 3.8 7-3.8M3 13.6l7 3.8 7-3.8"/></svg>',
  plan: '<svg viewBox="0 0 20 20"><path d="M6 4h8a1.5 1.5 0 0 1 1.5 1.5v10A1.5 1.5 0 0 1 14 17H6a1.5 1.5 0 0 1-1.5-1.5v-10A1.5 1.5 0 0 1 6 4z"/><path d="M7.5 8h5M7.5 11h5M7.5 14h3"/></svg>',
  spark: '<svg viewBox="0 0 20 20"><path d="M10 3c.5 3.6 2.2 5.4 7 7-4.8 1.6-6.5 3.4-7 7-.5-3.6-2.2-5.4-7-7 4.8-1.6 6.5-3.4 7-7z"/></svg>',
  term: '<svg viewBox="0 0 20 20"><rect x="3" y="4" width="14" height="12" rx="2"/><path d="M6.5 8l2.5 2-2.5 2M10.5 12.5h3"/></svg>',
  file: '<svg viewBox="0 0 20 20"><path d="M6 3h5l4 4v9a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1z"/><path d="M11 3v4h4"/></svg>',
  pen: '<svg viewBox="0 0 20 20"><path d="M4 16l.8-3.2L13 4.6a1.4 1.4 0 0 1 2 0l.4.4a1.4 1.4 0 0 1 0 2L7.2 15.2z"/></svg>',
  search: '<svg viewBox="0 0 20 20"><circle cx="9" cy="9" r="5"/><path d="M13 13l3.5 3.5"/></svg>',
  globe: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="M3 10h14M10 3c2 2 3 4.3 3 7s-1 5-3 7c-2-2-3-4.3-3-7s1-5 3-7z"/></svg>',
  bot: '<svg viewBox="0 0 20 20"><rect x="4" y="6" width="12" height="10" rx="3"/><path d="M10 3v3M7.5 11h.01M12.5 11h.01"/></svg>',
  plug: '<svg viewBox="0 0 20 20"><path d="M7 3v4M13 3v4M5 7h10v3a5 5 0 0 1-10 0zM10 15v2"/></svg>',
  list: '<svg viewBox="0 0 20 20"><path d="M8 6h8M8 10h8M8 14h8M4.5 6h.01M4.5 10h.01M4.5 14h.01"/></svg>',
  question: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="M8 8a2 2 0 1 1 2.8 1.8c-.5.3-.8.7-.8 1.2v.5M10 14h.01"/></svg>',
  book: '<svg viewBox="0 0 20 20"><path d="M4 4.5A1.5 1.5 0 0 1 5.5 3H16v12H5.5A1.5 1.5 0 0 0 4 16.5zM4 16.5A1.5 1.5 0 0 0 5.5 18H16"/></svg>',
  ok: '<svg viewBox="0 0 20 20"><path d="M5 10.5l3.2 3L15 6.5"/></svg>',
  bad: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="7"/><path d="M10 6.5v4.5M10 13.5h.01"/></svg>',
  spin: '<svg class="spin" viewBox="0 0 20 20"><path d="M10 3a7 7 0 1 1-7 7"/></svg>',
  gear: '<svg viewBox="0 0 20 20"><circle cx="10" cy="10" r="2.6"/><path d="M10 2.8v2M10 15.2v2M2.8 10h2M15.2 10h2M4.9 4.9l1.4 1.4M13.7 13.7l1.4 1.4M4.9 15.1l1.4-1.4M13.7 6.3l1.4-1.4"/></svg>',
  db: '<svg viewBox="0 0 20 20"><ellipse cx="10" cy="5.5" rx="6" ry="2.5"/><path d="M4 5.5v9c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5v-9M4 10c0 1.4 2.7 2.5 6 2.5s6-1.1 6-2.5"/></svg>',
};

// ----- helpers ------------------------------------------------------------
function esc(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function h(tag, attrs, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const kid of kids.flat()) {
    if (kid == null || kid === false) continue;
    el.append(kid instanceof Node ? kid : document.createTextNode(String(kid)));
  }
  return el;
}
function ago(t) {
  if (!t) return "";
  const s = Math.max(0, Date.now() / 1000 - t);
  if (s < 60) return "now";
  if (s < 3600) return Math.floor(s / 60) + "min";
  if (s < 86400) return Math.floor(s / 3600) + "h";
  return Math.floor(s / 86400) + "d";
}
async function api(path, body) {
  const opt = body === undefined ? {} : {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
  };
  const r = await fetch(path, opt);
  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.error || r.statusText);
  return data;
}
let toastTimer = null;
function toast(msg) {
  const t = $("toast");
  t.textContent = msg;
  t.hidden = false;
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => { t.hidden = true; }, 3800);
}

// ----- markdown (escape first, then a small safe subset) -----------------
function inline(s) {
  const codes = [];
  s = s.replace(/`([^`\n]+)`/g, (_, c) => { codes.push(c); return "\u0000" + (codes.length - 1) + "\u0000"; });
  s = esc(s);
  s = s.replace(/\[([^\]\n]+)\]\((https?:\/\/[^\s)]+)\)/g, (_, t, u) =>
    `<a href="${u}" target="_blank" rel="noopener noreferrer">${t}</a>`);
  s = s.replace(/(^|[\s(])(https?:\/\/[^\s<)]+)/g, (_, p, u) =>
    `${p}<a href="${u}" target="_blank" rel="noopener noreferrer">${u}</a>`);
  s = s.replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\n]+)\*(?!\*)/g, "$1<em>$2</em>");
  s = s.replace(/~~([^~\n]+)~~/g, "<del>$1</del>");
  return s.replace(/\u0000(\d+)\u0000/g, (_, i) => `<code>${esc(codes[+i])}</code>`);
}
function md(src) {
  const lines = String(src ?? "").replace(/\r\n/g, "\n").split("\n");
  const out = [];
  let i = 0;
  const para = [];
  const flush = () => { if (para.length) { out.push(`<p>${inline(para.join("\n")).replace(/\n/g, "<br>")}</p>`); para.length = 0; } };
  while (i < lines.length) {
    const line = lines[i];
    const fence = line.match(/^\s*(```|~~~)\s*([\w+-]*)/);
    if (fence) {
      flush();
      const body = [];
      i++;
      while (i < lines.length && !lines[i].trim().startsWith(fence[1])) body.push(lines[i++]);
      i++;
      out.push(`<pre>${fence[2] ? `<span class="lang">${esc(fence[2])}</span>` : ""}<code>${esc(body.join("\n"))}</code></pre>`);
      continue;
    }
    const hd = line.match(/^(#{1,4})\s+(.*)$/);
    if (hd) { flush(); out.push(`<h${hd[1].length}>${inline(hd[2])}</h${hd[1].length}>`); i++; continue; }
    if (/^\s*([-*_])(\s*\1){2,}\s*$/.test(line)) { flush(); out.push("<hr>"); i++; continue; }
    if (/^\s*>/.test(line)) {
      flush();
      const q = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) q.push(lines[i++].replace(/^\s*>\s?/, ""));
      out.push(`<blockquote>${md(q.join("\n"))}</blockquote>`);
      continue;
    }
    if (/^\s*\|.*\|\s*$/.test(line) && i + 1 < lines.length && /^\s*\|?\s*:?-{2,}/.test(lines[i + 1])) {
      flush();
      const cells = (l) => l.trim().replace(/^\||\|$/g, "").split("|").map((c) => inline(c.trim()));
      const head = cells(line);
      i += 2;
      const rows = [];
      while (i < lines.length && /^\s*\|.*\|\s*$/.test(lines[i])) rows.push(cells(lines[i++]));
      out.push(`<table><thead><tr>${head.map((c) => `<th>${c}</th>`).join("")}</tr></thead><tbody>${
        rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody></table>`);
      continue;
    }
    const li = line.match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
    if (li) {
      flush();
      const ordered = /\d/.test(li[2]);
      const items = [];
      while (i < lines.length) {
        const m = lines[i].match(/^(\s*)([-*+]|\d+[.)])\s+(.*)$/);
        if (m) { items.push(m[3]); i++; continue; }
        if (/^\s{2,}\S/.test(lines[i]) && items.length) { items[items.length - 1] += "\n" + lines[i].trim(); i++; continue; }
        break;
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((t) => {
        const box = t.match(/^\[([ xX])\]\s+(.*)$/s);
        return box ? `<li>${box[1] === " " ? "☐" : "☑"} ${inline(box[2])}</li>` : `<li>${inline(t)}</li>`;
      }).join("")}</${tag}>`);
      continue;
    }
    if (!line.trim()) { flush(); i++; continue; }
    para.push(line);
    i++;
  }
  flush();
  return out.join("");
}

// ----- theme ---------------------------------------------------------------
function pref(key, dflt) { try { return localStorage.getItem("hsh." + key) ?? dflt; } catch { return dflt; } }
function setPref(key, v) { try { localStorage.setItem("hsh." + key, v); } catch { /* private mode */ } }
function applyTheme() {
  const t = pref("theme", "system");
  const dark = t === "dark" || (t === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
}
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", applyTheme);
applyTheme();

// ----- state ---------------------------------------------------------------
async function loadState() {
  S.app = await api("/api/state");
  if (!S.draft.preset) S.draft.preset = S.app.default_preset;
  if (!S.draft.provider && S.app.default_model) {
    S.draft.provider = S.app.default_model.provider;
    S.draft.model = S.app.default_model.model;
  }
  renderSidebar();
  renderChips();
  $("ws-chip").innerHTML = ICON.folder + `<span>${esc(S.app.workspace_name || S.app.workspace)}</span>`;
}
async function loadSkills() {
  try { S.skills = await api("/api/skills"); } catch { S.skills = []; }
}

function renderSidebar() {
  const list = $("session-list");
  list.replaceChildren();
  list.append(h("div", { class: "group-head", title: S.app.workspace, html: ICON.folder + `<span>${esc(S.app.workspace_name)}</span>` }));
  if (!S.app.sessions.length) list.append(h("div", { class: "side-empty" }, "No sessions yet"));
  for (const s of S.app.sessions) {
    const item = h("button", { class: "s-item" + (s.id === S.sid ? " active" : ""), title: s.title, onclick: () => go(s.id) },
      s.status === "running" ? h("span", { class: "dot" }) : null,
      h("span", { class: "s-title" }, s.title || "New session"),
      h("span", { class: "s-time" }, ago(s.updated)));
    list.append(item);
  }
}

// ----- routing -------------------------------------------------------------
function go(sid, push = true) {
  if (push) history.pushState({}, "", sid ? "/s/" + sid : "/");
  openSession(sid);
  document.querySelector(".app").classList.remove("mobile-open");
}
window.addEventListener("popstate", () => openSession(sidFromPath()));
function sidFromPath() { const m = location.pathname.match(/^\/s\/([A-Za-z0-9_-]+)/); return m ? m[1] : null; }

async function openSession(sid) {
  if (S.es) { S.es.close(); S.es = null; }
  S.sid = sid;
  S.events = []; S.seen = new Set(); S.questions = []; S.live = { reasoning: "", text: "", start: 0 };
  S.status = "idle"; S.meta = null; S.plan = null;
  if (!sid) { render(); renderSidebar(); renderChips(); $("input").focus(); return; }
  let snap;
  try { snap = await api(`/api/sessions/${sid}`); } catch (e) { toast(e.message); go(null); return; }
  if (S.sid !== sid) return;
  S.meta = snap.meta; S.permission = snap.permission; S.status = snap.status; S.plan = snap.plan;
  for (const ev of snap.events) addEvent(ev);
  S.questions = snap.questions || [];
  render(true);
  renderSidebar();
  renderChips();
  refreshList();
  follow(sid, snap.cursor);
}

function follow(sid, cursor) {
  const es = new EventSource(`/api/sessions/${sid}/stream?cursor=${cursor}`);
  S.es = es;
  es.onmessage = (m) => { if (S.sid === sid) onFrame(JSON.parse(m.data)); };
  es.onerror = () => { /* the browser reconnects with Last-Event-ID */ };
}

function addEvent(ev) {
  if (S.seen.has(ev.seq)) return false;
  S.seen.add(ev.seq);
  S.events.push(ev);
  return true;
}

let rafPending = false;
function schedule(stick) {
  if (rafPending) return;
  rafPending = true;
  requestAnimationFrame(() => { rafPending = false; render(stick); });
}

function onFrame(f) {
  const sc = $("scroll");
  const nearBottom = sc.scrollHeight - sc.scrollTop - sc.clientHeight < 140;
  switch (f.kind) {
    case "event":
      if (!addEvent(f.event)) return;
      if (f.event.type === "assistant/message" || f.event.type === "assistant/attempt") S.live = { reasoning: "", text: "", start: 0 };
      if (f.event.type === "step/start") S.live = { reasoning: "", text: "", start: Date.now() };
      if (f.event.type === "user/message" && f.event.data.source === "human") refreshList();
      if (f.event.type === "plan/mode" || f.event.type === "turn/start") S.plan = null;
      break;
    case "chunk":
      if (!S.live.start) S.live.start = Date.now();
      if (f.chunk.type === "reasoning") S.live.reasoning += f.chunk.text; else S.live.text += f.chunk.text;
      break;
    case "status":
      S.status = f.status;
      if (f.status === "idle") { S.live = { reasoning: "", text: "", start: 0 }; refreshList(); }
      renderChips();
      break;
    case "question": case "approval":
      if (!S.questions.some((q) => q.qid === f.qid)) S.questions.push(f);
      renderPending();
      break;
    case "question-closed":
      S.questions = S.questions.filter((q) => q.qid !== f.qid);
      renderPending();
      break;
    case "meta":
      S.meta = f.meta; S.permission = f.permission; S.plan = f.plan; renderChips(); refreshList();
      break;
    case "error":
      toast(f.message);
      break;
  }
  schedule(nearBottom);
}

let listTimer = null;
function refreshList() {
  clearTimeout(listTimer);
  listTimer = setTimeout(async () => {
    try { const st = await api("/api/state"); S.app.sessions = st.sessions; renderSidebar(); } catch { /* offline */ }
  }, 250);
}

// ----- conversation rendering ---------------------------------------------
const TOOL_ICON = { bash: ICON.term, read: ICON.file, write: ICON.pen, edit: ICON.pen, str_replace_editor: ICON.pen,
  glob: ICON.search, grep: ICON.search, web_fetch: ICON.globe, subagent: ICON.bot, todo_write: ICON.list,
  skill: ICON.book, ask_user_question: ICON.question, exit_plan_mode: ICON.plan, run_code: ICON.term,
  job_output: ICON.term, job_list: ICON.term, job_kill: ICON.term, list_mcp_resources: ICON.plug, read_mcp_resource: ICON.plug };

function toolSummary(name, args) {
  if (!args || typeof args !== "object") return "";
  if (name === "bash") return args.description || args.command || "";
  if (["read", "write", "edit"].includes(name)) return args.path || "";
  if (name === "str_replace_editor") return `${args.command || ""} ${args.path || ""}`;
  if (name === "glob" || name === "grep") return args.pattern + (args.path ? `  in ${args.path}` : "");
  if (name === "skill") return args.name;
  if (name === "web_fetch") return args.url;
  if (name === "subagent") return args.task;
  if (name === "exit_plan_mode") return (args.plan || "").split("\n")[0].replace(/^#+\s*/, "");
  if (name === "ask_user_question") return (args.questions || []).map((q) => q.question).join(" · ");
  if (name.startsWith("job_")) return args.job_id || "";
  const first = Object.values(args).find((v) => typeof v === "string");
  return first || "";
}
function prettyName(name) {
  const m = name.match(/^mcp__([^_]+(?:_[^_]+)*)__(.+)$/);
  return m ? `${m[1]} · ${m[2]}` : name;
}

function render(stick) {
  const main = $("main");
  const conv = $("conversation");
  const empty = !S.sid;
  main.classList.toggle("empty", empty);
  $("conv-head").hidden = empty;
  if (!empty) {
    $("conv-title").textContent = (S.meta && S.meta.title) || "New session";
    const tot = usageTotal();
    $("conv-meta").textContent = tot ? `${tot} tokens` : "";
  }
  conv.replaceChildren();
  if (!empty) buildConversation(conv);
  renderPending();
  renderChips();
  if (stick) { const sc = $("scroll"); sc.scrollTop = sc.scrollHeight; }
}

function usageTotal() {
  let n = 0;
  for (const ev of S.events) if (ev.type === "assistant/message" || ev.type === "assistant/attempt") {
    const u = ev.data.usage || {}; n += (u.input_tokens || 0) + (u.output_tokens || 0);
  }
  return n ? n.toLocaleString() : "";
}

function thinkBlock(key, text, seconds, live) {
  const open = S.open.has(key) || live;
  const label = live ? "Thinking" : `Thought for ${seconds != null ? Math.max(1, Math.round(seconds)) + "s" : "a moment"}`;
  const el = h("div", { class: "think" + (open ? " open" : "") + (live ? " live" : "") },
    h("button", { class: "think-head", onclick: () => { toggle(key); } , html: `<span class="${live ? "shimmer" : ""}">${label}</span>` + ICON.chev }),
    h("div", { class: "think-body" }, text));
  return el;
}
function toggle(key) { if (S.open.has(key)) S.open.delete(key); else S.open.add(key); schedule(false); }

function toolCard(call, result) {
  let args = {};
  try { args = JSON.parse(call.arguments || "{}"); } catch { args = { _raw: call.arguments }; }
  const key = "tool:" + call.id;
  const state = result ? (result.message.is_error ? "err" : "done") : "run";
  if (call.name === "todo_write" && result && !result.message.is_error) {
    const items = (args.todos || []).map((t) => h("div", { class: "todo-item " + t.status }, h("span", { class: "box" }), t.content));
    return h("div", { class: "todo" }, items);
  }
  const head = h("button", { class: "tool-head", onclick: () => toggle(key) });
  head.innerHTML = (TOOL_ICON[call.name] || (call.name.startsWith("mcp__") ? ICON.plug : ICON.spark)).replace("<svg", '<svg class="t-icon"') +
    `<span class="t-name">${esc(prettyName(call.name))}</span><span class="t-sum">${esc(toolSummary(call.name, args))}</span>` +
    (state === "run" ? ICON.spin : state === "err" ? ICON.bad : ICON.ok).replace("<svg", '<svg class="t-state' + (state === "run" ? " spin" : "") + '"') +
    ICON.chev.replace("<svg", '<svg class="t-chev"');
  const body = h("div", { class: "tool-body" },
    h("div", { class: "tool-sec" }, h("h5", {}, "Input"), h("pre", {}, call.name === "bash" && args.command ? args.command : JSON.stringify(args, null, 2))),
    result ? h("div", { class: "tool-sec" }, h("h5", {}, result.message.is_error ? "Error" : "Output"),
      h("pre", { class: "res" }, result.message.content || "(empty)")) : null);
  return h("div", { class: `tool ${state}` + (S.open.has(key) ? " open" : "") }, head, body);
}

function buildConversation(conv) {
  const results = {};
  for (const ev of S.events) if (ev.type === "tool/result") results[ev.data.message.call_id] = ev.data;
  let stepStart = null;
  let lastTurnUsage = null;
  for (const ev of S.events) {
    const d = ev.data;
    switch (ev.type) {
      case "step/start": stepStart = ev.time || null; break;
      case "user/message": {
        if (d.source === "human") {
          let text = d.content;
          let skill = null;
          const m = text.match(/^<skill_content name="([^"]+)">[\s\S]*?<\/skill_content>\s*([\s\S]*)$/);
          if (m) { skill = m[1]; text = m[2]; }
          conv.append(h("div", { class: "u-row" }, h("div", { class: "u-bubble" },
            skill ? h("span", { class: "u-skill" }, "/" + skill) : null, text)));
        } else if (d.source === "compaction") {
          conv.append(note("Earlier conversation summarized to save context"));
        } else if (d.source === "turn-stopping") {
          conv.append(note("Continuing: " + d.content.slice(0, 120)));
        }
        break;
      }
      case "assistant/message": {
        const msg = d.message;
        const secs = stepStart && ev.time ? ev.time - stepStart : null;
        const block = h("div", { class: "a-block" });
        if (msg.reasoning) block.append(thinkBlock("think:" + ev.seq, msg.reasoning, secs, false));
        if (msg.content) block.append(h("div", { class: "md", html: md(msg.content) }));
        for (const c of msg.tool_calls || []) block.append(toolCard(c, results[c.id]));
        conv.append(block);
        lastTurnUsage = d.usage;
        break;
      }
      case "assistant/attempt":
        if (d.error) conv.append(note("Model request failed: " + d.error.slice(0, 200), "err"));
        break;
      case "plan/mode":
        conv.append(note(d.active ? "Plan mode on: the agent will plan before changing anything" : "Plan mode off", "plan"));
        break;
      case "turn/end":
        if (d.reason === "cancelled") conv.append(note("Stopped"));
        else if (d.reason === "error") conv.append(note("The turn ended with an error", "err"));
        else if (d.reason === "max-steps") conv.append(note("Stopped after the step limit"));
        else if (d.reason === "interrupted") conv.append(note("Interrupted (the server stopped)"));
        if (lastTurnUsage && (lastTurnUsage.input_tokens || lastTurnUsage.output_tokens)) {
          const u = lastTurnUsage;
          const cached = u.cached_input_tokens ? ` · ${u.cached_input_tokens} cached` : "";
          conv.append(h("div", { class: "usage" }, `${u.input_tokens || 0} in · ${u.output_tokens || 0} out${cached}`));
          lastTurnUsage = null;
        }
        break;
      case "context/shadow":
        break;
    }
  }
  if (S.status === "running") {
    const live = h("div", { class: "a-block" });
    if (S.live.reasoning) live.append(thinkBlock("live", S.live.reasoning, null, !S.live.text));
    if (S.live.text) live.append(h("div", { class: "md cursor", html: md(S.live.text) }));
    if (!S.live.reasoning && !S.live.text && !S.questions.length) {
      const pendingTool = S.events.some((e) => e.type === "tool/call" && !results[e.data.call_id]);
      if (!pendingTool) live.append(h("div", { class: "waiting", html: '<span class="shimmer">Working…</span>' }));
    }
    conv.append(live);
  }
}
function note(text, cls) { return h("div", { class: "note" + (cls ? " " + cls : "") }, h("span", {}, text)); }

// ----- questions and approvals -------------------------------------------
function renderPending() {
  const box = $("pending");
  box.replaceChildren();
  for (const q of S.questions) box.append(q.kind === "approval" ? approvalCard(q) : questionCard(q));
}

function approvalCard(q) {
  const args = q.arguments || {};
  const preview = q.tool === "bash" ? args.command : JSON.stringify(args, null, 2);
  return h("div", { class: "qcard" },
    h("div", { class: "q-head" }, "Permission"),
    h("div", { class: "q-text" }, `Allow the agent to run `, h("code", {}, prettyName(q.tool)), "?"),
    h("pre", {}, preview || ""),
    h("div", { class: "q-actions" },
      h("button", { class: "btn danger", onclick: () => answer(q.qid, "deny") }, "Deny"),
      h("button", { class: "btn primary", onclick: () => answer(q.qid, "allow") }, "Allow")));
}

function questionCard(q) {
  const isPlan = q.tool === "exit_plan_mode";
  const picks = {};
  const others = {};
  const card = h("div", { class: "qcard" });
  for (const item of q.questions) {
    picks[item.id] = new Set();
    card.append(h("div", { class: "q-head" }, item.header || (isPlan ? "Plan" : "Question")));
    card.append(isPlan ? h("div", { class: "q-text md", html: md(item.question) }) : h("div", { class: "q-text" }, item.question));
    if (item.options && item.options.length && !isPlan) {
      const opts = h("div", { class: "q-opts" });
      for (const o of item.options) {
        const b = h("button", { class: "q-opt", onclick: () => {
          if (!item.multi_select) { picks[item.id].clear(); opts.querySelectorAll(".q-opt").forEach((x) => x.classList.remove("sel")); }
          if (picks[item.id].has(o.label)) { picks[item.id].delete(o.label); b.classList.remove("sel"); }
          else { picks[item.id].add(o.label); b.classList.add("sel"); }
        } }, o.label, o.description ? h("small", {}, o.description) : null);
        opts.append(b);
      }
      card.append(opts);
    }
    if (!isPlan) {
      others[item.id] = h("textarea", { class: "q-other", rows: 1, placeholder: item.options && item.options.length ? "Or type another answer" : "Your answer" });
      card.append(others[item.id]);
    }
  }
  if (isPlan) {
    const fb = h("textarea", { class: "q-other", rows: 2, placeholder: "Feedback for the agent (optional)" });
    card.append(fb);
    card.append(h("div", { class: "q-actions" },
      h("button", { class: "btn", onclick: () => answer(q.qid, { plan: fb.value.trim() || "Keep planning" }) }, "Keep planning"),
      h("button", { class: "btn primary", onclick: () => answer(q.qid, { plan: "Approve" }) }, "Approve plan")));
    return card;
  }
  card.append(h("div", { class: "q-actions" },
    h("button", { class: "btn", onclick: () => answer(q.qid, null) }, "Skip"),
    h("button", { class: "btn primary", onclick: () => {
      const out = {};
      for (const item of q.questions) {
        const typed = others[item.id].value.trim();
        const chosen = [...picks[item.id]];
        if (typed) chosen.push(typed);
        out[item.id] = item.multi_select ? chosen : (chosen[0] ?? null);
      }
      answer(q.qid, out);
    } }, "Submit")));
  return card;
}

async function answer(qid, value) {
  try { await api(`/api/sessions/${S.sid}/answer`, { qid, answer: value }); }
  catch (e) { toast(e.message); }
  S.questions = S.questions.filter((q) => q.qid !== qid);
  renderPending();
}

// ----- composer chips --------------------------------------------------------
const PERM = {
  auto: { label: "Workspace write", icon: ICON.shield, desc: "Run commands and edit files in the workspace" },
  ask: { label: "Ask first", icon: ICON.hand, desc: "Approve each command, edit and MCP call" },
  "read-only": { label: "Read only", icon: ICON.lock, desc: "Refuse commands and edits" },
};

function currentModel() {
  if (S.meta) return { provider: S.meta.provider, model: S.meta.model };
  return { provider: S.draft.provider, model: S.draft.model };
}
function planOn() {
  if (!S.sid) return S.draft.plan;
  let on = false;
  for (const ev of S.events) if (ev.type === "plan/mode") on = ev.data.active;
  return S.plan && S.plan.pending != null ? S.plan.pending : on;
}

function renderChips() {
  if (!S.app) return;
  const preset = S.meta ? S.meta.preset : S.draft.preset;
  const pc = $("preset-chip");
  pc.innerHTML = ICON.layers + `<span class="lbl">${esc(preset)}</span>` + (S.sid ? "" : ICON.caret);
  pc.disabled = !!S.sid;
  pc.title = S.sid ? "Preset (fixed for this session)" : "Agent preset";
  const perm = S.sid ? S.permission : S.draft.permission;
  const p = PERM[perm] || PERM.auto;
  const permChip = $("perm-chip");
  permChip.innerHTML = p.icon + `<span class="lbl">${p.label}</span>` + ICON.caret;
  permChip.classList.toggle("warn", perm === "auto");
  const plan = planOn();
  const planChip = $("plan-chip");
  planChip.innerHTML = ICON.plan + `<span class="lbl">Plan</span>`;
  planChip.classList.toggle("on", !!plan);
  planChip.title = plan ? "Plan mode is on (click to turn off)" : "Plan mode: plan before changing anything";
  const m = currentModel();
  const info = (S.app.models || []).find((x) => x.provider === m.provider && x.model === m.model);
  $("model-chip").innerHTML = `<span>${esc(m.model || "No model")}</span>` +
    (info && !info.ready ? ' <span class="muted">key missing</span>' : "") + ICON.caret;
  const send = $("send");
  const running = S.status === "running";
  send.classList.toggle("stop", running);
  send.title = running ? "Stop" : "Send (Enter)";
  send.disabled = !running && !$("input").value.trim();
  $("dock-note").textContent = running ? "The agent is working. Messages you send now run after this turn." :
    (info && !info.ready ? `Set ${providerEnv(m.provider)} and restart the server to use ${m.model}.` : "");
}
function providerEnv(name) { const p = (S.app.providers || []).find((x) => x.name === name); return p ? p.api_key_env : "the API key"; }

function openMenu(anchor, items) {
  const menu = $("menu");
  menu.replaceChildren();
  for (const it of items) {
    if (it.label && !it.onclick) { menu.append(h("div", { class: "menu-label" }, it.label)); continue; }
    if (it.sep) { menu.append(h("div", { class: "menu-sep" })); continue; }
    const b = h("button", { class: "menu-item" + (it.sel ? " sel" : "") + (it.danger ? " danger" : ""), disabled: it.disabled,
      onclick: () => { closeMenu(); it.onclick(); } });
    b.innerHTML = ICON.check + `<span>${esc(it.text)}</span>` + (it.hint ? `<small>${esc(it.hint)}</small>` : "");
    menu.append(b);
  }
  menu.hidden = false;
  const r = anchor.getBoundingClientRect();
  const mh = menu.offsetHeight, mw = menu.offsetWidth;
  let top = r.top - mh - 6;
  if (top < 8) top = r.bottom + 6;
  let left = Math.min(r.left, innerWidth - mw - 8);
  menu.style.top = top + "px";
  menu.style.left = Math.max(8, left) + "px";
}
function closeMenu() { $("menu").hidden = true; }
document.addEventListener("mousedown", (e) => { if (!$("menu").contains(e.target)) closeMenu(); });

$("preset-chip").onclick = (e) => {
  if (S.sid) return;
  openMenu(e.currentTarget, [{ label: "Agent preset" }, ...S.app.presets.map((p) => ({
    text: p.id, hint: { minimal: "bash + editor", standard: "full toolset", code: "programs call tools", creator: "+ inspect" }[p.id] || "",
    sel: p.id === S.draft.preset, onclick: () => { S.draft.preset = p.id; renderChips(); } }))]);
};
$("perm-chip").onclick = (e) => {
  const cur = S.sid ? S.permission : S.draft.permission;
  openMenu(e.currentTarget, [{ label: "Permissions" }, ...Object.entries(PERM).map(([k, v]) => ({
    text: v.label, hint: "", sel: k === cur, onclick: async () => {
      if (!S.sid) { S.draft.permission = k; renderChips(); return; }
      try { await api(`/api/sessions/${S.sid}/settings`, { permission: k }); S.permission = k; renderChips(); } catch (err) { toast(err.message); }
    } }))]);
};
$("plan-chip").onclick = async () => {
  const on = !planOn();
  if (!S.sid) { S.draft.plan = on; renderChips(); return; }
  try {
    await api(`/api/sessions/${S.sid}/settings`, { plan: on });
    S.plan = { active: !on, pending: on };
    renderChips();
    toast(on ? "Plan mode turns on with your next message" : "Plan mode turns off with your next message");
  } catch (err) { toast(err.message); }
};
$("model-chip").onclick = (e) => {
  const cur = currentModel();
  const items = [];
  let lastProv = null;
  for (const m of S.app.models) {
    if (m.provider !== lastProv) { items.push({ label: m.label }); lastProv = m.provider; }
    items.push({ text: m.model, hint: m.ready ? "" : "key missing", sel: m.provider === cur.provider && m.model === cur.model,
      onclick: async () => {
        if (!S.sid) { S.draft.provider = m.provider; S.draft.model = m.model; renderChips(); return; }
        try { await api(`/api/sessions/${S.sid}/settings`, { provider: m.provider, model: m.model }); } catch (err) { toast(err.message); }
      } });
  }
  if (!items.length) items.push({ label: "No model configured" });
  openMenu(e.currentTarget, items);
};
$("conv-menu").onclick = (e) => {
  openMenu(e.currentTarget, [
    { text: "Rename", onclick: async () => {
      const t = prompt("Session title", (S.meta && S.meta.title) || "");
      if (t != null) { try { await api(`/api/sessions/${S.sid}/settings`, { title: t }); } catch (err) { toast(err.message); } }
    } },
    { sep: true },
    { text: "Delete session", danger: true, onclick: async () => {
      if (!confirm("Delete this session and its log?")) return;
      try { await api(`/api/sessions/${S.sid}/delete`, {}); await loadState(); go(null); } catch (err) { toast(err.message); }
    } },
  ]);
};

// ----- sending ----------------------------------------------------------------
const input = $("input");
function autosize() { input.style.height = "auto"; input.style.height = Math.min(input.scrollHeight, 240) + "px"; }
input.addEventListener("input", () => { autosize(); renderChips(); updateSlash(); });
input.addEventListener("keydown", (e) => {
  const slash = $("slash");
  if (!slash.hidden) {
    const items = slash.querySelectorAll(".slash-item");
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      S.slashSel = (S.slashSel + (e.key === "ArrowDown" ? 1 : -1) + items.length) % items.length;
      items.forEach((x, i) => x.classList.toggle("sel", i === S.slashSel));
      return;
    }
    if ((e.key === "Enter" || e.key === "Tab") && items[S.slashSel]) { e.preventDefault(); items[S.slashSel].click(); return; }
    if (e.key === "Escape") { slash.hidden = true; return; }
  }
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); submit(); }
});

function slashItems() {
  const items = [
    { name: "plan", desc: "Turn plan mode on, then send the rest" },
    { name: "plan off", desc: "Turn plan mode off" },
  ];
  for (const s of S.skills) if (s.user) items.push({ name: s.name, desc: s.description });
  return items;
}
function updateSlash() {
  const slash = $("slash");
  const m = input.value.match(/^\/([\w-]*)$/);
  if (!m) { slash.hidden = true; return; }
  const q = m[1].toLowerCase();
  const items = slashItems().filter((it) => it.name.startsWith(q) || it.name.includes(q));
  if (!items.length) { slash.hidden = true; return; }
  S.slashSel = 0;
  slash.replaceChildren(...items.map((it, i) => h("button", { class: "slash-item" + (i === 0 ? " sel" : ""),
    onclick: () => { input.value = "/" + it.name + " "; slash.hidden = true; input.focus(); autosize(); renderChips(); } },
    h("b", {}, "/" + it.name), h("span", {}, it.desc))));
  slash.hidden = false;
}

async function submit() {
  if (S.status === "running" && !input.value.trim()) return;
  let text = input.value.trim();
  if (!text) return;
  input.value = ""; autosize(); $("slash").hidden = true;
  try {
    if (!S.sid) {
      if (S.draft.plan && !text.startsWith("/plan")) text = "/plan " + text;
      const r = await api("/api/sessions", { preset: S.draft.preset, provider: S.draft.provider, model: S.draft.model,
        permission: S.draft.permission, text });
      S.draft.plan = false;
      await loadState();
      go(r.id);
    } else {
      await api(`/api/sessions/${S.sid}/messages`, { text });
    }
  } catch (e) { toast(e.message); input.value = text; autosize(); }
  renderChips();
}
$("send").onclick = async () => {
  if (S.status === "running" && !input.value.trim()) {
    try { await api(`/api/sessions/${S.sid}/cancel`, {}); } catch (e) { toast(e.message); }
    return;
  }
  submit();
};

// ----- sidebar actions ------------------------------------------------------------
$("new-session").onclick = () => go(null);
$("brand").onclick = (e) => { e.preventDefault(); go(null); };
$("collapse").onclick = () => { document.querySelector(".app").classList.add("collapsed"); setPref("side", "closed"); };
$("reopen").onclick = () => {
  const app = document.querySelector(".app");
  if (innerWidth <= 760) app.classList.toggle("mobile-open");
  else { app.classList.remove("collapsed"); setPref("side", "open"); }
};
if (pref("side", "open") === "closed") document.querySelector(".app").classList.add("collapsed");
document.addEventListener("keydown", (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === "k") { e.preventDefault(); go(null); }
  if (e.key === "Escape") { closeMenu(); if (!$("settings").hidden) closeSettings(); }
});

// ----- settings -----------------------------------------------------------------
const TABS = [
  ["general", "General", ICON.gear], ["models", "Models", ICON.db], ["skills", "Skills", ICON.book],
  ["mcp", "MCP servers", ICON.plug], ["presets", "Agent presets", ICON.layers],
];
function openSettings(tab) {
  S.settingsTab = tab || S.settingsTab;
  $("settings").hidden = false;
  renderSettings();
}
function closeSettings() { $("settings").hidden = true; }
$("open-settings").onclick = () => openSettings();
$("close-settings").onclick = closeSettings;
$("settings").addEventListener("mousedown", (e) => { if (e.target === $("settings")) closeSettings(); });

function renderSettings() {
  const nav = $("settings-nav");
  nav.replaceChildren(...TABS.map(([id, label, icon]) => {
    const b = h("button", { class: "nav-item" + (id === S.settingsTab ? " sel" : ""), onclick: () => { S.settingsTab = id; renderSettings(); } });
    b.innerHTML = icon + `<span>${label}</span>`;
    return b;
  }));
  const body = $("settings-body");
  body.replaceChildren();
  const tab = S.settingsTab;
  if (tab === "general") {
    const theme = pref("theme", "system");
    const seg = h("div", { class: "seg" }, ...["system", "light", "dark"].map((t) => h("button", {
      class: t === theme ? "sel" : "", onclick: () => { setPref("theme", t); applyTheme(); renderSettings(); } }, t[0].toUpperCase() + t.slice(1))));
    body.append(h("h3", {}, "General"), h("p", { class: "lead" }, "This server and where it keeps your sessions."),
      h("div", { class: "card" },
        h("div", { class: "row" }, h("span", {}, "Appearance"), seg),
        h("div", { class: "row" }, h("span", {}, "Workspace"), h("code", {}, S.app.workspace)),
        h("div", { class: "row" }, h("span", {}, "Session logs"), h("code", {}, S.app.data_dir)),
        h("div", { class: "row" }, h("span", {}, "Default preset"), h("code", {}, S.app.default_preset))),
      h("div", { class: "card" },
        h("div", { class: "card-head" }, h("b", {}, "Keyboard")),
        h("div", { class: "row" }, h("span", {}, "Send"), h("code", {}, "Enter")),
        h("div", { class: "row" }, h("span", {}, "New line"), h("code", {}, "Shift + Enter")),
        h("div", { class: "row" }, h("span", {}, "New session"), h("code", {}, "Ctrl/⌘ + K")),
        h("div", { class: "row" }, h("span", {}, "Skills and commands"), h("code", {}, "/"))));
  } else if (tab === "models") {
    body.append(h("h3", {}, "Models"), h("p", { class: "lead" },
      "Keys are read from environment variables when the server starts; this page never asks for or stores a key."));
    for (const p of S.app.providers) {
      body.append(h("div", { class: "card" },
        h("div", { class: "card-head" }, h("b", {}, p.label), h("span", { class: "sub" }, p.name)),
        p.base_url ? h("div", { class: "row" }, h("span", {}, "Endpoint"), h("code", {}, p.base_url)) : null,
        p.api_key_env ? h("div", { class: "row" }, h("span", {}, "API key"),
          h("span", {}, h("code", {}, p.api_key_env), " ", h("span", { class: "pill " + (p.key_set ? "ok" : "bad") }, p.key_set ? "set" : "not set"))) :
          h("div", { class: "row" }, h("span", {}, "API key"), h("span", { class: "pill" }, "none needed")),
        h("div", { class: "row" }, h("span", {}, "Models"), h("span", {}, p.models.join(", ")))));
    }
    body.append(h("div", { class: "empty-card" }, "Add a local or other OpenAI-compatible server (llama.cpp, vLLM, SGLang) when you start the UI:",
      h("pre", {}, "python -m helioslm_v5.hsh.web --openai-base http://127.0.0.1:8080/v1 --openai-models my-model")));
  } else if (tab === "skills") {
    body.append(h("h3", {}, "Skills"), h("p", { class: "lead" },
      "Folders with a SKILL.md. The agent sees their names and descriptions and loads one when a task matches; type / to start one yourself."));
    if (!S.skills.length) {
      body.append(h("div", { class: "empty-card" }, "No skills found. Create one in the workspace:",
        h("pre", {}, ".hsh/skills/release-notes/SKILL.md\n---\nname: release-notes\ndescription: Use when asked for release notes.\n---\n1. Run git log since the last tag.\n2. Group the changes by area.")));
    }
    for (const s of S.skills) {
      body.append(h("div", { class: "card" },
        h("div", { class: "card-head" }, h("b", {}, s.name), h("span", { class: "sub" }, s.source),
          !s.model ? h("span", { class: "pill" }, "/ only") : null, !s.user ? h("span", { class: "pill" }, "agent only") : null),
        h("div", { class: "row" }, h("span", {}, s.description))));
    }
  } else if (tab === "mcp") {
    body.append(h("h3", {}, "MCP servers"), h("p", { class: "lead" }, "Tools from Model Context Protocol servers appear as mcp__server__tool."));
    body.append(S.app.mcp_config ? h("div", { class: "card" }, h("div", { class: "row" }, h("span", {}, "Config"), h("code", {}, S.app.mcp_config)))
      : h("div", { class: "empty-card" }, "No MCP config. Put a Claude-style .mcp.json in the workspace and start with --mcp:",
        h("pre", {}, '{"mcpServers": {"github": {"command": "npx",\n  "args": ["-y", "@modelcontextprotocol/server-github"],\n  "env": {"GITHUB_TOKEN": "${GITHUB_TOKEN}"}}}\n\npython -m helioslm_v5.hsh.web --mcp .mcp.json')));
  } else if (tab === "presets") {
    body.append(h("h3", {}, "Agent presets"), h("p", { class: "lead" }, "A preset decides which tools and prompt sections a new session gets."));
    for (const p of S.app.presets) {
      body.append(h("div", { class: "card" }, h("div", { class: "card-head" }, h("b", {}, p.id),
        p.id === S.app.default_preset ? h("span", { class: "pill" }, "default") : null),
        h("div", { class: "row" }, h("span", {}, p.rows.join(" · ")))));
    }
  }
}

// ----- boot ---------------------------------------------------------------------
(async function boot() {
  try {
    await Promise.all([loadState(), loadSkills()]);
  } catch (e) { toast("Cannot reach the harness server: " + e.message); return; }
  openSession(sidFromPath());
  setInterval(() => { if (!document.hidden) renderSidebar(); }, 60000);
})();
