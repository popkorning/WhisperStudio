import WaveSurfer from "./vendor/wavesurfer.esm.js";

// ================================================================== basics

const $ = (s, el = document) => el.querySelector(s);
const $$ = (s, el = document) => [...el.querySelectorAll(s)];
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
const LANGS = { ru: "Русский", de: "Deutsch", en: "English", uk: "Українська", fr: "Français", es: "Español", it: "Italiano", tr: "Türkçe", pl: "Polski" };
const FLAG_TEXT = { phantom: "известная фантомная фраза Whisper", no_speech: "похоже на тишину", repetition: "зацикливание, повтор",
  disagree: "модели услышали здесь разное" };
// a segment flagged only because the models disagree is not a hallucination on silence
const onlyDisagree = (s) => (s.flags || []).every((f) => f === "disagree");
const flagTitle = (s) => (onlyDisagree(s) ? "Неразборчиво" : "Возможная галлюцинация");
// models.SHORT_NAMES: ensemble members (seg.alts[].model) and meta.model
const MODEL_NAMES = { ensemble: "Ансамбль", "large-v3": "Whisper", "large-v3-turbo": "Whisper Turbo", "ru-large-v3": "Whisper RU",
  "ru-turbo": "Whisper RU Turbo", "de-turbo": "Whisper DE", "de-check": "Whisper DE", "gigaam-v3": "GigaAM", "t-one": "T-one",
  "canary-v2": "Canary", "qwen3-asr": "Qwen3" };
const mname = (k) => MODEL_NAMES[k] || S.models.find((m) => m.key === k)?.short || k || "";
// several model names: each stays on one line, the list may wrap between them
const mnames = (ks) => [...new Set(ks.map(mname))].map((n) => `<span class="nm">${esc(n)}</span>`).join(", ");
// Doubtful words, two levels (word p = agreement minus doubt penalties, see store.REVIEW_P):
// below LOW_P about 40 % of the words were wrong on the gold reference and they cover most errors -
// highlighted, counted, F3 goes through them; up to WEAK_P few more errors - only a thin dotted line.
const LOW_P = 0.35, WEAK_P = 0.5;
const RATES = [0.75, 1, 1.25, 1.5, 1.75, 2];
const AUDIO_EXT = /\.(mp3|m4a|aac|ogg|oga|opus|wav|flac|amr|wma|webm|mp4|m4v|mov|mkv|avi|3gp|caf|aiff?)$/i;
// Server palette (store.PALETTE) and the design palette — both map to --spk-N.
const SERVER_PAL = ["#0A84FF", "#FF9F0A", "#30D158", "#BF5AF2", "#FF375F", "#64D2FF", "#FFD60A", "#AC8E68"];
const PAL_LIGHT = ["#2F6FDE", "#D9731A", "#2E9E5B", "#8A4FD8", "#D6457F", "#1C98A8", "#9A6B3F", "#5A63C9"];
const PAL_DARK = ["#5B95FF", "#FF9F45", "#4CD07F", "#B484FF", "#FF6FA5", "#40C8D8", "#C99A6B", "#8C93FF"];

const S = {
  docs: [], doc: null, words: [], review: [], ws: null, audio: null, follow: true, stopAt: null, nowIdx: -1,
  userScrollAt: 0, reviewOn: true, reviewDocId: null, revT: null, picked: [], dropped: [], retranscribe: null,
  models: [], status: null, settings: null, dict: null, dictOk: null, opening: null, ticking: false,
  opts: { language: "ru", profile: "balanced", num_speakers: "2", model: "auto", quality: "max" },
  view: "library", sel: null, selBlocks: new Set(), lastBlock: null, pop: null, rate: 1, tab: "frag",
  progSeen: {}, findIdx: 0,
};

// localStorage can be unavailable (blocked site data): fall back to memory for this session
const memStore = new Map();
const ls = {
  get(k, d) {
    try { const v = localStorage.getItem("transcribe." + k); return v == null ? d : JSON.parse(v); }
    catch { return memStore.has(k) ? memStore.get(k) : d; }
  },
  set(k, v) { memStore.set(k, v); try { localStorage.setItem("transcribe." + k, JSON.stringify(v)); } catch {} },
};

// Gold reference (docs/GOLD.md): recordings whose text the owner checked by ear. A gold doc
// is doc.gold = {created, manifest:[{doc, source, start, end, at, label}]}; segment.verified
// marks a fragment confirmed by ear. Models' guesses are hidden there by default (goldVars),
// so they don't tell the listener what to hear.
S.gold = { docs: [], eval: null, history: [], running: false, error: null, ok: null, since: 0 };
S.goldVars = !!ls.get("goldVars", false);
const isGold = (d = S.doc) => !!d && !!d.gold && typeof d.gold === "object";
const goldQuiet = () => isGold() && !S.goldVars;
const inGold = (id) => (S.doc?.id === id ? isGold() : S.gold.docs.some((g) => g.id === id));

async function api(path, { method = "GET", body, form } = {}) {
  const init = { method };
  if (form) init.body = form;
  else if (body !== undefined) { init.body = JSON.stringify(body); init.headers = { "Content-Type": "application/json" }; }
  const r = await fetch(path, init);
  if (!r.ok) {
    let m = r.statusText; try { m = (await r.json()).detail || m; } catch {}
    if (r.status === 404 && (!m || m === "Not Found")) m = "сервер пока не поддерживает это действие";
    const err = new Error(m); err.status = r.status; throw err;
  }
  const ct = r.headers.get("content-type") || "";
  return ct.includes("json") ? r.json() : r.text();
}

function plural(n, one, few, many) {
  const a = n % 10, b = n % 100;
  return a === 1 && b !== 11 ? one : a >= 2 && a <= 4 && (b < 12 || b > 14) ? few : many;
}
const nfiles = (n) => `${n} ${plural(n, "файл", "файла", "файлов")}`;

function clock(t, ms = false) {
  t = Math.max(0, t || 0);
  const h = Math.floor(t / 3600), m = Math.floor((t % 3600) / 60), s = Math.floor(t % 60);
  const base = h ? `${h}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}` : `${m}:${String(s).padStart(2, "0")}`;
  return ms ? `${base}.${String(Math.floor((t % 1) * 100)).padStart(2, "0")}` : base;
}
const pct = (p) => `${Math.round((p || 0) * 100)} %`;
const norm = (w) => String(w || "").toLowerCase().replace(/[^\p{L}\p{N}]+/gu, "");

function toast(msg, ms = 2400, action) {
  const t = $("#toast");
  t.innerHTML = `<span>${esc(msg)}</span>${action ? `<button>${esc(action.label)}</button>` : ""}`;
  t.hidden = false;
  if (action) $("button", t).onclick = () => { t.hidden = true; action.run(); };
  clearTimeout(toast._t); toast._t = setTimeout(() => (t.hidden = true), action ? Math.max(ms, 5000) : ms);
}
const fail = (e) => toast("Ошибка: " + (e.message || e), 4000);
const typingIn = (el) => !!el?.closest?.("input, textarea, select, [contenteditable]");

// ================================================================== theme

function applyTheme(t) {
  if (t === "light" || t === "dark") document.documentElement.dataset.theme = t;
  else delete document.documentElement.dataset.theme;
  try { localStorage.setItem("transcribe.theme", t || "system"); } catch {}
  syncTitleBar();
  repaintWave();
}
// native title bar follows the theme chosen here, not only the system one
function syncTitleBar() {
  const t = document.documentElement.dataset.theme;
  const dark = t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches;
  window.pywebview?.api?.set_theme?.(dark);
}
addEventListener("pywebviewready", syncTitleBar);
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { syncTitleBar(); repaintWave(); });
$("#btn-theme").addEventListener("click", (e) => {
  const cur = document.documentElement.dataset.theme || "system";
  openMenu(e.currentTarget, menuItems([
    { a: "system", label: "Как в системе", checked: cur === "system" },
    { a: "light", label: "Светлая", checked: cur === "light" },
    { a: "dark", label: "Тёмная", checked: cur === "dark" },
  ], true), (a) => applyTheme(a), { above: true });
});
addEventListener("blur", () => document.body.classList.add("blurred"));
addEventListener("focus", () => { document.body.classList.remove("blurred"); refreshDocs(); });

// ================================================================== layout: sidebar and inspector (§7)

const L = { mode: "", side: ls.get("side", { wide: true, narrow: false }), insp: ls.get("insp", { wide: true, over: false }) };
function layoutMode() { const w = innerWidth; return w >= 1240 ? "wide" : w >= 1000 ? "mid" : "narrow"; }
function applyLayout(anim = true) {
  const app = $("#app"), m = layoutMode();
  if (!anim || m !== L.mode) { app.classList.add("no-anim"); requestAnimationFrame(() => requestAnimationFrame(() => app.classList.remove("no-anim"))); }
  L.mode = m;
  const sideKey = m === "narrow" ? "narrow" : "wide", inspKey = m === "wide" ? "wide" : "over";
  app.classList.toggle("side-overlay", m === "narrow");
  app.classList.toggle("insp-overlay", m !== "wide");
  app.classList.toggle("side-open", !!L.side[sideKey]);
  app.classList.toggle("insp-open", !!L.insp[inspKey] && !!S.doc && S.view === "library");
  $("#btn-insp").classList.toggle("on", app.classList.contains("insp-open"));
}
function setSide(open) { L.side[L.mode === "narrow" ? "narrow" : "wide"] = open; ls.set("side", L.side); applyLayout(); }
function setInsp(open) { L.insp[L.mode === "wide" ? "wide" : "over"] = open; ls.set("insp", L.insp); applyLayout(); }
const sideOpen = () => $("#app").classList.contains("side-open");
const inspOpen = () => $("#app").classList.contains("insp-open");
addEventListener("resize", () => { if (layoutMode() !== L.mode) applyLayout(false); positionPop(); });
document.addEventListener("click", (e) => { if (e.target.closest("[data-act=toggle-side]")) setSide(!sideOpen()); });
$("#btn-insp").addEventListener("click", () => setInsp(!inspOpen()));
// overlays close on a click outside them
document.addEventListener("mousedown", (e) => {
  const app = $("#app");
  if (app.classList.contains("side-overlay") && sideOpen() && !e.target.closest("#sidebar, [data-act=toggle-side], #menu, .sheet-wrap")) setSide(false);
  if (app.classList.contains("insp-overlay") && inspOpen() && e.target.closest("#transcript")) setInsp(false);
});

// ================================================================== library and polling

const working = (d) => d && (d.status === "queued" || d.status === "processing");
// Post-processing jobs (rediarize, ИИ) wait as status "processing" with stage "В очереди".
const isWaiting = (d) => d.status === "queued" || (d.status === "processing" && d.stage === "В очереди" && !d.progress);

// Poll often only while something is queued or processing; otherwise slowly, so files queued
// from elsewhere (MCP, CLI) still show up. Every /api/docs reads all doc.json files on the server.
function schedulePoll(ms) {
  clearTimeout(S.pollT);
  const busy = S.docs.some(working) || working(S.doc);
  S.pollT = setTimeout(async () => { await refreshDocs(); schedulePoll(); }, ms ?? (busy ? 1500 : document.hidden ? 15000 : 8000));
}
document.addEventListener("visibilitychange", () => { if (!document.hidden) refreshDocs(); });

let refreshing = null;
async function refreshDocs() {
  if (refreshing) return refreshing;
  refreshing = (async () => {
    try { S.docs = await api("/api/docs"); } catch { return; }
    trackProgress();
    renderDocList(); renderNav();
    if (S.view === "queue") renderQueue();
    // While a document is being opened, the poll must not start a second openDoc():
    // two concurrent opens used to build two WaveSurfer players into the same #wave.
    if (S.doc && !S.opening) {
      const s = S.docs.find((d) => d.id === S.doc.id);
      if (!s) { S.doc = null; showNoDoc(); return; }
      const busy = s.status !== "done";
      if (S.doc.status !== "done" && !busy) openDoc(s.id, { show: false });          // processing finished
      else if (S.doc.status === "done" && busy) openDoc(s.id, { show: false });      // a new job started (rediarize, ИИ…)
      else if (busy) { Object.assign(S.doc, s); renderState(); renderToolbar(); }
      else if ((s.rev || 0) !== (S.doc.rev || 0) && !$("#transcript textarea") && !S.pop) openDoc(s.id, { show: false });  // edited elsewhere (MCP)
    } else if (!S.doc && !S.opening && S.view === "library") renderState();
  })();
  try { await refreshing; } finally { refreshing = null; }
  if (S.docs.some(working)) schedulePoll(); // a new job appeared: switch to the fast poll
}

// remaining-time estimate from the observed progress rate
function trackProgress() {
  const now = Date.now();
  for (const d of S.docs) {
    if (working(d) && !S.queueIds.has(d.id)) { S.queueIds.add(d.id); ls.set("queueIds", [...S.queueIds].slice(-200)); }
    if (d.status !== "processing" || isWaiting(d)) { delete S.progSeen[d.id]; continue; }
    const p = d.progress || 0, r = S.progSeen[d.id];
    if (!r || p < r.p0) S.progSeen[d.id] = { p0: p, t0: now, p, t: now };
    else Object.assign(r, { p, t: now });
  }
}
function eta(d) {
  const r = S.progSeen[d.id];
  if (!r || r.t - r.t0 < 8000 || r.p - r.p0 < 0.02) return "";
  const left = (1 - r.p) / ((r.p - r.p0) / (r.t - r.t0)) / 1000;
  return left < 60 ? "осталось меньше минуты" : `осталось ≈ ${Math.round(left / 60)} мин`;
}

// queue order: the worker takes queued docs oldest first (ids start with a timestamp)
function queuePositions() {
  const q = S.docs.filter((d) => d.status === "queued").map((d) => d.id).sort();
  return Object.fromEntries(q.map((id, i) => [id, i + 1]));
}

const seen = new Set(ls.get("seen", null) || []);
function markSeen(id) { if (!seen.has(id)) { seen.add(id); ls.set("seen", [...seen].slice(-500)); } }

function docSub(d, pos) {
  if (d.status === "error") return `<span class="err">Ошибка — откройте для подробностей</span>`;
  if (isWaiting(d)) return `<span>В очереди${pos[d.id] ? ` · ${pos[d.id]}-й` : ""}</span>`;
  if (d.status === "processing") return `<span>${esc(d.stage || "Обработка")} · ${pct(d.progress)}</span>`;
  const hit = S.search?.q === $("#lib-search").value.trim().toLowerCase() && S.search.hits.get(d.id);
  if (hit) return `<span class="di-hit" title="${esc(hit.text)}">${hit.html}</span>`;
  const bits = [inGold(d.id) ? "Эталон" : "", d.duration ? clock(d.duration) : "", d.language ? d.language.toUpperCase() : "", d.speakers ? `${d.speakers} сп.` : "", mname(d.model)].filter(Boolean);
  const low = d.low ?? d.low_count;
  return `<span>${esc(bits.join(" · "))}</span>${low > 0 ? `<span class="di-low">${low}</span>` : ""}`;
}

function dayGroup(iso) {
  const d = new Date(iso || 0), now = new Date();
  const day = (x) => new Date(x.getFullYear(), x.getMonth(), x.getDate()).getTime();
  const diff = Math.round((day(now) - day(d)) / 864e5);
  if (diff <= 0) return "Сегодня";
  if (diff === 1) return "Вчера";
  const monday = day(now) - ((now.getDay() + 6) % 7) * 864e5;
  return day(d) >= monday ? "На этой неделе" : "Ранее";
}

let listSig = "";
function renderDocList() {
  const el = $("#doc-list");
  el.hidden = S.view !== "library";
  const q = $("#lib-search").value.trim().toLowerCase(), qn = q.replace(/ё/g, "е");
  const hits = S.search?.q === q ? S.search.hits : null;
  const fname = (p) => String(p || "").split(/[\\/]/).pop();
  const list = S.docs.filter((d) => !q || hits?.has(d.id) || ((d.title || "") + " " + fname(d.source) + " " + (d.preview || "")).toLowerCase().replace(/ё/g, "е").includes(qn));
  if (!list.length) {
    listSig = "";
    el.innerHTML = q ? `<div class="side-empty">Ничего не найдено по запросу «${esc(q)}»<br><button class="text-btn" data-act="clear-search">Очистить поиск</button></div>`
      : `<div class="side-empty">Записей пока нет</div>`;
    return;
  }
  const pos = queuePositions();
  const item = (d) => {
    const busy = working(d) && !isWaiting(d);
    const cls = `doc-item${S.doc?.id === d.id ? " active" : ""}${d.status === "done" && !seen.has(d.id) ? " unread" : ""}`;
    return { d, cls, sub: docSub(d, pos), busy, p: d.progress || 0 };
  };
  const items = list.map(item);
  const sig = q + (hits ? "+" : "") + "|" + items.map((x) => x.d.id + (x.busy ? "b" : "")).join(",");
  if (sig === listSig) {           // same rows: patch in place, so progress bars animate (§4.1)
    for (const x of items) {
      const b = el.querySelector(`.doc-item[data-id="${x.d.id}"]`);
      if (!b) continue;
      b.className = x.cls;
      const t = $(".di-title", b); if (t.textContent !== x.d.title) t.textContent = x.d.title;
      const s = $(".di-sub", b); if (s.innerHTML !== x.sub) s.innerHTML = x.sub;
      const bar = $(".di-bar i", b); if (bar) bar.style.transform = `scaleX(${x.p})`;
    }
    return;
  }
  listSig = sig;
  let html = "", group = "";
  for (const x of items) {
    const g = dayGroup(x.d.created);
    if (g !== group) { html += `<div class="group-label">${g}</div>`; group = g; }
    html += `<button class="${x.cls}" data-id="${x.d.id}" role="option"><div class="di-title">${esc(x.d.title)}</div><div class="di-sub">${x.sub}</div>`
      + (x.busy ? `<div class="di-bar"><i style="transform:scaleX(${x.p})"></i></div>` : "") + `</button>`;
  }
  el.innerHTML = html;
}

$("#doc-list").addEventListener("click", (e) => {
  if (e.target.closest("[data-act=clear-search]")) { $("#lib-search").value = ""; S.search = null; return renderDocList(); }
  const b = e.target.closest(".doc-item");
  if (!b) return;
  const id = b.dataset.id, q = S.search?.hits.has(id) ? S.search.q : null;
  openDoc(id).then(() => { if (q && S.doc?.id === id) showFound(q); });
  if (L.mode === "narrow") setSide(false);
});
$("#doc-list").addEventListener("contextmenu", (e) => {
  const b = e.target.closest(".doc-item");
  if (!b) return;
  e.preventDefault();
  const id = b.dataset.id, d = S.docs.find((x) => x.id === id);
  openMenu({ x: e.clientX, y: e.clientY }, menuItems([
    { a: "open", label: "Открыть" },
    { a: "rename", label: "Переименовать", kb: "F2" },
    { a: "source", label: "Показать исходный файл" },
    { a: "redo", label: "Распознать заново…" },
    ...(d && d.status === "done" ? [inGold(id) ? { a: "gold-off", label: "Убрать из эталона…" } : { a: "gold-on", label: "Сделать эталоном" }] : []),
    "-",
    { a: "del", label: "Удалить…", cls: "danger" },
  ]), async (a) => {
    if (a === "open") openDoc(id);
    if (a === "gold-on" || a === "gold-off") setGold(id, a === "gold-on", d?.title);
    if (a === "rename") { await openDoc(id); startRename(); }
    if (a === "source") revealPath(d?.source);
    if (a === "redo") { await openDoc(id); redoDoc(S.doc); }
    if (a === "del") deleteDoc(id, d?.title);
  });
});
// Content search: the title filter answers at once, the server's full-text hits follow.
function snippet(text, q) {
  const k = text.toLowerCase().replace(/ё/g, "е").indexOf(q.replace(/ё/g, "е"));
  if (k < 0) return esc(text.slice(0, 80));
  const a = Math.max(0, k - 12);
  return (a ? "…" : "") + esc(text.slice(a, k)) + `<mark>${esc(text.slice(k, k + q.length))}</mark>` + esc(text.slice(k + q.length, k + q.length + 60));
}
let searchT = 0;
async function runSearch() {
  const q = $("#lib-search").value.trim().toLowerCase();
  if (q.length < 2) { S.search = null; listSig = ""; return renderDocList(); }
  let res;
  try { res = await api(`/api/search?q=${encodeURIComponent(q)}`); } catch { return; }
  if ($("#lib-search").value.trim().toLowerCase() !== q) return;   // typed further meanwhile
  const hits = new Map();
  for (const h of res) {
    if (!h.segment_id || hits.has(h.doc_id)) continue;             // title matches show anyway
    const n = h.count || 1;
    hits.set(h.doc_id, { n, text: h.text, html: snippet(h.text, q) + (n > 1 ? ` <span class="di-n">+${n - 1}</span>` : "") });
  }
  S.search = { q, hits };
  listSig = ""; renderDocList();
}
$("#lib-search").addEventListener("input", () => { listSig = ""; renderDocList(); clearTimeout(searchT); searchT = setTimeout(runSearch, 200); });
$("#lib-search").addEventListener("keydown", (e) => { if (e.key === "Escape") { e.currentTarget.value = ""; S.search = null; listSig = ""; renderDocList(); e.currentTarget.blur(); } });

function renderNav() {
  $$(".nav-item[data-view]").forEach((b) => b.classList.toggle("active", b.dataset.view === S.view));
  const active = S.docs.filter(working).length;
  const batch = queueItems();
  const done = batch.filter((d) => !working(d)).length;
  $("#nav-queue").textContent = active ? `${done} из ${batch.length}` : "";
  const wm = S.models.filter((m) => m.kind === "whisper");
  const none = wm.length && !wm.some((m) => m.installed);
  $("#nav-models-n").textContent = none ? "нет модели" : "";
  $("#nav-models-n").classList.toggle("danger", !!none);
  const g = S.gold.docs, segs = g.reduce((n, d) => n + (d.segments || 0), 0), ver = g.reduce((n, d) => n + (d.verified || 0), 0);
  $("#nav-gold").textContent = segs ? `${ver} из ${segs}` : "";
  $("#nav-gold").title = segs ? `Проверено на слух ${ver} из ${segs} ${plural(segs, "фрагмента", "фрагментов", "фрагментов")}` : "";
}

$$(".nav-item[data-view]").forEach((b) => b.addEventListener("click", () => setView(b.dataset.view)));
$("#nav-dict").addEventListener("click", () => openDictSheet());
$("#nav-models").addEventListener("click", () => openModels());

function setView(v) {
  S.view = v;
  $("#view-doc").hidden = v !== "library";
  $("#view-queue").hidden = v !== "queue";
  $("#view-gold").hidden = v !== "gold";
  if (v === "queue") { pause(); renderQueue(); }
  if (v === "gold") { pause(); closePop(); renderGold(); loadGold(true); }
  renderNav(); listSig = ""; renderDocList(); applyLayout();
  if (L.mode === "narrow" && v !== "library") setSide(false);
}

async function deleteDoc(id, title) {
  if (!confirm(`Удалить «${title || "запись"}»? Это нельзя отменить.`)) return;
  try { await api(`/api/docs/${id}`, { method: "DELETE" }); } catch (e) { return fail(e); }
  if (S.doc?.id === id) { S.doc = null; showNoDoc(); }
  refreshDocs();
}

function revealPath(p) {
  if (!p) return toast("Путь к исходному файлу неизвестен");
  const a = window.pywebview?.api;
  if (a?.reveal_file) return a.reveal_file(p);
  navigator.clipboard?.writeText(p).then(() => toast("Путь скопирован: " + p, 3500), () => toast(p, 4000));
}

// ================================================================== document

function showNoDoc() {
  destroyPlayer();
  S.sel = null; S.selBlocks.clear(); closePop();
  $("#transcript").innerHTML = "";
  $("#player").hidden = true;
  setTitle();
  renderToolbar(); renderState(); renderInspector(); applyLayout(); listSig = ""; renderDocList();
}

let openSeq = 0;
async function openDoc(id, { show = true } = {}) {
  const seq = ++openSeq;
  S.opening = id;
  let doc;
  try { doc = await api(`/api/docs/${id}`); }
  catch (e) { if (seq === openSeq) S.opening = null; return fail(e); }
  if (seq !== openSeq) return;          // a newer openDoc() superseded this one
  S.opening = null;
  const same = S.doc?.id === id;
  if (!same) {
    S.sel = null; S.selBlocks.clear(); S.revT = null; closePop(); toggleFind(false, true);
  }
  S.doc = doc;
  ls.set("last", id);
  if (show && S.view !== "library") setView("library");
  if (doc.status === "done") markSeen(id);
  if (S.reviewDocId !== id) { S.reviewDocId = id; S.reviewOn = null; }
  listSig = ""; renderDocList(); renderToolbar(); setTitle(doc.title);
  applyLayout();
  if (doc.status === "done") {
    renderState(); renderTranscript();
    $("#player").hidden = !doc.segments.length && !doc.duration;
    if (!same || !S.ws) await setupPlayer();
    else paintStrip();
  } else {
    destroyPlayer();
    $("#transcript").innerHTML = ""; $("#player").hidden = true;
    renderState();
  }
  renderInspector();
  if (!same && isGold() && doc.status === "done") goGold(0, false);    // the cursor on the first unchecked fragment
}

function setTitle(t) {
  document.title = t ? `Транскрибация — ${t}` : "Транскрибация";
  window.pywebview?.api?.set_title?.(document.title);
}

function stateBox(h, p, btn) {
  return `<div class="state-box"><h2>${esc(h)}</h2>${p ? `<p>${p}</p>` : ""}${btn ? `<button class="btn primary" data-act="${btn.a}">${esc(btn.label)}</button>` : ""}</div>`;
}

function renderState() {
  const el = $("#state"), d = S.doc;
  let html = "";
  if (!d && !S.loaded) html = "";
  else if (!d) {
    const wm = S.models.filter((m) => m.kind === "whisper");
    if (!S.docs.length && wm.length && !wm.some((m) => m.installed))
      html = stateBox("Установите модель распознавания", "Всё работает офлайн. Нужна одна модель Whisper, ≈ 1,6 ГБ.", { a: "models", label: "Открыть модели" });
    else if (!S.docs.length)
      html = stateBox("Перетащите записи в окно", "MP3, M4A, OGG, OPUS, WAV, FLAC, AMR и видео. Распознаётся на этом компьютере.", { a: "pick", label: "Выбрать файлы…" });
    else html = stateBox("Выберите запись слева", "", null);
  } else if (d.status === "error") {
    html = `<div class="state-box"><h2>Не удалось распознать</h2><p class="mono">${esc(d.error || "неизвестная ошибка")}</p><button class="btn primary" data-act="retry">Повторить</button></div>`;
  } else if (d.status !== "done") {
    if (isWaiting(d)) {
      const pos = queuePositions()[d.id];
      html = `<div class="proc"><h3>${pos ? `В очереди — ${pos}-й` : "В очереди"}</h3><p>Файлы распознаются по одному, чтобы не делить видеокарту.</p></div>`;
    } else {
      const s = S.docs.find((x) => x.id === d.id) || d;
      html = `<div class="proc"><h3>${esc(d.stage || "Обработка")}</h3><div class="bar"><i style="transform:scaleX(${d.progress || 0})"></i></div>
        <div class="proc-nums"><span>${pct(d.progress)}</span><span>${esc(eta(s))}</span></div></div>`;
    }
  } else if (!d.segments.length) {
    html = stateBox("Речь не найдена", "Возможно, запись тихая или это музыка.", { a: "redo", label: "Распознать заново…" });
  }
  el.hidden = !html;
  const bar = $(".proc .bar i", el);
  if (bar && $(".proc h3", el)?.textContent === (d?.stage || "")) {   // keep the bar element so scaleX animates
    bar.style.transform = `scaleX(${d.progress || 0})`;
    $(".proc-nums", el).innerHTML = `<span>${pct(d.progress)}</span><span>${esc(eta(S.docs.find((x) => x.id === d.id) || d))}</span>`;
    return;
  }
  if (el.innerHTML !== html) el.innerHTML = html;
}

$("#state").addEventListener("click", (e) => {
  const a = e.target.closest("[data-act]")?.dataset.act;
  if (a === "models") openModels();
  if (a === "pick") openNewSheet();
  if (a === "redo") redoDoc(S.doc);
  if (a === "retry") retryDoc(S.doc);
});

// «Распознать заново»: asks for new options
function redoDoc(d) {
  if (d) openNewSheet({ retranscribe: d });
}

async function retryDoc(d) {
  if (!d) return;
  try { const n = await api(`/api/docs/${d.id}/retranscribe`, { method: "POST", body: { options: d.options || {} } }); toast("Запись снова в очереди"); await refreshDocs(); if (S.view === "library") openDoc(n.id); }
  catch (e) { fail(e); }
}

function renderToolbar() {
  const d = S.doc;
  const t = $("#doc-title");
  if (!t.isContentEditable) t.textContent = d ? d.title : (S.view === "library" ? "" : "");
  const bits = [];
  if (d) {
    if (d.status === "done") {
      if (d.duration) bits.push(esc(clock(d.duration)));
      if (d.language) bits.push(esc(LANGS[d.language] || d.language));
      const n = Object.keys(d.speakers || {}).length;
      if (n) bits.push(`${n} ${plural(n, "спикер", "спикера", "спикеров")}`);
      if (d.edited) bits.push("изменено");
      if (d.error) bits.push(`<span class="err" title="${esc(d.error)}" data-act="post-err">дополнительная обработка не удалась</span>`);
    } else if (d.status === "error") bits.push(`<span class="err">ошибка</span>`);
    else bits.push(esc(isWaiting(d) ? "в очереди" : `${d.stage || "обработка"} · ${pct(d.progress)}`));
  }
  $("#doc-meta").innerHTML = bits.join(" · ");
  const done = d?.status === "done" && d.segments.length;
  $("#tb-actions").hidden = !d;
  $("#review-group").hidden = !done || goldQuiet();
  $("#btn-find").hidden = !done;
  $("#btn-export").disabled = !done;
  renderReviewBtn(); renderGoldBar();
}
$("#doc-meta").addEventListener("click", (e) => { if (e.target.closest("[data-act=post-err]")) toast(S.doc.error, 6000); });

// ---- rename: double click or F2
function startRename() {
  const t = $("#doc-title");
  if (!S.doc || t.isContentEditable) return;
  t.contentEditable = "plaintext-only";
  t.focus();
  getSelection().selectAllChildren(t);
}
$("#doc-title").addEventListener("dblclick", startRename);
$("#doc-title").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); e.target.blur(); }
  if (e.key === "Escape") { e.target.textContent = S.doc?.title || ""; e.target.blur(); }
});
$("#doc-title").addEventListener("blur", async (e) => {
  const el = e.target;
  el.removeAttribute("contenteditable");
  const title = el.textContent.trim();
  if (!S.doc || !title || title === S.doc.title) { el.textContent = S.doc?.title || ""; return; }
  S.doc.title = title; setTitle(title);
  await api(`/api/docs/${S.doc.id}`, { method: "PATCH", body: { title } }).catch(fail);
  refreshDocs();
});

// ================================================================== transcript

// A new block starts with a new speaker, or after a pause longer than 30 s (the name repeats).
// A gold recording is glued from windows of several recordings: a window boundary starts a
// new block too (piece: segment -> index in goldPieces()).
function blocks(segs, piece = null) {
  const out = [];
  for (const s of segs) {
    const last = out[out.length - 1];
    const prev = last?.segs[last.segs.length - 1];
    const p = piece ? piece(s) : -1;
    if (last && last.speaker === s.speaker && s.start - prev.end <= 30 && last.piece === p) last.segs.push(s);
    else out.push({ speaker: s.speaker, segs: [s], piece: p });
  }
  return out;
}

// manifest pieces of a gold doc in audio order; a segment belongs to the piece its middle is in
function goldPieces(d = S.doc) {
  return (d?.gold?.manifest || []).filter((p) => p && Number.isFinite(+p.at)).slice().sort((a, b) => a.at - b.at);
}
function pieceOf(pieces) {
  return (s) => {
    const mid = (s.start + s.end) / 2;
    let k = -1;
    for (let i = 0; i < pieces.length && pieces[i].at <= mid + 0.05; i++) k = i;
    return k;
  };
}
const fname = (p) => String(p || "").split(/[\\/]/).pop();
function pieceHtml(p) {
  const win = Number.isFinite(+p.start) && Number.isFinite(+p.end) ? ` · ${clock(p.start)}–${clock(p.end)}` : "";
  return `<div class="gold-piece"><div class="gp-side"><button class="b-time" data-t="${+p.at}" title="Играть отсюда">${clock(p.at)}</button></div>`
    + `<div class="gp-body"><span class="gp-label">${esc(p.label || "Фрагмент")}</span><span class="gp-src" title="${esc(p.source || "")}">${esc(fname(p.source))}${win}</span></div></div>`;
}

function spkIdx(id) {
  const sp = S.doc?.speakers?.[id];
  if (!sp) return -1;
  const c = String(sp.color || "").toUpperCase();
  for (const pal of [PAL_LIGHT, PAL_DARK, SERVER_PAL]) { const k = pal.indexOf(c); if (k >= 0) return k; }
  return -2;
}
// speaker colour as CSS: a palette token (readable in both themes) or the stored custom colour
function spkColor(id) {
  const k = spkIdx(id);
  if (k >= 0) return `var(--spk-${k + 1})`;
  if (k === -2) return S.doc.speakers[id].color;
  return "var(--text-3)";
}
const spkList = () => Object.entries(S.doc?.speakers || {});
const segById = (id) => S.doc?.segments.find((s) => s.id === id);

// ---- "Верно": cosmetic, kept per document in localStorage (no server API)
const okKey = (segId, w, wi) => `${segId}|${w.s != null ? (+w.s).toFixed(2) : "#" + wi}|${norm(w.w)}`;
function okSet(docId = S.doc?.id) { return new Set(ls.get("ok." + docId, [])); }
function addOk(keys) { const s = okSet(); keys.forEach((k) => s.add(k)); ls.set("ok." + S.doc.id, [...s].slice(-3000)); }

function renderTranscript() {
  const d = S.doc;
  const scroller = $("#scroller"), scroll = scroller.scrollTop;
  S.words = []; S.review = [];
  const q = $("#findbar").hidden ? "" : $("#find-input").value.trim().toLowerCase();
  const qt = q.replace(/ё/g, "е").split(/\s+/).map(normFind).filter(Boolean);
  const ok = okSet();
  const gold = isGold(d), quiet = goldQuiet(), cur = gold ? curSegId() : null;
  if (gold) S.reviewOn = S.goldVars;
  // one fragment: word spans (S.words, S.review)
  const segHtml = (s) => {
    const flagged = s.flags?.length && !quiet;
    if (flagged) S.review.push({ k: "seg", id: s.id, t: s.start, e: s.end });
    let h = `<div class="seg${flagged ? " flagged" : ""}${gold && s.verified ? " ver" : ""}${gold && s.id === cur ? " cur" : ""}${S.sel?.k === "seg" && S.sel.id === s.id ? " sel" : ""}" data-id="${s.id}"${flagged ? ` title="${flagTitle(s)}: ${esc(s.flags.map((f) => FLAG_TEXT[f] || f).join(", "))}"` : ""}>`;
    if (gold) h += `<button class="ver-btn" data-ver tabindex="-1" title="${s.verified ? "Проверено на слух · щёлкните, чтобы снять отметку (Shift+Enter)" : "Отметить: верно на слух (Enter)"}" aria-label="${s.verified ? "Проверено" : "Не проверено"}"><svg class="ic"><use href="icons.svg#check"/></svg></button>`;
    const ws = s.words?.length ? s.words : [{ w: s.text, s: s.start, e: s.end, p: 1 }];
    const found = qt.length ? findIn(ws, qt) : null;
    h += ws.map((w, wi) => {
      const i = S.words.length;
      const okd = ok.has(okKey(s.id, w, wi));
      const low = w.p < LOW_P && !w.edited && !okd;
      const weak = !low && w.p < WEAK_P && !w.edited && !okd;
      const x = { s: w.s, e: w.e, seg: s.id, wi, p: w.p, w: w.w, low: low || weak };
      S.words.push(x);
      if (low) S.review.push({ k: "w", i, t: w.s, e: w.e });
      const cls = ["w"];
      if (low) cls.push("low");
      else if (weak) cls.push("weak");
      else if (w.edited) cls.push("ed");
      else if (w.p < WEAK_P) cls.push("ok");
      if (S.sel?.k === "w" && S.sel.seg === s.id && S.sel.wi === wi) cls.push("sel");
      let txt = esc(w.w);
      if (found?.has(wi)) txt = `<mark class="${found.get(wi)}">${txt}</mark>`;
      return `<span class="${cls.join(" ")}" data-i="${i}">${txt}</span>`;
    }).join(" ");
    return h + `</div>`;
  };
  let html = "";
  const pieces = gold ? goldPieces(d) : [];
  let piece = -1;
  for (const b of blocks(d.segments, pieces.length ? pieceOf(pieces) : null)) {
    if (b.piece !== piece && b.piece >= 0) html += pieceHtml(pieces[b.piece]);
    piece = b.piece;
    const who = d.speakers?.[b.speaker];
    const key = b.segs[0].id;
    const aiSegs = b.segs.filter((s) => s.ai), ai = aiSegs.length;
    const aiTip = ai ? `Исправлено ИИ: ${ai} ${plural(ai, "фрагмент", "фрагмента", "фрагментов")} из ${b.segs.length}`
      + aiSegs.filter((s) => s.orig_text && s.orig_text !== s.text).slice(0, 4).map((s) => `\n\nБыло: ${s.orig_text.length > 220 ? s.orig_text.slice(0, 220) + "…" : s.orig_text}`).join("") : "";
    const aiMark = ai ? `<span class="ai-mark" title="${esc(aiTip)}">✦</span>` : "";
    html += `<div class="block${S.selBlocks.has(key) ? " sel" : ""}" data-key="${key}" style="--c:${who ? spkColor(b.speaker) : "var(--text-3)"}">
      <div class="b-side">${who ? `<button class="b-spk" data-spk="${b.speaker}" title="Сменить спикера"><span class="dot"></span><span>${esc(who.name)}</span></button>` : ""}
      <div class="b-tm"><button class="b-time" data-t="${b.segs[0].start}" title="Играть отсюда">${clock(b.segs[0].start)}</button>${aiMark}</div></div><div class="b-body">`;
    html += b.segs.map(segHtml).join("") + `</div></div>`;
  }
  S.review.sort((a, b) => a.t - b.t);
  if (S.reviewOn == null) S.reviewOn = S.review.length > 0;     // on by default when there is something to check
  $("#transcript").innerHTML = html;
  $("#transcript").classList.toggle("review-on", S.reviewOn);
  $("#transcript").classList.toggle("gold", gold);
  scroller.scrollTop = scroll;
  S.nowIdx = -1;
  if (S.audio) highlight(S.audio.currentTime, true);
  if (q) updateFindCount();
  renderReviewBtn(); paintTicks(); updateEdges();
  if (S.sel?.k === "w") { const i = S.words.findIndex((x) => x.seg === S.sel.seg && x.wi === S.sel.wi); if (i >= 0) S.sel.i = i; else S.sel = null; }
}

// "ещё" finds "еще"; "25-го" finds "25-го,"; a phrase is matched word by word
function normFind(w) { return w.toLowerCase().replace(/ё/g, "е").replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, ""); }
function findIn(ws, qt) {
  const out = new Map(), n = qt.length, W = ws.map((w) => normFind(w.w));
  for (let i = 0; i + n <= W.length; i++) {
    let hit = n === 1 ? W[i].includes(qt[0])
      : W[i].endsWith(qt[0]) && W[i + n - 1].startsWith(qt[n - 1]) && qt.slice(1, -1).every((t, k) => W[i + 1 + k] === t);
    if (!hit) continue;
    for (let k = 0; k < n; k++) out.set(i + k, k ? "hit-more" : "hit");
  }
  return out;
}

function renderReviewBtn() {
  const n = S.review.length, b = $("#btn-review");
  b.innerHTML = n ? `Проверка<b>${n}</b>` : "Проверка";
  b.classList.toggle("on", S.reviewOn);
  b.title = S.reviewOn ? "Скрыть подсветку сомнительных слов" : "Подсветить сомнительные слова";
  $("#btn-rev-prev").disabled = $("#btn-rev-next").disabled = !n;
}
function setReview(on) {
  S.reviewOn = on;
  $("#transcript").classList.toggle("review-on", on);
  renderReviewBtn(); paintTicks();
  if (!on) closePop();
}
$("#btn-review").addEventListener("click", () => (isGold() ? setGoldVars(!S.goldVars) : setReview(!S.reviewOn)));
$("#btn-rev-prev").addEventListener("click", () => goReview(-1));
$("#btn-rev-next").addEventListener("click", () => goReview(1));

async function apply(promise, msg) {
  try {
    const r = await promise;
    const doc = r?.segments ? r : r?.doc || S.doc;
    S.doc = doc;
    renderToolbar(); renderTranscript(); paintStrip(); renderInspector(); refreshDocs();
    if (S.pop) positionPop();
    if (msg) toast(msg);
    return doc;
  } catch (e) { fail(e); }
}

// ---- selection

function select(sel) {
  S.sel = sel;
  $$(".w.sel, .seg.sel", $("#transcript")).forEach((x) => x.classList.remove("sel"));
  if (sel?.k === "w") $(`.w[data-i="${sel.i}"]`)?.classList.add("sel");
  if (sel?.k === "seg") $(`.seg[data-id="${sel.id}"]`)?.classList.add("sel");
  if (isGold()) {       // the gold cursor: the fragment Enter / E / R act on
    const id = curSegId();
    $$(".seg.cur", $("#transcript")).forEach((x) => x.classList.toggle("cur", x.dataset.id === id));
    if (id) $(`.seg[data-id="${id}"]`)?.classList.add("cur");
  }
  renderInspector();
}
const curSegId = () => (S.sel?.k === "w" ? S.sel.seg : S.sel?.k === "seg" ? S.sel.id : null);
function selectWord(i) { const x = S.words[i]; if (x) select({ k: "w", i, seg: x.seg, wi: x.wi }); }

function blockKeys() { return $$(".block", $("#transcript")).map((b) => b.dataset.key); }
function selectBlock(key, range) {
  const keys = blockKeys();
  if (range && S.lastBlock && keys.includes(S.lastBlock)) {
    const [a, b] = [keys.indexOf(S.lastBlock), keys.indexOf(key)].sort((x, y) => x - y);
    S.selBlocks = new Set(keys.slice(a, b + 1));
  } else if (S.selBlocks.size === 1 && S.selBlocks.has(key)) { S.selBlocks.clear(); key = null; }
  else S.selBlocks = new Set([key]);
  S.lastBlock = key;
  $$(".block", $("#transcript")).forEach((b) => b.classList.toggle("sel", S.selBlocks.has(b.dataset.key)));
}
function segsOfBlocks(keys) {
  const out = [];
  for (const b of $$(".block", $("#transcript"))) if (keys.has(b.dataset.key)) out.push(...$$(".seg", b).map((s) => s.dataset.id));
  return out;
}

$("#transcript").addEventListener("click", (e) => {
  const segEl = e.target.closest(".seg");
  if (segEl?.querySelector("textarea")) return;
  if (e.target.closest("[data-ver]")) {
    const s = segById(segEl.dataset.id);
    select({ k: "seg", id: s.id });
    return setVerified(s.id, !s.verified);
  }
  const w = e.target.closest(".w");
  if (w) {
    const i = +w.dataset.i, x = S.words[i];
    selectWord(i);
    if (e.ctrlKey || e.metaKey || e.altKey) return playFrom(x.s);
    playRange(x.s, x.e);
    if (S.reviewOn && (w.classList.contains("low") || w.classList.contains("weak"))) { S.revT = x.s; openWordPop(i); }
    else if (segEl.classList.contains("flagged") && S.reviewOn) { S.revT = segById(x.seg).start; openSegPop(x.seg); }
    return;
  }
  const t = e.target.closest(".b-time");
  if (t) return playFrom(+t.dataset.t);
  const spk = e.target.closest(".b-spk");
  if (spk) return speakerPop(spk, spk.dataset.spk, spk.closest(".block").dataset.key);
  if (segEl) { select({ k: "seg", id: segEl.dataset.id }); return; }
  const side = e.target.closest(".b-side");
  if (side) return selectBlock(side.closest(".block").dataset.key, e.shiftKey);
});

$("#transcript").addEventListener("dblclick", (e) => {
  const seg = e.target.closest(".seg");
  if (seg && !seg.querySelector("textarea")) { e.preventDefault(); closePop(); startEdit(seg.dataset.id, e.target.closest(".w")); }
});

$("#transcript").addEventListener("contextmenu", (e) => {
  const seg = e.target.closest(".seg");
  if (!seg || seg.querySelector("textarea")) return;
  e.preventDefault();
  const w = e.target.closest(".w");
  segMenu({ x: e.clientX, y: e.clientY }, seg.dataset.id, w ? S.words[+w.dataset.i] : null);
});

// ---- inline editing

function startEdit(id, wordEl) {
  const segEl = $(`.seg[data-id="${id}"]`);
  if (!segEl || segEl.querySelector("textarea")) return;
  const s = segById(id);
  pause(); closePop();
  const ta = document.createElement("textarea");
  ta.value = s.text;
  ta.spellcheck = true;
  ta.lang = s.lang || S.doc.language || "ru";
  segEl.innerHTML = "";
  segEl.append(ta);
  const hint = document.createElement("div");
  hint.className = "edit-hint";
  hint.textContent = isGold() ? "Enter — сохранить и дальше · Esc — отмена · Ctrl+/ — ??? · Ctrl+Пробел — прослушать"
    : "Enter — сохранить · Esc — отмена · Tab — следующий · Ctrl+Пробел — прослушать";
  segEl.append(hint);
  const fit = () => { ta.style.height = "auto"; ta.style.height = ta.scrollHeight + "px"; };
  fit(); ta.addEventListener("input", fit);
  ta.focus();
  let pos = s.text.length;       // caret near the clicked word
  if (wordEl) {
    const x = S.words[+wordEl.dataset.i];
    pos = (s.words || []).slice(0, x.wi).reduce((n, w) => n + w.w.length + 1, 0);
  }
  ta.setSelectionRange(pos, pos);
  let done = false;
  // advance (Enter in a gold doc): the fragment is checked by ear, go on to the next unchecked one
  const finish = async (save, next, advance) => {
    if (done) return; done = true;
    const text = ta.value.replace(/\s+/g, " ").trim();
    const changed = save && text !== s.text;
    if (changed) await apply(api(`/api/docs/${S.doc.id}/segments/${id}/text`, { method: "POST", body: { text } }));
    else renderTranscript();
    if (advance && isGold()) {
      // a text edit marks the fragment verified on the server; an unchanged text is confirmed here
      if (!changed && !segById(id)?.verified) setVerified(id, true);
      return goGold(1, true, s.start);
    }
    if (next) {
      const i = S.doc.segments.findIndex((x) => x.id === id);
      const n = S.doc.segments[i >= 0 ? i + 1 : 0] || (i < 0 ? null : null);
      if (n) startEdit(n.id);
    }
  };
  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); finish(true, false, true); }
    else if (e.key === "Escape") { e.preventDefault(); e.stopPropagation(); finish(false); }
    else if (e.key === "Tab") { e.preventDefault(); finish(true, true); }
    else if (e.key === " " && e.ctrlKey) { e.preventDefault(); playRange(s.start, s.end); }
    else if (e.code === "Slash" && e.ctrlKey && isGold()) {    // inaudible here: «???» at the caret (docs/GOLD.md)
      e.preventDefault();
      const a = ta.selectionStart, b = ta.selectionEnd, v = ta.value;
      const ins = (a > 0 && !/\s/.test(v[a - 1]) ? " " : "") + "???" + (b < v.length && !/\s/.test(v[b]) ? " " : "");
      ta.setRangeText(ins, a, b, "end"); fit();
    }
  });
  ta.addEventListener("blur", () => setTimeout(() => finish(true), 120));
}

const segApi = (id, tail = "", init) => api(`/api/docs/${S.doc.id}/segments/${id}${tail}`, init);
const mergeNext = (id) => apply(segApi(id, "/merge", { method: "POST" }));
const deleteSeg = (id) => apply(segApi(id, "", { method: "DELETE" }), "Фрагмент удалён · Ctrl+Z — вернуть");
const keepSeg = (id) => apply(segApi(id, "/keep", { method: "POST" }));
const setSegText = (id, text) => apply(segApi(id, "/text", { method: "POST", body: { text } }));

// ================================================================== menus (§4.11)

// items: { a, label, kb, cls, checked, sub, title, off, dot, num, data } or "-" or { label: "...", kind: "label" }
function menuItems(items, checkable = false) {
  return items.map((it) => {
    if (it === "-") return "<hr>";
    if (it.kind === "label") return `<div class="m-label">${esc(it.label)}</div>`;
    if (it.kind === "html") return it.html;
    const data = Object.entries(it.data || {}).map(([k, v]) => ` data-${k}="${esc(v)}"`).join("");
    const cls = ["mi", it.cls, it.checked ? "checked" : "", it.off ? "off" : ""].filter(Boolean).join(" ");
    return `<button class="${cls}" data-a="${esc(it.a)}"${data}${it.title ? ` title="${esc(it.title)}"` : ""} role="menuitem">`
      + (checkable ? `<span class="chk"><svg class="ic"><use href="icons.svg#check"/></svg></span>` : "")
      + (it.num ? `<span class="num">${it.num}</span>` : "")
      + (it.dot ? `<span class="dot" style="--c:${it.dot}"></span>` : "")
      + `<span>${esc(it.label)}</span>`
      + (it.kb ? `<span class="kb">${esc(it.kb)}</span>` : it.subHtml ? `<span class="mi-sub">${it.subHtml}</span>` : it.sub ? `<span class="mi-sub">${esc(it.sub)}</span>` : "") + `</button>`;
  }).join("");
}

// anchor: an element (menu goes under it, right-aligned when it would overflow) or {x, y}
function openMenu(anchor, html, onClick, { above = false, align = "start" } = {}) {
  const m = $("#menu");
  clearTimeout(m._t);
  m.classList.remove("closing");
  m.innerHTML = html; m.hidden = false;
  m.style.left = "0px"; m.style.top = "0px";
  const r = m.getBoundingClientRect();
  let x, y, oy = "top";
  if (anchor instanceof Element) {
    const a = anchor.getBoundingClientRect();
    x = align === "end" ? a.right - r.width : a.left;
    y = above ? a.top - r.height - 6 : a.bottom + 6;
    if (!above && y + r.height > innerHeight - 8) { y = a.top - r.height - 6; oy = "bottom"; }
    if (above) oy = "bottom";
  } else { x = anchor.x; y = anchor.y; if (y + r.height > innerHeight - 8) { y = anchor.y - r.height; oy = "bottom"; } }
  x = Math.max(8, Math.min(x, innerWidth - r.width - 8));
  y = Math.max(8, Math.min(y, innerHeight - r.height - 8));
  m.style.left = x + "px"; m.style.top = y + "px";
  m.style.transformOrigin = `${anchor instanceof Element ? "left" : "left"} ${oy}`;
  m.onclick = (e) => {
    const b = e.target.closest("button[data-a]");
    if (!b) return;
    if (b.classList.contains("off")) { if (b.title) toast(b.title, 3200); return; }
    closeMenu(); onClick(b.dataset.a, b.dataset, b);
  };
  m.onkeydown = (e) => {
    const items = $$("button.mi", m);
    let k = items.indexOf($("button.mi.hl", m));
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      k = e.key === "ArrowDown" ? (k + 1) % items.length : (k - 1 + items.length) % items.length;
      items.forEach((b, j) => b.classList.toggle("hl", j === k)); items[k]?.focus();
    }
  };
  const inp = $("input", m);
  setTimeout(() => (inp || $("button.mi", m))?.focus({ preventScroll: true }), 0);
  return m;
}
function closeMenu() {
  const m = $("#menu");
  if (m.hidden || m.classList.contains("closing")) return;
  m.classList.add("closing");
  m._t = setTimeout(() => { m.hidden = true; m.classList.remove("closing"); }, 100);
}
const menuOpen = () => !$("#menu").hidden && !$("#menu").classList.contains("closing");
document.addEventListener("mousedown", (e) => {
  if (!e.target.closest("#menu")) closeMenu();
  if (S.pop && !e.target.closest("#pop") && !e.target.closest(".w.low, .w.weak") && !e.target.closest("#menu")) closePop();
});

// ---- segment / word context menu
function segMenu(at, segId, w) {
  const s = segById(segId);
  const others = spkList().filter(([id]) => id !== s.speaker).map(([id, sp]) => ({ a: "split-to", label: sp.name, dot: spkColor(id), data: { spk: id } }));
  const items = [];
  if (isGold()) items.push(s.verified ? { a: "unver", label: "Снять отметку «проверено»", kb: "Shift+Enter" } : { a: "ver", label: "Верно на слух", kb: "Enter" },
    { a: "inaudible", label: "Не разобрать весь фрагмент (???)", kb: "Ctrl+/" }, "-");
  if (w) items.push({ a: "word", label: "Прослушать слово" }, { a: "from", label: "Играть отсюда", kb: "Ctrl+клик" });
  else items.push({ a: "seg-play", label: "Прослушать фрагмент", kb: isGold() ? "R" : "" });
  items.push({ a: "edit", label: "Править фрагмент", kb: isGold() ? "E" : "Enter" }, "-");
  if (w && w.wi > 0) {
    items.push({ a: "split", label: "Разделить перед словом" }, { a: "split-new", label: "С этого слова — новый спикер" });
    if (others.length) items.push({ kind: "label", label: "С этого слова говорит" }, ...others);
    items.push("-");
  }
  items.push({ a: "merge", label: "Объединить со следующим" }, { a: "del", label: "Удалить фрагмент", cls: "danger" }, "-");
  items.push({ a: "copy", label: `Скопировать время ${clock(w ? w.s : s.start, true)}` });
  openMenu(at, menuItems(items), (a, ds) => {
    const base = `/api/docs/${S.doc.id}/segments/${segId}`;
    if (a === "ver" || a === "unver") setVerified(segId, a === "ver");
    if (a === "inaudible") markInaudible(segId);
    if (a === "word") playRange(w.s, w.e);
    if (a === "from") playFrom(w.s);
    if (a === "seg-play") playRange(s.start, s.end);
    if (a === "edit") startEdit(segId);
    if (a === "split") apply(api(base + "/split", { method: "POST", body: { word_index: w.wi } }));
    if (a === "split-new") apply(api(base + "/split", { method: "POST", body: { word_index: w.wi, new_speaker: "new" } }));
    if (a === "split-to") apply(api(base + "/split", { method: "POST", body: { word_index: w.wi, new_speaker: ds.spk } }));
    if (a === "merge") mergeNext(segId);
    if (a === "del") deleteSeg(segId);
    if (a === "copy") navigator.clipboard?.writeText(clock(w ? w.s : s.start, true)).then(() => toast("Скопировано"));
  });
}

// ---- speaker reassignment popover (§4.3): click on a name
async function assignSpeaker(segIds, spk) {
  if (!segIds.length) return;
  let doc;
  for (const sid of segIds) {
    doc = await api(`/api/docs/${S.doc.id}/segments/${sid}/speaker`, { method: "POST", body: spk ? { speaker: spk } : { speaker: null } }).catch((e) => { fail(e); return null; });
    if (!doc) break;
  }
  if (doc) apply(Promise.resolve(doc));
}

function speakerPop(anchor, spk, blockKey) {
  const sp = S.doc.speakers[spk];
  const keys = S.selBlocks.has(blockKey) && S.selBlocks.size > 1 ? S.selBlocks : new Set([blockKey]);
  const segIds = segsOfBlocks(keys);
  const list = spkList().map(([id, x], k) => ({ a: "assign", label: x.name, dot: spkColor(id), num: k < 9 ? k + 1 : "", data: { spk: id }, cls: id === spk ? "checked" : "" }));
  const m = openMenu(anchor, `<input class="m-input" id="spk-name" value="${esc(sp.name)}" placeholder="Имя спикера" title="Enter — сохранить">`
    + menuItems(["-", { kind: "label", label: keys.size > 1 ? `Эти реплики (${keys.size}) —` : "Эта реплика —" }, ...list, { a: "assign-new", label: "Новый спикер" }]),
  (a, ds) => {
    if (a === "assign" && ds.spk !== spk) assignSpeaker(segIds, ds.spk);
    if (a === "assign-new") assignSpeaker(segIds, null);
  });
  const inp = $("#spk-name", m);
  inp.select();
  inp.addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
      const name = inp.value.trim();
      closeMenu();
      if (name && name !== sp.name) apply(api(`/api/docs/${S.doc.id}/speakers/${spk}`, { method: "POST", body: { name } }));
    }
    if (e.key === "Escape") closeMenu();
    if (/^[1-9]$/.test(e.key) && e.altKey) { e.preventDefault(); const it = spkList()[+e.key - 1]; if (it) { closeMenu(); assignSpeaker(segIds, it[0]); } }
  });
}

// ================================================================== review: doubtful words (§4.4, §4.5)

function lcsPairs(a, b) {
  const n = a.length, m = b.length;
  if (!n || !m) return [];
  const dp = Array.from({ length: n + 1 }, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--)
    dp[i][j] = a[i] && a[i] === b[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  const out = [];
  let i = 0, j = 0;
  while (i < n && j < m) {
    if (a[i] && a[i] === b[j]) { out.push([i, j]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) i++; else j++;
  }
  return out;
}

const segTokens = (s) => (s.words?.length ? s.words.map((w) => w.w) : s.text.split(/\s+/).filter(Boolean));
const altTokens = (t) => String(t || "").split(/\s+/).filter(Boolean);

// The alternative texts (seg.alts = [{model, text}]) are hypotheses of other ensemble models for
// the whole segment. For one word: align by LCS of normalized tokens, take the stretch of the
// alternative between the nearest matched neighbours. Returns {agree:[models], vars:[...]}.
function wordVariants(seg, wi) {
  const A = segTokens(seg), an = A.map(norm);
  const agree = [], byKey = new Map();
  for (const alt of seg.alts || []) {
    const B = altTokens(alt.text), bn = B.map(norm);
    if (!B.length) continue;
    const mapA = new Array(A.length).fill(-1);
    for (const [i, j] of lcsPairs(an, bn)) mapA[i] = j;
    let v;
    if (mapA[wi] >= 0) { agree.push(alt.model); continue; }
    let a = wi - 1; while (a >= 0 && mapA[a] < 0) a--;
    let b = wi + 1; while (b < A.length && mapA[b] < 0) b++;
    const ja = a >= 0 ? mapA[a] : -1, jb = b < A.length ? mapA[b] : B.length;
    const tokens = B.slice(ja + 1, jb);
    if (b - (a + 1) <= 3 && tokens.length <= 5) v = { from: a + 1, to: b, tokens };
    else v = { from: 0, to: A.length, tokens: B, phrase: true };
    if (!v.phrase && tokens.map(norm).join(" ") === an.slice(v.from, v.to).join(" ")) { agree.push(alt.model); continue; }
    const key = `${v.from}-${v.to}-${v.tokens.map(norm).join(" ")}`;
    const prev = byKey.get(key);
    if (prev) { if (!prev.models.includes(alt.model)) prev.models.push(alt.model); }
    else byKey.set(key, { ...v, models: [alt.model], B });
  }
  const vars = [...byKey.values()].map((v) => {
    if (v.phrase) {
      const keep = new Set(lcsPairs(an, v.B.map(norm)).map(([, j]) => j));
      v.html = v.B.map((t, j) => (keep.has(j) ? esc(t) : `<mark>${esc(t)}</mark>`)).join(" ");
    } else {
      v.html = v.tokens.length ? esc(v.tokens.join(" ")) : "<i>без этого слова</i>";
      if (v.to - v.from > 1) v.html += ` <span class="mi-sub">вместо «${esc(A.slice(v.from, v.to).join(" "))}»</span>`;
    }
    return v;
  });
  return { agree, vars, A };
}

// whole-segment alternatives with differences marked (inspector)
function altDiffs(seg) {
  const an = segTokens(seg).map(norm);
  return (seg.alts || []).map((alt) => {
    const B = altTokens(alt.text);
    const keep = new Set(lcsPairs(an, B.map(norm)).map(([, j]) => j));
    return { model: alt.model, text: alt.text, html: B.map((t, j) => (keep.has(j) ? esc(t) : `<mark>${esc(t)}</mark>`)).join(" "), same: keep.size === B.length && B.length === an.length };
  });
}

function goReview(dir) {
  const list = S.review;
  if (!list.length) return toast("Сомнительных слов нет");
  if (!S.reviewOn) setReview(true);
  const cur = S.revT ?? (dir > 0 ? -Infinity : Infinity);
  let it = dir > 0 ? list.find((x) => x.t > cur + 1e-3) : [...list].reverse().find((x) => x.t < cur - 1e-3);
  if (!it) it = dir > 0 ? list[0] : list[list.length - 1];
  S.revT = it.t;
  const el = it.k === "w" ? $(`.w[data-i="${it.i}"]`) : $(`.seg[data-id="${it.id}"]`);
  if (!el) return;
  scrollToEl(el);
  playRange(Math.max(0, it.t - 0.4), it.e + 0.4);
  if (it.k === "w") { selectWord(it.i); openWordPop(it.i); }
  else { select({ k: "seg", id: it.id }); openSegPop(it.id); }
}

function scrollToEl(el) {
  const sc = $("#scroller"), r = el.getBoundingClientRect(), sr = sc.getBoundingClientRect();
  const mid = sr.top + (sr.height + 52 - 72) / 2;
  const dy = r.top + r.height / 2 - mid;
  if (Math.abs(dy) < 4) return;
  const near = Math.abs(dy) < sr.height * 1.5 && !matchMedia("(prefers-reduced-motion: reduce)").matches;
  sc.scrollTo({ top: sc.scrollTop + dy, behavior: near ? "smooth" : "instant" });
}

// ---- popover
function closePop() {
  const p = $("#pop");
  if (!S.pop) return;
  S.pop = null;
  p.classList.add("closing");
  clearTimeout(p._t); p._t = setTimeout(() => { if (!S.pop) { p.hidden = true; p.classList.remove("closing"); } }, 100);
}

function showPop(html, anchorEl) {
  const p = $("#pop");
  clearTimeout(p._t);
  p.classList.remove("closing");
  p.innerHTML = html; p.hidden = false;
  p.style.animation = "none"; p.offsetWidth; p.style.animation = "";
  positionPop(anchorEl);
}
function popAnchor() {
  if (!S.pop) return null;
  return S.pop.type === "word" ? $(`.w[data-i="${S.pop.i}"]`) : $(`.seg[data-id="${S.pop.id}"]`);
}
function positionPop(anchorEl = popAnchor()) {
  const p = $("#pop");
  if (!S.pop || !anchorEl) return;
  const rects = anchorEl.getClientRects();
  const a = rects[0] || anchorEl.getBoundingClientRect();
  const r = p.getBoundingClientRect();
  let x = a.left + a.width / 2 - r.width / 2, y = a.bottom + 6, oy = "top";
  if (y + r.height > innerHeight - 8) { y = a.top - r.height - 6; oy = "bottom"; }
  x = Math.max(8, Math.min(x, innerWidth - r.width - 8));
  y = Math.max(8, Math.min(y, innerHeight - r.height - 8));
  p.style.left = x + "px"; p.style.top = y + "px";
  p.style.transformOrigin = `${Math.round(a.left + a.width / 2 - x)}px ${oy}`;
}
$("#scroller").addEventListener("scroll", () => { if (S.pop) positionPop(); }, { passive: true });

function popItemsHtml() {
  const P = S.pop;
  return P.items.map((it, k) => {
    if (it === "-") return "<hr>";
    const cls = ["mi", it.row ? "vrow" : "", it.phrase ? "phrase" : "", k === P.hl ? "hl" : "", it.cls || ""].filter(Boolean).join(" ");
    return `<button class="${cls}" data-k="${k}">${it.num ? `<span class="num">${it.num}</span>` : ""}<span class="v-text">${it.html ?? esc(it.label)}</span>`
      + (it.kb ? `<span class="kb">${esc(it.kb)}</span>` : it.subHtml ? `<span class="mi-sub">${it.subHtml}</span>` : it.sub ? `<span class="mi-sub">${esc(it.sub)}</span>` : "") + `</button>`;
  }).join("");
}

function openWordPop(i, mode = null) {
  const x = S.words[i];
  if (!x) return;
  const seg = segById(x.seg), w = seg.words?.[x.wi] || { w: x.w, s: x.s, e: x.e, p: x.p };
  const { agree, vars } = wordVariants(seg, x.wi);
  const items = [];
  items.push({ row: true, num: 1, label: w.w, subHtml: agree.length ? `текущий · ${mnames(agree)}` : "текущий вариант", act: "ok" });
  vars.slice(0, 8).forEach((v, k) => items.push({ row: true, num: k + 2, html: v.html, subHtml: mnames(v.models), phrase: v.phrase, act: "var", v }));
  items.push("-");
  const okIdx = items.length;
  items.push({ label: "Верно", kb: "Enter", act: "ok" }, { label: "Исправить вручную…", kb: "E", act: "edit" },
    { label: "Добавить в словарь…", kb: "D", act: "dict" }, { label: "Прослушать ещё раз", kb: "Пробел", act: "replay" });
  S.pop = { type: "word", i, seg: seg.id, wi: x.wi, word: w, items, hl: okIdx, mode };
  renderWordPop();
}

function renderWordPop() {
  const P = S.pop, w = P.word;
  const head = `<div class="pop-head"><b>«${esc(w.w)}»</b><span><span class="${w.p < WEAK_P ? "low" : ""}">${Math.round((w.p ?? 1) * 100)} %</span> · ${clock(w.s, true)}</span></div><hr>`;
  let body;
  if (P.mode === "edit") {
    body = `<div class="pop-form"><label for="pop-edit">Как правильно</label><input class="field" id="pop-edit" value="${esc(w.w)}" spellcheck="true">
      <button class="btn primary compact" data-f="edit">Исправить</button></div>`;
  } else if (P.mode === "dict") {
    const from = w.w.replace(/^[^\p{L}\p{N}]+|[^\p{L}\p{N}]+$/gu, "");
    body = `<div class="pop-form"><label for="pop-from">Как распознано</label><input class="field" id="pop-from" value="${esc(from)}" spellcheck="false">
      <label for="pop-to">Как правильно</label><input class="field" id="pop-to" value="" spellcheck="false">
      <button class="btn primary compact" data-f="dict">Добавить замену</button></div>`;
  } else body = popItemsHtml();
  showPop(head + body, popAnchor());
  const f = $("#pop input");
  if (f) { f.focus(); if (P.mode === "edit") f.select(); else $("#pop-to")?.focus(); }
}

function openSegPop(id) {
  const s = segById(id);
  if (!s) return;
  const why = (s.flags || []).map((f) => FLAG_TEXT[f] || f).join(", ");
  const items = [{ label: "Оставить", kb: "Enter", act: "keep" }, { label: "Удалить фрагмент", kb: "Del", act: "del", cls: "danger" },
    { label: "Прослушать ещё раз", kb: "Пробел", act: "replay" }];
  S.pop = { type: "seg", id, items, hl: 0 };
  const note = onlyDisagree(s) ? "Каждая модель услышала своё: текст может быть неверным. Прослушайте и исправьте или удалите."
    : `${esc(why)}. Whisper мог выдумать текст на тишине или музыке.`;
  showPop(`<div class="pop-head"><b>${flagTitle(s)}</b><span>${clock(s.start, true)}</span></div><div class="m-note">${note}</div><hr>${popItemsHtml()}`, popAnchor());
}

function popReplay() {
  const P = S.pop;
  if (!P) return;
  if (P.type === "word") playRange(Math.max(0, P.word.s - 0.4), P.word.e + 0.4);
  else { const s = segById(P.id); if (s) playRange(s.start, s.end); }
}

async function popActivate(k, thenNext = false) {
  const P = S.pop;
  if (!P) return;
  const it = P.items[k];
  if (!it || it === "-") return;
  if (it.act === "replay") return popReplay();
  if (it.act === "edit" || it.act === "dict") { P.mode = it.act; return renderWordPop(); }
  closePop();
  if (P.type === "seg") {
    if (it.act === "keep") await keepSeg(P.id);
    if (it.act === "del") await deleteSeg(P.id);
  } else if (it.act === "ok") {
    addOk([okKey(P.seg, P.word, P.wi)]);
    renderTranscript(); renderInspector();
  } else if (it.act === "var") {
    const seg = segById(P.seg), A = segTokens(seg), v = it.v;
    const text = [...A.slice(0, v.from), ...v.tokens, ...A.slice(v.to)].join(" ");
    await setSegText(P.seg, text);
  }
  if (thenNext) goReview(1);
}

async function popSubmit(kind) {
  const P = S.pop;
  if (!P) return;
  if (kind === "edit") {
    const val = $("#pop-edit").value.trim();
    closePop();
    if (val === P.word.w) { addOk([okKey(P.seg, P.word, P.wi)]); renderTranscript(); return; }
    const A = segTokens(segById(P.seg));
    await setSegText(P.seg, [...A.slice(0, P.wi), ...(val ? [val] : []), ...A.slice(P.wi + 1)].join(" "));
  }
  if (kind === "dict") {
    const from = $("#pop-from").value.trim(), to = $("#pop-to").value.trim();
    if (!from || !to) return $(to ? "#pop-from" : "#pop-to").focus();
    try {
      let saved, replace;
      for (let i = 0; ; i++) {   // an agent may save the dictionary in between: then read it again
        await loadDict();
        if (!S.dictOk) throw new Error("сервер не отдаёт словарь");
        replace = (S.dict.replace || []).filter((r) => r.from !== from).concat([{ from, to }]);
        try { saved = await api("/api/dictionary", { method: "PUT", body: { ...S.dict, replace } }); break; }
        catch (e) { if (e.status !== 409 || i >= 2) throw e; }
      }
      S.dict = { ...DICT_EMPTY, ...(saved || {}), replace: saved?.replace || replace };
      renderDictLine(); renderNav();
      closePop();
      toast(`Замена «${from}» → «${to}» добавлена в словарь`, 4000, { label: "Применить к записи", run: applyDictionary });
    } catch (e) { fail(e); }
  }
}

$("#pop").addEventListener("click", (e) => {
  const f = e.target.closest("[data-f]");
  if (f) return popSubmit(f.dataset.f);
  const b = e.target.closest("button[data-k]");
  if (b) popActivate(+b.dataset.k);
});
$("#pop").addEventListener("mousemove", (e) => {
  const b = e.target.closest("button[data-k]");
  if (b && S.pop && S.pop.hl !== +b.dataset.k) { S.pop.hl = +b.dataset.k; $$("#pop .mi").forEach((x) => x.classList.toggle("hl", x === b)); }
});

function popKey(e) {
  const P = S.pop;
  if (!P) return false;
  const inForm = e.target.closest?.("#pop input");
  if (inForm) {
    if (e.key === "Escape") { e.preventDefault(); P.mode = null; renderWordPop(); return true; }
    if (e.key === "Enter") { e.preventDefault(); popSubmit(P.mode); return true; }
    if (e.key === "Tab" && P.mode === "dict") return false;
    return e.key !== "F3";
  }
  if (e.ctrlKey || e.metaKey) return false;
  const sel = P.items.map((x, k) => (x === "-" ? -1 : k)).filter((k) => k >= 0);
  if (e.key === "Escape") { e.preventDefault(); closePop(); return true; }
  if (e.key === "ArrowDown" || e.key === "ArrowUp") {
    e.preventDefault();
    const pos = sel.indexOf(P.hl);
    P.hl = sel[(pos + (e.key === "ArrowDown" ? 1 : sel.length - 1)) % sel.length];
    $$("#pop .mi").forEach((x) => x.classList.toggle("hl", +x.dataset.k === P.hl));
    return true;
  }
  if (e.key === "Enter") { e.preventDefault(); popActivate(P.hl); return true; }
  if (e.code === "Space") { e.preventDefault(); popReplay(); return true; }
  if (e.key === "F3") { e.preventDefault(); if (e.shiftKey) goReview(-1); else popActivate(P.hl, true); return true; }
  if (P.type === "word") {
    if (/^[1-9]$/.test(e.key) && !e.ctrlKey && !e.altKey) {
      const k = P.items.findIndex((x) => x.row && x.num === +e.key);
      if (k >= 0) { e.preventDefault(); popActivate(k); }
      return true;
    }
    if (e.code === "KeyE") { e.preventDefault(); P.mode = "edit"; renderWordPop(); return true; }
    if (e.code === "KeyD") { e.preventDefault(); P.mode = "dict"; renderWordPop(); return true; }
  } else if (e.key === "Delete") { e.preventDefault(); popActivate(1); return true; }
  return false;
}

// ================================================================== inspector

$("#insp-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) { S.tab = b.dataset.v; renderInspector(); } });

function renderInspector() {
  $$("#insp-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.v === S.tab));
  const body = $("#insp-body"), d = S.doc;
  if (!d) { body.innerHTML = ""; return; }
  if (S.tab === "spk") body.innerHTML = inspSpeakers();
  else if (S.tab === "file") body.innerHTML = inspFile();
  else body.innerHTML = inspFragment();
}

function inspFragment() {
  const d = S.doc;
  if (d.status !== "done") return `<div class="insp-empty">Запись ещё обрабатывается</div>`;
  const sel = S.sel;
  const seg = sel?.k === "w" ? segById(sel.seg) : sel?.k === "seg" ? segById(sel.id) : null;
  if (!seg && isGold(d)) return `<div class="insp-empty">Enter или F3 — к первому непроверенному фрагменту</div>${goldKeysHtml()}`;
  if (!seg) return `<div class="insp-empty">Выберите слово или реплику</div>`;
  let html = "";
  if (sel.k === "w") {
    const w = seg.words?.[sel.wi];
    if (w) {
      const x = S.words[sel.i];
      html += `<div class="insp-sec"><div class="insp-cap">Слово</div><div class="insp-word">${esc(w.w)}</div>
        <div class="insp-meta"><span>${clock(w.s, true)}–${clock(w.e, true)}</span><span class="${w.p < WEAK_P ? "low" : ""}">${Math.round((w.p ?? 1) * 100)} %</span>${w.edited ? "<span>исправлено</span>" : ""}</div>
        <div class="insp-actions"><button class="btn" data-a="play-word">Прослушать</button>${x?.low ? `<button class="btn" data-a="ok-word">Верно</button>` : ""}</div></div>`;
    }
  }
  const sp = d.speakers?.[seg.speaker];
  if (isGold(d)) html += inspGold(seg);
  html += `<div class="insp-sec"><div class="insp-cap">Реплика${sp ? ` · ${esc(sp.name)}` : ""}</div><div class="insp-text">${esc(seg.text)}</div>
    <div class="insp-meta"><span>${clock(seg.start, true)}–${clock(seg.end, true)}</span>${seg.ai ? "<span>✦ исправлено ИИ</span>" : seg.edited ? "<span>изменено</span>" : ""}</div>
    <div class="insp-actions"><button class="btn" data-a="play-seg">Прослушать</button><button class="btn" data-a="edit">Править</button>
    <button class="btn" data-a="merge">Объединить со следующим</button><button class="btn danger" data-a="del">Удалить</button></div></div>`;
  if (goldQuiet()) return html;          // no flags and variants: they anchor the listener to the models' guesses
  if (seg.flags?.length) html += `<div class="insp-sec"><div class="insp-cap">${flagTitle(seg)}</div>
    <div class="flag-why">${esc(seg.flags.map((f) => FLAG_TEXT[f] || f).join(", "))}</div>
    <div class="insp-actions"><button class="btn" data-a="keep">Оставить</button><button class="btn danger" data-a="del">Удалить</button></div></div>`;
  const alts = altDiffs(seg);
  if (alts.length) html += `<div class="insp-sec"><div class="insp-cap">Варианты других моделей</div>`
    + alts.map((a, k) => `<button class="alt" data-a="alt" data-k="${k}" title="${a.same ? "Совпадает с текущим текстом" : "Принять этот вариант"}"><div class="alt-model"><span>${esc(mname(a.model))}</span><span>${k < 4 ? `Alt+${k + 1}` : ""}</span></div><div class="alt-text">${a.html}</div></button>`).join("")
    + `</div>`;
  return html;
}

// gold doc: the fragment's check state, its window of the source recording, the keys of the loop
function inspGold(seg) {
  const pieces = goldPieces(), k = pieceOf(pieces)(seg), p = pieces[k];
  return `<div class="insp-sec"><div class="insp-cap">Эталон</div>
    <div class="g-state ${seg.verified ? "ok" : ""}">${seg.verified ? "Проверено на слух" : "Ещё не проверено"}</div>
    ${p ? `<div class="insp-meta"><span>${esc(p.label || "Фрагмент")}</span><span title="${esc(p.source || "")}">${esc(fname(p.source))}${Number.isFinite(+p.start) ? ` · ${clock(p.start + (seg.start - p.at))}` : ""}</span></div>` : ""}
    <div class="insp-actions">${seg.verified ? `<button class="btn" data-a="unver">Снять отметку</button>` : `<button class="btn" data-a="ver">Верно на слух</button>`}
    <button class="btn" data-a="inaudible" title="Весь фрагмент не разобрать: текст станет «???», при оценке он пропускается">Не разобрать</button></div></div>${goldKeysHtml()}`;
}
function goldKeysHtml() {
  const k = [["Enter", "верно, к следующему"], ["E", "исправить"], ["R", "прослушать ещё раз"], ["↑ ↓", "соседний фрагмент"],
    ["Ctrl+/", "не разобрать: ???"], ["Shift+Enter", "снять отметку"], ["F3", "к непроверенному"]];
  return `<div class="insp-sec"><div class="insp-cap">Проверка на слух</div><dl class="kv keys">${k.map(([a, b]) => `<dt>${a}</dt><dd>${b}</dd>`).join("")}</dl></div>`;
}

function speakerTimes() {
  const t = {};
  for (const s of S.doc.segments) if (s.speaker) t[s.speaker] = (t[s.speaker] || 0) + (s.end - s.start);
  return t;
}
function inspSpeakers() {
  const d = S.doc, t = speakerTimes();
  if (d.status !== "done") return `<div class="insp-empty">Запись ещё обрабатывается</div>`;
  const rows = spkList().map(([id, sp]) => `<div class="spk-row" data-spk="${id}">
    <button class="swatch" data-a="color" title="Цвет"><span class="dot" style="--c:${spkColor(id)}"></span></button>
    <input value="${esc(sp.name)}" data-a="name" spellcheck="false" aria-label="Имя спикера">
    <span class="spk-time">${clock(t[id] || 0)}</span>
    <button class="icon-btn" data-a="spk-more" title="Ещё" aria-label="Ещё"><svg class="ic"><use href="icons.svg#more"/></svg></button></div>`).join("");
  return `<div class="insp-sec">${rows || `<div class="insp-empty">Спикеры не размечены</div>`}</div>
    <div class="insp-sec"><button class="btn" data-a="rediarize">Разделить спикеров заново…</button></div>`;
}
function inspFile() {
  const d = S.doc, m = d.meta || {}, o = d.options || {};
  const kv = (k, v) => (v ? `<dt>${esc(k)}</dt><dd>${v}</dd>` : "");
  const created = d.created ? new Date(d.created).toLocaleString("ru-RU", { dateStyle: "medium", timeStyle: "short" }) : "";
  const params = [
    o.quality ? (o.quality === "fast" ? "быстро" : "максимальная точность") : "",
    o.profile ? ({ balanced: "универсальный", clean: "чистая речь", noisy: "совещание, шум" }[o.profile] || o.profile) : "",
    o.diarize === false ? "без спикеров" : o.num_speakers ? `спикеров: ${o.num_speakers}` : "спикеры: авто",
    o.mixed || o.extra?.multilingual ? "смешанная речь" : "", o.stereo_split ? "стерео = спикеры" : "", o.highpass ? "фильтр 80 Гц" : "",
    o.beam_size ? `beam ${o.beam_size}` : "",
  ].filter(Boolean).join(", ");
  return `<div class="insp-sec"><dl class="kv">
    ${kv("Длительность", d.duration ? clock(d.duration) : "")}
    ${kv("Язык", d.language ? esc(LANGS[d.language] || d.language) : "")}
    ${kv("Модель", esc(mname(m.model || o.model || "")) + (m.members?.length ? `<br><span class="kv-sub">${esc(m.members.map(mname).join(", "))}</span>` : ""))}
    ${kv("Таймкоды", d.status === "done" ? (m.aligned ? "wav2vec2" : "Whisper") : "")}
    ${kv("Устройство", m.device ? (m.device === "cuda" ? "GPU" : "CPU") : "")}
    ${kv("Создано", esc(created))}
    ${kv("Исходный файл", d.source_path ? `${esc(d.source_path)}<br><button class="text-btn" data-a="reveal">${window.pywebview?.api?.reveal_file ? "Показать в проводнике" : "Скопировать путь"}</button>` : "")}
    ${kv("Параметры", esc(params))}
    ${kv("Тема", esc(o.topic || ""))}
    ${kv("Доп. слова", esc(o.glossary || ""))}
    ${kv("Ошибка", d.error ? `<span style="color:var(--danger)">${esc(d.error)}</span>` : "")}
  </dl></div><div class="insp-sec"><button class="btn" data-a="redo">Распознать заново…</button></div>`;
}

$("#insp-body").addEventListener("click", (e) => {
  const b = e.target.closest("[data-a]");
  if (!b || b.tagName === "INPUT") return;
  const a = b.dataset.a, sel = S.sel;
  const segId = sel?.k === "w" ? sel.seg : sel?.id;
  const seg = segId && segById(segId);
  if (a === "play-word") { const w = seg?.words?.[sel.wi]; if (w) playRange(w.s, w.e); }
  if (a === "ok-word") { const w = seg?.words?.[sel.wi]; if (w) { addOk([okKey(seg.id, w, sel.wi)]); renderTranscript(); renderInspector(); } }
  if (a === "play-seg" && seg) playRange(seg.start, seg.end);
  if ((a === "ver" || a === "unver") && seg) setVerified(seg.id, a === "ver");
  if (a === "inaudible" && seg) markInaudible(seg.id);
  if (a === "edit" && seg) startEdit(seg.id);
  if (a === "merge" && seg) mergeNext(seg.id);
  if (a === "del" && seg) { S.sel = null; deleteSeg(seg.id); }
  if (a === "keep" && seg) keepSeg(seg.id);
  if (a === "alt" && seg) acceptAlt(seg, +b.dataset.k);
  if (a === "rediarize") rediarizeMenu(b);
  if (a === "redo") redoDoc(S.doc);
  if (a === "reveal") revealPath(S.doc.source_path);
  const row = b.closest(".spk-row");
  if (row && a === "color") colorMenu(b, row.dataset.spk);
  if (row && a === "spk-more") spkMoreMenu(b, row.dataset.spk);
});
$("#insp-body").addEventListener("keydown", (e) => {
  const inp = e.target.closest(".spk-row input");
  if (!inp) return;
  if (e.key === "Enter") { e.preventDefault(); inp.blur(); }
  if (e.key === "Escape") { inp.value = S.doc.speakers[inp.closest(".spk-row").dataset.spk]?.name || ""; inp.blur(); }
});
$("#insp-body").addEventListener("focusout", (e) => {
  const inp = e.target.closest(".spk-row input");
  if (!inp) return;
  const spk = inp.closest(".spk-row").dataset.spk, name = inp.value.trim(), sp = S.doc.speakers[spk];
  if (sp && name && name !== sp.name) apply(api(`/api/docs/${S.doc.id}/speakers/${spk}`, { method: "POST", body: { name } }));
});

function acceptAlt(seg, k) {
  const a = (seg.alts || [])[k];
  if (!a || a.text.trim() === seg.text.trim()) return;
  setSegText(seg.id, a.text.replace(/\s+/g, " ").trim()).then((d) => d && toast(`Принят вариант ${a.model} · Ctrl+Z — отменить`));
}

function colorMenu(anchor, spk) {
  const cur = spkIdx(spk);
  const html = `<div class="swatches">${PAL_LIGHT.map((c, k) => `<button class="${k === cur ? "cur" : ""}" data-a="c" data-k="${k}" title="Цвет ${k + 1}" style="--c:var(--spk-${k + 1})"><span class="dot"></span></button>`).join("")}</div>`;
  openMenu(anchor, html, (a, ds) => {
    const sp = S.doc.speakers[spk];
    apply(api(`/api/docs/${S.doc.id}/speakers/${spk}`, { method: "POST", body: { name: sp.name, color: PAL_LIGHT[+ds.k] } }));
  });
}
function spkMoreMenu(anchor, spk) {
  const others = spkList().filter(([id]) => id !== spk);
  openMenu(anchor, menuItems([
    { a: "merge", label: "Объединить с…", off: !others.length, title: others.length ? "" : "Других спикеров нет" },
    { a: "listen", label: "Прослушать первую реплику" },
  ]), (a) => {
    if (a === "listen") { const s = S.doc.segments.find((x) => x.speaker === spk); if (s) playRange(s.start, s.end); }
    if (a === "merge") openMenu(anchor, menuItems([{ kind: "label", label: `Объединить «${S.doc.speakers[spk].name}» с` },
      ...others.map(([id, sp]) => ({ a: "into", label: sp.name, dot: spkColor(id), data: { spk: id } }))]),
    (_, ds) => apply(api(`/api/docs/${S.doc.id}/speakers/${spk}/merge`, { method: "POST", body: { into: ds.spk } }), "Спикеры объединены"), { align: "end" });
  }, { align: "end" });
}

// ================================================================== document jobs

async function refreshStatus() {
  try { S.status = await api("/api/status"); renderBadge(); } catch {}
  return S.status;
}
function renderBadge() {
  const st = S.status;
  if (!st) return;
  $("#device-badge").textContent = st.device === "cuda" ? "GPU" : "CPU";
  $("#device-badge").title = `Вычисления: ${st.device} / ${st.compute_type}\nДанные: ${st.home}`;
}

function rediarizeMenu(anchor) {
  if (!S.doc || S.doc.status !== "done") return;
  const cur = Object.keys(S.doc.speakers || {}).length;
  openMenu(anchor, `<div class="m-label">Сколько спикеров в записи?</div>
    <div class="m-seg">${[[0, "Авто"], [2, "2"], [3, "3"], [4, "4"]].map(([n, l]) => `<button class="mi" data-a="n" data-n="${n}">${l}</button>`).join("")}</div>
    <div class="m-note">Сейчас ${cur} ${plural(cur, "спикер", "спикера", "спикеров")}. Спикеры будут размечены заново.</div>`,
  (a, ds) => runJob("rediarize", { num_speakers: +ds.n }, "Пересчитываю спикеров…"), { align: "end" });
}

// rediarize / proofread return a summary; the document then goes through the usual processing poll.
async function runJob(kind, body, msg) {
  const id = S.doc.id;
  pause();
  try { await api(`/api/docs/${id}/${kind}`, { method: "POST", body: body ?? {} }); } catch (e) { return fail(e); }
  if (msg) toast(msg);
  await openDoc(id);
  refreshDocs();
}

async function applyDictionary() {
  if (!S.doc) return;
  let r;
  try { r = await api(`/api/docs/${S.doc.id}/apply-dictionary`, { method: "POST" }); } catch (e) { return fail(e); }
  const n = r?.changed ?? 0;
  apply(Promise.resolve(r?.doc || r), n ? `Словарь применён · ${n} ${plural(n, "исправление", "исправления", "исправлений")}` : "Словарь применён · исправлять нечего");
}

function exportMenu() {
  if (!S.doc || S.doc.status !== "done") return;
  openMenu($("#btn-export"), menuItems([
    { a: "txt", label: "Текст с таймкодами (.txt)" }, { a: "txt-plain", label: "Текст без таймкодов (.txt)" },
    { a: "md", label: "Markdown (.md)" }, { a: "srt", label: "Субтитры (.srt)" }, { a: "vtt", label: "Субтитры (.vtt)" },
    { a: "json", label: "JSON со словами (.json)" }, "-", { a: "copy", label: "Скопировать текст" },
  ]), async (a) => {
    const plain = a === "txt-plain", fmt = plain ? "txt" : a;
    try {
      if (a === "copy") { const t = await api(`/api/docs/${S.doc.id}/export/txt`); await navigator.clipboard.writeText(t); return toast("Текст скопирован"); }
      if (window.pywebview?.api) {
        const p = await window.pywebview.api.save_export(S.doc.id, fmt, S.doc.title, !plain);
        if (p) toast("Сохранено: " + p);
      } else {
        const t = await api(`/api/docs/${S.doc.id}/export/${fmt}?timestamps=${!plain}`);
        const a2 = document.createElement("a");
        a2.href = URL.createObjectURL(new Blob([t], { type: "text/plain" }));
        a2.download = `${S.doc.title}.${fmt}`; a2.click();
      }
    } catch (e) { fail(e); }
  }, { align: "end" });
}
$("#btn-export").addEventListener("click", exportMenu);

$("#btn-more").addEventListener("click", async (e) => {
  const anchor = e.currentTarget;
  if (!S.doc) return;
  const statusP = refreshStatus();
  const build = () => {
    const done = S.doc.status === "done";
    const wait = done ? {} : { off: true, title: "Дождитесь окончания обработки" };
    const llm = S.status?.llm, aiOk = done && llm?.available;
    const aiTitle = !done ? "Дождитесь окончания обработки"
      : !llm ? "Исправление с ИИ пока недоступно: сервер не сообщает о локальной модели"
      : !llm.available ? "Локальная ИИ-модель не запущена — исправление с ИИ недоступно"
      : `Исправить опечатки и ошибки распознавания · модель ${llm.model || "локальная"}`;
    const flagged = S.doc.segments.filter((s) => s.flags?.length && !onlyDisagree(s)).length;
    // a gold doc is corrected by ear only: no machine rewriting of its text (the server refuses it too)
    if (isGold()) return menuItems([
      { a: "redo", label: "Распознать заново…" },
      { a: "rediarize", label: "Пересчитать спикеров…", ...wait },
      "-",
      { a: "dict", label: "Применить словарь", off: true, title: "Эталон: только правки на слух", sub: "эталон" },
      { a: "ai", label: "✦ Исправить с ИИ", off: true, title: "Эталон: только правки на слух", sub: "эталон" },
      "-",
      { a: "undo", label: "Отменить правку", kb: "Ctrl+Z", ...wait },
      { a: "rename", label: "Переименовать", kb: "F2" },
      { a: "gold-off", label: "Убрать из эталона…" },
      "-",
      { a: "del", label: "Удалить транскрипцию…", cls: "danger" },
    ]);
    return menuItems([
      { a: "redo", label: "Распознать заново…" },
      { a: "rediarize", label: "Пересчитать спикеров…", ...wait },
      "-",
      { a: "dict", label: "Применить словарь", ...wait },
      { a: "ai", label: "✦ Исправить с ИИ", off: !aiOk, title: aiTitle, sub: done && !aiOk ? "недоступно" : "" },
      ...(flagged ? [{ a: "del-flags", label: `Удалить подозрительные фрагменты (${flagged})` }] : []),
      "-",
      { a: "undo", label: "Отменить правку", kb: "Ctrl+Z", ...wait },
      { a: "rename", label: "Переименовать", kb: "F2" },
      { a: "gold-on", label: "Сделать эталоном", ...wait, title: done ? "Проверить текст на слух и оценивать по нему модели" : wait.title },
      "-",
      { a: "del", label: "Удалить транскрипцию…", cls: "danger" },
    ]);
  };
  const onClick = async (a) => {
    if (a === "gold-on" || a === "gold-off") setGold(S.doc.id, a === "gold-on", S.doc.title);
    if (a === "redo") redoDoc(S.doc);
    if (a === "rediarize") rediarizeMenu(anchor);
    if (a === "dict") applyDictionary();
    if (a === "ai") runJob("proofread", {}, "Исправляю с ИИ…");
    if (a === "del-flags") apply(api(`/api/docs/${S.doc.id}/flagged/delete`, { method: "POST" }), "Подозрительные фрагменты удалены · Ctrl+Z — вернуть");
    if (a === "undo") undo();
    if (a === "rename") startRename();
    if (a === "del") deleteDoc(S.doc.id, S.doc.title);
  };
  openMenu(anchor, build(), onClick, { align: "end" });
  await statusP;     // the status (LLM availability) arrives in parallel; refresh the open menu
  if (menuOpen() && S.doc) { const m = $("#menu"); m.innerHTML = build(); }
});

function undo() { if (S.doc?.status === "done") apply(api(`/api/docs/${S.doc.id}/undo`, { method: "POST" }), "Отменено"); }

// ================================================================== find and replace

// a document opened from the library search: highlight the words in it, jump to the first one
function showFound(q) {
  toggleFind(true, true);
  $("#find-input").value = q; S.findIdx = 0;
  renderTranscript();
  const h = $("mark.hit"); if (h) scrollToEl(h);
  $("#find-input").blur();
}

function toggleFind(show = $("#findbar").hidden, silent = false) {
  if (show && S.doc?.status !== "done") return;
  $("#findbar").hidden = !show;
  $("#view-doc").classList.toggle("find-open", show);
  if (show) { $("#find-input").focus(); $("#find-input").select(); }
  else { $("#find-input").value = ""; $("#find-count").textContent = ""; if (!silent && S.doc?.status === "done") renderTranscript(); }
}
function updateFindCount() {
  const hits = $$("mark.hit");
  if (!hits.length) { $("#find-count").textContent = $("#find-input").value.trim() ? "Нет совпадений" : ""; return; }
  S.findIdx = Math.min(S.findIdx, hits.length - 1);
  hits.forEach((h, k) => h.classList.toggle("cur", k === S.findIdx));
  $("#find-count").textContent = `${S.findIdx + 1} из ${hits.length}`;
}
function findStep(dir) {
  const hits = $$("mark.hit");
  if (!hits.length) return;
  S.findIdx = (S.findIdx + dir + hits.length) % hits.length;
  updateFindCount();
  scrollToEl(hits[S.findIdx]);
}
$("#btn-find").addEventListener("click", () => toggleFind());
$("#btn-find-close").addEventListener("click", () => toggleFind(false));
$("#find-input").addEventListener("input", () => { S.findIdx = 0; renderTranscript(); const h = $("mark.hit"); if (h) scrollToEl(h); });
$("#findbar").addEventListener("keydown", (e) => {
  if (e.key === "Escape") { e.preventDefault(); toggleFind(false); }
  if (e.key === "Enter" && e.target.id === "find-input") { e.preventDefault(); findStep(e.shiftKey ? -1 : 1); }
  if (e.key === "Enter" && e.target.id === "replace-input") { e.preventDefault(); replaceOne(); }
});
async function replaceOne() {
  const find = $("#find-input").value.trim(), hit = $$("mark.hit")[S.findIdx];
  if (!find || !hit) return;
  const x = S.words[+hit.closest(".w").dataset.i], seg = segById(x.seg), A = segTokens(seg);
  const w = A[x.wi], k = w.toLowerCase().indexOf(find.toLowerCase());
  if (k < 0) return;
  A[x.wi] = w.slice(0, k) + $("#replace-input").value + w.slice(k + find.length);
  await setSegText(seg.id, A.join(" ").replace(/\s+/g, " ").trim());
}
$("#btn-replace-one").addEventListener("click", replaceOne);
$("#btn-replace").addEventListener("click", async () => {
  const find = $("#find-input").value.trim();
  if (!find) return;
  const r = await api(`/api/docs/${S.doc.id}/replace`, { method: "POST", body: { find, replace: $("#replace-input").value, whole_word: !/\s/.test(find) } }).catch(fail);
  if (r) apply(Promise.resolve(r.doc), `Заменено: ${r.count}`);
});

// ================================================================== player (§4.6)

let playerSeq = 0;
function destroyPlayer() {
  playerSeq++;                          // cancels any setupPlayer() still awaiting peaks
  if (S.ws) { try { S.ws.destroy(); } catch {} S.ws = null; }
  if (S.audio) { S.audio.pause(); S.audio.removeAttribute("src"); S.audio.load(); S.audio = null; }
  $("#wave").replaceChildren();         // never leave a stale waveform behind
  $("#strip").replaceChildren(); $("#ticks").replaceChildren();
  S.stopAt = null;
  document.body.classList.remove("playing");
}

// WaveSurfer draws nothing into a zero-width container and does not recover by itself,
// e.g. when processing finishes while the window is minimized. Wait for a real width,
// and redraw if the container ever comes back from zero width.
function waitForWidth(el) {
  if (el.clientWidth > 0) return Promise.resolve();
  return new Promise((done) => {
    const ro = new ResizeObserver(() => { if (el.clientWidth > 0) { ro.disconnect(); done(); } });
    ro.observe(el);
  });
}
let waveW = 0;
new ResizeObserver(() => {
  const w = $("#wave").clientWidth;
  if (w > 0 && waveW === 0 && S.ws) { try { S.ws.setOptions({}); } catch {} }
  waveW = w;
}).observe($("#wave"));

function waveColors() {
  const css = getComputedStyle(document.documentElement);
  const v = (k, d) => css.getPropertyValue(k).trim() || d;
  return { waveColor: v("--fill-3", "rgba(0,0,0,.11)"), progressColor: v("--text-2", "#5C5C63"), cursorColor: v("--accent", "#0071E3") };
}
function repaintWave() { requestAnimationFrame(() => { if (S.ws) try { S.ws.setOptions(waveColors()); } catch {} }); }

async function setupPlayer() {
  destroyPlayer();
  const seq = playerSeq, doc = S.doc;
  let peaks = [];
  try { peaks = await api(`/api/docs/${doc.id}/peaks`); } catch {}
  await waitForWidth($("#wave"));
  if (seq !== playerSeq || S.doc?.id !== doc.id) return;   // superseded while loading
  const old = $("#audio");
  const audio = document.createElement("audio");
  audio.id = "audio"; audio.preload = "auto";
  old.replaceWith(audio);
  S.audio = audio;
  audio.src = `/api/docs/${doc.id}/audio`;
  audio.playbackRate = S.rate;
  const h = parseInt(getComputedStyle(document.documentElement).getPropertyValue("--wave-h")) || 40;
  S.ws = WaveSurfer.create({
    container: "#wave", media: audio, peaks: peaks.length ? [peaks] : undefined, duration: doc.duration,
    height: h, barWidth: 2, barGap: 1, barRadius: 1, cursorWidth: 1.5, normalize: true, dragToSeek: true,
    ...waveColors(),
  });
  paintStrip(); paintTicks();
  $("#t-dur").textContent = clock(doc.duration);
  $("#t-cur").textContent = clock(0);
  const mine = () => S.audio === audio;
  audio.addEventListener("play", () => { if (!mine()) return; document.body.classList.add("playing"); $("#btn-play").title = "Пауза (Пробел)"; if (!S.ticking) { S.ticking = true; tick(); } });
  const stopped = () => { if (!mine()) return; document.body.classList.remove("playing"); $("#btn-play").title = "Воспроизвести (Пробел)"; };
  audio.addEventListener("pause", stopped);
  audio.addEventListener("ended", stopped);
  audio.addEventListener("seeked", () => { if (mine()) highlight(audio.currentTime, true); });
  // backup for tick(): requestAnimationFrame stops while the window is hidden
  audio.addEventListener("timeupdate", () => { if (mine() && S.stopAt != null && audio.currentTime >= S.stopAt) { audio.pause(); S.stopAt = null; } });
}

// speaker strip under the wave and doubtful-word ticks above it
function paintStrip() {
  const el = $("#strip"), d = S.doc;
  if (!d?.duration || d.status !== "done") return el.replaceChildren();
  el.innerHTML = blocks(d.segments).filter((b) => d.speakers?.[b.speaker]).map((b) => {
    const a = b.segs[0].start / d.duration, e = b.segs[b.segs.length - 1].end / d.duration;
    return `<i style="left:${(a * 100).toFixed(3)}%;width:calc(${((e - a) * 100).toFixed(3)}% - 1px);--c:${spkColor(b.speaker)}"></i>`;
  }).join("");
}
function paintTicks() {
  const el = $("#ticks"), d = S.doc;
  if (!d?.duration || (!S.reviewOn && !isGold(d))) return el.replaceChildren();
  // a gold doc: where the next window of a source recording starts
  const cuts = isGold(d) ? goldPieces(d).filter((p) => p.at > 0.5).map((p) => `<i class="cut" style="left:${(p.at / d.duration * 100).toFixed(3)}%"></i>`).join("") : "";
  el.innerHTML = cuts + (S.reviewOn ? S.review.map((x) => `<i style="left:${(x.t / d.duration * 100).toFixed(3)}%"></i>`).join("") : "");
}

$("#wave-wrap").addEventListener("mousemove", (e) => {
  const d = S.doc, tip = $("#wave-tip"), r = e.currentTarget.getBoundingClientRect();
  if (!d?.duration) return;
  const f = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)), t = f * d.duration;
  const s = d.segments.find((x) => x.start <= t && x.end >= t);
  const sp = s && d.speakers?.[s.speaker];
  const pc = isGold(d) ? goldPieces(d).filter((p) => p.at <= t).pop() : null;
  tip.innerHTML = `${sp ? `<span class="dot" style="--c:${spkColor(s.speaker)}"></span>${esc(sp.name)} · ` : ""}${pc?.label ? `${esc(pc.label)} · ` : ""}${clock(t)}`;
  tip.hidden = false;
  tip.style.left = Math.max(40, Math.min(r.width - 40, e.clientX - r.left)) + "px";
});
$("#wave-wrap").addEventListener("mouseleave", () => ($("#wave-tip").hidden = true));

function play() { S.audio?.play().catch(() => {}); }
function pause() { S.audio?.pause(); }
function playFrom(t) { if (!S.audio) return; S.stopAt = null; S.audio.currentTime = Math.max(0, t - 0.05); play(); }
function playRange(a, b) { if (!S.audio) return; S.audio.currentTime = Math.max(0, a - 0.04); S.stopAt = b + 0.08; play(); }
function togglePlay() { if (!S.audio) return; if (S.audio.paused) { S.stopAt = null; play(); } else pause(); }
function seekBy(dt) { if (S.audio) S.audio.currentTime = Math.max(0, Math.min(S.audio.duration || 1e9, S.audio.currentTime + dt)); }

function tick() {
  const a = S.audio;
  if (!a || a.paused) { S.ticking = false; return; }
  const t = a.currentTime;
  if (S.stopAt != null && t >= S.stopAt) { a.pause(); S.stopAt = null; }
  highlight(t);
  $("#t-cur").textContent = clock(t);
  renderFollow();
  requestAnimationFrame(tick);
}

function highlight(t, force) {
  const W = S.words;
  let lo = 0, hi = W.length - 1, idx = -1;
  while (lo <= hi) { const mid = (lo + hi) >> 1; if (W[mid].s <= t) { idx = mid; lo = mid + 1; } else hi = mid - 1; }
  if (idx >= 0 && t > W[idx].e + 0.3) idx = -1;
  if (idx === S.nowIdx && !force) return;
  $(".w.now")?.classList.remove("now");
  S.nowIdx = idx;
  $("#t-cur").textContent = clock(t);
  if (idx < 0) return;
  const el = $(`.w[data-i="${idx}"]`);
  if (!el) return;
  el.classList.add("now");
  if (S.follow && S.stopAt == null && !S.pop && Date.now() - S.userScrollAt > 4000) {
    const sc = $("#scroller").getBoundingClientRect(), r = el.getBoundingClientRect();
    if (r.top < sc.top + 52 + 40 || r.bottom > sc.bottom - 72 - 40) scrollToEl(el);
  }
}

// manual scrolling pauses "Следовать" for 4 s; the button then offers to jump back
function renderFollow() {
  const b = $("#btn-follow"), paused = S.follow && Date.now() - S.userScrollAt < 4000 && S.audio && !S.audio.paused;
  b.classList.toggle("on", S.follow);
  const txt = paused ? "Вернуться к звуку" : "Следовать";
  if (b.textContent !== txt) b.textContent = txt;
}
const userScrolled = () => { S.userScrollAt = Date.now(); renderFollow(); };
$("#scroller").addEventListener("wheel", userScrolled, { passive: true });
$("#scroller").addEventListener("keydown", (e) => { if (["PageUp", "PageDown", "Home", "End"].includes(e.key)) userScrolled(); });
$("#btn-follow").addEventListener("click", () => {
  if (S.follow && Date.now() - S.userScrollAt < 4000) {
    S.userScrollAt = 0;
    const el = $(".w.now"); if (el) scrollToEl(el);
  } else S.follow = !S.follow;
  renderFollow();
});
$("#btn-play").addEventListener("click", togglePlay);
$("#btn-back").addEventListener("click", () => seekBy(-5));
$("#btn-fwd").addEventListener("click", () => seekBy(5));

function setRate(r) {
  S.rate = r; ls.set("rate", r);
  if (S.audio) S.audio.playbackRate = r;
  $("#btn-rate").textContent = String(r).replace(".", ",") + "×";
}
function rateStep(dir) { const k = RATES.indexOf(S.rate); setRate(RATES[Math.max(0, Math.min(RATES.length - 1, (k < 0 ? 1 : k) + dir))]); toast(`Скорость ${$("#btn-rate").textContent}`, 1200); }
function rateMenu(anchor) {
  openMenu(anchor, menuItems(RATES.map((r) => ({ a: String(r), label: String(r).replace(".", ",") + "×", checked: r === S.rate })), true),
    (a) => setRate(+a), { above: true, align: "end" });
}
$("#btn-rate").addEventListener("click", (e) => rateMenu(e.currentTarget));
$("#btn-player-more").addEventListener("click", (e) => {
  const anchor = e.currentTarget;
  openMenu(anchor, menuItems([
    { a: "rate", label: "Скорость", sub: String(S.rate).replace(".", ",") + "×" },
    { a: "follow", label: S.follow ? "Не следовать за звуком" : "Следовать за звуком" },
  ]), (a) => { if (a === "rate") rateMenu(anchor); if (a === "follow") { S.follow = !S.follow; renderFollow(); } }, { above: true, align: "end" });
});

// scroll edge effects (§3.2): material under the toolbar, gradient above the player
function updateEdges() {
  const sc = $("#scroller");
  $("#topbar").classList.toggle("scrolled", sc.scrollTop > 4);
  $("#player").classList.toggle("edge", sc.scrollHeight - sc.scrollTop - sc.clientHeight > 4);
}
$("#scroller").addEventListener("scroll", updateEdges, { passive: true });

// ================================================================== keyboard (§8)

function setDocSize(px) {
  px = Math.max(14, Math.min(22, px));
  document.documentElement.style.setProperty("--doc-size", px + "px");
  ls.set("docSize", px);
}

document.addEventListener("keydown", (e) => {
  const ctrl = e.ctrlKey || e.metaKey, k = e.key.toLowerCase(), code = e.code;
  const sheetOpen = $$(".sheet-wrap").some((s) => !s.hidden);
  if (S.pop && !sheetOpen && popKey(e)) return;
  if (ctrl && (code === "KeyO" || code === "KeyN")) { e.preventDefault(); return openNewSheet(); }
  if (e.key === "Escape") {
    if (menuOpen()) { closeMenu(); return; }
    if (sheetOpen) return closeTopSheet();
    if (!$("#findbar").hidden) return toggleFind(false);
    if ($("#app").classList.contains("insp-overlay") && inspOpen()) return setInsp(false);
    if ($("#app").classList.contains("side-overlay") && sideOpen()) return setSide(false);
    if (S.selBlocks.size) { S.selBlocks.clear(); $$(".block.sel").forEach((b) => b.classList.remove("sel")); }
    return;
  }
  if (sheetOpen) return;
  if (ctrl && e.shiftKey && code === "KeyF") { e.preventDefault(); if (!sideOpen()) setSide(true); $("#lib-search").focus(); return; }
  if (ctrl && e.shiftKey && code === "KeyD") { e.preventDefault(); return openDictSheet(); }
  if (ctrl && (code === "KeyF" || code === "KeyH")) { e.preventDefault(); toggleFind(true); if (code === "KeyH") $("#replace-input").focus(); return; }
  if (ctrl && code === "Digit1") { e.preventDefault(); return setView("library"); }
  if (ctrl && code === "Digit2") { e.preventDefault(); return setView("queue"); }
  if (ctrl && code === "Digit3") { e.preventDefault(); return setView("gold"); }
  if (ctrl && (code === "Backslash" || code === "IntlBackslash")) { e.preventDefault(); return setSide(!sideOpen()); }
  if (ctrl && code === "KeyI") { e.preventDefault(); if (S.doc) setInsp(!inspOpen()); return; }
  if (ctrl && code === "KeyE") { e.preventDefault(); return exportMenu(); }
  if (ctrl && (k === "=" || k === "+" || code === "Equal" || code === "NumpadAdd")) { e.preventDefault(); return setDocSize((ls.get("docSize", 16)) + 1); }
  if (ctrl && (k === "-" || code === "Minus" || code === "NumpadSubtract")) { e.preventDefault(); return setDocSize((ls.get("docSize", 16)) - 1); }
  if (ctrl && code === "BracketLeft") { e.preventDefault(); return rateStep(-1); }
  if (ctrl && code === "BracketRight") { e.preventDefault(); return rateStep(1); }
  if (e.key === "F3") { e.preventDefault(); return goldQuiet() ? goGold(e.shiftKey ? -1 : 1) : goReview(e.shiftKey ? -1 : 1); }
  if (typingIn(e.target) || menuOpen()) return;
  if (e.key === "F2") { e.preventDefault(); return startRename(); }
  if (ctrl && code === "KeyZ") { e.preventDefault(); return undo(); }
  if (code === "Space") { e.preventDefault(); return togglePlay(); }
  if (e.key === "ArrowLeft") { e.preventDefault(); return seekBy(e.shiftKey ? -1 : -5); }
  if (e.key === "ArrowRight") { e.preventDefault(); return seekBy(e.shiftKey ? 1 : 5); }
  if (!S.doc || S.doc.status !== "done") return;
  if (isGold() && S.view === "library" && goldKey(e, ctrl, code)) return;
  if (e.altKey && /^Digit[1-4]$/.test(code) && !goldQuiet()) {
    const seg = S.sel?.k === "w" ? segById(S.sel.seg) : S.sel?.k === "seg" ? segById(S.sel.id) : null;
    if (seg?.alts?.length) { e.preventDefault(); acceptAlt(seg, +code.slice(5) - 1); }
    return;
  }
  if (/^Digit[1-9]$/.test(code) && !ctrl && !e.altKey && S.selBlocks.size) {
    const it = spkList()[+code.slice(5) - 1];
    if (it) { e.preventDefault(); assignSpeaker(segsOfBlocks(S.selBlocks), it[0]); }
    return;
  }
  if ((e.key === "ArrowDown" || e.key === "ArrowUp") && S.selBlocks.size) {
    e.preventDefault();
    const keys = blockKeys(), cur = keys.indexOf(S.lastBlock);
    const next = keys[Math.max(0, Math.min(keys.length - 1, cur + (e.key === "ArrowDown" ? 1 : -1)))];
    if (next) { S.selBlocks = new Set([next]); S.lastBlock = next; $$(".block", $("#transcript")).forEach((b) => b.classList.toggle("sel", b.dataset.key === next)); scrollToEl($(`.block[data-key="${next}"]`)); }
    return;
  }
  if (e.key === "Enter") {
    const id = S.sel?.k === "w" ? S.sel.seg : S.sel?.k === "seg" ? S.sel.id : S.selBlocks.size ? segsOfBlocks(new Set([S.lastBlock]))[0] : null;
    if (id) { e.preventDefault(); startEdit(id); }
  }
});

// the loop of checking a gold doc by ear: listen, fix if needed, confirm, next
function goldKey(e, ctrl, code) {
  const plain = !ctrl && !e.altKey, id = curSegId();
  if (e.key === "Enter" && plain && !S.selBlocks.size) {
    e.preventDefault();
    if (e.shiftKey) { if (id) setVerified(id, false); }
    else if (id) confirmAndNext(id);
    else goGold(1);
    return true;
  }
  if (code === "KeyE" && plain && !e.shiftKey && id) { e.preventDefault(); startEdit(id); return true; }
  if (code === "KeyR" && plain && !e.shiftKey) { e.preventDefault(); const s = segById(id); if (s) playRange(s.start, s.end); else goGold(1); return true; }
  if (code === "Slash" && ctrl && id) { e.preventDefault(); markInaudible(id); return true; }
  if ((e.key === "ArrowDown" || e.key === "ArrowUp") && plain && !e.shiftKey && !S.selBlocks.size) {
    e.preventDefault();
    const segs = S.doc.segments, k = segs.findIndex((s) => s.id === id);
    const n = segs[k < 0 ? 0 : Math.max(0, Math.min(segs.length - 1, k + (e.key === "ArrowDown" ? 1 : -1)))];
    if (n) focusSeg(n.id, true);
    return true;
  }
  return false;
}

// WebView2's own context menu («Обновить», «Сохранить как») only in text fields
document.addEventListener("contextmenu", (e) => { if (!typingIn(e.target) && !e.defaultPrevented) e.preventDefault(); });

// ================================================================== new transcription sheet

function segControl(id, key) {
  const el = $("#" + id);
  const sync = () => $$("button", el).forEach((b) => b.classList.toggle("on", b.dataset.v === String(S.opts[key])));
  el.addEventListener("click", (e) => { const b = e.target.closest("button"); if (!b) return; S.opts[key] = b.dataset.v; sync(); onOptsChange(); });
  sync();
  return sync;
}
const syncQuality = segControl("opt-quality", "quality");
const syncLang = segControl("opt-lang", "language");
const syncProfile = segControl("opt-profile", "profile");
const syncSpk = segControl("opt-spk", "num_speakers");

function effectiveLang() {
  if (S.opts.language === "other") return $("#opt-lang-other").value.trim().toLowerCase() || "auto";
  return S.opts.language;
}

function onOptsChange() {
  $("#opt-lang-other").hidden = S.opts.language !== "other";
  const lang = effectiveLang();
  const inst = (k) => S.models.find((m) => m.key === k)?.installed;
  const hints = [];
  const sel = $("#opt-model").value;
  if (sel === "auto") {
    if (lang === "ru" && !inst("ru-large-v3")) hints.push("Для русского точнее дообученная «Русский · large-v3» — её можно поставить в «Моделях».");
    if (lang === "de" && !inst("de-turbo")) hints.push("Для немецкого точнее «Deutsch · large-v3-turbo (primeline)».");
  }
  if ($("#opt-align").checked && ["ru", "de", "en"].includes(lang) && !inst("align-" + lang)) hints.push("Модель точных таймкодов для этого языка не установлена — будут таймкоды Whisper.");
  if (S.opts.num_speakers !== "1" && !$("#opt-stereo").checked && !inst("diarization")) hints.push("Модель спикеров не установлена — разметки спикеров не будет.");
  if (lang === "auto") hints.push("Если язык известен — укажите его: это заметно повышает точность.");
  if (S.opts.quality === "max") hints.push("Максимальная точность: запись распознают несколько моделей, это дольше.");
  $("#new-hint").textContent = hints.join(" ");
  $("#new-hint").classList.toggle("warn", hints.length > 0 && lang !== "auto" && S.opts.quality !== "max");
  if (!S.retranscribe && !$("#sheet-new").hidden) savePrefs();
}

function savePrefs() {
  const spk = S.opts.num_speakers;
  writePrefs({ language: effectiveLang(), profile: S.opts.profile, num_speakers: spk === "auto" ? 0 : +spk,
    mixed: $("#opt-mixed").checked, model: $("#opt-model").value, quality: S.opts.quality });
}
["opt-align", "opt-stereo", "opt-model", "opt-lang-other", "opt-mixed"].forEach((id) => $("#" + id).addEventListener("input", onOptsChange));

// ---- remembered choices (localStorage) on top of server defaults (GET /api/settings)
const PREF_KEY = "transcribe.newOptions";
const FALLBACK_DEFAULTS = { language: "ru", profile: "balanced", num_speakers: 2, mixed: false, model: "auto", quality: "max" };
const PROFILES = ["balanced", "clean", "noisy"];
function readPrefs() { try { return JSON.parse(localStorage.getItem(PREF_KEY) || "{}") || {}; } catch { return {}; } }
function writePrefs(p) { try { localStorage.setItem(PREF_KEY, JSON.stringify(p)); } catch {} }
const spkToUi = (n) => (n == null || n === "" || n === "auto" || +n === 0 ? "auto" : +n >= 6 ? "6" : String(+n));

function setLangOpt(lang) {
  lang = (lang || "auto").toLowerCase();
  if (["auto", "ru", "de", "en"].includes(lang)) S.opts.language = lang;
  else { S.opts.language = "other"; $("#opt-lang-other").value = lang; }
}
function applyOpts(p) {
  setLangOpt(p.language);
  S.opts.profile = PROFILES.includes(p.profile) ? p.profile : "balanced";
  S.opts.num_speakers = spkToUi(p.num_speakers);
  S.opts.model = p.model || "auto";
  S.opts.quality = p.quality === "fast" ? "fast" : "max";
  $("#opt-mixed").checked = !!p.mixed;
}
function selectModel(key) {
  const sel = $("#opt-model");
  const ok = [...sel.options].some((o) => o.value === key && !o.disabled);
  sel.value = ok ? key : "auto";
}
const serverDefaults = () => S.settings?.defaults || {};

// ---- project dictionary data
const DICT_EMPTY = { terms: [], replace: [], context: "" };
async function loadDict() {
  try {
    const d = await api("/api/dictionary");
    S.dict = { ...DICT_EMPTY, ...(d && typeof d === "object" ? d : {}) }; S.dictOk = true;
  } catch { S.dict = S.dict || { ...DICT_EMPTY }; S.dictOk = false; }
  renderDictLine(); renderNav();
  return S.dict;
}
function dictCounts(d) {
  const n = d?.terms?.length || 0, k = d?.replace?.length || 0;
  return `${n} ${plural(n, "термин", "термина", "терминов")} · ${k} ${plural(k, "замена", "замены", "замен")}`;
}
function renderDictLine() { $("#dict-count-line").textContent = `Словарь проекта: ${dictCounts(S.dict)}`; }

function startOptions() {
  const lang = effectiveLang(), spk = S.opts.num_speakers;
  return {
    language: lang === "auto" ? null : lang, profile: S.opts.profile, model: $("#opt-model").value, quality: S.opts.quality,
    num_speakers: spk === "auto" ? null : +spk, diarize: spk !== "1",
    glossary: $("#opt-glossary").value, topic: $("#opt-topic").value, mixed: $("#opt-mixed").checked,
    align: $("#opt-align").checked, stereo_split: $("#opt-stereo").checked, highpass: $("#opt-hp").checked,
  };
}
// options from the remembered choices, for Shift+drop (no sheet)
function quickOptions() {
  const p = { ...FALLBACK_DEFAULTS, ...serverDefaults(), ...readPrefs() };
  const lang = (p.language || "auto").toLowerCase(), n = +p.num_speakers || 0;
  return { language: lang === "auto" ? null : lang, profile: p.profile, model: p.model || "auto", quality: p.quality === "fast" ? "fast" : "max",
    num_speakers: n || null, diarize: n !== 1, mixed: !!p.mixed, align: true, glossary: "", topic: "" };
}

async function fillModelSelect() {
  try { S.models = await api("/api/models"); } catch {}
  const wm = S.models.filter((m) => m.kind === "whisper");
  $("#opt-model").innerHTML = `<option value="auto">Авто — лучшая установленная для языка</option>` +
    wm.map((m) => `<option value="${m.key}" ${m.installed ? "" : "disabled"}>${esc(m.title)}${m.installed ? "" : " — не установлена"}</option>`).join("");
}

async function openNewSheet({ files, retranscribe } = {}) {
  S.retranscribe = retranscribe || null;
  if (files) { S.dropped = files; S.picked = []; }
  else if (!retranscribe) { S.dropped = []; S.picked = []; }
  closeMenu(); closePop();
  await fillModelSelect();
  loadDict();
  if (retranscribe) {
    const o = retranscribe.options || {};
    applyOpts({ ...FALLBACK_DEFAULTS, ...serverDefaults(), ...o, language: o.language || "auto", num_speakers: o.diarize === false ? 1 : o.num_speakers });
    $("#opt-glossary").value = o.glossary || ""; $("#opt-topic").value = o.topic || "";
  } else {
    applyOpts({ ...FALLBACK_DEFAULTS, ...serverDefaults(), ...readPrefs() });
    if (S.lastSheetWasRetranscribe) { $("#opt-glossary").value = ""; $("#opt-topic").value = ""; }
  }
  S.lastSheetWasRetranscribe = !!retranscribe;
  selectModel(S.opts.model);
  syncQuality(); syncLang(); syncProfile(); syncSpk();
  $("#new-title").textContent = retranscribe ? "Распознать заново" : "Новая транскрипция";
  $("#new-files").hidden = !!retranscribe;
  renderPicked(); onOptsChange();
  $("#sheet-new").hidden = false;
  if (!retranscribe && !S.picked.length && !S.dropped.length) pickFiles();
}

function renderPicked() {
  const names = [...S.picked.map((p) => p.split(/[\\/]/).pop()), ...S.dropped.map((f) => f.name)];
  $("#file-list").innerHTML = names.length
    ? names.slice(0, 4).map((n) => `<div><b>${esc(n)}</b></div>`).join("") + (names.length > 4 ? `<div>и ещё ${names.length - 4}</div>` : "")
    : "Файлы не выбраны";
  if (!S.retranscribe) $("#new-title").textContent = "Новая транскрипция";
  $("#btn-start").disabled = !S.retranscribe && !names.length;
  $("#btn-start").textContent = !S.retranscribe && names.length > 1 ? `В очередь (${names.length})` : "Распознать";
}

async function pickFiles() {
  if (window.pywebview?.api) {
    const paths = await window.pywebview.api.pick_files();
    if (paths?.length) { S.picked = paths; S.dropped = []; }
    renderPicked();
  } else {
    const inp = document.createElement("input");
    inp.type = "file"; inp.multiple = true; inp.accept = "audio/*,video/*,.ogg,.opus,.oga,.amr,.wma";
    inp.onchange = () => { S.dropped = [...inp.files]; S.picked = []; renderPicked(); };
    inp.click();
  }
}

// queue files; afterwards: one file -> open it, several -> the queue view (§4.8)
async function enqueue({ paths = [], files = [], options, retranscribe = null }) {
  let created;
  if (retranscribe) created = [await api(`/api/docs/${retranscribe.id}/retranscribe`, { method: "POST", body: { options } })];
  else if (paths.length) created = await api("/api/transcribe", { method: "POST", body: { paths, options } });
  else {
    created = [];
    for (const f of files)
      created.push(...await api(`/api/upload?name=${encodeURIComponent(f.name)}&options=${encodeURIComponent(JSON.stringify(options))}`, { method: "POST", form: f }));
  }
  created = created || [];
  created.forEach((d) => S.queueIds.add(d.id));
  await refreshDocs();
  schedulePoll();
  if (created.length > 1) setView("queue");
  else if (created[0]) openDoc(created[0].id);
  return created;
}

$("#btn-pick").addEventListener("click", pickFiles);
$("#btn-dict-2").addEventListener("click", () => openDictSheet());
$("#btn-q-add").addEventListener("click", () => openNewSheet());
$$("[data-close]").forEach((b) => b.addEventListener("click", () => (b.closest(".sheet-wrap").hidden = true)));
$$(".sheet-wrap").forEach((w) => w.addEventListener("mousedown", (e) => { if (e.target === w) { w._down = true; } }));
$$(".sheet-wrap").forEach((w) => w.addEventListener("click", (e) => { if (e.target === w && w._down) closeSheet(w); w._down = false; }));
function closeSheet(w) { if (w.id === "sheet-dict") return cancelDict(); w.hidden = true; }
function closeTopSheet() { const w = $$(".sheet-wrap").filter((s) => !s.hidden).pop(); if (w) closeSheet(w); }

$("#btn-start").addEventListener("click", async () => {
  const options = startOptions();
  if (!S.retranscribe) savePrefs();
  $("#btn-start").disabled = true;
  try {
    await enqueue({ paths: S.picked, files: S.dropped, options, retranscribe: S.retranscribe });
    $("#sheet-new").hidden = true;
  } catch (e) { fail(e); } finally { $("#btn-start").disabled = false; }
});
$("#sheet-new").addEventListener("keydown", (e) => { if ((e.ctrlKey || e.metaKey) && e.key === "Enter") { e.preventDefault(); $("#btn-start").click(); } });

// ================================================================== drag and drop on the whole window (§4.8)

const isMediaType = (t) => !t || /^(audio|video)\//.test(t) || t === "application/ogg";
let dragDepth = 0;
addEventListener("dragenter", (e) => {
  if (!e.dataTransfer?.types?.includes("Files")) return;
  dragDepth++;
  if (dragDepth > 1) return;
  const items = [...(e.dataTransfer.items || [])].filter((i) => i.kind === "file");
  const n = items.length, bad = items.filter((i) => !isMediaType(i.type)).length;
  $("#drop-title").textContent = n ? `Отпустите, чтобы добавить ${nfiles(n)}` : "Отпустите, чтобы добавить файлы";
  $("#drop-warn").hidden = !bad;
  $("#drop-warn").textContent = bad ? `${nfiles(bad)} не ${bad === 1 ? "поддерживается и будет пропущен" : "поддерживаются и будут пропущены"}` : "";
  $("#dropzone").hidden = false;
});
addEventListener("dragleave", () => { if (--dragDepth <= 0) { dragDepth = 0; $("#dropzone").hidden = true; } });
addEventListener("dragover", (e) => e.preventDefault());
addEventListener("drop", async (e) => {
  e.preventDefault(); dragDepth = 0; $("#dropzone").hidden = true;
  const all = [...(e.dataTransfer?.files || [])];
  const files = all.filter((f) => /^(audio|video)\//.test(f.type) || f.type === "application/ogg" || AUDIO_EXT.test(f.name));
  if (all.length > files.length) toast(`Пропущено: ${nfiles(all.length - files.length)} — формат не поддерживается`, 3500);
  if (!files.length) return;
  if (e.shiftKey) {
    try {
      await enqueue({ files, options: quickOptions() });
      toast(`${nfiles(files.length)} в очереди`, 4000, { label: "Показать", run: () => setView("queue") });
    } catch (err) { fail(err); }
    return;
  }
  openNewSheet({ files });
});

// ================================================================== queue view (§4.2)

S.queueIds = new Set(ls.get("queueIds", []));
function queueItems() {
  return S.docs.filter((d) => working(d) || S.queueIds.has(d.id));
}
function renderQueue() {
  if (S.view !== "queue") return;
  const docs = S.docs;
  docs.filter(working).forEach((d) => S.queueIds.add(d.id));
  ls.set("queueIds", [...S.queueIds].filter((id) => docs.some((d) => d.id === id)).slice(-200));
  const items = queueItems();
  const proc = items.filter((d) => d.status === "processing" && !isWaiting(d));
  const wait = items.filter(isWaiting).sort((a, b) => a.id.localeCompare(b.id));
  const done = items.filter((d) => !working(d)).sort((a, b) => (b.updated || "").localeCompare(a.updated || ""));
  const parts = [];
  parts.push(`Обрабатывается ${proc.length}`, `Ждут ${wait.length}`, `Готово ${done.filter((d) => d.status === "done").length}`);
  $("#q-summary").textContent = items.length ? parts.join(" · ") : "";
  const total = items.reduce((s, d) => s + (d.duration || 0), 0);
  const ready = done.reduce((s, d) => s + (d.duration || 0), 0) + proc.reduce((s, d) => s + (d.duration || 0) * (d.progress || 0), 0);
  $("#q-total-bar").style.transform = `scaleX(${total ? ready / total : items.length ? done.length / items.length : 0})`;
  const el = $("#queue");
  if (!items.length) { el.innerHTML = `<div class="state-box" style="margin:96px auto 0"><h2>Очередь пуста</h2><p>Перетащите файлы в окно — они встанут сюда.</p></div>`; return; }
  let n = 0;
  const row = (d, kind) => {
    n++;
    const o = d.options || {};
    let sub, act, bar = "";
    if (kind === "proc") {
      sub = [d.stage || "Обработка", pct(d.progress), eta(d)].filter(Boolean).join(" · ");
      act = `<button class="btn compact" data-a="open" data-id="${d.id}">Открыть</button>`;
      bar = `<div class="bar"><i style="transform:scaleX(${d.progress || 0})"></i></div>`;
    } else if (kind === "wait") {
      sub = ["Ждёт", o.language ? (LANGS[o.language] || o.language) : d.language ? (LANGS[d.language] || d.language) : "", `модель ${o.model && o.model !== "auto" ? o.model : "Авто"}`].filter(Boolean).join(" · ");
      act = `<button class="btn compact hover-only" data-a="remove" data-id="${d.id}">Убрать из очереди</button>`;
    } else if (d.status === "error") {
      sub = `<span class="err">${esc(d.error || "Ошибка")}</span>`;
      act = `<button class="btn compact" data-a="retry" data-id="${d.id}">Повторить</button>`;
    } else {
      const took = d.created && d.updated ? (new Date(d.updated) - new Date(d.created)) / 1000 : 0;
      sub = took > 0 ? `Готово за ${clock(took)}` : "Готово";
      act = `<button class="btn compact" data-a="open" data-id="${d.id}">Открыть</button>`;
    }
    return `<div class="q-row" data-id="${d.id}"><span class="q-num">${n}</span><div class="q-main"><div class="q-title">${esc(d.title)}</div>
      <div class="q-sub${d.status === "error" ? " err" : ""}">${kind === "done" && d.status === "error" ? sub : esc(sub)}</div></div>
      <span class="q-dur">${d.duration ? clock(d.duration) : ""}</span><div class="q-act">${act}</div>${bar}</div>`;
  };
  let html = wait.length > 1 ? `<p class="q-note">Файлы распознаются по одному, чтобы не делить видеокарту.</p>` : "";
  html += proc.map((d) => row(d, "proc")).join("") + wait.map((d) => row(d, "wait")).join("");
  if (done.length) {
    const showAll = S.queueAll || done.length <= 10;
    html += `<div class="q-group">Готово</div>` + (showAll ? done : done.slice(0, 10)).map((d) => row(d, "done")).join("");
    if (!showAll) html += `<button class="text-btn q-more" data-a="all">Показать все (${done.length})</button>`;
  }
  // keep existing rows (and their bars) when the set of rows is the same, so progress animates
  const sig = items.map((d) => d.id + d.status + isWaiting(d)).join(",") + S.queueAll;
  if (el._sig === sig) {
    const tmp = document.createElement("div"); tmp.innerHTML = html;
    $$(".q-row", tmp).forEach((r) => {
      const cur = el.querySelector(`.q-row[data-id="${r.dataset.id}"]`);
      if (!cur) return;
      $(".q-sub", cur).innerHTML = $(".q-sub", r).innerHTML;
      const b = $(".bar i", cur), nb = $(".bar i", r);
      if (b && nb) b.style.transform = nb.style.transform;
    });
    return;
  }
  el._sig = sig;
  el.innerHTML = html;
}
$("#queue").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-a]");
  if (!b) return;
  const d = S.docs.find((x) => x.id === b.dataset.id);
  if (b.dataset.a === "all") { S.queueAll = true; $("#queue")._sig = ""; return renderQueue(); }
  if (b.dataset.a === "open") openDoc(b.dataset.id);
  if (b.dataset.a === "retry") retryDoc(await api(`/api/docs/${b.dataset.id}`).catch(() => d));
  if (b.dataset.a === "remove") deleteDoc(b.dataset.id, d?.title);
});

// ================================================================== gold reference (§10, docs/GOLD.md)

// ---- checking a gold doc by ear: the cursor is the selected fragment (curSegId)

function focusSeg(id, play) {
  const s = segById(id);
  if (!s) return;
  closePop();
  select({ k: "seg", id });
  const el = $(`.seg[data-id="${id}"]`);
  if (el) scrollToEl(el);
  if (play) playRange(s.start, s.end);
}
// the next (dir 1) / previous (-1) / first (0) fragment not yet checked, after the time from
function goGold(dir, play = true, from = null) {
  const segs = S.doc?.segments || [];
  const todo = segs.filter((s) => !s.verified);
  if (!todo.length) { if (dir && segs.length) toast("Все фрагменты проверены на слух"); renderGoldBar(); return; }
  const t = from ?? segById(curSegId())?.start;
  let it;
  if (!dir || t == null) it = dir < 0 ? todo[todo.length - 1] : todo[0];
  else if (dir > 0) it = todo.find((s) => s.start > t + 1e-3) || todo[0];
  else it = [...todo].reverse().find((s) => s.start < t - 1e-3) || todo[todo.length - 1];
  focusSeg(it.id, play);
}
function confirmAndNext(id) {
  const s = segById(id);
  if (!s) return goGold(1);
  if (!s.verified) setVerified(id, true);
  goGold(1, true, s.start);
}

function paintVer(id) {
  const s = segById(id), el = $(`.seg[data-id="${id}"]`);
  if (!s || !el) return;
  el.classList.toggle("ver", !!s.verified);
  const b = $(".ver-btn", el);
  if (b) { b.title = s.verified ? "Проверено на слух · щёлкните, чтобы снять отметку (Shift+Enter)" : "Отметить: верно на слух (Enter)"; b.ariaLabel = s.verified ? "Проверено" : "Не проверено"; }
}
// optimistic: the loop does not wait for the server; only the answer to the latest request is
// taken (an older one would lack the marks made since), a failure puts the mark back. Requests go
// one after another: sent in parallel, the server may finish them out of order, and the "latest"
// answer would then miss marks of requests it overtook.
let verSeq = 0, verChain = Promise.resolve();
async function setVerified(id, on) {
  const s = segById(id);
  if (!s || !isGold() || !!s.verified === on) return;
  const docId = S.doc.id, seq = ++verSeq;
  s.verified = on;
  paintVer(id); renderGoldBar(); renderInspector();
  try {
    const req = verChain.then(() => segApi(id, "/verified", { method: "POST", body: { on } }));
    verChain = req.catch(() => {});
    const doc = await req;
    if (seq === verSeq && S.doc?.id === docId && doc?.segments) {
      S.doc = doc;
      if (!$("#transcript textarea")) { renderTranscript(); renderToolbar(); renderInspector(); if (S.pop) positionPop(); }
    }
  } catch (e) {
    const x = S.doc?.id === docId && segById(id);
    if (x) { x.verified = !on; paintVer(id); renderGoldBar(); renderInspector(); }
    fail(e);
  }
  goldSoon();
}

async function markInaudible(id) {
  const s = segById(id);
  if (!s) return;
  const t = s.start;
  if (s.text.trim() !== "???") {
    const d = await setSegText(id, "???");
    if (!d) return;
    toast("Фрагмент не разобрать: «???» — при оценке он пропускается · Ctrl+Z — вернуть", 3500);
  }
  if (isGold()) goGold(1, true, t);
  goldSoon();
}

function setGoldVars(on) {
  S.goldVars = on; ls.set("goldVars", on);
  if (!on) closePop();
  renderToolbar(); renderTranscript(); renderInspector();
}
$("#gb-vars").addEventListener("change", (e) => setGoldVars(e.target.checked));
$("#gb-next").addEventListener("click", () => goGold(1));
$("#gb-list").addEventListener("click", () => setView("gold"));

// the band under the toolbar: how much of the doc is checked by ear
function renderGoldBar() {
  const on = isGold() && S.doc.status === "done" && S.doc.segments.length > 0;
  $("#goldbar").hidden = !on;
  $("#view-doc").classList.toggle("gold-open", on);
  if (!on) return;
  const segs = S.doc.segments, ver = segs.filter((s) => s.verified);
  const len = (a) => a.reduce((n, s) => n + Math.max(0, s.end - s.start), 0);
  const all = len(segs), vs = len(ver), f = all ? vs / all : 0;
  $("#gb-bar").style.transform = `scaleX(${f})`;
  $("#gb-text").innerHTML = `<b>${pct(f)}</b> проверено ${ver.length} из ${segs.length}<span class="gb-time"> · ${clock(vs)} из ${clock(all)} речи</span>`;
  $("#gb-vars").checked = S.goldVars;
  $("#gb-next").disabled = ver.length >= segs.length;
}

// mark / unmark a doc as gold (its «…» menu, the library context menu)
async function setGold(id, on, title) {
  if (!on && !confirm(`Убрать «${title || "запись"}» из эталона? Она перестанет участвовать в оценке моделей.`)) return;
  let r;
  try { r = await api(`/api/docs/${id}/gold`, { method: "POST", body: { on } }); } catch (e) { return fail(e); }
  if (S.doc?.id === id) {
    const doc = r?.segments ? r : r?.doc?.segments ? r.doc : null;
    if (doc) S.doc = doc;
    else S.doc.gold = on ? (r?.gold || S.doc.gold || { created: new Date().toISOString(), manifest: [] }) : null;
    S.sel = null; closePop();
    if (on) S.reviewOn = S.goldVars; else S.reviewOn = null;
    renderToolbar(); renderTranscript(); renderInspector(); paintTicks();
    if (on && isGold()) goGold(0, false);
  }
  toast(on ? "Запись — эталон: подтвердите каждый фрагмент на слух" : "Запись убрана из эталона", 3200);
  await loadGold();
}

// ---- the Эталон view: gold docs, the latest evaluation, past runs

let goldT = 0;
const goldSoon = () => { clearTimeout(goldT); goldT = setTimeout(() => loadGold(), 1500); };

async function loadGold(full = false) {
  try {
    const g = await api("/api/gold");
    Object.assign(S.gold, { ok: true, docs: Array.isArray(g?.docs) ? g.docs : [], running: !!g?.running, error: g?.error || null });
    if (g?.eval) S.gold.eval = g.eval;
  } catch (e) { S.gold.ok = false; S.gold.why = e.message; }
  if (S.gold.ok && (full || S.gold.running)) await loadEval();
  renderNav(); listSig = ""; renderDocList(); renderGold();
  if (S.gold.running) pollEval();
}
async function loadEval() {
  try {
    const r = await api("/api/gold/eval");
    S.gold.running = !!r?.running; S.gold.error = r?.error || null;
    if (r?.result) S.gold.eval = r.result;
    if (Array.isArray(r?.history)) S.gold.history = r.history;
    S.gold.fails = 0;
    return r;
  } catch { S.gold.fails = (S.gold.fails || 0) + 1; return null; }
}
function pollEval() {
  clearTimeout(S.gold.pollT);
  S.gold.pollT = setTimeout(async () => {
    const r = await loadEval();
    if (!r && S.gold.fails < 5) return pollEval();
    if (!r) { S.gold.running = false; S.gold.error = "сервер не отвечает о ходе оценки"; }
    if (S.gold.running) { renderGoldRun(); return pollEval(); }
    S.gold.since = 0;
    const res = S.gold.eval, err = S.gold.error;
    await loadGold();
    if (err) toast("Оценка не удалась: " + err, 5000);
    else if (res) toast(`Оценка готова · ошибка ансамбля ${fmtP(res.merged?.wer)}`, 4000, S.view !== "gold" ? { label: "Показать", run: () => setView("gold") } : null);
  }, 2000);
}
async function startEval(tune) {
  if (S.gold.running) return;
  try { await api("/api/gold/eval", { method: "POST", body: { tune } }); }
  catch (e) { if (e.status !== 409) return fail(e); }    // 409: already running — just follow it
  Object.assign(S.gold, { running: true, error: null, since: Date.now(), tuning: tune });
  renderGold(); pollEval();
}
$("#btn-gold-eval").addEventListener("click", () => startEval(false));
$("#btn-gold-tune").addEventListener("click", () => startEval(true));

const fmtP = (x, d = 1) => (x == null || !Number.isFinite(+x) ? "—" : `${(x * 100).toFixed(d).replace(".", ",")} %`);
const fmtPP = (x) => (x == null || !Number.isFinite(+x) ? "—" : `${x > 0 ? "+" : x < 0 ? "−" : ""}${Math.abs(x * 100).toFixed(1).replace(".", ",")}`);
const fmtN = (n) => (n == null || !Number.isFinite(+n) ? "—" : Number(n).toLocaleString("ru-RU"));
const fmtW = (w) => (w == null || !Number.isFinite(+w) ? "—" : (+w).toFixed(2).replace(".", ","));
function evalDate(t) {
  if (t == null || t === "") return "";
  const d = new Date(typeof t === "number" ? (t < 1e12 ? t * 1000 : t) : t);
  return isNaN(d) ? String(t) : d.toLocaleString("ru-RU", { day: "numeric", month: "short", hour: "2-digit", minute: "2-digit" });
}
const KIND_NAMES = { "число": "Числа", "латиница": "Латиница", "имя": "Имена", "термин": "Термины" };
const KIND_TIPS = { "число": "даты, время, номера", "латиница": "немецкие и английские слова", "имя": "люди и места", "термин": "слова из словаря проекта" };
const cnt = (x) => (Array.isArray(x) ? x.length : x);     // «docs» may come as a count or as a list of ids
const goldTitle = (id) => S.gold.docs.find((d) => d.id === id)?.title || S.docs.find((d) => d.id === id)?.title || id;

function goldFrac(d) {
  if (d.segments && d.verified >= d.segments) return 1;
  return d.duration ? Math.min(1, (d.verified_seconds || 0) / d.duration) : d.segments ? (d.verified || 0) / d.segments : 0;
}

function renderGoldRun() {
  const el = $("#g-run");
  if (!el) return;
  const s = S.gold.since ? ` · ${clock((Date.now() - S.gold.since) / 1000)}` : "";
  el.textContent = `${S.gold.tuning ? "Оценка и подбор весов" : "Оценка"} идёт${s}`;
}

function renderGold() {
  if (S.view !== "gold") return;
  const G = S.gold, docs = G.docs, run = G.running;
  $("#btn-gold-eval").disabled = $("#btn-gold-tune").disabled = run || !docs.length || G.ok === false;
  const dur = docs.reduce((n, d) => n + (d.duration || 0), 0), vs = docs.reduce((n, d) => n + Math.min(d.duration || Infinity, d.verified_seconds || 0), 0);
  const segs = docs.reduce((n, d) => n + (d.segments || 0), 0), ver = docs.reduce((n, d) => n + (d.verified || 0), 0);
  const all = segs && ver >= segs ? 1 : dur ? vs / dur : 0;
  $("#g-summary").textContent = docs.length ? `${docs.length} ${plural(docs.length, "запись", "записи", "записей")} · ${clock(dur)} · проверено на слух ${clock(vs)} (${pct(all)})` : "";
  $("#g-total-bar").style.transform = `scaleX(${all})`;
  const el = $("#gold");
  if (G.ok === false) { el.innerHTML = `<div class="state-box g-empty"><h2>Эталон недоступен</h2><p>${esc(G.why || "Сервер не отвечает")}</p></div>`; return; }
  if (G.ok == null) { el.innerHTML = ""; return; }
  if (!docs.length) {
    el.innerHTML = `<div class="state-box g-empty"><h2>Эталона пока нет</h2><p>Эталон — записи, текст которых вы проверили на слух. По нему видно, сколько ошибается каждая модель и их ансамбль.
      Откройте запись, выберите в меню «…» «Сделать эталоном» и подтвердите каждый фрагмент: Enter — верно, E — исправить, Ctrl+/ — не разобрать.</p>
      <button class="btn primary" data-a="library">Открыть библиотеку</button></div>`;
    return;
  }
  let html = `<div class="g-head"><h2>Записи</h2></div>`;
  html += docs.map((d) => {
    const f = goldFrac(d), done = d.segments && d.verified >= d.segments;
    const sub = d.status && d.status !== "done" ? (d.status === "error" ? `<span class="err">Ошибка распознавания</span>` : "В обработке")
      : `Проверено ${d.verified || 0} из ${d.segments || 0} · ${clock(Math.min(d.verified_seconds || 0, d.duration || Infinity))} из ${clock(d.duration)}`;
    return `<div class="q-row g-row" data-id="${esc(d.id)}"><div class="q-main"><div class="q-title">${esc(d.title)}</div><div class="q-sub">${sub}</div></div>
      <span class="g-pct${done ? " ok" : ""}">${pct(f)}</span><span class="q-dur">${d.duration ? clock(d.duration) : ""}</span>
      <div class="q-act"><button class="btn compact" data-a="open" data-id="${esc(d.id)}">${done ? "Открыть" : "Проверить"}</button></div>
      <div class="bar"><i style="transform:scaleX(${f})"></i></div></div>`;
  }).join("");
  html += goldResultsHtml();
  el.innerHTML = html;
  if (run) renderGoldRun();
}

function goldResultsHtml() {
  const G = S.gold, R = G.eval;
  const nd = R ? cnt(R.docs) : null;
  let h = `<div class="g-head"><h2>Оценка</h2>${R ? `<span class="g-when">${esc([evalDate(R.t), nd != null ? `${nd} ${plural(nd, "запись", "записи", "записей")}` : "",
    R.words != null ? `${fmtN(R.words)} ${plural(R.words, "слово", "слова", "слов")}` : "", R.chunks != null ? `${fmtN(R.chunks)} ${plural(R.chunks, "кусок", "куска", "кусков")}` : ""].filter(Boolean).join(" · "))}</span>` : ""}</div>`;
  if (G.running) h += `<p class="g-run" id="g-run"></p>`;
  if (G.error && !G.running) h += `<p class="g-err">Оценка не удалась: ${esc(G.error)}</p>`;
  if (!R) return h + (G.running ? "" : `<p class="g-note">Оценки ещё не было. Проверьте фрагменты на слух и нажмите «Оценить»: ошибки моделей считаются по проверенному тексту, куски с «???» пропускаются.</p>`);
  const m = R.merged || {};
  h += `<div class="g-hero"><div class="g-big"><div class="g-num">${fmtP(m.wer)}</div><div class="g-cap">ошибка ансамбля (WER)${m.errors != null ? ` · ${fmtN(m.errors)} ${plural(m.errors, "ошибка", "ошибки", "ошибок")} на ${fmtN(m.words ?? R.words)} ${plural(m.words ?? R.words ?? 0, "слово", "слова", "слов")}` : ""}</div></div>`
    + (R.oracle?.wer != null ? `<div class="g-big dim"><div class="g-num">${fmtP(R.oracle.wer)}</div><div class="g-cap">оракул: если бы в каждом куске выбиралась лучшая модель — предел для ансамбля</div></div>` : "") + `</div>`;
  const mem = [...(R.members || [])].sort((a, b) => (a.wer ?? 9) - (b.wer ?? 9));
  if (mem.length) {
    const max = Math.max(...mem.map((x) => x.wer || 0), m.wer || 0) || 1;
    const bar = (w) => `<span class="g-bar"><i style="transform:scaleX(${Math.min(1, (w || 0) / max)})"></i></span>`;
    h += `<h3>Модели</h3><table class="g-table"><thead><tr><th>Модель</th><th class="num">WER</th><th class="barc"></th><th class="num">Слов</th><th class="num" title="Ошибка ансамбля на тех же кусках, где отвечала эта модель">Ансамбль там же</th></tr></thead><tbody>`
      + `<tr class="g-ens"><td>Ансамбль</td><td class="num">${fmtP(m.wer)}</td><td class="barc">${bar(m.wer)}</td><td class="num">${fmtN(m.words ?? R.words)}</td><td class="num"></td></tr>`
      + mem.map((x) => `<tr><td>${esc(mname(x.name))}</td><td class="num">${fmtP(x.wer)}</td><td class="barc">${bar(x.wer)}</td><td class="num">${fmtN(x.words)}</td>`
        + `<td class="num${x.merged_wer != null && x.wer != null && x.wer < x.merged_wer ? " bad" : ""}" title="${x.merged_wer != null && x.wer != null && x.wer < x.merged_wer ? "На этих кусках модель одна точнее ансамбля" : ""}">${fmtP(x.merged_wer)}</td></tr>`).join("")
      + `</tbody></table>`;
  }
  const abl = [...(R.ablation || [])].sort((a, b) => (a.delta ?? 0) - (b.delta ?? 0));
  if (abl.length) h += `<h3>Вклад моделей</h3><p class="g-note">Ошибка ансамбля, если убрать модель. Плюс — модель помогает, минус — без неё ансамбль точнее.</p>`
    + `<table class="g-table"><thead><tr><th>Без модели</th><th class="num">WER</th><th class="num">Разница, п.п.</th></tr></thead><tbody>`
    + abl.map((x) => `<tr><td>${esc(mname(x.name))}</td><td class="num">${fmtP(x.wer)}</td><td class="num ${x.delta < 0 ? "bad" : x.delta > 0 ? "good" : ""}">${fmtPP(x.delta)}</td></tr>`).join("")
    + `</tbody></table>`;
  const kinds = Object.entries(R.kinds || {});
  if (kinds.length) h += `<h3>Трудные слова</h3><div class="g-tiles">` + kinds.map(([k, v]) => `<div class="g-tile" title="${esc(KIND_TIPS[k] || "")}"><div class="g-tl">${esc(KIND_NAMES[k] || k)}</div>`
    + `<div class="g-tv">${fmtP(v?.wer)}</div><div class="g-ts">${fmtN(v?.words)} ${plural(v?.words || 0, "слово", "слова", "слов")}</div></div>`).join("") + `</div>`;
  const dz = R.doubts;
  if (dz) {
    const share = dz.marked == null ? "—" : dz.marked <= 1 ? fmtP(dz.marked) : fmtN(dz.marked);
    h += `<h3>Подсветка сомнительных слов</h3><dl class="kv g-kv">`
      + (dz.threshold != null ? `<dt>Порог</dt><dd>уверенность ниже ${String(dz.threshold).replace(".", ",")}</dd>` : "")
      + `<dt>Подсвечено</dt><dd>${share}${dz.marked != null && dz.marked <= 1 ? " слов" : ""}</dd>`
      + `<dt>Точность</dt><dd>${fmtP(dz.precision, 0)} <span class="kv-sub">подсвеченных слов действительно ошибочны</span></dd>`
      + `<dt>Полнота</dt><dd>${fmtP(dz.recall, 0)} <span class="kv-sub">всех ошибок подсвечено</span></dd></dl>`;
  }
  if (R.tune) h += tuneHtml(R.tune);
  if (R.note) h += `<p class="g-note">${esc(R.note)}</p>`;
  h += historyHtml();
  return h;
}

function tuneHtml(T) {
  const names = [...new Set([...Object.keys(T.base || {}), ...Object.keys(T.weights || {})])];
  let h = `<h3>Подбор весов</h3><p class="g-line">Ошибка ансамбля на записях подбора: ${fmtP(T.wer_before)} → <b class="${T.wer_after < T.wer_before ? "good" : ""}">${fmtP(T.wer_after)}</b></p>`;
  if (names.length) h += `<table class="g-table"><thead><tr><th>Модель</th><th class="num">Сейчас</th><th class="num">Предлагается</th></tr></thead><tbody>`
    + names.map((n) => { const a = T.base?.[n], b = T.weights?.[n], ch = a !== b && Math.abs((a ?? 0) - (b ?? 0)) > 1e-6;
      return `<tr${ch ? ` class="g-ch"` : ""}><td>${esc(mname(n))}</td><td class="num">${fmtW(a)}</td><td class="num">${fmtW(b)}</td></tr>`; }).join("")
    + `</tbody></table>`;
  if (T.heldout?.length) h += `<h4>Проверка на отложенной записи</h4><table class="g-table"><thead><tr><th>Запись</th><th class="num">Сейчас</th><th class="num">С новыми весами</th></tr></thead><tbody>`
    + T.heldout.map((x) => `<tr><td>${esc(goldTitle(x.doc))}</td><td class="num">${fmtP(x.before)}</td><td class="num ${x.after < x.before ? "good" : x.after > x.before ? "bad" : ""}">${fmtP(x.after)}</td></tr>`).join("")
    + `</tbody></table>`;
  return h + `<p class="g-note">Веса в приложении сами не меняются: их переносит в код (ensemble.MEMBERS) владелец или агент — и только если улучшение держится на отложенной записи.</p>`;
}

function historyHtml() {
  const H = [...(S.gold.history || [])].filter((x) => x && x.wer != null).sort((a, b) => String(a.t).localeCompare(String(b.t), undefined, { numeric: true }));
  if (H.length < 1) return "";
  // bars from the best run (lowest) to the worst: the change shows, not the distance from zero
  const last = H.slice(-24), max = Math.max(...last.map((x) => x.wer)), min = Math.min(...last.map((x) => x.wer));
  const hgt = (w) => (max > min ? 15 + 85 * (w - min) / (max - min) : 50).toFixed(1);
  const spark = last.length > 1 ? `<div class="g-spark" aria-hidden="true">${last.map((x, k) => `<i class="${k === last.length - 1 ? "cur" : ""}" style="height:${hgt(x.wer)}%" title="${esc(evalDate(x.t))} · ${fmtP(x.wer)}"></i>`).join("")}</div>` : "";
  return `<h3>Прошлые оценки</h3>${spark}<div class="g-hist">` + [...H].reverse().slice(0, 6).map((x) => `<div class="g-hrow"><span>${esc(evalDate(x.t))}</span><b>${fmtP(x.wer)}</b>`
    + `<span>${x.words != null ? `${fmtN(x.words)} ${plural(x.words, "слово", "слова", "слов")}` : ""}${x.docs != null ? ` · ${cnt(x.docs)} ${plural(cnt(x.docs), "запись", "записи", "записей")}` : ""}</span></div>`).join("") + `</div>`;
}

$("#gold").addEventListener("click", (e) => {
  const b = e.target.closest("[data-a]"), row = e.target.closest(".g-row");
  if (b?.dataset.a === "library") return setView("library");
  if (b?.dataset.a === "open" || (row && !b)) openDoc((b || row).dataset.id);
});

// ================================================================== models sheet (§4.10)

async function openModels() {
  $("#sheet-models").hidden = false;
  await renderModels();
  pollModels();
}
function pollModels() {
  clearTimeout(S.modelsT);
  if ($("#sheet-models").hidden) return;
  S.modelsT = setTimeout(async () => { await renderModels(); pollModels(); }, S.models.some((m) => m.installing) ? 1200 : 4000);
}
async function renderModels() {
  try { S.models = await api("/api/models"); } catch { return; }
  const st = S.status;
  $("#models-sub").textContent = st ? `Устройство: ${st.device === "cuda" ? "GPU" : "CPU"} (${st.compute_type}). Всё работает офлайн.` : "Всё работает офлайн.";
  $("#model-list").innerHTML = S.models.map((m) => `<div class="model">
    <div><div class="m-title">${esc(m.title)}</div><div class="m-note">${esc([m.note, m.size_gb ? `${String(m.size_gb).replace(".", ",")} ГБ` : ""].filter(Boolean).join(" · "))}</div>
    ${m.error && !m.installing ? `<div class="m-err">${esc(m.error)}</div>` : ""}${m.installing && m.message ? `<div class="m-note">${esc(m.message)}</div>` : ""}</div>
    <div>${m.installed ? `<span class="m-ok">Установлена</span>`
      : m.installing ? `<div class="m-prog"><div class="bar"><i style="transform:scaleX(${m.progress || 0})"></i></div>${pct(m.progress)}</div>`
      : `<button class="btn" data-key="${m.key}">Установить</button>`}</div></div>`).join("");
  renderNav();
  return S.models;
}
$("#model-list").addEventListener("click", async (e) => {
  const b = e.target.closest("[data-key]");
  if (b) { await api(`/api/models/${b.dataset.key}/install`, { method: "POST" }).catch(fail); await renderModels(); pollModels(); }
});
$("#btn-install-rec").addEventListener("click", async () => {
  for (const m of S.models) if (!m.installed && !m.installing) await api(`/api/models/${m.key}/install`, { method: "POST" }).catch(fail);
  await renderModels(); pollModels();
});
$("#sheet-models [data-close]").addEventListener("click", () => { clearTimeout(S.modelsT); if (!S.doc) renderState(); });

// ================================================================== dictionary sheet (§4.9)

const DS = { terms: [], replace: [], context: "", tab: "terms", base: "" };
const dictSnapshot = () => JSON.stringify({ t: DS.terms, r: DS.replace.filter((r) => r.from.trim() || r.to.trim()), c: $("#dict-context").value });

async function openDictSheet() {
  closeMenu(); closePop();
  $("#sheet-dict").hidden = false;
  $("#dict-status").textContent = "Загрузка…"; $("#dict-status").classList.remove("danger");
  $("#dict-save").disabled = true;
  $("#dict-apply-wrap").hidden = !(S.doc?.status === "done");
  $("#dict-apply").checked = false;
  await loadDict();
  DS.server = JSON.parse(JSON.stringify(S.dict));
  DS.terms = [...(S.dict.terms || [])];
  DS.replace = (S.dict.replace || []).map((r) => ({ from: r.from || "", to: r.to || "" }));
  $("#dict-context").value = S.dict.context || "";
  $("#dict-term-new").value = ""; $("#dict-term-filter").value = "";
  setDictTab(DS.tab);
  renderTerms(); renderRepl();
  DS.base = dictSnapshot();
  $("#dict-save").disabled = false;
  if (!S.dictOk) { $("#dict-status").textContent = "Сервер пока не отдаёт словарь — показаны пустые значения."; $("#dict-status").classList.add("danger"); }
  else dictStatus();
  (DS.tab === "terms" ? $("#dict-term-new") : DS.tab === "ctx" ? $("#dict-context") : $("#dict-repl input"))?.focus();
}
function setDictTab(t) {
  DS.tab = t;
  $$("#dict-tabs button").forEach((b) => b.classList.toggle("on", b.dataset.v === t));
  $("#dict-pane-terms").hidden = t !== "terms"; $("#dict-pane-repl").hidden = t !== "repl"; $("#dict-pane-ctx").hidden = t !== "ctx";
}
$("#dict-tabs").addEventListener("click", (e) => { const b = e.target.closest("button"); if (b) setDictTab(b.dataset.v); });
function dictStatus() {
  if (!S.dictOk) return;
  const dups = dupFroms();
  $("#dict-status").classList.toggle("danger", dups.size > 0);
  $("#dict-status").textContent = dups.size ? `Повторяются замены: ${[...dups].slice(0, 3).join(", ")}` : dictCounts({ terms: DS.terms, replace: DS.replace.filter((r) => r.from.trim()) });
}

function renderTerms() {
  const f = $("#dict-term-filter").value.trim().toLowerCase();
  $("#dict-term-filter").hidden = DS.terms.length <= 20;
  const list = DS.terms.map((t, k) => [t, k]).filter(([t]) => !f || t.toLowerCase().includes(f));
  $("#dict-terms").innerHTML = list.map(([t, k]) => `<div class="term"><span title="${esc(t)}">${esc(t)}</span><button class="icon-btn" data-k="${k}" title="Удалить" aria-label="Удалить"><svg class="ic"><use href="icons.svg#close"/></svg></button></div>`).join("")
    || `<div class="insp-empty">${f ? "Ничего не найдено" : "Терминов пока нет. Имена, названия мест и редкие слова — Whisper будет узнавать их лучше."}</div>`;
  dictStatus();
}
function addTerms(text) {
  const parts = String(text).split(/[,;\n\r]+/).map((s) => s.trim()).filter(Boolean);
  let added = 0;
  for (const p of parts) if (!DS.terms.some((t) => t.toLowerCase() === p.toLowerCase())) { DS.terms.push(p); added++; }
  renderTerms();
  return added;
}
$("#dict-term-new").addEventListener("keydown", (e) => {
  if (e.key === "Enter") { e.preventDefault(); if (e.ctrlKey) return; addTerms(e.target.value); e.target.value = ""; }
});
$("#dict-term-new").addEventListener("paste", (e) => {
  const t = e.clipboardData?.getData("text") || "";
  if (/[,;\n]/.test(t)) { e.preventDefault(); const n = addTerms(t); toast(`Добавлено терминов: ${n}`, 1600); }
});
$("#dict-term-filter").addEventListener("input", renderTerms);
$("#dict-terms").addEventListener("click", (e) => { const b = e.target.closest("[data-k]"); if (b) { DS.terms.splice(+b.dataset.k, 1); renderTerms(); } });

function dupFroms() {
  const seen = new Map(), dups = new Set();
  for (const r of DS.replace) { const k = r.from.trim().toLowerCase(); if (!k) continue; if (seen.has(k)) dups.add(r.from.trim()); seen.set(k, 1); }
  return dups;
}
function renderRepl() {
  const rows = [...DS.replace];
  if (!rows.length || rows[rows.length - 1].from || rows[rows.length - 1].to) rows.push({ from: "", to: "" });
  DS.replace = rows;
  $("#dict-repl").innerHTML = rows.map((r, k) => `<div class="repl-row" data-k="${k}">
    <input class="field" data-k="from" placeholder="как распознано" spellcheck="false" value="${esc(r.from)}">
    <span class="arrow">→</span>
    <input class="field" data-k="to" placeholder="как правильно" spellcheck="false" value="${esc(r.to)}">
    <button class="icon-btn" type="button" data-act="del" title="Удалить" aria-label="Удалить" tabindex="-1"><svg class="ic"><use href="icons.svg#close"/></svg></button></div>`).join("");
  markDups();
}
function markDups() {
  const dups = new Set([...dupFroms()].map((s) => s.toLowerCase())), seen = new Set();
  $$(".repl-row", $("#dict-repl")).forEach((row) => {
    const k = DS.replace[+row.dataset.k]?.from.trim().toLowerCase();
    row.classList.toggle("dup", !!k && dups.has(k) && seen.has(k));
    if (k) seen.add(k);
  });
  dictStatus();
}
$("#dict-repl").addEventListener("input", (e) => {
  const row = e.target.closest(".repl-row"), k = +row.dataset.k;
  DS.replace[k][e.target.dataset.k] = e.target.value;
  if (k === DS.replace.length - 1 && e.target.value) {       // typing into the empty last row adds a new empty row
    DS.replace.push({ from: "", to: "" });
    const n = document.createElement("div"); n.innerHTML = `<div class="repl-row" data-k="${k + 1}">
      <input class="field" data-k="from" placeholder="как распознано" spellcheck="false"><span class="arrow">→</span>
      <input class="field" data-k="to" placeholder="как правильно" spellcheck="false">
      <button class="icon-btn" type="button" data-act="del" title="Удалить" aria-label="Удалить" tabindex="-1"><svg class="ic"><use href="icons.svg#close"/></svg></button></div>`;
    $("#dict-repl").append(n.firstElementChild);
  }
  markDups();
});
$("#dict-repl").addEventListener("click", (e) => {
  const b = e.target.closest("[data-act=del]");
  if (!b) return;
  DS.replace.splice(+b.closest(".repl-row").dataset.k, 1);
  renderRepl();
});
$("#dict-repl").addEventListener("keydown", (e) => {
  if (e.key !== "Enter" || !e.target.matches("input") || e.ctrlKey) return;
  e.preventDefault();
  const row = e.target.closest(".repl-row");
  if (e.target.dataset.k === "from") return $("[data-k=to]", row).focus();
  $("input", row.nextElementSibling || row)?.focus();
});
$("#dict-context").addEventListener("input", dictStatus);

function collectDict() {
  const seen = new Set(), replace = [];
  for (const r of DS.replace) {
    const from = r.from.trim(), to = r.to.trim(), k = from.toLowerCase();
    if (!from || seen.has(k)) continue;           // duplicates are not saved
    seen.add(k); replace.push({ from, to });
  }
  return { terms: [...DS.terms], replace, context: $("#dict-context").value.trim() };
}
async function saveDict() {
  const pending = $("#dict-term-new").value.trim();
  if (pending) { addTerms(pending); $("#dict-term-new").value = ""; }
  if (dupFroms().size) { setDictTab("repl"); return toast("Уберите повторяющиеся замены — они подсвечены"); }
  $("#dict-save").disabled = true;
  try {
    for (let i = 0; ; i++) {
      const body = { ...(S.dict || {}), ...collectDict() };   // keep any fields the UI does not edit
      // terms and rules someone else (an agent through MCP) added since the sheet opened stay
      const base = DS.server || {}, cur = await api("/api/dictionary").catch(() => null);
      if (cur) {
        body.rev = cur.rev;
        const had = new Set((base.terms || []).map((t) => t.toLowerCase())), have = new Set(body.terms.map((t) => t.toLowerCase()));
        body.terms = body.terms.concat((cur.terms || []).filter((t) => !had.has(t.toLowerCase()) && !have.has(t.toLowerCase())));
        const hadR = new Set((base.replace || []).map((r) => r.from.toLowerCase())), haveR = new Set(body.replace.map((r) => r.from.toLowerCase()));
        body.replace = body.replace.concat((cur.replace || []).filter((r) => !hadR.has(r.from.toLowerCase()) && !haveR.has(r.from.toLowerCase())));
        if ((cur.context || "") !== (base.context || "") && body.context === (base.context || "")) body.context = cur.context;
      }
      let saved;
      try { saved = await api("/api/dictionary", { method: "PUT", body }); }
      catch (e) { if (e.status === 409 && i < 2) continue; throw e; }   // saved in between: merge again
      S.dict = { ...DICT_EMPTY, ...(saved && typeof saved === "object" ? saved : body) };
      S.dictOk = true;
      renderDictLine(); renderNav();
      $("#sheet-dict").hidden = true;
      if ($("#dict-apply").checked && S.doc?.status === "done") applyDictionary();
      else toast("Словарь сохранён");
      break;
    }
  } catch (e) { fail(e); } finally { $("#dict-save").disabled = false; }
}
function cancelDict() {
  if (!$("#sheet-dict").hidden && dictSnapshot() !== DS.base && !confirm("Отменить изменения?")) return;
  $("#sheet-dict").hidden = true;
}
$("#dict-save").addEventListener("click", saveDict);
$("#dict-cancel").addEventListener("click", cancelDict);
$("#sheet-dict").addEventListener("keydown", (e) => {
  if ((e.ctrlKey || e.metaKey) && (e.key.toLowerCase() === "s" || e.key === "Enter")) { e.preventDefault(); saveDict(); }
});

// ================================================================== boot

async function boot() {
  setDocSize(ls.get("docSize", 16));
  setRate(RATES.includes(ls.get("rate", 1)) ? ls.get("rate", 1) : 1);
  renderFollow();
  applyLayout(false);
  showNoDoc();
  // everything in parallel: /api/status can take up to ~1.5 s (device probe) and must not delay the list
  const pStatus = api("/api/status").then((s) => { S.status = s; renderBadge(); }).catch(() => {});
  const pSettings = api("/api/settings").then((s) => (S.settings = s)).catch(() => (S.settings = null));
  const pModels = api("/api/models").then((m) => { S.models = m; renderNav(); if (!S.doc) renderState(); }).catch(() => {});
  loadDict();
  loadGold();
  await refreshDocs();
  S.loaded = true;
  if (!ls.get("seen", null)) { S.docs.forEach((d) => seen.add(d.id)); ls.set("seen", [...seen]); listSig = ""; renderDocList(); }
  const last = ls.get("last", null);
  const first = S.docs.find((d) => d.id === last) || S.docs.find((d) => d.status === "done") || S.docs[0];
  if (first) await openDoc(first.id, { show: false });
  schedulePoll();
  await Promise.all([pStatus, pSettings, pModels]);
}
boot();
