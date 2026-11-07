// The phone capture flow: consent → mode → checklist → record (two passes) → send → status.

import { store, Uploader, serverStatus, projectInfo } from "./store.js";
import { TurnTracker, QualityCheck } from "./guide.js";
import { Live, drawSectors, drawCloud } from "./live.js";

const $ = (id) => document.getElementById(id);
const token = location.pathname.split("/").filter(Boolean).pop();
const CHUNK = 4 * 1024 * 1024; // at most this per piece: kinder to weak mobile links

// --- Text -----------------------------------------------------------------------------------
let strings = {};
async function loadStrings() {
  const en = await (await fetch("/i18n/en.json")).json();
  const lang = (navigator.language || "en").slice(0, 2);
  let local = {};
  if (lang !== "en") local = await fetch(`/i18n/${lang}.json`).then((r) => (r.ok ? r.json() : {})).catch(() => ({}));
  strings = { ...en, ...local };
  document.documentElement.lang = lang;
  document.querySelectorAll("[data-t]").forEach((el) => { el.textContent = t(el.dataset.t); });
}
function t(key, vars = {}) {
  return (strings[key] || key).replace(/\{(\w+)\}/g, (_, k) => vars[k] ?? "");
}

// --- Screens --------------------------------------------------------------------------------
function show(name) {
  document.querySelectorAll("body > main").forEach((m) => { m.hidden = m.id !== `s-${name}`; });
}
function message(text) { $("message").textContent = text; show("message"); }

// --- State ----------------------------------------------------------------------------------
let info = null;
let mode = "object";
let stream = null, recorder = null, recording = null; // recording: the stored capture record
let pass = 1, startedAt = 0, timerId = 0, checkId = 0;
const turn = new TurnTracker();
let quality = null;
let waitingFor = null; // capture id whose result the status screen shows
let live = null; // live coverage while filming (null: off)
const FULL = 15 / 16; // live coverage that counts as "all the way round"

const uploader = new Uploader(onUpload);

function onUpload(p) {
  if (p.id === waitingFor) {
    if (p.state === "sending" && p.total) setUpload(p.sent / p.total);
    if (p.state === "offline") $("upload-note").textContent = t("upload_offline");
    if (p.state === "done") { setUpload(1); pollStatus(p.id); }
    if (p.state === "failed") {
      if (recorder && recorder.state === "recording") { recording.cancelled = true; stopRecording(); }
      closeCamera();
      showStatus(t("status_failed", { error: p.error }));
    }
  } else if (p.state === "sending" && p.total) {
    $("pending").hidden = false;
    $("pending").textContent = t("pending_banner", { pct: Math.round((100 * p.sent) / p.total) });
  } else if (p.state === "done" || p.state === "failed") {
    $("pending").hidden = true;
  }
}

function setUpload(f) {
  $("upload-bar").style.width = `${Math.round(f * 100)}%`;
  $("upload-pct").textContent = t("upload_progress", { pct: Math.round(f * 100) });
}

// --- Start ----------------------------------------------------------------------------------
async function init() {
  await loadStrings();
  try {
    info = await projectInfo(token);
  } catch {
    if (!navigator.onLine && (await store.allCaptures()).length) {
      message(t("upload_offline")); uploader.kick(); return;
    }
    message(t("invalid")); return;
  }
  mode = info.mode;
  $("project-name").textContent = t("project", { name: info.project });
  $("consent-1").textContent = t("consent_1", { name: info.project });
  uploader.kick(); // anything left from an earlier visit
  show("consent");
}

$("consent").addEventListener("change", (e) => { $("consent-next").disabled = !e.target.checked; });
$("consent-next").addEventListener("click", () => show("mode"));
document.querySelectorAll("[data-mode]").forEach((b) =>
  b.addEventListener("click", () => {
    mode = b.dataset.mode;
    document.body.className = `mode-${mode}`;
    show("check");
  }),
);

$("start-camera").addEventListener("click", async () => {
  if (!navigator.mediaDevices || !window.MediaRecorder) { message(t("no_camera")); return; }
  await turn.start(); // asks for motion permission on iOS; fine if refused
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: "environment" }, width: { ideal: 3840 }, height: { ideal: 2160 }, frameRate: { ideal: 30 } },
    });
  } catch {
    message(t("camera_denied")); return;
  }
  $("preview").srcObject = stream;
  quality = new QualityCheck($("preview"));
  pass = 1;
  resetRecordUi();
  show("record");
});

function resetRecordUi() {
  $("rec").hidden = false;
  $("rec").disabled = false;
  $("round-done").hidden = true;
  $("finish").hidden = true;
  $("flip").hidden = true;
  $("warning").hidden = true;
  $("timer").textContent = "0:00";
  $("pass-label").textContent = t("pass_label", { n: 1 });
  $("hint").textContent = hintFor(1);
  setRing(0);
}

function hintFor(p) {
  return t(`rec_hint_${mode === "object" ? "object" : "insitu"}_${p}`);
}
function setRing(f) { $("ring").style.strokeDashoffset = String(264 * (1 - f)); }

// --- Recording ------------------------------------------------------------------------------
// Best quality the phone can record. HEVC first: about twice the detail of H.264 per bit.
function pickMime() {
  const options = ["video/mp4;codecs=hvc1", "video/mp4;codecs=hev1", "video/mp4;codecs=avc1",
    "video/mp4", "video/webm;codecs=vp9", "video/webm;codecs=vp8", "video/webm"];
  return options.find((m) => MediaRecorder.isTypeSupported(m)) || "";
}

// Record as much detail as the phone gives: fine, low-contrast texture is what the encoder
// throws away first, and it is what reconstruction needs. The server can always drop data.
const MAX_BITS = 80e6;

// Android Chrome can lock focus, exposure and white balance (Safari ignores this). Locked,
// frames don't change brightness or colour mid-walk and focus doesn't hunt. Best effort.
async function lockCamera(track) {
  const applied = {};
  const caps = track.getCapabilities ? track.getCapabilities() : {};
  // Only "single-shot": measure once, then hold. ("manual" without a focus distance or colour
  // temperature can leave the lens at infinity or the colours wrong.)
  for (const key of ["focusMode", "exposureMode", "whiteBalanceMode"]) {
    if (!(caps[key] || []).includes("single-shot")) continue;
    try {
      await track.applyConstraints({ advanced: [{ [key]: "single-shot" }] });
      applied[key] = "single-shot";
    } catch { /* not supported on this phone */ }
  }
  return applied;
}

let wakeLock = null;
async function keepAwake() {
  try { wakeLock = await navigator.wakeLock.request("screen"); } catch { warn(t("warn_screen"), 4000); }
}

$("rec").addEventListener("click", async () => {
  if ($("rec").disabled) return; // a second tap while the camera is being set up
  $("rec").disabled = true;
  const mime = pickMime();
  const s = stream.getVideoTracks()[0].getSettings();
  const bits = Math.min(MAX_BITS, Math.max(12e6, (s.width || 1920) * (s.height || 1080) * (s.frameRate || 30) * 0.3));
  const locks = await lockCamera(stream.getVideoTracks()[0]);
  recorder = new MediaRecorder(stream, mime ? { mimeType: mime, videoBitsPerSecond: bits } : { videoBitsPerSecond: bits });
  recording = {
    id: crypto.randomUUID(), token, mode, mime: (recorder.mimeType || mime || "video/webm").split(";")[0],
    device: navigator.userAgent, total: null, created: Date.now(), created_on_server: false,
    meta: { mode, markers: [], sensor: false,
      camera: { width: s.width, height: s.height, fps: s.frameRate, codec: recorder.mimeType || mime, requested_bps: bits, locks } },
  };
  await store.putCapture(recording);
  waitingFor = recording.id; // so a refusal from the server shows even while recording

  const chunk = Math.min(CHUNK, info.chunk_bytes);
  let n = 0, size = 0, buf = new Blob([]), saving = Promise.resolve();
  const save = (blob) => {
    const i = n++;
    // Store a copy of the bytes, not the slice: Safari can persist a whole file-backed blob when
    // asked to store a slice of it, which made one piece far bigger than 4 MB.
    saving = saving
      .then(async () => store.putChunk(recording.id, i, await blob.arrayBuffer()))
      .then(() => uploader.kick());
  };
  recorder.ondataavailable = (e) => {
    if (!e.data || !e.data.size) return;
    buf = new Blob([buf, e.data]);
    size += e.data.size;
    while (buf.size >= chunk) { save(buf.slice(0, chunk)); buf = buf.slice(chunk); }
    if (size > 0.97 * info.max_bytes) stopRecording(); // the server's size limit
  };
  recorder.onstop = async () => {
    if (buf.size) save(buf);
    await saving;
    if (recording.cancelled) {
      const c = await store.getCapture(recording.id);
      await store.delCapture(recording.id);
      if (c && c.created_on_server) fetch(`/api/${token}/captures/${recording.id}/cancel`, { method: "POST" }).catch(() => {});
      return;
    }
    recording.meta.sensor = turn.hasSensor;
    recording.meta.heading = turn.samples;
    await store.patchCapture(recording.id, { total: n, meta: recording.meta });
    setUpload(0);
    show("upload");
    uploader.kick();
  };
  recorder.start(1000);
  startLive();
  keepAwake();
  turn.begin();
  startedAt = performance.now();
  mark("start");
  $("rec").hidden = true;
  $("round-done").hidden = false;
  timerId = setInterval(tick, 250);
  checkId = setInterval(checkPicture, 500);
});

// --- Live coverage --------------------------------------------------------------------------
function startLive() {
  live = new Live($("preview"));
  live.onUpdate = () => {
    if (live.status === "busy") warn(t("live_busy"), 4000);
    const cov = live.current();
    $("top").classList.toggle("live", !!cov);
    $("live-view").hidden = !cov;
    if (cov) drawSectors($("sectors"), cov);
  };
  live.start(token, recording.id);
  recording.meta.live = true;
  requestAnimationFrame(function draw(time) {
    if (!live) return;
    const cov = live.current();
    if (cov) drawCloud($("live-view"), live.points, cov, time);
    requestAnimationFrame(draw);
  });
}

function stopLive() {
  if (live) live.stop();
  live = null;
  $("top").classList.remove("live");
  $("live-view").hidden = true;
}

function elapsed() { return (performance.now() - startedAt) / 1000; }
function mark(event) {
  // `turned`: degrees the motion sensor measured this pass, to check the ring against reality
  const m = { t: +elapsed().toFixed(2), event };
  if (turn.hasSensor) m.turned = Math.round(Math.abs(turn.total));
  recording.meta.markers.push(m);
}

function tick() {
  const sec = elapsed();
  $("timer").textContent = `${Math.floor(sec / 60)}:${String(Math.floor(sec % 60)).padStart(2, "0")}`;
  // Without a gyroscope, assume about 40 s per loop.
  const passStart = recording.meta.markers.filter((m) => m.event === "start" || m.event === "pass2_start").pop().t;
  // Round complete when the live coverage or the motion sensor says so (without a sensor, the
  // clock): the live estimate can be a sector or two short while it settles.
  const cov = live && live.current();
  const moved = turn.hasSensor ? turn.fraction() : Math.min(1, (sec - passStart) / 40);
  const f = cov ? Math.max(Math.min(1, cov.covered / FULL), moved) : moved;
  setRing(f);
  if (cov && cov.placed === false) warn(t("warn_lost"));
  if (f >= 1 && pass === 1 && $("flip").hidden) endPass1();
  if (f >= 1 && pass === 2 && $("finish").hidden) {
    $("finish").hidden = false;
    $("round-done").hidden = true;
    $("hint").textContent = t("all_done");
  }
  if (turn.hasSensor && turn.rate > 40) warn(t("warn_fast"));
  if (sec >= info.max_seconds - 1) stopRecording();
}

function checkPicture() {
  const q = quality.measure();
  if (!q) return;
  if (q.brightness < 45) warn(t("warn_dark"));
  else if (q.brightness > 235) warn(t("warn_bright"));
  else if (q.blurry) warn(t("warn_blur"));
}

let warnTimer = 0;
function warn(text, ms = 1200) {
  $("warning").textContent = text;
  $("warning").hidden = false;
  clearTimeout(warnTimer);
  warnTimer = setTimeout(() => { $("warning").hidden = true; }, ms);
}

// End of pass 1: a full-screen card in both modes, so the change is obvious and a stray tap
// can't skip a pass. Object: turn it over. In the ground: walk round again, higher.
function endPass1() {
  mark("pass1_end");
  const object = mode === "object";
  $("pass-icon").textContent = object ? "⟲" : "✓";
  $("pass-icon").className = object ? "flip-anim" : "flip-anim still";
  $("pass-title").textContent = t(object ? "flip_title" : "pass1_done_title");
  $("pass-body").textContent = t(object ? "flip_body" : "pass1_done_body");
  $("flip-done").textContent = t(object ? "flip_done" : "pass2_go");
  $("pass-finish").hidden = object; // in the ground, one loop can be enough
  $("round-done").hidden = true;
  $("flip").hidden = false;
}

$("pass-finish").addEventListener("click", stopRecording);

// After a pass changes, hide "gone all the way round" briefly: a double tap must not skip a pass.
function pauseRoundButton() {
  $("round-done").hidden = true;
  setTimeout(() => { if (pass === 2 && $("finish").hidden) $("round-done").hidden = false; }, 2500);
}

function startPass2() {
  mark("pass2_start");
  pass = 2;
  pauseRoundButton();
  turn.reset();
  if (live) live.setPass(2);
  $("flip").hidden = true;
  $("pass-label").textContent = t("pass_label", { n: 2 });
  $("hint").textContent = hintFor(2);
  setRing(0);
}

$("flip-done").addEventListener("click", startPass2);

$("round-done").addEventListener("click", () => {
  // Trust the person over the sensor. (A confirm() here froze the page while recording went on.)
  mark("round_button");
  if (pass === 1) endPass1();
  else { $("finish").hidden = false; $("round-done").hidden = true; $("hint").textContent = t("all_done"); }
});

$("finish").addEventListener("click", stopRecording);

$("cancel").addEventListener("click", () => {
  if (recorder && recorder.state === "recording") {
    if (!confirm(t("cancel_confirm"))) return;
    recording.cancelled = true;
    stopRecording();
  }
  closeCamera();
  show("check");
});

function stopRecording() {
  if (!recorder || recorder.state !== "recording") return;
  mark("stop");
  clearInterval(timerId);
  clearInterval(checkId);
  recorder.stop();
  stopLive();
  if (wakeLock) wakeLock.release().catch(() => {});
  if (!recording.cancelled) closeCamera();
}

function closeCamera() {
  if (stream) stream.getTracks().forEach((tr) => tr.stop());
  stream = null;
}

// --- After sending --------------------------------------------------------------------------
function showStatus(text) {
  $("status-text").textContent = text;
  show("status");
}

async function pollStatus(id) {
  showStatus(t("status_received"));
  for (let i = 0; i < 360; i++) { // up to 30 minutes: reconstruction takes a while
    try {
      const s = await serverStatus(token, id);
      if (s.status === "checked") showStatus(t("status_checked"));
      if (s.status === "processing") {
        const p = s.progress || {};
        const pct = Math.round((p.fraction || 0) * 100);
        const mins = Math.max(1, Math.round(p.eta_seconds / 60));
        showStatus(p.eta_seconds == null ? t("status_checked")
          : `${t("status_checked")} ${t("status_progress", { pct, mins })}`);
      }
      if (s.status === "done") return showStatus(t("status_done"));
      if (s.status === "failed") return showStatus(t("status_model_failed", { error: s.error }));
      if (s.status === "rejected") return showStatus(t("status_rejected", { error: s.error }));
    } catch { /* keep trying */ }
    await new Promise((r) => setTimeout(r, 5000));
  }
}

$("again").addEventListener("click", () => { waitingFor = null; show("mode"); });

window.addEventListener("beforeunload", (e) => {
  if (uploader.running || (recorder && recorder.state === "recording")) {
    e.preventDefault();
    e.returnValue = t("leave_warning");
  }
});

init();
