// Live coverage while filming. Small frames go to the server, where MASt3R-SLAM tracks them;
// back come the directions covered so far (16 sectors round the object) and, for the latest
// frame, how well each part of the picture has been filmed, painted over the camera view.
// If the live view can't run (busy, offline), recording carries on exactly as before.

const SIDE = 512; // longest side of a live frame (the tracker works at 512)
const EVERY_MS = 220; // ~4.5 frames a second
const MAX_BUFFERED = 300_000; // bytes queued on a slow link: skip frames rather than lag
export const SECTORS = 16;
// Overlay colours by how many directions a spot has been filmed from (castlive/live/slam.py).
const PAINT = [null, [230, 60, 60], [240, 170, 40], [40, 200, 110]]; // 1, 2, 3+ directions
const ALPHA = 120;

export class Live {
  constructor(video) {
    this.video = video;
    this.canvas = document.createElement("canvas");
    this.ws = null; this.state = null; this.stateAt = 0; this.overlay = null; this.overlayAt = 0;
    this.pass = 1; this.status = "off"; this.timer = 0; this.sending = false;
    this.onUpdate = () => {};
  }

  start(token, cid) {
    const proto = location.protocol === "https:" ? "wss" : "ws";
    try { this.ws = new WebSocket(`${proto}://${location.host}/live/${token}/${cid}`); } catch { return; }
    this.ws.binaryType = "arraybuffer";
    this.status = "connecting";
    this.ws.onopen = () => { this.status = "on"; this.timer = setInterval(() => this.send(), EVERY_MS); };
    this.ws.onmessage = (e) => {
      if (typeof e.data === "string") {
        let s;
        try { s = JSON.parse(e.data); } catch { return; }
        if (s.status === "busy" || s.status === "refused") { this.status = s.status; this.why = s.why; }
        else { this.state = s; this.stateAt = Date.now(); }
      } else { this.overlay = e.data; this.overlayAt = Date.now(); }
      this.onUpdate();
    };
    this.ws.onclose = () => {
      clearInterval(this.timer);
      if (this.status === "on" || this.status === "connecting") this.status = "closed";
      this.onUpdate();
    };
  }

  send() {
    const v = this.video, ws = this.ws;
    if (this.sending || !ws || ws.readyState !== 1 || ws.bufferedAmount > MAX_BUFFERED || !v.videoWidth) return;
    // Sides in multiples of 16: the tracker would crop otherwise, and the overlay must line up.
    const k = SIDE / Math.max(v.videoWidth, v.videoHeight);
    const c = this.canvas;
    c.width = Math.max(16, Math.round((v.videoWidth * k) / 16) * 16);
    c.height = Math.max(16, Math.round((v.videoHeight * k) / 16) * 16);
    c.getContext("2d").drawImage(v, 0, 0, c.width, c.height);
    this.sending = true;
    c.toBlob((b) => { this.sending = false; if (b && ws.readyState === 1) ws.send(b); }, "image/jpeg", 0.8);
  }

  setPass(n) {
    this.pass = n;
    if (this.ws && this.ws.readyState === 1) this.ws.send(JSON.stringify({ pass: n }));
  }

  stop() {
    clearInterval(this.timer);
    const ws = this.ws;
    if (ws && ws.readyState === 1) {
      ws.send(JSON.stringify({ end: true }));
      setTimeout(() => ws.close(), 3000);
    }
  }

  /** This pass's latest state while it is fresh: {frames, tracked, placed, ...} or null. */
  info() {
    const s = this.state;
    if (!s || this.status !== "on" || Date.now() - this.stateAt > 4000) return null;
    return (s.passes && s.passes[String(this.pass)]) || null;
  }

  /** This pass's coverage once there is some: {covered, sectors, cameras, ...} or null. */
  current() {
    const p = this.info();
    return p && p.sectors ? p : null;
  }
}

// --- Drawing ----------------------------------------------------------------------------------

/** The 16 sectors as arcs on the progress ring (an <svg> group, viewBox 0..100, ring r=42),
 * turned so the sector the phone is in now sits at the bottom, nearest the person. */
export function drawSectors(group, p) {
  const ns = "http://www.w3.org/2000/svg";
  if (group.childElementCount !== SECTORS) {
    group.replaceChildren(...Array.from({ length: SECTORS }, () => document.createElementNS(ns, "path")));
  }
  const last = p.cameras && p.cameras[p.cameras.length - 1];
  const here = last ? (Math.atan2(last[2], last[0]) * 180) / Math.PI : 0;
  const step = 360 / SECTORS;
  [...group.children].forEach((path, i) => {
    // the ring's svg is turned -90deg in CSS: 180 here is the bottom of the screen
    const a0 = ((i * step - here + 180 + 1.5) * Math.PI) / 180;
    const a1 = (((i + 1) * step - here + 180 - 1.5) * Math.PI) / 180;
    const pt = (a) => `${(50 + 42 * Math.cos(a)).toFixed(2)} ${(50 + 42 * Math.sin(a)).toFixed(2)}`;
    path.setAttribute("d", `M ${pt(a0)} A 42 42 0 0 1 ${pt(a1)}`);
    path.setAttribute("class", p.sectors[i] > 0 ? "sector on" : "sector");
  });
}

/** Paint the coverage of the latest frame on `canvas`, which sits over the video with the same
 * object-fit, so cell (x, y) lands on the same part of the picture. Clears it when there is
 * nothing fresh to show. */
export function drawOverlay(canvas, live) {
  const buf = live.overlay;
  const fresh = buf && Date.now() - live.overlayAt < 2500;
  const dv = fresh ? new DataView(buf) : null;
  const w = dv ? dv.getUint16(0, true) : 0, h = dv ? dv.getUint16(2, true) : 0;
  if (!w || !h) { canvas.hidden = true; return; }
  canvas.width = w; canvas.height = h;
  const cells = new Uint8Array(buf, 4, w * h);
  const img = new ImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const c = PAINT[Math.min(cells[i], 3)];
    if (!c) continue;
    img.data.set([c[0], c[1], c[2], ALPHA], 4 * i);
  }
  canvas.getContext("2d").putImageData(img, 0, 0);
  canvas.hidden = false;
}
