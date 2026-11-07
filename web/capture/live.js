// Live coverage while filming. Small frames go to the server, where MASt3R-SLAM tracks them;
// back come the directions covered so far (16 sectors round the object) and a preview point
// cloud. If the live view can't run (busy, offline), recording carries on exactly as before.

const SIDE = 512; // longest side of a live frame (the tracker works at 512)
const EVERY_MS = 220; // ~4.5 frames a second
const MAX_BUFFERED = 300_000; // bytes queued on a slow link: skip frames rather than lag
export const SECTORS = 16;
const UNITS = 8000; // castlive/live/session.py POINT_UNITS

export function unpack(buf) {
  const n = new DataView(buf).getUint32(0, true);
  return { n, xyz: new Int16Array(buf, 4, 3 * n), rgb: new Uint8Array(buf, 4 + 6 * n, 3 * n) };
}

export class Live {
  constructor(video) {
    this.video = video;
    this.canvas = document.createElement("canvas");
    this.ws = null; this.state = null; this.stateAt = 0; this.points = null;
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
        if (s.status === "busy" || s.status === "refused") this.status = s.status;
        else { this.state = s; this.stateAt = Date.now(); }
      } else this.points = unpack(e.data);
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
    const k = SIDE / Math.max(v.videoWidth, v.videoHeight);
    const c = this.canvas;
    c.width = Math.round(v.videoWidth * k);
    c.height = Math.round(v.videoHeight * k);
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

/** The preview points, slowly turning, with the camera path. */
export function drawCloud(canvas, points, p, time) {
  const ctx = canvas.getContext("2d"), w = canvas.width, h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  const a = time / 5000, tilt = 0.5;
  const ca = Math.cos(a), sa = Math.sin(a), ct = Math.cos(tilt), st = Math.sin(tilt);
  const s = w / 2.4; // +-1.2 orbit radii across: the object, not the whole room
  const at = (x, y, z) => {
    const X = x * ca + z * sa, Z = -x * sa + z * ca;
    return [w / 2 + X * s, h / 2 - (y * ct - Z * st) * s];
  };
  if (points) {
    const { n, xyz, rgb } = points;
    for (let i = 0; i < n; i++) {
      const [px, py] = at(xyz[3 * i] / UNITS, xyz[3 * i + 1] / UNITS, xyz[3 * i + 2] / UNITS);
      ctx.fillStyle = `rgb(${rgb[3 * i]},${rgb[3 * i + 1]},${rgb[3 * i + 2]})`;
      ctx.fillRect(px - 1, py - 1, 2.5, 2.5);
    }
  }
  const cams = (p && p.cameras) || [];
  ctx.fillStyle = "rgba(255,255,255,.7)";
  cams.forEach(([x, y, z]) => { const [px, py] = at(x, y, z); ctx.fillRect(px - 1, py - 1, 2, 2); });
  if (cams.length) {
    const [px, py] = at(...cams[cams.length - 1]);
    ctx.fillStyle = "#3ddc84";
    ctx.beginPath(); ctx.arc(px, py, 5, 0, 2 * Math.PI); ctx.fill();
  }
}
