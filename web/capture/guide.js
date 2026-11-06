// Live guidance on the phone: how far round the object you have turned (gyroscope), how fast,
// and whether the picture is too dark or blurry. Works without sensors, just with less help.

// --- Turn tracking --------------------------------------------------------------------------
// How far round the object you have walked = how far the camera's facing direction has turned,
// seen from above. That direction comes from the phone's orientation (deviceorientation), so
// it does not depend on how a browser labels gyroscope axes (Safari's labelling made a summed
// gyroscope rate read about half the true turn) and it does not drift.

const DEG = Math.PI / 180;

// Compass direction (degrees) the camera looks along, or null when it points nearly straight
// down or up. W3C device orientation: R = Rz(alpha) Rx(beta) Ry(gamma); the camera looks along
// the device's -z axis, i.e. minus the third column of R.
export function cameraHeading(alpha, beta, gamma) {
  const cA = Math.cos(alpha * DEG), sA = Math.sin(alpha * DEG);
  const cB = Math.cos(beta * DEG), sB = Math.sin(beta * DEG);
  const cG = Math.cos(gamma * DEG), sG = Math.sin(gamma * DEG);
  const fx = -(cG * sA * sB + cA * sG);
  const fy = -(sA * sG - cA * cG * sB);
  if (Math.hypot(fx, fy) < 0.25) return null; // looking almost straight down: direction unclear
  return Math.atan2(fx, fy) / DEG;
}

export class TurnTracker {
  constructor() {
    this.total = 0; // degrees turned since reset (signed)
    this.rate = 0; // smoothed degrees per second
    this.samples = []; // [t seconds, heading degrees] at ~4 Hz, kept in the capture's metadata
    this.hasSensor = false;
    this.heading = 0; // unwrapped, since begin()
    this.lastDir = null;
    this.lastT = null;
    this.t0 = performance.now();
    this.handler = (e) => this.onOrientation(e);
  }

  // Must be called from a tap on iOS (permission prompt).
  async start() {
    try {
      for (const Ev of [window.DeviceOrientationEvent, window.DeviceMotionEvent]) {
        if (Ev && Ev.requestPermission && (await Ev.requestPermission()) !== "granted") return false;
      }
    } catch { return false; }
    window.addEventListener("deviceorientation", this.handler);
    return true;
  }

  stop() { window.removeEventListener("deviceorientation", this.handler); }

  reset() { this.total = 0; }

  begin() { this.t0 = performance.now(); this.samples = []; this.heading = 0; this.total = 0; }

  onOrientation(e) {
    if (e.alpha == null || e.beta == null || e.gamma == null) return;
    this.hasSensor = true;
    const now = performance.now();
    const dir = cameraHeading(e.alpha, e.beta, e.gamma);
    if (dir == null) { this.lastDir = null; return; }
    if (this.lastDir != null) {
      const d = ((dir - this.lastDir + 540) % 360) - 180; // shortest way round
      this.total += d;
      this.heading += d;
      const dt = (now - this.lastT) / 1000;
      if (dt > 0) this.rate = 0.9 * this.rate + 0.1 * Math.min(Math.abs(d) / dt, 360);
    }
    this.lastDir = dir;
    this.lastT = now;
    const t = (now - this.t0) / 1000;
    const prev = this.samples[this.samples.length - 1];
    if (!prev || t - prev[0] >= 0.25) this.samples.push([+t.toFixed(2), Math.round(this.heading)]);
  }

  fraction() { return Math.min(1, Math.abs(this.total) / 360); }
}

// --- Picture quality ------------------------------------------------------------------------
// Every half second, the centre of the frame (where the object is): mean brightness and edge
// strength (variance of the Laplacian). Sharpness is compared with the typical value of the last
// few seconds, since the right number depends on the object. Only sustained blur is reported:
// the server keeps the sharpest frames, so a moment of shake costs nothing.

const WINDOW = 12; // measurements (6 s) that set the typical sharpness
const SUSTAIN = 4; // blurry measurements in a row (2 s) before warning

export class QualityCheck {
  constructor(video) {
    this.video = video;
    this.canvas = document.createElement("canvas");
    this.canvas.width = 192; this.canvas.height = 144;
    this.ctx = this.canvas.getContext("2d", { willReadFrequently: true });
    this.recent = [];
    this.streak = 0;
  }

  measure() {
    const v = this.video;
    if (!v.videoWidth) return null;
    const { width: w, height: h } = this.canvas;
    // middle half of the picture in each direction
    this.ctx.drawImage(v, v.videoWidth / 4, v.videoHeight / 4, v.videoWidth / 2, v.videoHeight / 2, 0, 0, w, h);
    const px = this.ctx.getImageData(0, 0, w, h).data;
    const lum = new Float32Array(w * h);
    let sum = 0;
    for (let i = 0; i < w * h; i++) {
      lum[i] = 0.299 * px[4 * i] + 0.587 * px[4 * i + 1] + 0.114 * px[4 * i + 2];
      sum += lum[i];
    }
    let s = 0, s2 = 0, n = 0;
    for (let y = 1; y < h - 1; y++) {
      for (let x = 1; x < w - 1; x++) {
        const i = y * w + x;
        const l = 4 * lum[i] - lum[i - 1] - lum[i + 1] - lum[i - w] - lum[i + w];
        s += l; s2 += l * l; n++;
      }
    }
    const sharp = s2 / n - (s / n) ** 2;
    this.recent.push(sharp);
    if (this.recent.length > WINDOW) this.recent.shift();
    const sorted = [...this.recent].sort((a, b) => a - b);
    const typical = sorted[Math.floor(sorted.length / 2)];
    const soft = this.recent.length >= 4 && sharp < 0.4 * typical;
    this.streak = soft ? this.streak + 1 : 0;
    return { brightness: sum / (w * h), sharp, blurry: this.streak >= SUSTAIN };
  }
}
