// Watch a capture's live tracking: the preview points, the camera path and the 16 sectors of
// directions filmed so far, as the phone sends them. Polls the studio API once a second.

import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";

const cid = new URLSearchParams(location.search).get("c");
const stage = document.getElementById("stage");
const info = document.getElementById("info");
const UNITS = 8000; // castlive/live/session.py POINT_UNITS
const SECTORS = 16;

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(matchMedia("(prefers-color-scheme: dark)").matches ? 0x1b1b1b : 0xf2efe8);
const camera = new THREE.PerspectiveCamera(45, 1, 0.01, 100);
camera.position.set(2.2, 1.6, 2.2);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
scene.add(new THREE.GridHelper(4, 8, 0x888888, 0x888888));

const cloud = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.02, vertexColors: true }));
const path = new THREE.Points(new THREE.BufferGeometry(), new THREE.PointsMaterial({ size: 0.04, color: 0x2e9e5b }));
scene.add(cloud, path);

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);
renderer.setAnimationLoop(() => { controls.update(); renderer.render(scene, camera); });

function showPoints(buf) {
  const n = new DataView(buf).getUint32(0, true);
  const q = new Int16Array(buf, 4, 3 * n), c = new Uint8Array(buf, 4 + 6 * n, 3 * n);
  const xyz = new Float32Array(3 * n), rgb = new Float32Array(3 * n);
  for (let i = 0; i < 3 * n; i++) { xyz[i] = q[i] / UNITS; rgb[i] = c[i] / 255; }
  cloud.geometry.setAttribute("position", new THREE.BufferAttribute(xyz, 3));
  cloud.geometry.setAttribute("color", new THREE.BufferAttribute(rgb, 3));
}

function showSectors(p) {
  const svg = document.getElementById("sectors"), ns = "http://www.w3.org/2000/svg";
  svg.replaceChildren(...Array.from({ length: SECTORS }, (_, i) => {
    const a0 = ((i * 360) / SECTORS + 1.5) * (Math.PI / 180), a1 = (((i + 1) * 360) / SECTORS - 1.5) * (Math.PI / 180);
    const pt = (a) => `${(50 + 42 * Math.cos(a)).toFixed(2)} ${(50 + 42 * Math.sin(a)).toFixed(2)}`;
    const e = document.createElementNS(ns, "path");
    e.setAttribute("d", `M ${pt(a0)} A 42 42 0 0 1 ${pt(a1)}`);
    if (p.sectors[i] > 0) e.setAttribute("class", "on");
    return e;
  }));
}

let lastPointsAt = 0;
async function poll() {
  try {
    const r = await fetch(`/api/captures/${cid}/live`, { cache: "no-store" });
    if (r.ok) {
      const s = await r.json();
      const p = s.passes && s.passes[String(s.pass)];
      const parts = [s.status === "ended" ? "Finished" : s.status === "failed" ? `Failed: ${s.error}` : "Tracking"];
      if (p) {
        parts.push(`pass ${p.pass}: ${p.tracked}/${p.frames} frames placed${p.placed === false ? " (lost)" : ""}`);
        if (p.sectors) parts.push(`${Math.round(p.covered * 100)}% of the way round`, `heights ${p.elevation[0]}° to ${p.elevation[1]}°`);
        if (p.sectors) showSectors(p);
        if (p.cameras) path.geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array(p.cameras.flat()), 3));
      }
      if (s.fps) parts.push(`${s.fps} frames/s`);
      info.textContent = parts.join(" · ");
      if (s.updated && s.updated !== lastPointsAt) {
        lastPointsAt = s.updated;
        const b = await fetch(`/api/captures/${cid}/live/points`, { cache: "no-store" });
        if (b.ok) showPoints(await b.arrayBuffer());
      }
      if (s.status === "ended" || s.status === "failed") return;
    }
  } catch { /* try again */ }
  setTimeout(poll, 1000);
}
poll();
