// 3D viewer for one capture's model: orbit, soft or raking light (low, rotatable: shows wedges).

import * as THREE from "./vendor/three.module.min.js";
import { OrbitControls } from "./vendor/OrbitControls.js";
import { GLTFLoader } from "./vendor/GLTFLoader.js";
import { loadSplats } from "./splats.js";

const cid = new URLSearchParams(location.search).get("c");
const stage = document.getElementById("stage");
const info = document.getElementById("info");

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setPixelRatio(window.devicePixelRatio);
stage.prepend(renderer.domElement);
const scene = new THREE.Scene();
scene.background = new THREE.Color(matchMedia("(prefers-color-scheme: dark)").matches ? 0x1b1b1b : 0xf2efe8);
const camera = new THREE.PerspectiveCamera(35, 1, 0.001, 100);
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;

const ambient = new THREE.AmbientLight(0xffffff, 0.6);
const key = new THREE.DirectionalLight(0xffffff, 1.6);
const fill = new THREE.HemisphereLight(0xffffff, 0x444444, 0.6);
scene.add(ambient, key, fill);

let model = null, splats = null;
const shaded = [], flat = [];

function setLight() {
  const mode = document.getElementById("light").value;
  const a = (Number(document.getElementById("angle").value) * Math.PI) / 180;
  const raking = mode === "raking";
  const elev = raking ? 0.12 : 0.9; // raking: light skims the surface
  key.position.set(Math.cos(a), elev, Math.sin(a)).multiplyScalar(1);
  key.intensity = raking ? 2.4 : 1.6;
  ambient.intensity = raking ? 0.08 : 0.6;
  fill.intensity = raking ? 0.05 : 0.6;
  if (model) model.traverse((o) => { if (o.isMesh) o.material = mode === "flat" ? flat[o.id] : shaded[o.id]; });
}
function setShow() {
  const splat = document.getElementById("show").value === "splat" && splats;
  if (model) model.visible = !splat;
  if (splats) splats.mesh.visible = !!splat;
  // the splat carries its own colours and light: the light controls are for the mesh
  for (const id of ["light", "angle"]) document.getElementById(id).disabled = !!splat;
}
document.getElementById("show").addEventListener("change", setShow);
document.getElementById("light").addEventListener("change", setLight);
document.getElementById("angle").addEventListener("input", setLight);

function resize() {
  const { clientWidth: w, clientHeight: h } = stage;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(stage);

async function load() {
  const glb = `/api/captures/${cid}/model.glb`;
  document.getElementById("download").href = glb;
  const [gltf, report] = await Promise.all([
    new GLTFLoader().loadAsync(glb),
    fetch(`/api/captures/${cid}/report`).then((r) => (r.ok ? r.json() : null)),
  ]);
  model = gltf.scene;
  model.traverse((o) => {
    if (!o.isMesh) return;
    // the model's own normals run smoothly across texture seams: compute only if missing
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    // colour from a texture (photos) or per vertex (points), whichever the model has
    const look = { map: o.material.map || null, vertexColors: !!o.geometry.attributes.color, side: THREE.DoubleSide };
    shaded[o.id] = new THREE.MeshStandardMaterial({ ...look, roughness: 1, metalness: 0 });
    flat[o.id] = new THREE.MeshBasicMaterial(look);
    o.material = shaded[o.id];
  });
  scene.add(model);
  const box = new THREE.Box3().setFromObject(model);
  const size = box.getSize(new THREE.Vector3()).length();
  const centre = box.getCenter(new THREE.Vector3());
  controls.target.copy(centre);
  camera.position.copy(centre).add(new THREE.Vector3(0.6, 0.5, 0.8).multiplyScalar(size));
  camera.near = size / 100; camera.far = size * 20; camera.updateProjectionMatrix();
  setLight();
  info.textContent = report ? summary(report) : "";
  splats = await loadSplats(`/api/captures/${cid}/splat.bin`).catch(() => null);
  if (splats) {
    splats.mesh.visible = false;
    scene.add(splats.mesh);
    document.getElementById("show-label").hidden = false;
    const dl = document.getElementById("download-splat");
    dl.href = `/api/captures/${cid}/splat.ply`;
    dl.hidden = false;
    setShow();
  }
}

function summary(r) {
  const parts = [];
  if (r.poses) {
    const passes = Object.entries(r.poses.passes).map(([p, v]) => `${p}: ${v.placed}/${v.frames}`).join(", ");
    parts.push(`Frames placed ${r.poses.placed}/${r.poses.frames} (${passes})`);
    if (r.poses.overlap && "joined" in r.poses) parts.push(`${r.poses.overlap.shared || 0} points seen from both sides: ${r.poses.joined ? "joined" : "NOT joined"}`);
  }
  if (r.mesh) parts.push(`${r.mesh.triangles.toLocaleString()} triangles${r.mesh.open ? ", open where unseen" : ""}`);
  const q = r.quality || {};
  if ("outline_missed" in q) parts.push(`outline covered ${(100 * (1 - q.outline_missed)).toFixed(1)}% (worst view ${(100 * (1 - q.outline_missed_worst)).toFixed(0)}%)`);
  if ("point_distance_median" in q) parts.push(`surface within ${(100 * q.point_distance_median).toFixed(2)}% of size of the solved points (median; 90%: ${(100 * q.point_distance_p90).toFixed(1)}%)`);
  if (r.splat && r.splat.psnr_held_out) parts.push(`splat ${r.splat.psnr_held_out} dB on ${r.splat.held_out} unseen photos`);
  if (r.mesh && !r.mesh.scale_known) parts.push("size not known (no marker card)");
  const secs = Object.values(r.seconds || {}).reduce((a, b) => a + b, 0);
  if (secs) parts.push(`${Math.round(secs / 60)} min`);
  return parts.join(" · ");
}

renderer.setAnimationLoop(() => {
  controls.update();
  if (splats && splats.mesh.visible) splats.sort(camera);
  renderer.render(scene, camera);
});
load().catch((e) => { info.textContent = `Could not load the model: ${e.message || e}`; });
