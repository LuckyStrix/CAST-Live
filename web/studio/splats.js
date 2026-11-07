// The trained gaussian splat, drawn as what it is: flat discs (2D gaussians) in 3D, each an
// oriented quad fading as exp(-r²/2), blended back to front. File layout: per disc 16 float32
// (castlive/recon/splat.py write_view): centre xyz, opacity | axis 1 xyz, red | axis 2 xyz,
// green | blue, 0, 0, 0. The discs live in a float texture; only their drawing order changes
// as the camera moves.

import * as THREE from "./vendor/three.module.min.js";

const WIDTH = 2048; // texture width in texels (4 per disc)
const EXTENT = 3; // draw each disc out to 3 sigma

const vertexShader = `
  precision highp sampler2D;
  uniform sampler2D discs;
  in float order;
  out vec2 local;
  out vec4 colour;
  vec4 texel(int i) { return texelFetch(discs, ivec2(i % ${WIDTH}, i / ${WIDTH}), 0); }
  void main() {
    int i = int(order) * 4;
    vec4 a = texel(i), b = texel(i + 1), c = texel(i + 2), d = texel(i + 3);
    local = position.xy * ${EXTENT}.0;
    colour = vec4(b.w, c.w, d.x, a.w);
    vec3 p = a.xyz + local.x * b.xyz + local.y * c.xyz;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(p, 1.0);
  }`;

const fragmentShader = `
  in vec2 local;
  in vec4 colour;
  void main() {
    float alpha = colour.a * exp(-0.5 * dot(local, local));
    if (alpha < 1.0 / 255.0) discard;
    gl_FragColor = vec4(colour.rgb * alpha, alpha); // premultiplied
  }`;

export async function loadSplats(url) {
  const res = await fetch(url);
  if (!res.ok) return null;
  const data = new Float32Array(await res.arrayBuffer());
  const n = data.length / 16;
  const rows = Math.ceil((n * 4) / WIDTH);
  const texData = new Float32Array(rows * WIDTH * 4);
  texData.set(data);
  const tex = new THREE.DataTexture(texData, WIDTH, rows, THREE.RGBAFormat, THREE.FloatType);
  tex.needsUpdate = true;

  const geometry = new THREE.InstancedBufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(new Float32Array([-1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 1, 0]), 3));
  geometry.setIndex([0, 1, 2, 0, 2, 3]);
  const order = new Float32Array(n);
  const orderAttr = new THREE.InstancedBufferAttribute(order, 1);
  orderAttr.setUsage(THREE.DynamicDrawUsage);
  geometry.setAttribute("order", orderAttr);
  geometry.instanceCount = n;

  const material = new THREE.ShaderMaterial({
    glslVersion: THREE.GLSL3, vertexShader, fragmentShader, uniforms: { discs: { value: tex } },
    transparent: true, depthWrite: false, side: THREE.DoubleSide,
    blending: THREE.CustomBlending, blendSrc: THREE.OneFactor, blendDst: THREE.OneMinusSrcAlphaFactor,
  });
  const mesh = new THREE.Mesh(geometry, material);
  mesh.frustumCulled = false;

  const centres = new Float32Array(n * 3);
  for (let i = 0; i < n; i++) centres.set(data.subarray(i * 16, i * 16 + 3), i * 3);
  const box = new THREE.Box3();
  for (let i = 0; i < n; i++) box.expandByPoint(new THREE.Vector3(centres[3 * i], centres[3 * i + 1], centres[3 * i + 2]));

  // Back to front along the view direction: a counting sort on 16-bit depth (fast for ~1M).
  const depth = new Float32Array(n), key = new Uint16Array(n), counts = new Uint32Array(65536);
  const last = new THREE.Matrix4();
  function sort(camera) {
    camera.updateMatrixWorld();
    if (last.equals(camera.matrixWorld)) return;
    last.copy(camera.matrixWorld);
    const dir = new THREE.Vector3();
    camera.getWorldDirection(dir);
    let lo = Infinity, hi = -Infinity;
    for (let i = 0; i < n; i++) {
      const d = centres[3 * i] * dir.x + centres[3 * i + 1] * dir.y + centres[3 * i + 2] * dir.z;
      depth[i] = d;
      if (d < lo) lo = d;
      if (d > hi) hi = d;
    }
    const k = 65535 / Math.max(hi - lo, 1e-9);
    counts.fill(0);
    for (let i = 0; i < n; i++) counts[(key[i] = 65535 - ((depth[i] - lo) * k) | 0)]++; // far first
    for (let i = 1; i < 65536; i++) counts[i] += counts[i - 1];
    for (let i = n - 1; i >= 0; i--) order[--counts[key[i]]] = i;
    orderAttr.needsUpdate = true;
  }

  return { mesh, sort, box, count: n };
}
