// Studio: projects, invite links, captures. Plain DOM, hash routing (#p=<id>).

const $ = (sel) => document.querySelector(sel);

async function api(method, url, body) {
  const r = await fetch(url, {
    method, headers: body ? { "Content-Type": "application/json" } : {},
    body: body ? JSON.stringify(body) : undefined,
  });
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new Error((data && (data.detail || data.error)) || `HTTP ${r.status}`);
  return data;
}

function el(tag, attrs = {}, ...children) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") e.className = v;
    else if (k.startsWith("on")) e.addEventListener(k.slice(2), v);
    else e.setAttribute(k, v);
  }
  for (const c of children) e.append(c instanceof Node ? c : document.createTextNode(c ?? ""));
  return e;
}

const when = (s) => new Date(s * 1000).toLocaleString();

async function health() {
  try {
    const s = await api("GET", "/api/status");
    const parts = [
      el("span", { class: s.worker_ok ? "ok" : "bad" }, s.worker_ok ? "upload checks running" : "upload checks NOT running"),
      " · ",
      el("span", { class: s.reconstructor_ok ? "ok" : "bad" }, s.reconstructor_ok ? "GPU reconstruction running" : "GPU reconstruction NOT running"),
      " · ",
      el("span", { class: s.disk_ok ? "ok" : "bad" }, `${s.disk_free_gb} GB free`),
      ` · ${s.user}`,
    ];
    $("#health").replaceChildren(...parts);
  } catch (e) {
    $("#health").replaceChildren(el("span", { class: "bad" }, e.message));
  }
}

async function showHome() {
  $("#home").hidden = false; $("#project").hidden = true;
  const rows = await api("GET", "/api/projects");
  $("#projects tbody").replaceChildren(...rows.map((p) =>
    el("tr", { class: "link", onclick: () => { location.hash = `p=${p.id}`; } },
      el("td", {}, p.name), el("td", {}, p.mode), el("td", {}, String(p.n)))));
}

let current = null;
async function showProject(id) {
  current = id;
  $("#home").hidden = true; $("#project").hidden = false;
  const d = await api("GET", `/api/projects/${id}`);
  $("#p-name").textContent = d.project.name;
  $("#invites").replaceChildren(...d.invites.map((i) => {
    const status = i.revoked ? "revoked" : !i.active ? "expired" : `active${i.expires ? `, expires ${when(i.expires)}` : ""}`;
    return el("div", { class: `invite${i.active ? "" : " off"}` },
      i.active ? el("img", { src: `/api/invites/${i.token}/qr.svg`, alt: "QR code for this link" }) : "",
      el("div", {},
        el("strong", {}, i.label || "(no label)"), el("br"),
        el("code", {}, i.url), el("br"),
        el("span", { class: "muted" }, `${status} · used ${i.used}${i.max_captures ? ` of ${i.max_captures}` : ""}`), el("br"),
        i.active ? el("button", { class: "plain", onclick: () => navigator.clipboard.writeText(i.url) }, "Copy link") : "",
        " ",
        i.active ? el("button", { class: "plain", onclick: async () => {
          if (confirm("Revoke this link? Phones using it will no longer be able to send.")) {
            await api("POST", `/api/invites/${i.token}/revoke`); showProject(id);
          }
        } }, "Revoke") : ""));
  }));
  $("#captures tbody").replaceChildren(...d.captures.map((c) => {
    const v = c.meta.video;
    const notes = [];
    if (v) notes.push(`${v.width}×${v.height} ${v.codec}${v.bitrate ? ` ${Math.round(v.bitrate / 1e6)} Mbps` : ""}, ${v.duration}s`);
    if (c.meta.phone && c.meta.phone.markers) notes.push(c.meta.phone.markers.map((m) => `${m.event}@${m.t}s`).join(" "));
    if (c.meta.phone && c.meta.phone.sensor === false) notes.push("no motion sensor");
    const r = c.meta.report;
    if (r && r.poses) notes.push(`placed ${r.poses.placed}/${r.poses.frames} frames`);
    if (r && r.poses && "joined" in r.poses) notes.push(r.poses.joined ? "sides joined" : "sides NOT joined");
    if (r && r.seconds) notes.push(`${Math.round(Object.values(r.seconds).reduce((a, b) => a + b, 0) / 60)} min`);
    if (c.error) notes.push(c.error);
    const hasVideo = !["uploading", "cancelled", "abandoned"].includes(c.status);
    const bad = ["rejected", "failed"].includes(c.status);
    return el("tr", {},
      el("td", {}, when(c.created)), el("td", {}, c.mode),
      el("td", { class: bad ? "bad" : c.status === "done" ? "ok" : "" }, ...statusCell(c)),
      el("td", {}, hasVideo ? el("a", { href: `/api/captures/${c.id}/video`, target: "_blank" }, "play") : ""),
      el("td", {}, ...modelCell(c)),
      el("td", { class: "muted" }, notes.join(" · ")));
  }));
}

function minutes(s) { return s < 90 ? "under 2 min" : `about ${Math.round(s / 60)} min`; }

function statusCell(c) {
  if (c.status === "checked") return ["waiting for GPU"];
  if (c.status !== "processing") return [c.status];
  const p = c.progress || {};
  const pct = Math.round((p.fraction || 0) * 100);
  return [
    `${p.stage || "starting"} · ${pct}%`,
    el("div", { class: "pbar" }, el("div", { style: `width:${pct}%` })),
    el("span", { class: "muted" }, p.eta_seconds != null ? `${minutes(p.eta_seconds)} left` : ""),
  ];
}

function modelCell(c) {
  const retry = (label) => el("button", { class: "plain", onclick: async () => {
    await api("POST", `/api/captures/${c.id}/retry`); showProject(current);
  } }, label);
  const log = el("a", { href: `/view-log?c=${c.id}`, target: "_blank" }, "log");
  if (c.status === "done") return [el("a", { href: `/view?c=${c.id}`, target: "_blank" }, "view 3D"), " ", log, " ", retry("redo")];
  if (c.status === "failed") return [log, " ", retry("retry")];
  if (c.status === "processing") return [log];
  return [];
}

$("#new-project").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  const { id } = await api("POST", "/api/projects", { name: f.get("name"), mode: f.get("mode") });
  e.target.reset();
  location.hash = `p=${id}`;
});

$("#new-invite").addEventListener("submit", async (e) => {
  e.preventDefault();
  const f = new FormData(e.target);
  await api("POST", `/api/projects/${current}/invites`, {
    label: f.get("label"), days: Number(f.get("days")) || null, max_captures: Number(f.get("max_captures")) || null,
  });
  e.target.reset();
  showProject(current);
});

function route() {
  const m = location.hash.match(/p=(\d+)/);
  (m ? showProject(Number(m[1])) : showHome()).catch((e) => alert(e.message));
}
window.addEventListener("hashchange", route);
route();
health();
setInterval(health, 15000);
setInterval(() => { if (current && !$("#project").hidden) showProject(current); }, 4000);
