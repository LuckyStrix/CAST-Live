// Run log for one capture: the pipeline's steps, then the end of COLMAP's output.
const cid = new URLSearchParams(location.search).get("c");
const pre = document.getElementById("log");

async function refresh() {
  const atBottom = innerHeight + scrollY >= document.body.scrollHeight - 40;
  const r = await fetch(`/api/captures/${cid}/log`).catch(() => null);
  if (r && r.ok) pre.textContent = await r.text();
  if (atBottom) scrollTo(0, document.body.scrollHeight);
}
refresh();
setInterval(refresh, 3000);
