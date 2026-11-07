// Recordings live in IndexedDB until the server has every piece, so a reload, a dropped
// connection or a capture made offline loses nothing.
//
// captures: {id, token, mode, mime, device, total (null while recording), meta, created, created_on_server}
// chunks:   {id, n, data (ArrayBuffer)}, key [id, n]
// A capture the server refused as too large keeps its data and gets `failed` (not retried).

const DB_NAME = "castlive";
let dbp = null;

function open() {
  if (!dbp) {
    dbp = new Promise((resolve, reject) => {
      const req = indexedDB.open(DB_NAME, 1);
      req.onupgradeneeded = () => {
        const db = req.result;
        db.createObjectStore("captures", { keyPath: "id" });
        db.createObjectStore("chunks", { keyPath: ["id", "n"] }).createIndex("by_id", "id");
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => reject(req.error);
    });
  }
  return dbp;
}

async function tx(store, mode, fn) {
  const db = await open();
  return new Promise((resolve, reject) => {
    const t = db.transaction(store, mode);
    const result = fn(t.objectStore(store));
    t.oncomplete = () => resolve(result && "result" in result ? result.result : result);
    t.onerror = () => reject(t.error);
    t.onabort = () => reject(t.error || new Error("storage full"));
  });
}

export const store = {
  putCapture: (c) => tx("captures", "readwrite", (s) => s.put(c)),
  getCapture: (id) => tx("captures", "readonly", (s) => s.get(id)),
  // Change some fields in one transaction, so two writers never undo each other.
  patchCapture: (id, fields) => tx("captures", "readwrite", (s) => {
    const r = s.get(id);
    r.onsuccess = () => { if (r.result) s.put({ ...r.result, ...fields }); };
  }),
  allCaptures: () => tx("captures", "readonly", (s) => s.getAll()),
  putChunk: (id, n, data) => tx("chunks", "readwrite", (s) => s.put({ id, n, data })),
  chunkKeys: (id) => tx("chunks", "readonly", (s) => s.index("by_id").getAllKeys(IDBKeyRange.only(id))),
  getChunk: (id, n) => tx("chunks", "readonly", (s) => s.get([id, n])),
  delChunk: (id, n) => tx("chunks", "readwrite", (s) => s.delete([id, n])),
  async delCapture(id) {
    for (const [, n] of await this.chunkKeys(id)) await this.delChunk(id, n);
    await tx("captures", "readwrite", (s) => s.delete(id));
  },
};

// Ask the browser not to clear our data under storage pressure (best effort).
if (navigator.storage && navigator.storage.persist) navigator.storage.persist().catch(() => {});

class HttpError extends Error {
  constructor(status, body) {
    super((body && body.detail) || `HTTP ${status}`);
    this.status = status;
  }
}

async function api(method, url, body, isJson = true) {
  const opts = { method, headers: {} };
  if (body !== undefined) {
    opts.body = isJson ? JSON.stringify(body) : body;
    if (isJson) opts.headers["Content-Type"] = "application/json";
  }
  const r = await fetch(url, opts);
  const data = await r.json().catch(() => null);
  if (!r.ok) throw new HttpError(r.status, data);
  return data;
}

// Sends every stored capture, oldest first. One loop at a time; kick() wakes it.
export class Uploader {
  constructor(onProgress) {
    this.onProgress = onProgress; // ({id, sent, total, state, error})
    this.running = false;
    this.again = false;
    window.addEventListener("online", () => this.kick());
  }

  kick() {
    if (this.running) { this.again = true; return; }
    this.running = true;
    this.run().finally(() => {
      this.running = false;
      if (this.again) { this.again = false; this.kick(); }
    });
  }

  async run() {
    let delay = 2000;
    for (;;) {
      const caps = (await store.allCaptures()).filter((c) => !c.failed).sort((a, b) => a.created - b.created);
      if (!caps.length) return;
      let progressed = false;
      try {
        for (const c of caps) progressed = (await this.send(c)) || progressed;
        if (!progressed) return; // only recordings still in progress: wait for kick()
        delay = 2000;
      } catch (e) {
        if (e instanceof HttpError && e.status >= 400 && e.status < 500 && e.status !== 409 && e.status !== 429) {
          // The server will never accept this one (bad link, used up, stopped, too large).
          const c = caps.find((x) => x.id === this.current);
          if (c) {
            // Too large: keep the recording on the phone, it may be recoverable. Otherwise drop it.
            if (e.status === 413) await store.patchCapture(c.id, { failed: e.message });
            else await store.delCapture(c.id);
            this.onProgress({ id: c.id, state: "failed", error: e.message });
          }
          continue;
        }
        this.onProgress({ id: this.current, state: navigator.onLine ? "retrying" : "offline" });
        await new Promise((r) => setTimeout(r, delay));
        delay = Math.min(delay * 2, 30000);
      }
    }
  }

  /** Make the capture on the server now, even while another one is uploading (the live view
   * needs it as filming starts). Failures are left to the upload loop, which tries again. */
  async create(c) {
    const r = await api("POST", `/api/${c.token}/captures`,
      { id: c.id, mode: c.mode, mime: c.mime, consent: true, device: c.device });
    await store.patchCapture(c.id, { created_on_server: true });
    return new Set(r.have);
  }

  // Send what is stored for one capture. True if it made progress.
  async send(c) {
    this.current = c.id;
    const base = `/api/${c.token}/captures`;
    const fresh = await store.getCapture(c.id); // create() may have run since the list was read
    if (!fresh) return false;
    const have = fresh && fresh.created_on_server ? new Set() : await this.create(c);
    const keys = (await store.chunkKeys(c.id)).map(([, n]) => n).sort((a, b) => a - b);
    let progressed = false;
    for (const n of keys) {
      if (!have.has(n)) {
        const piece = await store.getChunk(c.id, n);
        await api("PUT", `${base}/${c.id}/chunks/${n}`, piece.data, false);
      }
      await store.delChunk(c.id, n);
      progressed = true;
      // Read the total afresh: filming may have stopped (and the total become known) since this
      // pass started, and the bar should move straight away rather than sit at 0%.
      const now = await store.getCapture(c.id);
      const total = now ? now.total : null;
      const left = total == null ? 0 : (await store.chunkKeys(c.id)).length;
      this.onProgress({ id: c.id, sent: total == null ? null : total - left, total, state: "sending" });
    }
    if (c.total != null) {
      try {
        await api("POST", `${base}/${c.id}/finish`, { chunks: c.total, meta: c.meta });
      } catch (e) {
        if (e.status === 409) {
          // The server lost pieces we already deleted: nothing more can be done for this one.
          await store.delCapture(c.id);
          this.onProgress({ id: c.id, state: "failed", error: "pieces were lost" });
          return true;
        }
        throw e;
      }
      await store.delCapture(c.id);
      this.onProgress({ id: c.id, sent: c.total, total: c.total, state: "done", token: c.token });
      return true;
    }
    return progressed;
  }
}

export async function serverStatus(token, id) {
  return api("GET", `/api/${token}/captures/${id}`);
}

export async function projectInfo(token) {
  return api("GET", `/api/${token}/info`);
}
