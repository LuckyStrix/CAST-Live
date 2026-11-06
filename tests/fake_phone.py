"""Send a video to a capture site the way the phone page does: chunked, out of order, with
retries, so the server path can be tried without a phone.

    python tests/fake_phone.py http://127.0.0.1:8001/c/<token> video.mp4 [--drop 0.2]
"""

import argparse
import json
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlparse

CHUNK = 4 * 1024 * 1024


def call(method, url, body=None, content_type="application/json"):
    data = json.dumps(body).encode() if isinstance(body, dict) else body
    req = urllib.request.Request(url, data=data, method=method)
    if data is not None:
        req.add_header("Content-Type", content_type)
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.status, json.loads(r.read() or b"null")
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"null")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("link", help="the invite link, e.g. https://castlive.x.ts.net/c/TOKEN")
    ap.add_argument("video")
    ap.add_argument("--mode", default="object")
    ap.add_argument("--drop", type=float, default=0.0, help="fraction of chunk sends to skip at first")
    a = ap.parse_args(argv)

    u = urlparse(a.link)
    token = u.path.rstrip("/").split("/")[-1]
    base = f"{u.scheme}://{u.netloc}/api/{token}"
    data = Path(a.video).read_bytes()
    mime = "video/webm" if a.video.endswith(".webm") else "video/mp4"
    chunks = [data[i:i + CHUNK] for i in range(0, len(data), CHUNK)]
    cid = str(uuid.uuid4())

    st, r = call("POST", f"{base}/captures",
                 {"id": cid, "mode": a.mode, "mime": mime, "consent": True, "device": "fake_phone"})
    if st != 200:
        sys.exit(f"create failed: {st} {r}")
    order = list(range(len(chunks)))
    random.shuffle(order)
    for n in order:
        if random.random() < a.drop:
            continue  # "lost": finish will report it missing
        st, r = call("PUT", f"{base}/captures/{cid}/chunks/{n}", chunks[n], "application/octet-stream")
        if st != 200:
            sys.exit(f"chunk {n} failed: {st} {r}")
    meta = {"markers": [{"t": 0, "event": "start"}], "sensor": False}
    for _ in range(5):
        st, r = call("POST", f"{base}/captures/{cid}/finish", {"chunks": len(chunks), "meta": meta})
        if st == 200:
            break
        if st != 409:
            sys.exit(f"finish failed: {st} {r}")
        print(f"resending {len(r['missing'])} missing chunks")
        for n in r["missing"]:
            call("PUT", f"{base}/captures/{cid}/chunks/{n}", chunks[n], "application/octet-stream")
    print(f"sent {cid} ({len(chunks)} chunks)")
    for _ in range(30):
        st, r = call("GET", f"{base}/captures/{cid}")
        print("status:", r["status"], r.get("error", ""))
        if r["status"] in ("checked", "rejected"):
            return
        time.sleep(2)


if __name__ == "__main__":
    main()
