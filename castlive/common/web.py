"""Small pieces shared by the capture and studio apps."""

import time
from collections import defaultdict, deque

from fastapi import Request

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    # Capture links carry the invite token; never leak it to another site.
    "Referrer-Policy": "no-referrer",
    "X-Frame-Options": "DENY",
    "Content-Security-Policy": (
        "default-src 'self'; img-src 'self' data: blob:; media-src 'self' blob:; "
        "connect-src 'self' blob:; "
        "style-src 'self' 'unsafe-inline'; frame-ancestors 'none'"
    ),
}


async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for k, v in SECURITY_HEADERS.items():
        response.headers.setdefault(k, v)
    return response


def client_ip(request: Request) -> str:
    """The phone's address. The capture app is reachable only through tailscaled (its own docker
    network), which appends the real peer to X-Forwarded-For. Earlier entries come from the
    client and can be anything, so only the last one is trusted."""
    fwd = request.headers.get("x-forwarded-for", "")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else ""


class RateLimit:
    """At most `count` events per `seconds` for each key. In memory: one process each."""

    def __init__(self, count: int, seconds: float):
        self.count, self.seconds = count, seconds
        self.events: dict[str, deque] = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.monotonic()
        if len(self.events) > 10_000:  # forget keys with nothing recent, so memory stays bounded
            for k in [k for k, q in self.events.items() if not q or q[-1] < now - self.seconds]:
                del self.events[k]
        q = self.events[key]
        while q and q[0] < now - self.seconds:
            q.popleft()
        if len(q) >= self.count:
            return False
        q.append(now)
        return True
