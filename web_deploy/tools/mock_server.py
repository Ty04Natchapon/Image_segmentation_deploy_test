"""
mock_server.py — stands in for the analysis server while it does not exist yet.

Accepts exactly the multipart POST that src/sync.js sends, writes each capture
to disk, and prints what it got. Zero dependencies: Python 3.9+ standard
library only, so anyone on the team can run it without a virtualenv.

This is also the reference for whoever builds the real endpoint. If this
accepts your request, so will they -- and if you change the contract in
sync.js, change it here too, or the two drift apart silently.

    python tools/mock_server.py                 # listens on :8001
    python tools/mock_server.py --port 9000 --out ./captures

Captures land in a tree mirroring the original Python prototype:

    received/<session_id>/Group_3_Front/<basename>.jpg       clean frame
    received/<session_id>/Group_3_Front/<basename>_mask.png  region mask
    received/<session_id>/Group_3_Front/<basename>.json      metadata

To test from a phone, this needs its own HTTPS tunnel -- a page served over
HTTPS cannot POST to a plain-HTTP address:

    python tools/mock_server.py                                  # terminal 1
    npx --yes cloudflared tunnel --url http://localhost:8001      # terminal 2

then open the app with the tunnel URL appended:

    https://<your-site>/?api=https://<tunnel>.trycloudflare.com
"""

import argparse
import functools
import json
import mimetypes
import os
import re
from email.parser import BytesParser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer

# Windows registry entries routinely get these wrong, and a .js served as
# text/plain makes the browser refuse the ES modules with no useful error.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("application/manifest+json", ".webmanifest")

GROUP_DIRS = {
    "GROUP_1": "Group_1_Right_Cheek",
    "GROUP_2": "Group_2_Left_Cheek",
    "GROUP_3": "Group_3_Front",
}

# capture_id is generated client-side and stays stable across retries, so the
# real server should use it the same way: to make ingestion idempotent rather
# than collecting duplicates every time a phone loses signal mid-upload.
seen_capture_ids = set()

SAFE = re.compile(r"[^A-Za-z0-9._-]")


def safe(name, fallback="unknown"):
    """Never let a client-supplied string escape the output directory."""
    cleaned = SAFE.sub("_", (name or "").strip())
    return cleaned[:120] or fallback


def parse_multipart(headers, body):
    """Split a multipart/form-data body into {name: (filename, bytes)}."""
    raw = b"Content-Type: " + headers["Content-Type"].encode() + b"\r\n\r\n" + body
    msg = BytesParser().parsebytes(raw)
    if not msg.is_multipart():
        return {}

    fields = {}
    for part in msg.get_payload():
        disposition = part.get("Content-Disposition", "")
        name = re.search(r'name="([^"]*)"', disposition)
        if not name:
            continue
        filename = re.search(r'filename="([^"]*)"', disposition)
        fields[name.group(1)] = (
            filename.group(1) if filename else None,
            part.get_payload(decode=True) or b"",
        )
    return fields


class Handler(SimpleHTTPRequestHandler):
    out_dir = "received"
    serving = False        # also handing out the app itself, not just the API

    def _cors(self):
        # The app is served from a different origin (GitHub Pages, or a tunnel),
        # so without these the browser discards the response and every upload
        # looks like a network failure.
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def _json(self, status, payload):
        body = json.dumps(payload).encode()
        self.send_response(status)
        self._cors()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self):
        # With --serve, this one process is both the app and the API. That
        # means one tunnel instead of two, and no CORS at all, because the page
        # and the endpoint are then the same origin.
        if self.serving and self.path != "/__health":
            return super().do_GET()
        self._json(200, {"status": "ok", "received": len(seen_capture_ids)})

    def end_headers(self):
        # Never cache during testing; a stale main.js after an edit wastes
        # more time than the requests it saves.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        if not length:
            return self._json(400, {"error": "empty body"})

        try:
            fields = parse_multipart(self.headers, self.rfile.read(length))
        except Exception as exc:                      # noqa: BLE001
            return self._json(400, {"error": f"could not parse multipart: {exc}"})

        text = {k: v[1].decode("utf-8", "replace") for k, v in fields.items() if v[0] is None}
        image = fields.get("image")
        mask = fields.get("mask")

        missing = [f for f in ("capture_id", "group") if not text.get(f)]
        if missing or not image or not mask:
            if not image:
                missing.append("image")
            if not mask:
                missing.append("mask")
            return self._json(400, {"error": "missing fields", "missing": missing})

        capture_id = text["capture_id"]
        if capture_id in seen_capture_ids:
            # A retry of something already stored. Answer 200 so the client
            # stops retrying, but do not write it twice.
            print(f"  duplicate {capture_id[:8]} - already have it, acking", flush=True)
            return self._json(200, {"status": "duplicate", "capture_id": capture_id})

        group = text.get("group", "")
        folder = os.path.join(
            self.out_dir,
            safe(text.get("session_id"), "no_session"),
            GROUP_DIRS.get(group, safe(group, "unknown_group")),
        )
        os.makedirs(folder, exist_ok=True)

        base = safe(os.path.splitext(image[0] or "capture")[0], "capture")
        with open(os.path.join(folder, base + ".jpg"), "wb") as fh:
            fh.write(image[1])
        with open(os.path.join(folder, base + "_mask.png"), "wb") as fh:
            fh.write(mask[1])
        with open(os.path.join(folder, base + ".json"), "w", encoding="utf-8") as fh:
            json.dump(text, fh, indent=2)

        seen_capture_ids.add(capture_id)
        print(
            f"  {text.get('region', group):<12} "
            f"{text.get('width')}x{text.get('height')}  "
            f"skin_px={text.get('skin_px')}  "
            f"image={len(image[1]) / 1024:.0f}KB mask={len(mask[1]) / 1024:.0f}KB  "
            f"-> {folder}",
            flush=True,
        )
        return self._json(201, {"status": "stored", "capture_id": capture_id})

    def log_message(self, fmt, *args):
        pass          # the per-capture line above is the useful log


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--out", default="received", help="where to write captures")
    ap.add_argument("--serve", metavar="DIR", default=None,
                    help="also serve the app from DIR, so one tunnel covers both")
    args = ap.parse_args()

    Handler.out_dir = args.out
    Handler.serving = bool(args.serve)
    os.makedirs(args.out, exist_ok=True)

    handler = Handler
    if args.serve:
        handler = functools.partial(Handler, directory=args.serve)

    print(f"Listening on http://localhost:{args.port}")
    print(f"Writing captures to {os.path.abspath(args.out)}")
    if args.serve:
        print(f"Serving the app from {os.path.abspath(args.serve)}")
        print(f"Open  http://localhost:{args.port}/?api=/   (same origin, no CORS)")
    else:
        print(f"Open the app with  ?api=http://localhost:{args.port}")
    print()
    try:
        ThreadingHTTPServer(("0.0.0.0", args.port), handler).serve_forever()
    except KeyboardInterrupt:
        print(f"\nStopped. {len(seen_capture_ids)} capture(s) received.")


if __name__ == "__main__":
    main()
