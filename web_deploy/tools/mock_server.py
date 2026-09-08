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
from datetime import datetime
from email.parser import BytesParser
from html import escape
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from string import Template

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

# --- the /__received inspection page ---------------------------------------
# string.Template rather than str.format, because the CSS below is full of
# braces and escaping every one of them would make this unreadable.

PAGE_HTML = Template("""<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Received captures</title>
<style>
  :root { color-scheme: dark; }
  body { margin:0; background:#0d1014; color:#e8edf3;
         font:15px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,sans-serif; }
  header { position:sticky; top:0; background:#0d1014ee; backdrop-filter:blur(8px);
           padding:16px 18px; border-bottom:1px solid #2b333d; }
  h1 { margin:0 0 4px; font-size:17px; }
  .sub { color:#93a1b1; font-size:13px; word-break:break-all; }
  .live { display:inline-block; width:8px; height:8px; border-radius:50%;
          background:#3ddc84; margin-right:6px; vertical-align:middle; }
  main { padding:14px; display:grid; gap:12px;
         grid-template-columns:repeat(auto-fill,minmax(280px,1fr)); }
  .card { border:1px solid #2b333d; border-radius:12px; background:#161b22; overflow:hidden; }
  .pair { display:grid; grid-template-columns:1fr 1fr; gap:1px; background:#2b333d; }
  .pair figure { margin:0; background:#000; }
  .pair img { display:block; width:100%; aspect-ratio:3/4; object-fit:cover; }
  .pair figcaption { padding:4px 8px; font-size:11px; color:#93a1b1; background:#161b22; }
  .meta { padding:10px 12px; font-size:13px; }
  .meta .row { display:flex; justify-content:space-between; gap:10px; }
  .meta .row span:last-child { color:#93a1b1; font-variant-numeric:tabular-nums; }
  .region { font-weight:650; margin-bottom:6px; color:#3ddc84; }
  .empty { padding:60px 20px; text-align:center; color:#93a1b1; }
  code { background:#1e252e; padding:1px 5px; border-radius:4px; font-size:12px; }
</style>
</head><body>
<header>
  <h1><span class="live"></span>$count capture(s) received</h1>
  <div class="sub">Written to $out &middot; refreshing automatically</div>
</header>
<main>$cards</main>
<script>
  // Poll the health count and reload only when it changes, so the page can sit
  // open on a laptop while captures are taken on a phone.
  let last = $count;
  setInterval(async () => {
    try {
      const r = await fetch('/__health', { cache: 'no-store' });
      const j = await r.json();
      if (j.received !== last) location.reload();
    } catch (e) { /* server restarting; try again next tick */ }
  }, 2000);
</script>
</body></html>""")

CARD_HTML = Template("""
<div class="card">
  <div class="pair">
    <figure><img src="$image" alt="capture" loading="lazy"><figcaption>image</figcaption></figure>
    <figure><img src="$mask" alt="mask" loading="lazy"><figcaption>mask</figcaption></figure>
  </div>
  <div class="meta">
    <div class="region">$region</div>
    <div class="row"><span>skin px</span><span>$skin</span></div>
    <div class="row"><span>size</span><span>$dims</span></div>
    <div class="row"><span>ratio / brightness</span><span>$ratio / $bright</span></div>
    <div class="row"><span>fill light</span><span>$light</span></div>
    <div class="row"><span>received</span><span>$when</span></div>
    <div class="row"><span>capture / session</span><span>$cid / $session</span></div>
  </div>
</div>""")

EMPTY_HTML = ('<div class="empty">Nothing yet. Take a capture in the app '
              'with <code>?api=/</code> on the URL.</div>')


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
        path = self.path.split("?")[0]

        if path == "/__health":
            return self._json(200, {"status": "ok", "received": len(seen_capture_ids)})
        if path == "/__received":
            return self._received_page()
        if path == "/__file":
            return self._send_stored_file()

        # With --serve, this one process is both the app and the API. That
        # means one tunnel instead of two, and no CORS at all, because the page
        # and the endpoint are then the same origin.
        if self.serving:
            return super().do_GET()
        return self._json(200, {"status": "ok", "received": len(seen_capture_ids)})

    # --- inspecting what actually arrived -----------------------------------
    # The app's own gallery can only tell you what it *believes* it sent. This
    # is the other end of the wire: what is genuinely on the server's disk.
    # When the two disagree, that gap is the bug.

    def _stored(self):
        """Every capture on disk, newest first, read back from its metadata."""
        rows = []
        for folder, _dirs, files in os.walk(self.out_dir):
            for name in files:
                if not name.endswith(".json"):
                    continue
                full = os.path.join(folder, name)
                try:
                    with open(full, encoding="utf-8") as fh:
                        meta = json.load(fh)
                except (OSError, ValueError):
                    continue
                base = name[:-5]
                rel = os.path.relpath(folder, self.out_dir).replace("\\", "/")
                meta["_image"] = f"{rel}/{base}.jpg"
                meta["_mask"] = f"{rel}/{base}_mask.png"
                meta["_received_at"] = os.path.getmtime(full)
                rows.append(meta)
        rows.sort(key=lambda r: r["_received_at"], reverse=True)
        return rows

    def _send_stored_file(self):
        from urllib.parse import parse_qs, urlparse

        rel = (parse_qs(urlparse(self.path).query).get("p") or [""])[0]
        root = os.path.abspath(self.out_dir)
        full = os.path.abspath(os.path.join(root, rel))
        # Refuse anything that resolves outside the capture directory.
        if not full.startswith(root + os.sep) or not os.path.isfile(full):
            return self._json(404, {"error": "not found"})

        ctype = "image/png" if full.endswith(".png") else "image/jpeg"
        with open(full, "rb") as fh:
            body = fh.read()
        self.send_response(200)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _received_page(self):
        rows = self._stored()
        cards = []
        for r in rows:
            when = datetime.fromtimestamp(r["_received_at"]).strftime("%H:%M:%S")
            skin = r.get("skin_px", "?")
            try:
                skin = f"{int(skin):,}"
            except (TypeError, ValueError):
                pass
            cards.append(CARD_HTML.substitute(
                image=escape("/__file?p=" + r["_image"]),
                mask=escape("/__file?p=" + r["_mask"]),
                region=escape(r.get("region") or r.get("group", "?")),
                skin=escape(str(skin)),
                dims=escape(f'{r.get("width", "?")}x{r.get("height", "?")}'),
                when=escape(when),
                ratio=escape(str(r.get("ratio", "?"))),
                bright=escape(str(r.get("brightness", "?"))),
                light="on" if r.get("fill_light") == "true" else "off",
                cid=escape((r.get("capture_id") or "")[:8]),
                session=escape((r.get("session_id") or "")[:8]),
            ))

        body = PAGE_HTML.substitute(
            count=len(rows),
            out=escape(os.path.abspath(self.out_dir)),
            cards="".join(cards) or EMPTY_HTML,
        ).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def end_headers(self):
        # Never cache during testing; a stale main.js after an edit wastes
        # more time than the requests it saves.
        self.send_header("Cache-Control", "no-store")
        super().end_headers()

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)

        # A phone has no console you can read without a tethered Mac, so the
        # app forwards its errors here and they come out in this terminal.
        if self.path.split("?")[0] == "/__log":
            raw = self.rfile.read(length) if length else b"{}"
            try:
                entry = json.loads(raw.decode("utf-8", "replace"))
            except ValueError:
                entry = {"message": raw.decode("utf-8", "replace")}
            stamp = datetime.now().strftime("%H:%M:%S")
            print(f"[{stamp}] PHONE {entry.get('level', 'log')}: {entry.get('message', '')}",
                  flush=True)
            for line in str(entry.get("detail") or "").splitlines():
                print(f"           {line}", flush=True)
            return self._json(200, {"status": "logged"})

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
