#!/usr/bin/env python3
"""
k1_head_server.py  -  runs ON the Booster K1 (or any Linux box with the Booster SDK).

Accepts head angles over HTTP and forwards them to B1LocoClient.RotateHead, so a
laptop that cannot install the Booster SDK (e.g. a Mac) can still drive the head.
Mirror image of k1_camera_stream.py: that one sends frames out, this one takes
commands in.

    robot:   python3 k1_head_server.py --iface eth0          # see `ip a` for the name
    laptop:  python head_track.py --source url --head http --head-url http://<robot-ip>:8081

Endpoints:
    GET /head?pitch=<rad>&yaw=<rad>     set head pose (radians by default)
    GET /center                         pitch=0, yaw=0
    GET /                               status

Safety: every command is clamped to the limits below before it reaches the SDK,
regardless of what the laptop sends. Adjust the limits once you know the real ones.
"""

import argparse
import math
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

# Limits in degrees. Keep these conservative; the laptop has its own clamp too.   VERIFY
YAW_LIMIT_DEG = 45.0
PITCH_MIN_DEG = -20.0
PITCH_MAX_DEG = 20.0

_client = None
_lock = threading.Lock()
_state = {"pitch": 0.0, "yaw": 0.0, "n": 0, "last_rc": None, "units": "radians"}


def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def rotate_head(pitch_rad, yaw_rad):
    yaw_rad = _clamp(yaw_rad, -math.radians(YAW_LIMIT_DEG), math.radians(YAW_LIMIT_DEG))
    pitch_rad = _clamp(pitch_rad, math.radians(PITCH_MIN_DEG), math.radians(PITCH_MAX_DEG))
    p, y = pitch_rad, yaw_rad
    if _state["units"] == "degrees":
        p, y = math.degrees(p), math.degrees(y)
    with _lock:
        rc = _client.RotateHead(p, y) if _client is not None else None
        _state.update(pitch=pitch_rad, yaw=yaw_rad, last_rc=rc)
        _state["n"] += 1
    return rc


class Handler(BaseHTTPRequestHandler):
    def _reply(self, code, body):
        data = body.encode()
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/":
            s = _state
            self._reply(200, f"ok  pitch={math.degrees(s['pitch']):+.1f}deg  yaw={math.degrees(s['yaw']):+.1f}deg  "
                             f"commands={s['n']}  last_rc={s['last_rc']}  sdk={'live' if _client else 'DRY RUN'}\n")
            return
        if url.path == "/center":
            rc = rotate_head(0.0, 0.0)
            self._reply(200, f"centered rc={rc}\n")
            return
        if url.path == "/head":
            q = parse_qs(url.query)
            try:
                pitch = float(q["pitch"][0])
                yaw = float(q["yaw"][0])
            except (KeyError, ValueError, IndexError):
                self._reply(400, "need /head?pitch=<rad>&yaw=<rad>\n")
                return
            rc = rotate_head(pitch, yaw)
            self._reply(200, f"rc={rc}\n")
            return
        self._reply(404, "not found\n")

    def log_message(self, *args):
        pass


def main():
    global _client
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0", help="network interface the SDK should bind to")
    ap.add_argument("--port", type=int, default=8081)
    ap.add_argument("--degrees", action="store_true", help="pass degrees to RotateHead instead of radians")
    ap.add_argument("--dry-run", action="store_true", help="no SDK, just log commands (test the plumbing)")
    args = ap.parse_args()
    _state["units"] = "degrees" if args.degrees else "radians"

    if not args.dry_run:
        from booster_robotics_sdk_python import ChannelFactory, B1LocoClient   # VERIFY names against SDK example
        ChannelFactory.Instance().Init(0, args.iface)
        _client = B1LocoClient()
        _client.Init()
        print(f"[sdk] connected on {args.iface}")
    else:
        print("[sdk] DRY RUN, commands will be logged only")

    server = ThreadingHTTPServer(("0.0.0.0", args.port), Handler)
    server.daemon_threads = True
    print(f"head server on http://0.0.0.0:{args.port}   (Ctrl+C to stop)")
    last = -1
    try:
        threading.Thread(target=server.serve_forever, daemon=True).start()
        while True:
            time.sleep(1.0)
            if _state["n"] != last:
                last = _state["n"]
                print(f"  pitch={math.degrees(_state['pitch']):+6.1f}  yaw={math.degrees(_state['yaw']):+6.1f}  "
                      f"n={last}  rc={_state['last_rc']}")
    except KeyboardInterrupt:
        pass
    finally:
        if _client is not None:
            rotate_head(0.0, 0.0)
        server.shutdown()


if __name__ == "__main__":
    main()
