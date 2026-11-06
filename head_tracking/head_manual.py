"""
head_manual.py - drive the K1 head by hand from the laptop while watching its camera.

Reuses the camera sources, head sinks and safety clamp from head_track.py, so what you
learn here (signs, units, limits) applies directly to the tracker.

Examples:
    # no robot: laptop webcam, head commands just printed
    python head_manual.py --source webcam --head print

    # robot camera stream + real head
    python head_manual.py --source url --url http://<robot-ip>:8080/stream.mjpg --head booster --iface <name>

Keys (click the video window first):
    arrow keys or W/A/S/D   nudge the head   (W/Up = pitch -, S/Down = pitch +, A/Left = yaw -, D/Right = yaw +)
    [  /  ]                 smaller / bigger nudge step
    c                       center the head (0, 0)
    q                       quit (centers the head first)

"+" and "-" here are just the sign of the number sent. The on-screen readout shows exactly what
was commanded, so use it with the real head to work out which physical direction is which and
then set YAW_SIGN / PITCH_SIGN / HEAD_UNITS in head_track.py.
"""

import argparse
import math
import os
import sys
import time

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_track as ht  # noqa: E402

# Windows arrow-key codes from cv2.waitKeyEx
KEY_LEFT, KEY_UP, KEY_RIGHT, KEY_DOWN = 2424832, 2490368, 2555904, 2621440
# Linux/macOS (GTK/Qt) codes
ALT_LEFT, ALT_UP, ALT_RIGHT, ALT_DOWN = 65361, 65362, 65363, 65364


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "url", "ros2"], default="webcam")
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--url", default="http://192.168.10.102:8080/stream.mjpg")
    ap.add_argument("--topic", default="/camera/color/image_raw")
    ap.add_argument("--head", choices=["print", "booster"], default="print")
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--step", type=float, default=2.0, help="starting nudge size in degrees")
    args = ap.parse_args()

    if args.source == "webcam":
        source = ht.WebcamSource(args.cam)
    elif args.source == "url":
        source = ht.UrlSource(args.url)
    else:
        source = ht.Ros2ImageSource(args.topic)
    head = ht.PrintHead() if args.head == "print" else ht.BoosterHead(args.iface)

    pitch = yaw = 0.0           # radians, what we last commanded
    step = args.step            # degrees
    t_last, fps = time.monotonic(), 0.0

    def command(p, y):
        nonlocal pitch, yaw
        pitch, yaw = ht.clamp_head(p, y)
        try:
            head.send(pitch, yaw)
        except Exception as e:      # noqa: BLE001  a flaky SDK call should not kill the viewer
            print(f"[head] send failed: {type(e).__name__}: {e}")

    command(0.0, 0.0)
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                time.sleep(0.01)
                continue

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - t_last))
            t_last = now

            h, w = frame.shape[:2]
            cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
            cv2.putText(frame, f"yaw {math.degrees(yaw):+6.1f}  pitch {math.degrees(pitch):+6.1f}  step {step:g} deg  {fps:4.1f} fps",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(frame, "arrows/WASD move   [ ] step   c center   q quit",
                        (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.imshow("K1 manual head control", frame)

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            d = math.radians(step)
            low = key & 0xFF
            if key in (KEY_LEFT, ALT_LEFT) or low == ord("a"):
                command(pitch, yaw - d)
            elif key in (KEY_RIGHT, ALT_RIGHT) or low == ord("d"):
                command(pitch, yaw + d)
            elif key in (KEY_UP, ALT_UP) or low == ord("w"):
                command(pitch - d, yaw)
            elif key in (KEY_DOWN, ALT_DOWN) or low == ord("s"):
                command(pitch + d, yaw)
            elif low == ord("["):
                step = max(0.5, step / 2)
            elif low == ord("]"):
                step = min(20.0, step * 2)
            elif low == ord("c"):
                command(0.0, 0.0)
            elif low == ord("q"):
                break
    finally:
        try:
            head.send(0.0, 0.0)
        except Exception:           # noqa: BLE001
            pass
        source.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
