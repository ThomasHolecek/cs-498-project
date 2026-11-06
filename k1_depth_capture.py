#!/usr/bin/env python3
# Runs ON the Booster K1 (needs ROS 2, mediapipe, opencv, numpy). Like k1_camera_stream.py, it
# serves the colour camera as an MJPEG stream, and in addition it:
#   - runs MediaPipe pose + hands on each colour frame,
#   - looks up the depth camera's distance (metres) at every landmark,
#   - writes one CSV row per frame: x, y, z, visibility AND depth for each landmark.
#
# Copy BOTH k1_camera_stream.py and this file to the robot (it imports the first one), plus
# pose_landmarker_full.task and gesture_recognizer.task (see record_gestures.py for the links).
#
#   robot:   source /opt/ros/humble/setup.bash
#            python3 k1_depth_capture.py --list
#            python3 k1_depth_capture.py --color-topic <colour> --depth-topic <depth> --label stop --distance 2
#   laptop:  open http://<robot-ip>:8080 to watch (landmarks + depth in metres are drawn on it)
#
# Depth image must be sensor_msgs/Image, 16UC1 (millimetres) or 32FC1 (metres). Landmarks are
# normalised 0-1, so they map onto the depth image even at a different resolution, but this
# ASSUMES THE DEPTH IMAGE IS ALIGNED TO THE COLOUR IMAGE. If the K1 publishes unaligned depth, the
# numbers will be off by a few pixels near edges (check with the overlay).        VERIFY on the robot

import argparse
import collections
import csv
import math
import os
import time
from pathlib import Path

import cv2
import numpy as np

from k1_camera_stream import image_to_bgr, image_topics, publish_jpeg, serve

HERE = Path(__file__).resolve().parent
POSE_MODEL = HERE / "pose_landmarker_full.task"
HAND_MODEL = HERE / "gesture_recognizer.task"

MIN_M, MAX_M = 0.15, 10.0     # depth readings outside this are treated as "no reading"
LEFT_WRIST, RIGHT_WRIST = 15, 16
TORSO = (11, 12, 23, 24)      # shoulders + hips, for the single "how far is the person" number

META_COLS = ["frame", "t_ms", "img_w", "img_h", "depth_w", "depth_h", "label", "person", "distance_m",
             "depth_age_ms", "torso_depth_m"]
POSE_COLS = [f"pose{i}_{c}" for i in range(33) for c in "xyzvd"]                  # d = depth in metres
HAND_COLS = [f"{s}hand{i}_{c}" for s in "lr" for i in range(21) for c in "xyzd"]
GESTURE_COLS = ["lhand_gesture", "lhand_score", "rhand_gesture", "rhand_score"]
COLUMNS = META_COLS + POSE_COLS + HAND_COLS + GESTURE_COLS


# ---------------------------------------------------------------- depth helpers
def depth_to_meters(msg):
    """sensor_msgs/Image (16UC1 mm or 32FC1 m) -> float32 array in metres."""
    enc = msg.encoding.lower()
    if enc in ("16uc1", "mono16"):
        arr = np.frombuffer(msg.data, np.uint16).reshape(msg.height, msg.step // 2)[:, :msg.width]
        return arr.astype(np.float32) * 0.001
    if enc == "32fc1":
        arr = np.frombuffer(msg.data, np.float32).reshape(msg.height, msg.step // 4)[:, :msg.width]
        return arr.copy()
    raise ValueError(f"unsupported depth encoding {msg.encoding!r} (expected 16UC1 or 32FC1)")


def sample_depth(depth, xn, yn, radius=2):
    """Median valid depth (metres) in a small patch around normalised point (xn, yn), or None."""
    h, w = depth.shape
    x, y = int(round(xn * (w - 1))), int(round(yn * (h - 1)))
    if not (0 <= x < w and 0 <= y < h):
        return None
    patch = depth[max(0, y - radius):y + radius + 1, max(0, x - radius):x + radius + 1]
    vals = patch[np.isfinite(patch) & (patch > MIN_M) & (patch < MAX_M)]
    return float(np.median(vals)) if vals.size else None


def stamp_s(msg):
    return msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9


# ---------------------------------------------------------------- landmarks -> row
def assign_hands(pose, hands, w, h):
    """{"l": index, "r": index} - each hand goes to the nearest pose wrist (same rule as record_gestures.py)."""
    n = len(hands.hand_landmarks)
    if n == 0:
        return {}
    if not pose.pose_landmarks:
        sides = {}
        for k in range(n):
            sides.setdefault("r" if hands.handedness[k][0].category_name == "Left" else "l", k)
        return sides
    body = pose.pose_landmarks[0]

    def gap(k, wrist):
        a, b = hands.hand_landmarks[k][0], body[wrist]
        return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

    if n == 1:
        return {"l" if gap(0, LEFT_WRIST) < gap(0, RIGHT_WRIST) else "r": 0}
    straight = gap(0, LEFT_WRIST) + gap(1, RIGHT_WRIST)
    swapped = gap(1, LEFT_WRIST) + gap(0, RIGHT_WRIST)
    return {"l": 0, "r": 1} if straight <= swapped else {"l": 1, "r": 0}


def make_row(meta, pose, hands, depth, w, h, radius):
    """meta: dict of the META_COLS values. depth: float32 metres array or None. Returns (row, drawn)
    where drawn lists (x_norm, y_norm, depth_or_None, id) for the preview overlay."""
    row = dict(meta)
    drawn = []

    def d_at(p):
        return sample_depth(depth, p.x, p.y, radius) if depth is not None else None

    def put(prefix, i, p, d, with_vis=False):
        row[f"{prefix}{i}_x"] = round(p.x, 5)
        row[f"{prefix}{i}_y"] = round(p.y, 5)
        row[f"{prefix}{i}_z"] = round(p.z, 5)
        if with_vis:
            row[f"{prefix}{i}_v"] = round(p.visibility or 0.0, 3)
        if d is not None:
            row[f"{prefix}{i}_d"] = round(d, 3)

    if pose.pose_landmarks:
        torso = []
        for i, p in enumerate(pose.pose_landmarks[0]):
            d = d_at(p)
            put("pose", i, p, d, with_vis=True)
            drawn.append((p.x, p.y, d, ("pose", i)))
            if i in TORSO and d is not None:
                torso.append(d)
        if torso:
            row["torso_depth_m"] = round(float(np.median(torso)), 3)
    for side, k in assign_hands(pose, hands, w, h).items():
        for i, p in enumerate(hands.hand_landmarks[k]):
            d = d_at(p)
            put(f"{side}hand", i, p, d)
            drawn.append((p.x, p.y, d, (f"{side}hand", i)))
        if hands.gestures and hands.gestures[k]:
            top = hands.gestures[k][0]
            row[f"{side}hand_gesture"] = top.category_name
            row[f"{side}hand_score"] = round(top.score, 3)
    return row, drawn


def draw_overlay(img, drawn, row):
    h, w = img.shape[:2]
    for x, y, d, (kind, i) in drawn:
        c = (int(x * w), int(y * h))
        cv2.circle(img, c, 3, (0, 255, 0) if d is not None else (0, 0, 255), -1)
        if kind == "pose" and i in (0, LEFT_WRIST, RIGHT_WRIST):   # label a few so it stays readable
            cv2.putText(img, f"{d:.2f}m" if d is not None else "no depth", (c[0] + 6, c[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    torso = row.get("torso_depth_m")
    cv2.putText(img, f"torso {torso:.2f} m" if torso else "no person / no depth", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)


# ---------------------------------------------------------------- main
def pick(topics, name, prefer, avoid=()):
    if name:
        return name
    for n, _ in topics:
        low = n.lower()
        if any(k in low for k in prefer) and not any(k in low for k in avoid):
            return n
    return None


def main():
    ap = argparse.ArgumentParser(description="Stream the K1 camera and log per-landmark depth to a CSV.")
    ap.add_argument("--color-topic")
    ap.add_argument("--depth-topic")
    ap.add_argument("--list", action="store_true", help="list image topics and exit")
    ap.add_argument("--out", help="CSV path (default depth_log_<time>.csv)")
    ap.add_argument("--seconds", type=float, default=0, help="stop after this long (0 = until Ctrl+C)")
    ap.add_argument("--label", default="", help="gesture label written to every row, e.g. stop")
    ap.add_argument("--person", default="")
    ap.add_argument("--distance", default="", help="true distance in metres, written to every row")
    ap.add_argument("--patch", type=int, default=2, help="depth patch radius in pixels (median of (2r+1)^2)")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=640, help="stream width (0 = full size)")
    ap.add_argument("--pose-model", default=str(POSE_MODEL))
    ap.add_argument("--hand-model", default=str(HAND_MODEL))
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image

    rclpy.init()
    node = Node("k1_depth_capture")
    topics = image_topics(node)
    if args.list or not topics:
        print("image topics:" if topics else "no image topics found - is the camera bridge running?")
        for n, t in topics:
            print(f"  {n}  [{t}]")
        print("\nCheck a depth topic's encoding with:  ros2 topic echo --once <topic> --field encoding")
        node.destroy_node()
        rclpy.shutdown()
        return

    color_topic = pick(topics, args.color_topic, ("color", "rgb", "left"), avoid=("depth",))
    depth_topic = pick(topics, args.depth_topic, ("depth",))
    types = dict(topics)
    if color_topic not in types:
        raise SystemExit(f"colour topic {color_topic!r} not found. Found: {[n for n, _ in topics]}")
    if depth_topic is None:
        print("WARNING: no depth topic found/given - depth columns will be empty (use --depth-topic).")
    elif depth_topic not in types:
        raise SystemExit(f"depth topic {depth_topic!r} not found. Found: {[n for n, _ in topics]}")
    elif types[depth_topic].endswith("CompressedImage"):
        raise SystemExit("compressed depth topics aren't supported; use the raw sensor_msgs/Image depth topic")

    for path in (args.pose_model, args.hand_model):
        if not os.path.exists(path):
            raise SystemExit(f"Missing model file {path}. Copy it next to this script (links in record_gestures.py).")
    import mediapipe as mp
    vision = mp.tasks.vision
    pose_model = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.pose_model),
        running_mode=vision.RunningMode.VIDEO, num_poses=1))
    hand_model = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=args.hand_model),
        running_mode=vision.RunningMode.VIDEO, num_hands=2))

    latest_color = [None]
    depth_buf = collections.deque(maxlen=8)   # recent depth messages, so we can pick the closest in time

    color_compressed = types[color_topic].endswith("CompressedImage")
    node.create_subscription(CompressedImage if color_compressed else Image, color_topic,
                             lambda m: latest_color.__setitem__(0, m), qos_profile_sensor_data)
    if depth_topic:
        node.create_subscription(Image, depth_topic, depth_buf.append, qos_profile_sensor_data)

    out_path = Path(args.out or time.strftime("depth_log_%Y%m%d-%H%M%S.csv"))
    f = open(out_path, "w", newline="")
    writer = csv.DictWriter(f, COLUMNS, restval="")
    writer.writeheader()
    serve(args.port)
    print(f"colour {color_topic} | depth {depth_topic} | stream http://0.0.0.0:{args.port}/stream.mjpg")
    print(f"writing {out_path}  (Ctrl+C to stop)")

    t0 = time.monotonic()
    frame_idx, last_ts, last_msg, warned_skew, warned_enc = 0, 0, None, False, False
    encode = [cv2.IMWRITE_JPEG_QUALITY, 80]
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.05)
            msg = latest_color[0]
            if msg is None or msg is last_msg:
                if args.seconds and time.monotonic() - t0 > args.seconds:
                    break
                continue
            last_msg = msg
            try:
                if color_compressed:
                    img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
                else:
                    img = image_to_bgr(msg)
            except ValueError as e:
                print(e)
                continue
            h, w = img.shape[:2]

            depth, age_ms, dh, dw = None, "", "", ""
            if depth_buf:
                cs = stamp_s(msg)
                best = min(depth_buf, key=lambda d: abs(stamp_s(d) - cs)) if cs > 0 else depth_buf[-1]
                age = (cs - stamp_s(best)) * 1000 if cs > 0 else 0.0
                age_ms = round(age, 1)
                if abs(age) > 150 and not warned_skew:
                    print(f"WARNING: nearest depth frame is {age:.0f} ms from the colour frame - values may lag.")
                    warned_skew = True
                try:
                    depth = depth_to_meters(best)
                    dh, dw = depth.shape
                except ValueError as e:
                    if not warned_enc:
                        print(e)
                        warned_enc = True

            ts = max(last_ts + 1, int((time.monotonic() - t0) * 1000))
            last_ts = ts
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            pose = pose_model.detect_for_video(mp_img, ts)
            hands = hand_model.recognize_for_video(mp_img, ts)

            meta = dict(frame=frame_idx, t_ms=ts, img_w=w, img_h=h, depth_w=dw, depth_h=dh,
                        label=args.label, person=args.person, distance_m=args.distance, depth_age_ms=age_ms)
            row, drawn = make_row(meta, pose, hands, depth, w, h, args.patch)
            writer.writerow(row)
            if frame_idx % 30 == 0:
                f.flush()
            frame_idx += 1

            view = img
            if args.width and w > args.width:
                view = cv2.resize(img, (args.width, int(h * args.width / w)))
            draw_overlay(view, drawn, row)
            ok, jpg = cv2.imencode(".jpg", view, encode)
            if ok:
                publish_jpeg(jpg.tobytes())

            if args.seconds and time.monotonic() - t0 > args.seconds:
                break
    except KeyboardInterrupt:
        pass
    finally:
        f.close()
        print(f"saved {frame_idx} frames to {out_path}")
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
