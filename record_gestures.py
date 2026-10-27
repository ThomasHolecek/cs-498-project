"""
Record the K1 gesture dataset: MediaPipe body + hand landmarks, saved as short labeled clips.

One command runs a whole guided session for one person on one camera. The script walks
through every distance x facing x gesture, shows a countdown big enough to read from 4 m,
records each clip, and saves:
    data/<person>_<camera>_<time>.csv           one row per frame (landmarks only, a few MB)
    clips/<person>_<camera>_<time>/*.mp4        the raw clips, so you can re-extract later

Setup:
    pip install mediapipe opencv-python pandas
    Put both models in the folder you run from:
    https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
    https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task

Usage:
    python record_gestures.py record --person joe --camera laptop
    python record_gestures.py record --person joe --camera k1 --ros-topic <head camera topic>
    python record_gestures.py record --person joe --camera laptop --start 20   # resume at step 20
    python record_gestures.py cameras                                          # find webcam indexes
    python record_gestures.py stats                                            # what you have so far
    python record_gestures.py extract clips/joe_laptop_20261019-140501         # re-run landmarks on saved clips

Keys during a session:
    SPACE start the step   r redo last clip   n skip step   b back a step   p pause
    m mirror preview   s sound on/off   f fullscreen   q quit
After every clip the status line shows how well the body/hands were detected, in red if it's poor.
Every finished clip is saved immediately, so quitting never loses work.
"""

import argparse
import csv
import math
import re
import shutil
import sys
import threading
import time
try:
    import winsound
except ImportError:   # not Windows
    winsound = None
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

HAND_MODEL = "gesture_recognizer.task"
POSE_MODEL = "pose_landmarker_full.task"
MODEL_URLS = {
    HAND_MODEL: "https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task",
    POSE_MODEL: "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
}
DATA_DIR = Path("data")
CLIPS_DIR = Path("clips")

# ---- The recording plan: every distance x facing x label, CLIPS[label] clips each ----
DISTANCES = [1, 2, 3, 4]                             # meters from the camera (tape marks on the floor)
FACINGS = ["front", "turned_left", "turned_right"]   # which way the performer's body faces
CLIPS = {"come_here": 4, "shoo": 4, "stop": 1, "none": 2}
PROMPTS = {
    "come_here": "COME HERE: beckon toward yourself",
    "shoo": "SHOO: flick your hand away from you",
    "stop": "STOP: palm out, arm up, hold still",
    "none": "NONE: wave hi, scratch head, phone, walk...",
}
# Which hand and style for each clip of a step (cycled if CLIPS asks for more). Hand = the PERFORMER's own hand.
VARIANTS = {
    "come_here": [("right", "normal speed"), ("left", "normal speed"), ("right", "fast and small"), ("left", "slow and big")],
    "shoo": [("right", "normal speed"), ("left", "normal speed"), ("right", "fast and small"), ("left", "slow and big")],
    "stop": [("right", "hold still")],
    "none": [("either", "something natural"), ("either", "something different")],
}
FACING_TEXT = {"front": "facing the camera", "turned_left": "body turned LEFT (~45 deg)",
               "turned_right": "body turned RIGHT (~45 deg)"}
CLIP_SECONDS = 1.5      # a bit longer than the classifier's 1 s window, so training can take random 1 s crops
FIRST_COUNTDOWN = 7.0   # time to walk to your mark after pressing SPACE
COUNTDOWN = 5.0         # rest + get ready between clips

# ---- CSV layout: one row per frame. Empty cells = not detected in that frame ----
META_COLS = ["clip_id", "label", "person", "camera", "distance_m", "facing", "hand", "frame", "t_ms", "img_w", "img_h"]
POSE_COLS = [f"pose{i}_{c}" for i in range(33) for c in "xyzv"]                # 33 body points (x, y, z, visibility)
HAND_COLS = [f"{s}hand{i}_{c}" for s in "lr" for i in range(21) for c in "xyz"]  # the person's left / right hand
GESTURE_COLS = ["lhand_gesture", "lhand_score", "rhand_gesture", "rhand_score"]  # MediaPipe's built-in guess
COLUMNS = META_COLS + POSE_COLS + HAND_COLS + GESTURE_COLS

MIN_DETECTED = 0.8   # a clip whose body is found in fewer frames than this gets flagged for redo
NOSE, LEFT_WRIST, RIGHT_WRIST = 0, 15, 16   # pose landmark ids for the PERSON's left and right wrist
WIN = "gesture recorder"


class Detector:
    """Runs MediaPipe pose (33 body points) and hands (21 points each + built-in gesture) on a frame."""

    def __init__(self):
        for model in (HAND_MODEL, POSE_MODEL):
            if not Path(model).exists():
                sys.exit(f"Missing {model}. Download it into this folder:\n  {MODEL_URLS[model]}")
        vision = mp.tasks.vision
        self.pose = vision.PoseLandmarker.create_from_options(vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=POSE_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_poses=1,
        ))
        self.hands = vision.GestureRecognizer.create_from_options(vision.GestureRecognizerOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=HAND_MODEL),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=2,
        ))
        self.ts = 0

    def __call__(self, frame_bgr):
        self.ts = max(self.ts + 1, int(time.monotonic() * 1000))  # VIDEO mode needs increasing timestamps
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        return self.pose.detect_for_video(image, self.ts), self.hands.recognize_for_video(image, self.ts)


def assign_hands(pose, hands, w, h):
    """Map detected hands to the person's arms: {"l": hand_index, "r": hand_index}.

    Each hand is matched to the nearest pose wrist. That's more reliable than MediaPipe's
    own left/right guess, which assumes a mirrored selfie image (our frames aren't mirrored).
    """
    n = len(hands.hand_landmarks)
    if n == 0:
        return {}
    if not pose.pose_landmarks:
        # No body found: use MediaPipe's guess, swapped because our frames aren't mirrored
        sides = {}
        for k in range(n):
            sides.setdefault("r" if hands.handedness[k][0].category_name == "Left" else "l", k)
        return sides

    body = pose.pose_landmarks[0]

    def gap(k, wrist):
        a, b = hands.hand_landmarks[k][0], body[wrist]   # hand landmark 0 is the wrist
        return math.hypot((a.x - b.x) * w, (a.y - b.y) * h)

    if n == 1:
        return {"l" if gap(0, LEFT_WRIST) < gap(0, RIGHT_WRIST) else "r": 0}
    straight = gap(0, LEFT_WRIST) + gap(1, RIGHT_WRIST)
    swapped = gap(1, LEFT_WRIST) + gap(0, RIGHT_WRIST)
    return {"l": 0, "r": 1} if straight <= swapped else {"l": 1, "r": 0}


def make_row(meta, frame_idx, t_ms, w, h, pose, hands):
    """One CSV row: the clip's labels plus every landmark found in this frame."""
    row = dict(meta, frame=frame_idx, t_ms=t_ms, img_w=w, img_h=h)
    if pose.pose_landmarks:
        for i, p in enumerate(pose.pose_landmarks[0]):
            row[f"pose{i}_x"] = round(p.x, 5)
            row[f"pose{i}_y"] = round(p.y, 5)
            row[f"pose{i}_z"] = round(p.z, 5)
            row[f"pose{i}_v"] = round(p.visibility or 0.0, 3)
    for side, k in assign_hands(pose, hands, w, h).items():
        for i, p in enumerate(hands.hand_landmarks[k]):
            row[f"{side}hand{i}_x"] = round(p.x, 5)
            row[f"{side}hand{i}_y"] = round(p.y, 5)
            row[f"{side}hand{i}_z"] = round(p.z, 5)
        if hands.gestures and hands.gestures[k]:
            top = hands.gestures[k][0]
            row[f"{side}hand_gesture"] = top.category_name
            row[f"{side}hand_score"] = round(top.score, 3)
    return row


def beep(freq, ms):
    """Audible cue so the performer doesn't need to read the screen from 4 m."""
    if winsound:
        threading.Thread(target=winsound.Beep, args=(freq, ms), daemon=True).start()
    else:
        print("", end="", flush=True)


def clip_quality(rows):
    """(summary text, ok) for a just-recorded clip: how often the body and hands were found."""
    n = len(rows)
    body = sum("pose0_x" in r for r in rows) / n
    hand = sum("lhand0_x" in r or "rhand0_x" in r for r in rows) / n
    text = f"last clip: body {body:.0%}, hand {hand:.0%}"
    return text, body >= MIN_DETECTED


def framing_warning(pose, w, h):
    """Why the performer may be out of frame right now, or '' if they look fine."""
    if not pose.pose_landmarks:
        return "No body found: step into frame / add light"
    body = pose.pose_landmarks[0]
    for name, i in (("head", NOSE), ("left hand", LEFT_WRIST), ("right hand", RIGHT_WRIST)):
        p = body[i]
        if not (0.02 < p.x < 0.98 and 0.02 < p.y < 0.98):
            return f"{name} is cut off by the frame edge"
    return ""


def write_video(path, frames, fps):
    h, w = frames[0].shape[:2]
    # Whole-number fps: the MP4 writer can refuse fractional rates like 23.7
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(1, round(fps)), (w, h))
    if not out.isOpened():
        print(f"WARNING: couldn't write {path} (landmarks were still saved to the CSV)")
        return
    for frame in frames:
        out.write(frame)
    out.release()


class Session:
    """One CSV + one folder of clips per recording session. Can undo the last clip."""

    def __init__(self, name):
        self.name = name
        self.clip_dir = CLIPS_DIR / name
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)
        self.csv_path = DATA_DIR / f"{name}.csv"
        self.file = open(self.csv_path, "w", newline="")
        self.writer = csv.DictWriter(self.file, COLUMNS, restval="")
        self.writer.writeheader()
        self.file.flush()
        self.history = []   # (CSV position before the clip, video path, what to rewind to on undo)
        self.n = 0          # clip numbers are never reused, even after an undo

    def save_clip(self, rows, frames, seconds, label, distance, facing, rewind_to):
        self.n += 1
        clip_id = f"{self.name}-{self.n:03d}"
        video = self.clip_dir / f"{self.n:03d}-{label}-{distance}m-{facing}.mp4"
        self.history.append((self.file.tell(), video, rewind_to))
        for row in rows:
            row["clip_id"] = clip_id
        self.writer.writerows(rows)
        self.file.flush()
        write_video(video, frames, len(frames) / seconds)

    def undo(self):
        pos, video, rewind_to = self.history.pop()
        self.file.seek(pos)
        self.file.truncate()
        self.file.flush()
        video.unlink(missing_ok=True)
        return rewind_to

    @property
    def count(self):
        return len(self.history)

    def close(self):
        self.file.close()
        if not self.history:   # nothing recorded: don't leave empty files behind
            self.csv_path.unlink(missing_ok=True)
            shutil.rmtree(self.clip_dir, ignore_errors=True)


# ---- Frame sources ----

def webcam_frames(source, size=None):
    is_webcam = source.isdigit()
    cap = cv2.VideoCapture(int(source) if is_webcam else source)
    if not cap.isOpened():
        sys.exit(f"Couldn't open camera {source!r}. Try: python record_gestures.py cameras")
    if size and is_webcam:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, size[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, size[1])
    misses = 0
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                misses = 0
                yield frame
            elif is_webcam and misses < 100:   # webcams (especially on macOS) skip frames while starting up
                misses += 1
                time.sleep(0.02)
            else:
                if is_webcam:
                    print(f"Camera {source} stopped sending frames.")
                return
    finally:
        cap.release()


def ros_to_bgr(msg):
    """sensor_msgs/Image -> BGR array. Handles nv12 (the K1 head camera), bgr8 and rgb8."""
    rows = np.frombuffer(msg.data, np.uint8).reshape(-1, msg.step)
    encoding = msg.encoding.lower()
    if encoding == "nv12":
        return cv2.cvtColor(np.ascontiguousarray(rows[:, :msg.width]), cv2.COLOR_YUV2BGR_NV12)
    image = rows[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
    if encoding == "bgr8":
        return image.copy()
    if encoding == "rgb8":
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    raise ValueError(f"Unsupported image encoding {msg.encoding!r}")


def ros_frames(topic):
    """Yield frames from a ROS2 image topic. Run from a shell with ROS2 sourced that can see the robot."""
    import rclpy
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image

    rclpy.init()
    node = rclpy.create_node("gesture_recorder")
    latest, fresh = [None], threading.Event()

    def on_image(msg):
        latest[0] = msg
        fresh.set()

    node.create_subscription(Image, topic, on_image, qos_profile_sensor_data)
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()
    try:
        while True:
            if not fresh.wait(timeout=5):
                sys.exit(f"No images on {topic} for 5 s. Check the name with: ros2 topic list")
            fresh.clear()
            yield ros_to_bgr(latest[0])
    finally:
        rclpy.shutdown()


# ---- Drawing ----

def put_text(img, text, org, scale, thickness=2):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)


CANVAS_W, BANNER_H, FOOTER_H, VIDEO_H = 1280, 170, 70, 480   # the preview is drawn at this fixed size so text stays sharp


def draw(frame, pose, hands, mirror, banner, big, color, status, progress, notice=None):
    """banner: 3 text lines on a colored bar above the video. notice: (text, bgr color) over the video bottom."""
    h, w = frame.shape[:2]
    scale = min(CANVAS_W / w, VIDEO_H / h)
    vw, vh = int(w * scale), int(h * scale)
    view = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    for points, dot in [(pose.pose_landmarks, (0, 255, 255)), (hands.hand_landmarks, (255, 0, 255))]:
        for person in points:
            for p in person:
                cv2.circle(view, (int(p.x * vw), int(p.y * vh)), 4, dot, -1)
    if mirror:
        view = cv2.flip(view, 1)   # display only; recorded frames are never mirrored
    canvas = np.zeros((BANNER_H + VIDEO_H + FOOTER_H, CANVAS_W, 3), np.uint8)
    x0, y0 = (CANVAS_W - vw) // 2, BANNER_H + (VIDEO_H - vh) // 2
    canvas[y0:y0 + vh, x0:x0 + vw] = view

    canvas[:BANNER_H] = color
    sizes = [(1.9, 4), (1.2, 3), (0.95, 2)]
    ys = [62, 112, 152]
    for text, (sc, th), y in zip(banner, sizes, ys):
        put_text(canvas, text, (30, y), sc, th)
    if big:
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 7, 14)[0]
        put_text(canvas, big, ((CANVAS_W - size[0]) // 2, BANNER_H + (VIDEO_H + size[1]) // 2), 7, 14)
    if notice:
        text, tint = notice
        cv2.rectangle(canvas, (0, BANNER_H + VIDEO_H - 50), (CANVAS_W, BANNER_H + VIDEO_H), (0, 0, 0), -1)
        cv2.putText(canvas, text, (20, BANNER_H + VIDEO_H - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.85, tint, 2, cv2.LINE_AA)

    foot = BANNER_H + VIDEO_H
    canvas[foot:] = (30, 30, 30)
    cv2.rectangle(canvas, (0, foot), (int(CANVAS_W * progress), foot + 8), (0, 200, 0), -1)   # progress through the plan
    cv2.putText(canvas, status, (20, foot + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, KEYS_HELP, (20, foot + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (170, 170, 170), 1, cv2.LINE_AA)
    cv2.imshow(WIN, canvas)


KEYS_HELP = "SPACE start | r redo | n skip | b back | p pause | m mirror | s sound | f fullscreen | q quit"


# ---- Commands ----

def record(args):
    detect = Detector()
    steps = [(d, f, label) for d in DISTANCES for f in FACINGS for label in CLIPS]
    if not 1 <= args.start <= len(steps):
        sys.exit(f"--start must be between 1 and {len(steps)}")
    session = Session(f"{args.person}_{args.camera}_{time.strftime('%Y%m%d-%H%M%S')}")
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    source = ros_frames(args.ros_topic) if args.ros_topic else webcam_frames(args.source, size)
    mirror = args.ros_topic is None   # mirror the preview for a webcam you're facing
    sound, paused, note = True, False, ""   # note: result of the last clip, shown until the next one starts

    step, clip_no = args.start - 1, 0
    state, until = "idle", 0.0        # idle -> countdown -> recording -> countdown ... -> idle
    rows, frames, t0 = [], [], 0.0
    fps, last = 0.0, time.monotonic()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WIN, CANVAS_W, BANNER_H + VIDEO_H + FOOTER_H)

    for frame in source:
        if paused:   # keep reading the camera so it doesn't buffer, but do nothing else
            key = cv2.waitKey(1) & 0xFF
            paused = key != ord("p")
            if key == ord("q"):
                break
            continue
        now = time.monotonic()
        fps = 0.9 * fps + 0.1 / max(now - last, 1e-3)
        last = now
        pose, hands = detect(frame)
        h, w = frame.shape[:2]
        distance, facing, label = steps[step]

        if state == "countdown" and now >= until:
            state, t0, rows, frames = "recording", now, [], []
            if sound:
                beep(1200, 250)
        if state == "recording":
            meta = dict(label=label, person=args.person, camera=args.camera, distance_m=distance, facing=facing,
                        hand=VARIANTS[label][clip_no % len(VARIANTS[label])][0])
            rows.append(make_row(meta, len(rows), round((now - t0) * 1000), w, h, pose, hands))
            frames.append(frame)
            if now - t0 >= CLIP_SECONDS:
                session.save_clip(rows, frames, now - t0, label, distance, facing, rewind_to=(step, clip_no))
                note, ok = clip_quality(rows)
                if not ok:
                    note += "  <- POOR, press r to redo"
                if sound:
                    beep(500 if ok else 300, 150 if ok else 500)
                clip_no += 1
                if clip_no < CLIPS[label]:
                    state, until = "countdown", now + args.rest
                else:
                    step, clip_no, state = step + 1, 0, "idle"
                    if step == len(steps):
                        break
                    distance, facing, label = steps[step]

        sides = assign_hands(pose, hands, w, h)
        status = (f"body {'OK' if pose.pose_landmarks else '--'}   L hand {'OK' if 'l' in sides else '--'}"
                  f"   R hand {'OK' if 'r' in sides else '--'}   {fps:.0f} fps   {session.count} clips saved")
        progress = (step + (clip_no / CLIPS[label] if state != "idle" else 0)) / len(steps)
        name = label.replace("_", " ").upper()
        hand, style = VARIANTS[label][clip_no % len(VARIANTS[label])]
        hand_text = "EITHER hand" if hand == "either" else f"YOUR {hand.upper()} hand"
        where = f"{distance} m away, {FACING_TEXT[facing]}"
        if state == "idle":
            hand, style = VARIANTS[label][0]
            hand_text = "EITHER hand" if hand == "either" else f"YOUR {hand.upper()} hand"
            banner = [f"NEXT: {name}  -  {hand_text}",
                      f"Step {step + 1}/{len(steps)}: stand {where}",
                      f"{PROMPTS[label].split(': ', 1)[1]}   ({CLIPS[label]} clips)   Press SPACE when ready"]
            warn = framing_warning(pose, w, h)
            notice = ("WARNING: " + warn, (0, 200, 255)) if warn else None
            if note and not warn:
                notice = (note, (0, 0, 255) if "POOR" in note else (120, 255, 120))
            draw(frame, pose, hands, mirror, banner, "", (150, 0, 0), status, progress, notice)
        else:
            banner = [f"{name}  -  {hand_text}",
                      f"clip {clip_no + 1} of {CLIPS[label]}:  {style}",
                      where]
            if state == "countdown":
                draw(frame, pose, hands, mirror, banner, str(math.ceil(until - now)), (0, 130, 220), status, progress)
            else:
                draw(frame, pose, hands, mirror, banner, "GO", (0, 150, 0), status, progress)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and state == "idle":
            state, until, note = "countdown", now + args.rest + (FIRST_COUNTDOWN - COUNTDOWN), ""
        elif key == ord("p"):
            paused = True
        elif key == ord("m"):
            mirror = not mirror
        elif key == ord("s"):
            sound = not sound
        elif key == ord("b") and state == "idle" and step > 0:
            step, clip_no = step - 1, 0   # revisit the previous step (recorded clips stay; use r to undo them)
        elif key == ord("n"):
            step, clip_no, state = step + 1, 0, "idle"
            if step == len(steps):
                break
        elif key == ord("r") and session.history:
            step, clip_no = session.undo()
            state = "idle"
        elif key == ord("f"):
            full = cv2.getWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN) == cv2.WINDOW_FULLSCREEN
            cv2.setWindowProperty(WIN, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL if full else cv2.WINDOW_FULLSCREEN)

    session.close()
    cv2.destroyAllWindows()
    if session.count:
        print(f"Saved {session.count} clips: {session.csv_path} and {session.clip_dir}/")
    else:
        print("No clips recorded, so no files were saved.")
    if step < len(steps):
        print(f"Stopped at step {step + 1}/{len(steps)}. Resume with --start {step + 1}")


def cameras(args):
    """List webcam indexes that open, so --source can be picked without guessing."""
    found = False
    for i in range(6):
        cap = cv2.VideoCapture(i)
        ok, frame = cap.read() if cap.isOpened() else (False, None)
        cap.release()
        if ok:
            found = True
            print(f"--source {i}: {frame.shape[1]}x{frame.shape[0]}")
    if not found:
        print("No webcams found.")


def extract(args):
    """Re-run landmark detection on saved clips (e.g. after changing the pipeline) into a new CSV."""
    clip_dir = Path(args.clip_dir)
    person, camera, _ = clip_dir.name.split("_")
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"{clip_dir.name}.csv"
    detect = Detector()
    videos = sorted(clip_dir.glob("*.mp4"))
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, COLUMNS, restval="")
        writer.writeheader()
        for video in videos:
            n, label, distance, facing = video.stem.split("-")
            meta = dict(clip_id=f"{clip_dir.name}-{n}", label=label, person=person, camera=camera,
                        distance_m=float(distance.rstrip("m")), facing=facing)
            cap = cv2.VideoCapture(str(video))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            i = 0
            while True:
                ok, frame = cap.read()
                if not ok:
                    break
                h, w = frame.shape[:2]
                writer.writerow(make_row(meta, i, round(i * 1000 / fps), w, h, *detect(frame)))
                i += 1
            cap.release()
    print(f"Re-extracted {len(videos)} clips -> {out_path}")


def stats(args):
    import pandas as pd

    files = sorted(Path(args.dir).glob("*.csv"))
    if not files:
        sys.exit(f"No CSVs in {args.dir}/ yet.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    clips = df.drop_duplicates("clip_id")
    print(f"{len(clips)} clips, {len(df)} frames, {len(files)} files\n")
    print("Clips per person and camera:")
    print(clips.pivot_table(index=["person", "camera"], columns="label", values="clip_id",
                            aggfunc="count", fill_value=0), "\n")
    found = pd.DataFrame({
        "camera": df["camera"],
        "distance_m": df["distance_m"],
        "body": df["pose0_x"].notna(),
        "any_hand": df["lhand0_x"].notna() | df["rhand0_x"].notna(),
    })
    print("Detection rate by camera and distance (fraction of frames):")
    print(found.groupby(["camera", "distance_m"]).mean().round(2), "\n")
    per_clip = df.groupby("clip_id").size()
    print(f"Frames per clip: median {per_clip.median():.0f}, min {per_clip.min()}")


def name_arg(text):
    if not re.fullmatch(r"[A-Za-z0-9-]+", text):
        raise argparse.ArgumentTypeError("letters, digits and hyphens only (no spaces or underscores)")
    return text.lower()


def main():
    parser = argparse.ArgumentParser(description="Record and check the K1 gesture dataset.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    rec = sub.add_parser("record", help="run a guided recording session")
    rec.add_argument("--person", required=True, type=name_arg, help="who is performing, e.g. joe")
    rec.add_argument("--camera", required=True, type=name_arg, help="laptop or k1")
    rec.add_argument("--source", default="0", help="webcam index or video file (default 0)")
    rec.add_argument("--rest", type=float, default=COUNTDOWN, help=f"seconds of rest/get-ready between clips (default {COUNTDOWN:g})")
    rec.add_argument("--size", help="webcam resolution to request, e.g. 1280x720")
    rec.add_argument("--ros-topic", help="read the K1 head camera from this ROS2 image topic instead")
    rec.add_argument("--start", type=int, default=1, help="step to start at, to resume a session")

    sub.add_parser("cameras", help="list webcams that work")

    st = sub.add_parser("stats", help="summarize what's been recorded")
    st.add_argument("--dir", default=str(DATA_DIR))

    ex = sub.add_parser("extract", help="re-run landmarks on a folder of saved clips")
    ex.add_argument("clip_dir")
    ex.add_argument("--out", default="data-reextract")

    args = parser.parse_args()
    {"record": record, "stats": stats, "extract": extract, "cameras": cameras}[args.cmd](args)


if __name__ == "__main__":
    main()