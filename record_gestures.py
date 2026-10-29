"""
Record the K1 gesture dataset: MediaPipe body + hand landmarks, saved as short labeled clips.

One command runs a whole guided session for one person on one camera. The script walks
through every distance x facing x gesture, shows a countdown big enough to read from 4 m,
records each clip, and saves:
    data/<person>_<camera>_<time>.csv           one row per frame (landmarks only, a few MB)
    clips/<person>_<camera>_<time>/*.mp4        the raw clips, so you can re-extract later

Setup:
    pip install mediapipe opencv-python pandas imageio-ffmpeg
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
import io
import json
import math
import re
import shutil
import subprocess
import tempfile
import sys
import threading
import time
import wave
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
CLIP_SECONDS = 3.0      # recording time per clip; longer than the classifier's 1 s window, so training takes random 1 s crops
STYLE_SECONDS = {"slow and big": 4.5, "hold still": 3.5}   # styles that need more time than CLIP_SECONDS
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


SOUND_LEAD = 0.12   # seconds before GO that the GO sound starts, to cancel playback latency (--sound-lead)


def tone(freq, ms, volume=0.6):
    """A sine beep as in-memory WAV bytes. Playing a ready-made WAV starts much faster than winsound.Beep."""
    rate = 22050
    n = int(rate * ms / 1000)
    fade = min(n // 4, 200)   # short fade in/out so it doesn't click
    wave_data = b"".join(
        int(32767 * volume * math.sin(2 * math.pi * freq * i / rate) * min(1, i / fade, (n - i) / fade)).to_bytes(2, "little", signed=True)
        for i in range(n))
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(wave_data)
    return buf.getvalue()


class Sounds:
    """Audible cues so the performer doesn't need to read the screen from 4 m.

    Cues are scheduled against the clock on timer threads, not played from the video loop, so a slow
    frame (MediaPipe) can't make them late. Windows only; other systems get silence.
    """

    def __init__(self, lead):
        self.lead = lead
        self.enabled = winsound is not None
        # Windows can't play from memory asynchronously, so write the tones to files once and play those
        self.dir = Path(tempfile.mkdtemp(prefix="gesture_sounds_"))
        self.files = {}
        for name, data in {"tick": tone(700, 70), "go": tone(1400, 350), "good": tone(500, 150), "poor": tone(300, 500)}.items():
            self.files[name] = str(self.dir / f"{name}.wav")
            (self.dir / f"{name}.wav").write_bytes(data)
        self.timers = []

    def play(self, name):
        if self.enabled:
            winsound.PlaySound(self.files[name], winsound.SND_FILENAME | winsound.SND_ASYNC)

    def countdown(self, go_at):
        """Schedule ticks at 3, 2, 1 and the GO sound for the monotonic time go_at."""
        self.cancel()
        now = time.monotonic()
        for name, when in [("tick", go_at - 3), ("tick", go_at - 2), ("tick", go_at - 1), ("go", go_at - self.lead)]:
            if when > now:
                timer = threading.Timer(when - now, self.play, args=(name,))
                timer.daemon = True
                timer.start()
                self.timers.append(timer)

    def cancel(self):
        for timer in self.timers:
            timer.cancel()
        self.timers = []

    def close(self):
        self.cancel()
        shutil.rmtree(self.dir, ignore_errors=True)


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


def to_h264(path):
    """Re-encode a clip as H.264 in place: OpenCV can only write MPEG-4 Part 2, which most players
    (browsers, VS Code video extensions) can't play. Needs `pip install imageio-ffmpeg`; returns success."""
    try:
        import imageio_ffmpeg
        ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        return False
    tmp = Path(path).with_suffix(".tmp.mp4")
    done = subprocess.run([ffmpeg, "-y", "-loglevel", "error", "-i", str(path), "-c:v", "libx264", "-crf", "18",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(tmp)])
    if done.returncode != 0:
        tmp.unlink(missing_ok=True)
        return False
    tmp.replace(path)
    return True


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
    to_h264(path)


def full_plan():
    """Every step of a fresh session: (step number, distance, facing, label, [(hand, style) per clip])."""
    plan = []
    for d in DISTANCES:
        for f in FACINGS:
            for label in CLIPS:
                variants = [VARIANTS[label][i % len(VARIANTS[label])] for i in range(CLIPS[label])]
                plan.append((len(plan) + 1, d, f, label, variants))
    return plan


def session_clips(name):
    """[(clip_id, label, distance, facing, hand)] for each clip saved in a session's CSV, in order."""
    path = DATA_DIR / f"{name}.csv"
    if not path.exists():
        sys.exit(f"No session {name!r}: {path} doesn't exist. Use a name from the data/ folder.")
    clips = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            clips.setdefault(row["clip_id"], (row["clip_id"], row["label"], float(row["distance_m"]),
                                              row["facing"], row.get("hand") or ""))
    return list(clips.values())


def resume_plan(name):
    """The steps still to record for an old session: missing clips, plus replacements for rejected ones.

    A clip counts toward its step unless it was rejected in review_clips.py (data/trims/<name>.json).
    Which hand/style is still missing is worked out from the hands the good clips already used.
    """
    trims_path = DATA_DIR / "trims" / f"{name}.json"
    trims = json.loads(trims_path.read_text()) if trims_path.exists() else {}
    good = {}
    for clip_id, label, distance, facing, hand in session_clips(name):
        if not trims.get(clip_id, {}).get("reject", False):
            good.setdefault((distance, facing, label), []).append(hand)
    plan = []
    for stepno, d, f, label, variants in full_plan():
        missing = list(variants)
        for hand in good.get((float(d), f, label), []):
            if missing:
                missing.remove(next((v for v in missing if v[0] == hand), missing[0]))
        if missing:
            plan.append((stepno, d, f, label, missing))
    return plan


class Session:
    """One CSV + one folder of clips per recording session. Can undo the last clip."""

    def __init__(self, name, append=False):
        self.name = name
        self.append = append   # continuing an old session: add to its files, never delete them
        self.clip_dir = CLIPS_DIR / name
        self.clip_dir.mkdir(parents=True, exist_ok=True)
        DATA_DIR.mkdir(exist_ok=True)
        self.csv_path = DATA_DIR / f"{name}.csv"
        self.history = []   # (CSV position before the clip, video path, what to rewind to on undo)
        self.n = 0          # clip numbers are never reused, even after an undo
        if append:
            for clip_id, *_ in session_clips(name):
                self.n = max(self.n, int(clip_id.rsplit("-", 1)[1]))
            for video in self.clip_dir.glob("*.mp4"):
                self.n = max(self.n, int(video.stem.split("-")[0]))
            self.file = open(self.csv_path, "r+", newline="")
            self.file.seek(0, 2)
            self.writer = csv.DictWriter(self.file, COLUMNS, restval="")
        else:
            self.file = open(self.csv_path, "w", newline="")
            self.writer = csv.DictWriter(self.file, COLUMNS, restval="")
            self.writer.writeheader()
            self.file.flush()

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
        if not self.history and not self.append:   # nothing recorded: don't leave empty files behind
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
    misses, started = 0, False
    try:
        while True:
            ok, frame = cap.read()
            if ok:
                misses, started = 0, True
                yield frame
            elif is_webcam and misses < (150 if started else 500):   # webcams can take several seconds to start up
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


CANVAS_W, BANNER_H, FOOTER_H, VIDEO_H = 1280, 150, 64, 480   # drawn at this fixed size so text stays sharp
BG_IDLE, BG_GET_READY, BG_RECORDING = (80, 55, 35), (0, 120, 220), (0, 0, 190)   # banner colors (BGR)
GREEN, RED, GREY = (90, 220, 90), (60, 60, 255), (120, 120, 120)


def screen_size():
    """(width, height) of the main screen in pixels, or None if it can't be found."""
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return user32.GetSystemMetrics(0), user32.GetSystemMetrics(1)
    except (AttributeError, OSError):
        pass
    try:
        import tkinter
        root = tkinter.Tk()
        size = root.winfo_screenwidth(), root.winfo_screenheight()
        root.destroy()
        return size
    except Exception:
        return None


def fit_window(win, width, height):
    """Open the window at its drawn size, or smaller so it fits comfortably on this screen."""
    screen = screen_size()
    scale = min(1.0, 0.92 * screen[0] / width, 0.82 * screen[1] / height) if screen else 1.0
    cv2.resizeWindow(win, int(width * scale), int(height * scale))


STYLES = list(dict.fromkeys(style for variants in VARIANTS.values() for _, style in variants))
LENGTH_BTN = (20, BANNER_H + 16, 260, 46)            # x, y, w, h of the "Clip lengths" button
PANEL_X, PANEL_Y, PANEL_W, ROW_H = 20, BANNER_H + 70, 600, 52


def panel_rects(row):
    """(minus button, plus button) rectangles for a row of the clip-length panel."""
    y = PANEL_Y + row * ROW_H + 6
    return (PANEL_X + 400, y, 46, 40), (PANEL_X + 530, y, 46, 40)


def inside(rect, x, y):
    return rect[0] <= x <= rect[0] + rect[2] and rect[1] <= y <= rect[1] + rect[3]


def button(canvas, rect, text, fill=(70, 70, 70)):
    x, y, w, h = rect
    cv2.rectangle(canvas, (x, y), (x + w, y + h), fill, -1)
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (200, 200, 200), 2)
    size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
    cv2.putText(canvas, text, (x + (w - size[0]) // 2, y + (h + size[1]) // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                (255, 255, 255), 2, cv2.LINE_AA)


def dot(canvas, x, y, ok, label):
    cv2.circle(canvas, (x, y), 9, GREEN if ok else RED, -1)
    cv2.putText(canvas, label, (x + 20, y + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (230, 230, 230), 1, cv2.LINE_AA)


def draw(frame, pose, hands, mirror, banner, big, color, det, info, progress, notice=None, recording=False, show_help=False,
         lengths=None, panel_open=False):
    """banner: 3 text lines on a colored bar above the video. det: (body, left, right) found flags.
    notice: (text, color) shown over the bottom of the video. recording: red frame + REC badge."""
    h, w = frame.shape[:2]
    scale = min(CANVAS_W / w, VIDEO_H / h)
    vw, vh = int(w * scale), int(h * scale)
    view = cv2.resize(frame, (vw, vh), interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR)
    for points, tint in [(pose.pose_landmarks, (0, 255, 255)), (hands.hand_landmarks, (255, 0, 255))]:
        for person in points:
            for p in person:
                cv2.circle(view, (int(p.x * vw), int(p.y * vh)), 4, tint, -1)
    if mirror:
        view = cv2.flip(view, 1)   # display only; recorded frames are never mirrored
    canvas = np.zeros((BANNER_H + VIDEO_H + FOOTER_H, CANVAS_W, 3), np.uint8)
    x0, y0 = (CANVAS_W - vw) // 2, BANNER_H + (VIDEO_H - vh) // 2
    canvas[y0:y0 + vh, x0:x0 + vw] = view

    canvas[:BANNER_H] = color
    for text, (sc, th), y in zip(banner, [(1.7, 4), (1.05, 2), (0.85, 2)], [60, 105, 138]):
        put_text(canvas, text, (30, y), sc, th)
    if recording:   # blinking badge, so it's obvious the camera is saving
        on = int(time.monotonic() * 3) % 2 == 0
        cv2.circle(canvas, (CANVAS_W - 290, 52), 22, (255, 255, 255) if on else (140, 140, 255), -1)
        put_text(canvas, "REC", (CANVAS_W - 250, 66), 1.4, 3)
    else:
        cv2.circle(canvas, (CANVAS_W - 290, 52), 22, (200, 200, 200), 3)
        put_text(canvas, "not recording", (CANVAS_W - 250, 60), 0.8, 2)

    if big:
        size = cv2.getTextSize(big, cv2.FONT_HERSHEY_SIMPLEX, 6, 12)[0]
        cx, cy = CANVAS_W // 2, BANNER_H + VIDEO_H // 2
        shade = canvas.copy()
        cv2.circle(shade, (cx, cy), 120, (0, 0, 0), -1)   # dark disc so the number reads on any background
        canvas = cv2.addWeighted(shade, 0.55, canvas, 0.45, 0)
        put_text(canvas, big, (cx - size[0] // 2, cy + size[1] // 2), 6, 12)
    if notice:
        text, tint = notice
        cv2.rectangle(canvas, (0, BANNER_H + VIDEO_H - 52), (CANVAS_W, BANNER_H + VIDEO_H), (0, 0, 0), -1)
        cv2.putText(canvas, text, (24, BANNER_H + VIDEO_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.85, tint, 2, cv2.LINE_AA)

    foot = BANNER_H + VIDEO_H
    canvas[foot:] = (32, 32, 32)
    cv2.rectangle(canvas, (0, foot), (int(CANVAS_W * progress), foot + 6), GREEN, -1)   # progress through the plan
    for k, (ok, label) in enumerate(zip(det, ("body", "left hand", "right hand"))):
        dot(canvas, 34 + k * 190, foot + 36, ok, label)
    cv2.putText(canvas, info, (600, foot + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.putText(canvas, "SPACE start   H help", (CANVAS_W - 260, foot + 42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (160, 160, 160), 1, cv2.LINE_AA)
    if lengths is not None:   # click the button to open a panel for changing how long each kind of clip records
        button(canvas, LENGTH_BTN, "Clip lengths " + ("^" if panel_open else "v"))
        if panel_open:
            shade = canvas.copy()
            cv2.rectangle(shade, (PANEL_X, PANEL_Y), (PANEL_X + PANEL_W, PANEL_Y + ROW_H * len(STYLES) + 12), (20, 20, 20), -1)
            canvas = cv2.addWeighted(shade, 0.9, canvas, 0.1, 0)
            for k, style in enumerate(STYLES):
                y = PANEL_Y + k * ROW_H
                cv2.putText(canvas, style, (PANEL_X + 20, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (230, 230, 230), 2, cv2.LINE_AA)
                minus, plus = panel_rects(k)
                button(canvas, minus, "-")
                button(canvas, plus, "+")
                cv2.putText(canvas, f"{lengths[style]:g} s", (PANEL_X + 458, y + 38), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 220, 255), 2, cv2.LINE_AA)
    if show_help:
        shade = canvas.copy()
        cv2.rectangle(shade, (0, BANNER_H), (CANVAS_W, foot), (0, 0, 0), -1)
        canvas = cv2.addWeighted(shade, 0.8, canvas, 0.2, 0)
        for k, (key, what) in enumerate(KEYS_HELP):
            y = BANNER_H + 55 + k * 40
            cv2.putText(canvas, key, (330, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 220, 255), 2, cv2.LINE_AA)
            cv2.putText(canvas, what, (560, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (230, 230, 230), 2, cv2.LINE_AA)
    if recording:
        cv2.rectangle(canvas, (0, 0), (CANVAS_W - 1, canvas.shape[0] - 1), (0, 0, 255), 14)
    cv2.imshow(WIN, canvas)


KEYS_HELP = [("SPACE", "start the step"), ("- / +", "shorter / longer clips (or click Clip lengths)"), ("R", "redo the last clip"), ("N", "skip this step"), ("B", "back one step"),
             ("P", "pause"), ("M", "flip the mirrored preview"), ("S", "sound on / off"), ("F", "fullscreen"),
             ("H", "show / hide this help"), ("Q", "quit (everything is saved)")]


# ---- Commands ----

def record(args):
    detect = Detector()
    total = len(full_plan())
    if args.resume:
        name = Path(args.resume).stem   # accepts the run name, data/<name>.csv or clips/<name>
        person, camera = name.split("_")[:2]
        steps = resume_plan(name)
        if not steps:
            sys.exit(f"Nothing left to record in {name}: every step has all its good (not rejected) clips.")
        redo = sum(len(v) for *_, v in steps)
        print(f"Continuing {name}: {redo} clips to record in {len(steps)} steps.")
        session, start = Session(name, append=True), 0
    else:
        if not (args.person and args.camera):
            sys.exit("Give --person and --camera, or --resume <session> to continue an old session.")
        person, camera = args.person, args.camera
        steps = full_plan()
        if not 1 <= args.start <= total:
            sys.exit(f"--start must be between 1 and {total}")
        session, start = Session(f"{person}_{camera}_{time.strftime('%Y%m%d-%H%M%S')}"), args.start - 1
    size = tuple(int(v) for v in args.size.lower().split("x")) if args.size else None
    source = ros_frames(args.ros_topic) if args.ros_topic else webcam_frames(args.source, size)
    mirror = args.ros_topic is None   # mirror the preview for a webcam you're facing
    lengths = {style: STYLE_SECONDS.get(style, CLIP_SECONDS) for style in STYLES}   # seconds per style, editable while recording
    panel_open, clicks = False, []

    def secs_for(style):
        return lengths.get(style, CLIP_SECONDS)

    sounds, scheduled, show_help = Sounds(args.sound_lead), False, False
    sound, paused, note = True, False, ""   # note: result of the last clip, shown until the next one starts

    step, clip_no = start, 0
    state, until = "idle", 0.0        # idle -> countdown -> recording -> countdown ... -> idle
    rows, frames, t0 = [], [], 0.0
    fps, last = 0.0, time.monotonic()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    fit_window(WIN, CANVAS_W, BANNER_H + VIDEO_H + FOOTER_H)
    cv2.setMouseCallback(WIN, lambda event, x, y, flags, param: clicks.append((x, y)) if event == cv2.EVENT_LBUTTONDOWN else None)

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
        stepno, distance, facing, label, variants = steps[step]

        if state == "countdown" and not scheduled:
            scheduled = True
            if sound:
                sounds.countdown(until)
        elif state != "countdown":
            scheduled = False
        if state == "countdown" and now >= until:
            state, t0, rows, frames = "recording", now, [], []
        if state == "recording":
            meta = dict(label=label, person=person, camera=camera, distance_m=distance, facing=facing,
                        hand=variants[clip_no][0])
            rows.append(make_row(meta, len(rows), round((now - t0) * 1000), w, h, pose, hands))
            frames.append(frame)
            if now - t0 >= secs_for(variants[clip_no][1]):
                session.save_clip(rows, frames, now - t0, label, distance, facing, rewind_to=(step, clip_no))
                note, ok = clip_quality(rows)
                if not ok:
                    note += "  <- POOR, press r to redo"
                if sound:
                    sounds.play("good" if ok else "poor")
                clip_no += 1
                if clip_no < len(variants):
                    state, until = "countdown", now + args.rest
                else:
                    step, clip_no, state = step + 1, 0, "idle"
                    if step == len(steps):
                        break
                    stepno, distance, facing, label, variants = steps[step]

        sides = assign_hands(pose, hands, w, h)
        det = (bool(pose.pose_landmarks), "l" in sides, "r" in sides)
        info = f"{session.count} clips saved   |   {fps:.0f} fps"
        progress = (step + (clip_no / len(variants) if state != "idle" else 0)) / len(steps)
        name = label.replace("_", " ").upper()
        hand, style = variants[min(clip_no, len(variants) - 1)]
        hand_text = "EITHER hand" if hand == "either" else f"YOUR {hand.upper()} hand"
        where = f"{distance} m away, {FACING_TEXT[facing]}"
        if state == "idle":
            hand, style = variants[0]
            hand_text = "EITHER hand" if hand == "either" else f"YOUR {hand.upper()} hand"
            banner = [f"NEXT: {name}  -  {hand_text}",
                      f"Step {stepno}/{total}: stand {where}",
                      f"{PROMPTS[label].split(': ', 1)[1]}   ({len(variants)} clips x {secs_for(style):g} s)"]
            warn = framing_warning(pose, w, h)
            notice = ("WARNING: " + warn, (0, 200, 255)) if warn else None
            if note and not warn:
                notice = (note, (0, 0, 255) if "POOR" in note else (120, 255, 120))
            draw(frame, pose, hands, mirror, banner, "", BG_IDLE, det, info, progress, notice, False, show_help,
                 lengths, panel_open)
        else:
            banner = [f"{name}  -  {hand_text}",
                      f"clip {clip_no + 1} of {len(variants)}:  {style}  ({secs_for(style):g} s)",
                      where]
            if state == "countdown":
                draw(frame, pose, hands, mirror, banner, str(math.ceil(until - now)), BG_GET_READY, det, info, progress,
                     show_help=show_help, lengths=lengths, panel_open=panel_open)
            else:
                draw(frame, pose, hands, mirror, banner, "GO", BG_RECORDING, det, info, progress, recording=True)

        while clicks:
            x, y = clicks.pop()
            if state == "recording":
                continue
            if inside(LENGTH_BTN, x, y):
                panel_open = not panel_open
            elif panel_open:
                for k, style in enumerate(STYLES):
                    for rect, delta in zip(panel_rects(k), (-0.5, 0.5)):
                        if inside(rect, x, y):
                            lengths[style] = min(10.0, max(1.0, lengths[style] + delta))

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" ") and state == "idle":
            state, until, note = "countdown", now + args.rest + (FIRST_COUNTDOWN - COUNTDOWN), ""
        elif key in (ord("-"), ord("="), ord("+")) and state != "recording":
            style = variants[min(clip_no, len(variants) - 1)][1] if state != "idle" else variants[0][1]
            lengths[style] = min(10.0, max(1.0, secs_for(style) + (-0.5 if key == ord("-") else 0.5)))
            note = f"{style}: clips are now {lengths[style]:g} s"
        elif key == ord("h"):
            show_help = not show_help
        elif key == ord("p"):
            paused = True
            sounds.cancel()
        elif key == ord("m"):
            mirror = not mirror
        elif key == ord("s"):
            sound = not sound
            sounds.cancel()
            scheduled = False   # re-arm: turning sound on mid-countdown schedules the rest
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

    sounds.close()
    session.close()
    cv2.destroyAllWindows()
    if session.count:
        print(f"Saved {session.count} clips: {session.csv_path} and {session.clip_dir}/")
    else:
        print("No clips recorded, so no files were saved.")
    if step < len(steps):
        if args.resume or session.count:
            print(f"Stopped early. Continue (and redo anything rejected) with: "
                  f"python record_gestures.py record --resume {session.name}")
        else:
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


def convert(args):
    """Make already-saved clips playable in browsers and video extensions (H.264)."""
    try:
        import imageio_ffmpeg   # noqa: F401
    except ImportError:
        sys.exit("Install the converter first: python -m pip install imageio-ffmpeg")
    videos = sorted(Path(args.path).rglob("*.mp4")) if Path(args.path).is_dir() else [Path(args.path)]
    for n, video in enumerate(videos, 1):
        cap = cv2.VideoCapture(str(video))
        already = int(cap.get(cv2.CAP_PROP_FOURCC)).to_bytes(4, "little") in (b"avc1", b"h264")
        cap.release()
        if not already:
            print(f"[{n}/{len(videos)}] {video}", "ok" if to_h264(video) else "FAILED")
    print("Done.")


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
    rec.add_argument("--person", type=name_arg, help="who is performing, e.g. joe")
    rec.add_argument("--camera", type=name_arg, help="laptop or k1")
    rec.add_argument("--source", default="0", help="webcam index or video file (default 0)")
    rec.add_argument("--resume", metavar="SESSION", help="continue an old session: record its missing clips and replace rejected ones")
    rec.add_argument("--rest", type=float, default=COUNTDOWN, help=f"seconds of rest/get-ready between clips (default {COUNTDOWN:g})")
    rec.add_argument("--sound-lead", type=float, default=SOUND_LEAD, help="start the GO sound this many seconds early (raise if it sounds late)")
    rec.add_argument("--size", help="webcam resolution to request, e.g. 1280x720")
    rec.add_argument("--ros-topic", help="read the K1 head camera from this ROS2 image topic instead")
    rec.add_argument("--start", type=int, default=1, help="step to start at, to resume a session")

    sub.add_parser("cameras", help="list webcams that work")

    st = sub.add_parser("stats", help="summarize what's been recorded")
    st.add_argument("--dir", default=str(DATA_DIR))

    cv = sub.add_parser("convert", help="re-encode saved clips as H.264 so video players can open them")
    cv.add_argument("path", nargs="?", default=str(CLIPS_DIR), help="a clip, or a folder of clips (default clips/)")

    ex = sub.add_parser("extract", help="re-run landmarks on a folder of saved clips")
    ex.add_argument("clip_dir")
    ex.add_argument("--out", default="data-reextract")

    args = parser.parse_args()
    {"record": record, "stats": stats, "extract": extract, "cameras": cameras, "convert": convert}[args.cmd](args)


if __name__ == "__main__":
    main()