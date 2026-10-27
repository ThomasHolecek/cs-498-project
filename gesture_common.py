# Shared pieces for the gesture classifier: train_gestures.py and live_gestures.py.
#
# Training data comes from record_gestures.py (see RECORDING.md): data/*.csv with
# one row per frame, grouped into clips by clip_id. Features are computed here
# from the raw landmarks, so the feature design can change without re-recording.

import glob
import json
import os
import threading

import cv2
import numpy as np
from mediapipe.tasks.python import vision

from record_gestures import CLIPS

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(HERE, "data")
CLIPS_DIR = os.path.join(HERE, "clips")
MODEL_DIR = os.path.join(HERE, "models")

LABELS = list(CLIPS)   # come_here, shoo, stop, none
WINDOW_S = 1.0         # the classifier looks at the last 1 s; clips are 1.5 s so training takes random crops
SEQ_LEN = 24           # each window is resampled to this many time steps

N_HAND = 21
N_POSE = 33
SIDES = ["l", "r"]     # the person's own left / right hand (record_gestures matches hands to pose wrists)
HAND_COLS = {s: [f"{s}hand{i}_{a}" for i in range(N_HAND) for a in "xyz"] for s in SIDES}
POSE_COLS = [f"pose{i}_{a}" for i in range(N_POSE) for a in "xyzv"]
ARM_JOINTS = [11, 13, 15, 12, 14, 16]  # shoulders, elbows, wrists: left arm then right arm

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS
POSE_LINES = [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]


# ---------------------------------------------------------------- trims
# review_clips.py saves, per clip: {"start": frame, "end": frame, "reject": bool,
# "label": new label (only if relabelled)} in data/trims/<session>.json.
# The raw CSVs and videos are never modified.

def session_of(clip_id):
    return clip_id.rsplit("-", 1)[0]


def load_trims(data_dir=DATA_DIR):
    trims = {}
    for path in sorted(glob.glob(os.path.join(data_dir, "trims", "*.json"))):
        with open(path) as f:
            trims.update(json.load(f))
    return trims


def save_trims(session, entries, data_dir=DATA_DIR):
    os.makedirs(os.path.join(data_dir, "trims"), exist_ok=True)
    with open(os.path.join(data_dir, "trims", f"{session}.json"), "w") as f:
        json.dump(entries, f, indent=1, sort_keys=True)


def crop_range(t0, t1, clip_end, length_ms):
    """Range of window start times (ms) such that a window covers as much of the
    trimmed gesture [t0, t1] as possible while staying inside the clip."""
    if t1 - t0 >= length_ms:
        lo, hi = t0, t1 - length_ms
    else:
        lo, hi = t1 - length_ms, t0
    last = max(0.0, clip_end - length_ms)
    lo, hi = min(max(lo, 0.0), last), min(max(hi, 0.0), last)
    return lo, max(lo, hi)


# ---------------------------------------------------------------- camera

class LatestFrameCapture:
    """Reads a network stream on a background thread and only keeps the newest
    frame, so slow processing never builds up a backlog of stale frames."""

    def __init__(self, url):
        self.cap = cv2.VideoCapture(url)
        self.opened = self.ok = self.cap.isOpened()
        self.frame, self.seq, self.seen = None, 0, 0
        self.cond = threading.Condition()
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def _loop(self):
        try:
            while self.ok:
                try:
                    ok, frame = self.cap.read()
                except cv2.error:
                    ok, frame = False, None
                with self.cond:
                    self.ok = self.ok and ok
                    if ok:
                        self.frame, self.seq = frame, self.seq + 1
                    self.cond.notify_all()
        finally:
            self.cap.release()  # only this thread touches cap once started

    def isOpened(self):
        return self.opened

    def read(self):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != self.seen or not self.ok, timeout=5)
            if self.frame is None or self.seq == self.seen:
                return False, None
            self.seen = self.seq
            return True, self.frame

    def release(self):
        self.ok = False
        self.thread.join(timeout=2)


def open_camera(src):
    """src is a webcam index ("0") or a stream URL (e.g. the K1 MJPEG stream)."""
    if str(src).isdigit():
        return cv2.VideoCapture(int(src))
    return LatestFrameCapture(src)


def draw_landmarks(frame, pose, hands):
    h, w = frame.shape[:2]
    if pose.pose_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in pose.pose_landmarks[0]]
        for a, b in POSE_LINES:
            cv2.line(frame, pts[a], pts[b], (255, 128, 0), 2)
    for lms in hands.hand_landmarks:
        pts = [(int(lm.x * w), int(lm.y * h)) for lm in lms]
        for c in HAND_CONNECTIONS:
            cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (0, 0, 255), -1)


# ---------------------------------------------------------------- features

def _fill(arr, mask):
    """Forward-fill rows of arr where mask is False (back-fill the start). Zeros if never valid."""
    idx = np.where(mask)[0]
    if len(idx) == 0:
        return np.zeros_like(arr)
    last = np.maximum.accumulate(np.where(mask, np.arange(len(mask)), -1))
    last[last < 0] = idx[0]
    return arr[last]


def _pick_active_hand(hands, present, pose, pose_ok):
    """The gesturing hand is the one whose wrist moves the most. Uses the pose wrists
    when available (they survive at 3-4 m where the hand detector drops out)."""
    best, best_score = "r", -1.0
    for s, wrist in (("l", 15), ("r", 16)):
        if pose_ok.sum() > 1:
            w = pose[pose_ok, wrist, :2]
        elif present[s].sum() > 1:
            w = hands[s][present[s], 0, :2]
        else:
            continue
        score = np.abs(np.diff(w, axis=0)).sum()
        if score > best_score:
            best, best_score = s, score
    return best


def frame_features(df):
    """Per-frame features (N, F), invariant to distance from camera and to which hand is used."""
    n = len(df)
    aspect = float(df["img_w"].iloc[0]) / float(df["img_h"].iloc[0])
    hands = {s: df[HAND_COLS[s]].to_numpy(float).reshape(n, N_HAND, 3) for s in SIDES}
    present = {s: ~np.isnan(hands[s][:, 0, 0]) for s in SIDES}
    pose = df[POSE_COLS].to_numpy(float).reshape(n, N_POSE, 4)[:, :, :3]
    pose_ok = ~np.isnan(pose[:, 0, 0])

    side = _pick_active_hand(hands, present, pose, pose_ok)
    hand_ok = present[side]
    hand = _fill(np.nan_to_num(hands[side]), hand_ok)
    pose = _fill(np.nan_to_num(pose), pose_ok)
    # undo the non-square normalised image coords
    hand[..., 0] *= aspect
    pose[..., 0] *= aspect

    # body frame: origin at shoulder midpoint, unit = shoulder width
    ls, rs = pose[:, 11, :2], pose[:, 12, :2]
    center = (ls + rs) / 2
    scale = np.linalg.norm(ls - rs, axis=1, keepdims=True)
    scale = np.where(scale < 1e-4, 1.0, scale)
    arms = (pose[:, ARM_JOINTS, :2] - center[:, None]) / scale[:, None]       # (N,6,2)
    wrist_body = (hand[:, 0, :2] - center) / scale                             # (N,2)

    # hand shape: wrist-relative, unit = wrist -> middle knuckle length
    palm = np.linalg.norm(hand[:, 9, :2] - hand[:, 0, :2], axis=1)[:, None, None]
    palm = np.where(palm < 1e-4, 1.0, palm)
    shape = (hand - hand[:, :1]) / palm                                        # (N,21,3)

    # mirror left-handed gestures onto the right hand
    if side == "l":
        shape[..., 0] *= -1
        wrist_body[:, 0] *= -1
        arms[..., 0] *= -1
        arms = arms[:, [3, 4, 5, 0, 1, 2]]

    return np.concatenate([
        hand_ok[:, None].astype(float),
        pose_ok[:, None].astype(float),
        shape.reshape(n, -1),
        wrist_body,
        arms.reshape(n, -1),
    ], axis=1)


def resample(x, t, n=SEQ_LEN):
    """Resample (N,F) frames at times t onto n evenly spaced time steps."""
    if len(x) == 1:
        return np.repeat(x, n, axis=0)
    tt = np.linspace(t[0], t[-1], n)
    return np.stack([np.interp(tt, t, x[:, j]) for j in range(x.shape[1])], axis=1)


def window(df, start_ms, length_s=WINDOW_S):
    """The rows of a clip within [start_ms, start_ms + length]."""
    t = df["t_ms"].to_numpy(float)
    return df[(t >= start_ms) & (t <= start_ms + length_s * 1000)]


def clip_features(df, n=SEQ_LEN):
    """(n, 2F) array: resampled per-frame features plus their frame-to-frame velocity."""
    x = resample(frame_features(df), df["t_ms"].to_numpy(float), n)
    v = np.diff(x, axis=0, prepend=x[:1])
    return np.concatenate([x, v], axis=1).astype(np.float32)
