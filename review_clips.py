# Review recorded gesture clips and trim each one down to just the gesture.
#
#   python review_clips.py                                  # starts at the first unreviewed clip
#   python review_clips.py --person joe --label come_here   # only some clips
#   python review_clips.py --all                            # start from the first clip
#
# Trims are saved to data/trims/<session>.json; the CSVs and videos are never
# changed, so a trim can always be redone. train_gestures.py uses only the
# trimmed part of each clip and skips rejected clips. Commit data/trims/ with the data.
#
# Keys:
#   SPACE   play / pause (loops over the trimmed part)
#   A / D   step one frame back / forward (arrow keys also work)
#   [  ]    set the start / end of the gesture at the current frame
#   T       auto-trim to where the arm is moving
#   C       clear the trim (use the whole clip)
#   X       reject / un-reject the clip (bad take, wrong gesture...)
#   1-4     relabel: 1 come_here  2 shoo  3 stop  4 none
#   N / ENTER  accept and go to the next clip      P  previous clip
#   Q / ESC    save and quit
# Mouse: click or drag on the timeline to scrub.

import argparse
import glob
import os
import time

import cv2
import numpy as np
import pandas as pd

from record_gestures import fit_window
from gesture_common import (CLIPS_DIR, DATA_DIR, HAND_CONNECTIONS, LABELS, POSE_LINES,
                            load_trims, save_trims, session_of)

WIN = "review clips"
# The window is drawn at a fixed size (scaled by the OS) so text stays sharp:
# header bar, video, then a panel with the info line, timeline and key help.
VIEW_W = 1280
HEADER_H = 84
VIDEO_H = 520
MARGIN = 30
TL_TOP = HEADER_H + VIDEO_H + 80          # timeline box, in window coordinates
TL_BOTTOM = TL_TOP + 70
TOTAL_H = TL_BOTTOM + 128
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEY_HELP = [("SPACE", "play/pause"), ("A / D", "step frame"), ("[ ]", "set start / end"), ("T", "auto-trim"),
            ("C", "clear trim"), ("X", "reject clip"), ("1-4", "relabel"), ("N / P", "next / prev"), ("Q", "save+quit")]
LEFT_KEYS = {2424832, 65361, 63234}   # arrow codes on Windows / Linux / macOS
RIGHT_KEYS = {2555904, 65363, 63235}
LABEL_COLORS = {"come_here": (0, 200, 0), "shoo": (0, 140, 255), "stop": (60, 60, 255), "none": (200, 200, 200)}


def wrist_motion(rows):
    """Per-frame speed of the faster-moving wrist (pose landmarks 15/16), lightly smoothed."""
    speeds = []
    for i in (15, 16):
        xy = rows[[f"pose{i}_x", f"pose{i}_y"]].ffill().bfill().to_numpy(float)
        xy[:, 0] *= rows["img_w"].iloc[0] / rows["img_h"].iloc[0]
        speeds.append(np.r_[0.0, np.linalg.norm(np.diff(xy, axis=0), axis=1)])
    speed = np.nan_to_num(np.max(speeds, axis=0))
    return np.convolve(speed, np.ones(3) / 3, mode="same")


class Clip:
    def __init__(self, clip_id, rows):
        self.id = clip_id
        self.session = session_of(clip_id)
        self.rows = rows.sort_values("frame").reset_index(drop=True)
        self.n = len(self.rows)
        self.label = self.rows["label"].iloc[0]
        self.t = self.rows["t_ms"].to_numpy(float)
        self.frame_dt = float(np.median(np.diff(self.t))) / 1000 if self.n > 1 else 1 / 30
        self.motion = wrist_motion(self.rows)
        self._frames = None

    def frames(self):
        """The clip's video frames, or [] if the video isn't on this computer."""
        if self._frames is None:
            num = self.id.rsplit("-", 1)[1]
            paths = glob.glob(os.path.join(CLIPS_DIR, self.session, f"{num}-*.mp4"))
            self._frames = []
            if paths:
                cap = cv2.VideoCapture(paths[0])
                while True:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    self._frames.append(frame)
                cap.release()
        return self._frames

    def unload(self):
        self._frames = None

    def auto_trim(self):
        """Frames where the arm is clearly moving, padded a little. None if there's no clear motion."""
        m = self.motion
        if m.max() < 1e-3:
            return None
        moving = np.where(m > 0.25 * m.max())[0]
        return max(0, int(moving[0]) - 3), min(self.n - 1, int(moving[-1]) + 3)


def draw_row(img, row):
    """Draw the landmarks saved in a CSV row (so clips can be reviewed even without the video)."""
    h, w = img.shape[:2]

    def pt(prefix, i):
        x, y = row.get(f"{prefix}{i}_x"), row.get(f"{prefix}{i}_y")
        if x is None or y is None or pd.isna(x) or pd.isna(y):
            return None
        return int(x * w), int(y * h)

    pose = [pt("pose", i) for i in range(33)]
    for a, b in POSE_LINES:
        if pose[a] and pose[b]:
            cv2.line(img, pose[a], pose[b], (255, 128, 0), 2)
    for side in "lr":
        hand = [pt(f"{side}hand", i) for i in range(21)]
        for c in HAND_CONNECTIONS:
            if hand[c.start] and hand[c.end]:
                cv2.line(img, hand[c.start], hand[c.end], (0, 255, 0), 1)


def text(img, s, org, scale=0.8, color=(255, 255, 255), thickness=2):
    # dark box behind the text (an outline drawn as thicker text misaligns in OpenCV 5)
    (w, h), base = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    x, y = org
    cv2.rectangle(img, (x - 4, y - h - 4), (x + w + 4, y + base + 2), (20, 20, 20), -1)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


class Reviewer:
    def __init__(self, clips, trims, data_dir, start_all=False):
        self.clips, self.trims, self.data_dir = clips, trims, data_dir
        self.i = 0
        if not start_all:
            unreviewed = [k for k, c in enumerate(clips) if c.id not in trims]
            self.i = unreviewed[0] if unreviewed else 0
        self.playing, self.last_step = False, 0.0
        self.message, self.message_until = "", 0.0
        self.dragging = False
        self.load()

    # ---- state
    @property
    def clip(self):
        return self.clips[self.i]

    def load(self):
        c = self.clip
        self.entry = dict(self.trims.get(c.id, {"start": 0, "end": c.n - 1, "reject": False}))
        self.entry["start"] = min(self.entry["start"], c.n - 1)
        self.entry["end"] = min(self.entry["end"], c.n - 1)
        self.original = dict(self.entry)
        self.pos = self.entry["start"]
        self.playing = False

    def commit(self):
        """Save the current clip's trim (this also marks it as reviewed)."""
        c = self.clip
        self.trims[c.id] = dict(self.entry)
        save_trims(c.session, {k: v for k, v in self.trims.items() if session_of(k) == c.session},
                   self.data_dir)

    def save_if_changed(self):
        """On quit: keep edits to the current clip, but don't mark an untouched clip as reviewed."""
        if self.entry != self.original:
            self.commit()

    def go(self, k):
        if not 0 <= k < len(self.clips):
            if k >= len(self.clips):
                self.commit()   # accepting the last clip still counts as reviewing it
                self.say("saved - that was the last clip (Q to quit)")
            else:
                self.say("this is the first clip")
            return
        self.commit()
        self.clip.unload()
        self.i = k
        self.load()

    def say(self, msg):
        self.message, self.message_until = msg, time.monotonic() + 2

    @property
    def label(self):
        return self.entry.get("label", self.clip.label)

    # ---- input
    def handle_key(self, key):
        """Returns False to quit."""
        if key == -1:
            return True
        c, e = self.clip, self.entry
        ch = chr(key & 0xFF).lower() if key < 0x10000 else ""
        if key in LEFT_KEYS or ch == "a":
            self.playing, self.pos = False, max(0, self.pos - 1)
        elif key in RIGHT_KEYS or ch == "d":
            self.playing, self.pos = False, min(c.n - 1, self.pos + 1)
        elif ch == " ":
            self.playing = not self.playing
            if self.playing and not e["start"] <= self.pos < e["end"]:
                self.pos = e["start"]
        elif ch == "[":
            e["start"] = min(self.pos, e["end"])
        elif ch == "]":
            e["end"] = max(self.pos, e["start"])
        elif ch == "t":
            trim = c.auto_trim()
            if trim:
                e["start"], e["end"] = trim
                self.pos = e["start"]
                self.say("auto-trimmed to the arm motion - check it with SPACE")
            else:
                self.say("no clear motion found - trim by hand")
        elif ch == "c":
            e["start"], e["end"] = 0, c.n - 1
        elif ch == "x":
            e["reject"] = not e.get("reject", False)
        elif ch in "1234" and ch:
            new = LABELS[int(ch) - 1]
            if new == c.label:
                e.pop("label", None)
            else:
                e["label"] = new
            self.say(f"label: {new}")
        elif ch in ("n", "\r"):
            self.go(self.i + 1)
        elif ch == "p":
            self.go(self.i - 1)
        elif ch in ("q", "\x1b"):
            self.save_if_changed()
            return False
        return True

    def frame_at_x(self, x):
        c = self.clip
        frac = (x - MARGIN) / (VIEW_W - 2 * MARGIN)
        return int(round(min(max(frac, 0.0), 1.0) * (c.n - 1)))

    def on_mouse(self, event, x, y, flags, param):
        top, bottom = TL_TOP - 14, TL_BOTTOM + 14
        if event == cv2.EVENT_LBUTTONDOWN and top <= y <= bottom:
            self.dragging, self.playing = True, False
            self.pos = self.frame_at_x(x)
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            self.pos = self.frame_at_x(x)
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False

    def tick(self):
        """Advance playback, looping over the trimmed part."""
        now = time.monotonic()
        if self.playing and now - self.last_step >= self.clip.frame_dt:
            self.last_step = now
            self.pos = self.pos + 1 if self.pos < self.entry["end"] else self.entry["start"]

    # ---- drawing
    def render(self):
        c, e = self.clip, self.entry
        frames = c.frames()
        row = c.rows.iloc[self.pos].to_dict()
        if frames:
            img = frames[min(self.pos, len(frames) - 1)].copy()
        else:
            img = np.full((int(row["img_h"]), int(row["img_w"]), 3), 40, np.uint8)
            text(img, "video not on this computer - showing saved landmarks", (20, 40))
        draw_row(img, row)
        scale = min(VIEW_W / img.shape[1], VIDEO_H / img.shape[0])
        img = cv2.resize(img, (int(img.shape[1] * scale), int(img.shape[0] * scale)))
        vh, vw = img.shape[:2]

        canvas = np.full((TOTAL_H, VIEW_W, 3), 25, np.uint8)
        vx, vy = (VIEW_W - vw) // 2, HEADER_H + (VIDEO_H - vh) // 2
        canvas[HEADER_H:HEADER_H + VIDEO_H] = 0
        canvas[vy:vy + vh, vx:vx + vw] = img
        rejected = e.get("reject", False)
        inside = e["start"] <= self.pos <= e["end"]
        if rejected:
            cv2.rectangle(canvas, (vx, vy), (vx + vw - 1, vy + vh - 1), (0, 0, 255), 10)
            text(canvas, "REJECTED", (VIEW_W // 2 - 190, HEADER_H + VIDEO_H // 2), 2.4, (0, 0, 255), 6)
        elif inside:
            cv2.rectangle(canvas, (vx, vy), (vx + vw - 1, vy + vh - 1), (0, 200, 0), 6)
        badge, badge_color = ("IN GESTURE", (0, 220, 0)) if inside else ("outside trim", (160, 160, 160))
        text(canvas, badge, (vx + vw - 250, vy + 40), 1.0, badge_color, 3)

        # header: where we are, what the clip is
        reviewed = sum(cl.id in self.trims for cl in self.clips)
        label = self.label
        r0 = c.rows.iloc[0]
        header_color = LABEL_COLORS.get(label, (200, 200, 200))
        canvas[:HEADER_H] = (45, 45, 45)
        cv2.rectangle(canvas, (0, 0), (14, HEADER_H), header_color, -1)
        text(canvas, f"Clip {self.i + 1} of {len(self.clips)}", (MARGIN, 52), 1.2)
        name = label.replace("_", " ").upper() + (f"  (was {c.label})" if label != c.label else "")
        text(canvas, name, (340, 52), 1.2, header_color, 3)
        hand = r0.get("hand")
        hand = f"{hand} hand   " if isinstance(hand, str) and hand else ""
        text(canvas, f"{hand}{r0['distance_m']:g} m   {str(r0['facing']).replace('_', ' ')}", (760, 36), 0.7, (220, 220, 220), 2)
        status = "SAVED" if c.id in self.trims else "not reviewed yet"
        text(canvas, f"{status}   |   {reviewed}/{len(self.clips)} done", (760, 70), 0.7,
             (120, 255, 120) if c.id in self.trims else (170, 170, 170), 2)

        # info line
        dur = (c.t[e["end"]] - c.t[e["start"]]) / 1000
        text(canvas, f"Gesture keeps {dur:.2f} s of {c.t[-1] / 1000:.2f} s    (frames {e['start']} to {e['end']})",
             (MARGIN, HEADER_H + VIDEO_H + 34), 0.8)
        text(canvas, f"now: frame {self.pos}  ({c.t[self.pos] / 1000:.2f} s)", (VIEW_W - 360, HEADER_H + VIDEO_H + 34), 0.8,
             (255, 255, 255))

        # timeline: trimmed part highlighted, wrist-motion curve, playhead
        x = lambda f: int(MARGIN + f / max(c.n - 1, 1) * (VIEW_W - 2 * MARGIN))
        cv2.rectangle(canvas, (MARGIN, TL_TOP), (VIEW_W - MARGIN, TL_BOTTOM), (70, 70, 70), -1)
        cv2.rectangle(canvas, (x(e["start"]), TL_TOP), (x(e["end"]), TL_BOTTOM),
                      (50, 50, 170) if rejected else (50, 120, 50), -1)
        m = c.motion / max(c.motion.max(), 1e-6)
        pts = np.array([(x(f), int(TL_BOTTOM - 4 - v * (TL_BOTTOM - TL_TOP - 8))) for f, v in enumerate(m)], np.int32)
        cv2.polylines(canvas, [pts], False, (0, 220, 255), 2, cv2.LINE_AA)
        for f, tag in ((e["start"], "START"), (e["end"], "END")):
            cv2.line(canvas, (x(f), TL_TOP - 8), (x(f), TL_BOTTOM + 8), (0, 255, 0), 3)
            cv2.putText(canvas, tag, (x(f) - (50 if tag == "END" else 0), TL_TOP - 14), FONT, 0.55, (0, 255, 0), 1, cv2.LINE_AA)
        cv2.line(canvas, (x(self.pos), TL_TOP - 10), (x(self.pos), TL_BOTTOM + 10), (255, 255, 255), 3)
        cv2.putText(canvas, "yellow line = how fast your arm moves", (MARGIN, TL_BOTTOM + 28), FONT, 0.55,
                    (0, 220, 255), 1, cv2.LINE_AA)

        # keys as a grid of chips, or a status message for a moment after an action
        if time.monotonic() < self.message_until:
            text(canvas, self.message, (MARGIN, TL_BOTTOM + 78), 0.9, (0, 255, 255), 2)
        else:
            col_w = (VIEW_W - 2 * MARGIN) // 5
            for k, (key, what) in enumerate(KEY_HELP):
                cx, cy = MARGIN + (k % 5) * col_w, TL_BOTTOM + 62 + (k // 5) * 36
                cv2.putText(canvas, key, (cx, cy), FONT, 0.7, (0, 220, 255), 2, cv2.LINE_AA)
                kw = cv2.getTextSize(key, FONT, 0.7, 2)[0][0]
                cv2.putText(canvas, what, (cx + kw + 10, cy), FONT, 0.7, (200, 200, 200), 1, cv2.LINE_AA)
        return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR, help="folder of session CSVs")
    ap.add_argument("--person")
    ap.add_argument("--label", choices=LABELS)
    ap.add_argument("--session", help="only clips from this session (CSV name without .csv)")
    ap.add_argument("--all", action="store_true", help="start from the first clip, not the first unreviewed")
    args = ap.parse_args()

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    if args.session:
        files = [f for f in files if os.path.basename(f) == f"{args.session}.csv"]
    if not files:
        raise SystemExit(f"No session CSVs found in {args.data}/")
    df = pd.concat((pd.read_csv(f, low_memory=False) for f in files), ignore_index=True)
    if args.person:
        df = df[df["person"] == args.person]
    if args.label:
        df = df[df["label"] == args.label]
    clips = [Clip(cid, g) for cid, g in df.groupby("clip_id", sort=True)]
    if not clips:
        raise SystemExit("No clips match those filters.")

    reviewer = Reviewer(clips, load_trims(args.data), args.data, start_all=args.all)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    fit_window(WIN, VIEW_W, TOTAL_H)
    cv2.setMouseCallback(WIN, reviewer.on_mouse)
    while True:
        reviewer.tick()
        cv2.imshow(WIN, reviewer.render())
        if not reviewer.handle_key(cv2.waitKeyEx(15)):
            break
        if cv2.getWindowProperty(WIN, cv2.WND_PROP_VISIBLE) < 1:  # window closed with the X button
            reviewer.save_if_changed()
            break
    cv2.destroyAllWindows()
    reviewed = sum(c.id in reviewer.trims for c in clips)
    rejected = sum(reviewer.trims.get(c.id, {}).get("reject", False) for c in clips)
    print(f"reviewed {reviewed}/{len(clips)} clips ({rejected} rejected). Trims are in {args.data}/trims/")


if __name__ == "__main__":
    main()
