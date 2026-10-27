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

from gesture_common import (CLIPS_DIR, DATA_DIR, HAND_CONNECTIONS, LABELS, POSE_LINES,
                            load_trims, save_trims, session_of)

WIN = "review clips"
VIEW_W = 960
PANEL_H = 140
MARGIN = 20
TL_TOP, TL_BOTTOM = 62, 104   # timeline position inside the panel
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


def text(img, s, org, scale=0.55, color=(255, 255, 255), thickness=1):
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
        self.view_h = 0
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
            self.say("that was the last clip" if k >= len(self.clips) else "this is the first clip")
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
        top = self.view_h + TL_TOP - 8
        bottom = self.view_h + TL_BOTTOM + 8
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
        img = cv2.resize(img, (VIEW_W, int(img.shape[0] * VIEW_W / img.shape[1])))
        self.view_h = img.shape[0]
        if e.get("reject"):
            cv2.rectangle(img, (0, 0), (VIEW_W - 1, self.view_h - 1), (0, 0, 255), 10)
            text(img, "REJECTED", (VIEW_W // 2 - 110, self.view_h // 2), 1.6, (0, 0, 255), 4)
        inside = e["start"] <= self.pos <= e["end"]
        text(img, "IN GESTURE" if inside else "outside trim", (VIEW_W - 190, 30), 0.7,
             (0, 255, 0) if inside else (150, 150, 150), 2)

        panel = np.full((PANEL_H, VIEW_W, 3), 25, np.uint8)
        reviewed = sum(cl.id in self.trims for cl in self.clips)
        label = self.label
        relabel = f" (was {c.label})" if label != c.label else ""
        r0 = c.rows.iloc[0]
        text(panel, f"{self.i + 1}/{len(self.clips)}  {c.id}", (MARGIN, 22))
        text(panel, f"{label}{relabel}", (MARGIN + 430, 22), 0.6, LABEL_COLORS.get(label, (255, 255, 255)), 2)
        text(panel, f"{r0['distance_m']} m {r0['facing']}   reviewed {reviewed}/{len(self.clips)}"
                    + ("   [saved]" if c.id in self.trims else ""), (MARGIN + 600, 22), 0.5)
        dur = (c.t[e["end"]] - c.t[e["start"]]) / 1000
        text(panel, f"gesture: frames {e['start']}-{e['end']}  ({dur:.2f} s of {c.t[-1] / 1000:.2f} s)"
                    f"    frame {self.pos}  ({c.t[self.pos] / 1000:.2f} s)", (MARGIN, 46))

        # timeline: trimmed part highlighted, wrist-motion curve, playhead
        x = lambda f: int(MARGIN + f / max(c.n - 1, 1) * (VIEW_W - 2 * MARGIN))
        cv2.rectangle(panel, (MARGIN, TL_TOP), (VIEW_W - MARGIN, TL_BOTTOM), (60, 60, 60), -1)
        cv2.rectangle(panel, (x(e["start"]), TL_TOP), (x(e["end"]), TL_BOTTOM),
                      (40, 40, 160) if e.get("reject") else (40, 110, 40), -1)
        m = c.motion / max(c.motion.max(), 1e-6)
        pts = np.array([(x(f), int(TL_BOTTOM - 3 - v * (TL_BOTTOM - TL_TOP - 6))) for f, v in enumerate(m)], np.int32)
        cv2.polylines(panel, [pts], False, (0, 220, 255), 1, cv2.LINE_AA)
        for f in (e["start"], e["end"]):
            cv2.line(panel, (x(f), TL_TOP - 4), (x(f), TL_BOTTOM + 4), (0, 255, 0), 2)
        cv2.line(panel, (x(self.pos), TL_TOP - 6), (x(self.pos), TL_BOTTOM + 6), (255, 255, 255), 2)

        if time.monotonic() < self.message_until:
            text(panel, self.message, (MARGIN, 128), 0.55, (0, 255, 255))
        else:
            text(panel, "SPACE play  A/D step  [ ] start/end  T auto  C clear  X reject  "
                        "1-4 label  N next  P prev  Q quit", (MARGIN, 128), 0.45, (180, 180, 180))
        return np.vstack([img, panel])


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
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    if args.person:
        df = df[df["person"] == args.person]
    if args.label:
        df = df[df["label"] == args.label]
    clips = [Clip(cid, g) for cid, g in df.groupby("clip_id", sort=True)]
    if not clips:
        raise SystemExit("No clips match those filters.")

    reviewer = Reviewer(clips, load_trims(args.data), args.data, start_all=args.all)
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
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
