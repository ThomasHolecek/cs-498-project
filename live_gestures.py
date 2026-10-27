# Run the trained gesture classifier live on a camera, using a sliding window
# the same length as the training crops. Without a trained model it still shows
# the landmarks and MediaPipe's built-in gestures, to check detection.
#
#   python live_gestures.py [--camera 0] [--threshold 0.8] [--save]
#   python live_gestures.py --camera http://<robot-ip>:8080/stream.mjpg --save   # K1 camera

import argparse
import logging
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np
import pandas as pd
import torch

from gesture_common import HERE, MODEL_DIR, clip_features, draw_landmarks, open_camera
from gesture_model import load_checkpoint
from record_gestures import COLUMNS, Detector, make_row

COOLDOWN_S = 1.5  # don't fire the same command again within this time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--camera", default="0",
                    help="webcam index, or a stream URL like http://<robot-ip>:8080/stream.mjpg")
    ap.add_argument("--model", default=os.path.join(MODEL_DIR, "gesture_gru.pt"))
    ap.add_argument("--threshold", type=float, default=0.8)
    ap.add_argument("--save", action="store_true", help="save the annotated view to run_logs/*.mp4")
    args = ap.parse_args()
    os.chdir(HERE)  # Detector loads the .task models from the current folder

    if os.path.exists(args.model):
        model, ckpt = load_checkpoint(args.model)
        labels, window_s, seq_len = ckpt["labels"], ckpt["window_s"], ckpt["seq_len"]
        mean, std = ckpt["mean"], ckpt["std"]
    else:
        print(f"no model at {args.model} - showing landmarks and built-in gestures only")
        model, labels, window_s, seq_len = None, [], 1.0, 0

    log_dir = os.path.join(HERE, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    run_name = datetime.now().strftime("live_%Y%m%d_%H%M%S")
    logging.basicConfig(filename=os.path.join(log_dir, run_name + ".log"),
                        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logger = logging.getLogger()
    logger.info("camera %s, model %s", args.camera, args.model)

    cap = open_camera(args.camera)
    if not cap.isOpened():
        raise SystemExit(f"Could not open camera {args.camera}")
    detect = Detector()
    writer = None
    fps_t, fps_n, fps = time.monotonic(), 0, 0.0

    buf = deque()
    probs = np.zeros(len(labels))
    last_fired, last_fired_at = None, 0.0
    start = time.monotonic()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        now = time.monotonic()
        ts = int((now - start) * 1000)
        pose, hands = detect(frame)
        h, w = frame.shape[:2]
        buf.append(make_row({}, len(buf), ts, w, h, pose, hands))
        while buf and ts - buf[0]["t_ms"] > window_s * 1000:
            buf.popleft()

        command = None
        if model and len(buf) > 2 and ts - buf[0]["t_ms"] >= 0.9 * window_s * 1000:
            x = (clip_features(pd.DataFrame(list(buf), columns=COLUMNS), seq_len) - mean) / std
            with torch.no_grad():
                p = torch.softmax(model(torch.tensor(x[None], dtype=torch.float32)), 1)[0].numpy()
            probs = 0.5 * probs + 0.5 * p  # smooth over frames
            top = int(probs.argmax())
            if labels[top] != "none" and probs[top] >= args.threshold:
                command = labels[top]

        if command and (command != last_fired or now - last_fired_at > COOLDOWN_S):
            logger.info("command %s (p=%.2f)", command, probs.max())
            print("command:", command)
            last_fired, last_fired_at = command, now
            if command != "stop":
                buf.clear()  # don't re-detect the same motion
                probs[:] = 0

        disp = frame.copy()
        draw_landmarks(disp, pose, hands)
        disp = cv2.flip(disp, 1)
        for i, lb in enumerate(labels):
            y = 30 + 28 * i
            cv2.rectangle(disp, (130, y - 16), (130 + int(200 * probs[i]), y + 4), (0, 200, 0), -1)
            cv2.putText(disp, lb, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        row = buf[-1] if buf else {}
        builtin = [f"{s.upper()}: {row[f'{s}hand_gesture']} {row[f'{s}hand_score']:.2f}"
                   for s in ("l", "r") if f"{s}hand_gesture" in row]
        status = f"body {'OK' if pose.pose_landmarks else '--'}  " + ("  ".join(builtin) or "no hand")
        cv2.putText(disp, status, (10, 30 + 28 * len(labels)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        if now - last_fired_at < 1.0 and last_fired:
            cv2.putText(disp, last_fired.upper(), (10, disp.shape[0] - 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.6, (0, 255, 255), 4)
        fps_n += 1
        if now - fps_t >= 1.0:
            fps, fps_t, fps_n = fps_n / (now - fps_t), now, 0
        cv2.putText(disp, f"{fps:.0f} fps", (disp.shape[1] - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if args.save:
            if writer is None:
                path = os.path.join(log_dir, run_name + ".mp4")
                writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), 15,
                                         (disp.shape[1], disp.shape[0]))
                print("saving to", path)
            writer.write(disp)
        cv2.imshow("live gestures (q to quit)", disp)
        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
            break

    if writer:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
