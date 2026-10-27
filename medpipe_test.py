# Test file for MediaPipe gesture recognition (Tasks API).
# Needs gesture_recognizer.task in the same folder (see README / setup notes).

import os
import time
import logging
import cv2
import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision
from datetime import datetime

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "gesture_recognizer.task")

HAND_CONNECTIONS = vision.HandLandmarksConnections.HAND_CONNECTIONS


def draw_hand(frame, landmarks):
    h, w = frame.shape[:2]
    pts = [(int(lm.x * w), int(lm.y * h)) for lm in landmarks]
    for c in HAND_CONNECTIONS:
        cv2.line(frame, pts[c.start], pts[c.end], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def main():
    if not os.path.exists(MODEL_PATH):
        raise SystemExit(f"Missing model file: {MODEL_PATH}")

    options = vision.GestureRecognizerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
    )


    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "run_logs")
    os.makedirs(log_dir, exist_ok=True)

    log_file = os.path.join(
        log_dir,
        datetime.now().strftime("run_%Y%m%d_%H%M%S.log"),
    )

    logging.basicConfig(
        filename=log_file,
        level=logging.DEBUG,
        format='%(asctime)s - %(levelname)s - %(message)s',
    )
    logger = logging.getLogger()
    logger.info("starting our run!")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logger.error("couldn't access camera")
        raise SystemExit("Could not open camera 0")

    start = time.monotonic()
    with vision.GestureRecognizer.create_from_options(options) as recognizer:
        last_gesture = None
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame = cv2.flip(frame, 1)  # mirror so it feels natural
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            ts_ms = int((time.monotonic() - start) * 1000)

            result = recognizer.recognize_for_video(image, ts_ms)

            for i, landmarks in enumerate(result.hand_landmarks):
                draw_hand(frame, landmarks)
                if result.gestures[i]:
                    g = result.gestures[i][0]
                    label = f"{g.category_name} {g.score:.3f}"
                    x = int(landmarks[0].x * frame.shape[1]) # go from a normalized location to actual pixel value
                    y = int(landmarks[0].y * frame.shape[0])
                    cv2.putText(frame, label, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.8, (255, 255, 0), 2)
                
                if g.score > 0.8 and  g.category_name != last_gesture:
                    logger.info("curt %s != last: %s", g.category_name, last_gesture )
                    last_gesture = g.category_name 
                    logger.info("found item, %s above 0.8 confidence", g.category_name )
                    #logger.debug("gestures %s", result.gestures[i])
            cv2.imshow("gestures (q to quit)", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    logger.info("run end")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
