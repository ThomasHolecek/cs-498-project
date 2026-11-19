"""
head_track.py  -  Booster K1 head tracking pipeline

Loop:  frame  ->  MediaPipe Pose (all people)  ->  pick the locked person
       ->  YOUR controller turns pixel error into head yaw/pitch  ->  send to head

The same file runs in three places. Pick with --source and --head.

1. Laptop, webcam, no robot (development):
    python head_track.py --source webcam --head print

2. ON THE ROBOT, headless (the main path; nobody on the team has Linux so the Booster
   SDK only exists on the robot). It subscribes to the ROS2 camera topic, runs MediaPipe
   and the controller locally, and calls the SDK in-process. You watch and control it
   from a laptop browser:
    robot:   source /opt/ros/humble/setup.bash
             python3 head_track.py --source ros2 --topic <color topic> --head booster --iface eth0 --no-window
    laptop:  open http://<robot-ip>:8082        (live overlay + Lock / Release / Center buttons)
             or curl http://<robot-ip>:8082/lock

3. Laptop brain, robot only does the SDK call (fallback if MediaPipe won't install on the robot):
    robot:   python3 k1_camera_stream.py --topic <color topic>
             python3 k1_head_server.py --iface eth0
    laptop:  python head_track.py --source url --url http://<robot-ip>:8080/stream.mjpg \
                                  --head http --head-url http://<robot-ip>:8081

Keys while the video window is focused (when not --no-window):
    l   lock onto the person nearest the frame center   (stand-in for "gesture fired")
    r   release the lock
    c   send the head back to center
    q   quit

HTTP control (always on unless --no-control):  /lock  /release  /center  /status  /stream.mjpg  /
"""

import argparse
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse
from urllib.request import urlopen

import cv2
import numpy as np

# k1_camera_stream.py lives one folder up (repo root). We reuse its image conversion.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# ============================================================================
# Config. Verify every number marked VERIFY at the robot tomorrow.
# ============================================================================

# Head joint limits in degrees. Conservative on purpose until you read the real
# ones off the robot or the K1 manual.                                    VERIFY
YAW_LIMIT_DEG = 45.0
PITCH_MIN_DEG = -20.0
PITCH_MAX_DEG = 20.0

# Sign conventions. If the head turns the wrong way, flip the sign here.    VERIFY
# +1 means: target is to the right in the image  ->  yaw command goes positive
YAW_SIGN = 1.0
# +1 means: target is below image center         ->  pitch command goes positive
PITCH_SIGN = 1.0

# Does RotateHead take radians or degrees?  Default assumption: radians.   VERIFY
HEAD_UNITS = "radians"

# Pose landmark indices (MediaPipe Pose, 33 landmarks)
NOSE = 0
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

# Where the pose model file lives. See README.md for the download link.
DEFAULT_MODEL = "pose_landmarker_lite.task"


# ============================================================================
# Data passed between stages
# ============================================================================

@dataclass
class Person:
    """One detected person. All coordinates are normalized to [0, 1] of the frame."""
    cx: float               # target point x (shoulder midpoint)
    cy: float               # target point y
    x0: float               # bounding box of all visible landmarks
    y0: float
    x1: float
    y1: float
    landmarks: list = field(default_factory=list)   # raw MediaPipe landmarks, if you want them

    @property
    def box_center(self):
        return ((self.x0 + self.x1) / 2.0, (self.y0 + self.y1) / 2.0)

    @property
    def box_area(self):
        return max(0.0, self.x1 - self.x0) * max(0.0, self.y1 - self.y0)


# ============================================================================
# Frame sources.  Each one has  .read() -> (ok, frame_bgr)  and  .release()
# ============================================================================

class WebcamSource:
    def __init__(self, index=0, width=640, height=480):
        self.cap = cv2.VideoCapture(index)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open webcam index {index}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class UrlSource:
    """
    Reads an MJPEG stream over HTTP, e.g. the one served by k1_camera_stream.py
    running on the robot:   http://<robot-ip>:8080/stream.mjpg
    No ROS2 needed on the laptop. This is the path to use from a Mac.
    """
    def __init__(self, url):
        self.url = url
        self.cap = cv2.VideoCapture(url)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open stream {url}  (is k1_camera_stream.py running on the robot?)")
        print(f"[url] reading {url}")

    def read(self):
        return self.cap.read()

    def release(self):
        self.cap.release()


class Ros2ImageSource:
    """
    Subscribes to a ROS2 camera topic and always hands back the latest frame.
    This is the source to use ON the robot. Spins rclpy in a background thread so the
    main loop never blocks on ROS.

    Handles sensor_msgs/Image (any encoding k1_camera_stream.image_to_bgr knows, which
    includes the odd ones like nv12 and yuyv) and sensor_msgs/CompressedImage (JPEG).
    Uses the sensor-data QoS profile; most camera drivers publish best-effort and a
    default reliable subscriber silently receives nothing.
    """
    def __init__(self, topic, downscale_width=640):
        import rclpy
        from rclpy.node import Node
        from rclpy.qos import qos_profile_sensor_data
        from sensor_msgs.msg import CompressedImage, Image

        try:
            from k1_camera_stream import image_to_bgr
            self._to_bgr = image_to_bgr
        except ImportError:
            print("[ros2] k1_camera_stream.py not found next to this folder, using basic conversion")
            self._to_bgr = self._basic_to_bgr

        self._rclpy = rclpy
        rclpy.init()
        self._node = Node("k1_head_track")
        self._lock = threading.Lock()
        self._frame = None
        self._downscale = downscale_width
        self._warned = False

        # Work out whether the topic is raw or compressed from what ROS says it publishes.
        types = dict(self._node.get_topic_names_and_types())
        tnames = types.get(topic, [])
        compressed = any(t.endswith("CompressedImage") for t in tnames)
        if not tnames:
            print(f"[ros2] WARNING: {topic} not visible yet. Is the camera driver running? "
                  f"Try: python3 ../k1_camera_stream.py --list")
        msg_type = CompressedImage if compressed else Image
        self._compressed = compressed

        self._node.create_subscription(msg_type, topic, self._on_image, qos_profile_sensor_data)
        self._thread = threading.Thread(target=rclpy.spin, args=(self._node,), daemon=True)
        self._thread.start()
        print(f"[ros2] subscribed to {topic} ({'compressed' if compressed else 'raw'}), waiting for first frame...")

    @staticmethod
    def _basic_to_bgr(msg):
        buf = np.frombuffer(msg.data, dtype=np.uint8)
        enc = msg.encoding.lower()
        if enc in ("bgr8", "rgb8"):
            img = buf.reshape(msg.height, msg.step)[:, :msg.width * 3].reshape(msg.height, msg.width, 3)
            return img if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if enc in ("bgra8", "rgba8"):
            img = buf.reshape(msg.height, msg.step)[:, :msg.width * 4].reshape(msg.height, msg.width, 4)
            return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR)
        raise ValueError(f"unsupported image encoding: {msg.encoding}")

    def _on_image(self, msg):
        try:
            if self._compressed:
                frame = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                frame = self._to_bgr(msg)
        except Exception as e:                   # noqa: BLE001  a bad frame is dropped, never fatal
            if not self._warned:
                print(f"[ros2] could not convert frame: {e}")
                self._warned = True
            return
        if frame is None:
            return
        if self._downscale and frame.shape[1] > self._downscale:
            h = int(frame.shape[0] * self._downscale / frame.shape[1])
            frame = cv2.resize(frame, (self._downscale, h))
        with self._lock:
            self._frame = np.ascontiguousarray(frame)

    def read(self):
        with self._lock:
            if self._frame is None:
                return False, None
            return True, self._frame.copy()

    def release(self):
        try:
            self._node.destroy_node()
            self._rclpy.shutdown()
        except Exception:                        # noqa: BLE001
            pass


# ============================================================================
# Person detection with MediaPipe Pose (Tasks API, supports multiple people)
# ============================================================================

class PoseDetector:
    def __init__(self, model_path=DEFAULT_MODEL, max_people=4):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        self._mp = mp
        options = mp_vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_poses=max_people,
            min_pose_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.PoseLandmarker.create_from_options(options)
        self._t0 = time.monotonic()
        self._last_t_ms = -1
        self._dropped = 0

    @staticmethod
    def _sanitize(frame):
        """
        Make the frame exactly what MediaPipe wants: uint8, HxWx3, C-contiguous, non-empty.
        Returns None if it cannot be fixed, so the caller drops the frame.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return None
        if frame.ndim == 2:                       # grayscale -> 3 channel
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        if frame.ndim != 3 or frame.shape[2] != 3:
            return None
        if frame.shape[0] < 2 or frame.shape[1] < 2:
            return None
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(frame)

    def detect(self, frame_bgr):
        """
        Returns a list of Person, one per detected human.
        Returns an empty list if nobody is found OR if the frame was bad / MediaPipe threw.
        A bad frame is dropped, never fatal.
        """
        frame_bgr = self._sanitize(frame_bgr)
        if frame_bgr is None:
            self._dropped += 1
            return []

        t_ms = int((time.monotonic() - self._t0) * 1000)
        if t_ms <= self._last_t_ms:              # detect_for_video requires strictly increasing timestamps
            t_ms = self._last_t_ms + 1
        self._last_t_ms = t_ms

        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
            result = self._landmarker.detect_for_video(mp_image, t_ms)
        except Exception as e:                   # noqa: BLE001  (we want to survive anything here)
            self._dropped += 1
            if self._dropped % 30 == 1:          # don't spam the console
                print(f"[pose] dropped frame ({self._dropped} total): {type(e).__name__}: {e}")
            return []

        people = []
        for lms in result.pose_landmarks:
            visible = [lm for lm in lms if getattr(lm, "visibility", 1.0) > 0.5]
            if not visible:
                continue
            xs = [lm.x for lm in visible]
            ys = [lm.y for lm in visible]
            ls, rs = lms[LEFT_SHOULDER], lms[RIGHT_SHOULDER]
            people.append(Person(
                cx=(ls.x + rs.x) / 2.0,
                cy=(ls.y + rs.y) / 2.0,
                x0=max(0.0, min(xs)), y0=max(0.0, min(ys)),
                x1=min(1.0, max(xs)), y1=min(1.0, max(ys)),
                landmarks=lms,
            ))
        return people


# ============================================================================
# Head command sinks.  Each one has  .send(pitch_rad, yaw_rad)
# ============================================================================

def _clamp(v, lo, hi):
    return max(lo, min(hi, v))


def clamp_head(pitch_rad, yaw_rad):
    """Hard safety clamp applied to every command no matter what the controller says."""
    yaw_rad = _clamp(yaw_rad, -math.radians(YAW_LIMIT_DEG), math.radians(YAW_LIMIT_DEG))
    pitch_rad = _clamp(pitch_rad, math.radians(PITCH_MIN_DEG), math.radians(PITCH_MAX_DEG))
    return pitch_rad, yaw_rad

class _Throttle:
    """
    Shared by the sinks: only pass a command through if it actually changed
    (by more than ~0.05 deg) and no more often than max_hz. Keeps the terminal
    readable and avoids spamming the robot with identical RPCs.
    """
    def __init__(self, max_hz=20.0):
        self._min_dt = 1.0 / max_hz
        self._last_t = 0.0
        self._last = (None, None)

    def should_send(self, pitch_rad, yaw_rad):
        now = time.monotonic()
        if now - self._last_t < self._min_dt:
            return False
        lp, ly = self._last
        if lp is not None and abs(pitch_rad - lp) < 1e-3 and abs(yaw_rad - ly) < 1e-3:
            return False
        self._last_t = now
        self._last = (pitch_rad, yaw_rad)
        return True
    
class PrintHead:
    def __init__(self):
        self._throttle = _Throttle()

    def send(self, pitch_rad, yaw_rad):
        pitch_rad, yaw_rad = clamp_head(pitch_rad, yaw_rad)
        if self._throttle.should_send(pitch_rad, yaw_rad):
            print(f"head  pitch={math.degrees(pitch_rad):+6.1f} deg   yaw={math.degrees(yaw_rad):+6.1f} deg")


class BoosterHead:
    """
    Drives the real K1 head through the Booster SDK's B1LocoClient.RotateHead.
    Names below come from the official Python binding. If the import or a call
    fails, open the SDK's python example and match its names.               VERIFY
    """
    def __init__(self, network_interface):
        from booster_robotics_sdk_python import ChannelFactory, B1LocoClient
        ChannelFactory.Instance().Init(0, network_interface)
        self._client = B1LocoClient()
        self._client.Init()
        self._throttle = _Throttle()
        print(f"[booster] connected on {network_interface}")

    def send(self, pitch_rad, yaw_rad):
        pitch_rad, yaw_rad = clamp_head(pitch_rad, yaw_rad)
        if not self._throttle.should_send(pitch_rad, yaw_rad):
            return
        if HEAD_UNITS == "degrees":
            pitch_rad, yaw_rad = math.degrees(pitch_rad), math.degrees(yaw_rad)
        rc = self._client.RotateHead(pitch_rad, yaw_rad)
        if rc not in (0, None):
            print(f"[booster] RotateHead returned {rc}")


class HttpHead:
    """
    Fallback path: send angles to k1_head_server.py running on the robot, which makes
    the SDK call there. Lets a Mac/Windows laptop drive the head without the SDK.
    """
    def __init__(self, base_url):
        self._base = base_url.rstrip("/")
        self._throttle = _Throttle()
        with urlopen(f"{self._base}/", timeout=2) as r:
            print(f"[http-head] {r.read().decode().strip()}")

    def send(self, pitch_rad, yaw_rad):
        pitch_rad, yaw_rad = clamp_head(pitch_rad, yaw_rad)
        if not self._throttle.should_send(pitch_rad, yaw_rad):
            return
        with urlopen(f"{self._base}/head?pitch={pitch_rad:.4f}&yaw={yaw_rad:.4f}", timeout=0.5) as r:
            body = r.read().decode().strip()
            if body not in ("rc=0", "rc=None"):
                print(f"[http-head] {body}")


# ============================================================================
# Remote control + live view over HTTP. This is how you watch and drive the
# tracker when it runs headless on the robot. Open http://<robot-ip>:8082 in a
# browser: live overlay video with Lock / Release / Center buttons.
# ============================================================================

_VIEW_HTML = b"""<!doctype html><html><head><meta name=viewport content="width=device-width">
<title>K1 head track</title>
<style>body{margin:0;background:#111;color:#ddd;font:14px system-ui}
.bar{padding:8px;display:flex;gap:8px;align-items:center}
button{font:16px system-ui;padding:8px 16px}
img{display:block;width:100%;max-width:960px}
#s{margin-left:auto;font-family:monospace}</style></head><body>
<div class=bar>
<button onclick="go('lock')">Lock (l)</button>
<button onclick="go('release')">Release (r)</button>
<button onclick="go('center')">Center (c)</button>
<span id=s></span></div>
<img src="/stream.mjpg">
<script>
function go(c){fetch('/'+c).then(r=>r.text()).then(t=>document.getElementById('s').textContent=t)}
document.addEventListener('keydown',e=>{if(e.key=='l')go('lock');if(e.key=='r')go('release');if(e.key=='c')go('center')})
setInterval(()=>fetch('/status').then(r=>r.text()).then(t=>document.getElementById('s').textContent=t),1000)
</script></body></html>"""


class RemoteControl:
    """
    Tiny HTTP server. Commands are queued and consumed by the main loop (so all robot
    calls stay on one thread); the main loop pushes overlay JPEGs in for /stream.mjpg.
    """
    def __init__(self, port=8082, jpeg_quality=70):
        self._cmds = []
        self._cmd_lock = threading.Lock()
        self._cond = threading.Condition()
        self._jpg, self._seq = None, 0
        self._status = "starting"
        self._quality = jpeg_quality
        rc = self

        class Handler(BaseHTTPRequestHandler):
            def _text(self, code, body):
                data = body.encode()
                self.send_response(code)
                self.send_header("Content-Type", "text/plain")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html")
                    self.send_header("Content-Length", str(len(_VIEW_HTML)))
                    self.end_headers()
                    self.wfile.write(_VIEW_HTML)
                elif path in ("/lock", "/release", "/center", "/quit"):
                    with rc._cmd_lock:
                        rc._cmds.append(path[1:])
                    self._text(200, f"{path[1:]} queued\n")
                elif path == "/status":
                    self._text(200, rc._status + "\n")
                elif path == "/stream.mjpg":
                    self.send_response(200)
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.send_header("Cache-Control", "no-cache")
                    self.end_headers()
                    seq = -1
                    try:
                        while True:
                            with rc._cond:
                                rc._cond.wait_for(lambda: rc._seq != seq, timeout=5)
                                jpg, seq = rc._jpg, rc._seq
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                             + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        pass
                else:
                    self._text(404, "not found\n")

            def log_message(self, *a):
                pass

        self._server = ThreadingHTTPServer(("0.0.0.0", port), Handler)
        self._server.daemon_threads = True
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        print(f"[control] open http://<this-machine-ip>:{port}  in a browser")

    def pop_commands(self):
        with self._cmd_lock:
            cmds, self._cmds = self._cmds, []
        return cmds

    def publish_frame(self, frame_bgr, status):
        self._status = status
        ok, jpg = cv2.imencode(".jpg", frame_bgr, [cv2.IMWRITE_JPEG_QUALITY, self._quality])
        if ok:
            with self._cond:
                self._jpg, self._seq = jpg.tobytes(), self._seq + 1
                self._cond.notify_all()

    def close(self):
        self._server.shutdown()


# ============================================================================
# ======================  YOUR PART STARTS HERE  =============================
# ============================================================================

class HeadTracker:
    """
    Two jobs:
      1. choose_target: given everyone in the frame, decide which Person is the one
         we locked onto (or None if we lost them).
      2. update: given that Person (or None), decide the next head pitch and yaw.

    State you probably want lives in __init__. Add whatever you need.
    """

    def __init__(self):
        # Current head pose, in radians. Start looking straight ahead.
        self.pitch = 0.0
        self.yaw = 0.0

        # The last Person we believed was our target. None means no lock.
        self.last_target = None

        self.kp_yaw = 0.08        #  starting value of yaw
        self.kp_pitch = 0.06      # starting value of pitch
        self.dead_zone = 0.05     # fraction of the frame to ignore 
        self.max_step = 0.03      #  max radians the head may move per frame
        self.max_jump = 0.2       # max distance the target can move in one frame

    # ---------------------------------------------------------------- lock
    def lock_on(self, person):
        """Called when the gesture fires (or you press 'l'). person is who gestured."""
        self.last_target = person

    def release(self):
        self.last_target = None

    @property
    def locked(self):
        return self.last_target is not None

    # ---------------------------------------------------------------- target
    
    def choose_target(self, people):
        """
        people: list[Person] detected this frame (may be empty).
        Returns the Person that is our locked target this frame, or None.

        """
        def distance( person):
            last_x , last_y = self.last_target.box_center
            person_x, person_y = person.box_center
            return math.hypot(last_x - person_x, last_y - person_y)
        if self.last_target is None: #if there is no target
            return None
        if not people: #if there are no people in the frame
            return None
        nearest_person = min(people, key=distance) #find person in the people the robot sees who is closest to the last frame's target

        if distance(nearest_person) > self.max_jump: #if the nearest person is to far away from the last frame
            return None
        self.last_target = nearest_person #update the last target to the new person

        return nearest_person #return the new target
        

    # ---------------------------------------------------------------- control
    def update(self, target):
        """
        target: the Person to center on, or None.
        Returns (pitch_rad, yaw_rad) for the head this frame.

        TODO 
        Optional once it works: smooth target.cx/cy with a moving average before
        computing error, so a jittery detection does not shake the head.
        """
        if target is None:
            return self.pitch, self.yaw

        error_x = (target.cx - 0.5) * 2 #0.5 center of our frame
        error_y = (target.cy - 0.5) * 2

        if abs(error_x) < self.dead_zone: #robots head does not have to perfectly be centered on the target, ignores small errors making head movement more smooth
            error_x = 0.0
        if abs(error_y) < self.dead_zone:
            error_y = 0.0

        step_yaw = self.kp_yaw * error_x #how much the head is off from the center multiplied by the proportional gain to get the step size, the greater the error the larger the step size
        step_pitch = self.kp_pitch * error_y 

        step_yaw = _clamp(step_yaw, -self.max_step, self.max_step) # if the step size is too large (if the person was barely in frame) it wont overshoot and jerk the head around, it will just move the max step size
        step_pitch = _clamp(step_pitch, -self.max_step, self.max_step)

        self.yaw += step_yaw * YAW_SIGN # which way is up and down for the robot
        self.pitch += step_pitch * PITCH_SIGN

        self.pitch, self.yaw = clamp_head(self.pitch, self.yaw) # makes sure pitch and yaw are never outside of the robots limits
        return self.pitch, self.yaw

    def center(self):
        self.pitch, self.yaw = 0.0, 0.0
        return self.pitch, self.yaw


# ============================================================================
# ======================  YOUR PART ENDS HERE  ===============================
# ============================================================================


def nearest_to_center(people):
    """Stand-in for 'the person who gestured' until the gesture code is wired in."""
    if not people:
        return None
    return min(people, key=lambda p: (p.cx - 0.5) ** 2 + (p.cy - 0.5) ** 2)


def draw_overlay(frame, people, target, tracker, fps):
    h, w = frame.shape[:2]
    cv2.drawMarker(frame, (w // 2, h // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
    for p in people:
        color = (0, 255, 0) if p is target else (128, 128, 128)
        cv2.rectangle(frame, (int(p.x0 * w), int(p.y0 * h)), (int(p.x1 * w), int(p.y1 * h)), color, 2)
        cv2.circle(frame, (int(p.cx * w), int(p.cy * h)), 6, color, -1)
    status = "LOCKED" if tracker.locked else "no lock (press l)"
    if tracker.locked and target is None:
        status = "LOCKED - target lost"
    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.putText(frame,
                f"yaw {math.degrees(tracker.yaw):+5.1f}  pitch {math.degrees(tracker.pitch):+5.1f}  {fps:4.1f} fps",
                (10, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["webcam", "url", "ros2"], default="webcam")
    ap.add_argument("--cam", type=int, default=0, help="webcam index")
    ap.add_argument("--url", default="http://192.168.10.102:8080/stream.mjpg",
                    help="MJPEG stream URL from k1_camera_stream.py on the robot")
    ap.add_argument("--topic", default="/camera/color/image_raw",
                    help="ROS2 image topic (on the robot; list them with k1_camera_stream.py --list)")
    ap.add_argument("--head", choices=["print", "booster", "http"], default="print")
    ap.add_argument("--iface", default="eth0", help="network interface for the Booster SDK (--head booster)")
    ap.add_argument("--head-url", default="http://192.168.10.102:8081",
                    help="k1_head_server.py address (--head http)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--no-window", action="store_true", help="headless; use the browser page instead")
    ap.add_argument("--no-control", action="store_true", help="disable the HTTP control/view server")
    ap.add_argument("--control-port", type=int, default=8082)
    ap.add_argument("--auto-lock", type=float, default=0.0,
                    help="seconds after start to lock onto whoever is nearest center (0 = off). Handy for solo tests.")
    args = ap.parse_args()

    if args.source == "webcam":
        source = WebcamSource(args.cam)
    elif args.source == "url":
        source = UrlSource(args.url)
    else:
        source = Ros2ImageSource(args.topic)

    if args.head == "print":
        head = PrintHead()
    elif args.head == "booster":
        head = BoosterHead(args.iface)
    else:
        head = HttpHead(args.head_url)

    detector = PoseDetector(args.model)
    tracker = HeadTracker()
    control = None if args.no_control else RemoteControl(args.control_port)
    if args.no_window and control is None:
        print("[warn] --no-window with --no-control: no way to lock. Use --auto-lock or drop --no-control.")

    def do(cmd, people):
        """One place for every command, whether it came from a key or from HTTP."""
        if cmd == "lock":
            p = nearest_to_center(people)
            if p is not None:
                tracker.lock_on(p)
                print("[lock] locked on")
            else:
                print("[lock] nobody in frame to lock onto")
        elif cmd == "release":
            tracker.release()
            print("[lock] released")
        elif cmd == "center":
            head.send(*tracker.center())
        return cmd == "quit"

    t_start = time.monotonic()
    auto_locked = False
    last_t = time.monotonic()
    fps = 0.0
    try:
        while True:
            ok, frame = source.read()
            if not ok:
                time.sleep(0.01)
                continue

            people = detector.detect(frame)

            # Commands from HTTP (browser buttons / curl), then auto-lock, then keyboard below.
            quit_now = False
            if control is not None:
                for cmd in control.pop_commands():
                    quit_now |= do(cmd, people)
            if args.auto_lock and not auto_locked and time.monotonic() - t_start >= args.auto_lock:
                auto_locked = True
                do("lock", people)

            target = tracker.choose_target(people) if tracker.locked else None
            pitch, yaw = tracker.update(target)
            try:
                head.send(pitch, yaw)
            except Exception as e:               # noqa: BLE001  (a flaky SDK call should not kill the loop)
                print(f"[head] send failed: {type(e).__name__}: {e}")

            now = time.monotonic()
            fps = 0.9 * fps + 0.1 * (1.0 / max(1e-6, now - last_t))
            last_t = now

            draw_overlay(frame, people, target, tracker, fps)
            if control is not None:
                lock_txt = "LOCKED" if tracker.locked else "no lock"
                if tracker.locked and target is None:
                    lock_txt = "LOCKED (lost)"
                control.publish_frame(frame, f"{lock_txt}  yaw {math.degrees(tracker.yaw):+.1f}  "
                                             f"pitch {math.degrees(tracker.pitch):+.1f}  {fps:.0f} fps  "
                                             f"{len(people)} people")

            if not args.no_window:
                cv2.imshow("K1 head track", frame)
                key = cv2.waitKey(1) & 0xFF
                keymap = {ord("l"): "lock", ord("r"): "release", ord("c"): "center", ord("q"): "quit"}
                if key in keymap:
                    quit_now |= do(keymap[key], people)
            if quit_now:
                break
    finally:
        try:
            head.send(*tracker.center())
        except Exception:                        # noqa: BLE001
            pass
        source.release()
        if control is not None:
            control.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()