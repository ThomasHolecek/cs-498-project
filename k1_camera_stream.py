#!/usr/bin/env python3
# Runs ON the Booster K1. Subscribes to a ROS 2 camera topic and serves it as an
# MJPEG stream over HTTP, so the laptop scripts can use it as a camera:
#
#   robot:   source /opt/ros/humble/setup.bash
#            python3 k1_camera_stream.py --list                # show image topics
#            python3 k1_camera_stream.py --topic <image topic>
#   laptop:  open http://<robot-ip>:8080 in a browser to check the view, then
#            python live_gestures.py --camera http://<robot-ip>:8080/stream.mjpg
#
# Handles sensor_msgs/Image (bgr8, rgb8, mono8, nv12, yuyv/yuv422) and
# sensor_msgs/CompressedImage (forwarded as-is if JPEG).

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2
import numpy as np

IMAGE_TYPES = ("sensor_msgs/msg/Image", "sensor_msgs/msg/CompressedImage")

_latest = {"jpg": None, "seq": 0}
_cond = threading.Condition()


def publish_jpeg(jpg):
    with _cond:
        _latest["jpg"] = jpg
        _latest["seq"] += 1
        _cond.notify_all()


def image_to_bgr(msg):
    enc = msg.encoding.lower()
    h, w = msg.height, msg.width
    buf = np.frombuffer(msg.data, np.uint8)
    if enc in ("bgr8", "rgb8"):
        img = buf.reshape(h, msg.step)[:, :w * 3].reshape(h, w, 3)
        return img if enc == "bgr8" else cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    if enc in ("bgra8", "rgba8"):
        img = buf.reshape(h, msg.step)[:, :w * 4].reshape(h, w, 4)
        return cv2.cvtColor(img, cv2.COLOR_BGRA2BGR if enc == "bgra8" else cv2.COLOR_RGBA2BGR)
    if enc == "mono8":
        return cv2.cvtColor(buf.reshape(h, msg.step)[:, :w], cv2.COLOR_GRAY2BGR)
    if enc == "nv12":
        return cv2.cvtColor(buf[:w * h * 3 // 2].reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if enc in ("yuyv", "yuv422_yuy2", "yuv422"):
        img = buf.reshape(h, msg.step)[:, :w * 2].reshape(h, w, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_YUY2)
    if enc == "uyvy":
        img = buf.reshape(h, msg.step)[:, :w * 2].reshape(h, w, 2)
        return cv2.cvtColor(img, cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"unsupported image encoding: {msg.encoding}")


class StreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = b"<html><body style='margin:0;background:#000'><img src='/stream.mjpg' style='width:100%'></body></html>"
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path != "/stream.mjpg":
            self.send_error(404)
            return
        self.send_response(200)
        self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        seq = -1
        try:
            while True:
                with _cond:
                    _cond.wait_for(lambda: _latest["seq"] != seq, timeout=5)
                    jpg, seq = _latest["jpg"], _latest["seq"]
                if jpg is None:
                    continue
                self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n"
                                 + f"Content-Length: {len(jpg)}\r\n\r\n".encode() + jpg + b"\r\n")
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            pass

    def log_message(self, *args):
        pass


def serve(port):
    server = ThreadingHTTPServer(("0.0.0.0", port), StreamHandler)
    server.daemon_threads = True
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def image_topics(node, wait_s=3.0):
    end = time.time() + wait_s
    found = []
    while time.time() < end:  # discovery takes a moment
        found = [(n, t[0]) for n, t in node.get_topic_names_and_types() if t and t[0] in IMAGE_TYPES]
        if found:
            time.sleep(0.5)
            found = [(n, t[0]) for n, t in node.get_topic_names_and_types() if t and t[0] in IMAGE_TYPES]
            break
        time.sleep(0.2)
    return sorted(found)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--topic", help="ROS 2 image topic (default: first one that looks like a colour/left image)")
    ap.add_argument("--list", action="store_true", help="list image topics and exit")
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--width", type=int, default=640, help="downscale to this width (0 = keep)")
    ap.add_argument("--fps", type=float, default=15, help="max frames per second to stream")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality")
    args = ap.parse_args()

    import rclpy
    from rclpy.node import Node
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import CompressedImage, Image

    rclpy.init()
    node = Node("k1_camera_stream")
    topics = image_topics(node)
    if args.list or not topics:
        print("image topics:" if topics else "no image topics found - is the camera bridge running?")
        for n, t in topics:
            print(f"  {n}  [{t}]")
        node.destroy_node()
        rclpy.shutdown()
        return

    topic = args.topic
    if topic is None:
        prefer = [n for n, _ in topics if any(k in n.lower() for k in ("color", "rgb", "left"))
                  and "depth" not in n.lower()]
        topic = (prefer or [topics[0][0]])[0]
    types = dict(topics)
    if topic not in types:
        raise SystemExit(f"{topic} is not an image topic. Found: {[n for n, _ in topics]}")
    compressed = types[topic].endswith("CompressedImage")

    min_dt = 1.0 / args.fps
    state = {"last": 0.0, "n": 0, "warned": False}
    encode = [cv2.IMWRITE_JPEG_QUALITY, args.quality]

    def on_image(msg):
        now = time.time()
        if now - state["last"] < min_dt:
            return
        state["last"] = now
        try:
            if compressed:
                fmt = msg.format.lower()
                if "jpeg" in fmt or "jpg" in fmt:
                    if not args.width:
                        publish_jpeg(bytes(msg.data))
                        state["n"] += 1
                        return
                img = cv2.imdecode(np.frombuffer(msg.data, np.uint8), cv2.IMREAD_COLOR)
            else:
                img = image_to_bgr(msg)
        except ValueError as e:
            if not state["warned"]:
                print(e)
                state["warned"] = True
            return
        if args.width and img.shape[1] > args.width:
            img = cv2.resize(img, (args.width, int(img.shape[0] * args.width / img.shape[1])))
        ok, jpg = cv2.imencode(".jpg", img, encode)
        if ok:
            publish_jpeg(jpg.tobytes())
            state["n"] += 1

    node.create_subscription(CompressedImage if compressed else Image, topic, on_image, qos_profile_sensor_data)
    serve(args.port)
    print(f"streaming {topic} on http://0.0.0.0:{args.port}/stream.mjpg  (Ctrl+C to stop)")

    last_report = time.time()
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.5)
            if time.time() - last_report >= 5:
                print(f"  {state['n'] / (time.time() - last_report):.1f} fps streamed")
                state["n"], last_report = 0, time.time()
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()
