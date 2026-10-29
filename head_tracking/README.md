# K1 head tracking

## Setup (laptop, tonight)

```bash
pip install opencv-python mediapipe numpy
pip install booster_robotics_sdk_python --user      # only needed for the robot, Ubuntu 22.04

# MediaPipe pose model (multi-person capable)
wget -O pose_landmarker_lite.task \
  https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task
```

Test the webcam half:

```bash
python head_track.py --source webcam --head print
```

It will crash with `NotImplementedError` the first time you press `l`. That is the point. The two TODO methods in `HeadTracker` are yours.

## Tomorrow at the robot (1 hour)

| When | Do | Done when |
|---|---|---|
| 0:00 | Plug in, find the interface name (`ip a`), `python head_sweep.py --iface <name>` | Head moves. Write down units, +yaw dir, +pitch dir. Fix `YAW_SIGN`, `PITCH_SIGN`, `HEAD_UNITS`, limits in `head_track.py` |
| 0:15 | `ros2 topic list`, find the color image topic, `ros2 topic hz <topic>` | Know the topic name and frame rate |
| 0:25 | `python head_track.py --source ros2 --topic <topic> --head print` | Window shows robot camera with people boxed |
| 0:35 | Same but `--head booster --iface <name>`, low gains, robot standing still | Head follows you when you press `l` |
| 0:45 | Tune `kp_*`, `dead_zone`, `max_step`. Try walking across the room | Smooth-ish tracking |
| 0:55 | Dump camera topic, image size, fps, signs, final gains into `notes.md` | |

Fallback if the camera topic is a fight: keep `--source webcam` (laptop pointed at the room) with `--head booster`. Still proves the controller and the SDK call.

## Handoff from gesture code

When the gesture fires, call `tracker.lock_on(person)` with the `Person` who gestured. Until that exists, the `l` key locks onto whoever is nearest the frame center.
