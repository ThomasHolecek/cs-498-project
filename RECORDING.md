# Gesture Dataset Recorder

`record_gestures.py` records our gesture dataset for the K1 follow-me project. It uses MediaPipe to turn camera frames into body and hand points, then saves short labeled clips of four gestures:

| Label | Meaning for the robot |
|---|---|
| `come_here` | start following me |
| `shoo` | let go of me and back off |
| `stop` | stop moving |
| `none` | normal movement, not a command |

Each recording session saves:
- `data/<person>_<camera>_<time>.csv`: one row per frame (landmarks only, small). **Commit these.**
- `clips/<person>_<camera>_<time>/`: the raw video clips. **Don't commit these.** Upload them to the shared drive.

---

## 1. Setup (once per laptop)

### Install the packages

**Mac**
```
python3 -m pip install mediapipe opencv-python pandas
```

**Windows**
```
python -m pip install mediapipe opencv-python pandas
```

Use the same command (`python3` on Mac, `python` on Windows) for installing and for running the script. That way the packages go to the Python you actually run. Any Python 3.9 or newer works.

### Download the two model files

Run these in this folder:

**Mac**
```
curl -L -o gesture_recognizer.task https://storage.googleapis.com/mediapipe-models/gesture_recognizer/gesture_recognizer/float16/latest/gesture_recognizer.task
curl -L -o pose_landmarker_full.task https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task
```

**Windows (PowerShell):** same commands, but type `curl.exe` instead of `curl`.

The `.task` files are in `.gitignore`, so everyone downloads their own.

### Do a test run

```
python3 record_gestures.py record --person test --camera laptop
```

- **Mac:** the first time, macOS asks if Terminal can use the camera. Click **Allow**, then run the command again.
- You should see a window with your camera feed and dots on your body and hands.
- Press **SPACE**, record a clip or two, then press **q**.
- Delete the test files: `data/test_laptop_*.csv` and `clips/test_laptop_*/`.

---

## 2. Set up the room

- Put the laptop on a chair or low table so the camera is at about the K1's head height.
- Put tape on the floor at **1, 2, 3 and 4 m**, measured from the camera.
- Only the performer should be in frame.
- At 1 m it's fine if your legs are cut off. Your head and both arms need to be visible.

---

## 3. The gestures

Make every gesture **arm-sized**, because the camera can't see fingers at 3–4 m.

| Gesture | How to do it |
|---|---|
| `come_here` | Arm out toward the camera, then sweep your forearm back toward your chest. Do it 2–3 times. |
| `shoo` | Hand near your body, then flick your forearm outward, away from you. Do it 2–3 times. |
| `stop` | Arm up, palm facing the camera. Hold still. |
| `none` | Anything that isn't a command: wave hi, scratch your head, check your phone, stretch, point, walk past. |

**Vary each clip.** The classifier only learns versions it has seen. Change the speed, size, arm height and hand from clip to clip. Pattern for the come_here / shoo clips at each spot:

1. Right hand, normal
2. Left hand
3. Fast and small
4. Slow and big

The screen now tells the performer which hand to use for every clip (the performer's own hand; the preview is mirrored like a mirror). The plan lives in `VARIANTS` in `record_gestures.py` and the hand is saved in the CSV's `hand` column.

**Keep the core of each gesture the same:**
- come_here always moves *toward* you
- shoo always moves *away* from you
- stop is always palm out and still

---

## 4. Record a session (about 15–20 min per person)

Work in pairs. One person **performs**, the other **operates** the keyboard. Rotate so everyone performs once.

The operator runs this, using the performer's name:
```
python3 record_gestures.py record --person joe --camera laptop
```

Then press **f** for fullscreen so the performer can see the screen from 4 m.

### What the screen means

| Screen | Meaning |
|---|---|
| Dark blue bar, `not recording` | Waiting. It shows where to stand, which hand to use and the next gesture. The operator presses **SPACE**. |
| Orange bar and a big countdown | Get ready. The first countdown of each step is 7 s, so you have time to walk to your mark. |
| Red bar, red frame around the whole window, blinking **REC**, big **GO** | The camera is recording. Do the gesture **now**. You have 3 seconds (3.5 s for `stop`, 4.5 s for the slow-and-big clips). The banner shows the time for each clip. |

The footer shows green or red dots for whether the body, left hand and right hand are being detected. Press **h** for the key list.

After GO, the next clip's countdown starts automatically until the step is done. Then it goes back to the blue waiting screen for the next step.

### The order

At 1 m, facing the camera: come_here, shoo, stop, none. Then the same at 1 m turned left (body rotated about 45°), then turned right. Then all of that again at 2, 3 and 4 m.

The number of clips per gesture is set by `CLIPS` near the top of `record_gestures.py`. By default it's 4 come_here, 4 shoo, 1 stop and 2 none at each spot. That makes 48 steps and 132 clips per session.

### Keys

| Key | Action |
|---|---|
| **SPACE** | Start the current step |
| **r** | Redo the last clip. Press again to go back further. |
| **n** | Skip this step |
| **f** | Toggle fullscreen |
| **Clip lengths** button (click it) | Opens a panel with a **-** / **+** for each kind of clip (normal speed, slow and big, hold still...). Click the button again to close it. |
| **-** / **+** keys | Same change from the keyboard, for the style shown on screen. Each step is 0.5 s (range 1�10 s) and lasts for the rest of the session. |
| **h** | Show / hide the key list |
| **b** | Go back one step (while waiting) |
| **p** | Pause / resume |
| **m** | Flip the mirrored preview |
| **s** | Sound on/off (beep at GO, high/low beep when a clip is good/poor) |
| **q** | Quit. Everything recorded so far is already saved. |

### Operator checklist

- Watch the bottom line of the screen. **body OK** should show at every distance. **hand --** at 3–4 m is normal.
- After each clip the red screen shows `last clip: body 98%, hand 70%`. If body is under 80% it says POOR and beeps low: press **r**.
- A `WARNING:` line means your head or a wrist is cut off by the frame edge. Fix it before pressing SPACE.
- Wrong webcam? `python record_gestures.py cameras` lists them; use `--source N`. Use `--size 1280x720` to request a higher resolution.
- Press **r** if a clip went wrong: started late, wrong gesture, or someone walked into frame.

---

## 5. Fixing mistakes

- **Bad clip during a session:** press **r**.
- **Bad run afterward:** delete both `data/<run name>.csv` and `clips/<run name>/`.
- **Continue a past session and replace rejected clips:** reject the bad clips in `review_clips.py` (**X**), then run
  ```
  python record_gestures.py record --resume <run name>
  ```
  It adds to that same session (same CSV and `clips/` folder) and records only what's missing: the steps you never got to, the rest of a step you stopped partway through, and a replacement for every rejected clip, with the hand and style that clip needed. The rejected clips stay on disk but are ignored by training. Quitting early prints the same command to continue again. Review again afterward to trim the new clips.
- **Stopped partway:** the terminal prints the step to resume from. Run the same command with `--start <step>`. The resumed part saves as its own run, which is fine.

---

## 6. After each session

1. Check what you have:
   ```
   python3 record_gestures.py stats
   ```
   This shows the clip counts per person and gesture, and how often a body or hand was detected at each distance. If body detection is low at some distance, fix the lighting or camera height before the next person records.
2. Review and trim the clips:
   ```
   python3 review_clips.py --session <run name>
   ```
   This plays each clip back with the landmarks drawn on it. Trim each clip to just the gesture, cutting the standing still before and after. Training then only learns from the gesture itself. The originals are never changed: trims are saved to `data/trims/<run name>.json`, so you can redo them any time.

   | Key | Action |
   |---|---|
   | **SPACE** | Play / pause (loops over the trimmed part) |
   | **A / D** or arrows | Step one frame back / forward |
   | **[** and **]** | Set the gesture's start / end at the current frame |
   | **T** | Auto-trim to where the arm is moving (check it with SPACE) |
   | **C** | Clear the trim |
   | **X** | Reject the clip (wrong gesture, someone walked in, started late) |
   | **1–4** | Relabel: come_here, shoo, stop, none |
   | **N / ENTER** | Accept and go to the next clip. **P** goes back. |
   | **Q** | Save and quit. Next time it starts at the first clip you haven't reviewed. |

   Click or drag on the timeline to scrub. The yellow line shows how fast the arm is moving, so the gesture is usually where it's high. Don't trim `stop` or `none` clips unless something is wrong with them.

   The video only shows on the computer that has the session's `clips/` folder. Without it, the reviewer draws the saved landmarks on a blank background, which is still enough to trim.
3. Commit the data and the trims:
   ```
   git add data/*.csv data/trims/
   git commit -m "Add <name> laptop session"
   git push
   ```
4. Upload that session's `clips/` folder to the shared drive.

**Test set:** one teammate's data is held out and never used for training, so our accuracy number is measured on someone the model hasn't seen. Decide who before training.

---

## 7. Recording on the K1 camera

Same process, with these changes:

1. Keep the robot still and in a safe, non-walking mode, with its head facing straight ahead.
2. Measure the tape marks from the **robot's head**.
3. Get the camera to the laptop, using one of these:
   - **No ROS on the laptop (easiest, works on Windows/Mac).** Copy `k1_camera_stream.py` to the robot and stream the camera over HTTP:
     ```
     scp k1_camera_stream.py booster@<robot-ip>:~/          # on the laptop
     source /opt/ros/humble/setup.bash                       # on the robot (ssh booster@<robot-ip>)
     python3 k1_camera_stream.py --list                      # find the camera topic
     python3 k1_camera_stream.py --topic <camera topic>
     ```
     Check it by opening `http://<robot-ip>:8080` in a browser. Then record with:
     ```
     python3 record_gestures.py record --person joe --camera k1 --source http://<robot-ip>:8080/stream.mjpg
     ```
   - **Laptop with ROS2** that can see the robot's topics. Find the camera topic and check that it's publishing:
     ```
     ros2 topic list
     ros2 topic hz <camera topic>
     ```
     Then record with:
     ```
     python3 record_gestures.py record --person joe --camera k1 --ros-topic <camera topic>
     ```

The default wired robot IP is `192.168.10.102` (set the laptop's Ethernet to `192.168.10.10`, mask `255.255.255.0`). The K1 options haven't been tested on the robot yet, so do a short test run first.

---

## 8. Training and testing the classifier

Once there's data in `data/`:
```
python3 train_gestures.py --val-person <held-out teammate>
```
This trains on random 1 s crops of every clip, prints accuracy and a confusion matrix for the held-out person, and saves `models/gesture_gru.pt`.

Try it live on the laptop webcam or the K1 stream (`--save` writes the annotated video to `run_logs/`):
```
python3 live_gestures.py
python3 live_gestures.py --camera http://<robot-ip>:8080/stream.mjpg --save
```
Without a trained model, `live_gestures.py` still shows the landmarks and MediaPipe's built-in gestures, which is handy for checking detection on the K1 camera. Training also needs `pip install torch scikit-learn`.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `command not found: python` | On Mac use `python3`. |
| `No module named 'cv2'` | The packages went to a different Python. Rerun the install with the same command you run the script with (`python3 -m pip install ...`). |
| `not authorized to capture video` / black window | Mac: System Settings → Privacy & Security → Camera → turn on Terminal, then restart Terminal. |
| `Missing gesture_recognizer.task` | Download the model files (step 1) into this folder. |
| Lots of `I0000` / `W0000` / `INFO` lines at startup | Normal MediaPipe logging. Ignore them. |
| It quits right away | Get the latest `record_gestures.py` from the repo. If it still happens, share the terminal output. |

---

## Repo setup (already done once)

`.gitignore` keeps the raw clips, model downloads, trained models and run logs out of the repo:
```
clips/
*.task
models/
run_logs/
__pycache__/
*.log
```
