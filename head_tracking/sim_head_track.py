"""
sim_head_track.py - closed-loop simulation of HeadTracker, no camera, MediaPipe or robot.

World: people move at angles (azimuth right-positive, elevation down-positive, degrees)
relative to the robot body. The camera sits on the head, so where a person lands in
the image depends on the head pose the controller commanded:

    cx = 0.5 + (azimuth   - head_yaw)   / HFOV
    cy = 0.5 + (elevation - head_pitch) / VFOV

That matches the sign convention in head_track.py (YAW_SIGN = PITCH_SIGN = +1).

Run:
    python sim_head_track.py             # run all scenarios, print pass/fail
    python sim_head_track.py --plot      # also save sim_results.png (needs matplotlib)
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import head_track as ht  # noqa: E402

HFOV = 90.0     # degrees, camera horizontal field of view
VFOV = 65.0     # degrees
FPS = 30.0
LATENCY = 2     # frames between command and what the camera shows


def make_person(az, el, half_w=0.07, half_h=0.25, head_yaw=0.0, head_pitch=0.0):
    """Project a world-space person into a normalized image-space Person (or None if off screen)."""
    cx = 0.5 + (az - math.degrees(head_yaw)) / HFOV
    cy = 0.5 + (el - math.degrees(head_pitch)) / VFOV
    if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
        return None
    return ht.Person(cx=cx, cy=cy,
                     x0=max(0.0, cx - half_w), y0=max(0.0, cy - half_h),
                     x1=min(1.0, cx + half_w), y1=min(1.0, cy + half_h))


def run(name, paths, seconds, lock_at=0.0, target_idx=0, release_at=None, dark=None):
    """
    paths: list of f(t) -> (az_deg, el_deg), one per person. paths[target_idx] is the one we lock.
    dark: optional (t0, t1) window where detection returns nobody (occlusion / lost frames).
    Returns a dict of metrics and a trace.
    """
    tracker = ht.HeadTracker()
    n = int(seconds * FPS)
    pose_hist = [(tracker.pitch, tracker.yaw)] * (LATENCY + 1)   # camera sees pose from LATENCY frames ago
    trace, locked = [], False
    for i in range(n):
        t = i / FPS
        cam_pitch, cam_yaw = pose_hist[0]
        people, tgt_obj = [], None
        for k, f in enumerate(paths):
            az, el = f(t)
            p = make_person(az, el, head_yaw=cam_yaw, head_pitch=cam_pitch)
            if p is not None and not (dark and dark[0] <= t < dark[1]):
                people.append(p)
                if k == target_idx:
                    tgt_obj = p
        if not locked and t >= lock_at and tgt_obj is not None:
            tracker.lock_on(tgt_obj)
            locked = True
        if release_at is not None and t >= release_at:
            tracker.release()

        target = tracker.choose_target(people) if tracker.locked else None
        pitch, yaw = tracker.update(target)
        pose_hist = pose_hist[1:] + [(pitch, yaw)]

        az, el = paths[target_idx](t)
        err_az = az - math.degrees(yaw)
        err_el = el - math.degrees(pitch)
        trace.append((t, az, math.degrees(yaw), err_az, el, math.degrees(pitch), err_el,
                      target is not None, tgt_obj is not None,
                      [f(t) for f in paths]))
    return {"name": name, "trace": trace, "locked": locked}


def metrics(res, settle_after=0.0):
    tr = [r for r in res["trace"] if r[0] >= settle_after]
    errs = [abs(r[3]) for r in tr if r[8]]
    in_view = sum(1 for r in tr if r[8]) / max(1, len(tr))
    yaw_steps = [abs(res["trace"][i][2] - res["trace"][i - 1][2]) for i in range(1, len(res["trace"]))]
    return {
        "mean_abs_yaw_err": sum(errs) / len(errs) if errs else float("nan"),
        "max_abs_yaw_err": max(errs) if errs else float("nan"),
        "frac_target_in_view": in_view,
        "max_yaw_step_deg": max(yaw_steps),
        "final_yaw": res["trace"][-1][2],
        "tracking_at_end": res["trace"][-1][7],
    }


# ---------------------------------------------------------------- scenarios
def still(az, el=0.0):
    return lambda t: (az, el)


def walk_across(t):             # slow walk left to right, 8 deg/s
    return (-30.0 + 8.0 * t, 5.0)


def walk_fast(t):               # brisk walk, 25 deg/s
    return (-40.0 + 25.0 * t, 5.0)


def walk_up_down(t):            # sways vertically too
    return (20.0 * math.sin(0.5 * t), 10.0 * math.sin(0.8 * t))


def crosser(t):                 # second person crossing the other way
    return (30.0 - 10.0 * t, 5.0)


SCENARIOS = [
    # name, kwargs, checks(metrics)->list[(label, ok)]
    ("stand still, off to the right",
     dict(paths=[still(25.0, 5.0)], seconds=8),
     lambda m, r: [("centers within 5 deg", abs(m["final_yaw"] - 25.0) < 5.0)]),
    ("slow walk across the room",
     dict(paths=[walk_across], seconds=10, ),
     lambda m, r: [("stays in view >95%", m["frac_target_in_view"] > 0.95),
                ("mean error < 8 deg", m["mean_abs_yaw_err"] < 8.0)]),
    ("fast walk (25 deg/s)",
     dict(paths=[walk_fast], seconds=4),
     lambda m, r: [("stays in view >80%", m["frac_target_in_view"] > 0.80)]),
    ("moves up and down and sideways",
     dict(paths=[walk_up_down], seconds=12),
     lambda m, r: [("stays in view >95%", m["frac_target_in_view"] > 0.95)]),
    ("second person crosses; do not swap targets",
     dict(paths=[walk_across, crosser], seconds=10),
     lambda m, r: [("never swapped to the crosser (mean err < 8)", m["mean_abs_yaw_err"] < 8.0),
                ("max step <= max_step", m["max_yaw_step_deg"] <= math.degrees(0.03) + 1e-6)]),
    ("target occluded 1s then returns",
     dict(paths=[still(15.0, 0.0)], seconds=6, dark=(2.0, 3.0)),
     lambda m, r: [("head holds still while lost", yaw_const(r, 2.1, 3.0)),
                ("re-acquires after", m["tracking_at_end"])]),
    ("person walks out of the head's range",
     dict(paths=[lambda t: (20.0 + 10.0 * t, 0.0)], seconds=10),
     lambda m, r: [("yaw clamped at limit", m["final_yaw"] <= ht.YAW_LIMIT_DEG + 1e-6)]),
    ("release drops the lock",
     dict(paths=[still(20.0)], seconds=6, release_at=2.0),
     lambda m, r: [("head stops moving after release", yaw_const(r, 2.2, 6.0))]),
]


def yaw_const(res, t0, t1, tol_deg=1e-6):
    ys = [r[2] for r in res["trace"] if t0 <= r[0] < t1]
    return max(ys) - min(ys) < tol_deg


def view(res, scale=1.0):
    """Replay one scenario in a window: camera view (left) and top-down map (right). space=pause, q=quit."""
    import cv2
    import numpy as np
    W, H = 480, 360
    MAP = 360
    pause = False
    i = 0
    tr = res["trace"]
    while True:
        t, az, yaw, err_az, el, pitch, err_el, tracking, in_view, people = tr[i]
        cam = np.full((H, W, 3), 30, np.uint8)
        cv2.drawMarker(cam, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 20, 1)
        for k, (pa, pe) in enumerate(people):
            cx = 0.5 + (pa - yaw) / HFOV
            cy = 0.5 + (pe - pitch) / VFOV
            if 0 <= cx <= 1 and 0 <= cy <= 1:
                color = (0, 255, 0) if k == 0 else (160, 160, 160)
                x0, x1 = int((cx - 0.07) * W), int((cx + 0.07) * W)
                y0, y1 = int((cy - 0.25) * H), int((cy + 0.25) * H)
                cv2.rectangle(cam, (x0, y0), (x1, y1), color, 2)
                cv2.circle(cam, (int(cx * W), int(cy * H)), 6, color, -1)
        status = "TRACKING" if tracking else "target not seen"
        cv2.putText(cam, "robot camera", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(cam, f"{status}   err {err_az:+.1f} deg", (10, H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)

        top = np.full((MAP, MAP, 3), 30, np.uint8)
        rc = (MAP // 2, MAP - 40)                      # robot position, facing up
        R = 280
        def pt(angle_deg, r):
            a = math.radians(angle_deg)
            return (int(rc[0] + r * math.sin(a)), int(rc[1] - r * math.cos(a)))
        for a in (yaw - HFOV / 2, yaw + HFOV / 2):     # field-of-view edges
            cv2.line(top, rc, pt(a, R), (90, 90, 90), 1)
        cv2.line(top, rc, pt(yaw, R), (0, 200, 255), 2)  # where the head points
        for k, (pa, pe) in enumerate(people):
            cv2.circle(top, pt(pa, 200), 9, (0, 255, 0) if k == 0 else (160, 160, 160), -1)
        cv2.circle(top, rc, 10, (255, 255, 255), -1)
        cv2.putText(top, "top-down (yellow = head, grey = view)", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(top, f"t={t:4.1f}s  head yaw {yaw:+5.1f}  person {az:+5.1f}", (10, MAP - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        top = cv2.resize(top, (MAP, H))
        frame = np.hstack([cam, top])
        if scale != 1.0:
            frame = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.imshow("sim: " + res["name"] + "  (space=pause, r=restart, q=quit)", frame)
        key = cv2.waitKey(int(1000 / FPS)) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            pause = not pause
        if key == ord("r"):
            i = 0
        if not pause:
            i = min(i + 1, len(tr) - 1)
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--view", type=int, nargs="?", const=0, default=None, metavar="N",
                    help="watch scenario N (0-7) in a window instead of running checks")
    args = ap.parse_args()

    if args.view is not None:
        name, kw, _ = SCENARIOS[args.view]
        view(run(name, **kw))
        return

    results, failures = [], 0
    for name, kw, check in SCENARIOS:
        try:
            res = run(name, **kw)
        except Exception as e:      # noqa: BLE001
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            failures += 1
            continue
        m = metrics(res)
        checks = check(m, res)
        ok = all(c[1] for c in checks)
        failures += 0 if ok else 1
        print(f"[{'PASS' if ok else 'FAIL'}] {name}")
        print(f"        mean|err|={m['mean_abs_yaw_err']:.1f} deg  max|err|={m['max_abs_yaw_err']:.1f} deg  "
              f"in view={m['frac_target_in_view']*100:.0f}%  max step={m['max_yaw_step_deg']:.2f} deg/frame")
        for label, good in checks:
            print(f"        {'ok ' if good else 'BAD'} {label}")
        results.append(res)

    if args.plot and results:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(len(results), 1, figsize=(8, 2.2 * len(results)), sharex=False)
        for ax, r in zip(axes, results):
            t = [x[0] for x in r["trace"]]
            ax.plot(t, [x[1] for x in r["trace"]], label="person azimuth")
            ax.plot(t, [x[2] for x in r["trace"]], label="head yaw")
            ax.set_title(r["name"], fontsize=9)
            ax.set_ylabel("deg")
        axes[0].legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sim_results.png"), dpi=110)
        print("saved sim_results.png")

    print(f"\n{len(SCENARIOS) - failures}/{len(SCENARIOS)} scenarios passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
