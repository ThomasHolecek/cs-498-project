"""
head_sweep.py  -  first thing to run at the robot tomorrow.

Moves the K1 head through a few fixed angles with a pause between each, then recenters.
It answers four questions before you touch the tracking code:
    1. Can the laptop talk to the robot at all?
    2. Does RotateHead take radians or degrees?
    3. Which direction is +yaw and +pitch?
    4. How far can the head actually go?

Usage:
    python head_sweep.py --iface eth0
    python head_sweep.py --iface eth0 --degrees      # if radians clearly did nothing / too little

Robot should be in a standing mode (PREP or walking) with someone on the e-stop.
"""

import argparse
import math
import time

from booster_robotics_sdk_python import ChannelFactory, B1LocoClient   # names: VERIFY against SDK python example

# (pitch_deg, yaw_deg, label)
SWEEP = [
    (0,   0,  "center"),
    (0,  20,  "yaw +20   -> note which way the head turned"),
    (0,   0,  "center"),
    (0, -20,  "yaw -20"),
    (0,   0,  "center"),
    (10,  0,  "pitch +10 -> note up or down"),
    (0,   0,  "center"),
    (-10, 0,  "pitch -10"),
    (0,   0,  "center"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iface", default="eth0")
    ap.add_argument("--degrees", action="store_true", help="send degrees instead of radians")
    ap.add_argument("--pause", type=float, default=2.0)
    args = ap.parse_args()

    ChannelFactory.Instance().Init(0, args.iface)
    client = B1LocoClient()
    client.Init()
    print("connected")

    for pitch_deg, yaw_deg, label in SWEEP:
        if args.degrees:
            p, y = float(pitch_deg), float(yaw_deg)
        else:
            p, y = math.radians(pitch_deg), math.radians(yaw_deg)
        rc = client.RotateHead(p, y)
        print(f"{label:45s}  sent pitch={p:+.3f} yaw={y:+.3f}  rc={rc}")
        time.sleep(args.pause)

    print("done. write down: units, +yaw direction, +pitch direction, any limit errors.")


if __name__ == "__main__":
    main()
