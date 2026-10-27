# Train the gesture classifier on the sessions recorded with record_gestures.py.
#
#   python train_gestures.py                    # random 20% of clips for validation
#   python train_gestures.py --val-person joe   # validate on a person never trained on

import argparse
import glob
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn

from gesture_common import DATA_DIR, LABELS, MODEL_DIR, POSE_COLS, SEQ_LEN, WINDOW_S, clip_features, window
from gesture_model import GestureGRU


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA_DIR, help="folder of session CSVs")
    ap.add_argument("--val-person", help="hold out this person for validation")
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--crops", type=int, default=5, help="random 1 s crops per training clip")
    ap.add_argument("--min-detect", type=float, default=0.5,
                    help="skip clips where the body was found in less than this fraction of frames")
    ap.add_argument("--out", default=os.path.join(MODEL_DIR, "gesture_gru.pt"))
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    torch.manual_seed(0)

    files = sorted(glob.glob(os.path.join(args.data, "*.csv")))
    if not files:
        raise SystemExit(f"No CSVs in {args.data}/ - record some with record_gestures.py first.")
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)

    clips = []
    skipped = 0
    for clip_id, g in df.groupby("clip_id", sort=False):
        g = g.sort_values("frame")
        if g[POSE_COLS[0]].notna().mean() < args.min_detect:
            skipped += 1
            continue
        clips.append((g, g["label"].iloc[0], g["person"].iloc[0]))
    if skipped:
        print(f"skipping {skipped} clips with poor body detection")
    labels = [lb for lb in LABELS if any(c[1] == lb for c in clips)]
    print("clips per label:", pd.Series([c[1] for c in clips]).value_counts().to_dict())

    y_all = np.array([labels.index(c[1]) for c in clips])
    if args.val_person:
        val = np.array([c[2] == args.val_person for c in clips])
        if not val.any():
            raise SystemExit(f"no clips from person {args.val_person}")
    else:
        _, val_idx = train_test_split(np.arange(len(clips)), test_size=0.2, stratify=y_all, random_state=0)
        val = np.isin(np.arange(len(clips)), val_idx)

    X_tr, y_tr, X_va, y_va = [], [], [], []
    for (g, _, _), y, is_val in zip(clips, y_all, val):
        spare = max(0.0, g["t_ms"].max() - WINDOW_S * 1000)
        if is_val:  # the middle 1 s of the clip
            X_va.append(clip_features(window(g, spare / 2)))
            y_va.append(y)
        else:
            for _ in range(args.crops):
                X_tr.append(clip_features(window(g, rng.uniform(0, spare))))
                y_tr.append(y)
    X_tr, X_va = np.stack(X_tr), np.stack(X_va)
    y_tr, y_va = np.array(y_tr), np.array(y_va)
    print(f"train {len(y_tr)} windows, val {len(y_va)} clips")

    mean = X_tr.reshape(-1, X_tr.shape[-1]).mean(0)
    std = X_tr.reshape(-1, X_tr.shape[-1]).std(0)
    std[std < 1e-6] = 1.0
    norm = lambda x: torch.tensor((x - mean) / std, dtype=torch.float32)
    Xt, Xv = norm(X_tr), norm(X_va)
    yt, yv = torch.tensor(y_tr), torch.tensor(y_va)

    # weight classes so e.g. 4x as many come_here clips as stop clips doesn't bias the model
    counts = np.bincount(y_tr, minlength=len(labels))
    weights = torch.tensor(counts.sum() / (len(labels) * np.maximum(counts, 1)), dtype=torch.float32)

    model = GestureGRU(Xt.shape[-1], len(labels), args.hidden)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(weight=weights)

    best_acc, best_state = -1.0, None
    for epoch in range(args.epochs):
        model.train()
        perm = torch.randperm(len(Xt))
        for b in range(0, len(perm), 32):
            idx = perm[b:b + 32]
            xb = Xt[idx] + 0.05 * torch.randn_like(Xt[idx])  # jitter
            loss = loss_fn(model(xb), yt[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            acc = (model(Xv).argmax(1) == yv).float().mean().item()
        if acc >= best_acc:
            best_acc, best_state = acc, {k: v.clone() for k, v in model.state_dict().items()}
        if epoch % 10 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  loss {loss.item():.3f}  val acc {acc:.3f}")

    model.load_state_dict(best_state)
    with torch.no_grad():
        pred = model(Xv).argmax(1).numpy()
    print(f"\nbest val acc {best_acc:.3f}")
    print("confusion matrix (rows = true, cols = predicted):", labels)
    print(confusion_matrix(y_va, pred, labels=range(len(labels))))
    print(classification_report(y_va, pred, labels=range(len(labels)), target_names=labels, zero_division=0))

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": best_state, "labels": labels, "hidden": args.hidden,
        "n_features": Xt.shape[-1], "seq_len": SEQ_LEN, "window_s": WINDOW_S,
        "mean": mean, "std": std,
    }, args.out)
    print("saved", args.out)


if __name__ == "__main__":
    main()
