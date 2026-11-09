#!/usr/bin/env python3
"""Tiny U-Net that segments active-fire pixels, so the fire area can be measured.

Ground truth: the 100 human-annotated patches (data/manual). With so few labelled patches the
evaluation is 5-fold cross-validation: each fold trains on 80 patches and tests on the 20 held
out. The physics detector (Murphy, C++ masks in build/masks/manual) is scored on exactly the same
test patches, so the comparison is fair.

Input: 7 OLI bands as TOA reflectance without solar correction, (DN * 2e-5 - 0.1), clipped to
[-0.1, 1.5]. Output: fire probability per pixel. Fire area = fire pixels x 0.09 ha.

Usage: python model/train_unet.py [--epochs 60] [--folds 5]
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
H = W = 256
N_IN = 7
torch.manual_seed(0)
np.random.seed(0)
torch.set_num_threads(max(1, torch.get_num_threads()))


def conv_block(cin, cout):
    return nn.Sequential(nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
                         nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class TinyUNet(nn.Module):
    """3-level U-Net, 12-24-48 channels (66,661 parameters)."""

    def __init__(self, c=(12, 24, 48)):
        super().__init__()
        self.e1, self.e2, self.e3 = conv_block(N_IN, c[0]), conv_block(c[0], c[1]), conv_block(c[1], c[2])
        self.u2, self.d2 = nn.ConvTranspose2d(c[2], c[1], 2, 2), conv_block(2 * c[1], c[1])
        self.u1, self.d1 = nn.ConvTranspose2d(c[1], c[0], 2, 2), conv_block(2 * c[0], c[0])
        self.head = nn.Conv2d(c[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(F.max_pool2d(e1, 2))
        e3 = self.e3(F.max_pool2d(e2, 2))
        d2 = self.d2(torch.cat([self.u2(e3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.head(d1)


def load_split():
    raws = sorted((ROOT / "data" / "manual").glob("*.raw"))
    x = np.stack([np.fromfile(r, dtype=np.uint16).reshape(H, W, 10)[..., :N_IN] for r in raws])
    y = np.stack([np.fromfile(r.with_suffix(".manual.u8"), dtype=np.uint8).reshape(H, W) for r in raws])
    murphy = np.stack([np.fromfile(ROOT / "build" / "masks" / "manual" / (r.stem + ".u8"), dtype=np.uint8)
                       .reshape(H, W) for r in raws])
    return [r.stem for r in raws], x, y, murphy


def to_input(x_u16):
    rho = x_u16.astype(np.float32) * 2e-5 - 0.1
    return torch.from_numpy(np.clip(rho, -0.1, 1.5)).permute(0, 3, 1, 2).contiguous()


def random_crops(x, y, n, size=96, fire_bias=0.7):
    """Crops centred on fire pixels with probability fire_bias (fire is <1% of pixels)."""
    xs, ys = [], []
    for _ in range(n):
        i = np.random.randint(len(x))
        fy, fx = np.nonzero(y[i])
        if len(fy) and np.random.rand() < fire_bias:
            k = np.random.randint(len(fy))
            cy = np.clip(fy[k] + np.random.randint(-size // 3, size // 3), size // 2, H - size // 2)
            cx = np.clip(fx[k] + np.random.randint(-size // 3, size // 3), size // 2, W - size // 2)
        else:
            cy, cx = np.random.randint(size // 2, H - size // 2 + 1, 2)
        sl = (i, slice(cy - size // 2, cy + size // 2), slice(cx - size // 2, cx + size // 2))
        xc, yc = x[sl], y[sl]
        k = np.random.randint(4)                       # rotations + flip: fire has no orientation
        xc, yc = np.rot90(xc, k), np.rot90(yc, k)
        if np.random.rand() < 0.5:
            xc, yc = xc[:, ::-1], yc[:, ::-1]
        xs.append(np.ascontiguousarray(xc)); ys.append(np.ascontiguousarray(yc))
    return to_input(np.stack(xs)), torch.from_numpy(np.stack(ys)).float().unsqueeze(1)


def dice_bce(logits, target):
    p = torch.sigmoid(logits)
    dice = 1 - (2 * (p * target).sum() + 1) / (p.sum() + target.sum() + 1)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=torch.tensor(5.0)) + dice


def scores(pred, ref):
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    return np.array([tp, fp, fn])


def summarize(s):
    tp, fp, fn = s
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "iou": tp / max(1, tp + fp + fn)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--steps", type=int, default=40, help="batches per epoch")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    names, x, y, murphy = load_split()
    y, murphy = y > 0, murphy > 0
    perm = np.random.permutation(len(names))
    folds = np.array_split(perm, a.folds)
    n_params = sum(p.numel() for p in TinyUNet().parameters())
    print(f"{len(names)} annotated patches | {a.folds}-fold CV | TinyUNet {n_params:,} parameters")

    cnn_s, mur_s = np.zeros(3, int), np.zeros(3, int)
    cnn_area_err, mur_area_err, true_area, per_patch = [], [], [], []
    for f, test_idx in enumerate(folds):
        train_idx = np.setdiff1d(perm, test_idx)
        model = TinyUNet()
        opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-3, total_steps=a.epochs * a.steps)
        t0 = time.time()
        for ep in range(a.epochs):
            model.train()
            for _ in range(a.steps):
                xb, yb = random_crops(x[train_idx], y[train_idx], 16)
                loss = dice_bce(model(xb), yb)
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        model.eval()
        with torch.no_grad():
            prob = torch.cat([torch.sigmoid(model(to_input(x[test_idx][i:i + 4])))
                              for i in range(0, len(test_idx), 4)]).squeeze(1).numpy()
        pred = prob > 0.5
        for j, i in enumerate(test_idx):
            cnn_s += scores(pred[j], y[i]); mur_s += scores(murphy[i], y[i])
            ta = y[i].sum() * 0.09
            cnn_area_err.append(abs(pred[j].sum() * 0.09 - ta)); mur_area_err.append(abs(murphy[i].sum() * 0.09 - ta))
            true_area.append(ta)
            per_patch.append({"patch": names[i], "true_ha": round(float(ta), 2),
                              "cnn_ha": round(float(pred[j].sum() * 0.09), 2),
                              "murphy_ha": round(float(murphy[i].sum() * 0.09), 2)})
        print(f"fold {f+1}: CNN IoU {summarize(scores(pred, y[test_idx]))['iou']:.3f} | "
              f"Murphy IoU {summarize(scores(murphy[test_idx], y[test_idx]))['iou']:.3f} | {time.time()-t0:.0f}s")
        torch.save(model.state_dict(), ROOT / "model" / f"unet_fold{f+1}.pt")

    res = {"parameters": n_params, "folds": a.folds, "patches": len(names),
           "cnn": {**summarize(cnn_s), "area_mae_ha": float(np.mean(cnn_area_err))},
           "murphy": {**summarize(mur_s), "area_mae_ha": float(np.mean(mur_area_err))},
           "mean_true_area_ha": float(np.mean(true_area)), "per_patch": per_patch}
    for k in ["cnn", "murphy"]:
        r = res[k]
        print(f"{k:7s}: precision {r['precision']:.3f} | recall {r['recall']:.3f} | IoU {r['iou']:.3f} "
              f"| area MAE {r['area_mae_ha']:.2f} ha")
    print(f"mean true fire area per patch: {res['mean_true_area_ha']:.2f} ha")
    (ROOT / "results").mkdir(exist_ok=True)
    (ROOT / "results" / "unet_cv.json").write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
