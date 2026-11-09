#!/usr/bin/env python3
"""Validate the C++ integer detector.

1. bit-exactness: C++ integer masks == a float64 NumPy implementation of the same equations
   (the reference implementation's formulas, see murphy_float) on every patch
2. agreement with the dataset's Murphy masks (computed by the authors on full scenes)
3. accuracy against the human annotations (manual split): pixel precision / recall / IoU,
   and fire-area error per patch (area = fire pixels x 0.09 ha)

Usage: python tools/validate_detector.py [--sun-elev 90] [--paper-eq3]
"""
import argparse
import json
import math
import subprocess
from pathlib import Path

import numpy as np

H = W = 256
ROOT = Path(__file__).resolve().parent.parent


def load_raw(p):
    return np.fromfile(p, dtype=np.uint16).reshape(H, W, 10)


def load_mask(p):
    return np.fromfile(p, dtype=np.uint8).reshape(H, W) > 0


def dilate3(m):
    p = np.pad(m, 1)
    out = np.zeros_like(m)
    for dy in range(3):
        for dx in range(3):
            out |= p[dy:dy + H, dx:dx + W]
    return out


def murphy_float(img, sun_elev, paper_eq3):
    """Murphy et al. (2016) as in the dataset's reference code (create_masks.py), in float64."""
    s = math.sin(math.radians(sun_elev))
    rho = (img.astype(np.float64) * 2e-5 - 0.1) / s
    b = {i + 1: rho[..., i] for i in range(7)}
    eps = np.finfo(float).eps
    p5 = np.where(b[5] == 0, eps, b[5])
    p6 = np.where(b[6] == 0, eps, b[6])
    unamb = (b[7] >= 0.15) & (b[7] / p6 >= 1.4) & (b[7] / p5 >= 1.4)
    with np.errstate(divide="ignore", invalid="ignore"):
        if paper_eq3:
            r65 = np.where(b[5] > 0, b[6] / np.where(b[5] > 0, b[5], 1), -np.inf) >= 2.0
        else:
            p5b = np.where(b[5] > 0, eps, b[5])          # (sic) reference implementation
            r65 = b[6] / p5b >= 2.0
    cand = dilate3(unamb) & r65 & (b[6] >= 0.5)
    return unamb | cand


def scores(pred, ref):
    tp = int((pred & ref).sum()); fp = int((pred & ~ref).sum()); fn = int((~pred & ref).sum())
    return tp, fp, fn


def summarize(tp, fp, fn):
    return {"precision": tp / max(1, tp + fp), "recall": tp / max(1, tp + fn), "iou": tp / max(1, tp + fp + fn),
            "tp": tp, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sun-elev", type=float, default=90.0)
    ap.add_argument("--paper-eq3", action="store_true")
    ap.add_argument("--data", type=Path, default=ROOT / "data")
    a = ap.parse_args()

    results = {"sun_elevation_deg": a.sun_elev, "paper_eq3": a.paper_eq3}
    extra = ["--sun-elev", str(a.sun_elev)] + (["--paper-eq3"] if a.paper_eq3 else [])
    for split in ["random", "manual"]:
        raws = sorted((a.data / split).glob("*.raw"))
        out = ROOT / "build" / "masks" / split
        out.mkdir(parents=True, exist_ok=True)
        lst = ROOT / "build" / f"{split}.txt"
        lst.write_text("\n".join(str(r) for r in raws) + "\n")
        bench = subprocess.run([str(ROOT / "firedet"), "batch", str(lst), str(out)] + extra + ["--repeat", "20"],
                               check=True, capture_output=True, text=True).stdout
        print(f"[{split}] {bench.strip()}")

        mism, ref_s, man_s, area_err, area_ref = 0, [0, 0, 0], [0, 0, 0], [], []
        for r in raws:
            cpp = load_mask(out / (r.stem + ".u8"))
            mism += int((cpp != murphy_float(load_raw(r), a.sun_elev, a.paper_eq3)).sum())
            ref_s = [x + y for x, y in zip(ref_s, scores(cpp, load_mask(r.with_suffix(".murphy.u8"))))]
            if split == "manual":
                man = load_mask(r.with_suffix(".manual.u8"))
                man_s = [x + y for x, y in zip(man_s, scores(cpp, man))]
                area_err.append(abs(int(cpp.sum()) - int(man.sum())) * 0.09)
                area_ref.append(int(man.sum()) * 0.09)
        res = {"patches": len(raws), "pixel_mismatches_vs_float_reference": mism,
               "vs_dataset_murphy_masks": summarize(*ref_s), "timing": bench.strip()}
        print(f"[{split}] C++ integer vs float reference: {mism} differing pixels")
        print(f"[{split}] vs dataset Murphy masks: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                                   for k, v in res['vs_dataset_murphy_masks'].items()}))
        if split == "manual":
            res["vs_manual_annotations"] = summarize(*man_s)
            res["area_mae_ha"] = float(np.mean(area_err))
            res["mean_true_area_ha"] = float(np.mean(area_ref))
            print(f"[manual] vs human annotations: " + json.dumps({k: round(v, 4) if isinstance(v, float) else v
                                                                   for k, v in res['vs_manual_annotations'].items()}))
            print(f"[manual] fire area per patch: mean abs error {res['area_mae_ha']:.2f} ha "
                  f"(mean true area {res['mean_true_area_ha']:.2f} ha)")
        results[split] = res
    (ROOT / "results").mkdir(exist_ok=True)
    tag = f"sun{int(a.sun_elev)}" + ("_paper" if a.paper_eq3 else "")
    (ROOT / "results" / f"detector_{tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
