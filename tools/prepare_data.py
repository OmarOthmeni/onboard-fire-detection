#!/usr/bin/env python3
"""Convert the Landsat-8 active-fire patches (GeoTIFF) into raw files the C++ detector reads.

Input  : the "samples" folder of the Pereira et al. (2021) dataset
         (https://github.com/pereira-gha/activefire, CC-BY-4.0), already unzipped.
Output : data/<split>/<name>.raw       uint16, 256 x 256 x 10, band-interleaved per pixel (as stored)
         data/<split>/<name>.<mask>.u8  uint8 0/1 masks (Murphy, Schroeder, GOLI, Voting, manual)
         data/<split>/index.csv

Splits:
  random : 100 randomly selected fire patches with the 3 algorithm masks (+ voting)   -> CNN training/validation
  manual : 100 patches with human-annotated fire masks                                 -> final test set
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import tifffile


def write(img, path):
    np.ascontiguousarray(img).tofile(path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", required=True, type=Path, help="unzipped samples/ folder")
    ap.add_argument("--out", default=Path("data"), type=Path)
    a = ap.parse_args()

    # ---- random split: images/patches + masks/patches/<scene>_<ALGO>_pXXXXX.tif
    out = a.out / "random"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for f in sorted((a.samples / "images" / "patches").glob("*.tif")):
        name = f.stem
        img = tifffile.imread(f)
        assert img.shape == (256, 256, 10) and img.dtype == np.uint16
        write(img, out / f"{name}.raw")
        masks = {}
        for algo, tag in [("murphy", "Murphy"), ("schroeder", "Schroeder"), ("goli", "GOLI_v2")]:
            mf = a.samples / "masks" / "patches" / (name.replace("_RT_p", f"_RT_{tag}_p") + ".tif")
            masks[algo] = (tifffile.imread(mf) > 0).astype(np.uint8)
            write(masks[algo], out / f"{name}.{algo}.u8")
        votes = masks["murphy"].astype(int) + masks["schroeder"] + masks["goli"]
        write((votes >= 2).astype(np.uint8), out / f"{name}.voting.u8")
        rows.append([name] + [int(m.sum()) for m in masks.values()] + [int((votes >= 2).sum())])
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "murphy_px", "schroeder_px", "goli_px", "voting_px"])
        w.writerows(rows)
    print(f"random: {len(rows)} patches")

    # ---- manual split: landsat_patches/<scene>_pXXXXX.tif + manual_annotations_patches/<scene>_v1_pXXXXX.tif
    mdir = a.samples / "manual_annotations"
    out = a.out / "manual"
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for af in sorted((mdir / "manual_annotations_patches").glob("*.tif")):
        name = af.stem.replace("_RT_v1_p", "_RT_p")
        imgf = mdir / "landsat_patches" / f"{name}.tif"
        if not imgf.exists():
            continue
        img = tifffile.imread(imgf)
        assert img.shape == (256, 256, 10) and img.dtype == np.uint16
        write(img, out / f"{name}.raw")
        manual = (tifffile.imread(af) > 0).astype(np.uint8)
        write(manual, out / f"{name}.manual.u8")
        row = [name, int(manual.sum())]
        mf = mdir / "masks_patches" / (name.replace("_RT_p", "_RT_Murphy_p") + ".tif")
        if mf.exists():
            m = (tifffile.imread(mf) > 0).astype(np.uint8)
            write(m, out / f"{name}.murphy.u8")
            row.append(int(m.sum()))
        else:
            row.append(-1)
        rows.append(row)
    with open(out / "index.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["name", "manual_px", "murphy_px"])
        w.writerows(rows)
    print(f"manual: {len(rows)} patches")


if __name__ == "__main__":
    main()
