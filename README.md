# Onboard Active-Fire Detection for Small Satellites (Landsat-8)

Detect active fires in multispectral satellite imagery **on board**, measure their size, and send a
few bytes of alerts to the ground instead of the images. Two detectors are compared on the same
human-annotated test patches:

1. **A physics-based detector in integer-only C++** (Murphy et al., 2016 conditions), built for
   embedded CPUs and FPGAs: no floating point, no division, fixed memory.
2. **A tiny U-Net** (66,661 parameters) that segments fire pixels so the burning area can be
   measured.

*Concept from my ActInSpace 2026 hackathon entry (onboard wildfire alerts); this repository is an
independent rebuild on public data, with every number below measured.*

## Results

**Physics detector (C++, integer-only)**, 100 human-annotated 256×256 patches:

| Check | Result |
|---|---|
| Integer C++ vs float64 reference of the same equations | **0 differing pixels** on all 200 patches (bit-exact) |
| vs the dataset authors' Murphy masks (computed on full scenes) | precision **1.00**, recall **0.96**, IoU **0.96** |
| vs **human annotations** | precision **0.79**, recall **0.95**, IoU **0.75** |
| Speed (x86 laptop, 1 core, `-O2`) | **0.9 ms per patch** (73 Mpixel/s) |
| Downlink | **131 MB of raw patches → 3.4–20 KB of alerts** (6,500–38,000× smaller) |

**Physics vs CNN**, same 100 human-annotated patches, 5-fold cross-validation (each patch is scored
by a model that never saw it):

| | Precision | Recall | IoU | Fire-area error per patch (mean) | (median, relative) |
|---|---|---|---|---|---|
| Murphy (C++) | 0.787 | **0.949** | 0.755 | 15.4 ha | **20.0%** |
| Tiny U-Net | **0.828** | 0.938 | **0.785** | **13.0 ha** | 23.1% |

Mean true fire area per patch: 64.7 ha.

Reading the comparison honestly:
- The CNN removes part of the physics rules' **false alarms** (precision 0.79 → 0.83) at almost the
  same recall, and its pixel IoU is higher in 4 of the 5 folds.
- It measures **large fires** better (lower mean area error), but on a **typical patch** the
  physics rules are still slightly more accurate (lower median relative error).
- With only 100 human-labelled patches the gain is modest and fold-dependent (fold IoU 0.64–0.89);
  more labelled data is the main lever.

## How it works

### Physics detector: integer reformulation
Landsat-8 Level-1 pixels are 16-bit digital numbers (DN). Top-of-atmosphere reflectance is

ρ = (2·10⁻⁵ · DN − 0.1) / sin(sun elevation) = 2·10⁻⁵ · (DN − 5000) / sin(sun elevation)

so with `x = DN − 5000`:
- every reflectance threshold (ρ₇ ≥ 0.15, ρ₆ ≥ 0.5) becomes an **integer threshold** on x,
  precomputed once per scene from the sun elevation;
- every reflectance ratio (ρ₇/ρ₆ ≥ 1.4, ρ₇/ρ₅ ≥ 1.4) becomes a **cross-multiplication**
  (5·x₇ ≥ 7·x₆), because the 1/sin factor cancels: no division and no floating point.

Then a 3×3 neighbourhood test for potential fires, 8-connected cluster labelling, and one 12-byte
alert per fire (centroid, pixel count → area in hectares, bounding box, peak short-wave-infrared intensity).

Notes:
- The patches carry no scene metadata, so the default is **no solar-angle correction**
  (`--sun-elev 90`). With the true sun elevation the agreement with the reference masks rises
  further (IoU 0.98 at 60° in a quick test); without it the detector is conservative (zero
  false-positive pixels versus the reference).
- The dataset's reference code for Murphy's eq. 3 divides by ε exactly when ρ₅ > 0, which reduces
  the ratio test to the sign of ρ₅. This repository reproduces that behaviour by default (to
  validate against the reference masks) and offers the equation as written in the paper with
  `--paper-eq3`.
- Saturation flags (Landsat QA band) are not in the patches, so that branch of eq. 3 is not used.

### Tiny U-Net
- Input: 7 OLI bands as reflectance (no solar correction), 96×96 random crops centred on fire
  70% of the time (fire is < 1% of pixels), rotations and flips.
- 3-level encoder/decoder with 12/24/48 channels, BatchNorm, ReLU; loss = weighted BCE + Dice.
- Fire area = predicted fire pixels × 0.09 ha (30 m × 30 m pixels).

## Data

[Landsat-8 active fire dataset](https://github.com/pereira-gha/activefire) by G. H. de Almeida
Pereira et al., *Active fire detection in Landsat-8 imagery: A large-scale dataset and a
deep-learning study*, ISPRS J. Photogramm. Remote Sens. 178 (2021), licensed **CC-BY-4.0**.
This project uses the authors' public sample subset (242 MB): 100 random fire patches with the
masks of three published algorithms, and 100 patches with **manual annotations**. The data is not
redistributed here; download the `samples` archive from the authors' repository.

## Reproduce

```bash
pip install torch numpy tifffile
python tools/prepare_data.py --samples <path to unzipped samples/> --out data
make                                   # builds ./firedet (C++17, no dependencies)
python tools/validate_detector.py      # bit-exactness, reference agreement, human-annotation scores, timing
python model/train_unet.py             # 5-fold cross-validation of the U-Net vs the physics detector
./firedet detect data/manual/<patch>.raw mask.u8 --sun-elev 55    # one patch -> mask + JSON alerts
```
Results are written to `results/`.

## Limitations

- Area = active-fire **pixel extent** at 30 m; fires smaller than a pixel cannot be sized precisely,
  and this is not the burned-area perimeter.
- 256×256 patches cut from full scenes: the reference masks were computed on whole scenes, so a few
  pixels differ at patch borders.
- Small labelled test set (100 patches), hence cross-validation and the fold spread above.

## Next steps

- Run the C++ detector on embedded targets (ARM Cortex-A9 on PYNQ, RISC-V on PolarFire SoC) and
  report time and energy per patch.
- INT8 quantization of the U-Net and an FPGA implementation with the HLS flow of
  [accelerated_network_SoC](https://github.com/OmarOthmeni/accelerated_network_SoC).
- Use more labelled patches from the full dataset.

*Omar Othmeni*
