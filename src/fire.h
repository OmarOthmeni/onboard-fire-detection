// Onboard active-fire detection for Landsat-8 OLI patches.
//
// Integer-only implementation of the Murphy et al. (2016) active-fire conditions
// (Remote Sensing of Environment 175, 16-21), designed for embedded / FPGA use:
// no floating point, no division, one pass over the pixels, fixed memory.
//
// Pixels are Landsat-8 Level-1 digital numbers (DN). TOA reflectance is
//   rho = (2e-5 * DN - 0.1) / sin(sun_elevation) = 2e-5 * (DN - 5000) / sin(sun_elevation)
// so every reflectance threshold becomes an integer threshold on x = DN - 5000, and
// every reflectance ratio becomes a cross-multiplication (the 1/sin factor cancels).
#pragma once
#include <stddef.h>
#include <stdint.h>

#define PATCH_H 256
#define PATCH_W 256
#define N_BANDS 10           // bands 1..7, 9, 10, 11 as stored in the dataset (interleaved per pixel)
#define PIXEL_AREA_HA 0.09   // 30 m x 30 m

struct FireParams {
    // integer thresholds on x = DN - 5000, precomputed from the sun elevation
    int32_t t_b7_unamb;   // rho7 >= 0.15
    int32_t t_b6_cand;    // rho6 >= 0.50
    bool paper_eq3;       // false: reproduce the reference implementation bit for bit
                          // true : equation 3 exactly as written in the paper (rho6/rho5 >= 2)
};

// sun_elevation_deg = 90 means "no solar-angle correction" (use when the scene metadata is missing)
FireParams fire_params(double sun_elevation_deg, bool paper_eq3);

// img: PATCH_H x PATCH_W x N_BANDS uint16 (pixel-interleaved). mask: PATCH_H x PATCH_W, 0/1.
// Returns the number of fire pixels.
int detect_fire(const uint16_t *img, uint8_t *mask, const FireParams &p);

// One alert per connected fire cluster (8-connectivity): this is what would be downlinked.
struct FireAlert {          // 12 bytes
    uint16_t row, col;      // centroid in the patch (pixels)
    uint16_t n_pixels;      // cluster size; area = n_pixels * 0.09 ha
    uint16_t peak_b7;       // max band-7 DN in the cluster (intensity proxy)
    uint16_t h, w;          // bounding box size
};

// Labels the clusters of `mask` and fills `alerts` (up to max_alerts). Returns the cluster count.
int extract_alerts(const uint8_t *mask, const uint16_t *img, FireAlert *alerts, int max_alerts);
