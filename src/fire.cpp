#include "fire.h"

#include <math.h>
#include <string.h>

// band b (1-based Landsat band number, 1..7) of pixel i, as x = DN - 5000
static inline int32_t X(const uint16_t *img, int i, int band) {
    return (int32_t)img[(size_t)i * N_BANDS + (band - 1)] - 5000;
}

// smallest integer x such that  2e-5 * x / s >= thr  (s = sin(sun elevation))
static int32_t int_threshold(double thr, double s) {
    int32_t t = (int32_t)ceil(thr * s / 2e-5) - 2;
    while (!((2e-5 * t) / s >= thr)) t++;   // settle on the exact double-precision boundary
    return t;
}

FireParams fire_params(double sun_elevation_deg, bool paper_eq3) {
    const double s = sin(sun_elevation_deg * 3.14159265358979323846 / 180.0);
    FireParams p;
    p.t_b7_unamb = int_threshold(0.15, s);
    p.t_b6_cand = int_threshold(0.50, s);
    p.paper_eq3 = paper_eq3;
    return p;
}

// a / b >= num/den, for integers a, b (b may be negative or zero), without division.
// b == 0 follows the reference code: the denominator is replaced by epsilon, so the
// ratio is +huge, 0 or -huge depending on the sign of a.
static inline bool ratio_ge(int32_t a, int32_t b, int32_t num, int32_t den) {
    if (b > 0) return (int64_t)a * den >= (int64_t)b * num;
    if (b < 0) return (int64_t)a * den <= (int64_t)b * num;
    return a > 0;
}

int detect_fire(const uint16_t *img, uint8_t *mask, const FireParams &p) {
    static uint8_t unamb[PATCH_H * PATCH_W];
    const int n = PATCH_H * PATCH_W;

    // Murphy eq. 2 (unambiguous fire): rho7 >= 0.15, rho7/rho6 >= 1.4, rho7/rho5 >= 1.4
    for (int i = 0; i < n; i++) {
        const int32_t x5 = X(img, i, 5), x6 = X(img, i, 6), x7 = X(img, i, 7);
        unamb[i] = x7 >= p.t_b7_unamb && ratio_ge(x7, x6, 7, 5) && ratio_ge(x7, x5, 7, 5);
    }

    // Murphy eq. 3 (potential fire next to an unambiguous one): rho6/rho5 >= 2 and rho6 >= 0.5,
    // within the 3x3 neighbourhood of an unambiguous fire pixel. (Saturation flags are not
    // available in the patches.)
    int count = 0;
    for (int y = 0; y < PATCH_H; y++) {
        for (int x = 0; x < PATCH_W; x++) {
            const int i = y * PATCH_W + x;
            uint8_t fire = unamb[i];
            if (!fire) {
                bool near = false;
                for (int dy = -1; dy <= 1 && !near; dy++)
                    for (int dx = -1; dx <= 1; dx++) {
                        const int yy = y + dy, xx = x + dx;
                        if (yy >= 0 && yy < PATCH_H && xx >= 0 && xx < PATCH_W && unamb[yy * PATCH_W + xx]) {
                            near = true;
                            break;
                        }
                    }
                if (near) {
                    const int32_t x5 = X(img, i, 5), x6 = X(img, i, 6);
                    bool r65;
                    if (p.paper_eq3)
                        r65 = ratio_ge(x6, x5, 2, 1);
                    else
                        // reference implementation: the denominator is epsilon when rho5 > 0,
                        // so the test only depends on the sign of rho5 (rho6 > 0 is implied below)
                        r65 = x5 >= 0;
                    fire = r65 && x6 >= p.t_b6_cand;
                }
            }
            mask[i] = fire;
            count += fire;
        }
    }
    return count;
}

int extract_alerts(const uint8_t *mask, const uint16_t *img, FireAlert *alerts, int max_alerts) {
    static int32_t label[PATCH_H * PATCH_W];
    static int32_t stack[PATCH_H * PATCH_W];
    memset(label, 0, sizeof label);
    int clusters = 0;

    for (int start = 0; start < PATCH_H * PATCH_W; start++) {
        if (!mask[start] || label[start]) continue;
        clusters++;
        int sp = 0, npx = 0;
        int64_t sy = 0, sx = 0;
        int y0 = PATCH_H, y1 = -1, x0 = PATCH_W, x1 = -1;
        uint16_t peak = 0;
        stack[sp++] = start;
        label[start] = clusters;
        while (sp) {                                   // flood fill, 8-connectivity
            const int i = stack[--sp], y = i / PATCH_W, x = i % PATCH_W;
            npx++;
            sy += y;
            sx += x;
            if (y < y0) y0 = y;
            if (y > y1) y1 = y;
            if (x < x0) x0 = x;
            if (x > x1) x1 = x;
            const uint16_t b7 = img[(size_t)i * N_BANDS + 6];
            if (b7 > peak) peak = b7;
            for (int dy = -1; dy <= 1; dy++)
                for (int dx = -1; dx <= 1; dx++) {
                    const int yy = y + dy, xx = x + dx;
                    if (yy < 0 || yy >= PATCH_H || xx < 0 || xx >= PATCH_W) continue;
                    const int j = yy * PATCH_W + xx;
                    if (mask[j] && !label[j]) {
                        label[j] = clusters;
                        stack[sp++] = j;
                    }
                }
        }
        if (clusters <= max_alerts) {
            FireAlert &a = alerts[clusters - 1];
            a.row = (uint16_t)(sy / npx);
            a.col = (uint16_t)(sx / npx);
            a.n_pixels = (uint16_t)npx;
            a.peak_b7 = peak;
            a.h = (uint16_t)(y1 - y0 + 1);
            a.w = (uint16_t)(x1 - x0 + 1);
        }
    }
    return clusters;
}
