// firedet: run the detector on raw Landsat-8 patches.
//
//   firedet detect <patch.raw> <mask_out.u8> [--sun-elev DEG] [--paper-eq3]
//       writes the fire mask and prints the alerts as JSON
//   firedet batch <list.txt> <out_dir> [--sun-elev DEG] [--paper-eq3] [--repeat N]
//       runs every patch listed (one .raw path per line), writes <out_dir>/<name>.u8,
//       and reports timing and downlink size
#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>

#include "fire.h"

static const size_t PATCH_BYTES = (size_t)PATCH_H * PATCH_W * N_BANDS * sizeof(uint16_t);

static bool read_patch(const char *path, std::vector<uint16_t> &img) {
    img.resize((size_t)PATCH_H * PATCH_W * N_BANDS);
    FILE *f = fopen(path, "rb");
    if (!f) return false;
    const bool ok = fread(img.data(), 1, PATCH_BYTES, f) == PATCH_BYTES;
    fclose(f);
    return ok;
}

static bool write_mask(const char *path, const uint8_t *mask) {
    FILE *f = fopen(path, "wb");
    if (!f) return false;
    const bool ok = fwrite(mask, 1, PATCH_H * PATCH_W, f) == (size_t)PATCH_H * PATCH_W;
    fclose(f);
    return ok;
}

static std::string stem(const std::string &path) {
    size_t s = path.find_last_of('/');
    std::string n = s == std::string::npos ? path : path.substr(s + 1);
    size_t d = n.rfind(".raw");
    return d == std::string::npos ? n : n.substr(0, d);
}

int main(int argc, char **argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: firedet detect <patch.raw> <mask.u8> [--sun-elev DEG] [--paper-eq3]\n"
                        "       firedet batch <list.txt> <out_dir> [--sun-elev DEG] [--paper-eq3] [--repeat N]\n");
        return 2;
    }
    double sun = 90.0;
    bool paper = false;
    int repeat = 1;
    for (int i = 4; i < argc; i++) {
        if (!strcmp(argv[i], "--sun-elev") && i + 1 < argc) sun = atof(argv[++i]);
        else if (!strcmp(argv[i], "--paper-eq3")) paper = true;
        else if (!strcmp(argv[i], "--repeat") && i + 1 < argc) repeat = atoi(argv[++i]);
    }
    const FireParams p = fire_params(sun, paper);
    static uint8_t mask[PATCH_H * PATCH_W];
    static FireAlert alerts[256];
    std::vector<uint16_t> img;

    if (!strcmp(argv[1], "detect")) {
        if (!read_patch(argv[2], img)) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        const int npx = detect_fire(img.data(), mask, p);
        const int nc = extract_alerts(mask, img.data(), alerts, 256);
        if (!write_mask(argv[3], mask)) return 1;
        printf("{\"fire_pixels\": %d, \"area_ha\": %.2f, \"alerts\": [", npx, npx * PIXEL_AREA_HA);
        for (int k = 0; k < nc && k < 256; k++)
            printf("%s{\"row\": %u, \"col\": %u, \"pixels\": %u, \"area_ha\": %.2f, \"peak_b7\": %u}",
                   k ? ", " : "", alerts[k].row, alerts[k].col, alerts[k].n_pixels,
                   alerts[k].n_pixels * PIXEL_AREA_HA, alerts[k].peak_b7);
        printf("]}\n");
        return 0;
    }

    if (!strcmp(argv[1], "batch")) {
        FILE *lf = fopen(argv[2], "r");
        if (!lf) { fprintf(stderr, "cannot read %s\n", argv[2]); return 1; }
        std::vector<std::string> paths;
        char line[4096];
        while (fgets(line, sizeof line, lf)) {
            line[strcspn(line, "\r\n")] = 0;
            if (*line) paths.push_back(line);
        }
        fclose(lf);

        std::vector<std::vector<uint16_t>> imgs(paths.size());
        for (size_t k = 0; k < paths.size(); k++)
            if (!read_patch(paths[k].c_str(), imgs[k])) { fprintf(stderr, "cannot read %s\n", paths[k].c_str()); return 1; }

        long alert_bytes = 0, clusters = 0;
        double best = 1e30;
        for (int r = 0; r < repeat; r++) {
            const auto t0 = std::chrono::steady_clock::now();
            for (size_t k = 0; k < paths.size(); k++) {
                detect_fire(imgs[k].data(), mask, p);
                const int nc = extract_alerts(mask, imgs[k].data(), alerts, 256);
                if (r == 0) {
                    clusters += nc;
                    alert_bytes += 8 + (long)nc * sizeof(FireAlert);   // 8-byte header per patch
                    write_mask((std::string(argv[3]) + "/" + stem(paths[k]) + ".u8").c_str(), mask);
                }
            }
            const double dt = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
            if (dt < best) best = dt;
        }
        const double raw_bytes = (double)PATCH_BYTES * paths.size();
        printf("patches: %zu | clusters: %ld | best time: %.3f ms/patch (%.1f Mpixel/s)\n",
               paths.size(), clusters, best / paths.size() * 1e3,
               paths.size() * (double)PATCH_H * PATCH_W / best / 1e6);
        printf("downlink: raw %.1f MB -> alerts %ld bytes (%.0fx smaller)\n",
               raw_bytes / 1e6, alert_bytes, raw_bytes / alert_bytes);
        return 0;
    }
    fprintf(stderr, "unknown command %s\n", argv[1]);
    return 2;
}
