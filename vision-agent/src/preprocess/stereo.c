#include "preprocess/stereo.h"
#include <string.h>
#include <limits.h>

/*
 * IMX219-83 at 64x48 (10x downsample from 640x480):
 *   sensor width = 3.674mm, focal length = 2.6mm
 *   fx_full = 640 * 2.6 / 3.674 ≈ 453 px
 *   fx_small = 453 / 10 ≈ 45.3 px
 *   baseline = 60mm
 *   depth_mm = (45.3 * 60) / disparity ≈ 2718 / disparity
 */

void stereo_default_config(stereo_config_t *cfg) {
    cfg->block_size = 5;
    cfg->max_disparity = 16;
    cfg->focal_length_px = 45.3f;
    cfg->baseline_mm = 60.0f;
}

static inline int abs_diff(int a, int b) {
    int d = a - b;
    return d < 0 ? -d : d;
}

static uint32_t block_sad(const uint8_t *left, const uint8_t *right,
                          int lx, int rx, int y,
                          int width, int height, int half_block) {
    uint32_t sad = 0;
    for (int dy = -half_block; dy <= half_block; dy++) {
        int row = y + dy;
        if (row < 0 || row >= height) continue;
        int row_off = row * width;
        for (int dx = -half_block; dx <= half_block; dx++) {
            int lc = lx + dx;
            int rc = rx + dx;
            if (lc < 0 || lc >= width || rc < 0 || rc >= width) continue;
            sad += abs_diff(left[row_off + lc], right[row_off + rc]);
        }
    }
    return sad;
}

void stereo_compute_disparity(const uint8_t *left, const uint8_t *right,
                              int width, int height,
                              uint8_t *disparity_out,
                              const stereo_config_t *cfg) {
    int half_block = cfg->block_size / 2;
    int max_d = cfg->max_disparity;

    memset(disparity_out, 0, width * height);

    for (int y = 0; y < height; y++) {
        for (int x = max_d; x < width; x++) {
            uint32_t best_sad = UINT32_MAX;
            int best_d = 0;
            uint32_t sad_scores[3] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};

            for (int d = 0; d < max_d; d++) {
                uint32_t sad = block_sad(left, right, x, x - d, y,
                                         width, height, half_block);
                if (sad < best_sad) {
                    best_sad = sad;
                    best_d = d;
                }
            }

            if (best_d > 0 && best_d < max_d - 1) {
                sad_scores[0] = block_sad(left, right, x, x - (best_d - 1), y,
                                          width, height, half_block);
                sad_scores[1] = best_sad;
                sad_scores[2] = block_sad(left, right, x, x - (best_d + 1), y,
                                          width, height, half_block);

                int denom = (int)(sad_scores[0] + sad_scores[2]) - 2 * (int)sad_scores[1];
                if (denom > 0) {
                    int sub = (int)(sad_scores[0] - sad_scores[2]) * 8 / denom;
                    int refined = best_d * 16 + sub;
                    disparity_out[y * width + x] = (uint8_t)(refined > 255 ? 255 :
                                                             (refined < 0 ? 0 : refined));
                    continue;
                }
            }

            disparity_out[y * width + x] = (uint8_t)(best_d * 16);
        }
    }
}

void stereo_disparity_to_depth(const uint8_t *disparity,
                               uint16_t *depth_mm_out,
                               int width, int height,
                               const stereo_config_t *cfg) {
    float fb = cfg->focal_length_px * cfg->baseline_mm;

    for (int i = 0; i < width * height; i++) {
        uint8_t d = disparity[i];
        if (d == 0) {
            depth_mm_out[i] = 0;
        } else {
            float disp_real = d / 16.0f;
            float depth = fb / disp_real;
            depth_mm_out[i] = depth > 65535.0f ? 0 : (uint16_t)depth;
        }
    }
}

void stereo_compute_depth(const uint8_t *left, const uint8_t *right,
                          int width, int height,
                          uint16_t *depth_mm_out,
                          const stereo_config_t *cfg) {
    uint8_t disparity[VA_FRAME_PIXELS];
    stereo_compute_disparity(left, right, width, height, disparity, cfg);
    stereo_disparity_to_depth(disparity, depth_mm_out, width, height, cfg);
}
