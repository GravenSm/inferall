#ifndef VA_STEREO_H
#define VA_STEREO_H

#include "util/types.h"

typedef struct {
    int     block_size;
    int     max_disparity;
    float   focal_length_px;
    float   baseline_mm;
} stereo_config_t;

void stereo_default_config(stereo_config_t *cfg);

void stereo_compute_disparity(const uint8_t *left, const uint8_t *right,
                              int width, int height,
                              uint8_t *disparity_out,
                              const stereo_config_t *cfg);

void stereo_disparity_to_depth(const uint8_t *disparity,
                               uint16_t *depth_mm_out,
                               int width, int height,
                               const stereo_config_t *cfg);

void stereo_compute_depth(const uint8_t *left, const uint8_t *right,
                          int width, int height,
                          uint16_t *depth_mm_out,
                          const stereo_config_t *cfg);

#endif
