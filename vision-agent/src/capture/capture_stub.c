#include "capture/capture.h"
#include "util/timing.h"
#include <string.h>
#include <stdlib.h>

static capture_config_t s_cfg;

int capture_init(const capture_config_t *cfg) {
    s_cfg = *cfg;
    srand(42);
    return 0;
}

int capture_frame(uint8_t *gray_out, uint16_t *depth_out, uint64_t *timestamp_us) {
    int pixels = s_cfg.src_width * s_cfg.src_height;
    for (int i = 0; i < pixels; i++) {
        gray_out[i] = (uint8_t)(rand() & 0xFF);
        depth_out[i] = (uint16_t)(200 + (rand() % 2000));
    }
    *timestamp_us = timing_now_us();
    return 0;
}

int capture_frame_stereo(uint8_t *left_out, uint8_t *right_out, uint64_t *timestamp_us) {
    int pixels = s_cfg.src_width * s_cfg.src_height;
    for (int i = 0; i < pixels; i++) {
        uint8_t val = (uint8_t)(rand() & 0xFF);
        left_out[i] = val;
        int shift = (i % s_cfg.src_width) > 40 ? 3 : 0;
        int ri = i - shift;
        right_out[i] = ri >= 0 ? left_out[ri] : val;
    }
    *timestamp_us = timing_now_us();
    return 0;
}

void capture_shutdown(void) {
}
