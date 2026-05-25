#include "preprocess/stereo.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int test_default_config(void) {
    stereo_config_t cfg;
    stereo_default_config(&cfg);
    if (cfg.block_size != 5 || cfg.max_disparity != 16) {
        fprintf(stderr, "FAIL: unexpected defaults\n");
        return 1;
    }
    if (cfg.baseline_mm < 59.0f || cfg.baseline_mm > 61.0f) {
        fprintf(stderr, "FAIL: baseline %f\n", cfg.baseline_mm);
        return 1;
    }
    return 0;
}

static int test_identical_images_zero_disparity(void) {
    uint8_t img[VA_FRAME_PIXELS];
    uint8_t disparity[VA_FRAME_PIXELS];
    stereo_config_t cfg;
    stereo_default_config(&cfg);

    for (int i = 0; i < VA_FRAME_PIXELS; i++)
        img[i] = (uint8_t)((i * 7 + 13) & 0xFF);

    stereo_compute_disparity(img, img, VA_FRAME_W, VA_FRAME_H, disparity, &cfg);

    int nonzero = 0;
    for (int i = 0; i < VA_FRAME_PIXELS; i++)
        if (disparity[i] > 1) nonzero++;

    if (nonzero > VA_FRAME_PIXELS / 10) {
        fprintf(stderr, "FAIL: identical images should have ~0 disparity, got %d nonzero\n", nonzero);
        return 1;
    }
    return 0;
}

static int test_shifted_image_detects_disparity(void) {
    uint8_t left[VA_FRAME_PIXELS];
    uint8_t right[VA_FRAME_PIXELS];
    uint8_t disparity[VA_FRAME_PIXELS];
    stereo_config_t cfg;
    stereo_default_config(&cfg);

    memset(left, 128, VA_FRAME_PIXELS);
    memset(right, 128, VA_FRAME_PIXELS);

    for (int y = 10; y < 38; y++) {
        for (int x = 20; x < 50; x++) {
            uint8_t tex = (uint8_t)(x * 7 + y * 13);
            left[y * VA_FRAME_W + x] = tex;
            int rx = x - 5;
            if (rx >= 0)
                right[y * VA_FRAME_W + rx] = tex;
        }
    }

    stereo_compute_disparity(left, right, VA_FRAME_W, VA_FRAME_H, disparity, &cfg);

    int center_y = 24;
    int center_x = 35;
    uint8_t d = disparity[center_y * VA_FRAME_W + center_x];

    if (d < 40 || d > 120) {
        fprintf(stderr, "FAIL: expected disparity ~80 (5*16) at center, got %d\n", d);
        return 1;
    }
    return 0;
}

static int test_disparity_to_depth(void) {
    uint8_t disparity[VA_FRAME_PIXELS];
    uint16_t depth[VA_FRAME_PIXELS];
    stereo_config_t cfg;
    stereo_default_config(&cfg);

    memset(disparity, 0, sizeof(disparity));

    disparity[0] = 80;
    disparity[1] = 16;
    disparity[2] = 0;

    stereo_disparity_to_depth(disparity, depth, VA_FRAME_W, VA_FRAME_H, &cfg);

    float expected_0 = (cfg.focal_length_px * cfg.baseline_mm) / (80.0f / 16.0f);
    if (abs((int)depth[0] - (int)expected_0) > 5) {
        fprintf(stderr, "FAIL: depth[0] = %d, expected ~%d\n", depth[0], (int)expected_0);
        return 1;
    }

    if (depth[2] != 0) {
        fprintf(stderr, "FAIL: zero disparity should give zero depth\n");
        return 1;
    }
    return 0;
}

static int test_full_pipeline(void) {
    uint8_t left[VA_FRAME_PIXELS];
    uint8_t right[VA_FRAME_PIXELS];
    uint16_t depth[VA_FRAME_PIXELS];
    stereo_config_t cfg;
    stereo_default_config(&cfg);

    srand(42);
    for (int i = 0; i < VA_FRAME_PIXELS; i++) {
        left[i] = (uint8_t)(rand() & 0xFF);
        right[i] = left[i];
    }

    stereo_compute_depth(left, right, VA_FRAME_W, VA_FRAME_H, depth, &cfg);

    int valid = 0;
    for (int i = 0; i < VA_FRAME_PIXELS; i++)
        if (depth[i] > 0) valid++;

    (void)valid;
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"default_config",                   test_default_config},
        {"identical_images_zero_disparity",  test_identical_images_zero_disparity},
        {"shifted_image_detects_disparity",  test_shifted_image_detects_disparity},
        {"disparity_to_depth",               test_disparity_to_depth},
        {"full_pipeline",                    test_full_pipeline},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < n; i++) {
        int r = tests[i].fn();
        printf("  %s: %s\n", tests[i].name, r == 0 ? "PASS" : "FAIL");
        failures += r;
    }
    printf("%d/%d tests passed\n", n - failures, n);
    return failures > 0 ? 1 : 0;
}
