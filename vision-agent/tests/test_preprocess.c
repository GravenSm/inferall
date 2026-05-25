#include "preprocess/preprocess.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static int test_downsample_gray_identity(void) {
    uint8_t src[64 * 48];
    uint8_t dst[64 * 48];
    for (int i = 0; i < 64 * 48; i++) src[i] = (uint8_t)(i & 0xFF);
    preprocess_downsample_gray(src, dst, 64, 48, 64, 48);
    for (int i = 0; i < 64 * 48; i++) {
        if (src[i] != dst[i]) {
            fprintf(stderr, "FAIL: identity mismatch at %d: %d != %d\n", i, src[i], dst[i]);
            return 1;
        }
    }
    return 0;
}

static int test_downsample_gray_10x(void) {
    uint8_t *src = calloc(640 * 480, 1);
    uint8_t dst[64 * 48];

    for (int y = 0; y < 480; y++)
        for (int x = 0; x < 640; x++)
            src[y * 640 + x] = (uint8_t)((x + y) & 0xFF);

    preprocess_downsample_gray(src, dst, 640, 480, 64, 48);

    for (int y = 0; y < 48; y++) {
        for (int x = 0; x < 64; x++) {
            uint8_t expected = src[(y * 10) * 640 + (x * 10)];
            if (dst[y * 64 + x] != expected) {
                fprintf(stderr, "FAIL: 10x at (%d,%d): got %d, expected %d\n",
                        x, y, dst[y * 64 + x], expected);
                free(src);
                return 1;
            }
        }
    }
    free(src);
    return 0;
}

static int test_downsample_depth(void) {
    uint16_t *src = calloc(640 * 480, sizeof(uint16_t));
    uint16_t dst[64 * 48];

    for (int i = 0; i < 640 * 480; i++)
        src[i] = (uint16_t)(i % 3000);

    preprocess_downsample_depth(src, dst, 640, 480, 64, 48);

    for (int y = 0; y < 48; y++) {
        for (int x = 0; x < 64; x++) {
            uint16_t expected = src[(y * 10) * 640 + (x * 10)];
            if (dst[y * 64 + x] != expected) {
                fprintf(stderr, "FAIL: depth at (%d,%d): got %d, expected %d\n",
                        x, y, dst[y * 64 + x], expected);
                free(src);
                return 1;
            }
        }
    }
    free(src);
    return 0;
}

static int test_preprocess_frame(void) {
    uint8_t *raw_gray = calloc(640 * 480, 1);
    uint16_t *raw_depth = calloc(640 * 480, sizeof(uint16_t));
    frame_t frame;

    for (int i = 0; i < 640 * 480; i++) {
        raw_gray[i] = 128;
        raw_depth[i] = 500;
    }

    preprocess_frame(raw_gray, raw_depth, 640, 480, &frame);

    if (frame.width != 64 || frame.height != 48) {
        fprintf(stderr, "FAIL: frame dims %dx%d\n", frame.width, frame.height);
        free(raw_gray); free(raw_depth);
        return 1;
    }

    for (int i = 0; i < VA_FRAME_PIXELS; i++) {
        if (frame.gray[i] != 128 || frame.depth[i] != 500) {
            fprintf(stderr, "FAIL: frame data at %d\n", i);
            free(raw_gray); free(raw_depth);
            return 1;
        }
    }

    free(raw_gray);
    free(raw_depth);
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"downsample_gray_identity", test_downsample_gray_identity},
        {"downsample_gray_10x",      test_downsample_gray_10x},
        {"downsample_depth",         test_downsample_depth},
        {"preprocess_frame",         test_preprocess_frame},
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
