#include "preprocess/preprocess.h"

void preprocess_downsample_gray(const uint8_t *src, uint8_t *dst,
                                int src_w, int src_h, int dst_w, int dst_h) {
    int sx = src_w / dst_w;
    int sy = src_h / dst_h;
    for (int y = 0; y < dst_h; y++)
        for (int x = 0; x < dst_w; x++)
            dst[y * dst_w + x] = src[(y * sy) * src_w + (x * sx)];
}

void preprocess_downsample_depth(const uint16_t *src, uint16_t *dst,
                                 int src_w, int src_h, int dst_w, int dst_h) {
    int sx = src_w / dst_w;
    int sy = src_h / dst_h;
    for (int y = 0; y < dst_h; y++)
        for (int x = 0; x < dst_w; x++)
            dst[y * dst_w + x] = src[(y * sy) * src_w + (x * sx)];
}

void preprocess_frame(const uint8_t *raw_gray, const uint16_t *raw_depth,
                      int src_w, int src_h, frame_t *out) {
    out->width = VA_FRAME_W;
    out->height = VA_FRAME_H;
    preprocess_downsample_gray(raw_gray, out->gray, src_w, src_h,
                               VA_FRAME_W, VA_FRAME_H);
    preprocess_downsample_depth(raw_depth, out->depth, src_w, src_h,
                                VA_FRAME_W, VA_FRAME_H);
}
