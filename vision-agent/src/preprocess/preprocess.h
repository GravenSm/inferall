#ifndef VA_PREPROCESS_H
#define VA_PREPROCESS_H

#include "util/types.h"

void preprocess_downsample_gray(const uint8_t *src, uint8_t *dst,
                                int src_w, int src_h, int dst_w, int dst_h);

void preprocess_downsample_depth(const uint16_t *src, uint16_t *dst,
                                 int src_w, int src_h, int dst_w, int dst_h);

void preprocess_frame(const uint8_t *raw_gray, const uint16_t *raw_depth,
                      int src_w, int src_h, frame_t *out);

#endif
