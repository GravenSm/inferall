#include "perception/perception.h"
#include <string.h>
#include <stdint.h>

int perception_init(void) {
    return 0;
}

static uint16_t region_mean_mm(uint32_t sum, uint32_t count) {
    if (count == 0) return VA_DEPTH_INVALID;
    uint32_t mean = sum / count;
    /* A valid region can never legitimately average 0 mm; keep it distinct
     * from the VA_DEPTH_INVALID sentinel. */
    if (mean == 0) mean = 1;
    return mean > UINT16_MAX ? UINT16_MAX : (uint16_t)mean;
}

void perception_extract(const frame_t *frame, uint16_t *detections) {
    memset(detections, 0, VA_NUM_DETECTIONS * sizeof(detections[0]));

    int center_x = VA_FRAME_W / 2;
    int center_y = VA_FRAME_H / 2;
    int region_w = VA_FRAME_W / 4;
    int region_h = VA_FRAME_H / 4;

    uint32_t sum_left = 0, sum_right = 0, sum_center = 0, sum_top = 0;
    uint32_t count_left = 0, count_right = 0, count_center = 0, count_top = 0;

    for (int y = 0; y < VA_FRAME_H; y++) {
        for (int x = 0; x < VA_FRAME_W; x++) {
            uint16_t d = frame->depth[y * VA_FRAME_W + x];
            if (d == 0) continue;

            if (x < center_x - region_w) {
                sum_left += d; count_left++;
            } else if (x > center_x + region_w) {
                sum_right += d; count_right++;
            } else if (y < center_y - region_h) {
                sum_top += d; count_top++;
            } else {
                sum_center += d; count_center++;
            }
        }
    }

    detections[VA_DET_CENTER] = region_mean_mm(sum_center, count_center);
    detections[VA_DET_LEFT]   = region_mean_mm(sum_left,   count_left);
    detections[VA_DET_RIGHT]  = region_mean_mm(sum_right,  count_right);
    detections[VA_DET_TOP]    = region_mean_mm(sum_top,    count_top);

    uint32_t avg_gray = 0;
    for (int i = 0; i < VA_FRAME_PIXELS; i++)
        avg_gray += frame->gray[i];
    detections[VA_DET_GRAY] = (uint16_t)(avg_gray / VA_FRAME_PIXELS);

    detections[5] = 0;
    detections[6] = 0;
    detections[7] = 0;
}

void perception_shutdown(void) {
}
