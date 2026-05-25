#include "perception/perception.h"
#include <string.h>

int perception_init(void) {
    return 0;
}

void perception_extract(const frame_t *frame, uint8_t *detections) {
    memset(detections, 0, VA_NUM_DETECTIONS);

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

    detections[0] = count_center > 0 ? (uint8_t)(sum_center / count_center / 10) : 255;
    detections[1] = count_left   > 0 ? (uint8_t)(sum_left   / count_left   / 10) : 255;
    detections[2] = count_right  > 0 ? (uint8_t)(sum_right  / count_right  / 10) : 255;
    detections[3] = count_top    > 0 ? (uint8_t)(sum_top    / count_top    / 10) : 255;

    uint32_t avg_gray = 0;
    for (int i = 0; i < VA_FRAME_PIXELS; i++)
        avg_gray += frame->gray[i];
    detections[4] = (uint8_t)(avg_gray / VA_FRAME_PIXELS);

    detections[5] = 0;
    detections[6] = 0;
    detections[7] = 0;
}

void perception_shutdown(void) {
}
