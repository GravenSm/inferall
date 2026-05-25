#ifndef VA_CAPTURE_H
#define VA_CAPTURE_H

#include "util/types.h"

typedef struct {
    int fps;
    int src_width;
    int src_height;
} capture_config_t;

int  capture_init(const capture_config_t *cfg);
int  capture_frame(uint8_t *gray_out, uint16_t *depth_out, uint64_t *timestamp_us);
void capture_shutdown(void);

#endif
