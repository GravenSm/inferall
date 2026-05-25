#ifndef VA_PERCEPTION_H
#define VA_PERCEPTION_H

#include "util/types.h"

int  perception_init(void);
void perception_extract(const frame_t *frame, uint8_t *detections);
void perception_shutdown(void);

#endif
