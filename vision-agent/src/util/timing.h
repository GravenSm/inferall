#ifndef VA_TIMING_H
#define VA_TIMING_H

#include <stdint.h>

uint64_t timing_now_us(void);
uint64_t timing_elapsed_us(uint64_t start);

#endif
