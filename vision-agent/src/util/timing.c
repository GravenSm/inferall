#include "util/timing.h"
#include <time.h>

uint64_t timing_now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000ULL + (uint64_t)ts.tv_nsec / 1000ULL;
}

uint64_t timing_elapsed_us(uint64_t start) {
    return timing_now_us() - start;
}
