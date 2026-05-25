#ifndef VA_OCCUPANCY_H
#define VA_OCCUPANCY_H

#include "util/types.h"

typedef struct {
    float fx, fy;
    float cx, cy;
} camera_intrinsics_t;

void occupancy_init(occupancy_grid_t *grid, float cell_size_m);
void occupancy_update(occupancy_grid_t *grid, const frame_t *frame,
                      const camera_intrinsics_t *intrinsics);
void occupancy_decay(occupancy_grid_t *grid, uint8_t decay_amount);
int  occupancy_check_cone(const occupancy_grid_t *grid,
                          int center_x, int center_y,
                          int radius, int threshold);

#endif
