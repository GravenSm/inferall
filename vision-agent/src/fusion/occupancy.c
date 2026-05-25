#include "fusion/occupancy.h"
#include <string.h>

void occupancy_init(occupancy_grid_t *grid, float cell_size_m) {
    memset(grid->cells, 0, sizeof(grid->cells));
    grid->cell_size_m = cell_size_m;
    grid->origin[0] = -(VA_GRID_X / 2.0f) * cell_size_m;
    grid->origin[1] = -(VA_GRID_Y / 2.0f) * cell_size_m;
    grid->origin[2] = 0.0f;
}

void occupancy_update(occupancy_grid_t *grid, const frame_t *frame,
                      const camera_intrinsics_t *intrinsics) {
    float inv_fx = 1.0f / intrinsics->fx;
    float inv_fy = 1.0f / intrinsics->fy;
    float cell_inv = 1.0f / grid->cell_size_m;

    int half_gx = VA_GRID_X / 2;
    int half_gy = VA_GRID_Y / 2;

    for (int py = 0; py < frame->height; py++) {
        for (int px = 0; px < frame->width; px++) {
            uint16_t d_mm = frame->depth[py * frame->width + px];
            if (d_mm == 0) continue;

            float z = d_mm * 0.001f;
            float x = (px - intrinsics->cx) * z * inv_fx;
            float y = (py - intrinsics->cy) * z * inv_fy;

            int gx = (int)(x * cell_inv) + half_gx;
            int gy = (int)(y * cell_inv) + half_gy;
            int gz = (int)(z * cell_inv);

            if (gx < 0 || gx >= VA_GRID_X) continue;
            if (gy < 0 || gy >= VA_GRID_Y) continue;
            if (gz < 0 || gz >= VA_GRID_Z) continue;

            uint8_t val = grid->cells[gx][gy][gz];
            if (val < 240) val += 15;
            grid->cells[gx][gy][gz] = val;
        }
    }
}

void occupancy_decay(occupancy_grid_t *grid, uint8_t decay_amount) {
    for (int x = 0; x < VA_GRID_X; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++) {
                uint8_t v = grid->cells[x][y][z];
                grid->cells[x][y][z] = v > decay_amount ? v - decay_amount : 0;
            }
}

int occupancy_check_cone(const occupancy_grid_t *grid,
                         int center_x, int center_y,
                         int radius, int threshold) {
    int count = 0;
    for (int dx = -radius; dx <= radius; dx++) {
        for (int dy = -radius; dy <= radius; dy++) {
            int gx = center_x + dx;
            int gy = center_y + dy;
            if (gx < 0 || gx >= VA_GRID_X) continue;
            if (gy < 0 || gy >= VA_GRID_Y) continue;
            for (int gz = 0; gz < VA_GRID_Z; gz++) {
                if (grid->cells[gx][gy][gz] >= (uint8_t)threshold)
                    count++;
            }
        }
    }
    return count;
}
