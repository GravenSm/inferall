#include "agent/fsm.h"
#include "fusion/occupancy.h"
#include <string.h>

/* All distances are true millimetres (detections[] carries mm). */
#define OBSTACLE_CLOSE_MM   300   /* EXPLORE -> AVOID when blocked within this range */
#define OBSTACLE_CLEAR_MM   500   /* AVOID -> EXPLORE only once clear out to this range */
#define EMERGENCY_STOP_MM   100
#define CRUISE_SPEED        400
#define AVOID_TURN_RATE     600
#define OCCUPANCY_THRESHOLD 100
#define BLOCKED_CELL_COUNT  4     /* more than this many occupied cells = blocked */

/* occupancy_update discards points beyond the grid, so a range check longer
 * than the grid would be silently shortened to the grid depth. */
_Static_assert(OBSTACLE_CLEAR_MM <= VA_GRID_Z * VA_CELL_SIZE_MM,
               "occupancy grid is shallower than OBSTACLE_CLEAR_MM");
_Static_assert(OBSTACLE_CLOSE_MM < OBSTACLE_CLEAR_MM,
               "hysteresis requires CLOSE < CLEAR");

void fsm_init(fsm_context_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->state = STATE_EXPLORE;
}

/* Number of depth cells (grid z) covered by distance_mm, clamped to the grid. */
static int range_cells(const occupancy_grid_t *grid, int distance_mm) {
    float cells = ((float)distance_mm * 0.001f) / grid->cell_size_m;
    int n = (int)cells;
    if ((float)n < cells) n++;           /* ceil */
    if (n < 1) n = 1;
    if (n > VA_GRID_Z) n = VA_GRID_Z;
    return n;
}

/* Blocked if more than BLOCKED_CELL_COUNT occupied cells lie in the forward
 * cone within distance_mm. Because the CLEAR range contains the CLOSE range,
 * blocked(CLEAR) >= blocked(CLOSE) always holds, which is what gives the
 * EXPLORE/AVOID transitions their hysteresis. */
static int forward_blocked(const occupancy_grid_t *grid, int distance_mm) {
    int center = VA_GRID_X / 2;
    int max_z = range_cells(grid, distance_mm);
    return occupancy_check_cone(grid, center, center, 2, max_z,
                                OCCUPANCY_THRESHOLD) > BLOCKED_CELL_COUNT;
}

static int find_clear_direction(const occupancy_grid_t *grid) {
    int left_count = 0, right_count = 0;
    int mid = VA_GRID_X / 2;

    for (int x = 0; x < mid; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++)
                if (grid->cells[x][y][z] >= OCCUPANCY_THRESHOLD)
                    left_count++;

    for (int x = mid; x < VA_GRID_X; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++)
                if (grid->cells[x][y][z] >= OCCUPANCY_THRESHOLD)
                    right_count++;

    return left_count <= right_count ? -1 : 1;
}

/* No valid depth in the centre region means we are blind, which is treated
 * exactly like an obstacle at point-blank range. */
static int emergency_obstacle(const uint16_t *detections) {
    uint16_t d = detections[VA_DET_CENTER];
    return d == VA_DEPTH_INVALID || d < EMERGENCY_STOP_MM;
}

void fsm_step(fsm_context_t *ctx, const occupancy_grid_t *grid,
              const uint16_t *detections, uint64_t now_us) {
    if (emergency_obstacle(detections)) {
        if (ctx->state != STATE_STOP) {
            ctx->state = STATE_STOP;
            ctx->state_entered_us = now_us;
        }
        return;
    }

    switch (ctx->state) {
    case STATE_STOP:
        /* Emergency condition has cleared (checked above). */
        ctx->state = STATE_EXPLORE;
        ctx->state_entered_us = now_us;
        break;

    case STATE_EXPLORE:
        if (forward_blocked(grid, OBSTACLE_CLOSE_MM)) {
            ctx->state = STATE_AVOID;
            ctx->state_entered_us = now_us;
            ctx->avoid_direction = find_clear_direction(grid);
        }
        break;

    case STATE_AVOID:
        if (!forward_blocked(grid, OBSTACLE_CLEAR_MM)) {
            ctx->state = STATE_EXPLORE;
            ctx->state_entered_us = now_us;
        }
        break;
    }
}

motor_cmd_t fsm_command(const fsm_context_t *ctx, const occupancy_grid_t *grid) {
    motor_cmd_t cmd = {0, 0};
    (void)grid;

    switch (ctx->state) {
    case STATE_EXPLORE:
        cmd.speed = CRUISE_SPEED;
        cmd.turn = 0;
        break;

    case STATE_AVOID:
        cmd.speed = CRUISE_SPEED / 4;
        cmd.turn = ctx->avoid_direction * AVOID_TURN_RATE;
        break;

    case STATE_STOP:
        cmd.speed = 0;
        cmd.turn = 0;
        break;
    }

    return cmd;
}
