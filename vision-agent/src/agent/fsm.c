#include "agent/fsm.h"
#include "fusion/occupancy.h"
#include <string.h>

#define OBSTACLE_CLOSE_MM   30
#define OBSTACLE_CLEAR_MM   50
#define EMERGENCY_STOP_MM   10
#define TARGET_LOST_TIMEOUT_US 2000000
#define CRUISE_SPEED        400
#define AVOID_TURN_RATE     600
#define OCCUPANCY_THRESHOLD 100

void fsm_init(fsm_context_t *ctx) {
    memset(ctx, 0, sizeof(*ctx));
    ctx->state = STATE_EXPLORE;
}

static int forward_blocked(const occupancy_grid_t *grid, int distance_mm) {
    (void)distance_mm;
    int center = VA_GRID_X / 2;
    return occupancy_check_cone(grid, center, center, 2, OCCUPANCY_THRESHOLD) > 4;
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

static int emergency_obstacle(const uint8_t *detections) {
    return detections[0] < EMERGENCY_STOP_MM;
}

void fsm_step(fsm_context_t *ctx, const occupancy_grid_t *grid,
              const uint8_t *detections, uint64_t now_us) {
    if (emergency_obstacle(detections)) {
        if (ctx->state != STATE_STOP) {
            ctx->state = STATE_STOP;
            ctx->state_entered_us = now_us;
        }
        return;
    }

    switch (ctx->state) {
    case STATE_STOP:
        if (!emergency_obstacle(detections)) {
            ctx->state = STATE_EXPLORE;
            ctx->state_entered_us = now_us;
        }
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

    case STATE_TRACK:
        if ((now_us - ctx->last_target_seen_us) > TARGET_LOST_TIMEOUT_US) {
            ctx->state = STATE_EXPLORE;
            ctx->state_entered_us = now_us;
        } else if (forward_blocked(grid, OBSTACLE_CLOSE_MM)) {
            ctx->state = STATE_AVOID;
            ctx->state_entered_us = now_us;
            ctx->avoid_direction = find_clear_direction(grid);
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

    case STATE_TRACK:
        cmd.speed = CRUISE_SPEED / 2;
        cmd.turn = 0;
        break;

    case STATE_STOP:
        cmd.speed = 0;
        cmd.turn = 0;
        break;
    }

    return cmd;
}
