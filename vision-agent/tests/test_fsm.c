#include "agent/fsm.h"
#include "fusion/occupancy.h"
#include <stdio.h>
#include <string.h>

static int test_init_state(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);
    if (ctx.state != STATE_EXPLORE) {
        fprintf(stderr, "FAIL: initial state %d, expected EXPLORE\n", ctx.state);
        return 1;
    }
    return 0;
}

static int test_explore_command(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);

    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    motor_cmd_t cmd = fsm_command(&ctx, &grid);
    if (cmd.speed <= 0) {
        fprintf(stderr, "FAIL: explore speed should be positive, got %d\n", cmd.speed);
        return 1;
    }
    if (cmd.turn != 0) {
        fprintf(stderr, "FAIL: explore turn should be 0, got %d\n", cmd.turn);
        return 1;
    }
    return 0;
}

static int test_emergency_stop(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);

    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    uint16_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 50;   /* 50 mm: inside the emergency range */

    fsm_step(&ctx, &grid, detections, 1000000);

    if (ctx.state != STATE_STOP) {
        fprintf(stderr, "FAIL: expected STOP, got %d\n", ctx.state);
        return 1;
    }

    motor_cmd_t cmd = fsm_command(&ctx, &grid);
    if (cmd.speed != 0 || cmd.turn != 0) {
        fprintf(stderr, "FAIL: STOP should have zero speed/turn\n");
        return 1;
    }
    return 0;
}

static int test_stop_to_explore_recovery(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);

    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    uint16_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 50;
    fsm_step(&ctx, &grid, detections, 1000000);

    if (ctx.state != STATE_STOP) {
        fprintf(stderr, "FAIL: should be STOP\n");
        return 1;
    }

    detections[0] = 2000;
    fsm_step(&ctx, &grid, detections, 2000000);

    if (ctx.state != STATE_EXPLORE) {
        fprintf(stderr, "FAIL: should recover to EXPLORE, got %d\n", ctx.state);
        return 1;
    }
    return 0;
}

static int test_avoid_turns(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);

    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    int center = VA_GRID_X / 2;
    for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++)
            for (int z = 0; z < VA_GRID_Z; z++) {
                int gx = center + dx, gy = center + dy;
                if (gx >= 0 && gx < VA_GRID_X && gy >= 0 && gy < VA_GRID_Y)
                    grid.cells[gx][gy][z] = 200;
            }

    uint16_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 2000;

    fsm_step(&ctx, &grid, detections, 1000000);

    if (ctx.state != STATE_AVOID) {
        fprintf(stderr, "FAIL: expected AVOID, got %d\n", ctx.state);
        return 1;
    }

    motor_cmd_t cmd = fsm_command(&ctx, &grid);
    if (cmd.turn == 0) {
        fprintf(stderr, "FAIL: AVOID should have non-zero turn\n");
        return 1;
    }
    return 0;
}

/* Far-away average depth used to wrap modulo 256 and read as an emergency. */
static int test_far_depth_is_not_emergency(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    uint16_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 2600;
    fsm_step(&ctx, &grid, detections, 1000000);
    if (ctx.state != STATE_EXPLORE) {
        fprintf(stderr, "FAIL: 2600 mm should not stop, got %d\n", ctx.state);
        return 1;
    }
    return 0;
}

/* A blind sensor (no valid centre depth) must stop, never cruise. */
static int test_blind_sensor_stops(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    uint16_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = VA_DEPTH_INVALID;
    fsm_step(&ctx, &grid, detections, 1000000);
    if (ctx.state != STATE_STOP) {
        fprintf(stderr, "FAIL: invalid depth should STOP, got %d\n", ctx.state);
        return 1;
    }
    motor_cmd_t cmd = fsm_command(&ctx, &grid);
    if (cmd.speed != 0) {
        fprintf(stderr, "FAIL: blind robot must not move\n");
        return 1;
    }
    return 0;
}

/* An obstacle between the CLOSE and CLEAR ranges must keep AVOID latched,
 * and must not trigger AVOID from EXPLORE. */
static int test_avoid_hysteresis(void) {
    fsm_context_t ctx;
    fsm_init(&ctx);
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    uint16_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 2000;
    int c = VA_GRID_X / 2;

    /* Obstacle at depth cell 7 (0.35-0.40 m): beyond CLOSE (0.30 m), within CLEAR (0.50 m). */
    for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++)
            grid.cells[c + dx][c + dy][7] = 200;

    fsm_step(&ctx, &grid, detections, 1000000);
    if (ctx.state != STATE_EXPLORE) {
        fprintf(stderr, "FAIL: obstacle beyond CLOSE range should not trigger AVOID, got %d\n", ctx.state);
        return 1;
    }

    /* Now also occupy a near cell layer: enters AVOID. */
    for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++)
            grid.cells[c + dx][c + dy][2] = 200;
    fsm_step(&ctx, &grid, detections, 2000000);
    if (ctx.state != STATE_AVOID) {
        fprintf(stderr, "FAIL: near obstacle should trigger AVOID, got %d\n", ctx.state);
        return 1;
    }

    /* Near cells clear, far (in-CLEAR-range) cells remain: must stay in AVOID. */
    for (int dx = -2; dx <= 2; dx++)
        for (int dy = -2; dy <= 2; dy++)
            grid.cells[c + dx][c + dy][2] = 0;
    fsm_step(&ctx, &grid, detections, 3000000);
    if (ctx.state != STATE_AVOID) {
        fprintf(stderr, "FAIL: AVOID should latch until CLEAR range is empty, got %d\n", ctx.state);
        return 1;
    }

    /* Everything clear: back to EXPLORE. */
    occupancy_init(&grid, 0.05f);
    fsm_step(&ctx, &grid, detections, 4000000);
    if (ctx.state != STATE_EXPLORE) {
        fprintf(stderr, "FAIL: should return to EXPLORE when clear, got %d\n", ctx.state);
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"init_state",                 test_init_state},
        {"explore_command",            test_explore_command},
        {"emergency_stop",             test_emergency_stop},
        {"stop_to_explore_recovery",   test_stop_to_explore_recovery},
        {"avoid_turns",                test_avoid_turns},
        {"far_depth_is_not_emergency", test_far_depth_is_not_emergency},
        {"blind_sensor_stops",         test_blind_sensor_stops},
        {"avoid_hysteresis",           test_avoid_hysteresis},
    };
    int n = sizeof(tests) / sizeof(tests[0]);
    for (int i = 0; i < n; i++) {
        int r = tests[i].fn();
        printf("  %s: %s\n", tests[i].name, r == 0 ? "PASS" : "FAIL");
        failures += r;
    }
    printf("%d/%d tests passed\n", n - failures, n);
    return failures > 0 ? 1 : 0;
}
