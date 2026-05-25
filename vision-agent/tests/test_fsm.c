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

    uint8_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 5;

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

    uint8_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 5;
    fsm_step(&ctx, &grid, detections, 1000000);

    if (ctx.state != STATE_STOP) {
        fprintf(stderr, "FAIL: should be STOP\n");
        return 1;
    }

    detections[0] = 200;
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

    uint8_t detections[VA_NUM_DETECTIONS] = {0};
    detections[0] = 200;

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

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"init_state",              test_init_state},
        {"explore_command",         test_explore_command},
        {"emergency_stop",          test_emergency_stop},
        {"stop_to_explore_recovery", test_stop_to_explore_recovery},
        {"avoid_turns",             test_avoid_turns},
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
