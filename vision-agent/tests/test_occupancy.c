#include "fusion/occupancy.h"
#include <stdio.h>
#include <string.h>

static int test_init_clears_grid(void) {
    occupancy_grid_t grid;
    memset(grid.cells, 0xFF, sizeof(grid.cells));
    occupancy_init(&grid, 0.05f);

    for (int x = 0; x < VA_GRID_X; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++)
                if (grid.cells[x][y][z] != 0) {
                    fprintf(stderr, "FAIL: cell[%d][%d][%d] = %d\n",
                            x, y, z, grid.cells[x][y][z]);
                    return 1;
                }

    if (grid.cell_size_m != 0.05f) {
        fprintf(stderr, "FAIL: cell_size %f\n", grid.cell_size_m);
        return 1;
    }
    return 0;
}

static int test_update_projects_depth(void) {
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    camera_intrinsics_t intr = { .fx = 38.6f, .fy = 38.6f, .cx = 32.0f, .cy = 24.0f };

    frame_t frame;
    memset(&frame, 0, sizeof(frame));
    frame.width = VA_FRAME_W;
    frame.height = VA_FRAME_H;

    int cx = VA_FRAME_W / 2;
    int cy = VA_FRAME_H / 2;
    frame.depth[cy * VA_FRAME_W + cx] = 200;

    occupancy_update(&grid, &frame, &intr);

    int any_occupied = 0;
    for (int x = 0; x < VA_GRID_X; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++)
                if (grid.cells[x][y][z] > 0)
                    any_occupied = 1;

    if (!any_occupied) {
        fprintf(stderr, "FAIL: no cells occupied after update\n");
        return 1;
    }
    return 0;
}

static int test_decay_reduces_values(void) {
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    grid.cells[5][5][3] = 100;
    grid.cells[8][8][1] = 20;

    occupancy_decay(&grid, 15);

    if (grid.cells[5][5][3] != 85) {
        fprintf(stderr, "FAIL: expected 85, got %d\n", grid.cells[5][5][3]);
        return 1;
    }
    if (grid.cells[8][8][1] != 5) {
        fprintf(stderr, "FAIL: expected 5, got %d\n", grid.cells[8][8][1]);
        return 1;
    }
    return 0;
}

static int test_decay_clamps_to_zero(void) {
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    grid.cells[0][0][0] = 3;
    occupancy_decay(&grid, 10);

    if (grid.cells[0][0][0] != 0) {
        fprintf(stderr, "FAIL: expected 0, got %d\n", grid.cells[0][0][0]);
        return 1;
    }
    return 0;
}

static int test_check_cone(void) {
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    grid.cells[8][8][0] = 200;
    grid.cells[8][8][1] = 200;
    grid.cells[7][8][0] = 200;

    int count = occupancy_check_cone(&grid, 8, 8, 1, VA_GRID_Z, 100);
    if (count != 3) {
        fprintf(stderr, "FAIL: expected 3 occupied, got %d\n", count);
        return 1;
    }

    count = occupancy_check_cone(&grid, 0, 0, 1, VA_GRID_Z, 100);
    if (count != 0) {
        fprintf(stderr, "FAIL: expected 0 at origin, got %d\n", count);
        return 1;
    }
    return 0;
}

/* Pixels just left and just right of the principal point must land in
 * different grid columns: truncation toward zero would merge them. */
static int test_update_floor_indexing(void) {
    occupancy_grid_t grid;
    camera_intrinsics_t intr = { .fx = 38.6f, .fy = 38.6f, .cx = 32.0f, .cy = 24.0f };
    int half = VA_GRID_X / 2;
    int cy = VA_FRAME_H / 2;

    /* depth 300 mm, offset 4 px -> x = 4 * 0.3 / 38.6 = 0.031 m, i.e. 0.62 cells */
    frame_t frame;
    memset(&frame, 0, sizeof(frame));
    frame.width = VA_FRAME_W;
    frame.height = VA_FRAME_H;
    frame.depth[cy * VA_FRAME_W + (32 - 4)] = 300;
    occupancy_init(&grid, 0.05f);
    occupancy_update(&grid, &frame, &intr);

    int left_col = -1;
    for (int x = 0; x < VA_GRID_X; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++)
                if (grid.cells[x][y][z] > 0) left_col = x;

    memset(&frame.depth, 0, sizeof(frame.depth));
    frame.depth[cy * VA_FRAME_W + (32 + 4)] = 300;
    occupancy_init(&grid, 0.05f);
    occupancy_update(&grid, &frame, &intr);

    int right_col = -1;
    for (int x = 0; x < VA_GRID_X; x++)
        for (int y = 0; y < VA_GRID_Y; y++)
            for (int z = 0; z < VA_GRID_Z; z++)
                if (grid.cells[x][y][z] > 0) right_col = x;

    if (left_col != half - 1 || right_col != half) {
        fprintf(stderr, "FAIL: left px -> col %d (want %d), right px -> col %d (want %d)\n",
                left_col, half - 1, right_col, half);
        return 1;
    }
    return 0;
}

static int test_check_cone_max_z(void) {
    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    grid.cells[8][8][1] = 200;
    grid.cells[8][8][5] = 200;

    if (occupancy_check_cone(&grid, 8, 8, 0, 2, 100) != 1) {
        fprintf(stderr, "FAIL: max_z=2 should see only the near cell\n");
        return 1;
    }
    if (occupancy_check_cone(&grid, 8, 8, 0, VA_GRID_Z, 100) != 2) {
        fprintf(stderr, "FAIL: full depth should see both cells\n");
        return 1;
    }
    if (occupancy_check_cone(&grid, 8, 8, 0, VA_GRID_Z + 10, 100) != 2) {
        fprintf(stderr, "FAIL: oversized max_z must be clamped\n");
        return 1;
    }
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"init_clears_grid",     test_init_clears_grid},
        {"update_projects_depth", test_update_projects_depth},
        {"decay_reduces_values", test_decay_reduces_values},
        {"decay_clamps_to_zero", test_decay_clamps_to_zero},
        {"check_cone",           test_check_cone},
        {"update_floor_indexing", test_update_floor_indexing},
        {"check_cone_max_z",     test_check_cone_max_z},
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
