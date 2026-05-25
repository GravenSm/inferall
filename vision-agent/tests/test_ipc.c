#include "ipc/ipc.h"
#include <stdio.h>
#include <string.h>

#define TEST_SHM_NAME "/va_test_ipc"

static int test_publish_and_read(void) {
    vision_state_t *pub = NULL;
    if (ipc_publisher_init(TEST_SHM_NAME, &pub) != 0) {
        fprintf(stderr, "FAIL: publisher init\n");
        return 1;
    }

    frame_t frame;
    memset(&frame, 0, sizeof(frame));
    frame.width = VA_FRAME_W;
    frame.height = VA_FRAME_H;
    frame.timestamp_us = 12345;
    for (int i = 0; i < VA_FRAME_PIXELS; i++) {
        frame.gray[i] = 42;
        frame.depth[i] = 500;
    }

    occupancy_grid_t grid;
    memset(&grid, 0, sizeof(grid));
    grid.cells[5][5][2] = 100;

    uint8_t detections[VA_NUM_DETECTIONS] = {10, 20, 30, 40, 50, 60, 70, 80};
    motor_cmd_t cmd = {200, -100};

    ipc_publish(pub, &frame, &grid, detections, STATE_EXPLORE, &cmd, 7);

    vision_state_t *client = ipc_client_connect(TEST_SHM_NAME);
    if (!client) {
        fprintf(stderr, "FAIL: client connect\n");
        ipc_publisher_shutdown(TEST_SHM_NAME, pub);
        return 1;
    }

    vision_state_t local;
    if (!ipc_client_read(client, &local)) {
        fprintf(stderr, "FAIL: client read\n");
        ipc_client_disconnect(client);
        ipc_publisher_shutdown(TEST_SHM_NAME, pub);
        return 1;
    }

    int fail = 0;
    if (local.frame_id != 7) {
        fprintf(stderr, "FAIL: frame_id %lu\n", (unsigned long)local.frame_id);
        fail = 1;
    }
    if (local.timestamp_us != 12345) {
        fprintf(stderr, "FAIL: timestamp %lu\n", (unsigned long)local.timestamp_us);
        fail = 1;
    }
    if (local.state != STATE_EXPLORE) {
        fprintf(stderr, "FAIL: state %d\n", local.state);
        fail = 1;
    }
    if (local.last_cmd.speed != 200 || local.last_cmd.turn != -100) {
        fprintf(stderr, "FAIL: cmd %d/%d\n", local.last_cmd.speed, local.last_cmd.turn);
        fail = 1;
    }
    if (local.gray[0] != 42) {
        fprintf(stderr, "FAIL: gray[0] = %d\n", local.gray[0]);
        fail = 1;
    }
    if (local.depth[0] != 500) {
        fprintf(stderr, "FAIL: depth[0] = %d\n", local.depth[0]);
        fail = 1;
    }
    if (local.occupancy[5][5][2] != 100) {
        fprintf(stderr, "FAIL: occupancy[5][5][2] = %d\n", local.occupancy[5][5][2]);
        fail = 1;
    }
    if (local.detections[0] != 10 || local.detections[7] != 80) {
        fprintf(stderr, "FAIL: detections\n");
        fail = 1;
    }

    ipc_client_disconnect(client);
    ipc_publisher_shutdown(TEST_SHM_NAME, pub);
    return fail;
}

static int test_seqlock_consistency(void) {
    vision_state_t *pub = NULL;
    if (ipc_publisher_init(TEST_SHM_NAME, &pub) != 0) return 1;

    frame_t frame;
    memset(&frame, 0, sizeof(frame));
    frame.width = VA_FRAME_W;
    frame.height = VA_FRAME_H;

    occupancy_grid_t grid;
    memset(&grid, 0, sizeof(grid));
    uint8_t det[VA_NUM_DETECTIONS] = {0};
    motor_cmd_t cmd = {0, 0};

    for (uint64_t i = 0; i < 100; i++) {
        frame.timestamp_us = i * 1000;
        ipc_publish(pub, &frame, &grid, det, STATE_EXPLORE, &cmd, i);
    }

    vision_state_t *client = ipc_client_connect(TEST_SHM_NAME);
    if (!client) {
        ipc_publisher_shutdown(TEST_SHM_NAME, pub);
        return 1;
    }

    vision_state_t local;
    if (!ipc_client_read(client, &local)) {
        fprintf(stderr, "FAIL: read after 100 publishes\n");
        ipc_client_disconnect(client);
        ipc_publisher_shutdown(TEST_SHM_NAME, pub);
        return 1;
    }

    if (local.frame_id != 99) {
        fprintf(stderr, "FAIL: expected frame 99, got %lu\n", (unsigned long)local.frame_id);
        ipc_client_disconnect(client);
        ipc_publisher_shutdown(TEST_SHM_NAME, pub);
        return 1;
    }

    ipc_client_disconnect(client);
    ipc_publisher_shutdown(TEST_SHM_NAME, pub);
    return 0;
}

int main(void) {
    int failures = 0;
    struct { const char *name; int (*fn)(void); } tests[] = {
        {"publish_and_read",    test_publish_and_read},
        {"seqlock_consistency", test_seqlock_consistency},
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
