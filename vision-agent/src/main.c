#include "capture/capture.h"
#include "preprocess/preprocess.h"
#include "perception/perception.h"
#include "fusion/occupancy.h"
#include "agent/fsm.h"
#include "ipc/ipc.h"
#include "hal/motor.h"
#include "util/types.h"
#include "util/timing.h"
#include "util/log.h"

#include <signal.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

static volatile sig_atomic_t running = 1;

static void handle_signal(int sig) {
    (void)sig;
    running = 0;
}

static void print_usage(const char *prog) {
    fprintf(stderr, "Usage: %s [--bench] [--dry-run] [--fps N] [--frames N]\n", prog);
}

int main(int argc, char **argv) {
    int bench_mode = 0;
    int dry_run = 0;
    int target_fps = 2;
    int max_frames = 0;

    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--bench") == 0) bench_mode = 1;
        else if (strcmp(argv[i], "--dry-run") == 0) dry_run = 1;
        else if (strcmp(argv[i], "--fps") == 0 && i + 1 < argc) target_fps = atoi(argv[++i]);
        else if (strcmp(argv[i], "--frames") == 0 && i + 1 < argc) max_frames = atoi(argv[++i]);
        else if (strcmp(argv[i], "--help") == 0) { print_usage(argv[0]); return 0; }
    }

    signal(SIGINT, handle_signal);
    signal(SIGTERM, handle_signal);

    capture_config_t cap_cfg = { .fps = target_fps, .src_width = VA_SRC_W, .src_height = VA_SRC_H };
    if (capture_init(&cap_cfg) != 0) { VA_ERR("capture init failed"); return 1; }
    if (perception_init() != 0)      { VA_ERR("perception init failed"); return 1; }
    if (!dry_run && motor_init() != 0) { VA_ERR("motor init failed"); return 1; }

    occupancy_grid_t grid;
    occupancy_init(&grid, 0.05f);

    camera_intrinsics_t intrinsics = { .fx = 386.0f, .fy = 386.0f, .cx = 32.0f, .cy = 24.0f };

    fsm_context_t fsm;
    fsm_init(&fsm);

    vision_state_t *ipc_state = NULL;
    if (ipc_publisher_init(VA_SHM_NAME, &ipc_state) != 0)
        VA_WARN("IPC init failed, continuing without shared memory");

    uint8_t *raw_gray = malloc(VA_SRC_W * VA_SRC_H);
    uint16_t *raw_depth = malloc(VA_SRC_W * VA_SRC_H * sizeof(uint16_t));
    if (!raw_gray || !raw_depth) { VA_ERR("alloc failed"); return 1; }

    frame_t frame;
    uint8_t detections[VA_NUM_DETECTIONS];
    uint64_t frame_id = 0;

    uint64_t total_capture_us = 0, total_preprocess_us = 0;
    uint64_t total_perception_us = 0, total_occupancy_us = 0;
    uint64_t total_fsm_us = 0;

    VA_LOG("vision-agent started (fps=%d, dry_run=%d, bench=%d)", target_fps, dry_run, bench_mode);

    while (running) {
        if (max_frames > 0 && (int)frame_id >= max_frames) break;

        uint64_t t0 = timing_now_us();

        uint64_t ts;
        capture_frame(raw_gray, raw_depth, &ts);
        uint64_t t1 = timing_now_us();

        preprocess_frame(raw_gray, raw_depth, VA_SRC_W, VA_SRC_H, &frame);
        frame.timestamp_us = ts;
        uint64_t t2 = timing_now_us();

        perception_extract(&frame, detections);
        uint64_t t3 = timing_now_us();

        occupancy_decay(&grid, 5);
        occupancy_update(&grid, &frame, &intrinsics);
        uint64_t t4 = timing_now_us();

        fsm_step(&fsm, &grid, detections, ts);
        motor_cmd_t cmd = fsm_command(&fsm, &grid);
        uint64_t t5 = timing_now_us();

        if (!dry_run) motor_send(&cmd);

        if (ipc_state)
            ipc_publish(ipc_state, &frame, &grid, detections,
                        fsm.state, &cmd, frame_id);

        total_capture_us    += t1 - t0;
        total_preprocess_us += t2 - t1;
        total_perception_us += t3 - t2;
        total_occupancy_us  += t4 - t3;
        total_fsm_us        += t5 - t4;
        frame_id++;

        if (bench_mode && frame_id % 100 == 0) {
            printf("--- %lu frames ---\n", (unsigned long)frame_id);
            printf("  capture:    %lu us avg\n", (unsigned long)(total_capture_us / frame_id));
            printf("  preprocess: %lu us avg\n", (unsigned long)(total_preprocess_us / frame_id));
            printf("  perception: %lu us avg\n", (unsigned long)(total_perception_us / frame_id));
            printf("  occupancy:  %lu us avg\n", (unsigned long)(total_occupancy_us / frame_id));
            printf("  fsm+motor:  %lu us avg\n", (unsigned long)(total_fsm_us / frame_id));
            printf("  total:      %lu us avg\n", (unsigned long)(
                (total_capture_us + total_preprocess_us + total_perception_us +
                 total_occupancy_us + total_fsm_us) / frame_id));
        }
    }

    if (bench_mode && frame_id > 0) {
        printf("\n=== Final (%lu frames) ===\n", (unsigned long)frame_id);
        printf("  capture:    %lu us avg\n", (unsigned long)(total_capture_us / frame_id));
        printf("  preprocess: %lu us avg\n", (unsigned long)(total_preprocess_us / frame_id));
        printf("  perception: %lu us avg\n", (unsigned long)(total_perception_us / frame_id));
        printf("  occupancy:  %lu us avg\n", (unsigned long)(total_occupancy_us / frame_id));
        printf("  fsm+motor:  %lu us avg\n", (unsigned long)(total_fsm_us / frame_id));
        uint64_t total = total_capture_us + total_preprocess_us + total_perception_us +
                         total_occupancy_us + total_fsm_us;
        printf("  total:      %lu us avg\n", (unsigned long)(total / frame_id));
        printf("  throughput: %lu FPS theoretical\n",
               (unsigned long)(frame_id > 0 && total > 0 ? (frame_id * 1000000ULL / total) : 0));
    }

    motor_stop();
    motor_shutdown();
    perception_shutdown();
    capture_shutdown();
    if (ipc_state) ipc_publisher_shutdown(VA_SHM_NAME, ipc_state);
    free(raw_gray);
    free(raw_depth);

    VA_LOG("vision-agent stopped after %lu frames", (unsigned long)frame_id);
    return 0;
}
