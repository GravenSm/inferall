#include "ipc/ipc.h"
#include "util/log.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

int ipc_publisher_init(const char *shm_name, vision_state_t **state_out) {
    int fd = shm_open(shm_name, O_CREAT | O_RDWR, 0666);
    if (fd < 0) {
        VA_ERR("shm_open failed for %s", shm_name);
        return -1;
    }

    if (ftruncate(fd, sizeof(vision_state_t)) < 0) {
        VA_ERR("ftruncate failed");
        close(fd);
        return -1;
    }

    void *ptr = mmap(NULL, sizeof(vision_state_t),
                     PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) {
        VA_ERR("mmap failed");
        return -1;
    }

    vision_state_t *state = (vision_state_t *)ptr;
    memset(state, 0, sizeof(*state));
    state->version = VA_IPC_VERSION;
    atomic_store(&state->seq, 0);
    *state_out = state;
    return 0;
}

void ipc_publish(vision_state_t *state, const frame_t *frame,
                 const occupancy_grid_t *grid, const uint8_t *detections,
                 agent_state_t agent_state, const motor_cmd_t *cmd,
                 uint64_t frame_id) {
    unsigned int seq = atomic_load(&state->seq);
    atomic_store(&state->seq, seq + 1);
    __sync_synchronize();

    state->frame_id = frame_id;
    state->timestamp_us = frame->timestamp_us;
    state->state = agent_state;
    state->last_cmd = *cmd;

    memcpy(state->occupancy, grid->cells, sizeof(grid->cells));
    memcpy(state->detections, detections, VA_NUM_DETECTIONS);
    memcpy(state->gray, frame->gray, sizeof(frame->gray));
    memcpy(state->depth, frame->depth, sizeof(frame->depth));

    __sync_synchronize();
    atomic_store(&state->seq, seq + 2);
}

void ipc_publisher_shutdown(const char *shm_name, vision_state_t *state) {
    if (state) munmap(state, sizeof(vision_state_t));
    if (shm_name) shm_unlink(shm_name);
}
