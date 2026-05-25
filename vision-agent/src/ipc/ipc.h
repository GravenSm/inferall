#ifndef VA_IPC_H
#define VA_IPC_H

#include "util/types.h"

int  ipc_publisher_init(const char *shm_name, vision_state_t **state_out);
void ipc_publish(vision_state_t *state, const frame_t *frame,
                 const occupancy_grid_t *grid, const uint8_t *detections,
                 agent_state_t agent_state, const motor_cmd_t *cmd,
                 uint64_t frame_id);
void ipc_publisher_shutdown(const char *shm_name, vision_state_t *state);

vision_state_t *ipc_client_connect(const char *shm_name);
bool             ipc_client_read(const vision_state_t *shm, vision_state_t *local);
void             ipc_client_disconnect(vision_state_t *shm);

#endif
