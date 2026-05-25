#ifndef VA_FSM_H
#define VA_FSM_H

#include "util/types.h"

typedef struct {
    agent_state_t state;
    uint64_t      state_entered_us;
    uint64_t      last_target_seen_us;
    int           avoid_direction;
} fsm_context_t;

void       fsm_init(fsm_context_t *ctx);
void       fsm_step(fsm_context_t *ctx, const occupancy_grid_t *grid,
                     const uint8_t *detections, uint64_t now_us);
motor_cmd_t fsm_command(const fsm_context_t *ctx, const occupancy_grid_t *grid);

#endif
