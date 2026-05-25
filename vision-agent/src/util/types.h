#ifndef VA_TYPES_H
#define VA_TYPES_H

#include <stdint.h>
#include <stdbool.h>
#include <stdatomic.h>

#define VA_FRAME_W 64
#define VA_FRAME_H 48
#define VA_FRAME_PIXELS (VA_FRAME_W * VA_FRAME_H)

#define VA_SRC_W 640
#define VA_SRC_H 480

#define VA_GRID_X 16
#define VA_GRID_Y 16
#define VA_GRID_Z 8

typedef struct {
    uint8_t  gray[VA_FRAME_PIXELS];
    uint16_t depth[VA_FRAME_PIXELS];
    uint64_t timestamp_us;
    uint16_t width;
    uint16_t height;
} frame_t;

typedef enum {
    STATE_EXPLORE,
    STATE_AVOID,
    STATE_TRACK,
    STATE_STOP
} agent_state_t;

typedef struct {
    int16_t speed;
    int16_t turn;
} motor_cmd_t;

typedef struct {
    uint8_t cells[VA_GRID_X][VA_GRID_Y][VA_GRID_Z];
    float   cell_size_m;
    float   origin[3];
} occupancy_grid_t;

#define VA_NUM_DETECTIONS 8

typedef struct {
    uint64_t      frame_id;
    uint64_t      timestamp_us;
    uint32_t      version;

    agent_state_t state;
    motor_cmd_t   last_cmd;

    uint8_t       occupancy[VA_GRID_X][VA_GRID_Y][VA_GRID_Z];
    uint8_t       detections[VA_NUM_DETECTIONS];
    uint8_t       gray[VA_FRAME_PIXELS];
    uint16_t      depth[VA_FRAME_PIXELS];

    atomic_uint   seq;
} vision_state_t;

#define VA_IPC_VERSION 1
#define VA_SHM_NAME "/vision_agent"

#endif
