#ifndef VA_MOTOR_H
#define VA_MOTOR_H

#include "util/types.h"

int  motor_init(void);
void motor_send(const motor_cmd_t *cmd);
void motor_stop(void);
void motor_shutdown(void);

#endif
