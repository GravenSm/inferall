#include "hal/motor.h"
#include "util/log.h"

int motor_init(void) {
    VA_LOG("motor: stub initialized (no GPIO)");
    return 0;
}

void motor_send(const motor_cmd_t *cmd) {
    (void)cmd;
}

void motor_stop(void) {
    VA_LOG("motor: stop");
}

void motor_shutdown(void) {
    VA_LOG("motor: shutdown");
}
