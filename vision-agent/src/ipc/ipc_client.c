#include "ipc/ipc.h"
#include "util/log.h"
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

vision_state_t *ipc_client_connect(const char *shm_name) {
    int fd = shm_open(shm_name, O_RDONLY, 0);
    if (fd < 0) {
        VA_ERR("client: shm_open failed for %s", shm_name);
        return NULL;
    }

    void *ptr = mmap(NULL, sizeof(vision_state_t),
                     PROT_READ, MAP_SHARED, fd, 0);
    close(fd);

    if (ptr == MAP_FAILED) {
        VA_ERR("client: mmap failed");
        return NULL;
    }

    return (vision_state_t *)ptr;
}

bool ipc_client_read(const vision_state_t *shm, vision_state_t *local) {
    for (int attempt = 0; attempt < 4; attempt++) {
        unsigned int seq1 = atomic_load(&((vision_state_t *)shm)->seq);
        if (seq1 & 1) continue;

        __sync_synchronize();
        memcpy(local, shm, sizeof(vision_state_t));
        __sync_synchronize();

        unsigned int seq2 = atomic_load(&((vision_state_t *)shm)->seq);
        if (seq1 == seq2)
            return true;
    }
    return false;
}

void ipc_client_disconnect(vision_state_t *shm) {
    if (shm) munmap((void *)shm, sizeof(vision_state_t));
}
