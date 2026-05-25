#include "capture/capture.h"
#include "util/timing.h"
#include "util/log.h"

#ifdef VA_HAS_V4L2

#include <linux/videodev2.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <errno.h>

#define NUM_BUFFERS 2

typedef struct {
    int fd;
    void *buffers[NUM_BUFFERS];
    uint32_t buf_lengths[NUM_BUFFERS];
} v4l2_cam_t;

static v4l2_cam_t s_left, s_right;
static capture_config_t s_cfg;

static int cam_open(v4l2_cam_t *cam, const char *dev, int width, int height) {
    cam->fd = open(dev, O_RDWR | O_NONBLOCK);
    if (cam->fd < 0) {
        VA_ERR("open %s: %s", dev, strerror(errno));
        return -1;
    }

    struct v4l2_format fmt = {0};
    fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    fmt.fmt.pix.width = width;
    fmt.fmt.pix.height = height;
    fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_GREY;
    fmt.fmt.pix.field = V4L2_FIELD_NONE;

    if (ioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) {
        fmt.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
        if (ioctl(cam->fd, VIDIOC_S_FMT, &fmt) < 0) {
            VA_ERR("VIDIOC_S_FMT %s: %s", dev, strerror(errno));
            close(cam->fd);
            return -1;
        }
    }

    struct v4l2_requestbuffers req = {0};
    req.count = NUM_BUFFERS;
    req.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    req.memory = V4L2_MEMORY_MMAP;
    if (ioctl(cam->fd, VIDIOC_REQBUFS, &req) < 0) {
        VA_ERR("VIDIOC_REQBUFS %s: %s", dev, strerror(errno));
        close(cam->fd);
        return -1;
    }

    for (int i = 0; i < NUM_BUFFERS; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        if (ioctl(cam->fd, VIDIOC_QUERYBUF, &buf) < 0) {
            VA_ERR("VIDIOC_QUERYBUF: %s", strerror(errno));
            close(cam->fd);
            return -1;
        }
        cam->buf_lengths[i] = buf.length;
        cam->buffers[i] = mmap(NULL, buf.length, PROT_READ | PROT_WRITE,
                               MAP_SHARED, cam->fd, buf.m.offset);
        if (cam->buffers[i] == MAP_FAILED) {
            VA_ERR("mmap: %s", strerror(errno));
            close(cam->fd);
            return -1;
        }
    }

    for (int i = 0; i < NUM_BUFFERS; i++) {
        struct v4l2_buffer buf = {0};
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;
        ioctl(cam->fd, VIDIOC_QBUF, &buf);
    }

    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    if (ioctl(cam->fd, VIDIOC_STREAMON, &type) < 0) {
        VA_ERR("VIDIOC_STREAMON %s: %s", dev, strerror(errno));
        close(cam->fd);
        return -1;
    }

    return 0;
}

static int cam_grab(v4l2_cam_t *cam, uint8_t *out, int expected_bytes) {
    struct v4l2_buffer buf = {0};
    buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    buf.memory = V4L2_MEMORY_MMAP;

    fd_set fds;
    FD_ZERO(&fds);
    FD_SET(cam->fd, &fds);
    struct timeval tv = {.tv_sec = 2, .tv_usec = 0};
    if (select(cam->fd + 1, &fds, NULL, NULL, &tv) <= 0)
        return -1;

    if (ioctl(cam->fd, VIDIOC_DQBUF, &buf) < 0)
        return -1;

    int copy_len = (int)buf.bytesused < expected_bytes ? (int)buf.bytesused : expected_bytes;
    memcpy(out, cam->buffers[buf.index], copy_len);

    ioctl(cam->fd, VIDIOC_QBUF, &buf);
    return 0;
}

static void cam_close(v4l2_cam_t *cam) {
    if (cam->fd < 0) return;
    int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(cam->fd, VIDIOC_STREAMOFF, &type);
    for (int i = 0; i < NUM_BUFFERS; i++)
        if (cam->buffers[i] && cam->buffers[i] != MAP_FAILED)
            munmap(cam->buffers[i], cam->buf_lengths[i]);
    close(cam->fd);
    cam->fd = -1;
}

int capture_init(const capture_config_t *cfg) {
    s_cfg = *cfg;
    if (cam_open(&s_left, "/dev/video0", cfg->src_width, cfg->src_height) != 0)
        return -1;
    if (cam_open(&s_right, "/dev/video1", cfg->src_width, cfg->src_height) != 0) {
        cam_close(&s_left);
        return -1;
    }
    VA_LOG("stereo CSI capture initialized (%dx%d)", cfg->src_width, cfg->src_height);
    return 0;
}

int capture_frame(uint8_t *gray_out, uint16_t *depth_out, uint64_t *timestamp_us) {
    (void)depth_out;
    *timestamp_us = timing_now_us();
    return cam_grab(&s_left, gray_out, s_cfg.src_width * s_cfg.src_height);
}

int capture_frame_stereo(uint8_t *left_out, uint8_t *right_out, uint64_t *timestamp_us) {
    int pixels = s_cfg.src_width * s_cfg.src_height;
    int r1 = cam_grab(&s_left, left_out, pixels);
    int r2 = cam_grab(&s_right, right_out, pixels);
    *timestamp_us = timing_now_us();
    return (r1 == 0 && r2 == 0) ? 0 : -1;
}

void capture_shutdown(void) {
    cam_close(&s_left);
    cam_close(&s_right);
}

#endif /* VA_HAS_V4L2 */
