#ifndef VA_LOG_H
#define VA_LOG_H

#include <stdio.h>

#ifdef NDEBUG
#define VA_LOG(fmt, ...) ((void)0)
#define VA_WARN(fmt, ...) ((void)0)
#else
#define VA_LOG(fmt, ...) fprintf(stderr, "[VA] " fmt "\n", ##__VA_ARGS__)
#define VA_WARN(fmt, ...) fprintf(stderr, "[VA WARN] " fmt "\n", ##__VA_ARGS__)
#endif

#define VA_ERR(fmt, ...) fprintf(stderr, "[VA ERR] " fmt "\n", ##__VA_ARGS__)

#endif
