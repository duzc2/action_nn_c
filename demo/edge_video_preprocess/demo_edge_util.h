/**
 * @file demo_edge_util.h
 * @brief Shared utility for the edge video preprocessing demo.
 *
 * Provides a tiny string-copy helper used by both the CIFAR-10 dataset
 * loader and the video processor error-reporting paths.
 */

#ifndef DEMO_EDGE_VIDEO_PREPROCESS_DEMO_EDGE_UTIL_H
#define DEMO_EDGE_VIDEO_PREPROCESS_DEMO_EDGE_UTIL_H

#include <stddef.h>

/**
 * @brief Copy a C string into a bounded buffer (no stdlib string functions).
 *
 * If the message is longer than the buffer, it is silently truncated.
 *
 * @param buf  Destination buffer.
 * @param size Size of the destination buffer.
 * @param msg  Null-terminated message to copy.
 */
static void demo_set_error(char* buf, size_t size, const char* msg) {
    size_t n;

    if (buf == NULL || size == 0U) return;

    n = 0U;
    while (n + 1U < size && msg[n] != '\0') {
        buf[n] = msg[n];
        ++n;
    }
    buf[n] = '\0';
}

#endif /* DEMO_EDGE_VIDEO_PREPROCESS_DEMO_EDGE_UTIL_H */
