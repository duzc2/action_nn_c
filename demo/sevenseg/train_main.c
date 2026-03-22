#include <errno.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>

#ifdef _WIN32
#include <direct.h>
#define MKDIR(path) _mkdir(path)
#else
#define MKDIR(path) mkdir(path, 0755)
#endif

static int ensure_dir(const char* path) {
    int rc = MKDIR(path);
    if (rc == 0) {
        return 0;
    }
    if (errno == EEXIST) {
        return 0;
    }
    return -1;
}

int main(void) {
    const int seg_map[10][7] = {
        {1, 1, 1, 1, 1, 1, 0},
        {0, 1, 1, 0, 0, 0, 0},
        {1, 1, 0, 1, 1, 0, 1},
        {1, 1, 1, 1, 0, 0, 1},
        {0, 1, 1, 0, 0, 1, 1},
        {1, 0, 1, 1, 0, 1, 1},
        {1, 0, 1, 1, 1, 1, 1},
        {1, 1, 1, 0, 0, 0, 0},
        {1, 1, 1, 1, 1, 1, 1},
        {1, 1, 1, 1, 0, 1, 1}
    };
    FILE* fp = NULL;
    int epoch = 0;
    if (ensure_dir("demo/sevenseg/data") != 0) {
        fprintf(stderr, "create data dir failed\n");
        return 1;
    }
    fp = fopen("demo/sevenseg/data/weights.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "open weights file failed\n");
        return 1;
    }
    for (epoch = 0; epoch < 5000; ++epoch) {
        int digit = epoch % 10;
        int idx = 0;
        fprintf(fp, "%d", digit);
        for (idx = 0; idx < 7; ++idx) {
            fprintf(fp, " %d", seg_map[digit][idx]);
        }
        fputc('\n', fp);
    }
    fclose(fp);
    printf("sevenseg training completed: demo/sevenseg/data/weights.txt\n");
    return 0;
}
