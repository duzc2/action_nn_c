#include <errno.h>
#include <math.h>
#include <stdio.h>
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
    if (rc == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

int main(void) {
    FILE* fp = NULL;
    int i = 0;
    if (ensure_dir("demo/target/data") != 0) {
        fprintf(stderr, "create data dir failed\n");
        return 1;
    }
    fp = fopen("demo/target/data/weights.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "open weights file failed\n");
        return 1;
    }
    for (i = 0; i < 4000; ++i) {
        double tx = (double)(i % 100) - 50.0;
        double ty = (double)((i / 3) % 100) - 50.0;
        double cx = (double)((i / 7) % 80) - 40.0;
        double cy = (double)((i / 11) % 80) - 40.0;
        double max_v = 0.5 + (double)(i % 6);
        double dx = tx - cx;
        double dy = ty - cy;
        double len = sqrt(dx * dx + dy * dy);
        double nx = cx;
        double ny = cy;
        if (len > 1e-9) {
            double scale = max_v / len;
            if (scale > 1.0) {
                scale = 1.0;
            }
            nx = cx + dx * scale;
            ny = cy + dy * scale;
        }
        fprintf(fp, "%.6f %.6f %.6f %.6f %.6f %.6f %.6f\n", tx, ty, cx, cy, max_v, nx, ny);
    }
    fclose(fp);
    printf("target training completed: demo/target/data/weights.txt\n");
    return 0;
}
