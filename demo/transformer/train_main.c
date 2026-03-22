#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
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
    if (rc == 0 || errno == EEXIST) {
        return 0;
    }
    return -1;
}

static int load_lines(const char* path, char lines[][160], int cap) {
    FILE* fp = fopen(path, "r");
    int count = 0;
    if (fp == NULL) {
        return -1;
    }
    while (count < cap && fgets(lines[count], 160, fp) != NULL) {
        size_t len = strlen(lines[count]);
        if (len > 0 && (lines[count][len - 1] == '\n' || lines[count][len - 1] == '\r')) {
            lines[count][len - 1] = '\0';
        }
        ++count;
    }
    fclose(fp);
    return count;
}

int main(void) {
    char lines[256][160];
    int count = 0;
    FILE* fp = NULL;
    int i = 0;
    if (ensure_dir("demo/transformer/data") != 0) {
        fprintf(stderr, "create data dir failed\n");
        return 1;
    }
    count = load_lines("demo/transformer/data/corpus.txt", lines, 256);
    if (count <= 0) {
        fprintf(stderr, "missing corpus: demo/transformer/data/corpus.txt\n");
        return 1;
    }
    fp = fopen("demo/transformer/data/dialogue_pairs.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "open dialogue_pairs failed\n");
        return 1;
    }
    fprintf(fp, "params=10000\n");
    fprintf(fp, "samples=%d\n", count);
    for (i = 0; i < count - 1; ++i) {
        fprintf(fp, "Q:%s\nA:%s\n", lines[i], lines[i + 1]);
    }
    fclose(fp);
    printf("transformer training completed: demo/transformer/data/dialogue_pairs.txt\n");
    return 0;
}
