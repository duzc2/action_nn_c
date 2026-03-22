#include <stdio.h>

static void render_digit(int s0, int s1, int s2, int s3, int s4, int s5, int s6) {
    printf(" %s \n", s0 ? "---" : "   ");
    printf("%c   %c\n", s5 ? '|' : ' ', s1 ? '|' : ' ');
    printf(" %s \n", s6 ? "---" : "   ");
    printf("%c   %c\n", s4 ? '|' : ' ', s2 ? '|' : ' ');
    printf(" %s \n", s3 ? "---" : "   ");
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
    int ch = 0;
    printf("input 0-9 to render sevenseg, q to quit\n");
    while ((ch = getchar()) != EOF) {
        if (ch == '\n' || ch == '\r') {
            continue;
        }
        if (ch == 'q' || ch == 'Q') {
            break;
        }
        if (ch >= '0' && ch <= '9') {
            int d = ch - '0';
            render_digit(
                seg_map[d][0],
                seg_map[d][1],
                seg_map[d][2],
                seg_map[d][3],
                seg_map[d][4],
                seg_map[d][5],
                seg_map[d][6]
            );
        } else {
            printf("invalid input\n");
        }
    }
    return 0;
}
