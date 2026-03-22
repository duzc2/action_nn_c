#include <math.h>
#include <stdio.h>

static void step_to_target(double tx, double ty, double cx, double cy, double max_v, double* ox, double* oy) {
    double dx = tx - cx;
    double dy = ty - cy;
    double len = sqrt(dx * dx + dy * dy);
    if (len <= 1e-9) {
        *ox = cx;
        *oy = cy;
        return;
    }
    if (max_v < 0.0) {
        max_v = 0.0;
    }
    if (max_v > len) {
        max_v = len;
    }
    *ox = cx + dx * (max_v / len);
    *oy = cy + dy * (max_v / len);
}

int main(void) {
    double tx = 0.0;
    double ty = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double max_v = 1.0;
    double nx = 0.0;
    double ny = 0.0;
    printf("input: targetX targetY currentX currentY maxSpeed\n");
    while (scanf("%lf %lf %lf %lf %lf", &tx, &ty, &cx, &cy, &max_v) == 5) {
        step_to_target(tx, ty, cx, cy, max_v, &nx, &ny);
        printf("move_to_x=%.6f move_to_y=%.6f\n", nx, ny);
    }
    return 0;
}
