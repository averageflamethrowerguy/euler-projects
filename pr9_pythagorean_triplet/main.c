#include <stdio.h>

int square(int num) {
    return num * num;
}

int get_pythagorean_triplet(int sum_to) {
    // we're going to have the easiest time constructing loops and avoiding dumb cases
    for (int c = sum_to / 2; c > 0; c--) {
        for (int a = 0; a < c; a++) {
            int b = sum_to - ( a + c );

            if (b > 0 && b < c) {
                int asquare = square(a);
                int bsquare = square(b);
                int csquare = square(c);

                printf("a: %d, b: %d, a+b: %d, c: %d\n", asquare, bsquare, asquare + bsquare, csquare);
                if ((asquare + bsquare) == csquare) {
                    return a * b * c;
                }
            }
        }
    }

    return 0;
}

int main() {
    printf("%d", get_pythagorean_triplet(1000));
    return 0;
}
