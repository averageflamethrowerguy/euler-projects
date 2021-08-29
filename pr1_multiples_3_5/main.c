#include <stdio.h>

int multiplesSum() {
    int sum = 0;
    int num = 1;

    while (num < 1000) {
        if (num % 3 == 0 || num % 5 == 0) {
            sum += num;
            printf("%d\n", num);
        }

        num++;
    }

    return sum;
}

int main() {
    int sum = multiplesSum();
    printf("%d", sum);
    return 0;
}
