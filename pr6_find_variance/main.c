#include <stdio.h>

long square(int num) {
    return (long) num * (long) num;
}

long getDifference(int maxNum) {
    int sum = 0;
    long sumOfSquares = 0;

    int i = 1;
    while (i <= maxNum) {
        sum += i;
        sumOfSquares += square(i);
        i++;
    }

    printf("%d\n", sum);
    printf("%ld\n", sumOfSquares);

    long squaredSum = square(sum);

    return squaredSum - sumOfSquares;
}

int main() {
    printf("%ld\n", getDifference(100));
    return 0;
}
