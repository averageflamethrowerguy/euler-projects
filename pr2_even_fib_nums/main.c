#include <stdio.h>

int find_even_fibs() {
    int num1 = 1;
    int num2 = 2;
    int sum = 0;

    while (num2 < 4000000) {
        if (num2 % 2 == 0) {
            printf("%d\n", num2);
            sum += num2;
        }

        int temp = num2 + num1;
        num1 = num2;
        num2 = temp;
    }

    return sum;
}

int main() {
    printf("%d", find_even_fibs());
    return 0;
}
