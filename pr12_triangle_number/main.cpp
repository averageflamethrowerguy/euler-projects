#include <iostream>
#include <valarray>

// getting the triangle number can be split into two problems:
// 1: finding the triangle number itself
// 2: getting the number of divisors of a number

// we can simply get the number of factors by looping up to the root of the number.
// any factor below the root is guaranteed to have a factor LARGER than the root.
// if the factor is the root, we avoid double-counting.
int get_number_of_factors_of_number(long number) {
    double root = sqrt( (double) number);
    int numberFactors = 0;

    int currentCheck = 1;
    while (currentCheck <= root) {
        if (number % currentCheck == 0) {
            if (currentCheck != root) {
                numberFactors += 2;
            }
            else {
                numberFactors++;
            }
        }

        currentCheck++;
    }

    return numberFactors;
}

// we just store the last triangle and the current number.
long get_triangle_number_with_500_factors() {
    int has_over_500_factors = 0;
    long currentTriangle = 1;
    int currentNumber = 2;

    while (!has_over_500_factors) {
        currentTriangle = currentTriangle + currentNumber;

        if (get_number_of_factors_of_number(currentTriangle) > 500)  {
            has_over_500_factors = 1;
        }

        currentNumber++;
    }

    return currentTriangle;
}

int main() {
    std::cout << get_triangle_number_with_500_factors() << std::endl;
    return 0;
}
