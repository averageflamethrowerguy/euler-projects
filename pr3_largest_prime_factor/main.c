#include <stdio.h>
#include <math.h>
#include <malloc.h>

typedef struct {
    int *array;
    size_t used;
    size_t size;
} Array;

void initArray(Array *a, size_t initialSize) {
    a->array = malloc(initialSize * sizeof(int));
    a->used = 0;
    a->size = initialSize;
}

void insertArray(Array *a, int element) {
    if (a->used == a->size) {
        a->size *= 2;
        a->array = realloc(a->array, a->size * sizeof (int));
    }
    a->array[a->used++] = element;
}

void freeArray(Array *a) {
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

// note boolean is 0 vs 1, rather than a separate bool type
int is_prime(int number, Array lowerPrimes) {
    double root = sqrt( (double) number);

    int testIndex = 0;
    int testNum = lowerPrimes.array[testIndex];
    int isPrime = 1;

    while (testNum <= root && testIndex < lowerPrimes.size) {
        if (testNum != 0 && testNum != 1 && (number % testNum == 0)) {
            isPrime = 0;
            break;
        }

        testIndex++;
        testNum = lowerPrimes.array[testIndex];
    }

    return isPrime;
}

int get_largest_prime(long number) {
    double root = sqrt( (double) number);

    printf("Root calculated\n");

    int testNum = 1;
    int greatestPrime = 2;
    Array smallPrimes;
    Array bigPossiblePrimes;
    initArray(&smallPrimes, 256);
    initArray(&bigPossiblePrimes, 256);

    printf("Entering the first loop to find primes below the root\n");

    while (testNum <= root) {
        // by finding ALL the primes within the root, we find all possible prime factors of the
        // factors of "number", because in the worst case, "number" may have a prime of 1/2 it's size.
        if (is_prime(testNum, smallPrimes)) {
            if (number % testNum == 0) {
                // we're always guaranteed to be finding a bigger prime.
                greatestPrime = testNum;

                // this step is cool. Larger prime factors can only exist when connected to smaller prime factors.
                // therefore, we assemble larger factors rather than looping over the entire number
                insertArray(&bigPossiblePrimes, (int) (number / testNum));
            }
            insertArray(&smallPrimes, testNum);
        }
        testNum += 1;
    }

    printf("Entering the second loop to find large primes...\n");

    int index = 0;
    while (index < bigPossiblePrimes.size) {
        // we will completely ignore any possible prime smaller than the current largest prime
        if (bigPossiblePrimes.array[index] > greatestPrime) {
            // we check if the possible prime is, in fact, prime. If it is, we already know it is a factor.
            if (is_prime(bigPossiblePrimes.array[index], smallPrimes)) {
                greatestPrime = bigPossiblePrimes.array[index];
            }
        }

        index++;
    }

    freeArray(&smallPrimes);
    freeArray(&bigPossiblePrimes);

    return greatestPrime;
}

int main() {
    printf("Beginning the calculation...\n");
    printf("%d\n", get_largest_prime(600851475143));
    return 0;
}
