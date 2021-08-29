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

long get_primes_under(int underVal) {
    Array primes;
    initArray(&primes, 1024);
    long accumulatedPrimes = 0;
    int i = 2;
    while (i < underVal) {
        if (is_prime(i, primes)) {
            insertArray(&primes, i);
            accumulatedPrimes += i;
        }

        i++;
    }

    freeArray(&primes);
    return accumulatedPrimes;
}

int main() {
    printf("%ld\n", get_primes_under(2000000));
    return 0;
}
