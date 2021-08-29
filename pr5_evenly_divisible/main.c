#include <stdio.h>
#include <malloc.h>
#include <math.h>

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

int get_leftovers(int number, Array nums) {

    int testIndex = 0;
    while (testIndex < nums.size) {
        if (nums.array[testIndex] != 0 && number % nums.array[testIndex] == 0) {
            number = number / nums.array[testIndex];
        }

        testIndex++;
    }

    return number;
}

int get_smallest_evenly_divisible(int maxNum) {
    Array leftovers;
    initArray(&leftovers, 4);

    int currentNumber = 1;
    int minDivisible = 1;
    while (currentNumber <= maxNum) {
        int leftover = get_leftovers(currentNumber, leftovers);
        minDivisible *= leftover;
        // we will insert leftovers into the array
        insertArray(&leftovers, leftover);
        currentNumber += 1;
    }

    freeArray(&leftovers);
    return minDivisible;
}

int main() {
    printf("Smallest evenly divisible number is: %d\n", get_smallest_evenly_divisible(20));
    return 0;
}
