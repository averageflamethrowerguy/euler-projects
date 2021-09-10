#include <iostream>
#include "DevArray.cuh"

/*
 * We'd REALLY prefer to not have to evaluate 99 ^ 100 or something similar
 * We can eliminate numbers with a square root or a power that is another number on the list...
 *
 * Smaller numbers that comprise a larger number clearly eliminate the large number powers from being unique
 * if A has factors C and D, then A^N = C^N * D^N (so D^N must be a power of C for A^N to be a power of C)
 */

// returns 1 if it has a root, 0 otherwise
__device__ int hasAnyRoot(int number) {
    double squareRoot = sqrt((double) number);

    // we know the square root is the largest root; we'll look for other roots
    for (int i = 2; i <= squareRoot; i++) {
        int testNumber = i * i;
        int power = 2;

        // when we overshoot the number, we can leave
        while (testNumber <= number) {
            // we have found a valid root
            if (number == testNumber) {
                return power;
            }
            // we try for the next root (sqrt -> cube root -> fourth root ... )
            testNumber *= i;
            power++;
        }
    }

    return 0;
}

__global__ void getNonDuplicates(int lowerBoundA, int upperBoundA, int lowerBoundB, int upperBoundB, int* non_duplicates) {
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;

    int number = threadNumber + lowerBoundA;
    int widthB = upperBoundB - lowerBoundB + 1;

    if (number <= upperBoundA) {
        int rootPower = hasAnyRoot(number);

        if (rootPower == 0) {
            non_duplicates[threadNumber] = widthB;
        }
        else {
            // initialize an array to tick off
            int* powerArray = (int *) malloc(sizeof(int) * (widthB));
            memset(powerArray, 1, sizeof(int) * (widthB));

            // start with 1st root, then 2nd, etc
            for (int i = 1; i < rootPower; i++) {
                for (int j = lowerBoundB; j <= upperBoundB; j++) {
                    // if this is also a root of the current number
                    if ((i * j) % rootPower == 0) {
                        // we will mark it as a duplicate
                        powerArray[((i * j) / rootPower) - lowerBoundB] = 0;
                    }
                }
            }

            int counter = 0;
            for (int i = 0; i < widthB; i++) {
                if (powerArray[i]) {
                    counter++;
                }
            }

            non_duplicates[threadNumber] = counter;
            free(powerArray);
        }
    }
}

// lower and upper bounds are BOTH inclusive
int getUniquePowersCount(int lowerBoundA, int upperBoundA, int lowerBoundB, int upperBoundB) {
    int widthA = upperBoundA - lowerBoundA + 1;
    int* non_duplicates = (int*) malloc(sizeof (int) * widthA);
    memset(non_duplicates, 0, sizeof(int) * widthA);

    DevArray<int> dev_non_duplicates(widthA);
    dev_non_duplicates.set(non_duplicates, widthA);

    int blockSize = 128;
    int numberBlocks = ceil((double) widthA / blockSize);
    getNonDuplicates<<<numberBlocks, blockSize>>>(
            lowerBoundA,
            upperBoundA,
            lowerBoundB,
            upperBoundB,
            dev_non_duplicates.getData()
    );
    cudaDeviceSynchronize();

    dev_non_duplicates.get(non_duplicates, widthA);
    dev_non_duplicates.clear();

    // we clean entire rows of data
    int uniquePowersCount = 0;
    for (int i = 0; i < widthA; i++) {
        uniquePowersCount += non_duplicates[i];
    }

    free(non_duplicates);

    return uniquePowersCount;
}

int main() {
    std::cout << getUniquePowersCount(2, 100, 2, 100) << std::endl;
    return 0;
}
