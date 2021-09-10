#include <iostream>
#include "DevArray.cuh"

__device__ int isPrime(long number) {
    // deal with negatives ...
    if (number < 0) {
        number = -1 * number;
    }

    double root = sqrt((double) number);

    for (int i = 2; i <= root; i++) {
        if (number % i == 0) {
            return 0;
        }
    }

    return 1;
}

__global__ void evaluateQuadraticPrimes(int* numberPrimesArray, int maxA, int maxB, int iterationCounts) {
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadNumber < iterationCounts) {
        int A = (threadNumber % (2 * maxA + 1)) - maxA;
        int B = ((threadNumber - A) / (2 * maxA + 1)) - maxB;

        int numberPrimes = 0;
        int n = 0;
        int willContinue = 1;
        while (willContinue) {
            long possiblePrime = n * n + A * n + B;
            if (threadNumber == 0) {
                printf("Possible prime: %ld\n", possiblePrime);
            }

            if (isPrime(possiblePrime)) {
                n++;
                numberPrimes++;
            }
            else {
                willContinue = 0;
            }
        }

//        printf("A is: %d, B is %d, numberPrimes are: %d\n", A, B, numberPrimes);

        numberPrimesArray[threadNumber] = numberPrimes;
    }
}

// maxA and maxB ARE inclusive
long getCoefficientMultiple(int maxA, int maxB) {
    int iterationCounts = (2 * maxA + 1) * (2 * maxB + 1);
    int memorySize = (int)sizeof(int) * iterationCounts;
    int* numberQuadraticPrimes = (int*) malloc(memorySize);
    memset(numberQuadraticPrimes, 0, memorySize);

    DevArray<int> dev_numberQuadraticPrimes(iterationCounts);

    dev_numberQuadraticPrimes.set(numberQuadraticPrimes, iterationCounts);

    int blockSize = 256;
    int numberBlocks = ceil((double) iterationCounts / blockSize);
    evaluateQuadraticPrimes<<<numberBlocks, blockSize>>>(
        dev_numberQuadraticPrimes.getData(), maxA, maxB, iterationCounts
    );

    cudaDeviceSynchronize();
    dev_numberQuadraticPrimes.get(numberQuadraticPrimes, iterationCounts);
    dev_numberQuadraticPrimes.clear();

    int maxQuadraticChain = 0;
    int bestA;
    int bestB;

    for (int i = 0; i < iterationCounts; i++) {
        if (numberQuadraticPrimes[i] > maxQuadraticChain) {
            maxQuadraticChain = numberQuadraticPrimes[i];
            bestA = (i % (2 * maxA + 1)) - maxA;
            bestB = ((i - bestA) / (2 * maxA + 1)) - maxB;

            std::cout << "A is: " << bestA << ", B is: " << bestB <<
                    ", numberPrimes are: " << numberQuadraticPrimes[i] << std::endl;
        }
    }

    free(numberQuadraticPrimes);

    return bestA * bestB;
}

int main() {
    std::cout << getCoefficientMultiple(999, 1000) << std::endl;
    return 0;
}
