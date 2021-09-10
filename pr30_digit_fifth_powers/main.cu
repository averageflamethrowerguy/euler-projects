#include <iostream>
#include "DevArray.cuh"

__device__ int getFifthPower(int number) {
    int power = 1;
    for (int i = 0; i < 5; i++) {
        power *= number;
    }

    return power;
}

__global__ void checkSumOfFifthPowers(int* powerSums, int maxNumberGuess) {
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadNumber > 1 && threadNumber < maxNumberGuess) {
        int sum = 0;
        int number = threadNumber;

        while (number > 0) {
            sum += getFifthPower(number % 10);
            number = (number - (number % 10)) / 10;
        }

        if (sum == threadNumber) {
            powerSums[threadNumber] = 1;
        }
    }
}

int sumOfSumOfFifthPowers(int maxNumberGuess) {
    int* powerSums = (int *) malloc(sizeof(int) * maxNumberGuess);
    memset(powerSums, 0, sizeof(int) * maxNumberGuess);

    DevArray<int> dev_powerSums(maxNumberGuess);
    dev_powerSums.set(powerSums, maxNumberGuess);

    int blockSize = 256;
    int blockNumber = ceil(maxNumberGuess * blockSize);
    checkSumOfFifthPowers<<<blockNumber, blockSize>>>(dev_powerSums.getData(), maxNumberGuess);
    cudaDeviceSynchronize();

    dev_powerSums.get(powerSums, maxNumberGuess);
    dev_powerSums.clear();

    int sum = 0;
    for (int i = 0; i < maxNumberGuess; i++) {
        if (powerSums[i]) {
            sum += i;
        }
    }

    free(powerSums);
    return sum;
}

int main() {
    std::cout << sumOfSumOfFifthPowers(300000) << std::endl;
    return 0;
}
