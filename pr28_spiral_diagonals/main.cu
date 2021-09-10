#include <iostream>
#include "DevArray.cuh"

/*
 * We can get the number in a square by:
 * Diagonal(Right, Bottom) = (N - 2)^2 + N - 1
 * Diagonal(Left, Bottom) = (N - 2)^2 + (N - 1)*2
 * Diagonal(Left, Top) = (N - 2)^2 + (N - 1)*3
 * Diagonal(Right, Top) = N^2 = (N - 2)^2 + (N - 1)*4
 * SUM -> 4 * (N - 2)^2 + 10 * (N - 1)
 *
 * where N is the side length of the square.
 */

__global__ void get_diagonal_sum(long* sums, int maximumSquare) {
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    int N = 2 * threadNumber + 1;
    long sum = 0;

    if (N <= maximumSquare) {
        if (N == 1) {
            sum = 1;
        }
        else {
            long smallerSquare = (N - 2) * (N - 2);
            sum = 4 * smallerSquare + 10 * (N - 1);
        }

        sums[threadNumber] = sum;
    }
}

long sumOfDiagonals(int matrixWidth) {
    int numberSquares = (matrixWidth - 1) / 2 + 1;

    long* sums = (long*) malloc(sizeof(long) * numberSquares);
    memset(sums, 0, sizeof(long) * numberSquares);

    DevArray<long> dev_sums(numberSquares);
    dev_sums.set(sums, numberSquares);

    int blockSize = 256;
    int numberBlocks = ceil((double) numberSquares / blockSize);
    get_diagonal_sum<<<numberBlocks, blockSize>>>(
            dev_sums.getData(),
            matrixWidth
    );
    cudaDeviceSynchronize();

    dev_sums.get(sums, numberSquares);
    dev_sums.clear();

    long finalSum = 0;

    for (int i = 0; i < numberSquares; i++) {
        finalSum += sums[i];
    }

    free(sums);
    return finalSum;
}

int main() {
    std::cout << sumOfDiagonals(1001) << std::endl;
    return 0;
}
