#include <iostream>
#include <vector>
// NOTE -- this problem exhibits none of the parallelism or recursion we've seen in the previous
// two problems. It is most easily (and sanely) represented as a for-loop. (I think...?)
// you just have to loop from 0123456789 to 9876543210 and check if a num is a valid permutation
// return the millionth value

/*
 * Implementation details:
 * 1. Create an algorithm to efficiently generate a list of permutations (that requires limited sorting)
 * 2. Sort using quickSort on the GPU.
 */

#define DIGIT_LENGTH 10
#define SEARCH_LOC 1000000

// we recursively move the permutations pointer and reduce problemSize as numberUsed increases
__global__ void cudaGetPermutations(char* permutations, int offset, int problemSize, int numberUsed) {
    int newProblemSize = problemSize / (DIGIT_LENGTH - numberUsed);
    int newNumberUsed = numberUsed + 1;
    int newNumberIndex = 0;

    for (int i = 0; i < DIGIT_LENGTH; i++) {
        int isUsed = 0;
        for (int digitIndex = 0; digitIndex < numberUsed; digitIndex++) {
            if (i == (int)((permutations + offset)[digitIndex] - '0')) {
                isUsed = 1;
                break;
            }
        }

        if (!isUsed) {
            // we create a new pointer to the location in memory where we will continue building the permutation
            int newOffset = offset + (newProblemSize * newNumberIndex * DIGIT_LENGTH);
            newNumberIndex++;

            // constructs a new partial permutation
            int digitIndex = 0;
            // copies over the permutation from the old location in memory to the new location
            for (; digitIndex < numberUsed;) {
                (permutations + newOffset)[digitIndex] = (permutations + offset)[digitIndex];
                digitIndex++;
            }

            (permutations + newOffset)[digitIndex] = (char)(i + '0');

            if (newNumberUsed < DIGIT_LENGTH) {
                // recursively calls the function with the partial permutation
                cudaStream_t s;
                cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
                // NOTE: ignore this error as long as you've set compute capability 3.5 or higher
                cudaGetPermutations<<< 1, 1, 0, s>>>(
                        permutations,
                        newOffset,
                        newProblemSize,
                        newNumberUsed
                );
                cudaStreamDestroy(s);
            }
        }
    }
}

void get_permutations(char* permutations, int problemSize) {
    char* dev_permutations = nullptr;

    int memory_size = problemSize * DIGIT_LENGTH * (int)sizeof(char);

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_permutations, memory_size);

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_permutations, permutations, memory_size, cudaMemcpyHostToDevice);

    cudaGetPermutations<<<1, 1, 0>>>(
            dev_permutations,
            0,
            problemSize,
            0
    );

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(permutations, dev_permutations, memory_size, cudaMemcpyDeviceToHost);

    cudaFree(dev_permutations);
    cudaDeviceReset();
}

long get_millionth_permutation() {
    // we figure out how much memory we need to allocate to the process.
    // this is ~3.6 million for length 10 (we will allocate 36 MB)
    int problemSize = 1;
    for (int i = DIGIT_LENGTH; i > 0; i--) {
        problemSize *= i;
    }

    // we initialize the memory for our two-level char array
    char* permutations = (char *)malloc(sizeof(char) * DIGIT_LENGTH * problemSize);

    // we set the first DIGIT_LENGTH digits to be ascending order
    for (int i = 0; i < DIGIT_LENGTH; i++) {
        permutations[i] = (char)(i + '0');
    }

    get_permutations(permutations, problemSize);

    printf("%.10s\n", (permutations + ((SEARCH_LOC - 1) * DIGIT_LENGTH)));

    free(permutations);

    return 0;
}

int main() {
    std::cout << get_millionth_permutation() << std::endl;
    return 0;
}
