#include <iostream>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

/*
 * GamePlan:
 * 1. find all abundant numbers (GPU)
 * 2. assemble list of abundant numbers (will be pre-sorted)
 * 3. find all non-summed abundant numbers (GPU)
 */

#define TOP 28124

// 0 if not abundant, 1 if abundant
__global__ void is_abundant(char* abundance, int size) {
    int number = blockIdx.x * blockDim.x + threadIdx.x;
    if (number < size && number != 0) {
        double root = sqrt((double) number);
        int sum = 0;

        for (int i = 1; i <= root; i++) {
            if (number % i == 0) {
                sum += i;

                if ((number / i) != number && (double)((double)number / i) != root) {
                    sum += number / i;
                }
            }
        }

        if (sum > number) {
            abundance[number] = 1;
        }
    }
}

void get_abundant_numbers(char* abundance, int size) {
    char* dev_abundance = nullptr;
    int memSize = size * (int)sizeof(char);
    cudaMalloc((void**)&dev_abundance, memSize);

    cudaMemcpy(dev_abundance, abundance, memSize, cudaMemcpyHostToDevice);

    // configure the CUDA kernel
    int numBlocks = ceil((double) size / 256);
    dim3 threadsPerBlock(256);
    is_abundant<<<numBlocks, threadsPerBlock>>>(dev_abundance, size);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(abundance, dev_abundance, memSize, cudaMemcpyDeviceToHost);

    cudaFree(dev_abundance);
}

__global__ void is_sum_of_abundant(const int* abundantNumbers, int numberAbundantNumbers,
                                   char* not_sum_of_two, int size
                                   ) {
    int number = blockIdx.x * blockDim.x + threadIdx.x;
    if (number < size && number != 0) {
        int i = 0;
        int abundant1 = abundantNumbers[i];
        int willBreak = 0;
        while (abundant1 <= number && !willBreak) {
            int j = 0;
            int abundant2 = abundantNumbers[j];
            // abundant numbers are ordered, so we only have to check up to number (and could be less!)
            while (abundant1 + abundant2 <= number) {
                // we will break both loops if we have a sum of abundant numbers
                if (abundant1 + abundant2 == number) {
                    willBreak = 1;
                    break;
                }

                j++;
                if (j >= numberAbundantNumbers) {
                    break;
                }
                abundant2 = abundantNumbers[j];
            }

            i++;
            if (i >= numberAbundantNumbers) {
                break;
            }
            abundant1 = abundantNumbers[i];
        }

        // in this case, we never detected an abundant sum, and we add one to our not_sum array
        if (!willBreak) {
            not_sum_of_two[number] = 1;
        }
    }
}

void get_not_sum_of_two(std::vector<int>abundantNumbers, char* not_sum_of_two, int size) {
    int* dev_abundantNumbers = nullptr;
    char* dev_not_sum_of_two = nullptr;

    int memSizeNotSum = size * (int)sizeof(char);
    cudaMalloc((void**)&dev_not_sum_of_two, memSizeNotSum);
    int memSizeAbundantNumbers = (int)abundantNumbers.size() * (int)sizeof(int);
    cudaMalloc((void**)&dev_abundantNumbers, memSizeAbundantNumbers);

    cudaMemcpy(dev_not_sum_of_two, not_sum_of_two, memSizeNotSum, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_abundantNumbers, abundantNumbers.data(), memSizeAbundantNumbers, cudaMemcpyHostToDevice);

    // configure the CUDA kernel
    int numBlocks = ceil((double) size / 256);
    dim3 threadsPerBlock(256);
    is_sum_of_abundant<<<numBlocks, threadsPerBlock>>>(
            dev_abundantNumbers,
            (int)abundantNumbers.size(),
            dev_not_sum_of_two,
            size
    );

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(not_sum_of_two, dev_not_sum_of_two, memSizeNotSum, cudaMemcpyDeviceToHost);

    cudaFree(dev_not_sum_of_two);
    cudaFree(dev_abundantNumbers);

    cudaDeviceReset();
}

long get_non_abundant_sum() {
    char* abundance;
    abundance = (char *) std::malloc(sizeof(char) * TOP);
    memset(abundance, 0, sizeof(char) * TOP);

    get_abundant_numbers(abundance, TOP);

    std::vector<int> abundantNumbers;
    for (int i = 0; i < TOP; i++) {
        if (abundance[i] == 1) {
            abundantNumbers.push_back(i);
        }
    }

    char* not_sum_of_two;
    not_sum_of_two = (char * ) std::malloc(sizeof(char) * TOP);
    memset(not_sum_of_two, 0, sizeof(char) * TOP);

    get_not_sum_of_two(abundantNumbers, not_sum_of_two, TOP);

    long sum = 0;
    for (int i = 0; i < TOP; i++) {
        if (not_sum_of_two[i] == 1) {
            sum += i;
        }
    }

    free(abundance);
    free(not_sum_of_two);
    abundantNumbers.clear();

    return sum;
}

int main() {
    std::cout << get_non_abundant_sum() << std::endl;
    return 0;
}
