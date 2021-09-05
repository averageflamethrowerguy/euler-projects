#include <iostream>
#include <stdio.h>

/*
 * LOGIC:
 * 1. Simultaneously calculate n divisor sums
 * 2. Add these sums to the array
 * 3. When looped over entire array, go back and check pairs.
 */

__global__ void VecDivisorSum(int* number, int size) {
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;
    if (threadNumber < size) {
        // NOTE: printf works within GPU code as well.
        // printf("Got here");
        double root = sqrt((double) threadNumber);
        int sum = 0;

        for (int i = 1; i <= root; i++) {
            if (threadNumber % i == 0) {
                sum += i;

                if ((threadNumber / i) != threadNumber && (double)((double)threadNumber / i) != root) {
                    sum += threadNumber / i;
                }
            }
        }

        number[threadNumber] = sum;
    }
}

void get_divisors_with_cuda(int* divisor_sum, int size) {
    int* dev_sum = nullptr;

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_sum, size * sizeof(int));

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_sum, divisor_sum, size * sizeof(int), cudaMemcpyHostToDevice);

    // configure the CUDA kernel
    int numBlocks = ceil((double) size / 256);
    dim3 threadsPerBlock(256);
    VecDivisorSum<<<numBlocks, threadsPerBlock>>>(dev_sum, size);

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(divisor_sum, dev_sum, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(dev_sum);
    cudaDeviceReset();
}

int count_pairs(int highest_num) {
    int size = highest_num + 1;
    int divisor_sum[size];
    memset(divisor_sum, 0, (size)*sizeof(int) );

    get_divisors_with_cuda(divisor_sum, size);

    int pairs_sum = 0;

//    for (int i = 0; i < size; i++) {
//        std::cout << i << ", " << divisor_sum[i] << std::endl;
//    }

    // loop over all the components of the list
    for (int i = 1; i <= highest_num; i++) {
        int temp_sum = divisor_sum[i];
        if (temp_sum > i && temp_sum != 0 && temp_sum != i && temp_sum <= highest_num) {
            temp_sum = divisor_sum[temp_sum];

            if (temp_sum == i) {
                pairs_sum += i;
                pairs_sum += divisor_sum[i];
            }
        }
    }

    return pairs_sum;
}

int main() {
    std::cout << count_pairs(9999) << std::endl;
    return 0;
}
