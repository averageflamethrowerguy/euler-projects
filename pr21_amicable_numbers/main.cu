#include <iostream>

/*
 * LOGIC:
 * 1. Simultaneously calculate n divisor sums
 * 2. Add these sums to the array
 * 3. When looped over entire array, go back and check pairs.
 */

__global__ void VecDivisorSum(int* number) {
    int threadNumber = number[threadIdx.x];
    double root = sqrt((double) threadNumber);
    int sum = 0;

    for (int i = 1; i <= root; i++) {
        if (threadNumber % i == 0) {
            sum++;

            if ((threadNumber / i) != threadNumber && (double)(threadNumber / i) != root) {
                sum++;
            }
        }
    }

    number[threadNumber] = sum;
}

//int sum_of_divisors(int number) {
//    double root = sqrt(number);
//    int sum = 0;
//
//    for (int i = 1; i <= root; i++) {
//        if (number % i == 0) {
//            sum++;
//
//            if ((number / i) != number && (double)(number / i) != root) {
//                sum++;
//            }
//        }
//    }
//
//    return sum;
//}

int count_pairs(int highest_num) {
    int divisor_sum[highest_num + 1];

    int numBlocks = 1;
    dim3 threadsPerBlock(1);
    VecDivisorSum<<<numBlocks, threadsPerBlock>>>(divisor_sum);

    int pairs_sum = 0;

    // loop over all the components of the list
    for (int i = 1; i <= highest_num; i++) {
        int temp_sum = divisor_sum[i];
        if (temp_sum != i && temp_sum <= highest_num) {
            temp_sum = divisor_sum[temp_sum];

            if (temp_sum == i) {
                pairs_sum++;
            }
        }
    }

    return pairs_sum;
}

int main() {
    std::cout << count_pairs(9999) << std::endl;
    return 0;
}
