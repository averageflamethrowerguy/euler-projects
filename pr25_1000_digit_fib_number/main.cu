#include <iostream>
/*
 * This problem is ugly for the GPU because finding the next fibonacci number is
 * a single-threaded recurrence relation.
 *
 * we can use:
 * a>n = [Phi^n - phi^n] / sqrt(5)
 * where
 * Phi = (1 + sqrt(5)) / 2
 * and
 * phi = (1 - sqrt(5)) / 2 (which is much smaller for a large exponent)
 *
 * https://math.hmc.edu/funfacts/fibonacci-number-formula/
 *
 * and a>n ~= a>k * Phi^(n - k)
 *
 * because we're looking at 1000 digits, we need an alternative datatype.
 *
 * We can also use:
 * F(n + k) = F(n + 1) * F(k - 1) + F(n) * F(k - 2)
 * where F is zero-indexed
 * k = 33, n = 32
 *
 * 1  1  2  3       2  3
 * 5  8  13 21      13 21
 * 34
 *
 * Ex: F(7 => 2 + 5) = F(3) * F(4) + F(2) * F(3) = 3 * 5 + 2 * 3 = 21
 *     F(8 => 5 + 3) = F(6) * F(2) + F(5) * F(1) = 13 * 2 + 8 * 1 = 34
 *     F(33 => 31 + 2) = F(31) * F(2) + F(30) * F(1) = 2 * 2178309 + 1346269 = 5702887
 *
*/

/*
 * GamePlan:
 * 1. Initialize the first 33 terms of the Fib sequence (in ints)
 * 2. Allocate 32 1000-int sections of memory dedicated to storing the fib numbers; zero mem
 * 3. Allocate 2 1000-int sections of memory dedicated to storing fib numbers to build from; zero mem
 * 4. Initialize the build memory with the last two terms of the Fib (i.e. m 30 & 31)
 *
 * For every 32 fib numbers
 * 1. Copy the last build memory into Loc 0 in the storage memory
 * 2. For the middle 30 numbers, use build memory 0 & 1 and the standard algorithm
 * 3. For the last location, use build memory 1 & 2.
 * 4. Synchronize; then copy Loc 30 & 31 from storage into build 0 & 1. Add build 0 & 1 to build 2.
 * 5. Scan memory top locations to look for a value with the 1000th digit filled.
 *
 */

# define MAX_FIB_LENGTH 1000
# define NUMBER_PARALLEL 32

__device__ void multiplyAndAdd(
            int8_t* target,
            const int8_t* build_left, int leftMultiplier,
            const int8_t* build_right, int rightMultiplier,
            int size
        ) {

    // we start from the base of the numbers and scroll upward
    int index = size - 1;
    long accumulator = 0;
    while (index >= 0) {
        accumulator += build_left[index] * leftMultiplier;
        accumulator += build_right[index] * rightMultiplier;

        target[index] = (int8_t)(accumulator % 10);
        accumulator = (accumulator - accumulator % 10) / 10;
        index--;
    }
}

__global__ void generateFibSet(const int* initialFibs, int8_t* storage, int8_t* build, int8_t* is_over_1000) {
    int threadNumber = (int)(threadIdx.x);
    // k is the number to add to n (n starts on Fib 30 (31st fib number))
    int k = threadNumber + 2;
    int leftMultiplier = initialFibs[k - 1];
    int rightMultiplier = initialFibs[k - 2];

    multiplyAndAdd(
            storage + (MAX_FIB_LENGTH * threadNumber),
            build + MAX_FIB_LENGTH, leftMultiplier,
            build, rightMultiplier,
            MAX_FIB_LENGTH
    );

    if ((int)(storage + (MAX_FIB_LENGTH * threadNumber))[0] != 0) {
        is_over_1000[threadNumber] = 1;
    }
}

long run_fib_loop(int* initialFibs, int8_t* storage, int8_t* build, int8_t* is_over_1000) {
    int *dev_initialFibs = nullptr;
    int8_t *dev_storage = nullptr;
    int8_t *dev_build = nullptr;
    int8_t *dev_is_over_1000 = nullptr;

    int initialFibs_memSize = (NUMBER_PARALLEL + 1) * (int)sizeof(int);
    int storage_memSize = (NUMBER_PARALLEL) * MAX_FIB_LENGTH * (int)sizeof(int8_t);
    int build_memSize = (NUMBER_PARALLEL) * 2 * MAX_FIB_LENGTH * (int)sizeof(int8_t);
    int is_over_1000_memSize = (NUMBER_PARALLEL) * (int)sizeof(int8_t);

    // allocate memory
    cudaMalloc((void**)&dev_initialFibs, initialFibs_memSize);
    cudaMalloc((void**)&dev_storage, storage_memSize);
    cudaMalloc((void**)&dev_build, build_memSize);
    cudaMalloc((void**)&dev_is_over_1000, is_over_1000_memSize);

    // copy memory
    cudaMemcpy(dev_initialFibs, initialFibs, initialFibs_memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_storage, storage, storage_memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_build, build, build_memSize, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_is_over_1000, is_over_1000, is_over_1000_memSize, cudaMemcpyHostToDevice);

    long index = 0;
    int willContinue = 1;

    while (willContinue) {
        generateFibSet<<<1, NUMBER_PARALLEL>>>(
                dev_initialFibs,
                dev_storage,
                dev_build,
                dev_is_over_1000
        );

        cudaDeviceSynchronize();

        // copy memory back to the cpu
        cudaMemcpy(is_over_1000, dev_is_over_1000, is_over_1000_memSize, cudaMemcpyDeviceToHost);
        cudaMemcpy(storage, dev_storage, storage_memSize, cudaMemcpyDeviceToHost);
        // copy the last two storage memory locations to the build locations
        cudaMemcpy(
                dev_build,
                dev_storage + MAX_FIB_LENGTH * (NUMBER_PARALLEL - 2),
                sizeof(int8_t) * MAX_FIB_LENGTH * 2,
                cudaMemcpyDeviceToDevice
        );

//        if (index <= 1031) {
//            for (int i = 0; i < NUMBER_PARALLEL; i++) {
//                std::cout << "Index is: " << i + index << ", ";
//                for (int j = 0; j < MAX_FIB_LENGTH; j++) {
//                    std::cout << (int)(storage + (i * MAX_FIB_LENGTH))[j];
//                }
//                std::cout << std::endl;
//            }
//        }

        for (int i = 0; i < NUMBER_PARALLEL; i++) {
            if (0 != (int) is_over_1000[i]) {
                willContinue = 0;
            }
            else {
                index++;
            }
        }
    }

    // free memory
    cudaFree(dev_initialFibs);
    cudaFree(dev_storage);
    cudaFree(dev_build);
    cudaFree(dev_is_over_1000);
    cudaDeviceReset();

    return index;
}

void get_initial_fibs(int* initialFibs) {
    initialFibs[0] = 1;
    initialFibs[1] = 1;

    for (int i = 2; i < NUMBER_PARALLEL + 1; i++) {
        initialFibs[i] = initialFibs[i - 1] + initialFibs[i - 2];
    }
}

void convertIntToArray(int number, int8_t* array, int size) {
    // use a series of %10 and /10 operations to convert number into a base 10 int8_t array
    int index = size - 1;
    while (number > 0 && index >= 0) {
        array[index] = (int8_t)(number % 10);
        number = (number - number % 10) / 10;
        index--;
    }
}

void initialize_memory(int* initialFibs, int8_t* storage, int8_t* build, int8_t* is_over_1000) {
    get_initial_fibs(initialFibs);
    // zero the storage array
    memset(storage, 0, sizeof(int8_t) * MAX_FIB_LENGTH * NUMBER_PARALLEL);
    // zero the build array
    memset(build, 0, sizeof(int8_t) * MAX_FIB_LENGTH * 2);
    // zero the checking array
    memset(is_over_1000, 0, sizeof(int8_t) * NUMBER_PARALLEL);

    convertIntToArray(initialFibs[NUMBER_PARALLEL - 2], build, MAX_FIB_LENGTH);
    convertIntToArray(initialFibs[NUMBER_PARALLEL - 1], build + MAX_FIB_LENGTH, MAX_FIB_LENGTH);
}

void get_first_1000_digits() {
    auto* storage = (int8_t* )malloc(sizeof(int8_t) * MAX_FIB_LENGTH * NUMBER_PARALLEL);
    auto* is_over_1000 = (int8_t*)malloc(sizeof(int8_t) * NUMBER_PARALLEL);
    auto* build = (int8_t* )malloc(sizeof(int8_t) * MAX_FIB_LENGTH * 2);
    int* initialFibs = (int *)malloc(sizeof(int) * (NUMBER_PARALLEL + 1));

    initialize_memory(initialFibs, storage, build, is_over_1000);

//    for (int i = 0; i < NUMBER_PARALLEL + 1; i++) {
//        std::cout << initialFibs[i] << std::endl;
//    }
//    for (int i = 0; i < 2; i++) {
//        for (int j = 0; j < MAX_FIB_LENGTH; j++) {
//            std::cout << (int)(build + (i * MAX_FIB_LENGTH))[j];
//        }
//        std::cout << std::endl;
//    }

    long index = run_fib_loop(initialFibs, storage, build, is_over_1000);
    std::cout << index + NUMBER_PARALLEL << std::endl;
    // should be 4782 ...

    free(is_over_1000);
    free(storage);
    free(build);
    free(initialFibs);
}

int main() {
    get_first_1000_digits();
    return 0;
}
