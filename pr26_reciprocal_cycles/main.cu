#include <iostream>
#include "DevArray.cuh"

__global__ void get_reciprocal_cycle_length(int* cycleLengthStorage, size_t size) {
    int threadNumber = blockIdx.x * blockDim.x + threadIdx.x;

    DevArray<int> remainderStore(16);
    DevArray<int> quotientStore(16);

    if (threadNumber > 1 && threadNumber < size) {
        int willContinue = 1;
        int remainder = 1;
        int quotient = 0;
        int loopBegins;
        int failsToLoop = 0;

        while (willContinue) {
            quotient = ((remainder * 10) - (remainder * 10)
                        % threadNumber)
                        / threadNumber;
            remainder = (remainder * 10) % threadNumber;

            if (remainder == 0) {
//                printf("Fails to loop: %d \n", threadNumber);
                failsToLoop = 1;
                break;
            }

            // loop through all remainders and see if this one is a repeat.
            // if so, we know that the loop will repeat
            for (int i = 0; i < remainderStore.sizeUsed; i++) {
                if (remainder == remainderStore.getData()[i]) {
                    willContinue = 0;
                    loopBegins = i;
                    break;
                }
            }

            // store the quotient and remainder (ONLY if we haven't found a loop yet)
            quotientStore.push_back(quotient);
            remainderStore.push_back(remainder);
        }

        if (!failsToLoop) {
            // we start at the index where the loop begins, adding 1 to the length every cycle
            // remainder var has the latest remainder (the one with the first repeat)
            int cycleLength = 0;
            for (int i = loopBegins; i < remainderStore.sizeUsed; i++) {
                cycleLength++;
            }

            // free consumed memory
            remainderStore.clear();
            quotientStore.clear();

            cycleLengthStorage[threadNumber] = cycleLength;
        }
    }
}

// Note: maxNumber is NOT inclusive
int get_longest_reciprocal_cycle(int maxNumber) {
    int storageSize = sizeof(int) * maxNumber;
    int* cycleLengthStorage = (int*) malloc(storageSize);
    memset(cycleLengthStorage, 0, storageSize);

    DevArray<int> dev_cycleLengthStorage(storageSize);
    dev_cycleLengthStorage.set(cycleLengthStorage, storageSize);

    int blockSize = 128;
    int numberBlocks = ceil((double) maxNumber / (double) blockSize);
    get_reciprocal_cycle_length<<<numberBlocks, blockSize>>>(
            dev_cycleLengthStorage.getData(),
            maxNumber
    );

    cudaDeviceSynchronize();
    dev_cycleLengthStorage.get(cycleLengthStorage, storageSize);
    dev_cycleLengthStorage.clear();
    cudaDeviceReset();

    int longestCycle = 0;
    int longestCycleIndex;
    for (int i = 0; i < maxNumber; i++) {
//        std::cout << cycleLengthStorage[i] << std::endl;
        if (cycleLengthStorage[i] > longestCycle) {
            longestCycle = cycleLengthStorage[i];
            longestCycleIndex = i;
        }
    }

    return longestCycleIndex;
}

int main() {
    std::cout << get_longest_reciprocal_cycle(1000) << std::endl;
    return 0;
}
