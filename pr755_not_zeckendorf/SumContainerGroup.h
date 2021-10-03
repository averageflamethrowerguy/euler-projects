//
// Created by smooth_operator on 10/2/21.
//

#ifndef PR755_NOT_ZECKENDORF_SUMCONTAINERGROUP_H
#define PR755_NOT_ZECKENDORF_SUMCONTAINERGROUP_H

#include "SumContainer.h"

class SumContainerGroup {
public:
    // the thing we're summing to
    long sum;
    // the index of the largest Fib number smaller that sum
    int indexOfLargestFib;
    // the index of the smallest Fib number which has a cumulative sum greater than sum
    int indexOfSmallestFib;

    // the list of all the SumContainers
    std::vector<SumContainer> containers{};

    SumContainerGroup(long sum, int indexOfLargestFib, int indexOfSmallestFib) {
        this->sum = sum;
        this->indexOfLargestFib = indexOfLargestFib;
        this->indexOfSmallestFib = indexOfSmallestFib;
    }

    void clear() {
        for (SumContainer container : containers) {
            container.usedFibNumbers.clear();
        }
    }
};


#endif //PR755_NOT_ZECKENDORF_SUMCONTAINERGROUP_H
