//
// Created by smooth_operator on 10/2/21.
//

#ifndef PR755_NOT_ZECKENDORF_SUMCONTAINER_H
#define PR755_NOT_ZECKENDORF_SUMCONTAINER_H

#include "SumContainer.h"

class SumContainer {
public:
    // the thing we're summing to
    long sum;
    // the index of the largest Fib number smaller than sum (this will be modified if this is
    // larger than the maxPossibleIndex (which may be constrained by previous selections)
    int indexOfLargestFib;
    // the index of the smallest Fib number which has a cumulative sum greater than sum
    int indexOfSmallestFib;

    // a map from the largestIndex to the number of combinations
    std::map<int, long> combinationsMap;

    SumContainer(long sum, int indexOfLargestFib, int indexOfSmallestFib) {
        this->sum = sum;
        this->indexOfLargestFib = indexOfLargestFib;
        this->indexOfSmallestFib = indexOfSmallestFib;
    }

    void clear() {
        combinationsMap.clear();
    }
};


#endif //PR755_NOT_ZECKENDORF_SUMCONTAINER_H
