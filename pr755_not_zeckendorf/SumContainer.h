//
// Created by smooth_operator on 10/2/21.
//

#ifndef PR755_NOT_ZECKENDORF_SUMCONTAINER_H
#define PR755_NOT_ZECKENDORF_SUMCONTAINER_H


#include <set>

class SumContainer {
public:
    // the thing we're summing to
    long sum;
    // the set of Fib numbers that have already been used (and are relevant)
    std::set<long> usedFibNumbers;
    // the number of possible combinations
    long numberPossibleCombinations;

    SumContainer(long sum, std::set<long> usedFibNumbers, long numberPossibleCombinations) {
        this->sum = sum;
        this->usedFibNumbers = usedFibNumbers;
        this->numberPossibleCombinations = numberPossibleCombinations;
    }
};


#endif //PR755_NOT_ZECKENDORF_SUMCONTAINER_H
