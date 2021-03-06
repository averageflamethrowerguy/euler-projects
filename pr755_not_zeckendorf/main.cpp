#include <iostream>
#include <vector>
#include <unordered_map>
#include <chrono>
#include "SumContainer.h"

/*
 * This is supposed to be a combinatorics problem, but I'm attempting it with no actual background.
 *
 * Algorithm:
 * 1. Initialize the helper values
 *   a: Create a list of all the Fib numbers from 1 to the Fib number below the target number
 *   b: Create another list of all the sums of all previous Fib numbers
 *   c: Create a Map that will store mappings from targetNumber to a Vector of SumContainers
 * 2. Do the summation
 */

std::vector<long> fibNumbers{};
std::vector<long> fibSums{};
std::unordered_map<long, SumContainer*> sumContainerGroups(100000000);

long getFibCombinationsHelper(long sum, int indexOfHighestPossibleFib) {
    // first, we check if we already have this value stored in sumContainerGroups
    auto it = sumContainerGroups.find(sum);

    int indexOfLargestFib = indexOfHighestPossibleFib; // this caps our upward value

    SumContainer* container;
    // we have an existing container
    if (it != sumContainerGroups.end()) {
        container = it->second;

        // update indexOfLargestFib based on stored value
        if (container->indexOfLargestFib < indexOfLargestFib) {
            indexOfLargestFib = container->indexOfLargestFib;
        }
    }
    else {
        // we find the index of the highest Fib less than sum
        while (indexOfLargestFib >= 0 && fibNumbers[indexOfLargestFib] > sum) {
            indexOfLargestFib--;
        }
        if (indexOfLargestFib < 0) {
            indexOfLargestFib = 0;
        }

        // we find the index of the lowest Fib that still has a fibSum greater than sum
        // we iterate downward because that should be faster in general
        int indexOfSmallestFib = indexOfLargestFib;
        while(indexOfSmallestFib >= 0 && fibSums[indexOfSmallestFib] >= sum) {
            indexOfSmallestFib--;
        }
        // we overshoot by one and correct it
        indexOfSmallestFib++;

        container = new SumContainer(sum, indexOfLargestFib, indexOfSmallestFib);
        sumContainerGroups.insert({sum, container});
    }

    auto combo_iterator = container->combinationsMap.find(indexOfLargestFib);
    // we already have the value for the number of combinations from a previous iteration;
    // we use that instead of calculating again
    if (combo_iterator != container->combinationsMap.end()) {
        // return so we can recover up the chain
        return combo_iterator->second;
    }
    else {
        long numberPossibleCombinations = 0;
        // we've overrun the possible sums; must return 0 (or 1 if it's a Fib)
        if (indexOfLargestFib < container->indexOfSmallestFib) {}
        // otherwise, we continue recursing

        else {
            for (int i = container->indexOfSmallestFib;
                 i <= indexOfLargestFib;
                 i++
            ) {
                long newFib = fibNumbers[i];
                if (sum - newFib > 0) {
                    numberPossibleCombinations += getFibCombinationsHelper(
                            sum - newFib,
                            // this is the trick! We prevent selection of larger fibs
                            i - 1
                    );
                }

            }
        }

        // test to see if this number is itself a fib; if so, add 1 to the sum (because the sum is
        // itself a valid combination
        if (fibNumbers[indexOfLargestFib] == sum) {
            numberPossibleCombinations++;
        }

        container->combinationsMap.insert({indexOfLargestFib, numberPossibleCombinations});

        return numberPossibleCombinations;
    }
}

long getFibCombinationsSum(long maxNumber) {
    // initialize the vectors
    fibNumbers.push_back(1);
    fibNumbers.push_back(2);
    fibSums.push_back(1);
    fibSums.push_back(3);

    while (fibNumbers[fibNumbers.size() - 1] < maxNumber) {
        // we add the next fib number
        fibNumbers.push_back(fibNumbers[fibNumbers.size() - 1] + fibNumbers[fibNumbers.size() - 2]);
        // we update the sum
        fibSums.push_back(fibNumbers[fibNumbers.size() - 1] + fibSums[fibSums.size() - 1]);
    }

    long combinationsSum = 0;
    int currentPowerTen = 10;
    int sizeOfPower = 1;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i <= maxNumber; i++) {
        if (i == 0) combinationsSum++;

        if (i % currentPowerTen == 0) {
            std::cout << sizeOfPower << std::endl;
            currentPowerTen *= 10;
            sizeOfPower++;
        }

        // 10^7
        if (i % 10000000 == 0) {
            auto current = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
                    current - start
            );
            std::cout << duration.count() << std::endl;
            start = current;
        }

        combinationsSum += getFibCombinationsHelper(i,
                             (int)fibNumbers.size() - 1
                             );

    }

    for (auto entry : sumContainerGroups) {
        entry.second->clear();
    }
    sumContainerGroups.clear();

    fibNumbers.clear();
    fibSums.clear();
    return combinationsSum;
}

long getFibCombinationsSumWithMath(long maxNumber) {
    fibNumbers.push_back(1);
    fibNumbers.push_back(2);

    while (fibNumbers[fibNumbers.size() - 1] <= maxNumber) {
        // we add the next fib number
        fibNumbers.push_back(fibNumbers[fibNumbers.size() - 1] + fibNumbers[fibNumbers.size() - 2]);
    }

    long cumulativeSum = 6;
    int fibIndexToStart = 3;  // start with 5
    long sumToCarry = 3;   // 3...4 has sum 3
    int oneModifier = -1;

    // we rapidly calculate up to the top of the pattern
    while (fibNumbers[fibIndexToStart + 1] - 1 <= maxNumber) {
        sumToCarry = 2 * sumToCarry + oneModifier * 1;
        oneModifier *= -1;
        cumulativeSum += sumToCarry;
        fibIndexToStart++;
    }
    // now we need to deal with the remainder of numbers in the pattern

    fibNumbers.clear();
    fibSums.clear();

    return cumulativeSum;
}

int main() {
   // std::cout << getFibCombinationsSumWithMath(5) << std::endl;
    //std::cout << getFibCombinationsSum(88) << std::endl;
    //std::cout << getFibCombinationsSum(87) << std::endl;
    std::cout << getFibCombinationsSum(10000000000000) << std::endl;
    // this implementation starts by taking ~20 secs per 10 million points; it gets worse over time, slowly.
    return 0;
}
