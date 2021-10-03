#include <iostream>
#include <vector>
#include <map>
#include "SumContainerGroup.h"
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
std::map<long, SumContainerGroup> sumContainerGroups;

long getFibCombinationsHelper(long sum, const std::vector<long>& consumedFibs, int indexOfHighestPossibleFib) {
    // first, we check if we already have this value stored in sumContainerGroups
    std::map<long, SumContainerGroup>::iterator it;
    it = sumContainerGroups.find(sum);

    SumContainerGroup* containerGroup;
    // we have an existing containerGroup
    if (it != sumContainerGroups.end()) {
        containerGroup = &it->second;
    }
    else {
        // TODO: is there a case where the index of fibSum >= sum is larger than largestSum ...?
        // answer -- not if the math is done right!

        // we find the index of the highest Fib less than sum
        int indexOfLargestFib = indexOfHighestPossibleFib;
        while (indexOfLargestFib >= 0 && fibNumbers[indexOfLargestFib] >= sum) {
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

        containerGroup = new SumContainerGroup(sum, indexOfLargestFib, indexOfSmallestFib);
    }

    SumContainer* targetContainer;
    bool targetContainerInitialized = false;

    for (SumContainer container : containerGroup->containers) {
        bool isMismatched = false;
        for (long consumedFib : consumedFibs) {
            // we only check in-scope Fibs; the ones larger than Sum are irrelevant.
            if (consumedFib < sum) {
                // checks to see if the given usedFibNumber is NOT in the Set
                if (container.usedFibNumbers.find(consumedFib) == container.usedFibNumbers.end()) {
                    isMismatched = true;
                    break;
                }
            }
        }

        if (!isMismatched) {
            targetContainer = &container;
            targetContainerInitialized = true;
            break;
        }
    }

    // we already have the value for the number of combinations from a previous iteration;
    // we use that instead of calculating again
    if (targetContainerInitialized) {
        // return so we can recover up the chain
        return targetContainer->numberPossibleCombinations;
    }
    else {
        long numberPossibleCombinations = 0;

        // TODO: there may be a necessary step here to remove duplicates
        // though I don't think so, because if we constrain the next possible Fib
        // numbers to add to be underneath the Fib number we just added, we will
        // generate uniqueness based on the Fib we added
        for (int i = containerGroup->indexOfSmallestFib;
            i <= containerGroup->indexOfLargestFib;
            i++
        ) {
            std::vector<long> newConsumedFibs{consumedFibs};
            long newFib = fibNumbers[i];
            newConsumedFibs.push_back(newFib);
            numberPossibleCombinations += getFibCombinationsHelper(
                    sum - newFib,
                    newConsumedFibs,
                    i - 1      // this is the trick!
            );

            // construct a Set of only the relevant fibs
            std::set<long> usedFibNumbers{};
            for (long consumedFib : consumedFibs) {
                // we only check in-scope Fibs; the ones larger than Sum are irrelevant.
                if (consumedFib < sum) {
                    usedFibNumbers.insert(consumedFib);
                }
            }
            SumContainer newContainer(sum,
                                      usedFibNumbers,
                                      numberPossibleCombinations
                                      );
            containerGroup->containers.push_back(newContainer);
        }

        // in this case, we are at the base of the chain and return 1
        if (numberPossibleCombinations == 0) {
            return 1;
        }
        else {
            return numberPossibleCombinations;
        }
    }
}

long getFibCombinations(long desiredSum) {
    // initialize the vectors
    fibNumbers.push_back(1);
    fibNumbers.push_back(2);
    fibSums.push_back(1);
    fibSums.push_back(3);

    while (fibNumbers[fibNumbers.size() - 1] < desiredSum) {
        // we add the next fib number
        fibNumbers.push_back(fibNumbers[fibNumbers.size() - 1] + fibNumbers[fibNumbers.size() - 2]);
        // we update the sum
        fibSums.push_back(fibNumbers[fibNumbers.size() - 1] + fibSums[fibSums.size() - 1]);
    }

    std::vector<long> consumedFibs{};
//    getFibCombinationsHelper(desiredSum,
//                             consumedFibs,
//                             (int)fibNumbers.size() - 1
//                             );

    long combinations = 0;
    std::map<long, SumContainerGroup>::iterator it;
    it = sumContainerGroups.find(desiredSum);
    if (it != sumContainerGroups.end()) {
        combinations = it->second.containers[0].numberPossibleCombinations;
    }

    for (auto entry : sumContainerGroups) {
        entry.second.clear();
    }

    fibNumbers.clear();
    fibSums.clear();
    return combinations;
}

int main() {
    std::cout << getFibCombinations(100) << std::endl;
    return 0;
}
