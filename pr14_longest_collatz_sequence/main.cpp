#include <iostream>
#include <map>

// the collatz sequence tester
// n is even -> n = n/2
// n is odd -> n = 3n + 1
int get_collatz_length(long number, const std::map<long, int>& collatz_history) {
    // there will always be at least one term in the sequence
    int numberIterations = 1;

    while (number != 1) {
        auto it = collatz_history.find(number);

        // we will intercept the loop and truncate it immediately if the current number is in the history
        if (it != collatz_history.end()) {
            numberIterations += it->second;
            number = 1;
        }
        else {
            if (number % 2 == 0) {
                number = number / 2;
            }
            else {
                number = 3 * number + 1;
            }
        }

        numberIterations++;
    }

    return numberIterations;
}

// the control loop
long get_longest_collatz(int max_num) {
    std::map<long, int> collatz_history;
    long longestCollatz = 0;
    int longestCollatzLength = 0;

    for (long testNum = 1; testNum < max_num; testNum++) {
        int length = get_collatz_length(testNum, collatz_history);
        collatz_history.insert({testNum, length});

        if (length > longestCollatzLength) {
            longestCollatz = testNum;
            longestCollatzLength = length;
        }
    }

    collatz_history.clear();
    return longestCollatz;
}

int main() {
    std::cout << get_longest_collatz(1000000) << std::endl;
    return 0;
}
