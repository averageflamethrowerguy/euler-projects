#include <iostream>
#include <vector>
#include <cmath>

// TODO -- we need to allow multiplyBy to be >10 for this to work.
std::vector<int> multiplyVector(std::vector<int> vec, int multiplyBy) {
    int carryOver = 0;
    for (int i = 0; i < vec.size(); i++) {
        int temp = vec.at(i);
        temp = temp * multiplyBy + carryOver;

        carryOver = floor((double) temp / 10);
        if (carryOver > 0) {
            temp = temp % 10;
        }
        else {
            carryOver = 0;
        }

        vec.at(i) = temp;
    }

    if (carryOver > 0) {
        while (carryOver) {
            vec.push_back(carryOver % 10);
            carryOver = floor((double)carryOver / 10);
        }
    }

    return vec;
}

long sumVector(const std::vector<int>& vec) {
    long sum = 0;

    for (int i : vec) {
        std::cout << i << std::endl;
        sum += i;
    }

    return sum;
}

long factorial_sum(int number) {
    std::vector<int> vec;
    vec.push_back(1);

    int currentMultiplier = 1;
    while (currentMultiplier <= number) {
        vec = multiplyVector(vec, currentMultiplier);
        currentMultiplier++;
    }

    long sum = sumVector(vec);
    vec.clear();
    return sum;
}

int main() {
    std::cout << factorial_sum(100) << std::endl;
    return 0;
}
