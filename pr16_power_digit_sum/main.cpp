#include <iostream>
#include <vector>
#include <cmath>

/*
 * 2: 2
 * 4: 4
 * 8: 8
 * 16: 7
 * 32: 5
 * 64: 10
 * 128: 11
 * 256: 13
 * 512: 8
 * 1024: 7
 * 2048: 14
 * 4096: 19
 *
 *  ... the ones follows the pattern 2,4,8,6
 *  ... the last two digits follow a cycle of length 20
 *  ... the last three have a cycle of length 100
 *  https://www.exploringbinary.com/patterns-in-the-last-digits-of-the-positive-powers-of-two/
 *
 *  ...but this probably gets less useful for digits > 3
 */


std::vector<int> multiplyVector(std::vector<int> vec, int multiplyBy) {
    int carryOver = 0;
    for (int i = 0; i < vec.size(); i++) {
        int temp = vec.at(i);
        temp = temp * multiplyBy + carryOver;
        std::cout << temp << std::endl;

        carryOver = floor(temp / 10);
        if (carryOver > 0) {
            temp = temp % 10;
        }
        else {
            carryOver = 0;
        }

        vec.at(i) = temp;
    }

    if (carryOver > 0) {
        vec.push_back(carryOver);
    }

    return vec;
}

long sumVector(const std::vector<int>& vec) {
    long sum = 0;

    for (int i : vec) {
        std::cout << i << "\n" << std::endl;
        sum += i;
    }

    return sum;
}

long power_digit_sum(int number, int power) {
    std::vector<int> vec;
    vec.push_back(1);

    int currentPower = 0;
    while (currentPower < power) {
        vec = multiplyVector(vec, number);
        currentPower++;
    }

    long sum = sumVector(vec);
    vec.clear();
    return sum;
}

int main() {
    std::cout << power_digit_sum(2, 1000) << std::endl;
    return 0;
}
