#include <iostream>
#include <map>
#include <cmath>

long getLetters() {
    std::map<int, int> basic_name_lengths = {
            {1, 3},
            {2, 3},
            {3, 5},
            {4, 4},
            {5, 4},
            {6, 3},
            {7, 5},
            {8, 5},
            {9, 4},
    };

    // overrides for if the number is in the 10s
    std::map<int, int> tens_name_length_overrides = {
            {10, 3},
            {11,6},
            {12,6},
            {13,8},
            {14,8},
            {15,7},
            {16,7},
            {17,9},
            {18,8},
            {19,8},
    };

    // overrides for a number in the 10s place (twenty, thirty)
    std::map<int, int> tens_name_length = {
            {2,6},
            {3,6},
            {4,5},
            {5,5},
            {6,5},
            {7,7},
            {8,6},
            {9,6},
    };

    int add_for_hundred = 7;
    int add_for_and = 3;
    int add_for_thousand = 8;

    long sum = 0;
    for (int i = 1; i <= 1000; i++) {
        int thousands = floor((double) i / 1000);
        int hundreds = floor((double) (i % 1000) / 100);
        int tens = floor((double) (i % 100) / 10);
        int ones = floor((double ) (i % 10) );

        if (thousands) {
            auto it = basic_name_lengths.find(thousands);
            sum += it->second + add_for_thousand;
        }

        if (hundreds) {
            auto it = basic_name_lengths.find(hundreds);
            sum += it->second + add_for_hundred;

            if (tens || ones) {
                sum += add_for_and;
            }
        }

        if (tens) {
            if (tens == 1) {
                sum += tens_name_length_overrides.find(tens * 10 + ones)->second;
            }
            else {
                sum += tens_name_length.find(tens)->second;
            }
        }

        if (ones && tens != 1) {
            sum += basic_name_lengths.find(ones)->second;
        }
    }

    tens_name_length.clear();
    tens_name_length_overrides.clear();
    basic_name_lengths.clear();

    return sum;
}

int main() {
    std::cout << getLetters() << std::endl;
    return 0;
}
