#include <iostream>
#include <fstream>

double getPower10(int number, int power) {
    auto powered = (double) number;

    while (power > 0) {
        powered *= 10;
        power--;
    }

    return powered;
}

// we will load only the first 15 digits of each number
double sum_of_big_numbers(int digitsToGet) {
    double sum = 0;

    std::fstream newFile;
    newFile.open("/home/smooth_operator/fun/euler/pr13_large_sum/large_numbers.txt");
    if (newFile.is_open()){   //checking whether the file is open
        std::string numberString;
        while(getline(newFile, numberString)){ //read data from file object and put it into string.
            for (int digitLocation = 0; digitLocation < digitsToGet; digitLocation++) {
                sum += getPower10(
                        numberString[digitLocation] - '0',
                        digitsToGet - digitLocation - 1
                );
            }

        }
        newFile.close(); //close the file object.
    }

    return sum;
}

int main() {
    double sum = sum_of_big_numbers(15);
    // if we wanted JUST the first 10 digits, I would create an array, and use %10 and /10 repeatedly until the number
    // vanished. The first digits would end up in the last elements of the array.

    std::cout << sum << std::endl;
    return 0;
}
