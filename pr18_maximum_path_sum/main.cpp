#include <iostream>
#include <vector>
#include <fstream>

/*
 * CHALLENGES:
 *
 * Step 1: Read into mem as an array of arrays
 * Step 2: Start from the bottom. Each cell's score += score_of_larger_cell_below
 * Step 3: Read the score of the top cell.
 */

int get_greatest_path() {
    std::vector<std::vector<int>> vec;

    std::fstream newFile;
    newFile.open("/home/smooth_operator/fun/euler/pr18_maximum_path_sum/triangle.txt");
    if (newFile.is_open()){   //checking whether the file is open
        std::string numberString;
        while(getline(newFile, numberString)){ //read data from file object and put it into string.
            std::vector<int> lower_vec;

            int accumulator = 0;
            for (char digitLocation : numberString) {
                if (digitLocation == ' ') {
                    // save before zeroing
                    lower_vec.push_back(accumulator);
                    accumulator = 0;
                }
                else {
                    // to deal with numbers in the 10s place
                    if (accumulator) {
                        accumulator *= 10;
                    }
                    accumulator += digitLocation - '0';
                }
            }

            // always save the last one
            lower_vec.push_back(accumulator);
            vec.push_back(lower_vec);
        }
        newFile.close(); //close the file object.
    }

    for (int i = (int) vec.size() - 2; i >= 0; i--) {
        for (int j = 0; j < vec.at(i).size(); j++) {
            if (vec.at(i + 1).at(j) > vec.at(i + 1).at(j + 1)) {
                vec.at(i).at(j) += vec.at(i + 1).at(j);
            }
            else {
                vec.at(i).at(j) += vec.at(i + 1).at(j + 1);
            }
        }
    }

    int greatest = vec.at(0).at(0);

    for (std::vector<int> comp_vec : vec) {
        comp_vec.clear();
    }
    vec.clear();

    return greatest;
}

int main() {
    std::cout << get_greatest_path() << std::endl;
    return 0;
}
