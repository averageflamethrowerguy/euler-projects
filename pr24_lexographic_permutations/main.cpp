#include <iostream>
#include <vector>

#define DIGIT_LENGTH 10
#define SEARCH_LOC 1000000

std::vector<char *> getPermutations(const char* usedDigits, int numberUsed) {
    std::vector<char *> returnVector;
    if (numberUsed == DIGIT_LENGTH) {
        char* perm;
        perm = (char *) malloc(sizeof(char) * (numberUsed));
        for (int i = 0; i < numberUsed; i++) {
            perm[i] = usedDigits[i];
        }

        returnVector.push_back(perm);
    }

    else {
        int newNumberUsed = numberUsed + 1;
        for (int i = 0; i < DIGIT_LENGTH; i++) {
            int isUsed = 0;
            for (int digitIndex = 0; digitIndex < numberUsed; digitIndex++) {
                if (i == (int)(usedDigits[digitIndex] - '0')) {
                    isUsed = 1;
                }
            }

            if (!isUsed) {
                char* perm;
                perm = (char *) malloc(sizeof(char) * (newNumberUsed));

                // constructs a new partial permutation
                int digitIndex = 0;
                for (; digitIndex < numberUsed;) {
                    perm[digitIndex] = usedDigits[digitIndex];
                    digitIndex++;
                }

                perm[digitIndex] = (char)(i + '0');

                // recursively calls the function with the partial permutation
                std::vector<char *> tempVec = getPermutations(perm, newNumberUsed);

                for (char* tempPerm : tempVec) {
                    returnVector.push_back(tempPerm);
                }

                free(perm);
                tempVec.clear();
            }
        }
    }

    return returnVector;
}

long get_millionth_permutation() {
    char *empty = nullptr;
    std::vector<char *> permutations = getPermutations(empty, 0);

    int permNumber = 0;
    for (char* perm : permutations) {
        permNumber++;
        if (permNumber == SEARCH_LOC) {
            for (int permIndex = 0; permIndex < DIGIT_LENGTH; permIndex++) {
                std::cout << perm[permIndex];
            }
            std::cout << std::endl;
        }

        free(perm);
    }

    return 0;
}

int main() {
    std::cout << get_millionth_permutation() << std::endl;
    return 0;
}
