#include <iostream>
#include <vector>
#include <fstream>

/*
 * GAME_PLAN:
 * 1. Do sorting on GPU
 * 2. Assess scores on GPU
 * 3. Sum scores (try first with long, then other datatype?)
 */

#define MAX_DEPTH       16
#define INSERTION_SORT  32
#define max_name_length 13

// returns -1 if name1 is larger, 1 if name2 is larger (reverse order), 0 if they are the same
__device__ int compare_names(const char* name1, const char* name2, int name_length) {
    int currentCheckedChar = 0;
    char charName1 = name1[0];
    char charName2 = name2[0];

    // these handle the case of zeroed values as well -- zero is smaller than any letter and will come first
    while (currentCheckedChar < name_length) {
        if (charName1 > charName2) {
            return -1;
        }
        else if (charName2 > charName1) {
            return 1;
        }

        currentCheckedChar++;
        charName1 = name1[currentCheckedChar];
        charName2 = name2[currentCheckedChar];
    }
    return 0;
}

// replaces the name in memory (including the zero padding)
__device__ void replace_name(char* targetLoc, char* name, int name_length ) {
    int currentChar = 0;
    while (currentChar < name_length) {
        targetLoc[currentChar] = name[currentChar];

        currentChar++;
    }
}

// swaps the names in memory (including the zero padding)
__device__ void swap_names(char* name1, char* name2, int name_length ) {
    int currentChar = 0;
    while (currentChar < name_length) {
        char tempChar = name1[currentChar];
        name1[currentChar] = name2[currentChar];
        name2[currentChar] = tempChar;

        currentChar++;
    }
}

// this comes into play when the names list is getting smaller.
__device__ void insertion_sort(char* names, int left, int right, int name_length) {
    // we cache the value of the array
    char* name;
    name = (char *) malloc(sizeof(char) * name_length);

    // iterates through the entire array that we can legally access
    for (int i = left; i <= right; i++ ) {
        // we copy the char array from modifiable memory into the name array
        for (int copyLoc = 0; copyLoc < max_name_length; copyLoc++) {
            // (names + (i * name_length)) is a pointer
            name[copyLoc] = (names + (i * name_length))[copyLoc];
        }

        int j = i - 1;

        while (j >= left && compare_names(
                    names + (j * name_length),
                    name,
                    name_length
                ) == -1
            ) {
            replace_name(
                    names + ((j + 1) * name_length),
                    names + (j * name_length),
                    name_length
            );
            j--;
        }

        replace_name(
                names + ((j + 1) * name_length),
                name,
                name_length
        );
    }

    // free at the end
    free(name);
}

// the algorithm to partition the array
__device__ int partition(char* names, int left, int right, int name_length) {
    char* pivot = names + (name_length * right);
    int i = left - 1;

    for (int j = left; j < right; j++) {
        if (compare_names(
                names + (j * name_length),
                pivot,
                name_length
            ) == 1
        ) {
            i++;
            swap_names(
                    names + (j * name_length),
                    names + (i * name_length),
                    name_length
            );
        }
    }
    swap_names(
            names + (right * name_length),
            names + ((i + 1) * name_length),
            name_length
    );
    return (i + 1);
}

__global__ void quicksort(char* names, int left, int right, int depth, int name_length) {
    // we perform insertion sort if too deep (or on small arrays)
    if (depth >= MAX_DEPTH || right-left <= INSERTION_SORT) {
        insertion_sort(names, left, right, name_length);
        return;
    }
    else {
        int pivotIndex = partition(names, left, right, name_length);

        cudaStream_t s;
        cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
        // NOTE -- this error is shown even though I have set compilation to CC 8.6.
        // the code compiles fine.
        quicksort<<< 1, 1, 0, s >>>(names, left, pivotIndex - 1, depth+1, name_length);
        cudaStreamDestroy(s);

        cudaStream_t s1;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        // NOTE -- this error is shown even though I have set compilation to CC 8.6.
        // the code compiles fine.
        quicksort<<< 1, 1, 0, s1 >>>(names, pivotIndex + 1, right, depth+1, name_length);
        cudaStreamDestroy(s1);
    }
}

void sort_names_with_cuda(char* names_arrays, int number_names) {
    char* dev_sort = nullptr;

    int memory_size = number_names * max_name_length * (int)sizeof(char);

    // Allocate GPU buffers for three vectors (two input, one output)
    cudaMalloc((void**)&dev_sort, memory_size);

    // Copy input vectors from host memory to GPU buffers.
    cudaMemcpy(dev_sort, names_arrays, memory_size, cudaMemcpyHostToDevice);

    quicksort<<<1, 1, 0>>>(
            dev_sort,
            0,
            number_names-1,
            0,
            max_name_length
    );

    cudaDeviceSynchronize();

    // Copy output vector from GPU buffer to host memory.
    cudaMemcpy(names_arrays, dev_sort, memory_size, cudaMemcpyDeviceToHost);

    int currentNameIndex = 0;
    while (currentNameIndex < number_names) {
        for (int i = 0; i < max_name_length; i++) {
            printf("%c", names_arrays[(currentNameIndex * max_name_length) + i]);
        }

        printf("\n");
        currentNameIndex++;
    }

    cudaFree(dev_sort);
    cudaDeviceReset();
}

long get_name_scores_total() {
    std::vector<std::string> names;

    // load the file into strings
    std::fstream newFile;
    newFile.open("/home/smooth_operator/fun/euler/pr22_names_scores/names.txt");
    if (newFile.is_open()) {   //checking whether the file is open
        std::string nameString;
        while (getline(newFile, nameString)) { //read data from file object and put it into string.
            names.push_back(nameString);
        }
    }

    /*
     * We now do a few things to allow copying the strings to GPU memory:
     * 1. We find the longest string (which we will pad one past)
     * 2. We will initialize an n x m array of chars, where n is:
     *      the number of strings in the vector
     *      and m is:
     *      the length of the longest string (plus 1)
     */

    int numberNames = (int)names.size();
    char* names_arrays;
    names_arrays = (char *) std::malloc(numberNames * max_name_length * sizeof(char));
    memset(names_arrays, 0, numberNames * max_name_length * sizeof(char));

    for (int i = 0; i < names.size(); i++) {
        for (int j = 0; j < max_name_length; j++) {
            if (j < names.at(i).length()) {
                names_arrays[(i * max_name_length) + j] = names.at(i).at(j);
            }
            else {
                break;
            }
        }
    }

    sort_names_with_cuda(names_arrays, numberNames);

    long scores_total = 0;

    free(names_arrays);
    return scores_total;
}

int main() {
    std::cout << get_name_scores_total() << std::endl;
    return 0;
}
