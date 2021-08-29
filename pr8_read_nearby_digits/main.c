#include <stdio.h>
#include <malloc.h>
#include <stdlib.h>

typedef struct {
    int *array;
    size_t used;
    size_t size;
} StaticArray;

void initArray(StaticArray *a, size_t initialSize) {
    a->array = malloc(initialSize * sizeof(int));
    a->used = 0;
    a->size = initialSize;
}

void insertArray(StaticArray *a, int element) {
    // if the used array is the same size of the total array, we push all existing elements down by one
    if (a->used == a->size) {
        int temp = a->array[(int)a->size - 1];
        // starts one spot from the top of the array
        for (int i = (int)a->size - 2; i >= 0; i--) {
            int temp2 = a->array[i];
            a->array[i] = temp;
            temp = temp2;
        }
        // inserts element at the end of the array
        a->array[(int)a->size - 1] = element;
    }
    else {
        // we allocate the last index to the element
        a->array[a->used++] = element;
    }
}

void freeArray(StaticArray *a) {
    free(a->array);
    a->array = NULL;
    a->used = a->size = 0;
}

long get_greatest_product() {
    long score = 0;
    StaticArray lastThirteen;
    initArray(&lastThirteen, 13);

    FILE * fp = fopen(
    "/home/smooth_operator/fun/euler/pr8_read_nearby_digits/number.txt",
    "r"
    );
    if (fp == NULL) {
        printf("File is not available\n");
        return EXIT_FAILURE;
    }
    else {
        char ch;
        while((ch = fgetc(fp)) != EOF) {
            insertArray(&lastThirteen, ch - '0');
            if (lastThirteen.used == lastThirteen.size) {
                long tempScore = 1;
                for (int i = 0; i < lastThirteen.size; i++) {
//                    printf("%d\n", lastThirteen.array[i]);
                    tempScore *= lastThirteen.array[i];
                }

                if (tempScore > score) {
                    printf("%ld\n", tempScore);
                    score = tempScore;
                }
            }
        }
    }

    fclose(fp);
    freeArray(&lastThirteen);
    return score;
}

int main() {
    printf("%ld", get_greatest_product());
    return 0;
}
