#include <stdio.h>
#include <math.h>

int is_palindrome(int possiblePalindrome) {
    int isPalindrome = 1;
    int length = (int) floorf(log10f((float) possiblePalindrome)) + 1;
    int arr[length];
    int index = 0;

    do {
        arr[index] = possiblePalindrome % 10;
        possiblePalindrome /= 10;
        index++;
    } while (possiblePalindrome != 0);

    int leftIndex = 0;
    int rightIndex = length - 1;

    while (leftIndex < .5 * length && rightIndex >= leftIndex) {
        if (arr[leftIndex] != arr[rightIndex]) {
            isPalindrome = 0;
            break;
        }

        rightIndex -= 1;
        leftIndex += 1;
    }

    return isPalindrome;
}

int find_largest_palindrome(int largestNumber) {
    int left = largestNumber;
    int right = largestNumber;
    int largestPalindrome = 0;

    while (left > 0) {
        while (left * right > largestPalindrome) {
            if (is_palindrome(left * right)) {
                largestPalindrome = left * right;
            }
            right -= 1;
        }

        right = largestNumber;
        left -= 1;
    }

    return largestPalindrome;
}

int main() {
    printf("Largest palindrome is: %d\n", find_largest_palindrome(999));
    return 0;
}
