import std.stdio;
import std.math;
import std.format;

/+
We're just going to hard-code base10 and binary
+/

bool isBinaryPalidrome(int number) {
	// creats a string with the binary representation of the number
	auto binaryDigits = format("%b", number);

	// now we have the digits of the number in a string
	// we will compare the digits (up to half the array, no need to repeat!)
	for (int i = 0; i < ceil(cast(float)binaryDigits.length / 2); i++) {
		// start with the first and last binaryDigits and compare inward
		if (binaryDigits[i] != binaryDigits[binaryDigits.length - 1 - i]) {
			// we've failed
			return false;
		}
	}

	return true;
}

bool isDecimalPalindrome(int number) {
	auto digits = format("%d", number);

	// now we have the digits of the number in a string
	// we will compare the digits (up to half the array, no need to repeat!)
	for (int i = 0; i < ceil(cast(float)digits.length / 2); i++) {
		// start with the first and last digits and compare inward
		if (digits[i] != digits[digits.length - 1 - i]) {
			// we've failed
			return false;
		}
	}

	return true;
}

// highestNumber is inclusive
long getSumOfDoublePalindromes(int highestNumber) {
	long sum = 0;

	for (int i = 0; i <= highestNumber; i++) {
		if (isDecimalPalindrome(i) && isBinaryPalidrome(i)) {
			sum += i;
		}
	}

	return sum;
}

void main()
{
	long sum = getSumOfDoublePalindromes(999999);
	printf("Sum: %ld", sum);
	// supposedly 872187
}




//int numberDigits = 0;
//int tempNumber = number;
//// we count the number of digits in the number
//while (tempNumber > 0) {
//	numberDigits++;
//tempNumber = (tempNumber - (tempNumber % 10)) / 10;
//}
//
//int[] digits = new int[numberDigits];
//tempNumber = number;
//int index = 0;
//
//// we write into the array with a (reversed) representation of the
//// digits of the original number
//while (tempNumber > 0) {
//	digits[index] = tempNumber % 10;
//tempNumber = (tempNumber - (tempNumber % 10)) / 10;
//index++;
//}