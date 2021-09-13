import std.stdio;

/+
The ideal pandigital is 987654321.
We're trying to get as close to this as possible.

We need n to be at least 2.
That means that the first 4 digits are the number
and the last five may be the number * 2.
+/

bool testIfPandigital(int sum) {
	int[9] digits = [ 1, 2, 3, 4, 5, 6, 7, 8, 9];

	while (sum > 0) {
		int testDigit = sum % 10;
		// zeros are not allowed
		if (testDigit == 0) {
			return false;
		}

		sum = (sum - testDigit) / 10;

		// we make sure each digit may only be used once
		if (digits[testDigit - 1] == testDigit) {
			digits[testDigit - 1] = 0;
		}
		else {
			return false;
		}
	}

	// we check to make sure all digits have been used
	// this catches sums that are too small
	foreach (digit; digits) {
		if (digit) {
			return false;
		}
	}

	return true;
}

int getNumberDigits(int number) {
	int numberDigits = 0;

	while (number > 0) {
		numberDigits++;
		number /= 10;
	}

	return numberDigits;
}

int getPower(int number, int power) {
	int constructedPower = 1;

	while (power > 0) {
		constructedPower *= number;
		power--;
	}

	return constructedPower;
}

int findLargestPandigital() {
	int largestPandigital = 0;

	for (int i = 1; i <= 9876; i++) {
		int numberDigits = 0;
		int constructedSum = 0;

		int multiplier = 1;
		while (numberDigits < 9) {
			int newNumber = i * multiplier;
			int numberSize = getNumberDigits(newNumber);
			// concat the new number onto the end
			constructedSum = constructedSum * getPower(10, numberSize) + newNumber;
			numberDigits = getNumberDigits(constructedSum);
			multiplier++;
			//printf("%d\n", constructedSum);
		}

		if (numberDigits == 9 && testIfPandigital(constructedSum)) {
			if (constructedSum > largestPandigital) {
				printf("Current pandigital: %d\n", constructedSum);
				largestPandigital = constructedSum;
			}
		}
	}

	return largestPandigital;
}

void main()
{
	int pandigital = findLargestPandigital();
	printf("Largest Pandigital: %d\n", pandigital);
}
