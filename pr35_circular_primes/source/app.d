import std.stdio;
import std.math;

/+
We will cache previous primes and prime checks in an array to speed up
the prime finding process
+/

// inclusive
const int maxNumber = 999999;
// we will use this to cache primes; also intelligently find non-primes
bool[maxNumber + 1] isPrimeArray;

// maxSearchedNumber is the max value we have searched for primes; we
// can use the array underneath it
bool isPrime(int number, int maxSearchedNumber) {
	if (number < maxSearchedNumber) {
		return isPrimeArray[number];
	}
	// we use the prime finding algo
	else {
		double root = sqrt(cast(double)number);

		for (int i = 2; i <= root; i++) {
			if (number % i == 0) {
				isPrimeArray[number] = false;
				return false;
			}
		}

		isPrimeArray[number] = true;
		return true;
	}
}

int getPower(int number, int power) {
	int constructedPower = 1;
	while (power > 0) {
		constructedPower *= number;
		power--;
	}

	return constructedPower;
}

bool isCircularPrime(int number) {
	int tempNumber = number;

	do {
		if (!isPrime(tempNumber, number)) {
			return false;
		}
		// reorder the digits by popping the bottom one off and adding to the top.
		int bottomDigit = tempNumber % 10;

		int secondTemp = tempNumber;
		int numberLength = 0;

		// we count the number of digits
		while (secondTemp > 0) {
			numberLength++;
			secondTemp = secondTemp / 10;
		}

		// we cut the last digit off
		tempNumber = (tempNumber - (tempNumber % 10)) / 10;

		// we multiply the last digit by 10^(numberDigits - 1) to place it at the top
		tempNumber += bottomDigit * getPower(10, numberLength - 1);

		// we stop reordering when the numbers line up again
	} while (tempNumber != number);

	return true;
}

int getCircularPrimesCount() {
	int numberCircularPrimes = 0;

	for (int i = 2; i <= maxNumber; i++) {
		if (isCircularPrime(i)) {
			numberCircularPrimes++;
		}
	}

	return numberCircularPrimes;
}

void main()
{
	printf("Number Circular Primes: %d", getCircularPrimesCount());
	// 55
}
