import std.stdio;

/+
We guess 2200000 as our upper bound, as 9! is much larger than any other
factorial, and 6 * 9! is just under 2.2 mil.
We look at 6 digits because that is the maximum number of 9s that can fit
7 * 9! is smaller than 9,999,999
+/

int getFactorial(int number) {
	int factorial = 1;

	while (number > 0) {
		factorial *= number;
		number--;
	}

	return factorial;
}

bool isFactorialSum(int number) {
	int sum = 0;
	int tempNumber = number;

	while (tempNumber > 0) {
		sum += getFactorial(tempNumber % 10);
		tempNumber = (tempNumber - (tempNumber % 10)) / 10;
	}

	if (number == sum) {
		printf("Number: %d\n", number);
		return true;
	}
	else return false;
}

long sumOfFactorialSum(int upperBoundGuess) {
	long sum = 0;

	for (int i = 10; i < upperBoundGuess; i++) {
		if (isFactorialSum(i)) {
			sum += i;
		}
	}

	return sum;
}

void main()
{
	long sum = sumOfFactorialSum(2200000);
	printf("Sum: %ld", sum);
	// 40730 --> only two numbers count!
}
