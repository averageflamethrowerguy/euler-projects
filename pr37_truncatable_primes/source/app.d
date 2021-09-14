import std.stdio;
import std.container.rbtree;
import std.math;

/+
This is a bounding problem. We know that truncatable primes must be
composed of "Left truncatable" and "Right truncatable" primes.

Alternatively, we can just use the fact that there are only 11 primes truncatable
primes to bound our search...
+/

bool isPrime(int number) {
	double root = sqrt(cast(double)number);

	for (int i = 2; i <= root; i++) {
		if (number % i == 0) {
			return false;
		}
	}

	return true;
}

bool isRightTruncatable(int number, RedBlackTree!int tree) {
	// cut off the bottom digit
	int tempNumber = (number - (number % 10)) / 10;
	if (!tree.opBinaryRight!"in"(tempNumber)) {
		return false;
	}

	if (!isPrime(number)) {
		return false;
	}

	return true;
}

bool isLeftTruncatable(int number, RedBlackTree!int tree) {
	// cut off the top digit
	int powerTen = 1;
	while (powerTen < number) {
		powerTen *= 10;
	}
	powerTen /= 10;

	// kill the top digit
	int tempNumber = number % powerTen;
	if (!tree.opBinaryRight!"in"(tempNumber)) {
		return false;
	}

	if (!isPrime(number)) {
		return false;
	}

	return true;
}

long sumOfTruncatablePrimes() {
	int numberTruncatablePrimes = 0;
	long sum = 0;
	auto leftTruncatable = redBlackTree(2, 3, 5, 7);
	auto rightTruncatable = redBlackTree(2, 3, 5, 7);

	int number = 11;
	while (numberTruncatablePrimes < 11) {
		bool isRightTrunc = false;

		if (isRightTruncatable(number, rightTruncatable)) {
			rightTruncatable.insert(number);

			isRightTrunc = true;
		}
		if (isLeftTruncatable(number, leftTruncatable)) {
			leftTruncatable.insert(number);

			if (isRightTrunc) {
				sum += number;
				numberTruncatablePrimes++;
			}
		}

		number++;
	}

	return sum;
}

void main()
{
	long sum = sumOfTruncatablePrimes();
	printf("%ld", sum);
	//748317
}
