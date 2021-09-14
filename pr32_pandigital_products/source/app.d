import std.stdio;
import std.container.rbtree;

/+
Thinking about the problem:
Legal configurations:
9 digits total

1 2345 6789
12 345 6789

+/

bool isPanDigital(int multiplier1, int multiplier2, int product) {
	int totalDigits = 0;

	int[9] digits = [ 1, 2, 3, 4, 5, 6, 7, 8, 9 ];

	int[3] numbers = [multiplier1, multiplier2, product];
	foreach (int number; numbers) {
		int tempNumber = number;

		// TODO -- something is very slow (the while loop?)
		while (tempNumber > 0) {
			totalDigits++;
			int digit = tempNumber % 10;
			// screens out 0s
			if (digit != 0) {
				// repeat zeros are an auto-fail
				if (!digits[digit - 1]) {
					return false;
				}
				digits[digit - 1] = 0;
			}
			else {
				return false;
			}
			tempNumber = (tempNumber - digit) / 10;
		}
	}

	// catch the cases with extra digits
	if (totalDigits == 9) {
		printf("%d, %d, %d\n", multiplier1, multiplier2, product);
		return true;
	}
	else {
		return false;
	}
}

long getProductSumForFirstMultiplier(int multiplier) {
	long productSum = 0;
	//int[] products = new int[10000];
	// we need 0 in the initialization so the tree knows it contains ints
	auto productTree = redBlackTree(0);

	if (multiplier < 10) {
		// we loop for 4 digits if we only use 1 digit
		// 1111 is the first number without a 0 with 4 digits
		for (int i = 1111; i < 10000; i++) {
			const int product = multiplier * i;
			if (isPanDigital(multiplier, i, product)) {
				productTree.insert(product);
			}
		}
	}
	else {
		// we loop for 3 digits
		for (int i = 111; i < 1000; i++) {
			const int product = multiplier * i;
			if (isPanDigital(multiplier, i, product)) {
				productTree.insert(product);
			}
		}
	}

	foreach (int product; productTree[]) {
		//printf("%d\n", product);
		productSum += product;
	}

	return productSum;
}

long getProductSum() {
	auto productTree = redBlackTree(cast(long)0);

	long productSum = 0;

	// we will loop over numbers from 1-99, eliminating %11 (which are automatic fails)
	// and %10 (we can't use 0)
	for (int i = 1; i <= 99; i++) {
		if (i % 11 != 0 && i % 10 != 0) {
			productTree.insert(getProductSumForFirstMultiplier(i));
		}
	}

	// we use a tree for deduplication
	foreach (product; productTree[]) {
		productSum += product;
	}

	return productSum;
}

void main()
{
	printf("Sum: %ld", getProductSum());
}
