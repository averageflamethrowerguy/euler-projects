import std.stdio;
import std.format;

int getOvershotDigit(int number, int overshotAmount) {
	printf("Number: %d, overshotAmount: %d", number, overshotAmount);

	auto numberString = format("%d", number);

	int currentDigit = numberString[numberString.length - 1 - overshotAmount];
	printf("Current Digit: %d\n", currentDigit - '0');
	return currentDigit - '0';
}

int getPower(int number, int power) {
	int constructedPower = 1;
	while (power > 0) {
		constructedPower *= number;
		power--;
	}

	return constructedPower;
}

int getNumberDigits(int number) {
	int numberDigits = 0;

	while (number > 0) {
		numberDigits++;
		number /= 10;
	}

	return numberDigits;
}

int getOrdersOfMagProduct(int ordersOfMag) {
	int orderOfMagProduct = 1;

	int currentOrderOfMag = 0;
	int currentDigitCount = 0;
	int currentNumber = 1;

	while (currentOrderOfMag <= ordersOfMag) {
		int targetQuantity = getPower(10, currentOrderOfMag);

		while (currentDigitCount < targetQuantity) {
			currentDigitCount += getNumberDigits(currentNumber);
			currentNumber++;
		}

		// we check to see how much we overshot, and then collect the digit
		// from the corresponding location on the previous digit
		int overshotAmount = currentDigitCount - targetQuantity;

		orderOfMagProduct *= getOvershotDigit(currentNumber - 1, overshotAmount);

		currentOrderOfMag++;
	}

	return orderOfMagProduct;
}

void main()
{
	printf("%d", getOrdersOfMagProduct(6));
}
