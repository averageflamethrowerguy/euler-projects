import std.stdio;
import cancelledfraction;

// returns 4 CancelledFractions
auto getNonTrivialFractions() {
	CancelledFraction[4] fractions;
	int cancelledFractionCount = 0;

	// denominators
	for (int i = 11; i < 99; i++) {
		// if i or j are % 10, then they produce trivial outputs
		if (i % 10 != 0) {
			// numerators
			for (int j = 11; j < 99; j++) {
				// if i or j are % 10, then they produce trivial outputs
				// denominator > numerator if value < 1
				if (j % 10 != 0 && i > j) {
					// get the values of first and second digits
					int onesOfI = i % 10;
					int tensOfI = (i - onesOfI) / 10;
					int onesOfJ = j % 10;
					int tensOfJ = (j - onesOfJ) / 10;

					// we care about 4 cases
					int[2] iDigits = [onesOfI, tensOfI];
					int[2] jDigits = [onesOfJ, tensOfJ];

					for (int digitNumI = 0; digitNumI < 2; digitNumI++) {
						int digitNumJ = digitNumI ? 0 : 1;

						// check to see if the opposite digits may be eliminated
						if (iDigits[digitNumI != 0 ? 0 : 1] == jDigits[digitNumJ != 0 ? 0 : 1]) {
							//printf("propNum: %d, propDenom: %d\n", jDigits[digitNumJ], iDigits[digitNumI]);
							//printf("numerator: %d, denominator: %d\n", j, i);

							// check to see if the elimation is valid
							if ((cast(float)jDigits[digitNumJ] / cast(float)iDigits[digitNumI])
							== cast(float)j / cast(float)i
							) {
								//printf("numerator: %d, denominator: %d\n", j, i);
								fractions[cancelledFractionCount] = new CancelledFraction(j, i);
								cancelledFractionCount++;
							}
						}
					}
				}
			}
		}
	}

	foreach (fraction; fractions) {
		printf("numerator: %d, denominator: %d\n", fraction.numerator, fraction.denominator);
	}

	return fractions;
}

int getDenominatorOfFractionProduct() {
	CancelledFraction[4] fractions = getNonTrivialFractions();

	int cumulativeNumerator = 1;
	int cumulativeDenominator = 1;

	foreach (fraction; fractions) {
		cumulativeNumerator *= fraction.numerator;
		cumulativeDenominator *= fraction.denominator;
	}

	printf("numerator: %d, denominator: %d\n", cumulativeNumerator, cumulativeDenominator);

	return 0;
}

void main()
{
	getDenominatorOfFractionProduct();
}
