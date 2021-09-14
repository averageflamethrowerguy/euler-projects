import std.stdio;
import std.math;

/+
This one seems like a little bit nastier of a problem
It's similar to an earlier a^2 + b^2 = c^2 problem
+/

int getNumberOfSolutions(int parimeter) {
	int numberSolutions = 0;

	// we know that the shortest the hypotenuse can be is
	// 1 / sqrt(2) * the combined length of a and b
	// and the longest it can be is == a + b.
	// we will remove duplicates by asserting a >= b

	// the min size is parimeter / (1 + sqrt(2))

	int lowerBoundHypotenuse = cast(int) floor(cast(double) parimeter / (cast(double)1 + sqrt(cast(double)2)));
	// keep the hypotenuse bounded
	for (int c = lowerBoundHypotenuse; c <= parimeter / 2; c++) {
		// the upper bound of a is c; the lower bound is 1/2 c
		for (int a = cast(int) ceil(cast(double)c / 2); a <= c; a++) {
			int b = parimeter - c - a;

			// we verify that this is a valid solution
			if (c*c == a*a + b*b) {
				numberSolutions++;
			}
		}
	}

	return numberSolutions;
}

// maxValue is inclusive
int findMaximumSolutionsForParimeter(int maxValue) {
	int maximumNumberOfSolutions = 0;
	int maximumSolutionsParimeter;

	for (int i = 4; i <= maxValue; i++) {
		int numberOfSolutions = getNumberOfSolutions(i);

		if (numberOfSolutions > maximumNumberOfSolutions) {
			maximumNumberOfSolutions = numberOfSolutions;
			maximumSolutionsParimeter = i;
		}
	}

	return maximumSolutionsParimeter;
}

void main()
{
	printf("%d", findMaximumSolutionsForParimeter(1000));
	writeln("Edit source/app.d to start your project.");
}
