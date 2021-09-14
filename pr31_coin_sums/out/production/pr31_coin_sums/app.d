import std.stdio;

/+
<-- note these weird non-standard comments; they allow nesting

This problem can be broken into a series of recursive steps where each sub-problem
has a maximumSize (in pence) and maximumLevel (in pence)
+/

int[8] coinTypes = [ 1, 2, 5, 10, 20, 50, 100, 200 ];

int getCoinCombinations(int totalMoney, int coinIndex) {
	// this is the base case -- if we get to the coinIndex == 0, there are no more possible combos
	// clearly, the same is true if we run out of money
	//printf("Total money: %d\n", totalMoney);
	if (coinIndex == 0 || totalMoney == 0) {
		return 1;
	}

	// we loop through the totalMoney, starting with no instances of maxCoin and adding one
	// until we no longer can add another maxCoin
	const int maxCoin = coinTypes[coinIndex];
	int possibleComboSum = 0;
	int availableMoney = totalMoney;

	while (availableMoney >= 0) {
		possibleComboSum += getCoinCombinations(availableMoney, coinIndex - 1);
		availableMoney -= maxCoin;
	}

	return possibleComboSum;
}

void main() {
	printf("Coin Combinations: %d", getCoinCombinations(200, cast(int)(coinTypes.length) - 1));
}

/+
Test case: combine to 10:
10
5 5
2 2 1 5
2 1 1 1 5
1 1 1 1 1 1 5
2 2 2 2 2
2 2 2 2 1 1
2 2 2 1 1 1 1
2 2 1 1 1 1 1 1
2 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1
+/
