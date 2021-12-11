import kotlin.math.sqrt

/*
loop {
if num is prime {
check if there are digits 0, 1, 2 {
if so, take draw 1 permutations, draw 2 permutations, draw 3 permutations, etc {
sub out these permutations with 2,3,4,5,6,7,8,9 (and 1 if we started with 0) {
check if the resulting numbers are also prime
}
}
}
}
}
 */

fun isPrime(smallerPrimes: List<Int>, num: Int): Boolean {
    val root : Double = sqrt(num.toDouble())
    var currentPrimeIndex = 0
    var currentPrime = smallerPrimes[0]

    if (num == 1) {
        return false
    }

    while (currentPrime <= root && (currentPrimeIndex < smallerPrimes.size)) {
        if (num % currentPrime == 0) {
           return false
        }

        currentPrimeIndex += 1
        if (currentPrimeIndex < smallerPrimes.size) {
            currentPrime = smallerPrimes[currentPrimeIndex]
        }
    }

    return true
}

fun increment(key: Int, digitLocation: Int, map: HashMap<Int, MutableList<Int>>) {
    if (!map.contains(key)) {
        map[key] = mutableListOf(digitLocation)
    } else {
        (map[key])!!.add(digitLocation)
    }
}

/**
 * Gets the map of the digits that contains 0, 1, 2 from the number
 */
fun getTargetDigitMap(num: Int, threshold: Int) : HashMap<Int, MutableList<Int>>? {
    val targetDigits = (0..(10-threshold)).toList().toTypedArray()
    var numCopy = num
    var targetDigitMap : HashMap<Int, MutableList<Int>>? = null

    var digit = 0
    while (numCopy > 0) {
        if (targetDigits.contains(numCopy % 10)) {
            if (targetDigitMap == null) {
                targetDigitMap = HashMap()
            }
            increment(numCopy % 10, digit, targetDigitMap)
        }

        numCopy /= 10
        digit++
    }

    return targetDigitMap
}

/**
 * A recursive function to get all the location combinations
 * [0,1,2] ->
 * [[0], [1], [2], [0,1], [1,2], [0,2], [0,1,2]]
 */
fun getLocationCombinations(
    digitLocations: List<Int>,
    previousLocations: List<Int>,
) : List<List<Int>> {
    val combinations = mutableListOf<List<Int>>()

    for (i in digitLocations.indices) {
        // add to combinations, using any previousLocations generated prior
        combinations.add(previousLocations.toList() + digitLocations[i])
        // will recurse again if there is any array left
        if (i+1 < digitLocations.size) {
            // add the current digit location
            val tempPrevLocations = ArrayList(previousLocations)
            tempPrevLocations.add(digitLocations[i])

            // recursively call
            combinations.addAll(getLocationCombinations(
                // taking this sublist prevents reuse of digits toward the bottom
                digitLocations.subList(i+1, digitLocations.size),
                tempPrevLocations
            ))
        }
    }

    return combinations
}

fun getPowerOf(base: Int, power: Int) : Int {
    var result = base
    var currentPower = 1

    if (power == 0) {
        return 1
    }

    while (currentPower < power) {
        result *= base
        currentPower++
    }

    return result
}

/**
 * Tests all the combinations of the same digits
 * @return 0 if no valid combo, the smallest number if there is a valid combo
 */
fun testCombinations(
    threshold: Int,
    primes: List<Int>,
    num: Int,
    digit: Int,
    digitLocations: List<Int>
) : Int {
    // gets the location combinations
    val locationCombinations = getLocationCombinations(digitLocations, ArrayList())

    println("$num, $digit")
    println(locationCombinations)

    var outerSmallestPrime = Int.MAX_VALUE

    for (combination in locationCombinations) {
        var immutableComponent = num
        var mutableComponent = 0

        for (location in combination) {
            val multiplyNum = getPowerOf(10, location)

            // removes a digit at the critical location
            immutableComponent = ((immutableComponent / (10 * multiplyNum)) * 10 * multiplyNum) + immutableComponent % multiplyNum
            // adds one to the location we care about
            mutableComponent += multiplyNum
        }

        var primeCount = 1
        var remainingDigits = 9
        var hasFailed = false
        var smallestPrime : Int? = null
        for (i in 0..9) {
            // success is now impossible if too many primes have failed
            if (remainingDigits + primeCount < threshold) {
                hasFailed = true
                break
            }
            // we won't allow 0 if we're changing the leading digit
            else if (i == 0 && mutableComponent > immutableComponent) {
                remainingDigits--
            }
            // avoid testing the same digit
            else if (i != digit) {
                if (isPrime(primes, immutableComponent + mutableComponent * i)) {
                    primeCount++
                    // keep track of the smallest prime
                    if (smallestPrime == null || (immutableComponent + mutableComponent * i) < smallestPrime) {
                        smallestPrime = (immutableComponent + mutableComponent * i)
                    }
                }
                remainingDigits--
            }
        }

        if (primeCount < threshold) {
            hasFailed = true
        }

//        println("Prime count: $primeCount, hasFailed: $hasFailed")

        if (!hasFailed && smallestPrime!! < outerSmallestPrime) {
            outerSmallestPrime = smallestPrime
        }
    }

    return if (outerSmallestPrime == Int.MAX_VALUE) {
        0
    } else {
        if (outerSmallestPrime < num) {
            outerSmallestPrime
        } else {
            num
        }
    }
}

fun findSmallestEightPrimeMember(threshold: Int) : Int {
    var num = 3
    val primes : MutableList<Int> = mutableListOf()
    primes.add(2)
    var smallestWorkingPrime = Int.MAX_VALUE

    while (num < smallestWorkingPrime) {
        if (isPrime(primes, num)) {
            primes.add(num)

            if (num > 10) {
                val digits = getTargetDigitMap(num, threshold)
                if (digits != null) {
                    // iterates over the digits of interest
                    for (digitEntry in digits.entries) {
                        val returnVal = testCombinations(
                            threshold, primes, num, digitEntry.key, digitEntry.value
                        )
                        println("$num marked $returnVal")
                        if (returnVal != 0 && returnVal < smallestWorkingPrime) {
                            smallestWorkingPrime = returnVal
                        }
                    }
                }
            }
        }

        num++

    }

    return smallestWorkingPrime
}

fun main() {
//    val thing : List<Array<Int>> = getLocationCombinations(
//        listOf(0,1,2,3),
//        ArrayList()
//    )
//    for (thing1 in thing) {
//        var printer = "["
//        for (digit in thing1) {
//            printer += digit
//            printer += ", "
//        }
//        printer += "], "
//        println(printer)
//    }

    // 121313
    println(findSmallestEightPrimeMember(8))
}