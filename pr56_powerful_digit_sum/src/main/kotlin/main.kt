import java.math.BigInteger

fun power(base: Int, exponent: Int) : BigInteger {
    var result = BigInteger.valueOf(1)
    var counter = exponent
    val bigBase = BigInteger.valueOf(base.toLong())
    while (counter > 0) {
        result *= bigBase
        counter--
    }

    return result
}

data class DigitSumAndCount(val sum : Int, val count : Int)

fun digitSumCount(num: BigInteger) : DigitSumAndCount /* Int */ {
    var sum = 0
    var count = 0
    var tempNum = num
    val bigTen = BigInteger.valueOf(10)
    val bigZero = BigInteger.valueOf(0)

    while (tempNum > bigZero) {
        sum += (tempNum % bigTen).toInt()
        count++
        tempNum /= bigTen
    }

    return DigitSumAndCount(sum, count)
//    return sum
}

// gets the maximum sum of digits
fun findMaxDigitalSum(maxBase: Int, maxExponent: Int) : Int {
    // smallestDigitCount will start with the number of digits in 99^99
    var (smallestBiggestSum, smallestDigitCount) = digitSumCount(power(maxBase, maxExponent))
    var base = maxBase
    var maxSum = 0
    var maxDigitCount = 0

    // the outer loops will test that 9 * smallestDigitCount is larger than the max sum of digits.
    // this is because we know (eventually) a number like 50^99 will not have enough space to be
    // larger than the current largest digit sum
    while (smallestDigitCount*9 > maxSum) {
        // we reset the exponent to be the maximum one.
        var exponent = maxExponent
        var currentNum = power(base, exponent)
        var (tempSmallestBiggestSum, tempSmallestDigitCount) = digitSumCount(currentNum)
        smallestDigitCount = tempSmallestDigitCount
        var downwardDigitCount = smallestDigitCount

        println("$base")

        // the inner loop checks downward until a number (like 99^70) doesn't have enough space
        // to be larger than the current largest digit sum
        val bigBase = BigInteger.valueOf(base.toLong())
        while (9*downwardDigitCount > maxSum) {
            val (tempSum, tempDigitCount) = digitSumCount(currentNum)
            downwardDigitCount = tempDigitCount
            if (tempSum > maxSum) {
                maxSum = tempSum

                println("$base ^ $exponent = $currentNum with count $maxSum")
            }

            currentNum /= bigBase
            exponent -= 1
        }

        base -= 1
    }

    return maxSum
}

fun main() {
    // 972
    println(findMaxDigitalSum(99, 99))
}