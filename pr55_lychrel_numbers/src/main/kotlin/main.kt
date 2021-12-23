import java.math.BigInteger

fun getReversed(num: BigInteger) : BigInteger {
    var reversed : BigInteger = BigInteger.valueOf(0)
    var deconstructedNum = num

    while (deconstructedNum > BigInteger.valueOf(0)) {
        // adds the bottom of deconstructedNum to the top of reversed
        reversed += (deconstructedNum % BigInteger.valueOf(10))
        deconstructedNum /= BigInteger.valueOf(10)

        if (deconstructedNum > BigInteger.valueOf(0)) {
            reversed *= BigInteger.valueOf(10)
        }
    }

    return reversed
}

fun countPalindromesBelow10000() : Int {
    var count = 0
    for (i in 10..9999) {
        println("Testing $i")
        var formedPalindrome = false
        var iteration = 1
        var trackedNum = BigInteger.valueOf(i.toLong())
        var nextReversed = getReversed(trackedNum)
        println("$trackedNum, $nextReversed")

        while (!formedPalindrome && iteration < 50) {
            trackedNum += nextReversed
            nextReversed = getReversed(trackedNum)

            // if we found a palindrome, break
            if (trackedNum == nextReversed) {
                formedPalindrome = true
            }

            iteration++
        }

        if (!formedPalindrome) {
            count++
        }
    }
    return count
}

fun main() {
    // 249
    println(countPalindromesBelow10000())
}