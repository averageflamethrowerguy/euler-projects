import kotlin.math.ceil

/*
There are a few tricks to speed things up:
1. We can keep track of n!
2. We can double-count 4 choose 2 and 4 choose 3 (they are mirror images)
4:0=4:4
4:1=4:3
4:2-> unique

5 choose 0 -> 1
5 choose 1 = 5 choose 4
5 choose 2 = 5 choose 3
5 choose 5 -> 1
 */

fun countCombosGreaterThan(maxNumber: Int, threshold: Int) : Int {
    var count = 0

    var nFactorial = 1.0
    for (i in 1..maxNumber) {
        nFactorial *= i
        // starts calculating bottom up
        var denom1 = 1.0
        var denom2 = nFactorial

        val halfway = i / 2
//        for (j in 0..halfway) {
        for (j in 0..i) {
            if (j != 0) {
                denom1 *= j
                denom2 /= i-j+1
            }

            val result = nFactorial/(denom1 * denom2)

            if (result > threshold) {
                count += 1

                // not exactly sure why the below doesn't work; also only a ~2x speed improvement.

//                count += if (j != halfway) {
//                    println("$i choose $j = $result")
//                    println("Double count")
//                    // can double-count
//                    2
//                } else {
//                    // can't double-count
//                    println("$i choose $j = $result")
//                    println("Single count")
//                    1
//                }
            }
        }
    }

    return count
}

fun main() {
    //4036 -> actual 4075 (what's wrong...?)
    println(countCombosGreaterThan(100, 1000000))
}