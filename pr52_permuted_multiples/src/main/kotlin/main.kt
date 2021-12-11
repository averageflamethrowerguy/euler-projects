fun increment(key: Int, map: HashMap<Int, Int>) {
    if (!map.contains(key)) {
        map[key] = 1
    } else {
        map[key] = 1+map[key]!!
    }
}

fun getDigitMap(num: Int) : HashMap<Int, Int> {
    val map = HashMap<Int, Int>();
    var tempNum = num;
    while (tempNum > 0) {
        increment(tempNum % 10, map)
        tempNum /= 10
    }
    return map
}

fun mapsEqual(map1: HashMap<Int, Int>, map2: HashMap<Int, Int>) : Boolean {
    for (entry in map1) {
        if (!map2.contains(entry.key) || map2[entry.key] != entry.value) {
            return false
        }
    }
    return true
}

fun findSmallestPermuteMultiple(threshold: Int) : Int {
    var num = 1
    while (true) {
        val digits = getDigitMap(num)
        var didFail = false

        // check all the multiples
        for (i in 2..threshold) {
            val otherDigits = getDigitMap(num*i)
            if (!mapsEqual(digits, otherDigits)) {
                didFail = true
                break
            }
        }

        if (!didFail) {
            return num
        }

        num++
    }
}

fun main() {
    //142857
    println(findSmallestPermuteMultiple(6))
}