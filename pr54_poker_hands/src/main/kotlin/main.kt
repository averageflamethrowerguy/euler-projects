import java.io.File

// converts the type of the card to its numerical value
fun convertToInt(card : String) : Int {
    return when (card) {
        "A" -> 14
        "K" -> 13
        "Q" -> 12
        "J" -> 11
        "T" -> 10
        else -> card.toInt()
    }
}

// note -- we will denote values:
// 2 -> 2 on up to 10; J -> 11, Q -> 12, K -> 13, A -> 14
class Hand(cards: MutableList<String>) {
    val sortedCards : List<String>
    var highestCard : Int = 0
    var highestPair : Int = 0
    var secondPair : Int = 0
    var threeOfKind : Int = 0
    var straight : Boolean = false
    // all same suit
    var flush : Boolean = false
    var fullHouse = false
    var fourOfKind : Int = 0
    var straightFlush = false
    var royalFlush = false

    private class CardComparator: Comparator<String>{
        override fun compare(o1: String, o2: String): Int {
            val card1 = convertToInt(o1.substring(0,1))
            val card2 = convertToInt(o2.substring(0,1))

            return if (card1 > card2) {
                1
            } else if (card2 > card1) {
                -1
            } else 0
        }
    }

    init {
        sortedCards = cards.sortedWith(CardComparator())

        var last : String? = null
        var isValidFlush = true
        var isValidStraight = true
        for (i in sortedCards.indices) {
            val currentCard = sortedCards[i]

            // Check conditions for flushes, straights, etc
            if (last != null) {
                // Invalidate flushes if there is a suit change
                if (isValidFlush && last[1] != currentCard[1]) {
                    isValidFlush = false
                }

                val conversion = convertToInt(currentCard.substring(0,1))
                // check for doubles and triples and quads
                if (last[0] == currentCard[0]) {
                    // quads first
                    if (threeOfKind == conversion) {
                        fourOfKind = conversion
                        threeOfKind = 0
                    }
                    // triples, and takes care of any existing pairs
                    else if (highestPair == conversion) {
                        threeOfKind = conversion
                        highestCard = secondPair
                        secondPair = 0
                    }
                    // takes care of second and highest pairs
                    else {
                        secondPair = highestPair
                        highestPair = conversion
                    }
                }

                // check for straights to see if the current card is one larger than previous
                if (conversion != convertToInt(last.substring(0,1))+1) {
                    isValidStraight = false
                }
            }

            // updates the highest card
            if (highestCard < convertToInt(currentCard.substring(0,1))) {
                highestCard = convertToInt(currentCard.substring(0,1))
            }

            last = currentCard
        }

        // update variables for straight, flush, straightFlush, royalFlush, fullHouse
        if (isValidFlush) {
            flush = true
        }
        if (isValidStraight) {
            straight = true
        }
        if (straight && flush) {
            straightFlush = true
        }
        if (straightFlush && highestCard == 14) {
            royalFlush = true
        }
        if (threeOfKind != 0 && highestPair != 0) {
            fullHouse  =true
        }
    }

    // returns true if this hand is bigger
    fun compareHands(otherHand: Hand) : Boolean {
        if (royalFlush && !otherHand.royalFlush) {
            return true
        } else if (otherHand.royalFlush && !royalFlush) {
            return false
        }

        if (straightFlush && !otherHand.straightFlush) {
            return true
        } else if (otherHand.straightFlush && straightFlush) {
            return false
        }

        if (fourOfKind > otherHand.fourOfKind) {
            return true
        } else if (otherHand.fourOfKind > fourOfKind) {
            return false
        }

        if (fullHouse && !otherHand.fullHouse) {
            return true
        } else if (otherHand.fullHouse && !fullHouse) {
            return false
        }

        if (flush && !otherHand.flush) {
            return true
        } else if (otherHand.flush && !flush) {
            return false
        }

        if (straight && !otherHand.straight) {
            return true
        } else if (otherHand.straight && !straight) {
            return false
        }

        if (threeOfKind > otherHand.threeOfKind) {
            return true
        } else if (otherHand.threeOfKind > threeOfKind) {
            return false
        }

        // we break ties using the larger pairs... then go to the second pairs if needed
        if (secondPair != 0 && otherHand.secondPair == 0) {
            return true
        } else if (otherHand.secondPair != 0 && secondPair == 0) {
            return false
        }

        if (highestPair > otherHand.highestPair) {
            return true
        } else if (otherHand.highestPair > highestPair) {
            return false
        } else if (secondPair > otherHand.secondPair) {
            return true
        } else if (otherHand.secondPair > secondPair) {
            return false
        }

        return highestCard > otherHand.highestCard
    }

    override fun toString() : String {
        return "Royal Flush: $royalFlush, Straight Flush: $straightFlush, Four of Kind: $fourOfKind, Full House: $fullHouse, \n" +
                "Flush: $flush, Straight: $straight, Three of Kind: $threeOfKind,\n" +
                "Second Pair: $secondPair, Highest Pair: $highestPair, Highest Card: $highestCard"
    }
}

fun doesPlayer1Win(cards1: List<String>, cards2: List<String>) : Boolean {
    val hand1 = Hand(cards1 as MutableList<String>)
    val hand2 = Hand(cards2 as MutableList<String>)

    val comparison = hand1.compareHands(hand2)
    println("------------------------------------------")
    println(hand1.sortedCards.toString() + ", " + hand2.sortedCards.toString() + ": " + comparison)
    println(hand1.toString() + "\n\n" + hand2.toString())
    return comparison
}

fun countPlayer1Victories() : Int {
    var count = 0
    File("./src/main/resources/hands.txt").forEachLine {
        val splitLine = it.split(" ")
        if (doesPlayer1Win(splitLine.subList(0, 5), splitLine.subList(5, splitLine.size))) {
            count++
        }
    }
    return count
}

fun main() {
    // runs 377; should be 376 (what am I missing...?)
    println(countPlayer1Victories())
}