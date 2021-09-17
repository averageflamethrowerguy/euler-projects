/*
This problem is interesting because we now need to find and create arbitrarily sized pandigital numbers.
(Pandigital numbers can have at most 9 digits -- we aren't including 0 as per the example).

We start with the top (9 digits) and reduce digits until we find a prime (we search the entire level)
 */

// number_digits indicates the top level digit (i.e 8 would indicate 1-8).
fn get_pandigital_primes(number_digits: i32, digits_used: Vec<i32>, pandigital_primes: Vec<i32> ) {

}

fn find_largest_pandigital_prime() -> int {
    let mut number_pandigital_digits = 9;

    // a storage object for the primes we discover in a layer.
    let mut storage_vec: Vec<i32>;

    while (number_pandigital_digits > 0) {
        // gets the list of pandigital primes in the layer
        let temp_digits_used: Vec<i32> = Vec;
        get_pandigital_primes(
            number_pandigital_digits,
            temp_digits_used
        );

        // if the list is not empty, break the while loop
        let mut willBreak: bool = false;
        for i in storage_vec.iter() {
            willBreak = true;
            break;
        }
        if (willBreak) {
            break;
        }

        // otherwise, axe a digit and continue
        number_pandigital_digits -= 1;
    }

    let mut largest_prime_pandigital = 0;

    // loop through the primes we discovered
    for i in storage_vec.iter() {
        if i > &largest_prime_pandigital {
            &largest_prime_pandigital = i;
        }
    }

    // return the largest prime
    return largest_prime_pandigital;
}

fn main() {
    println!("Hello, world!");
}
