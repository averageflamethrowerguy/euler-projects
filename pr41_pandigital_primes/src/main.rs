/*
This problem is interesting because we now need to find and create arbitrarily sized pandigital numbers.
(Pandigital numbers can have at most 9 digits -- we aren't including 0 as per the example).

We start with the top (9 digits) and reduce digits until we find a prime (we search the entire level)
 */
use std::collections::HashSet;

fn is_prime(number: i32) -> bool {
    let root = (number as f64).sqrt();

    let mut i = 2;
    while i as f64 <= root {
        if number % i == 0 {
            return false;
        }
        i += 1;
    }
    return true;
}

fn get_pandigitals(number_digits: i32, used_digits: HashSet<i32>,
                   pandigitals: &mut Vec<i32>, leading_num: i32) {
    if used_digits.len() == number_digits as usize {
        pandigitals.push(leading_num);
    }

    // recurse down and find other pandigitals
    let mut i = 0;
    while i < number_digits {
        if !used_digits.contains(&(i + 1)) {
            let mut new_leading_num = leading_num;
            new_leading_num *= 10;
            new_leading_num += i+1;

            // if it's the last digit, we can eliminate some computation by
            // ignoring clearly non-prime cases
            if used_digits.len() < (number_digits - 1) as usize ||
                i+1 % 2 != 0 {
                let mut next_used_digits = used_digits.clone();
                next_used_digits.insert(i+1);

                get_pandigitals(number_digits, next_used_digits,
                                pandigitals, new_leading_num);
            }
        }
        i += 1;
    }
}

// number_digits indicates the top level digit (i.e 8 would indicate 1-8).
fn get_pandigital_primes(number_digits: i32,
                         pandigital_primes: &mut Vec<i32>,
) {
    let mut pandigitals : Vec<i32> = Vec::new();
    if number_digits > 0 {
        let mut i = 0;
        while i < number_digits {
            let mut used_digits: HashSet<i32> = HashSet::new();
            used_digits.insert(i + 1);
            let leading_num = i + 1;
            get_pandigitals(number_digits, used_digits,
                            &mut pandigitals, leading_num);
            i += 1;
        }

    }

    // if we're at the top level, we'll make sure the collected pandigitals are prime
    for pandigital in pandigitals {
        if is_prime(pandigital) {
            pandigital_primes.push(pandigital);
        }
    }
}

fn find_largest_pandigital_prime() -> i32 {
    let mut number_pandigital_digits = 9;

    // a storage object for the primes we discover in a layer.
    let mut storage_vec: Vec<i32> = Vec::new();

    while number_pandigital_digits > 0 {
        // gets the list of pandigital primes in the layer
        get_pandigital_primes(
            number_pandigital_digits,
            &mut storage_vec
        );

        // if the list is not empty, break the while loop
        if !storage_vec.is_empty() {
            break;
        }

        // otherwise, axe a digit and continue
        number_pandigital_digits -= 1;
    }

    let mut largest_prime_pandigital = 0;

    // loop through the primes we discovered
    for i in storage_vec {
        if i > largest_prime_pandigital {
            largest_prime_pandigital = i;
        }
    }

    // return the largest prime
    return largest_prime_pandigital;
}

fn main() {
    println!("{}", find_largest_pandigital_prime().to_string());
}
