use std::collections::HashSet;

fn is_prime(smaller_primes: &Vec<i32>, num: i32) -> bool {
    let root = (num as f64).sqrt();
    let mut current_prime_index = 0;
    let mut current_prime = smaller_primes.get(0).unwrap();

    while *current_prime as f64 <= root {
        if num % *current_prime == 0 {
            return false;
        }

        current_prime_index += 1;
        current_prime = smaller_primes.get(current_prime_index).unwrap();
    }

    return true;
}

fn fits_goldbach(primes_set: &HashSet<i32>, num: i32) -> bool {
    let mut num_to_double = 1;
    let mut doubled_squared_num = 2;

    while doubled_squared_num < num {
        if primes_set.contains(&(num - doubled_squared_num)) {
            return true;
        }

        num_to_double += 1;
        doubled_squared_num = 2 * (num_to_double * num_to_double);
    }

    return false;
}

fn find_first_comp_num_that_fails() -> i32 {
    let mut num: i32 = 2;
    let mut primes : Vec<i32> = Vec::new();
    let mut primes_set : HashSet<i32> = HashSet::new();
    primes.push(2);
    primes_set.insert(2);

    loop {
        if num % 2 != 0 {
            if is_prime(&primes, num) {
                primes.push(num);
                primes_set.insert(num);
            } else {
                if !fits_goldbach(&primes_set, num) {
                    return num;
                }
            }
        }

        num += 1;
    }
}

fn main() {
    // 5777
    println!("{}", find_first_comp_num_that_fails());
}
