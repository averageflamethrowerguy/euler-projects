fn is_prime(smaller_primes: &Vec<i32>, num: i32) -> bool {
    let root = (num as f64).sqrt();
    let mut current_prime_index = 0;
    let mut current_prime = smaller_primes.get(0).unwrap();

    if num == 1 {
        return false;
    }

    while *current_prime as f64 <= root && current_prime_index < smaller_primes.len() {
        if num % *current_prime == 0 {
            return false;
        }

        current_prime_index += 1;
        if current_prime_index < smaller_primes.len() {
            current_prime = smaller_primes.get(current_prime_index).unwrap();
        }
    }

    return true;
}

fn has_four_prime_factors(primes: &Vec<i32>, num: i32) -> bool {
    let root = (num as f64).sqrt();

    let mut prime_index = 0;
    let mut current_prime = *primes.get(0).unwrap();
    let mut number_prime_factors = 0;

    let mut prime_factors : Vec<i32> = Vec::new();

    // goes over smaller primes and checks if they are factors
    // we can constrain these to the square root and check larger factors
    // on a case-by-case basis
    while current_prime as f64 <= root && prime_index < primes.len() {
        if num % current_prime == 0 {
            number_prime_factors += 1;
            prime_factors.push(current_prime);
        }

        prime_index += 1;
        if prime_index < primes.len() {
            current_prime = *primes.get(prime_index).unwrap();
        }
    }

    // search for the (probably 1) additional factor above the root
    let mut temp_num = num;
    loop {
        let mut did_pass_cycle = true;
        for prime in &prime_factors {
            if temp_num % prime == 0 {
                temp_num /= prime;
                did_pass_cycle = false;
            }
        }
        if did_pass_cycle {
            break;
        }
    }
    // add it if it is indeed prime
    if is_prime(primes, temp_num) {
        // println!("{}", temp_num);
        number_prime_factors += 1;
    }

    return if number_prime_factors >= 4 {
        true
    } else {
        false
    }
}

fn find_four_consecutive_quad_primes() -> i32 {
    let mut min_seq_number = 3;
    let mut max_seq_number = 6;
    let mut primes_array : Vec<i32> = Vec::new();
    primes_array.push(2);

    let mut next_highest_safe = 0;

    loop {
        // first evaluate all numbers to see if they are primes
        let mut i = min_seq_number;
        while i <= max_seq_number {
            if i > next_highest_safe && is_prime(&primes_array, i) {
                // println!("Adding prime: {}", i);
                primes_array.push(i);
            }
            i += 1;
        }

        // we use a hopping algorithm where we will always evaluate
        // the number 4-away first (this is like text processing -- may
        // speed up computation by a factor of 4.)
        let mut will_pass = true;
        let mut i = max_seq_number;
        let mut next_next_highest_safe= next_highest_safe;
        while i >= min_seq_number && i > next_highest_safe {
            if has_four_prime_factors(&primes_array, i) {
                // we keep marking the highest safe, we can use this to avoid
                // recomputing factors on any numbers
                if i > next_next_highest_safe {
                    next_next_highest_safe = i;
                }
            } else {
                will_pass = false;
                break;
            }
            i -= 1;
        }

        if will_pass {
            return min_seq_number;
        } else {
            // the last number that we declared safe
            // (Or the number above max_seq_number, if we didn't find a safe one)
            min_seq_number = i + 1;
            max_seq_number = min_seq_number + 3;
            next_highest_safe = next_next_highest_safe;
        }
    }
}

fn main() {
    // 134043
    println!("{}", find_four_consecutive_quad_primes());
}
