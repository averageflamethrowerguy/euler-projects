use std::collections::{HashMap, HashSet};

fn increment(digit_map: &mut HashMap<i32, i32>, num: i32) {
    if !digit_map.contains_key(&num) {
        digit_map.insert(num, 1);
    }
    else {
        digit_map.insert(num, *digit_map.get(&num).unwrap()+1);
    }
}

fn decrement(digit_map: &mut HashMap<i32, i32>, num: i32) {
    let count = *digit_map.get(&num).unwrap();
    if count > 1 {
        digit_map.insert(num, count-1);
    }
    else {
        digit_map.remove(&num);
    }
}

fn get_digits_in(num: i32) -> HashMap<i32, i32> {
    let mut temp_num = num;
    let mut digit_map : HashMap<i32, i32> = HashMap::new();
    while temp_num > 0 {
        increment(&mut digit_map, temp_num % 10);
        temp_num /= 10;
    }
    return digit_map;
}

fn is_permutation_of(num1_num_map: &HashMap<i32, i32>, num2: i32) -> bool {
    let mut temp_num2 = num2;
    let mut temp_num1_num_map = num1_num_map.clone();

    while temp_num2 > 0 {
        if !temp_num1_num_map.contains_key(&(temp_num2 % 10)) {
            return false;
        }
        else {
            decrement(&mut temp_num1_num_map, temp_num2 % 10);
        }
        temp_num2 /= 10;
    }
    return true;
}

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

fn generate_primes_below(upper_bound: i32) -> HashSet<i32> {
    let mut num = 3;
    let mut primes_set : HashSet<i32> = HashSet::new();
    let mut primes_array : Vec<i32> = Vec::new();
    primes_set.insert(2);
    primes_array.push(2);

    while num < upper_bound {
        if is_prime(&primes_array, num) {
            primes_array.push(num);
            primes_set.insert(num);
        }

        num += 1;
    }
    return primes_set;
}

fn find_prime_permutations(lower_bound: i32, upper_bound: i32) -> i64 {
    let mut lowest_num = lower_bound;
    let prime_set : HashSet<i32> = generate_primes_below(upper_bound);

    while lowest_num < upper_bound - 3 {
        let mut increment_num = 1;
        let digits = get_digits_in(lowest_num);

        // we know the largest number must be less than 10000, so that constrains the max
        // upward search location
        while increment_num <= (upper_bound - lowest_num) / 2 {
            // if the three numbers are all prime
            if prime_set.contains(&lowest_num) &&
                prime_set.contains(&(lowest_num + increment_num)) &&
                prime_set.contains(&(lowest_num + 2 * increment_num)) {

                // if the other 2 numbers are also permutations
                if is_permutation_of(&digits, lowest_num+increment_num) &&
                    is_permutation_of(&digits, lowest_num+2*increment_num) {
                    println!("{}, {}, {}", lowest_num, lowest_num+increment_num, lowest_num+2*increment_num);
                }
            }

            increment_num += 1;
        }

        lowest_num += 1;
    }
    // we fail with zero if we complete the loop
    return 0;
}

fn main() {
    // 2969, 6299, 9629
    println!("{}", find_prime_permutations(1000, 10000));
}
