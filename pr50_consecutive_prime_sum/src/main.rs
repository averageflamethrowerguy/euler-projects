/*
This one is very easy if you know the trick:
2 is so low-cost that you almost HAVE to start from 2
then we can build primes upward;

if the built number is not a prime, then we can try axing lower
and upper primes until we find a prime
 */

use std::collections::HashSet;

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

// returns a tuple: (length_of_sum, sum)
// calls itself recursively to axe upper and lower primes until it reaches a prime sum
// min index inclusive, max exclusive
fn optimize_to_find_prime_sum(
    prime_array: &Vec<i32>,
    prime_set: &HashSet<i32>,
    sum_length: i32,
    min_index: i32,
    max_index: i32,
    max_index_ceiling: i32,
    original_max_index: i32,
    current_sum: i32,
    ceiling: i32,
    largest_discovered_chain: &mut [i32]
) -> (i32, i32) {
    // exits chains smaller than largest discovered chain (they are useless to us)
    if sum_length < largest_discovered_chain[0] {
        return (0, 0);
    }

    // if its a prime, returns immediately
    if prime_set.contains(&current_sum) {
        return (sum_length, current_sum);
    }

    // otherwise compares primes with tops and bottoms removed
    let small_removed = optimize_to_find_prime_sum(
        prime_array,
        prime_set,
        sum_length-1,
        min_index+1,
        max_index,
        max_index_ceiling,
        original_max_index,
        current_sum-prime_array.get(min_index as usize).unwrap(),
        ceiling,
        largest_discovered_chain
    );

    let mut large_removed = (0, 0);
    // if we add a larger prime, we won't bother removing it again
    if max_index_ceiling <= original_max_index {
        large_removed = optimize_to_find_prime_sum(
            prime_array,
            prime_set,
            sum_length-1,
            min_index,
            max_index-1,
            max_index_ceiling,
            original_max_index,
            current_sum-prime_array.get((max_index-1) as usize).unwrap(),
            ceiling,
            largest_discovered_chain
        );
    }
    let mut large_added = (0, 0);
    // if we've never removed a large prime, we can add it if we have space
    if max_index == max_index_ceiling &&
        current_sum+prime_array.get(max_index as usize).unwrap() < ceiling
    {
        large_added = optimize_to_find_prime_sum(
            prime_array,
            prime_set,
            sum_length+1,
            min_index,
            max_index+1,
            max_index_ceiling+1,
            original_max_index,
            current_sum+prime_array.get(max_index as usize).unwrap(),
            ceiling,
            largest_discovered_chain
        );
    }

    let return_val;
    if small_removed.0 > large_removed.0 && small_removed.0 > large_added.0 {
        return_val = small_removed;
    } else if large_removed.0 > small_removed.0 && large_removed.0 > large_added.0 {
        return_val = large_removed;
    } else {
        return_val = large_added;
    }

    if return_val.0 > largest_discovered_chain[0] {
        largest_discovered_chain[0] = return_val.0;
    }
    if current_sum == 997651 {
        println!("{}, {}", return_val.0, return_val.1);
    }
    return return_val;
}

fn find_largest_prime_sum_of_primes(upper_limit: i32) -> (i32, i32) {
    let mut prime_array : Vec<i32> = Vec::new();
    let mut prime_set : HashSet<i32> = HashSet::new();
    prime_array.push(2);
    prime_set.insert(2);
    let mut num = 3;
    let mut sum = 2;
    let mut prime_count = 1;

    while num < upper_limit {
        if is_prime(&prime_array, num) {
            if sum+num < upper_limit {
                sum += num;
                prime_count += 1;
            }

            prime_array.push(num);
            prime_set.insert(num);
        }

        num += 1;
    }

    let mut largest_discovered_chain = [0];

    // optimize up and down to try to seek a better solution
    return optimize_to_find_prime_sum(&prime_array,
                                      &prime_set,
                                      prime_count,
                                      0,
                                      prime_count+1,
                                      prime_count+1,
                                      prime_count+1,
                                      sum,
                                      upper_limit,
                                        &mut largest_discovered_chain
    );
}

fn main() {
    //997651
    let answer = find_largest_prime_sum_of_primes(1000000);
    println!("{}, {}", answer.0, answer.1);
}
