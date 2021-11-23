/*
(3n^2 - n) / 2 = (3k^2 - k ) / 2 + (3j^2 - j ) / 2
3n^2 - n = (3(k^2 + j^2) - (k + j))

We know that k + j - n = 3q
for some integer q.

Brute force solution:
Compute downward from indices
(index 5 will check 0,1,2,3,4)
We can lower-bound the search by computing the P num above the one we're currently checking
Find the difference between that P-num and this one
that is the first
compute difference using set of P numbers

 */

use std::collections::HashSet;

// check if a number is pentagonal
fn is_pentagonal(number : i32) -> bool {
    let pen_test = ((number as f64 + 1.0 ).sqrt() + 1.0) / 6.0;
    // do type coercion to check if modulo is equal.
    if pen_test == (pen_test as i64) as f64 {
        return true;
    }
    return false;
}

// gets a pent number from a certain number
fn get_pentagonal(pent_index : i32) -> i32 {
    return (3*pent_index - 1)*pent_index / 2
}

fn find_min_abs_diff_pent_nums() -> i32 {
    let mut min_diff : i32 = i32::MAX;
    let mut last_diff = 0;
    let mut next_index = 0;
    let mut search_downward_index:i32 = 0;

    // the last difference bounds the system
    while last_diff < min_diff {
        let current_num = next_index+1;
        let current_pent = get_pentagonal(current_num);
        let next_num = current_num+1;
        let next_pent = get_pentagonal(next_num);

        last_diff = next_pent-current_pent;

        // while the value search_downward_index is smaller than the difference to the
        // next pent number, increase the index
        while get_pentagonal(search_downward_index+1) < last_diff {
            search_downward_index += 1;
        }


        // searches a subset of the pent numbers underneath this one in order to
        // find the next pent number
        let mut temp_search_downward_index = search_downward_index;
        while temp_search_downward_index < next_index {
            let other_pent = get_pentagonal(temp_search_downward_index+1);
            // if the diff is a pent number
            if is_pentagonal(current_pent-other_pent) {
                // evaluate if the sum is a pent number
                let sum = current_pent + other_pent;
                println!("Current num: {}, Current pent: {}", current_num, current_pent);
                println!("Last diff: {}, min diff: {}", last_diff, min_diff);

                if is_pentagonal(sum) {
                    if (sum - current_pent) < min_diff {
                        min_diff = sum - current_pent;
                        println!("Last diff: {}, min diff: {}", last_diff, min_diff);
                        println!("pent1: {}, pent2: {}", temp_search_downward_index+1, next_num);
                    }
                }
            }

            temp_search_downward_index += 1;
        }

        next_index += 1;
    }

    return min_diff;
}

fn main() {
    println!("{}", find_min_abs_diff_pent_nums());
}
