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

use std::cmp::min;
use std::collections::HashSet;

fn find_min_abs_diff_pent_nums() -> i32 {
    let mut pent_num_set : HashSet<i32> = HashSet::new();
    let mut pent_num_vec : Vec<i32> = Vec::new();
    let mut min_diff : i32 = i32::MAX;
    let mut last_diff = 0;
    let mut next_index = 0;
    let mut search_downward_index:i32 = 0;

    // add 1 (the first pent num) into the vec and set
    pent_num_set.insert(1);
    pent_num_vec.push(1);

    // the last difference bounds the system
    while last_diff < min_diff {
        let current_num = next_index+1;
        let current_pent = (3*current_num*current_num - current_num) / 2;
        let next_num = current_num+1;
        let next_pent = (3*next_num*next_num - next_num) / 2;
        pent_num_vec.push(next_pent);
        pent_num_set.insert(next_pent);

        let pent_diff = current_pent-next_pent;

        // while the value search_downward_index is smaller than the difference to the
        // next pent number, increase the index
        while pent_num_vec.get(search_downward_index).unwrap() < &pent_diff {
            search_downward_index += 1;
        }

        let mut temp_search_downward_index = search_downward_index;
        while temp_search_downward_index < next_index {
            // if the diff is a pent number
            if pent_num_set.contains(current_pent - pent_num_vec.get(temp_search_downward_index)) {
                // evaluate if the sum is a pent number
                let sum = current_pent + pent_num_vec.get(temp_search_downward_index);
                let mut temp_search_upward_index = next_index;
                let mut last_pent = current_pent;

                while last_pent <= sum {
                    // if we hit the sum, store the difference
                    if last_pent == sum && pent_diff < min_diff {
                        min_diff = pent_diff;
                    }
                    let upward_num = temp_search_upward_index + 1;
                    let upward_pent = (3*upward_num - 1) * upward_num;
                    if !pent_num_set.contains(&upward_pent) {
                        pent_num_set.insert(upward_pent);
                        pent_num_vec.push(upward_pent);
                    }

                    temp_search_upward_index += 1;
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
