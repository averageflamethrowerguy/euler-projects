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

// check if a number is pentagonal
fn is_pentagonal(number : i64) -> bool {
    let pen_test = ((24.0 * (number as f64) + 1.0 ).sqrt() + 1.0) / 6.0;
    // do type coercion to check if modulo is equal.
    if pen_test == (pen_test as i64) as f64 {
        return true;
    }
    return false;
}

// gets a pent number from a certain number
fn get_pentagonal(input_num : i32) -> i64 {
    return (3*input_num - 1) as i64 *input_num as i64 / 2;
}

fn find_min_abs_diff_pent_nums() -> i64 {
    let mut min_diff : i64 = i64::MAX;
    let mut last_diff:i64 = 0;
    let mut last_bounding_diff:i64 = 0;
    let mut current_num = 1;
    let mut search_downward_num:i32 = 1;

    // the last difference bounds the system
    while last_bounding_diff < min_diff {
        let current_pent = get_pentagonal(current_num);
        let next_num = current_num+1;
        let next_pent = get_pentagonal(next_num);

        last_diff = next_pent-current_pent;
        /*
        This next section is very interesting:
        next_pent <= current_pent + other_pent
        so
        other_pent >= next_pent - current_pent

        We want to find current_pent - other_pent
        ~= 2 * current_pent - next_pent
        which is a very good bound!

        This allows us to super-efficiently bound the problem
        because this number is much smaller than last_diff.
         */
        last_bounding_diff = (2*current_pent) - next_pent;

        // while the value search_downward_index is smaller than the difference to the
        // next pent number, increase the index
        while get_pentagonal(search_downward_num) < last_diff {
            search_downward_num += 1;
        }


        // searches a subset of the pent numbers underneath this one in order to
        // find the next pent number
        let mut temp_search_downward_num = search_downward_num;
        while temp_search_downward_num < next_num {
            let other_pent = get_pentagonal(temp_search_downward_num);

            // if the diff is a pent number
            if is_pentagonal(current_pent-other_pent) {
                // evaluate if the sum is a pent number
                let sum = current_pent + other_pent;
                // println!("Current num: {}, Current pent: {}", current_num, current_pent);
                // println!("Last diff: {}, min diff: {}", last_diff, min_diff);

                if is_pentagonal(sum) {
                    println!("Sum is pentagonal!");
                    if (current_pent - other_pent) < min_diff {
                        min_diff = current_pent - other_pent;
                        println!("Last diff: {}, min diff: {}", last_diff, min_diff);
                        println!("pent1: {}, pent2: {}", temp_search_downward_num, next_num);
                    }
                }

                println!("{} {}", last_bounding_diff, min_diff);
            }

            temp_search_downward_num += 1;
        }

        current_num += 1;
    }

    return min_diff;
}

fn main() {
    println!("{}", find_min_abs_diff_pent_nums());
    // the answer is 5482660
}
