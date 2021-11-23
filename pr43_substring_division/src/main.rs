use std::collections::HashSet;

fn get_pandigitals(allow_zero: bool, max_digit: i32, used_digits: HashSet<i32>,
                   pandigitals: &mut Vec<i64>, leading_num: i64) {
    if used_digits.len() == (max_digit+1) as usize {
        pandigitals.push(leading_num);
    }

    // recurse down and find other pandigitals
    let mut i = 0;
    while allow_zero && i < max_digit+1 {
        if !used_digits.contains(&i) {
            let mut new_leading_num = leading_num;
            new_leading_num *= 10;
            new_leading_num += i as i64;

            let mut next_used_digits = used_digits.clone();
            next_used_digits.insert(i);

            get_pandigitals(allow_zero, max_digit, next_used_digits,
                            pandigitals, new_leading_num);
        }
        i += 1;
    }
}

fn is_factorable(pandigital: i64, primes: [i32; 8]) -> bool {
    // start from the back and move forward
    let mut i = primes.len() as i32 - 1;
    let mut pandigital_copy = pandigital.clone();

    while i >= 0 {
        // takes the last 3 digits of the pandigital_copy
        // we will slowly destroy the copy to inpect the last digits.
        let last_three = (pandigital_copy % 1000) as i32;
        if last_three % primes[i as usize] != 0 {
            return false;
        }

        // do integer division to lose the lower digit
        pandigital_copy /= 10;
        i -= 1;
    }

    return true;
}

fn get_factorable_pandigitals() -> i64 {
    let mut sum: i64 = 0;

    let mut pandigitals : Vec<i64> = Vec::new();
    get_pandigitals(true, 9, HashSet::new(), &mut pandigitals, 0);

    let primes = [1, 2, 3, 5, 7, 11, 13, 17];
    for pandigital in pandigitals {
        if is_factorable(pandigital, primes) {
            sum += pandigital;
        }
    }

    return sum;
}

fn main() {
    println!("{}", get_factorable_pandigitals());
}
