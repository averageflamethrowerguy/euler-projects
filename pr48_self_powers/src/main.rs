/*
This one is pretty funky.
We know a few things:
1: Anything ending in a zero is IRRELEVANT to our calculations. 10^10 is 11 digits. Any other multiples
of 10 will be larger.
2. We generally don't have to deal with higher digits in our numbers
 */

fn take_modulo_power(number: i32, power: i32, modulo: i64) -> i64 {
    let mut temp_pow = 1;
    let mut result = number as i64;
    let expanded_number = number as i64;
    while temp_pow < power {
        temp_pow += 1;
        result *= expanded_number;
        result = result % modulo;
    }
    return result;
}

fn get_self_power_sum(number_digits: i32, largest_number: i32) -> i64 {
    let mut current_num = 1;
    let mut sum : i64 = 0;
    let modulo = take_modulo_power(10, number_digits, i64::MAX);

    while current_num <= largest_number {
        sum += take_modulo_power(current_num, current_num, modulo);
        current_num += 1;
    }

    return sum % modulo;
}

fn main() {
    //9110846700
    println!("{}", get_self_power_sum(10, 1000));
}
