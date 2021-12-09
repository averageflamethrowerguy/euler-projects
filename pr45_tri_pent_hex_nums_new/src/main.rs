fn generate_hex_num(input_num: i32) -> i64 {
    return input_num as i64 * (2 * (input_num as i64) - 1);
}

// check if a number is pentagonal
fn is_pentagonal(number : i64) -> bool {
    let pen_test = ((24.0 * (number as f64) + 1.0 ).sqrt() + 1.0) / 6.0;
    // do type coercion to check if modulo is equal.
    if pen_test == (pen_test as i64) as f64 {
        return true;
    }
    return false;
}

// check if a number is triangular
fn is_triangular(number : i64) -> bool {
    let pen_test = ((8.0 * (number as f64) + 1.0 ).sqrt() - 1.0) / 2.0;
    // do type coercion to check if modulo is equal.
    if pen_test == (pen_test as i64) as f64 {
        return true;
    }
    return false;
}

fn find_next_multi_polygon_num() -> i64 {
    let mut next_hex_input = 144;

    // we will return out of the loop rather than imposing a termination condition
    loop {
        let next_hex = generate_hex_num(next_hex_input);

        if is_pentagonal(next_hex) {
            if is_triangular(next_hex) {
                return next_hex;
            }
        }

        next_hex_input += 1;
    }

    // this is needed for correct compilation
    return 0;
}

fn main() {
    // 1533776805
    println!("{}", find_next_multi_polygon_num());
}
