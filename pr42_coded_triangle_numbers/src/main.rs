// TODO:
// 1. Iterate through words
// 2. Maintain word sums -> whether they are triangle words
// 3. Hold a Vec of triangle numbers
// 4. Compute a larger triangle number and add to vec if word sum > largest

use std::collections::HashMap;
use std::fs;

// gets the sum of the values of characters of a word
fn get_word_sum(word: String) -> i32 {
    let mut sum = 0;

    for letter in word.to_lowercase().chars() {
        sum += (letter) as i32 - ('a') as i32 + 1;
    }

    return sum;
}

fn is_triangle_number(number: i32, triangle_nums: &mut Vec<i32>) -> bool {
    let num_triangle_nums = triangle_nums.len() as i32 - 1;

    let mut largest_triangle = 1;
    if num_triangle_nums >= 0 {
        largest_triangle = *triangle_nums.get(num_triangle_nums as usize).unwrap();
    }

    while number > largest_triangle {
        largest_triangle = ((triangle_nums.len() + 1) * (triangle_nums.len() + 2) / 2) as i32;
        triangle_nums.push(largest_triangle);
    }

    // iterate over triangle numbers; if a triangle num is greater than
    // the searched-for number, then we know that number is not a triangle number
    for triangle_num in triangle_nums {
        if *triangle_num == number {
            return true;
        }
        else if *triangle_num > number {
            return false;
        }
    }
    return false;
}

fn count_triangle_words() -> i32 {
    let contents = fs::read_to_string("./src/p042_words.txt")
        .expect("Something went wrong reading the file");

    let mut word_sum_map: HashMap<i32, bool> = HashMap::new();
    let mut triangle_nums: Vec<i32> = Vec::new();

    let mut count = 0;

    let split_contents = contents.split(",");
    for word in split_contents {
        let changed_word = word.replacen("\"", "", 2);
        let word_sum = get_word_sum(changed_word);

        if !word_sum_map.contains_key(&word_sum) {
            word_sum_map.insert(word_sum,
                                is_triangle_number(word_sum, &mut triangle_nums));
        }
        if *word_sum_map.get(&word_sum).unwrap() {
            count += 1;
        }
    }

    return count;
}

fn main() {
    println!("{}", count_triangle_words());
}
// 161, 146, 151 all wrong