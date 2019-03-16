extern crate csv;
extern crate serde_derive;

//use std::sync::{Arc, Mutex}; //TODO Concurrency
//use std::thread;

mod structs;

use std::collections::HashMap;
use std::time::Instant;
use structs::*;

fn make_partitions<T: Data<T> + Clone + Copy>(data: &Vec<T>, folds: usize) -> Vec<Vec<T>> {
    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<T>> = Vec::new();

    for _ in 0..folds {
        partitions.push(Vec::new());
    }

    for example in data {
        let counter = categories_count.entry(example.get_class()).or_insert(0);
        partitions[*counter].push(example.clone());

        *counter = (*counter + 1) % folds;
    }

    return partitions;
}

pub fn classifier_1nn_weights<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    exam: &Vec<T>,
    weights: &Vec<f32>,
    discard_low_weights: bool,
) -> i32 {
    let mut correct = 0;
    for result in exam.iter() {
        let mut nearest_example: T = T::new();
        let mut min_distance: f32 = std::f32::MAX;

        for known in knowledge.iter() {
            //NOTE Distance with weights
            let mut distance = 0.0;
            for index in 0..weights.len() {
                // NOTE weights below 0.2 aren't considered.
                if !discard_low_weights || weights[index] >= 0.2 {
                    distance += weights[index]
                        * (result.get_attr(index) - known.get_attr(index))
                        * (result.get_attr(index) - known.get_attr(index))
                }
            }

            distance = distance.sqrt();

            if distance < min_distance {
                min_distance = distance;
                nearest_example = known.clone();
            }
        }

        if nearest_example.get_class() == result.get_class() {
            correct += 1;
        }
    }
    return correct;
}

pub fn classifier_1nn<T: Data<T> + Clone + Copy>(knowledge: &Vec<T>, exam: &Vec<T>) -> () {
    let mut _correct: i32 = 0;
    for item in exam.iter() {
        let mut nearest_example: T = T::new();
        let mut min_distance: f32 = std::f32::MAX;

        for known in knowledge.iter() {
            let distance = item.euclidean_distance(known);
            if distance < min_distance {
                min_distance = distance;
                nearest_example = known.clone();
            }
        }

        if nearest_example.get_class() == item.get_class() {
            _correct += 1;
        }
    }

    println!(
        "\t\t1-NN Results: {}/{} = {}",
        _correct,
        exam.len(),
        _correct as f32 / exam.len() as f32
    );
}

pub fn relief<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    exam: &Vec<T>,
    n_attrs: usize,
    discard_low_weights: bool,
) -> Result<(), Box<std::error::Error>> {
    // NOTE For each partition
    let mut weights: Vec<f32> = vec![0.0; n_attrs];

    // NOTE Greedy
    for known in knowledge.iter() {
        let mut enemy_distance = std::f32::MAX;
        let mut friend_distance = std::f32::MAX;
        let mut ally_index = 0;
        let mut enemy_index = 0;

        for (index, candidate) in knowledge.iter().enumerate() {
            //NOTE Skip if cantidate == known
            if candidate.get_id() != known.get_id() {
                // NOTE Pre-calculate distance
                let dist = known.euclidean_distance(candidate);
                // NOTE Ally
                if known.get_class() == candidate.get_class() {
                    if dist < friend_distance {
                        ally_index = index;
                        friend_distance = dist;
                    }
                }
                // NOTE Enemy
                else {
                    if dist < enemy_distance {
                        enemy_index = index;
                        enemy_distance = dist;
                    }
                }
            }
        }
        let enemy: T = knowledge[enemy_index].clone();
        let ally: T = knowledge[ally_index].clone();

        // NOTE Re-calculate weights
        for attr in 0..n_attrs {
            weights[attr] += (known.get_attr(attr) - enemy.get_attr(attr)).abs()
                - (known.get_attr(attr) - ally.get_attr(attr)).abs();
        }
    } // NOTE END Greedy

    let mut highest_weight: f32 = 0.0;
    for attr in 0..n_attrs {
        // NOTE Find maximal element for future normalizing.
        if weights[attr] > highest_weight {
            highest_weight = weights[attr];
        }
        // NOTE Truncate negative values as 0.
        if weights[attr] < 0.0 {
            weights[attr] = 0.0;
        }
    }

    let mut reduction: f32 = 0.0;
    // NOTE Normalize weights
    for attr in 0..n_attrs {
        weights[attr] = weights[attr] / highest_weight;
        if weights[attr] < 0.2 {
            reduction += 1.0;
        }
    }

    let correct = classifier_1nn_weights(&knowledge, &exam, &weights, discard_low_weights);

    reduction = 100.0 * reduction / n_attrs as f32;

    let f = 0.5 * reduction + 0.5 * (100.0 * correct as f32 / exam.len() as f32);

    if discard_low_weights {
        println!(
            "\t\tRelief discarding weights under 0.2: \n\t\t\tReduction rate: {} \n\t\t\tSuccess percentage: {}/{} = {}\n\t\t\tEvaluation function: {}",
            reduction,
            correct,
            exam.len(),
            100.0 * correct as f32 / exam.len() as f32,
            f);
    } else {
        println!(
            "\t\tRelief without discarding weights under 0.2: \n\t\t\tReduction rate: {} \n\t\t\tSuccess percentage: {}/{} = {}\n\t\t\tEvaluation function: {}",
            reduction,
            correct,
            exam.len(),
            100.0 * correct as f32 / exam.len() as f32,
            f);
    }

    Ok(())
}

fn run<T: Data<T> + Clone + Copy>(
    _path: String,
    _n_attrs: usize,
    _folds: usize,
) -> Result<(), Box<std::error::Error>> {
    //NOTE Read CSV
    let mut csv_reader = csv::Reader::from_path(_path).expect("Error reading csv file");
    let mut data: Vec<T> = Vec::new();

    let mut id = 0;
    // NOTE CSV -> Data.
    for result in csv_reader.records() {
        let mut aux_record = T::new();
        let record = result?;
        let mut counter = 0;
        for field in record.iter() {
            // NOTE CSV structure: id , ... attributes ... , class
            if counter != _n_attrs {
                aux_record.set_attr(counter, field.parse::<f32>().unwrap());
            } else {
                aux_record.set_class(field.parse::<i32>().unwrap());
                aux_record.set_id(id);
                id += 1;
            }

            counter += 1;
        }

        data.push(aux_record);
    }

    let data: Vec<Vec<T>> = make_partitions(&data, 5);

    for i in 0.._folds {
        let mut knowledge: Vec<T> = Vec::new();
        for j in 0.._folds {
            if j != i {
                knowledge.extend(&data[j]);
            }
        }
        println!("\tPartition test: {}", i);
        let exam = data[i].clone();

        let mut now = Instant::now();
        classifier_1nn(&knowledge, &exam);
        println!("\t\t Time elapsed: {} ms.\n", now.elapsed().as_millis());

        now = Instant::now();
        relief(&knowledge, &exam, _n_attrs, true)?;
        println!("\t\t Time Elapsed: {} ms.\n", now.elapsed().as_millis());

        now = Instant::now();
        relief(&knowledge, &exam, _n_attrs, false)?;
        println!("\t\t Time elapsed: {} ms.\n", now.elapsed().as_millis());
    }

    Ok(())
}

fn main() {
    println!("Results for Texture.");
    if let Err(err) = run::<Texture>(String::from("data/texture.csv"), 40, 5) {
        println!("Error running Texture: {}", err);
        std::process::exit(1);
    }

    println!("Results for Colposcopy.");
    if let Err(err) = run::<Colposcopy>(String::from("data/colposcopy.csv"), 62, 5) {
        println!("Error running Texture: {}", err);
        std::process::exit(1);
    }

    println!("Results for Ionosphere.");
    if let Err(err) = run::<Ionosphere>(String::from("data/ionosphere.csv"), 34, 5) {
        println!("Error running Texture: {}", err);
        std::process::exit(1);
    }
}
