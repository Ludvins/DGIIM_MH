extern crate csv;
extern crate rand;
extern crate serde_derive;

//use std::sync::{Arc, Mutex}; //TODO Concurrency
//use std::thread;

pub mod structs;
use prettytable::Table;
use rand::distributions::{Distribution, Normal, Uniform};
use rand::seq::SliceRandom;
use rand::thread_rng;
use std::collections::HashMap;
use std::time::Instant;
use structs::*;
#[macro_use]
extern crate prettytable;

// Normalizes the given vector and sets negatives values to 0.
fn normalize_and_truncate_negative_weights(weights: &mut Vec<f32>) {
    // NOTE Normalize
    let mut highest_weight: f32 = 0.0;
    for attr in 0..weights.len() {
        // NOTE Find maximal element for future normalizing.
        if weights[attr] > highest_weight {
            highest_weight = weights[attr];
        }
        // NOTE Truncate negative values as 0.
        if weights[attr] < 0.0 {
            weights[attr] = 0.0;
        }
    }

    // NOTE Normalize weights
    for attr in 0..weights.len() {
        weights[attr] = weights[attr] / highest_weight;
    }
}

/// Makes partitions keeping the class diversity.
///
/// # Arguments
///
/// * `data` - Vec with all the data.
/// * `folds` - Number of partitions
///
/// # Returns
/// Returns a vector with `folds` vectors of `T`.
pub fn make_partitions<T: Data<T> + Clone + Copy>(data: &Vec<T>, folds: usize) -> Vec<Vec<T>> {
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

/// Clasifies all items in `exam`, using the ones in `knowledge` and weights for each attribute.
///
/// # Arguments
/// * `knowledge` - Items whose class is known.
/// * `exam` - Items whose class is unknown.
/// * `weights` - Vector of weights for each attribute.
/// * `discarding_low_weights` - Boolean value, if true all weights under 0.2 are used as 0.0.
///
/// # Returns
///
/// Returns an instance of Results.
pub fn classifier_1nn<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    exam: &Vec<T>,
    weights: &Vec<f32>,
    discard_low_weights: bool,
) -> Results {
    let mut correct: u32 = 0;
    // NOTE Make the test over each element in exam.
    for test in exam.iter() {
        let mut nearest_example: T = T::new();
        let mut min_distance: f32 = std::f32::MAX;

        for known in knowledge.iter() {
            if known.get_id() == test.get_id() {
                continue;
            }
            let mut distance = 0.0;
            for index in 0..weights.len() {
                // NOTE Weights below 0.2 aren't considered if `discarding_low_weights`.
                if !discard_low_weights || weights[index] >= 0.2 {
                    distance += weights[index]
                        * (test.get_attr(index) - known.get_attr(index))
                        * (test.get_attr(index) - known.get_attr(index))
                }
            }

            distance = distance.sqrt();

            if distance < min_distance {
                min_distance = distance;
                nearest_example = known.clone();
            }
        }

        // NOTE Verify it `test` was clasified correctly.
        if nearest_example.get_class() == test.get_class() {
            correct += 1;
        }
    }
    return Results::new(weights, correct, exam.len());
}

/// Prepares weights using a Greedy algorithm.
///
/// # Arguments
/// * `knowledge` - Items whose class is known.
/// * `n_attrs` - Number of attributes of the data.
///
/// # Return
/// Returns the vector of weights
///
/// **Note:** If no ally or enemy is found, this algorithm doesn't work, as it isn't contemplated. I'm not fixing this becasuse it can't happen in our project.
pub fn calculate_greedy_weights<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    n_attrs: usize,
) -> Vec<f32> {
    // NOTE Initialize vector of weights.
    let mut weights: Vec<f32> = vec![0.0; n_attrs];

    // NOTE Start Greedy loop
    for known in knowledge.iter() {
        let mut enemy_distance = std::f32::MAX;
        let mut friend_distance = std::f32::MAX;
        let mut ally_index = 0;
        let mut enemy_index = 0;

        for (index, candidate) in knowledge.iter().enumerate() {
            // NOTE Skip if cantidate == known
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
    }
    // NOTE END Greedy
    normalize_and_truncate_negative_weights(&mut weights);

    return weights;
}

/// Calls `calculate_greedy_weights` and then `classifier_1nn_weights`.
///
/// # Arguments
/// * `knowledge` - Items whose class is known.
/// * `exam` - Items whose class is unknown.
/// * `n_attrs` - Number of attributes of the data.
/// * `discarding_low_weights` - Boolean value, if true all weights under 0.2 are used as 0.0.
///
/// **Note**: Returns an instance of Results
pub fn relief<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    exam: &Vec<T>,
    n_attrs: usize,
    discard_low_weights: bool,
) -> Results {
    // NOTE Initialize vector of weights.
    let weights = calculate_greedy_weights(knowledge, n_attrs);

    return classifier_1nn(&knowledge, &exam, &weights, discard_low_weights);
}

/// Adds a random value of a normal distribution N(0.0, `desv`), to the vector `weights` at the position `index_to_mutate`
///
/// # Arguments
/// * `weights` - Vector to  be muted.
/// * `desv` - Typical desviation to be used.
/// * `index_to_mutate` - Index to mutate in the vector.
pub fn mutate_weights(weights: &mut Vec<f32>, desv: f64, index_to_mutate: usize) {
    let normal = Normal::new(0.0, desv);
    let mut rng = thread_rng();
    weights[index_to_mutate] += normal.sample(&mut rng) as f32;
    if weights[index_to_mutate] > 1.0 {
        weights[index_to_mutate] = 1.0;
    } else {
        if weights[index_to_mutate] < 0.0 {
            weights[index_to_mutate] = 0.0;
        }
    }
}
/// Prepares weights using a local search algorithm.
///
/// # Arguments
/// * `knowledge` - Items whose class is known.
/// * `n_attrs` - Number of attributes of the data.
/// * `discarding_low_weights` - Boolean value, if true all weights under 0.2 are used as 0.0.
/// # Returns
/// The vector of weights
/// **Note**: This generates 15000 neightbours and ends if 20*`n_attrs` neightbours are generated without any improve.
pub fn calculate_local_search_weights<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    discard_low_weights: bool,
    use_greedy_initial_weights: bool,
) -> Vec<f32> {
    let mut weights: Vec<f32> = vec![0.0; n_attrs];

    let mut rng = thread_rng();
    // NOTE Initialize weights using greedy or uniform
    if use_greedy_initial_weights {
        weights = calculate_greedy_weights(&training, n_attrs);
    } else {
        let uniform = Uniform::new(0.0, 1.0);
        for attr in 0..n_attrs {
            weights[attr] += uniform.sample(&mut rng);
        }
    }
    // NOTE Initialize vector of index
    let mut index_vec: Vec<usize> = (0..n_attrs).collect();
    index_vec.shuffle(&mut rng);

    let mut best_result = classifier_1nn(&training, &training, &weights, discard_low_weights);

    let max_neighbours_without_muting = 20 * n_attrs;
    let mut n_neighbours_generated_without_muting = 0;
    let mut _mutations = 0;
    for _ in 0..15000 {
        let index_to_mute = index_vec.pop().expect("Index vector empty!.");
        let mut muted_weights = weights.clone();
        mutate_weights(&mut muted_weights, 0.3, index_to_mute);

        let muted_result =
            classifier_1nn(&training, &training, &muted_weights, discard_low_weights);

        //NOTE if muted_weights is better
        if muted_result.evaluation_function() > best_result.evaluation_function() {
            _mutations += 1;
            // NOTE Reset neighbours count.
            n_neighbours_generated_without_muting = 0;

            // NOTE Save new best results.
            weights = muted_weights;
            best_result = muted_result;
            // NOTE Refresh index vector
            index_vec = (0..n_attrs).collect();
            index_vec.shuffle(&mut rng);
        } else {
            n_neighbours_generated_without_muting += 1;
            if n_neighbours_generated_without_muting == max_neighbours_without_muting {
                break;
            }
            //NOTE if no more index to mutate, recharge them.
            if index_vec.is_empty() {
                index_vec = (0..n_attrs).collect();
                index_vec.shuffle(&mut rng);
            }
        }
    }
    //println!("Mutations: {}", _mutations);
    return weights;
}

/// Calls `calculate_local_search_weights` and then `classifier_1nn_weights`.
///
/// # Arguments
/// * `knowledge` - Items whose class is known.
/// * `exam` - Items whose class is unknown.
/// * `n_attrs` - Number of attributes of the data.
/// * `discarding_low_weights` - Boolean value, if true all weights under 0.2 are used as 0.0.
/// # Returns
/// An instance of Results
pub fn local_search<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    exam: &Vec<T>,
    n_attrs: usize,
    discard_low_weights: bool,
    use_greedy_initial_weights: bool,
) -> Results {
    let weights: Vec<f32> = calculate_local_search_weights(
        knowledge,
        n_attrs,
        discard_low_weights,
        use_greedy_initial_weights,
    );

    let result = classifier_1nn(&knowledge, &exam, &weights, discard_low_weights);
    return result;
}

///  Using the csv in `path`, prepares everything to call the different classifiers.
///
/// # Arguments
/// * `path` - CSV file path.
/// * `n_attrs` - Number of attributes of the data.
/// * `folds` - Number of partitions to make (calls `make_partitions`).
///
/// **Note**: Doesn't return anything just print the result of each test.
pub fn run<T: Data<T> + Clone + Copy>(
    path: String,
    n_attrs: usize,
    folds: usize,
) -> Result<(), Box<std::error::Error>> {
    // NOTE Read CSV
    let mut csv_reader = csv::Reader::from_path(path).expect("Error reading csv file");
    let mut data: Vec<T> = Vec::new();

    let mut id = 0;
    // NOTE CSV -> Data.
    for result in csv_reader.records() {
        let mut aux_record = T::new();
        let record = result?;
        let mut counter = 0;
        for field in record.iter() {
            // NOTE CSV structure: attributes... ,class
            if counter != n_attrs {
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

    let mut table_1nn = Table::new();
    table_1nn.add_row(row![
        "Partición",
        "Tasa de clasificación",
        "Tasa de reducción",
        "Agregado",
        "Tiempo"
    ]);
    let mut table_relief1 = table_1nn.clone();
    let mut table_relief2 = table_1nn.clone();
    let mut table_ls1 = table_1nn.clone();
    let mut table_ls2 = table_1nn.clone();

    for i in 0..folds {
        let mut knowledge: Vec<T> = Vec::new();
        for j in 0..folds {
            if j != i {
                knowledge.extend(&data[j]);
            }
        }
        let exam = data[i].clone();

        let mut now = Instant::now();
        let nn_result = classifier_1nn(&knowledge, &exam, &vec![1.0; n_attrs], false);
        table_1nn.add_row(row![
            i,
            nn_result.success_percentage(),
            nn_result.reduction_rate(),
            nn_result.evaluation_function(),
            now.elapsed().as_millis()
        ]);

        now = Instant::now();
        let relief_result = relief(&knowledge, &exam, n_attrs, true);
        table_relief1.add_row(row![
            i,
            relief_result.success_percentage(),
            relief_result.reduction_rate(),
            relief_result.evaluation_function(),
            now.elapsed().as_millis()
        ]);

        now = Instant::now();
        let relief_result2 = relief(&knowledge, &exam, n_attrs, false);
        table_relief2.add_row(row![
            i,
            relief_result2.success_percentage(),
            relief_result2.reduction_rate(),
            relief_result2.evaluation_function(),
            now.elapsed().as_millis()
        ]);

        now = Instant::now();
        let ls_result = local_search(&knowledge, &exam, n_attrs, true, false);
        table_ls1.add_row(row![
            i,
            ls_result.success_percentage(),
            ls_result.reduction_rate(),
            ls_result.evaluation_function(),
            now.elapsed().as_millis()
        ]);

        now = Instant::now();
        let ls_result2 = local_search(&knowledge, &exam, n_attrs, true, true);

        table_ls2.add_row(row![
            i,
            ls_result2.success_percentage(),
            ls_result2.reduction_rate(),
            ls_result2.evaluation_function(),
            now.elapsed().as_millis()
        ]);
    }
    println!("1-NN");
    table_1nn.printstd();
    println!("Relief 1");
    table_relief1.printstd();
    println!("Relief 2");
    table_relief2.printstd();
    println!("Local Search 1");
    table_ls1.printstd();
    println!("Local Search 2");
    table_ls2.printstd();
    Ok(())
}

fn main() {
    println!("# Current Results.");
    println!("## Results for Texture.\n");
    if let Err(err) = run::<Texture>(String::from("data/texture.csv"), 40, 5) {
        println!("Error running Texture: {}", err);
        std::process::exit(1);
    }

    println!("## Results for Colposcopy.\n");
    if let Err(err) = run::<Colposcopy>(String::from("data/colposcopy.csv"), 62, 5) {
        println!("Error running Colposcopy: {}", err);
        std::process::exit(1);
    }

    println!("## Results for Ionosphere.\n");
    if let Err(err) = run::<Ionosphere>(String::from("data/ionosphere.csv"), 34, 5) {
        println!("Error running Ionosphere: {}", err);
        std::process::exit(1);
    }
}
