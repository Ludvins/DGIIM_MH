use crate::algorithms::common::*;
use crate::types::data::*;
use prettytable::Table;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::time::Instant;
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
pub fn calculate_relief_weights<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    n_attrs: usize,
) -> Vec<f32> {
    // NOTE Initialize vector of weights.
    let mut weights: Vec<f32> = vec![0.0; n_attrs];

    // NOTE Start Greedy loop
    for known in knowledge.iter() {
        let mut enemy_distance = std::f32::MAX;
        let mut ally_distance = std::f32::MAX;
        let mut ally_index = 0;
        let mut enemy_index = 0;

        for (index, candidate) in knowledge.iter().enumerate() {
            // NOTE Skip if cantidate == known
            if candidate.get_id() != known.get_id() {
                // NOTE Pre-calculate distance
                let dist = known.euclidean_distance(candidate);
                // NOTE Ally
                if known.get_class() == candidate.get_class() {
                    if dist < ally_distance {
                        ally_index = index;
                        ally_distance = dist;
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

/// Prepares weights using a local search algorithm.
///
/// # Arguments
/// * `knowledge` - Items whose class is known.
/// * `n_attrs` - Number of attributes of the data.
/// * `rng` - Random generator.
/// * `discarding_low_weights` - Boolean value, if true all weights under 0.2 are used as 0.0.
/// * `use_greedy_initial_weights` - Bollean value, if true, initial weights are the ones from RELIEF.
/// # Returns
/// The vector of weights
/// **Note**: This generates 15000 neightbours and ends if 20*`n_attrs` neightbours are generated without any improve.
pub fn calculate_local_search_weights<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    rng: &mut StdRng,
    discard_low_weights: bool,
    initial_weights: u32,
) -> Vec<f32> {
    let mut weights: Vec<f32> = vec![0.0; n_attrs];

    // NOTE Initialize weights using greedy or uniform

    match initial_weights {
        2 => weights = calculate_relief_weights(training, n_attrs),
        3 => weights = alter_greedy_weights(training, n_attrs),
        _ => {
            let uniform = Uniform::new(0.0, 1.0);
            for attr in 0..n_attrs {
                weights[attr] += uniform.sample(rng);
            }
        }
    }

    // NOTE Initialize vector of index
    let mut index_vec: Vec<usize> = (0..n_attrs).collect();
    index_vec.shuffle(rng);

    let mut best_result = classifier_1nn(training, training, &weights, discard_low_weights);

    let max_neighbours_without_muting = 20 * n_attrs;
    let mut n_neighbours_generated_without_muting = 0;
    let mut _mutations = 0;
    for _ in 0..15000 {
        let index_to_mute = index_vec.pop().expect("Index vector empty!.");
        let mut muted_weights = weights.clone();
        mutate_weights(&mut muted_weights, 0.3, index_to_mute, rng);

        let muted_result = classifier_1nn(training, training, &muted_weights, discard_low_weights);

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
            index_vec.shuffle(rng);
        } else {
            n_neighbours_generated_without_muting += 1;
            if n_neighbours_generated_without_muting == max_neighbours_without_muting {
                break;
            }
            //NOTE if no more index to mutate, recharge them.
            if index_vec.is_empty() {
                index_vec = (0..n_attrs).collect();
                index_vec.shuffle(rng);
            }
        }
    }
    //println!("Mutations: {}", _mutations);
    return weights;
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
/// **Note:** If no enemy is found, this algorithm doesn't work, as it isn't contemplated. I'm not fixing this becasuse it can't happen in our project.
pub fn alter_greedy_weights<T: Data<T> + Clone + Copy>(
    knowledge: &Vec<T>,
    n_attrs: usize,
) -> Vec<f32> {
    // NOTE Initialize vector of weights.
    let mut weights: Vec<f32> = vec![0.0; n_attrs];
    let mut attr_sum = vec![0.0; n_attrs];
    // NOTE Start Greedy loop
    for known in knowledge.iter() {
        let mut enemy_distance = std::f32::MAX;
        let mut enemy_index = 0;

        for (index, candidate) in knowledge.iter().enumerate() {
            // NOTE Skip if cantidate == known
            if known.get_class() != candidate.get_class() {
                let dist = known.euclidean_distance(candidate);
                if dist < enemy_distance {
                    enemy_index = index;
                    enemy_distance = dist;
                }
            }
        }

        let enemy: T = knowledge[enemy_index].clone();
        for attr in 0..n_attrs {
            attr_sum[attr] += (enemy.get_attr(attr) - known.get_attr(attr)).abs();
        }
    }

    let mut max_value = 0.0;
    let mut max_index = 0;
    for attr in 0..n_attrs {
        if attr_sum[attr] > max_value {
            max_index = attr;
            max_value = attr_sum[attr];
        }
    }

    weights[max_index] = 1.0;
    return weights;
}

///  Using the csv in `path`, prepares everything to call the different classifiers.
///
/// # Arguments
/// * `path` - CSV file path.
/// * `n_attrs` - Number of attributes of the data.
/// * `folds` - Number of partitions to make (calls `make_partitions`).
/// * `rng` - Random number generator.
///
/// **Note**: Doesn't return anything just print the result of each test.
pub fn run_p1<T: Data<T> + Clone + Copy>(
    path: String,
    n_attrs: usize,
    folds: usize,
    rng: &mut StdRng,
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
    let mut table_ls3 = table_1nn.clone();
    let mut table_greedy1 = table_1nn.clone();

    let do_1nn = true;
    let do_relief1 = true;
    let do_relief2 = true;
    let do_greedy1 = true;
    let do_ls1 = true;
    let do_ls2 = true;
    let do_ls3 = true;

    for i in 0..folds {
        let mut knowledge: Vec<T> = Vec::new();
        for j in 0..folds {
            if j != i {
                knowledge.extend(&data[j]);
            }
        }
        let exam = data[i].clone();

        if do_1nn {
            let now = Instant::now();
            let nn_result = classifier_1nn(&knowledge, &exam, &vec![1.0; n_attrs], false);
            table_1nn.add_row(row![
                i,
                nn_result.success_percentage(),
                nn_result.reduction_rate(),
                nn_result.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
        if do_relief1 {
            let now = Instant::now();
            let relief_result = classifier_1nn(
                &knowledge,
                &exam,
                &calculate_relief_weights(&knowledge, n_attrs),
                true,
            );
            table_relief1.add_row(row![
                i,
                relief_result.success_percentage(),
                relief_result.reduction_rate(),
                relief_result.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_relief2 {
            let now = Instant::now();
            let relief_result2 = classifier_1nn(
                &knowledge,
                &exam,
                &calculate_relief_weights(&knowledge, n_attrs),
                false,
            );
            table_relief2.add_row(row![
                i,
                relief_result2.success_percentage(),
                relief_result2.reduction_rate(),
                relief_result2.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_greedy1 {
            let now = Instant::now();
            let greedy_result = classifier_1nn(
                &knowledge,
                &exam,
                &alter_greedy_weights(&knowledge, n_attrs),
                true,
            );
            table_greedy1.add_row(row![
                i,
                greedy_result.success_percentage(),
                greedy_result.reduction_rate(),
                greedy_result.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_ls1 {
            let now = Instant::now();

            let ls_result = classifier_1nn(
                &knowledge,
                &exam,
                &calculate_local_search_weights(&knowledge, n_attrs, rng, true, 1),
                true,
            );

            table_ls1.add_row(row![
                i,
                ls_result.success_percentage(),
                ls_result.reduction_rate(),
                ls_result.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_ls2 {
            let now = Instant::now();
            let ls_result2 = classifier_1nn(
                &knowledge,
                &exam,
                &calculate_local_search_weights(&knowledge, n_attrs, rng, true, 2),
                true,
            );

            table_ls2.add_row(row![
                i,
                ls_result2.success_percentage(),
                ls_result2.reduction_rate(),
                ls_result2.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_ls3 {
            let now = Instant::now();
            let ls_result3 = classifier_1nn(
                &knowledge,
                &exam,
                &calculate_local_search_weights(&knowledge, n_attrs, rng, true, 3),
                true,
            );

            table_ls3.add_row(row![
                i,
                ls_result3.success_percentage(),
                ls_result3.reduction_rate(),
                ls_result3.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
    }
    if do_1nn {
        println!("1-NN");
        table_1nn.printstd();
    }
    if do_relief1 {
        println!("Relief 1");
        table_relief1.printstd();
    }
    if do_relief2 {
        println!("Relief 2");
        table_relief2.printstd();
    }
    if do_greedy1 {
        println!("Greedy 1");
        table_greedy1.printstd();
    }
    if do_ls1 {
        println!("Local Search 1");
        table_ls1.printstd();
    }
    if do_ls2 {
        println!("Local Search 2");
        table_ls2.printstd();
    }
    if do_ls3 {
        println!("Local Search 3");
        table_ls3.printstd();
    }

    Ok(())
}
