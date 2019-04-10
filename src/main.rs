extern crate csv;
extern crate rand;
extern crate serde_derive;
#[macro_use]
extern crate prettytable;

pub mod structs;
use prettytable::Table;
use rand::distributions::{Distribution, Normal, Uniform};
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::BTreeSet;
use std::collections::HashMap;
use std::time::Instant;
use structs::*;

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

/// Adds a random value of a normal distribution N(0.0, `desv`), to the vector `weights` at the position `index_to_mutate`
///
/// # Arguments
/// * `weights` - Vector to  be muted.
/// * `desv` - Typical desviation to be used.
/// * `index_to_mutate` - Index to mutate in the vector.
pub fn mutate_weights(weights: &mut Vec<f32>, desv: f64, index_to_mutate: usize, rng: &mut StdRng) {
    let normal = Normal::new(0.0, desv);
    weights[index_to_mutate] += normal.sample(rng) as f32;
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

// ********************************************************************************************************
// ********************************************************************************************************
// ********************************************************************************************************
// ********************************************************************************************************
// ********************************************************************************************************
// ********************************************************************************************************
// ********************************************************************************************************
// ********************************************************************************************************

pub fn low_intensity_local_search_weights<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    weights: &mut Vec<f32>,
    n_attrs: usize,
    rng: &mut StdRng,
) -> Results {
    // NOTE Initialize vector of index
    let mut index_vec: Vec<usize> = (0..n_attrs).collect();
    let mut aux_vec = index_vec.clone();
    index_vec.shuffle(rng);
    aux_vec.shuffle(rng);
    index_vec.append(&mut aux_vec);

    let mut best_result = classifier_1nn(training, training, weights, true);

    let mut _mutations = 0;
    for index_to_mute in index_vec {
        let mut muted_weights = weights.clone();
        mutate_weights(&mut muted_weights, 0.3, index_to_mute, rng);

        let muted_result = classifier_1nn(training, training, &muted_weights, true);

        //NOTE if muted_weights is better
        if muted_result.evaluation_function() > best_result.evaluation_function() {
            _mutations += 1;

            // NOTE Save new best results.
            *weights = muted_weights.clone();
            best_result = muted_result;
            // NOTE Refresh index vector
        }
    }
    //println!("Mutations: {}", _mutations);
    return best_result;
}

pub fn compite(a: &Chromosome, b: &Chromosome) -> Chromosome {
    if a.result.evaluation_function() >= b.result.evaluation_function() {
        return a.clone();
    }
    return b.clone();
}

pub fn generational_selection_and_cross<T: Data<T> + Clone + Copy>(
    generation: &Vec<Chromosome>,
    training: &Vec<T>,
    n_attrs: usize,
    rng: &mut StdRng,
) -> (Chromosome, Chromosome) {
    let parent1 = compite(
        &generation[rng.gen_range(0, generation.len())],
        &generation[rng.gen_range(0, generation.len())],
    );
    let parent2 = compite(
        &generation[rng.gen_range(0, generation.len())],
        &generation[rng.gen_range(0, generation.len())],
    );
    let parent3 = compite(
        &generation[rng.gen_range(0, generation.len())],
        &generation[rng.gen_range(0, generation.len())],
    );

    let parent4 = compite(
        &generation[rng.gen_range(0, generation.len())],
        &generation[rng.gen_range(0, generation.len())],
    );

    let mut weights1 = vec![0.0; n_attrs];
    let mut weights2 = weights1.clone();

    for i in 0..n_attrs {
        weights1[i] += (parent1.weights[i] + parent2.weights[i]) / 2.0;
        weights2[i] += (parent3.weights[i] + parent4.weights[i]) / 2.0;
    }

    let children1 = Chromosome::new(
        &weights1,
        classifier_1nn(&training, &training, &weights1, true),
    );
    let children2 = Chromosome::new(
        &weights2,
        classifier_1nn(&training, &training, &weights2, true),
    );

    return (children1, children2);
}

pub fn initial_generation<T: Data<T> + Clone + Copy>(
    generation_size: usize,
    n_attrs: usize,
    training: &Vec<T>,
    rng: &mut StdRng,
) -> Vec<Chromosome> {
    let mut generation = Vec::<Chromosome>::new();
    for _ in 0..generation_size {
        let mut weights: Vec<f32> = vec![0.0; n_attrs];
        let uniform = Uniform::new(0.0, 1.0);
        for attr in 0..n_attrs {
            weights[attr] += uniform.sample(rng);
        }
        let res = classifier_1nn(training, training, &weights, true);
        generation.push(Chromosome::new(&weights, res.clone()));
    }
    generation.sort();
    return generation;
}

pub fn generational_genetic_algorithm<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    prob_cruce: f32,
    prob_mut: f32,
    generation_size: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut generation: Vec<Chromosome> =
        initial_generation(generation_size, n_attrs, training, rng)
            .into_iter()
            .collect();

    let mut n_calls_to_ev = 0;
    let mut n_generation = 0;
    while n_calls_to_ev < 15000 {
        n_generation += 1;
        let mut next_generation = Vec::<Chromosome>::new();
        println!(
            "Generación {}:\n\tTamaño de la población: {}\n\tMejor de la generación: {}",
            n_generation,
            generation.len(),
            generation[generation.len() - 1]
                .result
                .evaluation_function()
        );

        for i in (0..generation_size).step_by(2) {
            if (i as f32) < generation_size as f32 * prob_cruce {
                let childrens =
                    generational_selection_and_cross(&generation, training, n_attrs, rng);
                next_generation.push(childrens.0.clone());
                next_generation.push(childrens.1.clone());
                n_calls_to_ev += 2;
            } else {
                next_generation.push(generation[rng.gen_range(0, generation_size)].clone());
                next_generation.push(generation[rng.gen_range(0, generation_size)].clone());
            }
        }
        let mut n_muts = (prob_mut * generation_size as f32) as usize;
        if n_muts < 1 {
            n_muts = 1;
        }
        // NOTE Mutation
        for _ in 0..n_muts {
            let random_value = rng.gen_range(0, generation_size * n_attrs);
            let chromosome = random_value / n_attrs;
            let attr = random_value % n_attrs;
            let item = next_generation[chromosome].clone();
            next_generation.remove(chromosome);
            let mut weights = item.weights;
            mutate_weights(&mut weights, 0.3, attr, rng);
            let res = classifier_1nn(training, training, &weights, true);
            next_generation.push(Chromosome::new(&weights, res));
            n_calls_to_ev += 1;
        }
        next_generation.sort();

        let best_of_last_generation = generation[generation_size - 1].clone();
        let best_of_this_generation = next_generation[generation_size - 1].clone();

        println!(
            "\tComparación: Anterior {} : Nueva {}",
            best_of_last_generation.ev(),
            best_of_this_generation.ev()
        );

        if best_of_this_generation.ev() < best_of_last_generation.ev() {
            next_generation.remove(next_generation.len() - 1);
            next_generation.push(best_of_last_generation);
        }
        generation = next_generation.clone();
    }

    return generation[generation_size - 1].weights.clone();
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
pub fn run_p2<T: Data<T> + Clone + Copy>(
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

    for i in 0..folds {
        let mut knowledge: Vec<T> = Vec::new();
        for j in 0..folds {
            if j != i {
                knowledge.extend(&data[j]);
            }
        }

        let exam = data[i].clone();

        let res = classifier_1nn(
            &knowledge,
            &exam,
            &generational_genetic_algorithm(&knowledge, n_attrs, 0.7, 0.001, 30, rng),
            true,
        );
        println!("Resultado {} ", res.evaluation_function());
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 1;
    if args.len() > 1 {
        seed = args[1].parse::<u64>().unwrap();
    }
    let mut rng: StdRng = SeedableRng::seed_from_u64(seed);

    let do_texture = true;
    let do_colpos = false;
    let do_iono = false;

    println!("# Current Results.");
    if do_texture {
        println!("## Results for Texture.\n");
        if let Err(err) = run_p2::<Texture>(String::from("data/texture.csv"), 40, 5, &mut rng) {
            println!("Error running Texture: {}", err);
            std::process::exit(1);
        }
    }

    if do_colpos {
        println!("## Results for Colposcopy.\n");
        if let Err(err) = run_p1::<Colposcopy>(String::from("data/colposcopy.csv"), 62, 5, &mut rng)
        {
            println!("Error running Colposcopy: {}", err);
            std::process::exit(1);
        }
    }

    if do_iono {
        println!("## Results for Ionosphere.\n");
        if let Err(err) = run_p1::<Ionosphere>(String::from("data/ionosphere.csv"), 34, 5, &mut rng)
        {
            println!("Error running Ionosphere: {}", err);
            std::process::exit(1);
        }
    }
}
