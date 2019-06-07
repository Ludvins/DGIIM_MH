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

pub fn local_search<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    weights: &mut Vec<f32>,
    result: &mut Results,
    max_calls_to_ev: usize,
    rng: &mut StdRng,
    discard_low_weights: bool,
) {
    // NOTE Initialize vector of index
    let mut index_vec: Vec<usize> = (0..n_attrs).collect();
    index_vec.shuffle(rng);

    let max_neighbours_without_muting = 20 * n_attrs;
    let mut n_neighbours_generated_without_muting = 0;
    let mut _mutations = 0;
    for _ in 0..max_calls_to_ev {
        let index_to_mute = index_vec.pop().expect("Index vector empty!.");
        let mut muted_weights = weights.clone();
        mutate_weights(&mut muted_weights, 0.3, index_to_mute, rng);

        let muted_result = classifier_1nn(training, training, &muted_weights, discard_low_weights);

        //NOTE if muted_weights is better
        if muted_result.evaluation_function() > result.evaluation_function() {
            _mutations += 1;
            // NOTE Reset neighbours count.
            n_neighbours_generated_without_muting = 0;

            // NOTE Save new best results.
            *weights = muted_weights;
            *result = muted_result;
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
    let mut result = classifier_1nn(training, training, &weights, true);
    local_search(
        training,
        n_attrs,
        &mut weights,
        &mut result,
        15000,
        rng,
        discard_low_weights,
    );
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
    let do_relief2 = false;
    let do_greedy1 = false;
    let do_ls1 = true;
    let do_ls2 = false;
    let do_ls3 = false;

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

pub fn truncate(value: f32) -> f32 {
    let mut ret = value;
    if ret < 0.0 {
        ret = 0.0;
    } else if ret > 1.0 {
        ret = 1.0;
    }

    return ret;
}

pub fn fitness_function<T: Data<T> + Clone + Copy>(training: &Vec<T>, chromosome: &mut Chromosome) {
    chromosome.result =
        classifier_1nn(training, training, &chromosome.weights, true).evaluation_function();
}

pub fn memetic_local_search_weights<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    chromosome: &mut Chromosome,
    n_attrs: usize,
    rng: &mut StdRng,
) -> usize {
    let mut n_evaluations = 0;
    let mut index_vec: Vec<usize> = (0..n_attrs).collect();
    index_vec.shuffle(rng);

    let mut _mutations = 0;
    for _ in 0..2 * n_attrs {
        if index_vec.is_empty() {
            index_vec = (0..n_attrs).collect();
            index_vec.shuffle(rng);
        }
        let index_to_mutate = index_vec.pop().expect("Vector of indexes is empty");
        let mut muted_weights = chromosome.weights.clone();
        mutate_weights(&mut muted_weights, 0.3, index_to_mutate, rng);

        let muted_result =
            classifier_1nn(training, training, &muted_weights, true).evaluation_function();
        n_evaluations += 1;
        if muted_result > chromosome.result {
            index_vec.clear();
            _mutations += 1;

            chromosome.weights = muted_weights;
            chromosome.result = muted_result;
        }
    }
    return n_evaluations;
}

pub fn compite(a: &Chromosome, b: &Chromosome) -> Chromosome {
    if a.result > b.result {
        return a.clone();
    }
    return b.clone();
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
        generation.push(Chromosome::new(&weights, res.evaluation_function()));
    }
    generation.sort();

    return generation;
}

pub fn initial_generation2<T: Data<T> + Clone + Copy>(
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
        generation.push(Chromosome::new(&weights, res.evaluation_function()));
    }
    generation.sort();

    generation.remove(0);
    generation.remove(0);
    let w = calculate_relief_weights(training, n_attrs);
    let res = classifier_1nn(training, training, &w, true);
    generation.push(Chromosome::new(&w, res.evaluation_function()));
    let w = alter_greedy_weights(training, n_attrs);
    let res = classifier_1nn(training, training, &w, true);
    generation.push(Chromosome::new(&w, res.evaluation_function()));

    generation.sort();

    return generation;
}

pub fn binary_tournament(
    generation: &Vec<Chromosome>,
    select_n: usize,
    rng: &mut StdRng,
) -> Vec<Chromosome> {
    let mut ret: Vec<Chromosome> = Vec::new();
    for _ in 0..select_n {
        ret.push(compite(
            &generation[rng.gen_range(0, generation.len())],
            &generation[rng.gen_range(0, generation.len())],
        ));
    }
    return ret;
}

pub fn weighted_selection(
    generation: &Vec<Chromosome>,
    select_n: usize,
    rng: &mut StdRng,
) -> Vec<Chromosome> {
    let mut ret: Vec<Chromosome> = Vec::new();

    let uniform = Uniform::new(0.0, 1.0);
    let total_sum: f32 = generation.iter().map(|x| x.result).sum();

    let mut weights: Vec<f32> = Vec::new();

    let mut acumulative = 0.0;
    for chromosome in generation {
        acumulative = acumulative + chromosome.result / total_sum;
        weights.push(acumulative);
    }

    for _ in 0..select_n {
        let random: f32 = uniform.sample(rng);
        for i in 0..weights.len() {
            if random < weights[i] {
                let parent1 = generation
                    .get(i)
                    .expect("Random number is over the limits of the generation");
                ret.push(parent1.clone());
                break;
            }
        }
    }
    return ret;
}

pub fn aritmethic_cross(
    parents: &mut Vec<Chromosome>,
    n_childs: usize,
    n_attrs: usize,
    _rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let mut children: Vec<Vec<f32>> = Vec::new();

    for _ in 0..(n_childs / 2) {
        let parent2 = parents.pop().expect("[ARITH]: No more parents");
        let parent1 = parents.pop().expect("[ARITH]: No more parents");
        let mut weights1 = vec![0.0; n_attrs];
        let mut weights2 = weights1.clone();

        for i in 0..n_attrs {
            weights1[i] += parent1.weights[i] * 0.4 + parent2.weights[i] * 0.6;
            weights2[i] += parent1.weights[i] * 0.6 + parent2.weights[i] * 0.4;
        }

        children.push(weights1);
        children.push(weights2);
    }

    return children;
}

pub fn blx_alpha_cross(
    parents: &mut Vec<Chromosome>,
    n_childs: usize,
    n_attrs: usize,
    rng: &mut StdRng,
) -> Vec<Vec<f32>> {
    let alpha = 0.3;
    let mut children: Vec<Vec<f32>> = Vec::new();
    for _ in 0..(n_childs / 2) {
        let parent2 = parents.pop().expect("[BLX]: No more parents");
        let parent1 = parents.pop().expect("[BLX]: No more parents");
        let mut weights1 = vec![0.0; n_attrs];
        let mut weights2 = vec![0.0; n_attrs];

        for i in 0..n_attrs {
            let c_max;
            let c_min;
            if parent1.weights[i] < parent2.weights[i] {
                c_max = parent2.weights[i];
                c_min = parent1.weights[i];
            } else if parent1.weights[i] > parent2.weights[i] {
                c_max = parent1.weights[i];
                c_min = parent2.weights[i];
            } else {
                weights1[i] = parent1.weights[i];
                weights2[i] = parent1.weights[i];
                continue;
            }

            let lower_bound = c_min - alpha * (c_max - c_min);
            let upper_bound = c_max + alpha * (c_max - c_min);

            let value1 = rng.gen_range(lower_bound, upper_bound);
            let value2 = rng.gen_range(lower_bound, upper_bound);

            weights1[i] += truncate(value1);
            weights2[i] += truncate(value2);
        }
        children.push(weights1);
        children.push(weights2);
    }

    return children;
}

pub fn stationary_iteration<T: Data<T> + Copy + Clone>(
    generation: &Vec<Chromosome>,
    training: &Vec<T>,
    n_attrs: usize,
    mut_prob: f32,
    selection_operator: fn(&Vec<Chromosome>, usize, &mut StdRng) -> Vec<Chromosome>,
    cross_operator: fn(&mut Vec<Chromosome>, usize, usize, &mut StdRng) -> Vec<Vec<f32>>,
    rng: &mut StdRng,
) -> (Vec<Chromosome>, usize) {
    let mut n_calls_to_ev = 0;

    let mut parents = selection_operator(&generation, 2, rng);

    let mut children = cross_operator(&mut parents, 2, n_attrs, rng);

    let n_muts = mut_prob * n_attrs as f32 * 2 as f32;
    let mut trunc = n_muts as usize;
    let dec = n_muts - trunc as f32;
    if rng.gen_range(0.0, 1.0) < dec as f32 {
        trunc += 1;
    }

    let mut nums: BTreeSet<usize> = BTreeSet::new();
    while nums.len() < trunc {
        nums.insert(rng.gen_range(0, 2 * n_attrs));
    }
    for random_value in nums.iter() {
        let child = random_value / n_attrs;
        let attr = random_value % n_attrs;
        mutate_weights(&mut children[child], 0.3, attr, rng);
    }

    let mut next_generation: Vec<Chromosome> = children
        .iter()
        .map(|x| {
            Chromosome::new(
                &x,
                classifier_1nn(training, training, &x, true).evaluation_function(),
            )
        })
        .collect();

    n_calls_to_ev += next_generation.len();

    next_generation.sort();

    let worst_child = next_generation.get(0).expect("Generation is empty").clone();
    let best_child = next_generation
        .get(1)
        .expect("Generation isnt big enought")
        .clone();

    let worst = generation.get(0).expect("Generation is empty");
    let best = generation.get(1).expect("Generation isnt big enought");
    if best_child.result < worst.result {
        next_generation.clear();
        next_generation.push(best.clone());
        next_generation.push(worst.clone());
    } else if worst_child.result < best.result {
        next_generation.remove(0);
        next_generation.push(best.clone());
    }

    next_generation.extend_from_slice(&mut (generation.clone())[2..]);

    next_generation.sort();

    return (next_generation, n_calls_to_ev);
}

pub fn generational_iteration<T: Data<T> + Clone + Copy>(
    generation: &Vec<Chromosome>,
    training: &Vec<T>,
    n_attrs: usize,
    cross_prob: f32,
    mut_prob: f32,
    generation_size: usize,
    selection_operator: fn(&Vec<Chromosome>, usize, &mut StdRng) -> Vec<Chromosome>,
    cross_operator: fn(&mut Vec<Chromosome>, usize, usize, &mut StdRng) -> Vec<Vec<f32>>,
    rng: &mut StdRng,
) -> (Vec<Chromosome>, usize) {
    let mut n_calls_to_ev = 0;

    let mut parents = selection_operator(&generation, generation_size, rng);

    let children = cross_operator(
        &mut parents,
        (cross_prob * generation_size as f32) as usize,
        n_attrs,
        rng,
    );

    let mut next_generation: Vec<Chromosome> =
        children.iter().map(|x| Chromosome::new_w(&x)).collect();

    // NOTE La generación no esta completa, no todos los padres se cruzan.
    for p in parents.iter() {
        next_generation.push(p.clone());
    }

    let n_muts = mut_prob * n_attrs as f32 * generation_size as f32;
    let mut trunc = n_muts as usize;
    let dec = n_muts - trunc as f32;
    if rng.gen_range(0.0, 1.0) < dec as f32 {
        trunc += 1;
    }

    // NOTE Mutation
    let mut nums: BTreeSet<usize> = BTreeSet::new();
    while nums.len() < trunc {
        nums.insert(rng.gen_range(0, generation_size * n_attrs));
    }

    for random_value in nums.iter() {
        let chromosome = random_value / n_attrs;
        let attr = random_value % n_attrs;
        mutate_weights(&mut next_generation[chromosome].weights, 0.3, attr, rng);
        next_generation[chromosome].result = -1.0;
    }

    for chromosome in next_generation.iter_mut() {
        if chromosome.result == -1.0 {
            fitness_function(training, chromosome);
            n_calls_to_ev += 1;
        }
    }

    next_generation.sort();

    let best_of_last_generation = generation
        .last()
        .expect("Last generation is empty.")
        .clone();
    let best_of_this_generation = next_generation
        .last()
        .expect("Current Generation is empty")
        .clone();

    if best_of_this_generation.result < best_of_last_generation.result {
        next_generation.remove(0);
        next_generation.push(best_of_last_generation.clone());
    }

    return (next_generation, n_calls_to_ev);
}

pub fn genetic_stationary_algorithm<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    mut_prob: f32,
    generation_size: usize,
    selection_operator: fn(&Vec<Chromosome>, usize, &mut StdRng) -> Vec<Chromosome>,
    cross_operator: fn(&mut Vec<Chromosome>, usize, usize, &mut StdRng) -> Vec<Vec<f32>>,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut generation: Vec<Chromosome> =
        initial_generation(generation_size, n_attrs, training, rng)
            .into_iter()
            .collect();

    let mut n_calls_to_ev = generation_size;
    let mut _n_generation = 0;
    while n_calls_to_ev < 15000 {
        let iteration = stationary_iteration::<T>(
            &generation,
            training,
            n_attrs,
            mut_prob,
            selection_operator,
            cross_operator,
            rng,
        );

        _n_generation += 1;
        generation = iteration.0;
        n_calls_to_ev += iteration.1;

        // println!(
        //     "Generation {} ({}/{}):\n\tGeneration size: {}\n\tBest of generátion: {}",
        //     _n_generation,
        //     n_calls_to_ev,
        //     15000,
        //     generation.len(),
        //     generation.last().expect("Generation is empty").result
        // );

        // for c in generation.iter() {
        //     println!(" {} ", c.result);
        // }
        // println!("\n");
    }

    return generation
        .last()
        .expect("Last generation is empty.")
        .weights
        .clone();
}

pub fn genetic_generational_algorithm<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    cross_prob: f32,
    mut_prob: f32,
    generation_size: usize,
    selection_operator: fn(&Vec<Chromosome>, usize, &mut StdRng) -> Vec<Chromosome>,
    cross_operator: fn(&mut Vec<Chromosome>, usize, usize, &mut StdRng) -> Vec<Vec<f32>>,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut generation: Vec<Chromosome> =
        initial_generation(generation_size, n_attrs, training, rng)
            .into_iter()
            .collect();

    let mut n_calls_to_ev = generation_size;
    let mut _n_generation = 0;
    while n_calls_to_ev < 15000 {
        let iteration = generational_iteration::<T>(
            &generation,
            training,
            n_attrs,
            cross_prob,
            mut_prob,
            generation_size,
            selection_operator,
            cross_operator,
            rng,
        );

        _n_generation += 1;
        generation = iteration.0;
        n_calls_to_ev += iteration.1;

        // println!(
        //     "Generation {} ({}/{}):\n\tGeneration size: {}\n\tBest of generátion: {}",
        //     _n_generation,
        //     n_calls_to_ev,
        //     15000,
        //     generation.len(),
        //     generation.last().expect("Generation is empty").result
        // );

        // for c in generation.iter() {
        //     println!(" {} ", c.result);
        // }
        // println!("\n");
    }

    return generation
        .last()
        .expect("Last generation is empty.")
        .weights
        .clone();
}

pub fn memetic_algorithm<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    cross_prob: f32,
    mut_prob: f32,
    generation_size: usize,
    memetic_type: usize,
    selection_operator: fn(&Vec<Chromosome>, usize, &mut StdRng) -> Vec<Chromosome>,
    cross_operator: fn(&mut Vec<Chromosome>, usize, usize, &mut StdRng) -> Vec<Vec<f32>>,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut generation: Vec<Chromosome> =
        initial_generation2(generation_size, n_attrs, training, rng)
            .into_iter()
            .collect();

    let mut n_calls_to_ev = generation_size;
    let mut _n_generation = 0;
    'outer: loop {
        for _ in 0..10 {
            if n_calls_to_ev >= 15000 {
                break 'outer;
            }

            let iteration = generational_iteration::<T>(
                &generation,
                training,
                n_attrs,
                cross_prob,
                mut_prob,
                generation_size,
                selection_operator,
                cross_operator,
                rng,
            );

            _n_generation += 1;
            generation = iteration.0;
            n_calls_to_ev += iteration.1;

            // println!(
            //     "Generation {} ({}/{}):\n\tGeneration size: {}\n\tBest of generátion: {}",
            //     _n_generation,
            //     n_calls_to_ev,
            //     15000,
            //     generation.len(),
            //     generation.last().expect("Generation is empty").result
            // );

            // for c in generation.iter() {
            //     println!(" {} ", c.result);
            // }
            // println!("\n");
        }

        match memetic_type {
            2 => {
                let selected: Vec<usize> =
                    (0..generation_size).choose_multiple(rng, generation_size / 10);
                // for c in generation.iter() {
                //     println!(" {} ", c.result);
                // }
                // println!("\n");

                for index in selected {
                    n_calls_to_ev +=
                        memetic_local_search_weights(training, &mut generation[index], n_attrs, rng)
                }
                // for c in generation.iter() {
                //     println!(" {} ", c.result);
                // }
                // println!("\n");
            }

            3 => {
                for index in generation_size - generation_size / 10..generation_size {
                    n_calls_to_ev +=
                        memetic_local_search_weights(training, &mut generation[index], n_attrs, rng)
                }
            }
            _ => {
                for index in 0..generation_size {
                    n_calls_to_ev +=
                        memetic_local_search_weights(training, &mut generation[index], n_attrs, rng)
                }
            }
        }

        generation.sort();
    }

    return generation
        .last()
        .expect("Last generation is empty.")
        .weights
        .clone();
}

///  Using the csv in `path`, prepares everything to call all the genetic algorithms.
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

    let do_generational_arith = true;
    let do_generational_blx = true;
    let do_stationary_arith = true;
    let do_stationary_blx = true;
    let do_roulette = false;
    let do_memetic1 = true;
    let do_memetic2 = true;
    let do_memetic3 = true;

    let mut table_generational_arith = Table::new();
    table_generational_arith.add_row(row![
        "Partición",
        "Tasa de clasificación",
        "Tasa de reducción",
        "Agregado",
        "Tiempo"
    ]);
    let mut table_generational_blx = table_generational_arith.clone();
    let mut table_stationary_arith = table_generational_arith.clone();
    let mut table_stationary_blx = table_generational_arith.clone();
    let mut table_roulette = table_stationary_arith.clone();
    let mut table_memetic1 = table_stationary_arith.clone();
    let mut table_memetic2 = table_stationary_arith.clone();
    let mut table_memetic3 = table_stationary_arith.clone();

    for i in 0..folds {
        let mut knowledge: Vec<T> = Vec::new();
        for j in 0..folds {
            if j != i {
                knowledge.extend(&data[j]);
            }
        }

        let exam = data[i].clone();

        if do_generational_arith {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &genetic_generational_algorithm(
                    &knowledge,
                    n_attrs,
                    0.7,
                    0.001,
                    30,
                    binary_tournament,
                    aritmethic_cross,
                    rng,
                ),
                true,
            );

            table_generational_arith.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_generational_blx {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &genetic_generational_algorithm(
                    &knowledge,
                    n_attrs,
                    0.7,
                    0.001,
                    30,
                    binary_tournament,
                    blx_alpha_cross,
                    rng,
                ),
                true,
            );

            table_generational_blx.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_stationary_arith {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &genetic_stationary_algorithm(
                    &knowledge,
                    n_attrs,
                    0.001,
                    30,
                    binary_tournament,
                    aritmethic_cross,
                    rng,
                ),
                true,
            );

            table_stationary_arith.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_stationary_blx {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &genetic_stationary_algorithm(
                    &knowledge,
                    n_attrs,
                    0.001,
                    30,
                    binary_tournament,
                    blx_alpha_cross,
                    rng,
                ),
                true,
            );

            table_stationary_blx.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_roulette {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &genetic_generational_algorithm(
                    &knowledge,
                    n_attrs,
                    0.7,
                    0.001,
                    30,
                    weighted_selection,
                    blx_alpha_cross,
                    rng,
                ),
                true,
            );

            table_roulette.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_memetic1 {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &memetic_algorithm(
                    &knowledge,
                    n_attrs,
                    0.7,
                    0.001,
                    10,
                    1,
                    binary_tournament,
                    blx_alpha_cross,
                    rng,
                ),
                true,
            );

            table_memetic1.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
        if do_memetic2 {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &memetic_algorithm(
                    &knowledge,
                    n_attrs,
                    0.7,
                    0.001,
                    10,
                    2,
                    binary_tournament,
                    blx_alpha_cross,
                    rng,
                ),
                true,
            );

            table_memetic2.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
        if do_memetic3 {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &memetic_algorithm(
                    &knowledge,
                    n_attrs,
                    0.7,
                    0.001,
                    10,
                    3,
                    binary_tournament,
                    blx_alpha_cross,
                    rng,
                ),
                true,
            );

            table_memetic3.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
    }

    if do_generational_arith {
        println!("Generacional con cruce aritmetico");
        table_generational_arith.printstd();
    }

    if do_generational_blx {
        println!("Generacional con cruce BLX");
        table_generational_blx.printstd();
    }

    if do_stationary_arith {
        println!("Estacionario con cruce aritmetico");
        table_stationary_arith.printstd();
    }

    if do_stationary_blx {
        println!("Estacionario con cruce BLX");
        table_stationary_blx.printstd();
    }

    if do_roulette {
        println!("Ruleta");
        table_roulette.printstd();
    }

    if do_memetic1 {
        println!("Memético 1");
        table_memetic1.printstd();
    }
    if do_memetic2 {
        println!("Memético 2");
        table_memetic2.printstd();
    }
    if do_memetic3 {
        println!("Memético 3");
        table_memetic3.printstd();
    }

    Ok(())
}
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////////////////////////

pub fn metrop(diff: f32, t: f32, rng: &mut StdRng) -> bool {
    let mut difference = diff;
    if diff == 0.0 {
        difference = 0.005;
    }
    let uniform = Uniform::new(0.0, 1.0);
    let random = uniform.sample(rng);
    let exp_value = (-1 as f32 * difference as f32 / t).exp();
    // println!(
    //     "[Metrop]: Valor aleatorio {}, exp {}, diff {}",
    //     random, exp_value, difference
    // );
    return difference < 0.0 || random <= exp_value;
}

pub fn annealing<T: Data<T> + Clone + Copy>(
    n_attrs: usize,
    training: &Vec<T>,
    max_neighbours: usize,
    max_success: usize,
    cooling_type: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    //let mut best_solution: Vec<f32> = vec![0.0; n_attrs];
    //let uniform = Uniform::new(0.0, 1.0);
    //for attr in 0..n_attrs {
    //    best_solution[attr] += uniform.sample(rng);
    //}

    let mut best_solution: Vec<f32> = calculate_relief_weights(training, n_attrs);

    let mut best_cost =
        1.0 - classifier_1nn(training, training, &best_solution, true).evaluation_function();
    let mut actual_solution = best_solution.clone();
    let mut actual_cost = best_cost;
    let initial_temp = 0.3 * actual_cost as f32 / (-1 as f32 * (0.3 as f32).ln());
    let mut temp = initial_temp;
    let mut final_temp = 0.001;
    if final_temp >= initial_temp {
        final_temp = initial_temp / 1000.0;
    }
    let mut n_calls_to_ev = 1;

    // NOTE Loop with tag
    'outer: loop {
        let mut n_success = 0;
        for _ in 0..max_neighbours {
            // println!(
            // "Valor: {}\nTemperatura: {}\nTemperatura final: {}\nNúmero de exitos: {}\nNúmero llamadas a la funcion de evaluación: {}",
            // best_cost, temp, final_temp, n_success, n_calls_to_ev
            // );
            let mut neighbour = actual_solution.clone();
            // NOTE Misma mutación que en las otras practicas.
            mutate_weights(&mut neighbour, 0.3, rng.gen_range(0, n_attrs), rng);

            let neighbour_cost =
                1.0 - classifier_1nn(training, training, &neighbour, true).evaluation_function();
            n_calls_to_ev += 1;
            let dif = neighbour_cost - actual_cost;

            if metrop(dif, temp, rng) {
                n_success += 1;
                actual_cost = neighbour_cost;
                actual_solution = neighbour.clone();

                if actual_cost < best_cost {
                    best_cost = actual_cost;
                    best_solution = neighbour;
                }
            }

            if n_calls_to_ev >= 15000 {
                break 'outer; // NOTE break outer loop.
            }
            if n_success >= max_success {
                break;
            }
        } // NOTE END INNER LOOP
        if n_success == 0 {
            break;
        }

        match cooling_type {
            1 => {
                let n_coolings = 15000 as f32 / max_neighbours as f32;
                let beta = (initial_temp - final_temp) / (initial_temp * final_temp * n_coolings);
                temp = temp / (1.0 + beta * temp);
            }
            _ => temp = 0.9 * temp,
        }
    }
    return best_solution;
}

pub fn iterated_local_search<T: Data<T> + Clone + Copy>(
    n_attrs: usize,
    training: &Vec<T>,
    rng: &mut StdRng,
) -> Vec<f32> {
    //let mut best_weights: Vec<f32> = vec![0.0; n_attrs];
    // let uniform = Uniform::new(0.0, 1.0);
    // for attr in 0..n_attrs {
    //     best_weights[attr] += uniform.sample(rng);
    // }

    //let mut best_weights: Vec<f32> = calculate_relief_weights(training, n_attrs);

    let mut best_weights: Vec<f32> = alter_greedy_weights(training, n_attrs);
    let mut best_results = classifier_1nn(training, training, &best_weights, true);
    local_search(
        training,
        n_attrs,
        &mut best_weights,
        &mut best_results,
        999,
        rng,
        true,
    );
    // println!("{}", now.elapsed().as_millis());
    for _ in 0..14 {
        println!("Mejor: {}", best_results);
        let mut muted_weights = best_weights.clone();
        let indexes = (0..n_attrs).choose_multiple(rng, n_attrs / 10);
        for index in indexes {
            mutate_weights(&mut muted_weights, 0.4, index, rng);
        }

        let mut muted_results = classifier_1nn(training, training, &muted_weights, true);

        local_search(
            training,
            n_attrs,
            &mut muted_weights,
            &mut muted_results,
            999,
            rng,
            true,
        );

        if muted_results.evaluation_function() >= best_results.evaluation_function() {
            best_results = muted_results;
            best_weights = muted_weights;
        }
    }

    return best_weights;
}

pub fn de_rand_iteration<T: Data<T> + Copy + Clone>(
    training: &Vec<T>,
    generation: &Vec<Chromosome>,
    generation_size: usize,
    n_attrs: usize,
    rng: &mut StdRng,
) -> Vec<Chromosome> {
    let uniform = Uniform::new(0.0, 1.0);
    let mut new_generation: Vec<Chromosome> = Vec::new();

    for index in 0..generation_size {
        let mut possible_parents = generation.clone();
        possible_parents.remove(index);
        let parents: Vec<Chromosome> = possible_parents.choose_multiple(rng, 3).cloned().collect();

        let mut offspring = vec![0.0; n_attrs];
        let random_gene = rng.gen_range(0, n_attrs);

        for attr in 0..n_attrs {
            if uniform.sample(rng) < 0.5 || attr == random_gene {
                offspring[attr] = truncate(
                    parents[0].weights[attr]
                        + (parents[1].weights[attr] - parents[2].weights[attr]),
                );
            } else {
                offspring[attr] = generation[index].weights[attr];
            }
        }
        let mut new: Chromosome = Chromosome::new_w(&offspring);
        fitness_function(training, &mut new);
        new_generation.push(new);
    }
    return new_generation;
}
pub fn de_best_iteration<T: Data<T> + Copy + Clone>(
    training: &Vec<T>,
    generation: &Vec<Chromosome>,
    generation_size: usize,
    n_attrs: usize,
    rng: &mut StdRng,
) -> Vec<Chromosome> {
    let uniform = Uniform::new(0.0, 1.0);
    let mut new_generation: Vec<Chromosome> = Vec::new();
    let best = generation.last().unwrap();

    for index in 0..generation_size {
        let mut possible_parents = generation.clone();
        possible_parents.remove(index);
        let parents: Vec<Chromosome> = possible_parents.choose_multiple(rng, 2).cloned().collect();

        let mut offspring = vec![0.0; n_attrs];
        let random_gene = rng.gen_range(0, n_attrs);

        for attr in 0..n_attrs {
            if uniform.sample(rng) < 0.5 || attr == random_gene {
                offspring[attr] = truncate(
                    generation[index].weights[attr]
                        + 1.0 * (best.weights[attr] - generation[index].weights[attr])
                        + 1.0 * (parents[0].weights[attr] - parents[1].weights[attr]),
                );
            } else {
                offspring[attr] = generation[index].weights[attr];
            }
        }
        let mut new: Chromosome = Chromosome::new_w(&offspring);
        fitness_function(training, &mut new);
        new_generation.push(new);
    }

    return new_generation;
}

pub fn diferential_evolution<T: Data<T> + Clone + Copy>(
    training: &Vec<T>,
    n_attrs: usize,
    generation_size: usize,
    iteration: fn(&Vec<T>, &Vec<Chromosome>, usize, usize, &mut StdRng) -> Vec<Chromosome>,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut generation: Vec<Chromosome> =
        initial_generation(generation_size, n_attrs, training, rng)
            .into_iter()
            .collect();

    generation.sort();

    // generation.remove(generation.len() - 1);
    // let w = alter_greedy_weights(training, n_attrs);
    // generation.push(Chromosome::new(
    //     &w,
    //     classifier_1nn(training, training, &w, true).evaluation_function(),
    // ));
    // generation.sort();

    let mut n_evaluations = generation_size;
    let mut c = 1;
    loop {
        for a in generation.iter() {
            println!("{} {}", c, a.result);
        }
        println!("\n");
        let new_generation = iteration(training, &generation, generation_size, n_attrs, rng);

        for index in 0..generation_size {
            if generation[index] < new_generation[index] {
                generation[index] = new_generation[index].clone();
            }
        }
        generation.sort();
        n_evaluations += generation_size;
        if n_evaluations >= 15000 {
            break;
        }
        c += 1;
    }

    return generation.last().unwrap().weights.clone();
}

///  Using the csv in `path`, prepares everything to call all the genetic algorithms.
///
/// # Arguments
/// * `path` - CSV file path.
/// * `n_attrs` - Number of attributes of the data.
/// * `folds` - Number of partitions to make (calls `make_partitions`).
/// * `rng` - Random number generator.
///
/// **Note**: Doesn't return anything just print the result of each test.
pub fn run_p3<T: Data<T> + Clone + Copy>(
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

    let do_es = false;
    let do_ils = false;
    let do_de1 = true;
    let do_de2 = false;

    let mut table_es = Table::new();
    table_es.add_row(row![
        "Partición",
        "Tasa de clasificación",
        "Tasa de reducción",
        "Agregado",
        "Tiempo"
    ]);
    let mut table_ils = table_es.clone();
    let mut table_de1 = table_es.clone();
    let mut table_de2 = table_es.clone();

    let data: Vec<Vec<T>> = make_partitions(&data, 5);
    for i in 0..folds {
        let mut knowledge: Vec<T> = Vec::new();
        for j in 0..folds {
            if j != i {
                knowledge.extend(&data[j]);
            }
        }
        let exam = data[i].clone();
        if do_es {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &annealing(n_attrs, &knowledge, 10 * n_attrs, n_attrs as usize, 1, rng),
                true,
            );

            table_es.add_row(row![
                i,
                100.0 * res.success_percentage(),
                100.0 * res.reduction_rate(),
                100.0 * res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_ils {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &iterated_local_search(n_attrs, &knowledge, rng),
                true,
            );

            table_ils.add_row(row![
                i,
                100.0 * res.success_percentage(),
                100.0 * res.reduction_rate(),
                100.0 * res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_de1 {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &diferential_evolution(&knowledge, n_attrs, 50, de_rand_iteration, rng),
                true,
            );

            table_de1.add_row(row![
                i,
                100.0 * res.success_percentage(),
                100.0 * res.reduction_rate(),
                100.0 * res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }

        if do_de2 {
            let now = Instant::now();
            let res = classifier_1nn(
                &knowledge,
                &exam,
                &diferential_evolution(&knowledge, n_attrs, 50, de_best_iteration, rng),
                true,
            );

            table_de2.add_row(row![
                i,
                100.0 * res.success_percentage(),
                100.0 * res.reduction_rate(),
                100.0 * res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
        break;
    }

    if do_es {
        println!("ES");
        table_es.printstd();
    }
    if do_ils {
        println!("ILS");
        table_ils.printstd();
    }
    if do_de1 {
        println!("DE1");
        table_de1.printstd();
    }
    if do_de2 {
        println!("DE2");
        table_de2.printstd();
    }

    Ok(())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut seed = 2;
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
        if let Err(err) = run_p3::<Texture>(String::from("data/texture.csv"), 40, 5, &mut rng) {
            println!("Error running Texture: {}", err);
            std::process::exit(1);
        }
    }

    if do_colpos {
        println!("## Results for Colposcopy.\n");
        if let Err(err) = run_p3::<Colposcopy>(String::from("data/colposcopy.csv"), 62, 5, &mut rng)
        {
            println!("Error running Colposcopy: {}", err);
            std::process::exit(1);
        }
    }

    if do_iono {
        println!("## Results for Ionosphere.\n");
        if let Err(err) = run_p3::<Ionosphere>(String::from("data/ionosphere.csv"), 34, 5, &mut rng)
        {
            println!("Error running Ionosphere: {}", err);
            std::process::exit(1);
        }
    }
}
