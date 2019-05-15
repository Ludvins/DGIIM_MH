use crate::algorithms::common::*;
use crate::algorithms::p1::*;
use crate::types::chromosome::*;
use crate::types::data::*;
use prettytable::Table;
use rand::distributions::Uniform;
use rand::prelude::*;
use rand::seq::SliceRandom;
use std::collections::BTreeSet;
use std::time::Instant;

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
        }

        match memetic_type {
            2 => {
                let selected: Vec<usize> =
                    (0..generation_size).choose_multiple(rng, generation_size / 10);
                for index in selected {
                    n_calls_to_ev +=
                        memetic_local_search_weights(training, &mut generation[index], n_attrs, rng)
                }
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
