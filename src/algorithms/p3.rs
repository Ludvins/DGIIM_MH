use crate::algorithms::common::*;
use crate::types::data::*;
use prettytable::Table;
use rand::distributions::Uniform;
use rand::prelude::*;
use std::time::Instant;

pub fn metrop(diff: f32, t: f32, rng: &mut StdRng) -> bool {
    //TODO aqui hay una k

    let uniform = Uniform::new(0.0, 1.0);
    let random = uniform.sample(rng);
    let exp_value = (-1 as f32 * diff as f32 / t).exp();
    // println!(
    //     "[Metrop]: Valor aleatorio {}, exp {}, diff {}",
    //     random, exp_value, diff
    //);
    return diff < 0.0 || random <= exp_value;
}

pub fn annealing<T: Data<T> + Clone + Copy>(
    n_attrs: usize,
    training: &Vec<T>,
    max_neighbours: usize,
    max_success: usize,
    cooling_type: usize,
    rng: &mut StdRng,
) -> Vec<f32> {
    let mut best_solution: Vec<f32> = vec![0.0; n_attrs];
    let uniform = Uniform::new(0.0, 1.0);
    for attr in 0..n_attrs {
        best_solution[attr] += uniform.sample(rng);
    }
    let mut best_cost =
        1.0 - classifier_1nn(training, training, &best_solution, true).evaluation_function();
    let mut actual_solution = best_solution.clone();
    let mut actual_cost = best_cost;
    let initial_temp = 0.3 * actual_cost as f32 / (-1 as f32 * (0.3 as f32).ln());
    let mut temp = initial_temp;
    let final_temp = 0.001; //TODO Test if lower than initial
    let mut n_calls_to_ev = 1;

    // NOTE Loop with tag
    'outer: loop {
        let mut n_success = 0;
        for _ in 0..max_neighbours {
            // println!(
            //     "Valor: {}\nTemperatura: {}\nTemperatura final: {}\nNúmero de exitos: {}\nNúmero llamadas a la funcion de evaluación: {}",
            //     best_cost, temp, final_temp, n_success, n_calls_to_ev
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
    let mut best_weights: Vec<f32> = vec![0.0; n_attrs];
    let uniform = Uniform::new(0.0, 1.0);
    for attr in 0..n_attrs {
        best_weights[attr] += uniform.sample(rng);
    }

    let now = Instant::now();
    let mut best_value = classifier_1nn(training, training, &best_weights, true);
    println!("{}", now.elapsed().as_millis());
    let now = Instant::now();
    local_search(
        training,
        n_attrs,
        &mut best_weights,
        &mut best_value,
        1000,
        rng,
        true,
    );
    println!("{}", now.elapsed().as_millis());
    for _ in 0..14 {
        println!("Mejor: {}", best_value);
        let mut muted_weights = best_weights.clone();
        let indexes = (0..n_attrs).choose_multiple(rng, n_attrs / 10);
        for index in indexes {
            mutate_weights(&mut muted_weights, 0.4, index, rng);
        }

        let mut muted_value = classifier_1nn(training, training, &muted_weights, true);

        local_search(
            training,
            n_attrs,
            &mut muted_weights,
            &mut muted_value,
            1000,
            rng,
            true,
        );

        if muted_value.evaluation_function() >= best_value.evaluation_function() {
            best_value = muted_value;
            best_weights = muted_weights;
        }
    }

    return best_weights;
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
    let do_ils = true;

    let mut table_es = Table::new();
    table_es.add_row(row![
        "Partición",
        "Tasa de clasificación",
        "Tasa de reducción",
        "Agregado",
        "Tiempo"
    ]);
    let mut table_ils = table_es.clone();

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
                &annealing(
                    n_attrs,
                    &knowledge,
                    10 * knowledge.len(),
                    (knowledge.len() as f32) as usize,
                    1,
                    rng,
                ),
                true,
            );

            table_es.add_row(row![
                i,
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
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
                res.success_percentage(),
                res.reduction_rate(),
                res.evaluation_function(),
                now.elapsed().as_millis()
            ]);
        }
    }

    if do_es {
        println!("Generacional con cruce aritmetico");
        table_es.printstd();
    }
    if do_ils {
        println!("Generacional con cruce aritmetico");
        table_ils.printstd();
    }

    Ok(())
}
