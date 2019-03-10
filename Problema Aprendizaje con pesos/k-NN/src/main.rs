extern crate csv;
extern crate rand;
extern crate rustc_serialize;
extern crate serde_derive;

use std::collections::HashMap;

#[allow(non_snake_case)]
#[derive(Copy, Clone)]
struct Texture {
    _id: i32,
    _attrs: [f32; 40],
    _class: i32,
}
impl Texture {
    pub fn euclidean_distance(self, other: &Texture) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum +=
                self._attrs[index] - other._attrs[index] * self._attrs[index] - other._attrs[index]
        }
        return sum;
    }

    pub fn euclidean_distance_with_weights(self, other: &Texture, w: [f32; 40]) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum += w[index] * self._attrs[index]
                - other._attrs[index] * self._attrs[index]
                - other._attrs[index]
        }

        return sum;
    }
}

fn make_partitions(data: Vec<Texture>) -> Vec<Vec<Texture>> {
    let folds = 5;

    let mut categories_count = HashMap::new();
    let mut partitions: Vec<Vec<Texture>> = Vec::new();

    for _ in 0..folds {
        partitions.push(Vec::new());
    }

    for example in data {
        let counter = categories_count.entry(example._class).or_insert(0);
        partitions[*counter].push(example);

        *counter = (*counter + 1) % folds;
    }

    return partitions;
}

fn test() -> Result<(), Box<std::error::Error>> {
    // TODO Normalize data in csv

    let _rng = rand::thread_rng();
    // Con particiones
    let folds = 5;

    let mut csv_reader = csv::Reader::from_path("data/csv_result-texture.csv")?;
    println!("Csv file read.");
    let mut data: Vec<Texture> = Vec::new();
    let _atributes: usize = 40;

    // TODO CSV -> Data.
    for result in csv_reader.records() {
        let result = result?;

        let mut id = 0;
        let mut attr = [0.0; 40];
        let mut class = 0;
        let mut counter = 0;

        for record in result.iter() {

            //     let result: Texture = result?;
            //     data.push(result);
        }
    }

    let mut _weights: [f32; 40] = [0.0; 40];

    let data: Vec<Vec<Texture>> = make_partitions(data);

    for i in 0..folds {
        // Exam is over partition i.
        let mut _correct = 0;
        let mut _attempts = 0;
        let mut knowledge: Vec<Texture> = Vec::new();

        // Learning from all other partitions.
        for j in 0..folds {
            if j != i {
                knowledge.extend(data[j].iter().cloned());
            }
        }
        println!("Aprendizaje hecho.");

        // NOTE Greedy
        for example in knowledge.iter() {
            let mut best_enemy: Texture = knowledge[0];
            let mut best_ally: Texture = knowledge[0];
            let mut enemy_distance = std::f32::MAX;
            let mut friend_distance = std::f32::MAX;

            for other in knowledge.iter() {
                // NOTE Ally
                if example._class == other._class {
                    if other._id != example._id {
                        let dist = example.euclidean_distance(other);
                        if dist < friend_distance {
                            best_ally = *other;
                            friend_distance = dist;
                        }
                    }
                }
                // NOTE Enemy
                else {
                    let dist = example.euclidean_distance(other);
                    if dist < enemy_distance {
                        best_enemy = *other;
                        enemy_distance = dist;
                    }
                }
            }
            // NOTE Re-calculate weights
            let mut highest_weight: f32 = _weights[0];
            for attr in 0.._atributes {
                _weights[attr] += f32::abs(example._attrs[attr] - best_enemy._attrs[attr])
                    - f32::abs(example._attrs[attr] - best_ally._attrs[attr]);

                if _weights[attr] > highest_weight {
                    highest_weight = _weights[attr];
                }
                if _weights[attr] < 0.0 {
                    _weights[attr] = 0.0;
                }
            }

            // NOTE Normalize weights
            for attr in 0.._atributes {
                _weights[attr] = _weights[attr] / highest_weight;

                //println!("Peso {}: {}", attr, _weights[attr]);
            }
        } // NOTE END Greedy

        // NOTE Test
        for result in data.get(i).unwrap() {
            _attempts += 1;
            let mut nearest_example: Texture = knowledge[0].clone();;
            let mut min_distance: f32 = (*result)
                .clone()
                .euclidean_distance_with_weights(&nearest_example, _weights);
            for known in knowledge.iter() {
                let distance = (*result).clone().euclidean_distance(known);
                if distance < min_distance {
                    min_distance = distance;
                    nearest_example = known.clone();
                }
            }

            if nearest_example._class == (*result)._class {
                _correct += 1;
            }
        }
        println!(
            "Total aciertos: {}/{} = {}",
            _correct,
            _attempts,
            _correct as f32 / _attempts as f32
        );
    }

    Ok(())
}

fn main() {
    if let Err(err) = test() {
        println!("error running text: {}", err);
        std::process::exit(1);
    }
}
