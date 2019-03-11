extern crate csv;
extern crate rand;
extern crate rustc_serialize;
extern crate serde_derive;

use std::collections::HashMap;

#[derive(Copy, Clone)]
struct Texture {
    _id: i32,
    _attrs: [f32; 40],
    _class: i32,
}
impl Texture {
    pub fn new() -> Texture {
        Texture {
            _id: -1,
            _attrs: [0.0; 40],
            _class: -1,
        }
    }

    pub fn euclidean_distance(self, other: &Texture) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum += (self._attrs[index] - other._attrs[index])
                * (self._attrs[index] - other._attrs[index])
        }
        return sum.sqrt();
    }

    pub fn euclidean_distance_with_weights(self, other: &Texture, w: [f32; 40]) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum += w[index]
                * (self._attrs[index] - other._attrs[index])
                * (self._attrs[index] - other._attrs[index])
        }

        return sum.sqrt();
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

    let folds = 5;

    let mut csv_reader = csv::Reader::from_path("data/csv_result-texture.csv")?;
    println!("Csv file read.");
    let mut data: Vec<Texture> = Vec::new();
    let _atributes: usize = 40;

    // NOTE CSV -> Data.
    for result in csv_reader.records() {
        let mut aux_record = Texture::new();
        let record = result?;
        let mut counter = 0;

        for field in record.iter() {
            // CSV structure: id , ... 40 data ... , class
            if counter == 0 {
                aux_record._id = field.parse::<i32>().unwrap();
            } else if counter != 41 {
                aux_record._attrs[counter - 1] = field.parse::<f32>().unwrap();
            } else {
                aux_record._class = field.parse::<i32>().unwrap();
            }

            counter += 1;
        }

        data.push(aux_record);
    }

    let mut _weights: [f32; 40] = [0.0; 40];

    let data: Vec<Vec<Texture>> = make_partitions(data);

    // NOTE Greedy iterations
    for _ in 0..100 {
        // NOTE For each partition
        for i in 0..folds {
            // NOTE Test over partition i
            let mut _correct = 0;
            let mut _attempts = 0;
            let mut knowledge: Vec<Texture> = Vec::new();

            // Learning from all other partitions.
            for j in 0..folds {
                if j != i {
                    knowledge.extend(data[j].iter().cloned());
                }
            }

            // NOTE Greedy
            for example in knowledge.iter() {
                let mut best_enemy: Texture = Texture::new();
                let mut best_ally: Texture = Texture::new();
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
                }
            } // NOTE END Greedy

            // NOTE Test
            for result in data[i].iter() {
                _attempts += 1;

                let mut nearest_example: Texture = Texture::new();
                let mut min_distance: f32 = std::f32::MAX;

                for known in knowledge.iter().cloned() {
                    let distance = (*result).euclidean_distance_with_weights(&known, _weights);
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
                "Total aciertos en test {}: {}/{} = {}",
                i,
                _correct,
                _attempts,
                _correct as f32 / _attempts as f32
            );
        }
    }

    Ok(())
}

fn main() {
    if let Err(err) = test() {
        println!("error running text: {}", err);
        std::process::exit(1);
    }
}
