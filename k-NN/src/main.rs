extern crate csv;
extern crate generic_array;
extern crate rand;
extern crate rustc_serialize;
extern crate serde_derive;

use std::collections::HashMap;

pub trait Data<T> {
    fn new() -> T;
    fn get_class(self) -> i32;
    fn get_id(self) -> i32;
    fn get_attr(self, index: usize) -> f32;
    fn euclidean_distance(self, other: &T) -> f32;
}

#[derive(Copy, Clone)]
pub struct Texture {
    pub _id: i32,
    pub _attrs: [f32; 40],
    pub _class: i32,
}
impl Texture {}

impl Data<Texture> for Texture {
    fn get_class(self) -> i32 {
        return self._class;
    }
    fn get_id(self) -> i32 {
        return self._id;
    }
    fn get_attr(self, index: usize) -> f32 {
        return self._attrs[index];
    }
    fn new() -> Texture {
        Texture {
            _id: -1,
            _attrs: [0.0; 40],
            _class: -1,
        }
    }
    fn euclidean_distance(self, other: &Texture) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum += (self._attrs[index] - other._attrs[index])
                * (self._attrs[index] - other._attrs[index])
        }
        return sum.sqrt();
    }
}

fn make_partitions<T: Data<T> + Clone + Copy>(data: Vec<T>, folds: usize) -> Vec<Vec<T>> {
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

pub fn _1_nn<T: Data<T> + Clone + Copy>(
    data: &Vec<T>,
    folds: usize,
) -> Result<(), Box<std::error::Error>> {
    let data: Vec<Vec<T>> = make_partitions(data.clone(), folds);
    // NOTE For each partition
    for i in 0..folds {
        // NOTE Test over partition i
        let mut _correct = 0;
        let mut _attempts = 0;
        let mut knowledge: Vec<T> = Vec::new();

        // Learning from all other partitions.
        for j in 0..folds {
            if j != i {
                knowledge.extend(data[j].iter().cloned());
            }
        }

        // NOTE Test
        for result in data[i].iter() {
            _attempts += 1;

            let mut nearest_example: T = T::new();
            let mut min_distance: f32 = std::f32::MAX;

            for known in knowledge.iter() {
                let distance = result.euclidean_distance(known);

                if distance < min_distance {
                    min_distance = distance;
                    nearest_example = known.clone();
                }
            }

            if nearest_example.get_class() == (*result).get_class() {
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

    Ok(())
}

pub fn relief<T: Data<T> + Clone + Copy>(
    data: &Vec<T>,
    folds: usize,
    atributes: usize,
) -> Result<(), Box<std::error::Error>> {
    let mut _weights: [f32; 40] = [0.0; 40];

    let data: Vec<Vec<T>> = make_partitions(data.clone(), folds);
    // NOTE For each partition
    for i in 0..folds {
        // NOTE Test over partition i
        let mut _correct = 0;
        let mut _attempts = 0;
        let mut knowledge: Vec<T> = Vec::new();

        // Learning from all other partitions.
        for j in 0..folds {
            if j != i {
                knowledge.extend(data[j].iter().cloned());
            }
        }

        // NOTE Greedy
        for known in knowledge.iter() {
            let mut enemy: T = T::new();
            let mut ally: T = T::new();
            let mut enemy_distance = std::f32::MAX;
            let mut friend_distance = std::f32::MAX;

            for candidate in knowledge.iter() {
                //NOTE Skip if cantidate == known
                if candidate.get_id() != known.get_id() {
                    // NOTE Pre-calculate distance
                    let dist = known.euclidean_distance(candidate);
                    // NOTE Ally
                    if known.get_class() == candidate.get_class() {
                        if dist < friend_distance {
                            ally = candidate.clone();
                            friend_distance = dist;
                        }
                    }
                    // NOTE Enemy
                    else {
                        if dist < enemy_distance {
                            enemy = candidate.clone();
                            enemy_distance = dist;
                        }
                    }
                }
            }
            // NOTE Re-calculate weights
            let mut highest_weight: f32 = _weights[0];
            for attr in 0..atributes {
                _weights[attr] += f32::abs(known.get_attr(attr) - enemy.get_attr(attr))
                    - f32::abs(known.get_attr(attr) - ally.get_attr(attr));

                if _weights[attr] > highest_weight {
                    highest_weight = _weights[attr];
                }
                if _weights[attr] < 0.0 {
                    _weights[attr] = 0.0;
                }
            }

            // NOTE Normalize weights
            for attr in 0..atributes {
                _weights[attr] = _weights[attr] / highest_weight;
            }
        } // NOTE END Greedy

        // NOTE Test
        for result in data[i].iter() {
            _attempts += 1;

            let mut nearest_example: T = T::new();
            let mut min_distance: f32 = std::f32::MAX;

            for known in knowledge.iter() {
                //NOTE Distance with weights
                let mut distance = 0.0;
                for index in 0..atributes {
                    distance += _weights[index]
                        * (result.get_attr(index) - known.get_attr(index))
                        * (result.get_attr(index) - known.get_attr(index))
                }

                distance = distance.sqrt();

                if distance < min_distance {
                    min_distance = distance;
                    nearest_example = known.clone();
                }
            }

            if nearest_example.get_class() == (*result).get_class() {
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

    Ok(())
}

pub fn _1_nn_texture() -> Result<(), Box<std::error::Error>> {
    let _rng = rand::thread_rng();

    let mut csv_reader = csv::Reader::from_path("data/csv_result-texture.csv")?;
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
    println!("Resultados 1-NN:");
    return _1_nn(&data, 5);
}

pub fn relief_texture() -> Result<(), Box<std::error::Error>> {
    let _rng = rand::thread_rng();

    let mut csv_reader = csv::Reader::from_path("data/csv_result-texture.csv")?;
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
    println!("Resultados Relief:");
    return relief(&data, 5, 40);
}

fn main() {
    if let Err(err) = _1_nn_texture() {
        println!("error running 1-nn: {}", err);
        std::process::exit(1);
    }

    if let Err(err) = relief_texture() {
        println!("error running relief: {}", err);
        std::process::exit(1);
    }
}
