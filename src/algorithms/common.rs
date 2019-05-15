use crate::types::data::*;
use crate::types::results::*;
use rand::distributions::Normal;
use rand::prelude::*;
use std::collections::HashMap;
// Normalizes the given vector and sets negatives values to 0.
pub fn normalize_and_truncate_negative_weights(weights: &mut Vec<f32>) {
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
    } else if weights[index_to_mutate] < 0.0 {
        weights[index_to_mutate] = 0.0;
    }
}
