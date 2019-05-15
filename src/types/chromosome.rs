use std::cmp::Ordering;
#[derive(Clone)]
pub struct Chromosome {
    pub weights: Vec<f32>,
    pub result: f32,
}

impl Chromosome {
    pub fn new(weights: &Vec<f32>, res: f32) -> Chromosome {
        Chromosome {
            weights: weights.clone(),
            result: res,
        }
    }
    pub fn new_w(weighted: &Vec<f32>) -> Chromosome {
        Chromosome {
            weights: weighted.clone(),
            result: -1.0,
        }
    }
}
impl PartialEq for Chromosome {
    fn eq(&self, other: &Chromosome) -> bool {
        return self.result == other.result;
    }
}
impl Eq for Chromosome {}

impl PartialOrd for Chromosome {
    fn partial_cmp(&self, other: &Chromosome) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Chromosome {
    fn cmp(&self, other: &Chromosome) -> Ordering {
        if self.result < other.result {
            return Ordering::Less;
        }
        if self.result > other.result {
            return Ordering::Greater;
        }
        return Ordering::Equal;
    }
}
