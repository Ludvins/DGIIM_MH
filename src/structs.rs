use std::cmp::Ordering;
use std::fmt;

#[derive(Clone)]
pub struct Chromosome {
    pub weights: Vec<f32>,
    pub result: Results,
}

impl Chromosome {
    pub fn new(weights: &Vec<f32>, res: Results) -> Chromosome {
        Chromosome {
            weights: weights.clone(),
            result: res,
        }
    }

    pub fn ev(&self) -> f32 {
        return self.result.evaluation_function();
    }
}
impl PartialEq for Chromosome {
    fn eq(&self, other: &Chromosome) -> bool {
        self.result.evaluation_function() == other.result.evaluation_function()
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
        let mine = self.result.evaluation_function();
        let him = other.result.evaluation_function();
        if mine < him {
            return Ordering::Less;
        }
        if mine > him {
            return Ordering::Greater;
        }
        return Ordering::Equal;
    }
}

/// Trait for CSV data
///
/// **Note**: The struct implementing this trait must also implement `Copy` and `Clone`.
pub trait Data<T: Copy + Clone> {
    fn new() -> T;
    fn get_id(&self) -> i32;
    fn get_attr(&self, index: usize) -> f32;
    fn get_class(&self) -> i32;
    fn set_id(&mut self, index: i32) -> ();
    fn set_attr(&mut self, index: usize, value: f32) -> ();
    fn set_class(&mut self, value: i32) -> ();
    fn euclidean_distance(&self, other: &T) -> f32;
}

#[derive(Clone)]
pub struct Results {
    pub low_weights: f32,
    pub correct_answers: f32,
    pub exam_len: f32,
    pub n_attrs: f32,
}

impl Results {
    pub fn new(weights: &Vec<f32>, correct_answers: u32, exam_len: usize) -> Results {
        let mut low = 0;
        for attr in 0..weights.len() {
            if weights[attr] < 0.2 {
                low += 1;
            }
        }

        Results {
            low_weights: low as f32,
            correct_answers: correct_answers as f32,
            exam_len: exam_len as f32,
            n_attrs: weights.len() as f32,
        }
    }
    pub fn reduction_rate(&self) -> f32 {
        if self.n_attrs == 0.0 {
            return 0.0;
        }
        return self.low_weights / self.n_attrs;
    }
    pub fn success_percentage(&self) -> f32 {
        return self.correct_answers / self.exam_len;
    }
    pub fn evaluation_function(&self) -> f32 {
        return 0.5 * self.reduction_rate() + 0.5 * self.success_percentage();
    }
}

impl fmt::Display for Results {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f, "\t\t\tReduction rate: {} \n\t\t\tSuccess percentage: {}/{} = {}\n\t\t\tEvaluation function: {}",
           self.reduction_rate(),
           self.correct_answers,
           self.exam_len,
           self.success_percentage(),
           self.evaluation_function(),
        )
    }
}

#[derive(Copy, Clone)]
pub struct Texture {
    id: i32,
    attrs: [f32; 40],
    class: i32,
}

impl Data<Texture> for Texture {
    fn new() -> Texture {
        Texture {
            id: -1,
            attrs: [0.0; 40],
            class: -1,
        }
    }
    fn get_id(&self) -> i32 {
        return self.id;
    }
    fn get_attr(&self, index: usize) -> f32 {
        return self.attrs[index];
    }
    fn get_class(&self) -> i32 {
        return self.class;
    }
    fn set_id(&mut self, value: i32) -> () {
        self.id = value;
    }
    fn set_attr(&mut self, index: usize, value: f32) -> () {
        self.attrs[index] = value;;
    }
    fn set_class(&mut self, value: i32) -> () {
        self.class = value;
    }
    fn euclidean_distance(&self, other: &Texture) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum +=
                (self.attrs[index] - other.attrs[index]) * (self.attrs[index] - other.attrs[index])
        }
        return sum.sqrt();
    }
}

#[derive(Copy, Clone)]
pub struct Colposcopy {
    id: i32,
    attrs: [f32; 62],
    class: i32,
}

impl Data<Colposcopy> for Colposcopy {
    fn new() -> Colposcopy {
        Colposcopy {
            id: -1,
            attrs: [0.0; 62],
            class: -1,
        }
    }
    fn get_id(&self) -> i32 {
        return self.id;
    }
    fn get_attr(&self, index: usize) -> f32 {
        return self.attrs[index];
    }
    fn get_class(&self) -> i32 {
        return self.class;
    }
    fn set_id(&mut self, value: i32) -> () {
        self.id = value;
    }
    fn set_attr(&mut self, index: usize, value: f32) -> () {
        self.attrs[index] = value;
    }
    fn set_class(&mut self, value: i32) -> () {
        self.class = value;
    }
    fn euclidean_distance(&self, other: &Colposcopy) -> f32 {
        let mut sum = 0.0;
        for index in 0..62 {
            sum +=
                (self.attrs[index] - other.attrs[index]) * (self.attrs[index] - other.attrs[index])
        }
        return sum.sqrt();
    }
}

#[derive(Copy, Clone)]
pub struct Ionosphere {
    id: i32,
    attrs: [f32; 34],
    class: i32,
}

impl Data<Ionosphere> for Ionosphere {
    fn new() -> Ionosphere {
        Ionosphere {
            id: -1,
            attrs: [0.0; 34],
            class: -1,
        }
    }
    fn get_id(&self) -> i32 {
        return self.id;
    }
    fn get_attr(&self, index: usize) -> f32 {
        return self.attrs[index];
    }
    fn get_class(&self) -> i32 {
        return self.class;
    }
    fn set_id(&mut self, value: i32) -> () {
        self.id = value;
    }
    fn set_attr(&mut self, index: usize, value: f32) -> () {
        self.attrs[index] = value;;
    }
    fn set_class(&mut self, value: i32) -> () {
        self.class = value;
    }
    fn euclidean_distance(&self, other: &Ionosphere) -> f32 {
        let mut sum = 0.0;
        for index in 0..34 {
            sum +=
                (self.attrs[index] - other.attrs[index]) * (self.attrs[index] - other.attrs[index])
        }
        return sum.sqrt();
    }
}
