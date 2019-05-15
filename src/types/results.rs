use std::fmt;
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
