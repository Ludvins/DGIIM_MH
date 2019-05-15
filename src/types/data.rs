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
