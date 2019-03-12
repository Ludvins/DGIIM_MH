pub trait Data<T> {
    fn new() -> T;
    fn get_class(&self) -> i32;
    fn get_id(&self) -> i32;
    fn get_attr(&self, index: usize) -> f32;
    fn set_id(&mut self, index: i32) -> ();
    fn set_attr(&mut self, index: usize, value: f32) -> ();
    fn set_class(&mut self, value: i32) -> ();
    fn euclidean_distance(&self, other: &T) -> f32;
}

#[derive(Copy, Clone)]
pub struct Texture {
    pub _id: i32,
    pub _attrs: [f32; 40],
    pub _class: i32,
}

impl Data<Texture> for Texture {
    fn get_class(&self) -> i32 {
        return self._class;
    }
    fn get_id(&self) -> i32 {
        return self._id;
    }
    fn get_attr(&self, index: usize) -> f32 {
        return self._attrs[index];
    }

    fn set_class(&mut self, value: i32) -> () {
        self._class = value;
    }
    fn set_id(&mut self, value: i32) -> () {
        self._id = value;;
    }
    fn set_attr(&mut self, index: usize, value: f32) -> () {
        self._attrs[index] = value;;
    }

    fn new() -> Texture {
        Texture {
            _id: -1,
            _attrs: [0.0; 40],
            _class: -1,
        }
    }
    fn euclidean_distance(&self, other: &Texture) -> f32 {
        let mut sum = 0.0;
        for index in 0..40 {
            sum += (self._attrs[index] - other._attrs[index])
                * (self._attrs[index] - other._attrs[index])
        }
        return sum.sqrt();
    }
}
