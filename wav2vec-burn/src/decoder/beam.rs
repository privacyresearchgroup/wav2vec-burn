use std::cell::Cell;
use std::cmp::Ordering;

use crate::util::LogSpaceF32;

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Beam {
    prefix: String,
    probability_with_blank: LogSpaceF32,
    probability_without_blank: LogSpaceF32,
    total_probability: Cell<Option<LogSpaceF32>>,
}

impl Beam {
    pub fn new(prefix: String) -> Self {
        Self {
            prefix,
            probability_with_blank: LogSpaceF32::default(),
            probability_without_blank: LogSpaceF32::default(),
            total_probability: Cell::new(Some(LogSpaceF32::default())),
        }
    }

    pub fn blank() -> Self {
        Self {
            prefix: String::default(),
            probability_with_blank: LogSpaceF32::ONE,
            probability_without_blank: LogSpaceF32::default(),
            total_probability: Cell::<Option<_>>::default(),
        }
    }

    pub fn add_to_probability_with_blank(&mut self, rhs: LogSpaceF32) {
        self.probability_with_blank += rhs;
        self.total_probability = Cell::<Option<_>>::default();
    }

    pub fn add_to_probability_without_blank(&mut self, rhs: LogSpaceF32) {
        self.probability_without_blank += rhs;
        self.total_probability = Cell::<Option<_>>::default();
    }

    pub fn prefix(&self) -> &str {
        &self.prefix
    }

    pub fn probability_with_blank(&self) -> LogSpaceF32 {
        self.probability_with_blank
    }

    pub fn probability_without_blank(&self) -> LogSpaceF32 {
        self.probability_without_blank
    }

    pub fn total_probability(&self) -> LogSpaceF32 {
        match self.total_probability.get() {
            Some(total_probability) => total_probability,
            None => {
                let total_probability = self.probability_with_blank + self.probability_without_blank;
                self.total_probability.set(Some(total_probability));
                total_probability
            }
        }
    }
}

impl Ord for Beam {
    fn cmp(&self, other: &Self) -> Ordering {
        match self.total_probability().cmp(&other.total_probability()) {
            Ordering::Equal => self.prefix.cmp(&other.prefix),
            ordering => ordering,
        }
    }
}

impl PartialOrd for Beam {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
