use std::cmp::Ordering;
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign};

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LogSpaceF32 {
    log: f32,
}

impl LogSpaceF32 {
    pub const ZERO: Self = Self { log: f32::NEG_INFINITY };
    pub const ONE: Self = Self { log: 0.0 };

    #[must_use]
    pub fn from_log(log: f32) -> Self {
        Self { log }
    }

    #[must_use]
    pub fn is_zero(self) -> bool {
        self == Self::ZERO
    }
}

impl Add for LogSpaceF32 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        if self.is_zero() {
            rhs
        } else if rhs.is_zero() {
            self
        } else if self > rhs {
            Self::from_log(self.log + (rhs.log - self.log).exp().ln_1p())
        } else {
            Self::from_log(rhs.log + (self.log - rhs.log).exp().ln_1p())
        }
    }
}

impl AddAssign for LogSpaceF32 {
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl Default for LogSpaceF32 {
    fn default() -> Self {
        Self::ZERO
    }
}

impl Div for LogSpaceF32 {
    type Output = Self;

    fn div(self, rhs: Self) -> Self::Output {
        #[expect(clippy::suspicious_arithmetic_impl, reason = "Divide is subtract in log-space")]
        Self::from_log(self.log - rhs.log)
    }
}

impl DivAssign for LogSpaceF32 {
    fn div_assign(&mut self, rhs: Self) {
        *self = *self / rhs;
    }
}

impl Mul for LogSpaceF32 {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        #[expect(clippy::suspicious_arithmetic_impl, reason = "Multiply is add in log-space")]
        Self::from_log(self.log + rhs.log)
    }
}

impl MulAssign for LogSpaceF32 {
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl Eq for LogSpaceF32 {}

impl Ord for LogSpaceF32 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.log.total_cmp(&other.log)
    }
}

impl PartialOrd for LogSpaceF32 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use assert_float_eq::{assert_f32_near, assert_float_absolute_eq};

    #[test]
    fn log_add() {
        let point_3 = LogSpaceF32::from_log(f32::ln(0.3));
        let point_7 = LogSpaceF32::from_log(f32::ln(0.7));
        assert_float_absolute_eq!((point_3 + point_7).log, f32::ln(1.0));
    }

    #[test]
    fn log_mul() {
        let half = LogSpaceF32::from_log(f32::ln(0.5));
        assert_f32_near!((half * half).log, f32::ln(0.5 * 0.5));
    }
}
