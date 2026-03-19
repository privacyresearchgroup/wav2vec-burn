pub mod evaluation;

pub use wav2vec_burn_cli::{audio, loader};

pub type TestBackend = burn::backend::NdArray;
pub type TestDevice = burn::backend::ndarray::NdArrayDevice;
