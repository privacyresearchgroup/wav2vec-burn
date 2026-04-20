//! Error types returned by the public API.

use burn::tensor::DataError;
use safetensors::SafeTensorError;

/// Error type returned by [`Model::new`](crate::Model::new).
#[derive(Debug, thiserror::Error)]
pub enum CreateError {
    /// An error occurred while loading a tensor from the model's safetensors file.
    #[error(transparent)]
    Tensor(#[from] SafeTensorError),

    /// An error occurred while loading a tensor from the model's safetensors file.
    #[error(transparent)]
    TensorData(#[from] DataError),

    /// A loaded tensor's shape didn't match what was expected.
    #[error("Unexpected tensor shape: expected {expected:?}, got {got:?}")]
    TensorShape {
        /// The expected tensor shape.
        expected: Vec<usize>,
        /// The encountered tensor shape.
        got: Vec<usize>,
    },
}
