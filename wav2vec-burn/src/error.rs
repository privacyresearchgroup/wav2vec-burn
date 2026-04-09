//! Error types returned by the public API.

use safetensors::SafeTensorError;

/// Error type returned by [`Model::new`](crate::Model::new).
#[derive(Debug, thiserror::Error)]
pub enum CreateError {
    /// An error occurred while loading a tensor from the model's safetensors file.
    #[error(transparent)]
    Tensor(#[from] SafeTensorError),
}
