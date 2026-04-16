pub mod evaluation;

pub use wav2vec_burn_cli::{audio, loader};

pub type TestBackend = burn::backend::NdArray;
pub type TestDevice = burn::backend::ndarray::NdArrayDevice;

/// Stack size for tests.
///
/// The CubeCL CPU backend uses deeply recursive MLIR/LLVM passes during kernel compilation.
pub const TEST_STACK_SIZE: usize = 512 * 1024 * 1024;

pub fn run_test<F, T>(test_fun: F) -> T
where
    F: FnOnce() -> T,
    F: Send + 'static,
    T: Send + 'static,
{
    std::thread::Builder::new()
        .stack_size(TEST_STACK_SIZE)
        .spawn(test_fun)
        .expect("error spawning test thread with large stack")
        .join()
        .unwrap_or_else(|err| std::panic::resume_unwind(err))
}
