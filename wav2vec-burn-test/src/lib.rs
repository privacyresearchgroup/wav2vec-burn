//! Common `wav2vec-burn` test code.

pub mod audio;
pub mod evaluation;
pub mod model;
pub mod test_data;

pub type TestBackend = burn::backend::NdArray;
pub type TestDevice = burn::backend::ndarray::NdArrayDevice;

/// Initializes the the `log` crate logger for use with tests.
pub fn init_logger() {
    // Ignore errors initializing the logger if tests race to configure it
    let _ignore = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .is_test(true)
        .try_init();
}
