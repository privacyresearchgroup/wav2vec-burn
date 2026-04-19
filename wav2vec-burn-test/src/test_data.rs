//! Locating and loading test data for `wav2vec-burn`.

use std::path::PathBuf;

pub mod librispeech;

/// Returns the directory containing test data (`./test-data`).
#[must_use]
pub fn test_data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap_or_else(|| unreachable!())
        .join("test-data")
}
