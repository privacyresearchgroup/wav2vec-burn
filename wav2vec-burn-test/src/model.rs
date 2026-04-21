//! Load a `wav2vec 2.0` model for tests.

use burn::prelude::*;
use wav2vec_burn::Model;
use wav2vec_burn::config::ConstConfig;
use wav2vec_burn_loader::model::PathConfig;

use crate::test_data::test_data_dir;

/// Loads a `wav2vec 2.0` model described by `C` from the `./test-data` directory.
pub fn load_model<C: ConstConfig + PathConfig>(device: &<C::Backend as Backend>::Device) -> anyhow::Result<Model<C>> {
    wav2vec_burn_loader::model::load_model(&test_data_dir(), device)
}
