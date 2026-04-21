//! Convenience for loading a `wav2vec 2.0` model.

use std::fs;
use std::path::Path;

use burn::prelude::*;
use wav2vec_burn::config::{ConstConfig, Wav2Vec2Base, Wav2Vec2Large};
use wav2vec_burn::{Model, Weights};

/// Trait for a model that has an associated name and filename.
pub trait PathConfig {
    /// The variant name of a `wav2vec 2.0` model.
    const MODEL_NAME: &str;

    /// The filename for a variant of the `wav2vec 2.0` model.
    const MODEL_FILENAME: &str;
}

impl<B: Backend> PathConfig for Wav2Vec2Base<B> {
    const MODEL_NAME: &str = "wav2vec2-base-960h";
    const MODEL_FILENAME: &str = "wav2vec2-base-960h.safetensors";
}

impl<B: Backend> PathConfig for Wav2Vec2Large<B> {
    const MODEL_NAME: &str = "wav2vec2-large-960h";
    const MODEL_FILENAME: &str = "wav2vec2-large-960h.safetensors";
}

/// Loads a `wav2vec 2.0` model described by `C` from `dir` .
pub fn load_model<C: ConstConfig + PathConfig>(dir: &Path, device: &<C::Backend as Backend>::Device) -> anyhow::Result<Model<C>> {
    let path = dir.join(C::MODEL_FILENAME);

    log::info!("Parsing safetensors for {}...", C::MODEL_FILENAME);
    let bytes = fs::read(&path)?;
    let weights = Weights::from_safetensors(bytes.into())?;

    log::info!("Loading model...");
    let model = Model::new(&weights, device)?;

    Ok(model)
}
