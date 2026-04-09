#![warn(missing_docs, clippy::missing_errors_doc, clippy::missing_panics_doc)]

//! `wav2vec-burn` is an implementation of Meta's [Wav2Vec 2.0](https://arxiv.org/abs/2006.11477) speech transcription using the [Burn ML
//! Framework](https://github.com/tracel-ai/burn).

mod decoder;
mod feature_encoder;
mod model;
mod transformer;
mod util;

mod safetensors;

pub mod config;
pub mod error;

pub use self::decoder::CTCDecoder;
pub use self::model::Model;
