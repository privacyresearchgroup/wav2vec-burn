use std::fmt::Debug;

use burn::module::ModuleDisplay;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::config::ConstConfig;
use crate::error::CreateError;
use crate::feature_encoder::FeatureEncoder;
use crate::transformer::Transformer;
use crate::weights::Weights;

/// Implementation of `wav2vec 2.0` for the `burn` ML Framework.
///
/// The variant of `wav2vec 2.0` model is selected at compile-time using the generic parameter `C`. Currently supported variants are:
///
/// * [`wav2vec2-base`](crate::config::Wav2Vec2Base)
/// * [`wav2vec2-large`](crate::config::Wav2Vec2Large)
#[derive(Clone, Debug, Module)]
pub struct Model<C: ConstConfig> {
    feature_encoding: FeatureEncoder<C>,
    feature_normalization: LayerNorm<C::Backend>,
    feature_projection: C::FeatureProjectionMode,
    transformation: Transformer<C>,
    language_modeling_projection: Linear<C::Backend>,
}

pub trait ProjectFeatures<B: Backend>: Clone + Debug + ModuleDisplay + Send {
    fn new(weights: &Weights, prefix: &str, input_len: usize, output_len: usize, device: &B::Device) -> Result<Self, CreateError>;
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>;
}

#[derive(Debug, Module)]
pub struct FeatureProjection<B: Backend> {
    projection: Linear<B>,
}

impl<C: ConstConfig> Model<C> {
    /// Creates a new `Model`.
    ///
    /// The given `weights` should correspond to the `wav2vec 2.0` model type described by the generic parameter `C`.
    ///
    /// # Errors
    ///
    /// If an error occurs accessing model values from `tensors`, then an error is returned.
    pub fn new(weights: &Weights, device: &<C::Backend as Backend>::Device) -> Result<Self, CreateError> {
        Ok(Self {
            feature_encoding: FeatureEncoder::new(weights, "wav2vec2.feature_extractor.conv_layers", device)?,
            feature_normalization: weights.load_layer_norm(
                "wav2vec2.feature_projection.layer_norm",
                &LayerNormConfig::new(C::FEATURES_LEN),
                device,
            )?,
            feature_projection: C::FeatureProjectionMode::new(
                weights,
                "wav2vec2.feature_projection.projection",
                C::FEATURES_LEN,
                C::POS_EMBEDDING_LEN,
                device,
            )?,
            transformation: Transformer::new(weights, "wav2vec2.encoder", device)?,
            language_modeling_projection: weights.load_linear("lm_head", &LinearConfig::new(C::POS_EMBEDDING_LEN, C::OUT_LEN), device)?,
        })
    }

    /// Runs model inference on the given `input`.
    ///
    /// The output returned should be fed into [`CTCDecoder`](crate::CTCDecoder) to produce a transcription.
    pub fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let features = self.feature_encoding.forward(input).swap_dims(1, 2);
        let normalized_features = self.feature_normalization.forward(features);
        let projected_features = self.feature_projection.forward(normalized_features);
        let transformed = self.transformation.forward(projected_features);
        let language_modelled = self.language_modeling_projection.forward(transformed);
        language_modelled
    }
}

impl<B: Backend> ProjectFeatures<B> for FeatureProjection<B> {
    fn new(weights: &Weights, prefix: &str, input_len: usize, output_len: usize, device: &B::Device) -> Result<Self, CreateError> {
        Ok(Self { projection: weights.load_linear(prefix, &LinearConfig::new(input_len, output_len), device)? })
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.projection.forward(input)
    }
}
