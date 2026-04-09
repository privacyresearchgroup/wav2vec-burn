//! Types used to specify the configuration of a `wav2vec 2.0` [`Model`](crate::Model).

use std::convert::Infallible;
use std::fmt::Debug;
use std::marker::PhantomData;

use burn::prelude::*;

use crate::{feature_encoder, model, transformer};

/// Compile-time configuration for [`Model`](crate::Model).
pub trait ConstConfig: Clone + Copy + Debug {
    /// The `burn` `Backend` to use.
    type Backend: Backend;

    /// The normalization mode used in the feature encoder.
    type FeatureEncoderNormalizationMode: feature_encoder::Normalize<Self::Backend>;
    /// The normalization mode used in the feature encoder.
    type FeatureProjectionMode: model::ProjectFeatures<Self::Backend>;

    /// Configuration of the convolutional layers in the feature encoder.
    const FEATURE_ENCODER_LAYERS: &[feature_encoder::LayerConfig] = feature_encoder::DEFAULT_LAYERS;
    /// Whether to include bias in the convolutional feature encoder.
    const FEATURE_ENCODER_BIAS: bool = false;
    /// Length of encoded features vector.
    const FEATURES_LEN: usize = Self::FEATURE_ENCODER_LAYERS[Self::FEATURE_ENCODER_LAYERS.len() - 1].output_len;

    /// Length of convolutional layer for positional embedding.
    const POS_EMBEDDING_LEN: usize = 768;
    /// Size of kernel in the convolutional layer for positional embedding.
    const POS_EMBEDDING_KERNEL: usize = 128;
    /// Number of groups in the convolutional layer for positional embedding.
    const POS_EMBEDDING_GROUPS: usize = 16;

    /// Length of the feed forward network layers.
    const FEED_FORWARD_LEN: usize = 3072;

    /// Number of transformer layers.
    const TRANSFORMER_LAYERS: usize = 12;
    /// Number of attention heads.
    const ATTENTION_HEADS: usize = 12;
    /// Mode of normalization in the attention layers in the transformer.
    const ATTENTION_NORMALIZER_MODE: transformer::AttentionNormalizerMode = transformer::AttentionNormalizerMode::After;

    /// Length of final representation.
    const OUT_LEN: usize = 768;
}

/// Compile-time configuration for [`Model`](crate::Model), for the `wav2vec2-base` model.
#[derive(Clone, Debug)]
pub enum Wav2Vec2Base<B: Backend> {
    #[doc(hidden)]
    _Phantom(Infallible, PhantomData<B>),
}

impl<B: Backend> Copy for Wav2Vec2Base<B> {}

impl<B: Backend> ConstConfig for Wav2Vec2Base<B> {
    type Backend = B;

    type FeatureEncoderNormalizationMode = feature_encoder::GroupNormalization<B>;
    type FeatureProjectionMode = model::FeatureProjection<B>;

    const OUT_LEN: usize = 32;
}

/// Compile-time configuration for [`Model`](crate::Model), for the `wav2vec2-large` model.
#[derive(Clone, Debug)]
pub enum Wav2Vec2Large<B: Backend> {
    #[doc(hidden)]
    _Phantom(Infallible, PhantomData<B>),
}

impl<B: Backend> Copy for Wav2Vec2Large<B> {}

impl<B: Backend> ConstConfig for Wav2Vec2Large<B> {
    type Backend = B;

    type FeatureEncoderNormalizationMode = feature_encoder::LayerNormalization<B>;
    type FeatureProjectionMode = model::FeatureProjection<B>;

    const FEATURE_ENCODER_BIAS: bool = true;

    const POS_EMBEDDING_LEN: usize = 1024;

    const FEED_FORWARD_LEN: usize = 4096;

    const TRANSFORMER_LAYERS: usize = 24;
    const ATTENTION_HEADS: usize = 16;
    const ATTENTION_NORMALIZER_MODE: transformer::AttentionNormalizerMode = transformer::AttentionNormalizerMode::Before;

    const OUT_LEN: usize = 32;
}
