use std::fmt::Debug;
use std::mem::replace;

use burn::module::ModuleDisplay;
use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Gelu, GroupNorm, GroupNormConfig, LayerNorm, LayerNormConfig};
use burn::prelude::*;

use crate::config::ConstConfig;
use crate::error::CreateError;
use crate::weights::Weights;

#[derive(Clone, Debug, Module)]
pub struct FeatureEncoder<C: ConstConfig> {
    first_layer: Layer<C, C::FeatureEncoderNormalizationMode>,
    layers: Vec<Layer<C, <C::FeatureEncoderNormalizationMode as Normalize<C::Backend>>::Rest>>,
}

#[derive(Clone, Debug, Module)]
struct Layer<C: ConstConfig, N: Normalize<C::Backend>> {
    convolution: Conv1d<C::Backend>,
    normalization: N,
    activation: Gelu,
}

#[derive(Clone, Copy, Debug)]
pub struct LayerConfig {
    pub output_len: usize,
    pub kernel_size: usize,
    pub stride: usize,
}

pub trait Normalize<B: Backend>: Clone + Debug + ModuleDisplay + Send {
    type Rest: Normalize<B>;

    fn new(weights: &Weights, prefix: &str, input_len: usize, device: &B::Device) -> Result<Self, CreateError>;
    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3>;
}

#[derive(Debug, Module)]
pub struct LayerNormalization<B: Backend> {
    layer_norm: LayerNorm<B>,
}

#[derive(Debug, Module)]
pub struct GroupNormalization<B: Backend> {
    group_norm: GroupNorm<B>,
}

#[derive(Clone, Debug, Module)]
pub struct NoNormalization;

pub const DEFAULT_LAYERS: &[LayerConfig] = &[
    layer(512, 10, 5),
    layer(512, 3, 2),
    layer(512, 3, 2),
    layer(512, 3, 2),
    layer(512, 3, 2),
    layer(512, 2, 2),
    layer(512, 2, 2),
];

#[must_use]
pub const fn layer(output_len: usize, kernel_size: usize, stride: usize) -> LayerConfig {
    LayerConfig { output_len, kernel_size, stride }
}

impl<C: ConstConfig> FeatureEncoder<C> {
    pub fn new(weights: &Weights, prefix: &str, device: &<C::Backend as Backend>::Device) -> Result<Self, CreateError> {
        let [first_config, configs @ ..] = C::FEATURE_ENCODER_LAYERS else {
            unreachable!("empty feature encoder config")
        };
        let first_layer = Layer::new(weights, &format!("{prefix}.0"), first_config, 1, device)?;
        let mut input_len = first_config.output_len;
        let layers = configs
            .iter()
            .enumerate()
            .map(|(idx, config)| {
                Layer::<C, _>::new(
                    weights,
                    &format!("{prefix}.{}", 1 + idx),
                    config,
                    replace(&mut input_len, config.output_len),
                    device,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;

        Ok(Self { first_layer, layers })
    }

    pub fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let acc = self.first_layer.forward(input);
        self.layers.iter().fold(acc, |acc, layer| layer.forward(acc))
    }
}

impl<C: ConstConfig, N: Normalize<C::Backend>> Layer<C, N> {
    fn new(
        weights: &Weights,
        prefix: &str,
        config: &LayerConfig,
        input_len: usize,
        device: &<C::Backend as Backend>::Device,
    ) -> Result<Self, CreateError> {
        Ok(Self {
            convolution: weights.load_conv1d(
                &format!("{prefix}.conv"),
                &Conv1dConfig::new(input_len, config.output_len, config.kernel_size)
                    .with_stride(config.stride)
                    .with_bias(C::FEATURE_ENCODER_BIAS),
                device,
            )?,
            normalization: N::new(weights, &format!("{prefix}.layer_norm"), config.output_len, device)?,
            activation: Gelu::new(),
        })
    }

    fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let convoluted = self.convolution.forward(input);
        let normalized = self.normalization.forward(convoluted);
        self.activation.forward(normalized)
    }
}

impl<B: Backend> Normalize<B> for LayerNormalization<B> {
    type Rest = Self;

    fn new(weights: &Weights, prefix: &str, input_len: usize, device: &B::Device) -> Result<Self, CreateError> {
        Ok(Self { layer_norm: weights.load_layer_norm(prefix, &LayerNormConfig::new(input_len), device)? })
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        let transposed = input.transpose();
        let normalized = self.layer_norm.forward(transposed);
        let untransposed = normalized.transpose();
        untransposed
    }
}

impl<B: Backend> Normalize<B> for GroupNormalization<B> {
    type Rest = NoNormalization;

    fn new(weights: &Weights, prefix: &str, input_len: usize, device: &B::Device) -> Result<Self, CreateError> {
        Ok(Self { group_norm: weights.load_group_norm(prefix, &GroupNormConfig::new(input_len, input_len), device)? })
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        self.group_norm.forward(input)
    }
}

impl<B: Backend> Normalize<B> for NoNormalization {
    type Rest = Self;

    fn new(_: &Weights, _: &str, _: usize, _: &<B as Backend>::Device) -> Result<Self, CreateError> {
        Ok(Self)
    }

    fn forward(&self, input: Tensor<B, 3>) -> Tensor<B, 3> {
        input
    }
}
