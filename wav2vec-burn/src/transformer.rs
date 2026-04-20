use burn::nn::conv::{Conv1d, Conv1dConfig};
use burn::nn::{Gelu, LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;

use crate::config::ConstConfig;
use crate::error::CreateError;
use crate::weights::Weights;

#[derive(Clone, Debug, Module)]
pub struct Transformer<C: ConstConfig> {
    positional_embedding: PositionalEmbedding<C>,
    normalization: LayerNorm<C::Backend>,
    layers: Vec<Layer<C>>,
}

#[derive(Clone, Debug, Module)]
struct PositionalEmbedding<C: ConstConfig> {
    convolution: Conv1d<C::Backend>,
    activation: Gelu,
}

#[derive(Clone, Copy, Debug)]
pub enum AttentionNormalizerMode {
    /// Layer norm is performed on input before the attention layer.
    Before,
    /// Layer norm is performed after input after the attention layer and residual is added.
    After,
}

#[derive(Clone, Debug, Module)]
struct Layer<C: ConstConfig> {
    attention: Attention<C>,
    attention_normalization: LayerNorm<C::Backend>,
    feed_forward: FeedForward<C>,
    feed_forward_normalization: LayerNorm<C::Backend>,
}

#[derive(Clone, Debug, Module)]
struct Attention<C: ConstConfig> {
    query_projection: Linear<C::Backend>,
    key_projection: Linear<C::Backend>,
    value_projection: Linear<C::Backend>,
    output_projection: Linear<C::Backend>,
}

#[derive(Clone, Debug, Module)]
struct FeedForward<C: ConstConfig> {
    intermediate_dense: Linear<C::Backend>,
    output_dense: Linear<C::Backend>,
    activation: Gelu,
}

impl<C: ConstConfig> Transformer<C> {
    pub fn new(weights: &Weights, prefix: &str, device: &<C::Backend as Backend>::Device) -> Result<Self, CreateError> {
        Ok(Self {
            positional_embedding: PositionalEmbedding::new(weights, &format!("{prefix}.pos_conv_embed.conv"), device)?,
            normalization: weights.load_layer_norm(&format!("{prefix}.layer_norm"), &LayerNormConfig::new(C::POS_EMBEDDING_LEN), device)?,
            layers: (0..C::TRANSFORMER_LAYERS)
                .map(|layer_idx| Layer::<C>::new(weights, &format!("{prefix}.layers.{layer_idx}"), device))
                .collect::<Result<Vec<_>, _>>()?,
        })
    }

    pub fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let residual = input.clone();
        let positions_embedded = residual + self.positional_embedding.forward(input);

        // NB: This is not a bug; "before" means "after" on the top-level. This
        // flag is used inside individual layers as well.
        let pre_normalized = match C::ATTENTION_NORMALIZER_MODE {
            AttentionNormalizerMode::Before => positions_embedded,
            AttentionNormalizerMode::After => self.normalization.forward(positions_embedded),
        };

        let transformed = self.layers.iter().fold(pre_normalized, |acc, layer| layer.forward(acc));

        // NB: same as above.
        let post_normalized = match C::ATTENTION_NORMALIZER_MODE {
            AttentionNormalizerMode::Before => self.normalization.forward(transformed),
            AttentionNormalizerMode::After => transformed,
        };

        post_normalized
    }
}

impl<C: ConstConfig> PositionalEmbedding<C> {
    pub fn new(weights: &Weights, prefix: &str, device: &<C::Backend as Backend>::Device) -> Result<Self, CreateError> {
        let config =
            Conv1dConfig::new(C::POS_EMBEDDING_LEN, C::POS_EMBEDDING_LEN, C::POS_EMBEDDING_KERNEL).with_groups(C::POS_EMBEDDING_GROUPS);
        Ok(PositionalEmbedding { convolution: weights.load_conv1d(prefix, &config, device)?, activation: Gelu::new() })
    }

    fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let transposed = input.swap_dims(1, 2);
        let input_len = transposed.dims()[2];
        let convoluted = self.convolution.forward(transposed);
        let unpadded = convoluted.slice(s![.., .., ..input_len]);
        let activated = self.activation.forward(unpadded);
        let untransposed = activated.swap_dims(1, 2);
        untransposed
    }
}

impl<C: ConstConfig> Layer<C> {
    fn new(weights: &Weights, prefix: &str, device: &<C::Backend as Backend>::Device) -> Result<Layer<C>, CreateError> {
        Ok(Layer {
            attention: Attention::new(weights, &format!("{prefix}.attention"), device)?,
            attention_normalization: weights.load_layer_norm(
                &format!("{prefix}.layer_norm"),
                &LayerNormConfig::new(C::POS_EMBEDDING_LEN),
                device,
            )?,
            feed_forward: FeedForward::new(weights, &format!("{prefix}.feed_forward"), device)?,
            feed_forward_normalization: weights.load_layer_norm(
                &format!("{prefix}.final_layer_norm"),
                &LayerNormConfig::new(C::POS_EMBEDDING_LEN),
                device,
            )?,
        })
    }

    fn forward(&self, hidden_states: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let residual = hidden_states.clone();

        let hidden_states = match C::ATTENTION_NORMALIZER_MODE {
            AttentionNormalizerMode::Before => self.attention_normalization.forward(hidden_states),
            AttentionNormalizerMode::After => hidden_states,
        };

        let attention = residual + self.attention.forward(hidden_states);

        let (normalized_attention, residual);
        match C::ATTENTION_NORMALIZER_MODE {
            AttentionNormalizerMode::Before => {
                residual = attention.clone();
                normalized_attention = self.feed_forward_normalization.forward(attention);
            }
            AttentionNormalizerMode::After => {
                normalized_attention = self.attention_normalization.forward(attention);
                residual = normalized_attention.clone();
            }
        }

        let fed_forward = residual + self.feed_forward.forward(normalized_attention);

        match C::ATTENTION_NORMALIZER_MODE {
            AttentionNormalizerMode::Before => fed_forward,
            AttentionNormalizerMode::After => self.feed_forward_normalization.forward(fed_forward),
        }
    }
}

impl<C: ConstConfig> Attention<C> {
    fn new(weights: &Weights, prefix: &str, device: &<C::Backend as Backend>::Device) -> Result<Attention<C>, CreateError> {
        let config = LinearConfig::new(C::POS_EMBEDDING_LEN, C::POS_EMBEDDING_LEN);
        Ok(Attention {
            query_projection: weights.load_linear(&format!("{prefix}.q_proj"), &config, device)?,
            key_projection: weights.load_linear(&format!("{prefix}.k_proj"), &config, device)?,
            value_projection: weights.load_linear(&format!("{prefix}.v_proj"), &config, device)?,
            output_projection: weights.load_linear(&format!("{prefix}.out_proj"), &config, device)?,
        })
    }

    fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let input_dims = input.dims();
        let [batch_len, sequence_len, ..] = input_dims;
        #[expect(clippy::cast_possible_truncation, reason = "Head len is const, should fit in u16")]
        let head_len = (C::POS_EMBEDDING_LEN / C::ATTENTION_HEADS) as u16;

        let reshape_to_heads = |input: Tensor<C::Backend, 3>| -> Tensor<C::Backend, 4> {
            input
                .reshape([batch_len, sequence_len, C::ATTENTION_HEADS, usize::from(head_len)])
                .swap_dims(1, 2)
        };

        let query = reshape_to_heads(self.query_projection.forward(input.clone()));
        let key = reshape_to_heads(self.key_projection.forward(input.clone()));
        let value = reshape_to_heads(self.value_projection.forward(input));
        let weights = softmax(query.matmul(key.transpose()).div_scalar(f32::from(head_len).sqrt()), 3);
        let context = weights.matmul(value).swap_dims(1, 2).reshape(input_dims);

        self.output_projection.forward(context)
    }
}

impl<C: ConstConfig> FeedForward<C> {
    fn new(weights: &Weights, prefix: &str, device: &<C::Backend as Backend>::Device) -> Result<FeedForward<C>, CreateError> {
        Ok(Self {
            intermediate_dense: weights.load_linear(
                &format!("{prefix}.intermediate_dense"),
                &LinearConfig::new(C::POS_EMBEDDING_LEN, C::FEED_FORWARD_LEN),
                device,
            )?,
            output_dense: weights.load_linear(
                &format!("{prefix}.output_dense"),
                &LinearConfig::new(C::FEED_FORWARD_LEN, C::POS_EMBEDDING_LEN),
                device,
            )?,
            activation: Gelu::new(),
        })
    }

    fn forward(&self, input: Tensor<C::Backend, 3>) -> Tensor<C::Backend, 3> {
        let intermediate = self.intermediate_dense.forward(input);
        let activated = self.activation.forward(intermediate);
        self.output_dense.forward(activated)
    }
}
