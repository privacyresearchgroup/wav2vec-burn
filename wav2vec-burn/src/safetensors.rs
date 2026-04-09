use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv1dRecord};
use burn::nn::{
    GroupNorm, GroupNormConfig, GroupNormRecord, LayerNorm, LayerNormConfig, LayerNormRecord, Linear, LinearConfig, LinearRecord,
    PaddingConfig1d,
};
use burn::prelude::*;
use safetensors::{Dtype, SafeTensorError, SafeTensors};

use crate::error::CreateError;

pub fn load_conv1d<B: Backend>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    config: &Conv1dConfig,
    device: &B::Device,
) -> Result<Conv1d<B>, CreateError> {
    // HuggingFace stores the pos_conv with weight_norm.
    // After saving, the weight_norm may be merged into a single `weight` tensor,
    // or stored as `weight_g` (norm) + `weight_v` (direction).
    // If the model uses weight_norm (weight_g + weight_v), merge them first.
    // Most HuggingFace checkpoints store the merged weight, but handle both cases.
    let mut config = config.clone();
    let weight = match read_tensor_f32(tensors, &format!("{prefix}.weight")) {
        Ok(weight) => Ok(Some(weight)),
        Err(CreateError::Tensor(SafeTensorError::TensorNotFound(_))) => Ok(None),
        Err(err) => Err(err),
    }?;
    let weight = if let Some(weight) = weight {
        let weight_shape = [config.channels_out, config.channels_in / config.groups, config.kernel_size];
        let weight = Param::from_data(TensorData::new(weight, weight_shape), device);
        weight
    } else {
        // Write merged weight into a temporary key by inserting via a temp SafeTensors
        // is not straightforward; instead we build the conv manually below using Param.
        // (handled by the load_conv1d_with_groups path with pre-merged data below)
        let weight_g = read_tensor_f32(tensors, &format!("{prefix}.weight_g"))?;
        let weight_v = read_tensor_f32(tensors, &format!("{prefix}.weight_v"))?;

        let in_ch_per_group = config.channels_in / config.groups;
        // Detect weight_norm axis from weight_g shape:
        //   dim=0 -> weight_g shape [dim, 1, 1], weight_g.len() == dim  (one scale per out-channel)
        //   dim=2 -> weight_g shape [1, 1, kernel], weight_g.len() == kernel (one scale per kernel pos)
        // HuggingFace wav2vec2 uses dim=2.
        let kernel = config.kernel_size;
        let merged: Vec<f32> = if weight_g.len() == kernel {
            // dim=2 weight_norm: normalize each kernel-position slice weight_v[:, :, k]
            // and scale by weight_g[k].
            // weight_v is C-contiguous [dim, in_ch_per_group, kernel].
            // weight_v[i, j, k] is at index i * in_ch_per_group_tmp * kernel + j * kernel + k.
            let mut norms = vec![0.0f32; kernel];
            for embed_idx in 0..config.channels_in {
                for channel_idx in 0..in_ch_per_group {
                    for kernel_idx in 0..kernel {
                        let value = weight_v[embed_idx * in_ch_per_group * kernel + channel_idx * kernel + kernel_idx];
                        norms[kernel_idx] += value * value;
                    }
                }
            }
            for norm in &mut norms {
                *norm = norm.sqrt();
            }
            let mut merged = vec![0.0f32; weight_v.len()];
            for embed_idx in 0..config.channels_in {
                for channel_idx in 0..in_ch_per_group {
                    for kernel_idx in 0..kernel {
                        let merged_idx = embed_idx * in_ch_per_group * kernel + channel_idx * kernel + kernel_idx;
                        merged[merged_idx] = weight_g[kernel_idx] * weight_v[merged_idx] / norms[kernel_idx];
                    }
                }
            }
            merged
        } else {
            // dim=0 weight_norm: one scale per output channel filter.
            let filter_size = weight_v.len() / config.channels_in; // in_ch_per_group * kernel
            weight_v
                .chunks(filter_size)
                .zip(weight_g.iter())
                .flat_map(|(v_filter, &weight_g)| {
                    let norm: f32 = v_filter.iter().map(|x| x * x).sum::<f32>().sqrt();
                    v_filter.iter().map(move |&x| x * weight_g / norm)
                })
                .collect()
        };

        config = config.with_padding(PaddingConfig1d::Explicit(kernel / 2));
        let weight = Param::from_data(TensorData::new(merged, [config.channels_in, in_ch_per_group, kernel]), device);
        weight
    };
    let record = Conv1dRecord {
        weight,
        bias: config
            .bias
            .then(|| {
                let bias = read_tensor_f32(tensors, &format!("{prefix}.bias"))?;
                Ok::<_, CreateError>(Param::from_data(TensorData::new(bias, [config.channels_out]), device))
            })
            .transpose()?,
        ..config.init(device).into_record()
    };
    Ok(config.init(device).load_record(record))
}

pub fn load_layer_norm<B: Backend>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    config: &LayerNormConfig,
    device: &B::Device,
) -> Result<LayerNorm<B>, CreateError> {
    let weight = read_tensor_f32(tensors, &format!("{prefix}.weight"))?;
    let bias = read_tensor_f32(tensors, &format!("{prefix}.bias"))?;
    let record = LayerNormRecord {
        gamma: Param::from_data(TensorData::new(weight, [config.d_model]), device),
        beta: Some(Param::from_data(TensorData::new(bias, [config.d_model]), device)),
        ..config.init(device).into_record()
    };
    Ok(config.init(device).load_record(record))
}

pub fn load_group_norm<B: Backend>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    config: &GroupNormConfig,
    device: &B::Device,
) -> Result<GroupNorm<B>, CreateError> {
    let weight = read_tensor_f32(tensors, &format!("{prefix}.weight"))?;
    let bias_value = read_tensor_f32(tensors, &format!("{prefix}.bias"))?;
    let record = GroupNormRecord {
        gamma: Some(Param::from_data(TensorData::new(weight, [config.num_channels]), device)),
        beta: Some(Param::from_data(TensorData::new(bias_value, [config.num_channels]), device)),
        ..config.init(device).into_record()
    };
    Ok(config.init(device).load_record(record))
}

pub fn load_linear<B: Backend>(
    tensors: &SafeTensors<'_>,
    prefix: &str,
    config: &LinearConfig,
    device: &B::Device,
) -> Result<Linear<B>, CreateError> {
    let weights = read_tensor_f32(tensors, &format!("{prefix}.weight"))?;
    // HuggingFace stores Linear weight as [out, in] (PyTorch convention).
    // Burn v0.20 stores it as [in, out] and computes O = I @ W (no transpose).
    let mut transposed_weights = vec![0.0f32; weights.len()];
    for output_idx in 0..config.d_output {
        for input_idx in 0..config.d_input {
            transposed_weights[input_idx * config.d_output + output_idx] = weights[output_idx * config.d_input + input_idx];
        }
    }
    let record = LinearRecord {
        weight: Param::from_data(TensorData::new(transposed_weights, [config.d_input, config.d_output]), device),
        bias: config
            .bias
            .then(|| {
                let bias = read_tensor_f32(tensors, &format!("{prefix}.bias"))?;
                Ok::<_, CreateError>(Param::from_data(TensorData::new(bias, [config.d_output]), device))
            })
            .transpose()?,
    };
    Ok(config.init(device).load_record(record))
}

fn read_tensor_f32(tensors: &SafeTensors<'_>, name: &str) -> Result<Vec<f32>, CreateError> {
    let view = tensors.tensor(name)?;
    let bytes = view.data();
    let converted = match view.dtype() {
        Dtype::F32 => bytes
            .chunks_exact(4)
            .map(|byte| f32::from_le_bytes(byte.try_into().unwrap_or_else(|_| unreachable!())))
            .collect(),
        Dtype::BF16 => bytes
            .chunks_exact(2)
            .map(|byte| {
                let bits = u16::from_le_bytes(byte.try_into().unwrap_or_else(|_| unreachable!()));
                f32::from_bits(u32::from(bits) << 16)
            })
            .collect(),
        dtype => panic!("unsupported dtype {dtype:?} for tensor {name}"),
    };
    Ok(converted)
}
