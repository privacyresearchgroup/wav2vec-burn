use std::collections::HashMap;
use std::mem;

use burn::module::Param;
use burn::nn::conv::{Conv1d, Conv1dConfig, Conv1dRecord};
use burn::nn::{
    GroupNorm, GroupNormConfig, GroupNormRecord, LayerNorm, LayerNormConfig, LayerNormRecord, Linear, LinearConfig, LinearRecord,
    PaddingConfig1d,
};
use burn::prelude::*;
use burn::tensor;
use bytes::Bytes;
use safetensors::tensor as safetensor;
use safetensors::{SafeTensorError, SafeTensors};

use crate::error::CreateError;

/// The loaded weights of a `wav2vec 2.0` model.
pub enum Weights {
    /// Weights loaded as safetensors.
    SafeTensors(SafeTensorsWeights),
    /// Dummy all-zero weights, for testing.
    None,
}

/// The loaded safetensors weights of a `wav2vec 2.0` model.
pub struct SafeTensorsWeights {
    data: Bytes,
    tensors: HashMap<String, safetensor::TensorInfo>,
}

impl Weights {
    /// Loads `Weights` from safetensors data.
    ///
    /// Only metadata is parsed by this function; weights are parsed lazily while loading the [`Model`](crate::Model).
    ///
    /// # Errors
    ///
    /// If an error occurs while parsing the safetensors metadata, then an error is returned.
    pub fn from_safetensors(data: Bytes) -> Result<Self, SafeTensorError> {
        let (metadata_len, metadata) = SafeTensors::read_metadata(&data)?;
        let data = data.slice(metadata_len + 8..);
        let tensors = metadata.tensors().into_iter().map(|(key, tensor)| (key, tensor.clone())).collect();
        Ok(Self::SafeTensors(SafeTensorsWeights { data, tensors }))
    }

    pub(crate) fn load_conv1d<B: Backend>(
        &self,
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
        let in_ch_per_group = config.channels_in / config.groups;
        let kernel_size = config.kernel_size;
        let weight_shape = [config.channels_out, in_ch_per_group, kernel_size];
        let weight = match self.load_tensor::<f32, _>(&format!("{prefix}.weight"), weight_shape) {
            Ok(weight) => Ok(Some(weight)),
            Err(CreateError::Tensor(SafeTensorError::TensorNotFound(_))) => Ok(None),
            Err(err) => Err(err),
        }?;
        let weight = if let Some(weight) = weight {
            weight
        } else {
            // Write merged weight into a temporary key by inserting via a temp SafeTensors
            // is not straightforward; instead we build the conv manually below using Param.
            // (handled by the load_conv1d_with_groups path with pre-merged data below)
            let weight_v = self.load_tensor::<f32, _>(&format!("{prefix}.weight_v"), [config.channels_in, in_ch_per_group, kernel_size])?;
            let weight_v = weight_v.as_slice::<f32>()?;

            // Detect weight_norm axis from weight_g shape:
            //   dim=0 -> weight_g shape [dim, 1, 1], weight_g.len() == dim  (one scale per out-channel)
            //   dim=2 -> weight_g shape [1, 1, kernel], weight_g.len() == kernel (one scale per kernel pos)
            // HuggingFace wav2vec2 uses dim=2.
            let merged: Vec<f32> = match self.load_tensor::<f32, _>(&format!("{prefix}.weight_g"), [1, 1, kernel_size]) {
                Ok(weight_g) => {
                    let weight_g = weight_g.as_slice::<f32>()?;
                    // dim=2 weight_norm: normalize each kernel-position slice weight_v[:, :, k]
                    // and scale by weight_g[k].
                    // weight_v is C-contiguous [dim, in_ch_per_group, kernel].
                    // weight_v[i, j, k] is at index i * in_ch_per_group_tmp * kernel + j * kernel + k.
                    let mut norms = vec![0.0f32; kernel_size];
                    for embed_idx in 0..config.channels_in {
                        for channel_idx in 0..in_ch_per_group {
                            for kernel_idx in 0..kernel_size {
                                let value = weight_v[embed_idx * in_ch_per_group * kernel_size + channel_idx * kernel_size + kernel_idx];
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
                            for kernel_idx in 0..kernel_size {
                                let merged_idx = embed_idx * in_ch_per_group * kernel_size + channel_idx * kernel_size + kernel_idx;
                                merged[merged_idx] = weight_g[kernel_idx] * weight_v[merged_idx] / norms[kernel_idx];
                            }
                        }
                    }
                    merged
                }
                Err(CreateError::TensorShape { got, .. }) if got == [config.channels_in, 1, 1] => {
                    // dim=0 weight_norm: one scale per output channel filter.
                    let weight_g = self.load_tensor::<f32, _>(&format!("{prefix}.weight_g"), [config.channels_in, 1, 1])?;
                    weight_v
                        .chunks(in_ch_per_group * kernel_size)
                        .zip(weight_g.iter::<f32>())
                        .flat_map(|(v_filter, weight_g)| {
                            let norm: f32 = v_filter.iter().map(|x| x * x).sum::<f32>().sqrt();
                            v_filter.iter().map(move |&x| x * weight_g / norm)
                        })
                        .collect()
                }
                Err(err) => return Err(err),
            };

            config = config.with_padding(PaddingConfig1d::Explicit(kernel_size / 2));
            TensorData::new(merged, [config.channels_in, in_ch_per_group, kernel_size])
        };
        let record = Conv1dRecord {
            weight: Param::from_data(weight, device),
            bias: config
                .bias
                .then(|| {
                    let bias = self.load_tensor::<f32, _>(&format!("{prefix}.bias"), [config.channels_out])?;
                    Ok::<_, CreateError>(Param::from_data(bias, device))
                })
                .transpose()?,
            ..config.init(device).into_record()
        };
        Ok(config.init(device).load_record(record))
    }

    pub(crate) fn load_layer_norm<B: Backend>(
        &self,
        prefix: &str,
        config: &LayerNormConfig,
        device: &B::Device,
    ) -> Result<LayerNorm<B>, CreateError> {
        let weight = self.load_tensor::<f32, _>(&format!("{prefix}.weight"), [config.d_model])?;
        let bias = self.load_tensor::<f32, _>(&format!("{prefix}.bias"), [config.d_model])?;
        let record = LayerNormRecord {
            gamma: Param::from_data(weight, device),
            beta: Some(Param::from_data(bias, device)),
            ..config.init(device).into_record()
        };
        Ok(config.init(device).load_record(record))
    }

    pub(crate) fn load_group_norm<B: Backend>(
        &self,
        prefix: &str,
        config: &GroupNormConfig,
        device: &B::Device,
    ) -> Result<GroupNorm<B>, CreateError> {
        let weight = self.load_tensor::<f32, _>(&format!("{prefix}.weight"), [config.num_channels])?;
        let bias_value = self.load_tensor::<f32, _>(&format!("{prefix}.bias"), [config.num_channels])?;
        let record = GroupNormRecord {
            gamma: Some(Param::from_data(weight, device)),
            beta: Some(Param::from_data(bias_value, device)),
            ..config.init(device).into_record()
        };
        Ok(config.init(device).load_record(record))
    }

    pub(crate) fn load_linear<B: Backend>(
        &self,
        prefix: &str,
        config: &LinearConfig,
        device: &B::Device,
    ) -> Result<Linear<B>, CreateError> {
        let weights = self.load_tensor::<f32, _>(&format!("{prefix}.weight"), [config.d_output, config.d_input])?;
        let weights = weights.as_slice::<f32>()?;
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
                    let bias = self.load_tensor::<f32, _>(&format!("{prefix}.bias"), [config.d_output])?;
                    Ok::<_, CreateError>(Param::from_data(bias, device))
                })
                .transpose()?,
        };
        Ok(config.init(device).load_record(record))
    }

    fn load_tensor<E: tensor::Element, S: Into<Vec<usize>>>(&self, name: &str, shape: S) -> Result<TensorData, CreateError> {
        match self {
            Self::SafeTensors(weights) => weights.load_tensor::<E, S>(name, shape),
            Self::None => Ok(TensorData::zeros::<E, S>(shape)),
        }
    }
}

impl SafeTensorsWeights {
    fn load_tensor<E: tensor::Element, S: Into<Vec<usize>>>(&self, name: &str, shape: S) -> Result<TensorData, CreateError> {
        let shape: Vec<_> = shape.into();
        let info = match self.tensors.get(name) {
            Some(info) if info.shape == shape => info,
            Some(info) => return Err(CreateError::TensorShape { expected: shape, got: info.shape.clone() }),
            None => return Err(SafeTensorError::TensorNotFound(name.to_string()))?,
        };

        let bytes = self.data.slice(info.data_offsets.0..info.data_offsets.1);
        let need_align = mem::align_of::<E>();
        let bytes = if bytes.as_ptr().align_offset(need_align) == 0 {
            tensor::Bytes::from_shared(bytes, tensor::AllocationProperty::Other)
        } else {
            let mut tensor_bytes = tensor::Bytes::from_bytes_vec(Vec::new());
            tensor_bytes.extend_from_byte_slice_aligned(&bytes, need_align);
            tensor_bytes
        };
        let dtype = match info.dtype {
            safetensor::Dtype::BOOL => tensor::DType::Bool,
            safetensor::Dtype::U8 => tensor::DType::U8,
            safetensor::Dtype::I8 => tensor::DType::I8,
            safetensor::Dtype::I16 => tensor::DType::I16,
            safetensor::Dtype::U16 => tensor::DType::U16,
            safetensor::Dtype::F16 => tensor::DType::F16,
            safetensor::Dtype::BF16 => tensor::DType::BF16,
            safetensor::Dtype::I32 => tensor::DType::I32,
            safetensor::Dtype::U32 => tensor::DType::U32,
            safetensor::Dtype::F32 => tensor::DType::F32,
            safetensor::Dtype::F64 => tensor::DType::F64,
            safetensor::Dtype::I64 => tensor::DType::I64,
            safetensor::Dtype::U64 => tensor::DType::U64,
            dtype => unimplemented!("Tensor datatype {dtype:?} not implemented"),
        };
        Ok(TensorData::from_bytes(bytes, shape, dtype).convert::<E>())
    }
}
