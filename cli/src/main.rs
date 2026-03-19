pub mod audio;

use std::io::{Write, stdout};
use std::path::PathBuf;

use burn::backend::NdArray;
use burn::backend::ndarray::NdArrayDevice;
use burn::prelude::*;
use burn::tensor::TensorData;
use clap::Parser;
use log::LevelFilter;
use wav2vec_burn::Model;
use wav2vec_burn::config::{ConstConfig, Wav2Vec2Base, Wav2Vec2Large};
use wav2vec_burn::decoder::CTCDecoder;
use wav2vec_burn_cli::loader;

use crate::audio::TARGET_SAMPLE_RATE;

#[derive(Parser, Debug)]
#[command(name = "transcribe", about = "Transcribe speech to text using wav2vec2")]
struct Args {
    /// Path to the input file.
    audio_file: PathBuf,

    /// Model variant to use: `base` (wav2vec2-base-960h) or `large` (wav2vec2-large-960h).
    #[arg(long, default_value = "base")]
    model: String,

    /// CTC beam search width (higher is more accurate but slower).
    #[arg(long, default_value_t = 50)]
    beam_width: usize,

    /// Directory for cached model weights [default: ~/.cache/wav2vec-burn].
    #[arg(long)]
    cache_dir: Option<PathBuf>,
}

fn main() -> anyhow::Result<()> {
    env_logger::Builder::new()
        .filter_level(LevelFilter::Info)
        .parse_default_env()
        .init();

    let args = Args::parse();

    let samples = audio::load_audio(&args.audio_file)?;
    #[expect(clippy::cast_precision_loss, reason = "Log can be imprecise")]
    let duration = samples.len() as f32 / TARGET_SAMPLE_RATE as f32;
    log::info!("Audio file of duration {duration:.2}s loaded: {}", args.audio_file.display(),);

    let cache_dir = args.cache_dir.unwrap_or_else(loader::default_cache_dir);
    log::info!("Cache dir: {}", cache_dir.display());

    let device = NdArrayDevice::Cpu;
    let text = match args.model.to_lowercase().as_str() {
        "base" => {
            let model: Model<Wav2Vec2Base<NdArray<f32>>> = loader::load_model(&cache_dir, &device)?;
            infer_and_decode(&samples, model, args.beam_width, &device)?
        }
        "large" => {
            let model: Model<Wav2Vec2Large<NdArray<f32>>> = loader::load_model(&cache_dir, &device)?;
            infer_and_decode(&samples, model, args.beam_width, &device)?
        }
        other => anyhow::bail!("Unknown model variant '{other}'; use 'base' or 'large'"),
    };

    log::info!("Finished transcribing");
    write!(&stdout(), "{text}")?;
    Ok(())
}

fn infer_and_decode<C: ConstConfig>(
    samples: &[f32],
    model: Model<C>,
    beam_width: usize,
    device: &<C::Backend as Backend>::Device,
) -> anyhow::Result<String> {
    let data = TensorData::new(samples.to_vec(), [1, 1, samples.len()]);
    let input = Tensor::from_data(data, device);

    log::info!("Running inference...");
    let logits = model.forward(input);

    log::info!("Decoding with beam width {beam_width}...");
    let mut decoder = CTCDecoder::new(beam_width);
    decoder.process_logits(logits)?;
    Ok(decoder.decode().to_string())
}
