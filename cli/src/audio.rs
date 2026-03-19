use std::path::Path;

use anyhow::Context as _;
use hound::{SampleFormat, WavReader, WavSpec, WavWriter};
use rubato::{Resampler, SincFixedIn, SincInterpolationParameters, SincInterpolationType, WindowFunction};

pub const TARGET_SAMPLE_RATE: u32 = 16_000;

/// Loads a FLAC or WAV file, downmixes to mono, resamples to 16 kHz, and returns normalised f32 samples.
pub fn load_audio(path: &Path) -> anyhow::Result<Vec<f32>> {
    log::info!("Opening audio file: {}", path.display());

    let extension = path.extension().ok_or(anyhow::anyhow!("no filename extension"))?;
    let extension = extension.to_str().context("invalid filename extension")?;
    match &extension.to_ascii_lowercase()[..] {
        "flac" => load_flac(path),
        "wav" => load_wav(path),
        ext => anyhow::bail!("unrecognized filename extension .{ext}"),
    }
}

/// Loads a WAV file, downmixes to mono, resamples to 16 kHz, and returns normalised f32 samples.
pub fn load_wav(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = WavReader::open(path)?;
    let spec = reader.spec();

    let interleaved: Vec<f32> = match spec.sample_format {
        SampleFormat::Float => reader.samples().collect::<hound::Result<_>>()?,
        #[expect(clippy::cast_precision_loss, reason = "Lossy conversion desired")]
        SampleFormat::Int => {
            let max = (1_i64 << (spec.bits_per_sample - 1)) as f32;
            let interleaved = reader
                .samples::<i32>()
                .map(|sample| sample.map(|value| value as f32 / max))
                .collect::<hound::Result<_>>()?;
            interleaved
        }
    };

    let mono_samples = downmix(interleaved, spec.channels);
    resample(mono_samples, spec.sample_rate, TARGET_SAMPLE_RATE)
}

/// Loads a FLAC file, downmixes to mono, resamples to 16 kHz, and returns normalised f32 samples.
pub fn load_flac(path: &Path) -> anyhow::Result<Vec<f32>> {
    let mut reader = claxon::FlacReader::open(path)?;
    let info = reader.streaminfo();
    #[expect(clippy::cast_precision_loss, reason = "Lossy conversion desired")]
    let interleaved: Vec<f32> = {
        let max = (1_i64 << (info.bits_per_sample - 1)) as f32;
        reader
            .samples()
            .map(|sample| sample.map(|value| value as f32 / max))
            .collect::<claxon::Result<_>>()?
    };

    #[expect(clippy::cast_possible_truncation, reason = "Channel count should fit in u16")]
    let mono_samples = downmix(interleaved, info.channels as u16);
    resample(mono_samples, info.sample_rate, TARGET_SAMPLE_RATE)
}

/// Writes a silent mono S16 WAV file with given `sample_rate` and `duration_secs`.
pub fn write_silent_wav(path: &Path, duration_secs: f32, sample_rate: u32) -> anyhow::Result<()> {
    let spec = WavSpec { channels: 1, sample_rate, bits_per_sample: 16, sample_format: SampleFormat::Int };
    #[expect(clippy::cast_sign_loss, reason = "Wont be negative")]
    #[expect(clippy::cast_precision_loss, reason = "Sample rate should fit in f32")]
    #[expect(clippy::cast_possible_truncation, reason = "Result should fit in u64")]
    let samples_len = (sample_rate as f32 * duration_secs) as u64;
    let mut writer = WavWriter::create(path, spec).expect("creating silent wav");
    for _ in 0..samples_len {
        writer.write_sample(0i16).context("writing samples to silent wav")?;
    }
    writer.finalize().context("finalizing silent wav")?;
    Ok(())
}

/// Reads the duration of a FLAC file, in seconds.
pub fn flac_duration_secs(path: &Path) -> anyhow::Result<f32> {
    let reader = claxon::FlacReader::open(path).context("opening audio file")?;
    let info = reader.streaminfo();
    let frames = info.samples.unwrap_or(0);
    #[expect(clippy::cast_precision_loss, reason = "Duration can be imprecise")]
    Ok(frames as f32 / info.sample_rate as f32)
}

/// Downmixes multi-channel f32 audio with given number of `channels` to mono.
#[must_use]
pub fn downmix(interleaved: Vec<f32>, channels: u16) -> Vec<f32> {
    if channels == 1 {
        return interleaved;
    }
    let frame_count = interleaved.len() / usize::from(channels);
    (0..frame_count)
        .map(|frame_idx| {
            let sum: f32 = (0..usize::from(channels))
                .map(|channel_idx| interleaved[frame_idx * usize::from(channels) + channel_idx])
                .sum();
            sum / f32::from(channels)
        })
        .collect()
}

/// Resamples mono f32 audio with given `sample_rate` to `target_sample_rate`.
pub fn resample(samples: Vec<f32>, sample_rate: u32, target_sample_rate: u32) -> anyhow::Result<Vec<f32>> {
    if sample_rate == target_sample_rate {
        return Ok(samples);
    }

    log::info!("Resampling {sample_rate} to {target_sample_rate}");
    let ratio = f64::from(target_sample_rate) / f64::from(sample_rate);
    let params = SincInterpolationParameters {
        sinc_len: 256,
        f_cutoff: 0.95,
        interpolation: SincInterpolationType::Linear,
        oversampling_factor: 256,
        window: WindowFunction::BlackmanHarris2,
    };

    let chunk_size = samples.len();
    let mut resampler = SincFixedIn::<f32>::new(ratio, 2.0, params, chunk_size, 1)?;
    let mut resampled = resampler.process(&[samples], None)?;

    let output = resampled.remove(0);
    Ok(output)
}
