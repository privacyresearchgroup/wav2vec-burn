//! Inspect and load audio to a format suitable for `wav2vec 2.0` input.

use std::path::Path;

use anyhow::Context as _;

pub use wav2vec_burn_loader::audio::*;

/// Reads the duration of a FLAC file, in seconds.
pub fn flac_duration_secs(path: &Path) -> anyhow::Result<f32> {
    let reader = claxon::FlacReader::open(path).context("opening audio file")?;
    let info = reader.streaminfo();
    let frames = info.samples.unwrap_or(0);
    #[expect(clippy::cast_precision_loss, reason = "Duration can be imprecise")]
    Ok(frames as f32 / info.sample_rate as f32)
}
