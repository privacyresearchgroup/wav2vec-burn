use std::fs;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use burn::prelude::Backend;
use safetensors::SafeTensors;
use wav2vec_burn::Model;
use wav2vec_burn::config::{ConstConfig, Wav2Vec2Base, Wav2Vec2Large};

pub trait PathConfig {
    const MODEL_URL: &str;
    const MODEL_NAME: &str;
    const MODEL_FILENAME: &str;
}

impl<B: Backend> PathConfig for Wav2Vec2Base<B> {
    const MODEL_URL: &str = "https://huggingface.co/facebook/wav2vec2-base-960h/resolve/main/model.safetensors";
    const MODEL_NAME: &str = "wav2vec2-base-960h";
    const MODEL_FILENAME: &str = "wav2vec2-base-960h.safetensors";
}

impl<B: Backend> PathConfig for Wav2Vec2Large<B> {
    const MODEL_URL: &str = "https://huggingface.co/facebook/wav2vec2-large-960h/resolve/main/model.safetensors";
    const MODEL_NAME: &str = "wav2vec2-large-960h";
    const MODEL_FILENAME: &str = "wav2vec2-large-960h.safetensors";
}

/// Returns the default cache directory (`$CACHE_DIR/wav2vec-burn`).
#[must_use]
pub fn default_cache_dir() -> PathBuf {
    dirs::cache_dir().unwrap_or_else(|| PathBuf::from(".")).join("wav2vec-burn")
}

/// Loads a wav2vec model downloading its safetensors from huggingface to `cache_dir` if necessary.
pub fn load_model<C: ConstConfig + PathConfig>(
    cache_dir: &Path,
    device: &<C::Backend as burn::prelude::Backend>::Device,
) -> anyhow::Result<Model<C>> {
    let cached_path = cache_dir.join(C::MODEL_FILENAME);
    ensure_downloaded(C::MODEL_URL, &cached_path)?;

    log::info!("Parsing safetensors for {}...", C::MODEL_FILENAME);
    let bytes = fs::read(&cached_path)?;
    let tensors = SafeTensors::deserialize(&bytes)?;
    log::info!("{} tensors loaded", tensors.names().len());

    log::info!("Loading model...");
    let model = Model::new(&tensors, device)?;

    Ok(model)
}

/// Download `url` to `dir` atomically, if it does not exist.
pub fn ensure_downloaded(url: &str, dir: &Path) -> anyhow::Result<()> {
    if dir.exists() {
        log::info!("Already downloaded: {}", dir.display());
        return Ok(());
    }

    log::info!("Downloading from {url} to {}...", dir.display());

    if let Some(parent) = dir.parent() {
        fs::create_dir_all(parent)?;
    }

    let mut response = reqwest::blocking::get(url)?;
    anyhow::ensure!(response.status().is_success(), "HTTP {}", response.status());

    let total = response.content_length();
    #[expect(clippy::cast_precision_loss, reason = "Log can be imprecise")]
    if let Some(total) = total {
        log::info!("Download size: {:.1} MB", total as f64 / 1_048_576.0);
    }

    let tmp_path = dir.with_extension("tmp");
    let mut file = fs::File::create(&tmp_path)?;

    let mut downloaded_len: u64 = 0;
    let mut last_logged_mb: u64 = 0;
    let mut buf = vec![0u8; 256 * 1024];

    loop {
        let n = response.read(&mut buf)?;
        if n == 0 {
            break;
        }
        file.write_all(&buf[..n])?;
        downloaded_len += n as u64;

        let downloaded_len_mb = downloaded_len / 10_485_760;
        if downloaded_len_mb > last_logged_mb {
            last_logged_mb = downloaded_len_mb;
            #[expect(clippy::cast_precision_loss, reason = "Log can be imprecise")]
            {
                let downloaded_len_mb = downloaded_len as f64 / 1_048_576.0;
                if let Some(total) = total {
                    let total_mb = total as f64 / 1_048_576.0;
                    let download_percent = downloaded_len as f64 / total as f64 * 100.0;
                    log::info!("  {downloaded_len_mb:.1} / {total_mb:.1} MB ({download_percent:.0}%)");
                } else {
                    log::info!("  {downloaded_len_mb:.1} MB downloaded");
                }
            }
        }
    }

    drop(file);
    fs::rename(&tmp_path, dir)?;

    #[expect(clippy::cast_precision_loss, reason = "Log can be imprecise")]
    let downloaded_len_mb = downloaded_len as f64 / 1_048_576.0;
    log::info!("Download complete: {downloaded_len_mb:.1} MB",);

    Ok(())
}
