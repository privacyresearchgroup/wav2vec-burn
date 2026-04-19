use std::collections::HashMap;
use std::fs;
use std::io::{BufReader, Read};
use std::path::Path;
use std::time::Instant;

use anyhow::Context as _;
use burn::prelude::*;
use burn::tensor::TensorData;
use wav2vec_burn::config::Wav2Vec2Base;
use wav2vec_burn::{CTCDecoder, Model};
use wav2vec_burn_test::evaluation::word_error_rate;
use wav2vec_burn_test::{TestBackend, TestDevice, audio, init_logger, loader};

const LIBRISPEECH_TEST_CLEAN_URL: &str = "https://www.openslr.org/resources/12/test-clean.tar.gz";
const FIRST_SPEAKER: &str = "1089";
const FIRST_CHAPTER: &str = "134686";
const MAX_AUDIO_SECS: f32 = 30.0;
const WER_THRESHOLD: f32 = 0.10;

#[test]
fn test_librispeech_wer() -> anyhow::Result<()> {
    init_logger();
    let cache_dir = loader::default_cache_dir();
    let audio_dir = cache_dir.join("librispeech").join(FIRST_SPEAKER).join(FIRST_CHAPTER);
    let transcription_path = audio_dir.join(format!("{FIRST_SPEAKER}-{FIRST_CHAPTER}.trans.txt"));
    if !transcription_path.exists() {
        log::info!("downloading librispeech test-clean speaker {FIRST_SPEAKER} chapter {FIRST_CHAPTER}...");
        stream_extract_chapter(&cache_dir).context("downloading librispeech")?;
    }

    let raw_transcriptions = fs::read_to_string(&transcription_path).context("reading transcript file")?;

    let transcriptions: HashMap<String, String> = raw_transcriptions
        .lines()
        .filter(|line| !line.trim().is_empty())
        .map(|line| {
            let (id, text) = line.split_once(' ').expect("malformed transcript line");
            (id.to_string(), text.to_string())
        })
        .collect();

    let mut utterance_ids: Vec<String> = transcriptions.keys().cloned().collect();
    utterance_ids.sort();
    let mut selected: Vec<(String, String)> = Vec::new();
    let mut selected_len_secs = 0f32;
    for utterance_id in &utterance_ids {
        let flac_path = audio_dir.join(format!("{utterance_id}.flac"));
        anyhow::ensure!(flac_path.exists(), "utterance {} exists", flac_path.display());
        let flac_duration = audio::flac_duration_secs(&flac_path).context("probing duration of utterance")?;
        if selected_len_secs + flac_duration > MAX_AUDIO_SECS + 10.0 {
            break;
        }
        selected.push((utterance_id.clone(), transcriptions[utterance_id].clone()));
        selected_len_secs += flac_duration;
        if selected_len_secs >= MAX_AUDIO_SECS {
            break;
        }
    }

    let device = TestDevice::default();
    let model: Model<Wav2Vec2Base<TestBackend>> = loader::load_model(&cache_dir, &device).context("loading model")?;

    let mut pairs: Vec<(String, String)> = Vec::new();
    for (utterance_id, reference) in selected {
        let flac_path = audio_dir.join(format!("{utterance_id}.flac"));
        let samples = audio::load_audio(&flac_path).context("loading utterance")?;
        let samples_len = samples.len();

        let start = Instant::now();
        let input = Tensor::from_data(TensorData::new(samples, [1, 1, samples_len]), &device);
        let logits = model.forward(input);
        let transcription = CTCDecoder::decode_logits(logits, 50)?;
        let elapsed = start.elapsed().as_millis() as f32 / 1000.0;
        log::info!("[{utterance_id}] inference complete in {elapsed:0.1}s:");
        log::info!("  reference:     \"{reference}\"");
        log::info!("  transcription: \"{transcription}\"");

        pairs.push((reference, transcription));
    }

    let pairs_len = pairs.len();
    let wer = word_error_rate(pairs);
    assert!(
        wer <= WER_THRESHOLD,
        "WER {:0.1}% higher than {:0.1}% on LibriSpeech test-clean ({} utterances, {:0.1}s)",
        wer * 100.0,
        WER_THRESHOLD * 100.0,
        pairs_len,
        selected_len_secs,
    );
    log::info!(
        "Total WER on {selected_len_secs:0.1}s of LibriSpeech test-clean: {:0.1}%",
        wer * 100.0
    );

    Ok(())
}

fn stream_extract_chapter(cache_dir: &Path) -> anyhow::Result<()> {
    let audio_dir = cache_dir.join("librispeech").join(FIRST_SPEAKER).join(FIRST_CHAPTER);
    fs::create_dir_all(&audio_dir).context("creating librispeech cache directory")?;

    let response = reqwest::blocking::get(LIBRISPEECH_TEST_CLEAN_URL)?;
    anyhow::ensure!(response.status().is_success(), "HTTP {}", response.status());

    let gz = flate2::read::GzDecoder::new(BufReader::new(response));
    let mut archive = tar::Archive::new(gz);

    log::info!("Extracting {LIBRISPEECH_TEST_CLEAN_URL}");

    let chapter_prefix = format!("LibriSpeech/test-clean/{FIRST_SPEAKER}/{FIRST_CHAPTER}/");

    let mut extracted = 0usize;
    for entry in archive.entries()? {
        let mut entry = entry?;
        let raw_path = entry.path()?.display().to_string();

        if let Some(filename) = raw_path.strip_prefix(&chapter_prefix) {
            if !filename.is_empty() {
                let dest = audio_dir.join(&filename);
                let mut data = Vec::new();
                entry.read_to_end(&mut data)?;
                fs::write(&dest, &data)?;
                log::info!("  extracted {filename} ({} bytes)", data.len());
                extracted += 1;
            }
        } else if extracted != 0 {
            break;
        } else {
            log::info!("  skipping {raw_path}");
        }
    }

    log::info!("Extracted {extracted} files to {}", audio_dir.display());
    if extracted == 0 {
        anyhow::bail!("No files extracted for the target chapter");
    }
    Ok(())
}
