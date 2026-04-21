use std::time::Instant;

use anyhow::Context as _;
use burn::prelude::*;
use burn::tensor::TensorData;
use wav2vec_burn::config::Wav2Vec2Base;
use wav2vec_burn::{CTCDecoder, Model};
use wav2vec_burn_test::audio::load_flac;
use wav2vec_burn_test::evaluation::word_error_rate;
use wav2vec_burn_test::model::load_model;
use wav2vec_burn_test::test_data::librispeech::{TestData, Utterance};
use wav2vec_burn_test::{TestBackend, TestDevice, init_logger};

const LIBRISPEECH_SPEAKER: &str = "1089";
const LIBRISPEECH_CHAPTER: &str = "134686";
const MAX_AUDIO_SECS: f32 = 30.0;
const WER_THRESHOLD: f32 = 0.10;

#[test]
fn test_librispeech_wer() -> anyhow::Result<()> {
    init_logger();

    let test_data = TestData::load(LIBRISPEECH_SPEAKER, LIBRISPEECH_CHAPTER)?;
    let mut selected_len_secs = 0.0;
    let selected = test_data.utterances.into_iter().take_while(|(_id, utterance)| {
        let select_utterance = selected_len_secs < MAX_AUDIO_SECS;
        selected_len_secs += utterance.duration;
        select_utterance
    });

    let device = TestDevice::default();
    let model: Model<Wav2Vec2Base<TestBackend>> = load_model(&device).context("loading model")?;

    let mut pairs: Vec<(String, String)> = Vec::new();
    for (utterance_id, Utterance { path: flac_path, text: reference, .. }) in selected {
        let samples = load_flac(&flac_path).context("loading utterance")?;
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
