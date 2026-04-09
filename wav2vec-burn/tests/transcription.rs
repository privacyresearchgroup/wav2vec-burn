use anyhow::Context as _;
use burn::prelude::*;
use burn::tensor::TensorData;
use wav2vec_burn::config::Wav2Vec2Base;
use wav2vec_burn::{CTCDecoder, Model};
use wav2vec_burn_test::{TestBackend, TestDevice, audio, loader};

#[test]
#[ignore = "An ndarray backend bug is crashing"]
fn test_transcribe_silence() -> anyhow::Result<()> {
    let dir = tempfile::tempdir().context("tempdir")?;
    let wav_path = dir.path().join("silence.wav");
    audio::write_silent_wav(&wav_path, 0.1, 16_000)?;

    let samples = audio::load_audio(&wav_path).context("loading audio")?;
    let samples_len = samples.len();

    let cache_dir = loader::default_cache_dir();
    let device = TestDevice::default();
    let model: Model<Wav2Vec2Base<TestBackend>> = loader::load_model(&cache_dir, &device).context("loading model")?;

    let data = TensorData::new(samples, [1, 1, samples_len]);
    let input = Tensor::from_data(data, &device);
    let logits = model.forward(input);
    let text = CTCDecoder::decode_logits(logits, 5).unwrap();

    assert_ne!(text.len(), 0);

    println!("Transcription of silence: {text:?}");
    Ok(())
}
