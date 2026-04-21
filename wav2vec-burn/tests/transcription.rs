use anyhow::Context as _;
use burn::prelude::*;
use burn::tensor::TensorData;
use wav2vec_burn::config::Wav2Vec2Base;
use wav2vec_burn::{CTCDecoder, Model};
use wav2vec_burn_test::model::load_model;
use wav2vec_burn_test::{TestBackend, TestDevice, init_logger};

#[test]
fn test_transcribe_silence() -> anyhow::Result<()> {
    init_logger();

    let device = TestDevice::default();
    let model: Model<Wav2Vec2Base<TestBackend>> = load_model(&device).context("loading model")?;

    let data = TensorData::zeros::<f32, _>([1, 1, 1_600]);
    let input = Tensor::from_data(data, &device);
    let logits = model.forward(input);
    let text = CTCDecoder::decode_logits(logits, 5).unwrap();

    assert_eq!(text, "");

    log::info!("Transcription of silence: {text:?}");
    Ok(())
}
