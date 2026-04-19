//! Locating and loading Librispeech test data for `wav2vec-burn`.

use std::collections::BTreeMap;
use std::fs;
use std::path::Path;
use std::path::PathBuf;

use anyhow::Context as _;
use wav2vec_burn_cli::audio;

use crate::test_data::test_data_dir;

/// Name of directory which should contain Librispeech test data.
pub const TEST_DATA_DIR_NAME: &str = "librispeech";

/// Holds Librispeech test data utterances for a particular speaker and chapter.
pub struct TestData {
    pub utterances: BTreeMap<String, Utterance>,
}

/// Holds data for a Librispeech test data utterance.
pub struct Utterance {
    pub path: PathBuf,
    pub duration: f32,
    pub text: String,
}

impl TestData {
    /// Returns Librispeech test data for given `speaker` and `chapter`.
    pub fn load(speaker: &str, chapter: &str) -> anyhow::Result<Self> {
        let dir = test_data_dir().join(TEST_DATA_DIR_NAME).join(speaker).join(chapter);
        let transcription_path = dir.join(format!("{speaker}-{chapter}.trans.txt"));
        let raw_transcriptions = fs::read_to_string(&transcription_path).context("reading transcript file")?;

        let transcription_lines = raw_transcriptions.lines().filter(|line| !line.trim().is_empty());
        let transcriptions = transcription_lines.map(|line| Utterance::from_line(line, &dir));
        let utterances = transcriptions.collect::<anyhow::Result<_>>()?;

        Ok(Self { utterances })
    }
}

impl Utterance {
    fn from_line(line: &str, dir: &Path) -> anyhow::Result<(String, Self)> {
        let (id, text) = line.split_once(' ').expect("malformed transcript line");
        let path = dir.join(format!("{id}.flac"));
        let duration = audio::flac_duration_secs(&path).context("probing duration of utterance")?;
        anyhow::ensure!(path.exists(), "utterance {} exists", path.display());
        Ok((id.to_string(), Self { path, duration, text: text.to_string() }))
    }
}
