mod beam;
mod beams;

use std::char;

use burn::prelude::*;
use burn::tensor::DataError;
use burn::tensor::activation::log_softmax;
use itertools::Itertools;

use crate::util::LogSpaceF32;

use self::beam::Beam;
use self::beams::Beams;

/// A CTC beam-search decoder for the `wav2vec 2.0` alphabet.
///
/// CTC is a technique used in transformer models for automatic speech recognition. A CTC decoder takes the output of such a model which was
/// fine-tuned using a CTC as input, and returns a string in the alphabet specified by the model. Beam search is an implementation of CTC
/// decoding.
///
/// This CTC decoder currently only works with the `wav2vec 2.0` model's alphabet.
pub struct CTCDecoder {
    beam_width: usize,
    beams: Beams,
    new_beams: Beams,
}

const BLANK_IDX: usize = 0;
const ALPHABET: [char; 32] = [
    // pad, begin sentence, end sentence, unknown, word separator
    '_', '{', '}', '?', ' ', 'e', 't', 'a', 'o', 'n', 'i', 'h', 's', 'r', 'd', 'l', 'u', 'm', 'w', 'c', 'f', 'g', 'y', 'p', 'b', 'v', 'k',
    '\'', 'x', 'j', 'q', 'z',
];

impl CTCDecoder {
    /// Constructs a new `CTCDecoder` with given `beam_width`.
    ///
    /// The larger the `beam_width`, the more accurate, but more computationally expensive.
    #[must_use]
    pub fn new(beam_width: usize) -> Self {
        let mut beams = Beams::default();
        beams.add(Beam::blank());
        Self { beam_width, beams, new_beams: Beams::default() }
    }

    /// Processes many timesteps of `logits` at once.
    ///
    /// The input `logits` can be, for example, the output of a `wav2vec 2.0` model. This method is more efficient than
    /// [`process_timestep`](Self::process_timestep), if multiple timesteps are available at once.
    ///
    /// # Errors
    ///
    /// If the wrong type of values is supplied as input, then an error is returned.
    pub fn process_logits<B: Backend>(&mut self, logits: Tensor<B, 3>) -> Result<(), DataError> {
        let log_probs = log_softmax(logits, 2).squeeze::<2>().to_data();
        let mut log_probs_iter = log_probs.as_slice()?.iter().copied();
        while let Some(logits) = log_probs_iter.next_array() {
            self.process_timestep(logits);
        }
        Ok(())
    }

    /// Processes one timestep of log probabilities.
    ///
    /// The input `log_probs` can be, for example, the softmax of the out of a `wav2vec 2.0` model. Use
    /// [`process_logits`](Self::process_logits) instead if multiple timesteps of logits are available at once.
    pub fn process_timestep(&mut self, log_probs: [f32; ALPHABET.len()]) {
        let symbol_probabilities = log_probs.map(LogSpaceF32::from_log);

        for beam in self.beams.iter() {
            for (symbol_idx, (&symbol, &symbol_probability)) in ALPHABET.iter().zip(&symbol_probabilities).enumerate() {
                if symbol_idx == BLANK_IDX {
                    self.new_beams
                        .add_to_probability_with_blank(beam.prefix(), beam.total_probability() * symbol_probability);
                } else if beam.prefix().ends_with(symbol) {
                    // allow repeated character with a blank in-between
                    self.new_beams.add_to_probability_without_blank(
                        &format!("{}{symbol}", beam.prefix()),
                        beam.probability_with_blank() * symbol_probability,
                    );
                    // collapse repeated characters with no blank in-between
                    self.new_beams
                        .add_to_probability_without_blank(beam.prefix(), beam.probability_without_blank() * symbol_probability);
                } else {
                    self.new_beams.add_to_probability_without_blank(
                        &format!("{}{symbol}", beam.prefix()),
                        beam.total_probability() * symbol_probability,
                    );
                }
            }
        }
        self.beams.extend_pruned(&mut self.new_beams, self.beam_width);
    }

    /// Returns the decoded most probable string up to the current timestep.
    #[must_use]
    pub fn decode(&self) -> &str {
        self.beams.max().unwrap_or_else(|| unreachable!()).prefix()
    }

    #[doc(hidden)]
    pub fn decode_logits<B: Backend>(logits: Tensor<B, 3>, beam_width: usize) -> Result<String, DataError> {
        let mut decoder = CTCDecoder::new(beam_width);
        decoder.process_logits(logits)?;
        let decoded = decoder.decode();
        let cleaned_up = decoded.replace(['{', '}', '?'], "").trim_matches(' ').to_string();
        Ok(cleaned_up)
    }

    #[doc(hidden)]
    pub fn decode_log_probs(timestep_log_probs: impl IntoIterator<Item = impl IntoIterator<Item = f32>>, beam_width: usize) -> String {
        let mut decoder = CTCDecoder::new(beam_width);
        for logits in timestep_log_probs {
            let logits = logits
                .into_iter()
                .collect_array()
                .unwrap_or_else(|| panic!("invalid number of logits per timestep"));
            decoder.process_timestep(logits);
        }
        decoder.decode().replace(['{', '}', '?'], "").trim_matches(' ').to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_blank_only() {
        let timestep_log_probs = (0..5).map(|_| {
            let mut logits = [f32::NEG_INFINITY; ALPHABET.len()];
            logits[BLANK_IDX] = 0.0; // log(1)
            logits
        });
        assert_eq!(CTCDecoder::decode_log_probs(timestep_log_probs, 5), "");
    }

    #[test]
    fn test_single_char() {
        let symbol_idx = ALPHABET.iter().position(|ch| *ch == 'e').unwrap();
        let timestep_log_probs = (0..5).map(|_| {
            let mut logits = vec![f32::NEG_INFINITY; ALPHABET.len()];
            logits[symbol_idx] = 0.0;
            logits
        });
        assert_eq!(CTCDecoder::decode_log_probs(timestep_log_probs, 5), "e");
    }

    #[test]
    fn test_hi() {
        let timestep_log_probs = Itertools::intersperse("hi".chars(), '_').map(|symbol| {
            let mut logits = [f32::NEG_INFINITY; ALPHABET.len()];
            logits[ALPHABET.iter().position(|ch| *ch == symbol).unwrap()] = 0.0;
            logits
        });
        assert_eq!(CTCDecoder::decode_log_probs(timestep_log_probs, 10), "hi");
    }
}
