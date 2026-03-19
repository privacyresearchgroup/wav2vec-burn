use std::mem::replace;

use itertools::Itertools;

pub fn word_error_rate(pairs: impl IntoIterator<Item = (impl AsRef<str>, impl AsRef<str>)>) -> f32 {
    let mut total_words = 0;
    let mut total_errors = 0;
    for (reference, transcribed) in pairs {
        let reference_words = words(reference.as_ref());
        total_words += reference_words.len();
        total_errors += edit_dist(&reference_words, &words(transcribed.as_ref()));
    }
    if total_words == 0 {
        return 0.0;
    }
    #[expect(clippy::cast_precision_loss, reason = "WER can be imprecise in huge case")]
    {
        total_errors as f32 / total_words as f32
    }
}

fn words(sentence: &str) -> Vec<String> {
    sentence.split_whitespace().map(str::to_ascii_lowercase).collect()
}

fn edit_dist<B>(words_a: impl IntoIterator<Item = impl AsRef<str>>, words_b: impl IntoIterator<IntoIter = B>) -> usize
where
    B: ExactSizeIterator + Clone,
    B::Item: AsRef<str> + Copy,
{
    let words_b = words_b.into_iter();
    let mut distances = (0..=words_b.len()).collect_vec();
    for (word_a_idx, word_a) in words_a.into_iter().enumerate() {
        let mut substitution_cost = replace(&mut distances[0], 1 + word_a_idx);
        for (word_b_idx, word_b) in words_b.clone().enumerate() {
            let deletion_cost = distances[1 + word_b_idx];
            let insertion_cost = distances[word_b_idx];
            let new_deletion_cost = if word_a.as_ref() == word_b.as_ref() {
                substitution_cost
            } else {
                1 + deletion_cost.min(insertion_cost).min(substitution_cost)
            };
            distances[1 + word_b_idx] = new_deletion_cost;
            substitution_cost = deletion_cost;
        }
    }
    *distances.last().unwrap_or_else(|| unreachable!())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical() {
        assert_eq!(word_error_rate([("hello world", "hello world")]), 0.0);
    }

    #[test]
    fn one_substitution() {
        assert!((word_error_rate([("hello world", "hello earth")]) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn case_insensitivity() {
        assert_eq!(word_error_rate([("HELLO WORLD", "hello world")]), 0.0);
    }

    #[test]
    fn multiple_pairs() {
        assert!((word_error_rate([("one two three", "one two three"), ("foo bar", "foo baz")]) - 0.2).abs() < 1e-6);
    }
}
