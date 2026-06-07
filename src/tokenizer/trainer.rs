use std::collections::{HashMap, HashSet};

use crate::{NixiaError, Result};

use super::{normalizer::normalize_text, special, vocab::Vocabulary};

#[derive(Clone, Debug)]
pub struct BpeTrainerConfig {
    pub vocab_size: usize,
    pub min_pair_frequency: usize,
}

impl Default for BpeTrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 8_000,
            min_pair_frequency: 2,
        }
    }
}

#[derive(Clone, Debug)]
struct WordEntry {
    pieces: Vec<std::rc::Rc<str>>,
    count: usize,
}

pub fn train_vocab(corpus: &str, config: BpeTrainerConfig) -> Result<Vocabulary> {
    let normalized = normalize_text(corpus);
    if normalized.is_empty() {
        return Err(NixiaError::EmptyCorpus);
    }

    let mut word_counts = HashMap::<String, usize>::new();
    for word in normalized.split_whitespace() {
        if special::is_reserved(word) {
            continue;
        }
        *word_counts.entry(word.to_string()).or_default() += 1;
    }

    let mut words = word_counts
        .into_iter()
        .map(|(word, count)| WordEntry {
            pieces: word_to_initial_pieces(&word),
            count,
        })
        .collect::<Vec<_>>();

    let mut vocab = special::default_special_tokens()
        .into_iter()
        .map(ToOwned::to_owned)
        .collect::<Vec<_>>();
    let mut known = vocab.iter().cloned().collect::<HashSet<_>>();

    add_current_pieces(&words, &mut vocab, &mut known, config.vocab_size);

    while vocab.len() < config.vocab_size {
        let Some((left, right, frequency)) = most_frequent_pair(&words) else {
            break;
        };

        if frequency < config.min_pair_frequency {
            break;
        }

        let merged = format!("{left}{right}");
        if known.insert(merged.clone()) {
            vocab.push(merged.clone());
        }

        merge_pair_in_words(&mut words, &left, &right, &merged);
        add_current_pieces(&words, &mut vocab, &mut known, config.vocab_size);
    }

    sort_tail_by_frequency(&mut vocab, &words);
    vocab.truncate(config.vocab_size);
    Vocabulary::new(vocab)
}

fn word_to_initial_pieces(word: &str) -> Vec<std::rc::Rc<str>> {
    let with_marker = format!("{}{}", special::SPACE_MARKER, word);
    with_marker
        .chars()
        .map(|ch| std::rc::Rc::from(ch.to_string().as_str()))
        .collect()
}

fn add_current_pieces(
    words: &[WordEntry],
    vocab: &mut Vec<String>,
    known: &mut HashSet<String>,
    vocab_size: usize,
) {
    let mut piece_frequency = HashMap::<&str, usize>::new();
    for word in words {
        for piece in &word.pieces {
            *piece_frequency.entry(&**piece).or_default() += word.count;
        }
    }

    let mut pieces = piece_frequency.into_iter().collect::<Vec<_>>();
    pieces.sort_by(|left, right| right.1.cmp(&left.1).then_with(|| left.0.cmp(right.0)));

    for (piece, _) in pieces {
        if vocab.len() >= vocab_size {
            break;
        }
        // We use contains + insert to avoid allocating `String` for pieces that are already known.
        // `HashSet::get_or_insert_owned` from hashbrown would be ideal here, but standard `HashSet`
        // doesn't have it. `contains` followed by `insert` is optimal without hashbrown raw entries.
        if !known.contains(piece) {
            let s = piece.to_string();
            known.insert(s.clone());
            vocab.push(s);
        }
    }
}

fn most_frequent_pair(words: &[WordEntry]) -> Option<(String, String, usize)> {
    let mut pair_counts = HashMap::<(&str, &str), usize>::new();

    for word in words {
        for pair in word.pieces.windows(2) {
            let key = (&*pair[0], &*pair[1]);
            *pair_counts.entry(key).or_default() += word.count;
        }
    }

    pair_counts
        .into_iter()
        .max_by(|left, right| {
            left.1
                .cmp(&right.1)
                .then_with(|| right.0.0.cmp(left.0.0))
                .then_with(|| right.0.1.cmp(left.0.1))
        })
        .map(|((left, right), count)| (left.to_string(), right.to_string(), count))
}

fn merge_pair_in_words(words: &mut [WordEntry], left: &str, right: &str, merged: &str) {
    let merged_rc: std::rc::Rc<str> = std::rc::Rc::from(merged);
    for word in words {
        let mut next = Vec::with_capacity(word.pieces.len());
        let mut index = 0usize;

        while index < word.pieces.len() {
            if index + 1 < word.pieces.len()
                && &*word.pieces[index] == left
                && &*word.pieces[index + 1] == right
            {
                next.push(std::rc::Rc::clone(&merged_rc));
                index += 2;
            } else {
                next.push(word.pieces[index].clone());
                index += 1;
            }
        }

        word.pieces = next;
    }
}

fn sort_tail_by_frequency(vocab: &mut [String], words: &[WordEntry]) {
    let special_len = special::default_special_tokens().len();
    if vocab.len() <= special_len {
        return;
    }

    let mut frequency = HashMap::<&str, usize>::new();
    for word in words {
        for piece in &word.pieces {
            *frequency.entry(&**piece).or_default() += word.count;
        }
    }

    vocab[special_len..].sort_by(|left, right| {
        frequency
            .get(right.as_str())
            .copied()
            .unwrap_or_default()
            .cmp(&frequency.get(left.as_str()).copied().unwrap_or_default())
            .then_with(|| left.cmp(right))
    });
}

#[cfg(test)]
mod tests {
    use super::{BpeTrainerConfig, special, train_vocab};

    #[test]
    fn trains_small_vocab() {
        let vocab = train_vocab(
            "aku lagi makan nih aku lagi santai",
            BpeTrainerConfig {
                vocab_size: 32,
                min_pair_frequency: 1,
            },
        )
        .unwrap();

        assert!(vocab.len() <= 32);
        assert_eq!(vocab.token(0), Some("<pad>"));
    }

    #[test]
    fn does_not_learn_special_token_fragments() {
        let vocab = train_vocab(
            "<user> aku capek <char> aku dengerin <user> makasih <char> sama-sama",
            BpeTrainerConfig {
                vocab_size: 64,
                min_pair_frequency: 1,
            },
        )
        .unwrap();

        assert_eq!(vocab.id(special::USER), Some(4));
        assert_eq!(vocab.id(special::CHARACTER), Some(5));
        assert!(
            vocab
                .id(&format!("{}<user>", special::SPACE_MARKER))
                .is_none()
        );
        assert!(vocab.id("user>").is_none());
        assert!(vocab.id("char>").is_none());
    }

    #[test]
    fn deterministic_ordering_with_ties() {
        let vocab1 = train_vocab(
            "a b c d e f a b c d e f", // all pairs occur same number of times
            BpeTrainerConfig {
                vocab_size: 20,
                min_pair_frequency: 1,
            },
        )
        .unwrap();

        let vocab2 = train_vocab(
            "a b c d e f a b c d e f",
            BpeTrainerConfig {
                vocab_size: 20,
                min_pair_frequency: 1,
            },
        )
        .unwrap();

        assert_eq!(vocab1.tokens(), vocab2.tokens());
    }

    #[test]
    fn most_frequent_pair_tie_breaking() {
        let vocab = train_vocab(
            "a b c d a b c d",
            BpeTrainerConfig {
                vocab_size: 11,
                min_pair_frequency: 1,
            },
        )
        .unwrap();

        assert!(!vocab.is_empty());
    }

    #[test]
    fn add_current_pieces_no_duplicates() {
        let vocab = train_vocab(
            "hello hello world world",
            BpeTrainerConfig {
                vocab_size: 20,
                min_pair_frequency: 1,
            },
        )
        .unwrap();

        let mut tokens = vocab.tokens().to_vec();
        tokens.sort();
        let len_before = tokens.len();
        tokens.dedup();
        let len_after = tokens.len();
        assert_eq!(
            len_before, len_after,
            "Vocabulary should not contain duplicate tokens"
        );
    }
}
