mod normalizer;
pub mod special;
pub mod trainer;
mod vocab;

use std::path::Path;

use crate::{NixiaError, Result};

pub use normalizer::normalize_text;
pub use trainer::{BpeTrainerConfig, train_vocab};
pub use vocab::Vocabulary;

#[derive(Clone, Debug)]
pub struct TinyTokenizer {
    vocab: Vocabulary,
    max_piece_chars: usize,
    pad_id: usize,
    bos_id: usize,
    eos_id: usize,
    unk_id: usize,
}

impl TinyTokenizer {
    pub fn new(vocab: Vocabulary) -> Result<Self> {
        let pad_id = required_id(&vocab, special::PAD)?;
        let bos_id = required_id(&vocab, special::BOS)?;
        let eos_id = required_id(&vocab, special::EOS)?;
        let unk_id = required_id(&vocab, special::UNK)?;
        let max_piece_chars = vocab
            .tokens()
            .iter()
            .map(|token| token.chars().count())
            .max()
            .ok_or(NixiaError::EmptyVocabulary)?;

        Ok(Self {
            vocab,
            max_piece_chars,
            pad_id,
            bos_id,
            eos_id,
            unk_id,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        Self::new(Vocabulary::load(path)?)
    }

    pub fn save_vocab(&self, path: impl AsRef<Path>) -> Result<()> {
        self.vocab.save(path)
    }

    pub fn encode(&self, text: &str, add_special: bool) -> Vec<usize> {
        let normalized = normalize_text(text);
        let mut ids = Vec::new();

        if add_special {
            ids.push(self.bos_id);
        }

        for word in normalized.split_whitespace() {
            if special::is_reserved(word) {
                if let Some(id) = self.vocab.id(word) {
                    ids.push(id);
                    continue;
                }
            }

            let remaining = format!("{}{}", special::SPACE_MARKER, word);
            if let Some(id) = self.vocab.id(&remaining) {
                ids.push(id);
                continue;
            }

            let mut remaining = remaining;
            while !remaining.is_empty() {
                let (id, consumed) = self.longest_piece(&remaining);
                ids.push(id);
                remaining = remaining.chars().skip(consumed).collect();
            }
        }

        if add_special {
            ids.push(self.eos_id);
        }

        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = String::new();

        for &id in ids {
            let Some(token) = self.vocab.token(id) else {
                continue;
            };

            match token {
                special::PAD | special::BOS | special::EOS => {}
                special::NEWLINE => out.push('\n'),
                token if token.starts_with(special::SPACE_MARKER) => {
                    if !out.is_empty() && !out.ends_with('\n') {
                        out.push(' ');
                    }
                    out.push_str(token.trim_start_matches(special::SPACE_MARKER));
                }
                token => out.push_str(token),
            }
        }

        out
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    pub fn pad_id(&self) -> usize {
        self.pad_id
    }

    pub fn bos_id(&self) -> usize {
        self.bos_id
    }

    pub fn eos_id(&self) -> usize {
        self.eos_id
    }

    fn longest_piece(&self, text: &str) -> (usize, usize) {
        let max_len = text.chars().count().min(self.max_piece_chars);

        for len in (1..=max_len).rev() {
            let piece = text.chars().take(len).collect::<String>();
            if let Some(id) = self.vocab.id(&piece) {
                return (id, len);
            }
        }

        (self.unk_id, 1)
    }
}

fn required_id(vocab: &Vocabulary, token: &str) -> Result<usize> {
    vocab.id(token).ok_or_else(|| {
        NixiaError::InvalidVocabulary(format!("required token {token:?} is missing"))
    })
}

#[cfg(test)]
mod tests {
    use super::{TinyTokenizer, Vocabulary, special};

    #[test]
    fn encode_decode_roundtrip() {
        let vocab = Vocabulary::new(vec![
            special::PAD.into(),
            special::BOS.into(),
            special::EOS.into(),
            special::UNK.into(),
            special::USER.into(),
            special::CHARACTER.into(),
            special::NEWLINE.into(),
            special::URL.into(),
            special::NUM.into(),
            format!("{}aku", special::SPACE_MARKER).into(),
            format!("{}makan", special::SPACE_MARKER).into(),
        ])
        .unwrap();
        let tokenizer = TinyTokenizer::new(vocab).unwrap();
        let ids = tokenizer.encode("Aku makan", true);

        assert_eq!(tokenizer.decode(&ids), "aku makan");
    }

    #[test]
    fn does_not_encode_normal_words_as_bare_suffixes() {
        let vocab = Vocabulary::new(vec![
            special::PAD.into(),
            special::BOS.into(),
            special::EOS.into(),
            special::UNK.into(),
            special::USER.into(),
            special::CHARACTER.into(),
            special::NEWLINE.into(),
            special::URL.into(),
            special::NUM.into(),
            format!("{}iyaa,", special::SPACE_MARKER).into(),
            "aku".into(),
            format!("{}aku", special::SPACE_MARKER).into(),
            "di".into(),
            format!("{}di", special::SPACE_MARKER).into(),
            format!("{}sini", special::SPACE_MARKER).into(),
        ])
        .unwrap();
        let tokenizer = TinyTokenizer::new(vocab).unwrap();
        let ids = tokenizer.encode("iyaa, aku di sini", true);

        assert_eq!(tokenizer.decode(&ids), "iyaa, aku di sini");
    }
}
