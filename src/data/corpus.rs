use std::{fs, path::Path};

use crate::{NixiaError, Result, tokenizer::TinyTokenizer};

#[derive(Clone, Debug)]
pub struct TokenizedCorpus {
    pub train_ids: Vec<usize>,
    pub valid_ids: Vec<usize>,
}

pub fn read_text(path: impl AsRef<Path>) -> Result<String> {
    let contents = fs::read_to_string(path)?;
    if contents.trim().is_empty() {
        return Err(NixiaError::EmptyCorpus);
    }
    Ok(contents)
}

pub fn tokenize_corpus(
    tokenizer: &TinyTokenizer,
    train_text: &str,
    valid_text: Option<&str>,
    valid_ratio: f32,
) -> Result<TokenizedCorpus> {
    let mut train_ids = tokenizer.encode(train_text, true);
    if train_ids.is_empty() {
        return Err(NixiaError::EmptyCorpus);
    }

    let valid_ids = match valid_text {
        Some(text) => tokenizer.encode(text, true),
        None => split_validation(&mut train_ids, valid_ratio),
    };

    Ok(TokenizedCorpus {
        train_ids,
        valid_ids,
    })
}

fn split_validation(train_ids: &mut Vec<usize>, valid_ratio: f32) -> Vec<usize> {
    if train_ids.len() < 4 {
        return train_ids.clone();
    }

    let ratio = valid_ratio.clamp(0.01, 0.5);
    let valid_len = ((train_ids.len() as f32) * ratio).ceil() as usize;
    let split_at = train_ids.len().saturating_sub(valid_len.max(1));
    train_ids.split_off(split_at)
}
