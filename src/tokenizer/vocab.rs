use std::{collections::HashMap, fs, path::Path};

use crate::{NixiaError, Result};

#[derive(Clone, Debug)]
pub struct Vocabulary {
    id_to_token: Vec<String>,
    token_to_id: HashMap<String, usize>,
}

impl Vocabulary {
    pub fn new(tokens: Vec<String>) -> Result<Self> {
        if tokens.is_empty() {
            return Err(NixiaError::EmptyVocabulary);
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        let mut id_to_token = Vec::with_capacity(tokens.len());

        for token in tokens {
            if token.is_empty() {
                return Err(NixiaError::InvalidVocabulary(
                    "empty token is not allowed".to_string(),
                ));
            }

            if token_to_id.contains_key(&token) {
                continue;
            }

            let id = id_to_token.len();
            token_to_id.insert(token.clone(), id);
            id_to_token.push(token);
        }

        Ok(Self {
            id_to_token,
            token_to_id,
        })
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let contents = fs::read_to_string(path)?;
        let tokens = contents
            .lines()
            .map(str::trim)
            .filter(|line| !line.is_empty() && !line.starts_with('#'))
            .map(ToOwned::to_owned)
            .collect();

        Self::new(tokens)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(parent) = path
            .parent()
            .filter(|parent| !parent.as_os_str().is_empty())
        {
            fs::create_dir_all(parent)?;
        }

        let mut contents = self.id_to_token.join("\n");
        contents.push('\n');
        fs::write(path, contents)?;
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.id_to_token.len()
    }

    pub fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    pub fn id(&self, token: &str) -> Option<usize> {
        self.token_to_id.get(token).copied()
    }

    pub fn token(&self, id: usize) -> Option<&str> {
        self.id_to_token.get(id).map(String::as_str)
    }

    pub fn tokens(&self) -> &[String] {
        &self.id_to_token
    }
}
