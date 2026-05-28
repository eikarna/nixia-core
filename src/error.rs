use std::{fmt, io};

pub type Result<T> = std::result::Result<T, NixiaError>;

#[derive(Debug)]
pub enum NixiaError {
    EmptyCorpus,
    EmptyVocabulary,
    InvalidArgument(String),
    InvalidVocabulary(String),
    Io(io::Error),
    Recorder(String),
}

impl fmt::Display for NixiaError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyCorpus => write!(formatter, "corpus is empty after normalization"),
            Self::EmptyVocabulary => write!(formatter, "vocabulary is empty"),
            Self::InvalidArgument(message) => write!(formatter, "invalid argument: {message}"),
            Self::InvalidVocabulary(message) => write!(formatter, "invalid vocabulary: {message}"),
            Self::Io(error) => write!(formatter, "io error: {error}"),
            Self::Recorder(message) => write!(formatter, "recorder error: {message}"),
        }
    }
}

impl std::error::Error for NixiaError {}

impl From<io::Error> for NixiaError {
    fn from(error: io::Error) -> Self {
        Self::Io(error)
    }
}
