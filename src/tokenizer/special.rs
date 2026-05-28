pub const PAD: &str = "<pad>";
pub const BOS: &str = "<bos>";
pub const EOS: &str = "<eos>";
pub const UNK: &str = "<unk>";
pub const USER: &str = "<user>";
pub const CHARACTER: &str = "<char>";
pub const NEWLINE: &str = "<nl>";
pub const URL: &str = "<url>";
pub const NUM: &str = "<num>";

pub const SPACE_MARKER: char = '▁';

pub fn default_special_tokens() -> [&'static str; 9] {
    [PAD, BOS, EOS, UNK, USER, CHARACTER, NEWLINE, URL, NUM]
}

pub fn is_reserved(token: &str) -> bool {
    matches!(
        token,
        PAD | BOS | EOS | UNK | USER | CHARACTER | NEWLINE | URL | NUM
    )
}
