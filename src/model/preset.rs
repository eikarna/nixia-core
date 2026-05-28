use super::TinyLmConfig;

pub const DEV_SMOKE: &str = "dev-smoke";
pub const REDMI_NANO: &str = "redmi-nano";
pub const REDMI_TINY: &str = "redmi-tiny";

pub fn preset(name: &str, vocab_size: usize, pad_token_id: usize) -> Option<TinyLmConfig> {
    match name {
        DEV_SMOKE => Some(TinyLmConfig {
            vocab_size,
            max_seq_len: 8,
            d_model: 16,
            n_layers: 1,
            n_heads: 4,
            d_ff: 32,
            dropout: 0.1,
            pad_token_id,
        }),
        REDMI_NANO => Some(TinyLmConfig {
            vocab_size,
            max_seq_len: 96,
            d_model: 192,
            n_layers: 6,
            n_heads: 4,
            d_ff: 512,
            dropout: 0.1,
            pad_token_id,
        }),
        REDMI_TINY => Some(TinyLmConfig {
            vocab_size,
            max_seq_len: 128,
            d_model: 256,
            n_layers: 8,
            n_heads: 4,
            d_ff: 768,
            dropout: 0.1,
            pad_token_id,
        }),
        _ => None,
    }
}

pub fn names() -> &'static [&'static str] {
    &[DEV_SMOKE, REDMI_NANO, REDMI_TINY]
}
