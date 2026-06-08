use super::TinyLmConfig;

pub const DEV_SMOKE: &str = "dev-smoke";
pub const NIXIA_MICRO: &str = "nixia-micro";
pub const NIXIA_TINY: &str = "nixia-tiny";
pub const NIXIA_CODER: &str = "nixia-coder";

pub fn get_preset_config(name: &str) -> Option<TinyLmConfig> {
    match name {
        DEV_SMOKE => Some(
            TinyLmConfig::new()
                .with_vocab_size(128)
                .with_max_seq_len(64)
                .with_d_model(64)
                .with_n_layers(2)
                .with_n_heads(2)
                .with_d_ff(128)
                .with_dropout(0.1)
                .with_pad_token_id(0)
                .with_quantization(None),
        ),
        NIXIA_MICRO => Some(
            TinyLmConfig::new()
                .with_vocab_size(32000)
                .with_max_seq_len(256)
                .with_d_model(256)
                .with_n_layers(4)
                .with_n_heads(4)
                .with_d_ff(1024)
                .with_dropout(0.1)
                .with_pad_token_id(0)
                .with_quantization(None),
        ),
        NIXIA_TINY => Some(
            TinyLmConfig::new()
                .with_vocab_size(32000)
                .with_max_seq_len(512)
                .with_d_model(512)
                .with_n_layers(8)
                .with_n_heads(8)
                .with_d_ff(2048)
                .with_dropout(0.1)
                .with_pad_token_id(0)
                .with_quantization(None),
        ),
        NIXIA_CODER => Some(
            TinyLmConfig::new()
                .with_vocab_size(32000)
                .with_max_seq_len(1024)
                .with_d_model(1024)
                .with_n_layers(16)
                .with_n_heads(16)
                .with_d_ff(4096)
                .with_dropout(0.1)
                .with_pad_token_id(0)
                .with_quantization(None),
        ),
        _ => None,
    }
}
