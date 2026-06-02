use burn::{
    module::Module,
    nn::{
        Dropout, DropoutConfig, RmsNorm, RmsNormConfig,
        attention::{MhaInput, MultiHeadAttention, MultiHeadAttentionConfig},
    },
    prelude::*,
    tensor::{Bool, Tensor},
};

use super::feed_forward::{SwiGluConfig, SwiGluFeedForward};

#[derive(Clone, Debug)]
pub struct DecoderBlockConfig {
    pub d_model: usize,
    pub n_heads: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct DecoderBlock<B: Backend> {
    norm_attn: RmsNorm<B>,
    self_attn: MultiHeadAttention<B>,
    norm_ff: RmsNorm<B>,
    feed_forward: SwiGluFeedForward<B>,
    dropout: Dropout,
}

impl DecoderBlockConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> DecoderBlock<B> {
        DecoderBlock {
            norm_attn: RmsNormConfig::new(self.d_model).init(device),
            self_attn: MultiHeadAttentionConfig::new(self.d_model, self.n_heads)
                .with_dropout(self.dropout)
                .with_min_float(-100.0)
                .with_quiet_softmax(false)
                .init(device),
            norm_ff: RmsNormConfig::new(self.d_model).init(device),
            feed_forward: SwiGluConfig {
                d_model: self.d_model,
                d_ff: self.d_ff,
                dropout: self.dropout,
            }
            .init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> DecoderBlock<B> {
    pub fn forward(&self, x: Tensor<B, 3>, causal_mask: Option<Tensor<B, 3, Bool>>) -> Tensor<B, 3> {
        let attn_input = self.norm_attn.forward(x.clone());
        let mut mha_input = MhaInput::self_attn(attn_input);
        if let Some(mask) = causal_mask {
            mha_input = mha_input.mask_attn(mask);
        }
        let attn = self.self_attn.forward(mha_input).context;
        let x = x + self.dropout.forward(attn);

        let ff = self.feed_forward.forward(self.norm_ff.forward(x.clone()));

        x + self.dropout.forward(ff)
    }
}
