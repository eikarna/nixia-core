use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig, Embedding, EmbeddingConfig, RmsNorm, RmsNormConfig,
        attention::generate_autoregressive_mask, loss::CrossEntropyLossConfig,
    },
    prelude::*,
    tensor::{Int, Tensor},
    train::ClassificationOutput,
};

use super::block::{DecoderBlock, DecoderBlockConfig};
use super::quantization::QuantizationConfig;

#[derive(Config, Debug)]
pub struct TinyLmConfig {
    #[config(default = 8000)]
    pub vocab_size: usize,

    #[config(default = 128)]
    pub max_seq_len: usize,

    #[config(default = 256)]
    pub d_model: usize,

    #[config(default = 8)]
    pub n_layers: usize,

    #[config(default = 4)]
    pub n_heads: usize,

    #[config(default = 1024)]
    pub d_ff: usize,

    #[config(default = 0.1)]
    pub dropout: f64,

    #[config(default = 0)]
    pub pad_token_id: usize,

    // Removing default = None because burn config macro has issues parsing None literals
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Module, Debug)]
pub struct TinyLm<B: Backend> {
    token_embedding: Embedding<B>,
    position_embedding: Embedding<B>,
    blocks: Vec<DecoderBlock<B>>,
    norm: RmsNorm<B>,
    dropout: Dropout,
    max_seq_len: usize,
    vocab_size: usize,
    pad_token_id: usize,
}

impl TinyLmConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> TinyLm<B> {
        assert!(
            self.d_model.is_multiple_of(self.n_heads),
            "d_model must be divisible by n_heads"
        );

        let block_config = DecoderBlockConfig {
            d_model: self.d_model,
            n_heads: self.n_heads,
            d_ff: self.d_ff,
            dropout: self.dropout,
            quantization: self.quantization.clone(),
        };

        TinyLm {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.d_model).init(device),
            position_embedding: EmbeddingConfig::new(self.max_seq_len, self.d_model).init(device),
            blocks: (0..self.n_layers)
                .map(|_| block_config.init(device))
                .collect(),
            norm: RmsNormConfig::new(self.d_model).init(device),
            dropout: DropoutConfig::new(self.dropout).init(),
            max_seq_len: self.max_seq_len,
            vocab_size: self.vocab_size,
            pad_token_id: self.pad_token_id,
        }
    }
}

impl<B: Backend> TinyLm<B> {
    pub fn forward(&self, token_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch_size, seq_len] = token_ids.dims();
        assert!(
            seq_len <= self.max_seq_len,
            "input sequence length exceeds configured context length"
        );

        let device = token_ids.device();
        let pos_ids = Tensor::<B, 1, Int>::arange(0..seq_len as i64, &device)
            .unsqueeze_dim(0)
            .repeat_dim(0, batch_size);

        let token_embed = self.token_embedding.forward(token_ids);
        let pos_embed = self.position_embedding.forward(pos_ids);
        let mut x = self.dropout.forward(token_embed + pos_embed);
        let mut mask_opt = if seq_len > 1 {
            Some(generate_autoregressive_mask::<B>(
                batch_size, seq_len, &device,
            ))
        } else {
            None
        };
        let last_idx = self.blocks.len().saturating_sub(1);

        for (i, block) in self.blocks.iter().enumerate() {
            let mask = if i == last_idx {
                mask_opt.take()
            } else {
                mask_opt.clone()
            };
            x = block.forward(x, mask);
        }

        self.output_projection(self.norm.forward(x))
    }

    fn output_projection(&self, hidden: Tensor<B, 3>) -> Tensor<B, 3> {
        let [batch_size, seq_len, d_model] = hidden.dims();
        let token_embedding_weight = self.token_embedding.weight.val();
        let logits = hidden
            .reshape([batch_size * seq_len, d_model])
            .matmul(token_embedding_weight.transpose());

        logits.reshape([batch_size, seq_len, self.vocab_size])
    }

    pub fn forward_classification(
        &self,
        inputs: Tensor<B, 2, Int>,
        targets: Tensor<B, 2, Int>,
    ) -> ClassificationOutput<B> {
        let logits = self.forward(inputs);
        let [batch_size, seq_len, vocab_size] = logits.dims();
        let logits = logits.reshape([batch_size * seq_len, vocab_size]);
        let targets = targets.reshape([batch_size * seq_len]);

        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());

        ClassificationOutput::new(loss, logits, targets)
    }

    pub fn calibrate(&mut self) {}

    pub fn apply_quantization(&mut self) {
        for block in self.blocks.iter_mut() {
            block.quantize();
        }
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
