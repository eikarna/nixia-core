use super::quantization::{QuantizationConfig, QuantizedLinear};
use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, LinearConfig},
    prelude::*,
    tensor::{Tensor, activation::silu},
};

#[derive(Clone, Debug)]
pub struct SwiGluConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout: f64,
    pub quantization: Option<QuantizationConfig>,
}

#[derive(Module, Debug)]
pub struct SwiGluFeedForward<B: Backend> {
    gate: QuantizedLinear<B>,
    up: QuantizedLinear<B>,
    down: QuantizedLinear<B>,
    dropout: Dropout,
}

impl SwiGluConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluFeedForward<B> {
        SwiGluFeedForward {
            gate: QuantizedLinear::new(
                LinearConfig::new(self.d_model, self.d_ff).init(device),
                self.quantization.clone(),
            ),
            up: QuantizedLinear::new(
                LinearConfig::new(self.d_model, self.d_ff).init(device),
                self.quantization.clone(),
            ),
            down: QuantizedLinear::new(
                LinearConfig::new(self.d_ff, self.d_model).init(device),
                self.quantization.clone(),
            ),
            dropout: DropoutConfig::new(self.dropout).init(),
        }
    }
}

impl<B: Backend> SwiGluFeedForward<B> {
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let gate = silu(self.gate.forward(x.clone()));
        let up = self.up.forward(x);
        let x = self.dropout.forward(gate * up);
        self.down.forward(x)
    }

    pub fn quantize(&mut self) {
        self.gate.quantize();
        self.up.quantize();
        self.down.quantize();
    }
}
