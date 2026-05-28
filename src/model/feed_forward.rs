use burn::{
    module::Module,
    nn::{Dropout, DropoutConfig, Linear, LinearConfig},
    prelude::*,
    tensor::{Tensor, activation::silu},
};

#[derive(Clone, Debug)]
pub struct SwiGluConfig {
    pub d_model: usize,
    pub d_ff: usize,
    pub dropout: f64,
}

#[derive(Module, Debug)]
pub struct SwiGluFeedForward<B: Backend> {
    gate: Linear<B>,
    up: Linear<B>,
    down: Linear<B>,
    dropout: Dropout,
}

impl SwiGluConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> SwiGluFeedForward<B> {
        SwiGluFeedForward {
            gate: LinearConfig::new(self.d_model, self.d_ff).init(device),
            up: LinearConfig::new(self.d_model, self.d_ff).init(device),
            down: LinearConfig::new(self.d_ff, self.d_model).init(device),
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
}
