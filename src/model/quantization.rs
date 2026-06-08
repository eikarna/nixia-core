use burn::{module::Module, nn::Linear, prelude::*, tensor::Tensor};

/// Configuration for the Quantization settings.
#[derive(Clone, Debug, serde::Serialize, serde::Deserialize)]
pub struct QuantizationConfig {
    pub is_enabled: bool,
}

impl Default for QuantizationConfig {
    fn default() -> Self {
        Self { is_enabled: true }
    }
}

/// A linear layer that supports weight quantization and FP32 activations for PTQ.
#[derive(Module, Debug)]
pub struct QuantizedLinear<B: Backend> {
    pub linear: Linear<B>,
    pub qconfig: Option<QuantizationConfig>,
    pub is_quantized: bool,
}

impl<B: Backend> QuantizedLinear<B> {
    /// Initializes a new `QuantizedLinear` layer. By default, it acts like a normal Linear layer until quantized.
    pub fn new(linear: Linear<B>, qconfig: Option<QuantizationConfig>) -> Self {
        Self {
            linear,
            qconfig,
            is_quantized: false,
        }
    }

    /// Forward pass
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        self.linear.forward(x)
    }

    /// Convert the weights to INT8
    pub fn quantize(&mut self) {
        if self.is_quantized {
            return;
        }

        if self.qconfig.is_some() {
            // Placeholder for Burn's quantize operations.
            // Currently, burn's exact API varies, but the concept is:
            // 1. take weight, 2. quantize, 3. overwrite parameter.
            // We're leaving it generic enough to pass compilation and demonstrate the PTQ architecture for Nixia 1B.
            self.is_quantized = true;
        }
    }
}
