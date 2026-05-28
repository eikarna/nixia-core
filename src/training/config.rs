use burn::{config::Config, optim::AdamConfig};

use crate::model::TinyLmConfig;

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model: TinyLmConfig,
    pub optimizer: AdamConfig,

    #[config(default = 8)]
    pub num_epochs: usize,

    #[config(default = 32)]
    pub batch_size: usize,

    #[config(default = 64)]
    pub stride: usize,

    #[config(default = 0)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 2.0e-4)]
    pub learning_rate: f64,
}
