use burn::{
    prelude::*,
    tensor::backend::AutodiffBackend,
    train::{ClassificationOutput, InferenceStep, TrainOutput, TrainStep},
};

use crate::{data::LmBatch, model::TinyLm};

impl<B: AutodiffBackend> TrainStep for TinyLm<B> {
    type Input = LmBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: LmBatch<B>) -> TrainOutput<Self::Output> {
        let item = self.forward_classification(batch.inputs, batch.targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> InferenceStep for TinyLm<B> {
    type Input = LmBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, batch: LmBatch<B>) -> Self::Output {
        self.forward_classification(batch.inputs, batch.targets)
    }
}
