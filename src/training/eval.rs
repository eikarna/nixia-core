use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
};

use crate::{
    NixiaError, Result,
    data::{LmBatcher, LmDataset},
    model::TinyLm,
};

#[derive(Clone, Copy, Debug)]
pub struct EvalReport {
    pub loss: f32,
    pub perplexity: f32,
    pub batches: usize,
}

pub fn evaluate<B: Backend>(
    model: &TinyLm<B>,
    token_ids: &[usize],
    seq_len: usize,
    batch_size: usize,
    device: &B::Device,
) -> Result<EvalReport> {
    let dataset = LmDataset::from_token_stream(token_ids, seq_len, seq_len);
    if dataset.is_empty() {
        return Err(NixiaError::InvalidArgument(
            "not enough evaluation tokens for configured seq_len".to_string(),
        ));
    }
    let batch_size = batch_size.min(dataset.len()).max(1);
    let dataset = dataset.drop_remainder(batch_size);

    let batcher = LmBatcher;
    let mut total_loss = 0.0f32;
    let mut batches = 0usize;
    let mut index = 0usize;

    while index < dataset.len() {
        let end = (index + batch_size).min(dataset.len());
        let items = (index..end)
            .filter_map(|item_index| dataset.get(item_index))
            .collect::<Vec<_>>();
        index = end;

        if items.is_empty() {
            continue;
        }

        let batch = batcher.batch(items, device);
        let output = model.forward_classification(batch.inputs, batch.targets);
        let loss = output
            .loss
            .into_data()
            .to_vec::<f32>()
            .map_err(|error| NixiaError::InvalidArgument(error.to_string()))?
            .into_iter()
            .next()
            .ok_or_else(|| NixiaError::InvalidArgument("empty loss tensor".to_string()))?;

        total_loss += loss;
        batches += 1;
    }

    if batches == 0 {
        return Err(NixiaError::InvalidArgument(
            "evaluation produced no batches".to_string(),
        ));
    }

    let loss = total_loss / batches as f32;
    Ok(EvalReport {
        loss,
        perplexity: loss.exp(),
        batches,
    })
}
