use burn::{
    data::{dataloader::batcher::Batcher, dataset::Dataset},
    prelude::*,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};

#[derive(Clone, Debug)]
pub struct LmItem {
    pub input: Vec<i64>,
    pub target: Vec<i64>,
}

#[derive(Clone, Debug)]
pub struct LmDataset {
    items: Vec<LmItem>,
}

impl LmDataset {
    pub fn from_token_stream(token_ids: &[usize], seq_len: usize, stride: usize) -> Self {
        if seq_len == 0 || token_ids.len() <= seq_len {
            return Self { items: Vec::new() };
        }

        let stride = stride.max(1);
        let max_start = token_ids.len() - seq_len - 1;
        let mut items = Vec::new();

        for start in (0..=max_start).step_by(stride) {
            let input = token_ids[start..start + seq_len]
                .iter()
                .map(|&id| id as i64)
                .collect();

            let target = token_ids[start + 1..start + seq_len + 1]
                .iter()
                .map(|&id| id as i64)
                .collect();

            items.push(LmItem { input, target });
        }

        Self { items }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn drop_remainder(mut self, batch_size: usize) -> Self {
        let batch_size = batch_size.max(1);
        let remainder = self.items.len() % batch_size;

        if remainder > 0 && self.items.len() > batch_size {
            self.items.truncate(self.items.len() - remainder);
        }

        self
    }
}

impl Dataset<LmItem> for LmDataset {
    fn get(&self, index: usize) -> Option<LmItem> {
        self.items.get(index).cloned()
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct LmBatcher;

#[derive(Clone, Debug)]
pub struct LmBatch<B: Backend> {
    pub inputs: Tensor<B, 2, Int>,
    pub targets: Tensor<B, 2, Int>,
}

impl<B: Backend> Batcher<B, LmItem, LmBatch<B>> for LmBatcher {
    fn batch(&self, items: Vec<LmItem>, device: &B::Device) -> LmBatch<B> {
        let batch_size = items.len();
        let seq_len = items
            .first()
            .map(|item| item.input.len())
            .unwrap_or_default();

        let capacity = batch_size * seq_len;
        let mut inputs = Vec::with_capacity(capacity);
        let mut targets = Vec::with_capacity(capacity);

        for item in items.iter() {
            inputs.extend(item.input.iter().map(|&id| id.elem::<B::IntElem>()));
            targets.extend(item.target.iter().map(|&id| id.elem::<B::IntElem>()));
        }

        LmBatch {
            inputs: Tensor::<B, 2, Int>::from_data(
                TensorData::new(inputs, [batch_size, seq_len]),
                device,
            ),
            targets: Tensor::<B, 2, Int>::from_data(
                TensorData::new(targets, [batch_size, seq_len]),
                device,
            ),
        }
    }
}
