use criterion::{criterion_group, criterion_main, Criterion};
use nixia::model::{TinyLmConfig, TinyLm};
use burn::tensor::Tensor;

fn bench_lm_forward(c: &mut Criterion) {
    type Backend = burn::backend::NdArray;
    let device = Default::default();

    let config = TinyLmConfig {
        vocab_size: 100,
        max_seq_len: 128,
        d_model: 64,
        n_layers: 2,
        n_heads: 2,
        d_ff: 128,
        dropout: 0.0,
        pad_token_id: 0,
    };

    let model: TinyLm<Backend> = config.init(&device);
    let token_ids = Tensor::<Backend, 2, burn::tensor::Int>::from_data(
        [[1]],
        &device,
    );

    c.bench_function("lm_forward_seq1", |b| {
        b.iter(|| model.forward(token_ids.clone()))
    });
}

criterion_group!(benches, bench_lm_forward);
criterion_main!(benches);
