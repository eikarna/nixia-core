use criterion::{criterion_group, criterion_main, Criterion};
use nixia::model::{TinyLmConfig, TinyLm};
use burn::tensor::Tensor;
use burn::backend::ndarray::NdArrayDevice;

fn bench_lm_forward(c: &mut Criterion) {
    type Backend = burn::backend::NdArray;
    let device = NdArrayDevice::Cpu;

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
        [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
        &device,
    );

    c.bench_function("lm_forward", |b| {
        b.iter(|| model.forward(token_ids.clone()))
    });
}

criterion_group!(benches, bench_lm_forward);
criterion_main!(benches);
