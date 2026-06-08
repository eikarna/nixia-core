use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use nixia::inference::sampling::{GenerationConfig, TokenSampler};
use std::hint::black_box;

fn bench_sampling(c: &mut Criterion) {
    let mut group = c.benchmark_group("TokenSampler::sample");

    let vocab_size = 32_000;

    // Generate pseudo-random logits without relying on rand specific version issues
    let mut logits: Vec<f32> = Vec::with_capacity(vocab_size);
    for i in 0..vocab_size {
        logits.push((i as f32 % 20.0) - 10.0);
    }

    for window_size in [64, 256, 1024].iter() {
        // Generate pseudo-random history history
        let mut history: Vec<usize> = Vec::with_capacity(*window_size);
        for i in 0..*window_size {
            history.push((i * 137) % vocab_size);
        }

        let config = GenerationConfig {
            repetition_window: *window_size,
            repetition_penalty: 1.2,
            ..GenerationConfig::default()
        };

        group.bench_with_input(
            BenchmarkId::new("window_size", window_size),
            window_size,
            |b, _| {
                let mut sampler = TokenSampler::new(42);
                b.iter(|| {
                    sampler.sample(black_box(&logits), black_box(&history), black_box(&config))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_sampling);
criterion_main!(benches);
