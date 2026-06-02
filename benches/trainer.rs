use criterion::{Criterion, criterion_group, criterion_main};
use nixia::tokenizer::trainer::{BpeTrainerConfig, train_vocab};
use std::fs;

fn bench_train(c: &mut Criterion) {
    let corpus = fs::read_to_string("data/sample_corpus.txt").unwrap_or_else(|_| {
        "This is a dummy corpus text used for benchmarking the BPE tokenizer. It contains various words, repeated structures, and some special tokens if needed. Let's make it a bit longer to simulate real text processing.".to_string()
    });
    let corpus = corpus.repeat(10);
    let config = BpeTrainerConfig {
        vocab_size: 500,
        min_pair_frequency: 2,
    };

    c.bench_function("train_vocab", |b| {
        b.iter(|| {
            train_vocab(
                std::hint::black_box(&corpus),
                std::hint::black_box(config.clone()),
            )
        })
    });
}

criterion_group!(benches, bench_train);
criterion_main!(benches);
