use burn::backend::ndarray::NdArrayDevice;
use burn::data::dataloader::batcher::Batcher;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use nixia::data::{LmBatcher, LmItem};

type Backend = burn::backend::NdArray;

fn create_mock_items(batch_size: usize, seq_len: usize) -> Vec<LmItem> {
    (0..batch_size)
        .map(|_| LmItem {
            input: vec![1_i64; seq_len],
            target: vec![2_i64; seq_len],
        })
        .collect()
}

fn bench_batcher(c: &mut Criterion) {
    let mut group = c.benchmark_group("lm_batcher");
    let device = NdArrayDevice::Cpu;
    let batcher = LmBatcher;

    for (batch_size, seq_len) in [(32, 128), (64, 512), (128, 1024)] {
        let items = create_mock_items(batch_size, seq_len);

        group.throughput(Throughput::Elements((batch_size * seq_len * 2) as u64));
        group.bench_with_input(
            BenchmarkId::new("batch", format!("bs{}_seq{}", batch_size, seq_len)),
            &items,
            |b, items: &Vec<LmItem>| {
                b.iter(|| {
                    <LmBatcher as Batcher<Backend, LmItem, _>>::batch(
                        &batcher,
                        items.clone(),
                        &device,
                    )
                });
            },
        );
    }
    group.finish();
}

criterion_group!(benches, bench_batcher);
criterion_main!(benches);
