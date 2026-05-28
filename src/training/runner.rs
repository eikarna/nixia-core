use std::{fs, path::Path};

use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    prelude::*,
    record::DefaultRecorder,
    tensor::backend::AutodiffBackend,
    train::{Learner, SupervisedTraining, metric::LossMetric},
};

use crate::{
    NixiaError, Result,
    data::{LmBatcher, LmDataset},
    model::TinyLmConfig,
    training::TrainingConfig,
};

#[derive(Clone, Debug, Default)]
pub struct TrainOptions {
    pub init_from: Option<String>,
    pub resume_epoch: Option<usize>,
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    train_ids: Vec<usize>,
    valid_ids: Vec<usize>,
    config: TrainingConfig,
    options: TrainOptions,
    device: B::Device,
) -> Result<()> {
    validate_train_options(artifact_dir, &config, &options)?;

    fs::create_dir_all(artifact_dir)?;
    config
        .save(format!("{artifact_dir}/config.json"))
        .map_err(|error| NixiaError::Recorder(error.to_string()))?;
    config
        .model
        .save(format!("{artifact_dir}/model_config.json"))
        .map_err(|error| NixiaError::Recorder(error.to_string()))?;

    B::seed(&device, config.seed);

    let train_dataset =
        LmDataset::from_token_stream(&train_ids, config.model.max_seq_len, config.stride);
    let valid_dataset = LmDataset::from_token_stream(
        &valid_ids,
        config.model.max_seq_len,
        config.model.max_seq_len,
    );

    if train_dataset.is_empty() {
        return Err(NixiaError::InvalidArgument(format!(
            "not enough training tokens for max_seq_len={}; got {} tokens, need at least {}. Use a larger corpus, --preset dev-smoke for sample data, or lower --seq-len.",
            config.model.max_seq_len,
            train_ids.len(),
            config.model.max_seq_len + 1
        )));
    }

    if valid_dataset.is_empty() {
        return Err(NixiaError::InvalidArgument(format!(
            "not enough validation tokens for max_seq_len={}; got {} tokens, need at least {}. Pass --valid with a larger validation file, use --preset dev-smoke for sample data, or lower --seq-len.",
            config.model.max_seq_len,
            valid_ids.len(),
            config.model.max_seq_len + 1
        )));
    }

    let train_batch_size = config.batch_size.min(train_dataset.len()).max(1);
    let valid_batch_size = config.batch_size.min(valid_dataset.len()).max(1);
    let train_dataset = train_dataset.drop_remainder(train_batch_size);
    let valid_dataset = valid_dataset.drop_remainder(valid_batch_size);

    let train_loader = DataLoaderBuilder::new(LmBatcher)
        .batch_size(train_batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(train_dataset);

    let valid_loader = DataLoaderBuilder::new(LmBatcher)
        .batch_size(valid_batch_size)
        .num_workers(config.num_workers)
        .build(valid_dataset);

    let mut training = SupervisedTraining::new(artifact_dir, train_loader, valid_loader)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(DefaultRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    if let Some(epoch) = options.resume_epoch {
        training = training.checkpoint(epoch);
    }

    let mut model = config.model.init::<B>(&device);
    if let Some(init_from) = options.init_from.as_deref() {
        model = model
            .load_file(
                format!("{init_from}/model"),
                &DefaultRecorder::new(),
                &device,
            )
            .map_err(|error| NixiaError::Recorder(error.to_string()))?;
    }

    let result = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));

    result
        .model
        .save_file(format!("{artifact_dir}/model"), &DefaultRecorder::new())
        .map_err(|error| NixiaError::Recorder(error.to_string()))?;

    Ok(())
}

fn validate_train_options(
    artifact_dir: &str,
    config: &TrainingConfig,
    options: &TrainOptions,
) -> Result<()> {
    if options.init_from.is_some() && options.resume_epoch.is_some() {
        return Err(NixiaError::InvalidArgument(
            "--init-from and --resume-epoch cannot be used together".to_string(),
        ));
    }

    if let Some(epoch) = options.resume_epoch {
        if epoch >= config.num_epochs {
            return Err(NixiaError::InvalidArgument(format!(
                "--resume-epoch ({epoch}) must be smaller than --epochs ({})",
                config.num_epochs
            )));
        }

        let existing = load_artifact_model_config(artifact_dir)?;
        ensure_compatible_model_config(&config.model, &existing, "resume checkpoint")?;
        ensure_checkpoint_files(artifact_dir, epoch)?;
    }

    if let Some(init_from) = options.init_from.as_deref() {
        ensure_different_artifacts(artifact_dir, init_from)?;
        let existing = load_artifact_model_config(init_from)?;
        ensure_compatible_model_config(&config.model, &existing, "--init-from artifact")?;
        ensure_file_exists(Path::new(init_from).join("model.mpk"))?;
    }

    Ok(())
}

fn load_artifact_model_config(artifact_dir: &str) -> Result<TinyLmConfig> {
    let model_config_path = Path::new(artifact_dir).join("model_config.json");
    if model_config_path.exists() {
        return TinyLmConfig::load(model_config_path)
            .map_err(|error| NixiaError::Recorder(error.to_string()));
    }

    TrainingConfig::load(Path::new(artifact_dir).join("config.json"))
        .map(|config| config.model)
        .map_err(|error| NixiaError::Recorder(error.to_string()))
}

fn ensure_compatible_model_config(
    expected: &TinyLmConfig,
    actual: &TinyLmConfig,
    source: &str,
) -> Result<()> {
    let mut differences = Vec::new();

    if expected.vocab_size != actual.vocab_size {
        differences.push(format!(
            "vocab_size expected {}, got {}",
            expected.vocab_size, actual.vocab_size
        ));
    }
    if expected.max_seq_len != actual.max_seq_len {
        differences.push(format!(
            "max_seq_len expected {}, got {}",
            expected.max_seq_len, actual.max_seq_len
        ));
    }
    if expected.d_model != actual.d_model {
        differences.push(format!(
            "d_model expected {}, got {}",
            expected.d_model, actual.d_model
        ));
    }
    if expected.n_layers != actual.n_layers {
        differences.push(format!(
            "n_layers expected {}, got {}",
            expected.n_layers, actual.n_layers
        ));
    }
    if expected.n_heads != actual.n_heads {
        differences.push(format!(
            "n_heads expected {}, got {}",
            expected.n_heads, actual.n_heads
        ));
    }
    if expected.d_ff != actual.d_ff {
        differences.push(format!(
            "d_ff expected {}, got {}",
            expected.d_ff, actual.d_ff
        ));
    }
    if (expected.dropout - actual.dropout).abs() > f64::EPSILON {
        differences.push(format!(
            "dropout expected {}, got {}",
            expected.dropout, actual.dropout
        ));
    }
    if expected.pad_token_id != actual.pad_token_id {
        differences.push(format!(
            "pad_token_id expected {}, got {}",
            expected.pad_token_id, actual.pad_token_id
        ));
    }

    if !differences.is_empty() {
        return Err(NixiaError::InvalidArgument(format!(
            "model config mismatch for {source}: {}",
            differences.join("; ")
        )));
    }

    Ok(())
}

fn ensure_checkpoint_files(artifact_dir: &str, epoch: usize) -> Result<()> {
    let checkpoint_dir = Path::new(artifact_dir).join("checkpoint");
    for name in ["model", "optim", "scheduler"] {
        ensure_file_exists(checkpoint_dir.join(format!("{name}-{epoch}.mpk")))?;
    }
    Ok(())
}

fn ensure_file_exists(path: impl AsRef<Path>) -> Result<()> {
    let path = path.as_ref();
    if !path.is_file() {
        return Err(NixiaError::InvalidArgument(format!(
            "required file does not exist: {}",
            path.display()
        )));
    }
    Ok(())
}

fn ensure_different_artifacts(artifact_dir: &str, init_from: &str) -> Result<()> {
    if Path::new(artifact_dir) == Path::new(init_from) {
        return Err(NixiaError::InvalidArgument(
            "--init-from must point to a different artifact directory than --artifacts".to_string(),
        ));
    }

    let output = Path::new(artifact_dir);
    let source = Path::new(init_from);
    if output.exists() && source.exists() && fs::canonicalize(output)? == fs::canonicalize(source)?
    {
        return Err(NixiaError::InvalidArgument(
            "--init-from must point to a different artifact directory than --artifacts".to_string(),
        ));
    }

    Ok(())
}
