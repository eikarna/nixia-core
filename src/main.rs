use std::{env, process::ExitCode};

use burn::optim::AdamConfig;
use nixia::{
    Result,
    data::{read_text, tokenize_corpus},
    inference::{GenerationConfig, chat, generate, load_model},
    model::{TinyLmConfig, preset},
    tokenizer::{BpeTrainerConfig, TinyTokenizer, train_vocab},
    training::{TrainOptions, TrainingConfig, evaluate, train},
};

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("error: {error}");
            ExitCode::FAILURE
        }
    }
}

fn run() -> Result<()> {
    let args = env::args().skip(1).collect::<Vec<_>>();
    let Some(command) = args.first().map(String::as_str) else {
        print_help();
        return Ok(());
    };

    match command {
        "tokenizer" => train_tokenizer(&args[1..]),
        "train" => train_model(&args[1..]),
        "eval" => eval_model(&args[1..]),
        "generate" => generate_text(&args[1..]),
        "help" | "--help" | "-h" => {
            print_help();
            Ok(())
        }
        other => Err(nixia::NixiaError::InvalidArgument(format!(
            "unknown command {other:?}"
        ))),
    }
}

fn train_tokenizer(args: &[String]) -> Result<()> {
    let corpus_path = flag(args, "--corpus").unwrap_or("data/sample_corpus.txt");
    let vocab_path = flag(args, "--vocab").unwrap_or("artifacts/vocab.txt");
    let vocab_size = flag(args, "--vocab-size")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(8_000);
    let min_pair_frequency = flag(args, "--min-pair-frequency")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(2);

    let corpus = read_text(corpus_path)?;
    let vocab = train_vocab(
        &corpus,
        BpeTrainerConfig {
            vocab_size,
            min_pair_frequency,
        },
    )?;
    let tokenizer = TinyTokenizer::new(vocab)?;
    tokenizer.save_vocab(vocab_path)?;

    println!(
        "saved tokenizer vocab to {vocab_path} ({} tokens)",
        tokenizer.vocab_size()
    );
    Ok(())
}

fn train_model(args: &[String]) -> Result<()> {
    let corpus_path = flag(args, "--corpus").unwrap_or("data/sample_corpus.txt");
    let valid_path = flag(args, "--valid");
    let vocab_path = flag(args, "--vocab").unwrap_or("artifacts/vocab.txt");
    let artifact_dir = flag(args, "--artifacts").unwrap_or("artifacts/run");
    let preset_name = flag(args, "--preset").unwrap_or(preset::REDMI_NANO);

    let batch_size = flag(args, "--batch-size")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(32);
    let epochs = flag(args, "--epochs")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(8);
    let learning_rate = flag(args, "--lr")
        .map(parse_f64)
        .transpose()?
        .unwrap_or(5.0e-5);
    let init_from = flag(args, "--init-from").map(ToOwned::to_owned);
    let resume_epoch = flag(args, "--resume-epoch").map(parse_usize).transpose()?;

    let tokenizer = TinyTokenizer::load(vocab_path)?;
    let train_text = read_text(corpus_path)?;
    let valid_text = valid_path.map(read_text).transpose()?;
    let corpus = tokenize_corpus(&tokenizer, &train_text, valid_text.as_deref(), 0.05)?;

    let model = model_config_from_args(args, &tokenizer, preset_name)?;
    let max_seq_len = model.max_seq_len;

    let config = TrainingConfig {
        model,
        optimizer: AdamConfig::new().with_epsilon(1.0e-6),
        num_epochs: epochs,
        batch_size,
        stride: max_seq_len / 2,
        num_workers: 0,
        seed: 42,
        learning_rate,
    };

    train_with_default_backend(
        artifact_dir,
        corpus.train_ids,
        corpus.valid_ids,
        config,
        TrainOptions {
            init_from,
            resume_epoch,
        },
    )
}

fn eval_model(args: &[String]) -> Result<()> {
    let corpus_path = flag(args, "--corpus").unwrap_or("data/sample_corpus.txt");
    let vocab_path = flag(args, "--vocab").unwrap_or("artifacts/vocab.txt");
    let artifact_dir = flag(args, "--artifacts").unwrap_or("artifacts/run");
    let batch_size = flag(args, "--batch-size")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(8);

    type Backend = burn::backend::Flex;
    let device = Default::default();
    let tokenizer = TinyTokenizer::load(vocab_path)?;
    let text = read_text(corpus_path)?;
    let token_ids = tokenizer.encode(&text, true);
    let model = load_model::<Backend>(artifact_dir, &device)?;
    let report = evaluate(&model, &token_ids, model.max_seq_len(), batch_size, &device)?;

    println!(
        "eval: loss={:.4}, perplexity={:.2}, batches={}",
        report.loss, report.perplexity, report.batches
    );
    Ok(())
}

fn generate_text(args: &[String]) -> Result<()> {
    let prompt = flag(args, "--prompt").unwrap_or("<user> halo <char>");
    let vocab_path = flag(args, "--vocab").unwrap_or("artifacts/vocab.txt");
    let artifact_dir = flag(args, "--artifacts").unwrap_or("artifacts/run");
    let max_new_tokens = flag(args, "--tokens")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(64);
    let temperature = flag(args, "--temperature")
        .map(parse_f32)
        .transpose()?
        .unwrap_or(0.8);
    let top_k = flag(args, "--top-k")
        .map(parse_usize)
        .transpose()?
        .unwrap_or(30);
    let top_p = flag(args, "--top-p")
        .map(parse_f32)
        .transpose()?
        .unwrap_or(0.92);
    let min_p = flag(args, "--min-p")
        .map(parse_f32)
        .transpose()?
        .unwrap_or(0.03);
    let chat_mode = has_flag(args, "--chat");

    type Backend = burn::backend::Flex;
    let device = Default::default();
    let tokenizer = TinyTokenizer::load(vocab_path)?;
    let model = load_model::<Backend>(artifact_dir, &device)?;
    let generation_config = GenerationConfig {
        max_new_tokens,
        temperature,
        top_k,
        top_p,
        min_p,
        ..GenerationConfig::default()
    };
    let text = if chat_mode {
        chat(&model, &tokenizer, prompt, generation_config, &device)?
    } else {
        generate(&model, &tokenizer, prompt, generation_config, &device)?
    };

    println!("{text}");
    Ok(())
}

fn model_config_from_args(
    args: &[String],
    tokenizer: &TinyTokenizer,
    preset_name: &str,
) -> Result<TinyLmConfig> {
    let mut model = preset::preset(preset_name, tokenizer.vocab_size(), tokenizer.pad_id())
        .ok_or_else(|| {
            nixia::NixiaError::InvalidArgument(format!(
                "unknown preset {preset_name:?}; available: {}",
                preset::names().join(", ")
            ))
        })?;

    if let Some(value) = flag(args, "--seq-len") {
        model.max_seq_len = parse_usize(value)?;
    }
    if let Some(value) = flag(args, "--d-model") {
        model.d_model = parse_usize(value)?;
    }
    if let Some(value) = flag(args, "--layers") {
        model.n_layers = parse_usize(value)?;
    }
    if let Some(value) = flag(args, "--heads") {
        model.n_heads = parse_usize(value)?;
    }
    if let Some(value) = flag(args, "--d-ff") {
        model.d_ff = parse_usize(value)?;
    }

    Ok(model)
}

fn train_with_default_backend(
    artifact_dir: &str,
    train_ids: Vec<usize>,
    valid_ids: Vec<usize>,
    config: TrainingConfig,
    options: TrainOptions,
) -> Result<()> {
    type Backend = burn::backend::Autodiff<burn::backend::Flex>;
    let device = Default::default();
    train::<Backend>(artifact_dir, train_ids, valid_ids, config, options, device)
}

fn flag<'a>(args: &'a [String], name: &str) -> Option<&'a str> {
    args.windows(2)
        .find(|window| window[0] == name)
        .map(|window| window[1].as_str())
}

fn has_flag(args: &[String], name: &str) -> bool {
    args.iter().any(|arg| arg == name)
}

fn parse_usize(value: &str) -> Result<usize> {
    value.parse::<usize>().map_err(|error| {
        nixia::NixiaError::InvalidArgument(format!("expected usize, got {value:?}: {error}"))
    })
}

fn parse_f32(value: &str) -> Result<f32> {
    value.parse::<f32>().map_err(|error| {
        nixia::NixiaError::InvalidArgument(format!("expected f32, got {value:?}: {error}"))
    })
}

fn parse_f64(value: &str) -> Result<f64> {
    value.parse::<f64>().map_err(|error| {
        nixia::NixiaError::InvalidArgument(format!("expected f64, got {value:?}: {error}"))
    })
}

fn print_help() {
    println!(
        "nixia - tiny Indonesian causal language model\n\n\
Commands:\n\
  tokenizer --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --vocab-size 8000\n\
  train --preset redmi-nano --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run\n\
  train --init-from artifacts/base --artifacts artifacts/finetune --corpus data/curated/train_corpus.txt --valid data/curated/valid_corpus.txt\n\
  train --resume-epoch 10 --epochs 15 --artifacts artifacts/run --corpus data/curated/train_corpus.txt --valid data/curated/valid_corpus.txt\n\
  eval --corpus data/sample_corpus.txt --vocab artifacts/vocab.txt --artifacts artifacts/run\n\
  generate --chat --artifacts artifacts/run --vocab artifacts/vocab.txt --prompt \"halo, kamu siapa?\"\n\n\
Presets: dev-smoke, redmi-nano, redmi-tiny\n\
Training uses Burn Flex CPU for stable, portable checkpoints.\n\
Use --init-from for fine-tuning compatible model weights, or --resume-epoch to continue an existing checkpoint."
    );
}
