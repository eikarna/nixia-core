use burn::{
    config::Config,
    module::Module,
    prelude::*,
    record::DefaultRecorder,
    tensor::{ElementConversion, Int, Tensor, TensorData},
};

use crate::{
    NixiaError, Result,
    inference::sampling::{GenerationConfig, TokenSampler},
    inference::{build_chat_prompt, clean_chat_output},
    model::{TinyLm, TinyLmConfig},
    tokenizer::TinyTokenizer,
    training::TrainingConfig,
};

pub fn load_model<B: Backend>(artifact_dir: &str, device: &B::Device) -> Result<TinyLm<B>> {
    let config = load_model_config(artifact_dir)?;

    config
        .init::<B>(device)
        .load_file(
            format!("{artifact_dir}/model"),
            &DefaultRecorder::new(),
            device,
        )
        .map_err(|error| NixiaError::Recorder(error.to_string()))
}

fn load_model_config(artifact_dir: &str) -> Result<TinyLmConfig> {
    let model_config_path = format!("{artifact_dir}/model_config.json");
    if std::path::Path::new(&model_config_path).exists() {
        return TinyLmConfig::load(model_config_path)
            .map_err(|error| NixiaError::Recorder(error.to_string()));
    }

    TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .map(|config| config.model)
        .map_err(|error| NixiaError::Recorder(error.to_string()))
}

pub fn generate<B: Backend>(
    model: &TinyLm<B>,
    tokenizer: &TinyTokenizer,
    prompt: &str,
    config: GenerationConfig,
    device: &B::Device,
) -> Result<String> {
    let mut ids = tokenizer.encode(prompt, true);
    let mut sampler = TokenSampler::new(config.seed);

    for _ in 0..config.max_new_tokens {
        let logits = last_token_logits(model, &ids, device)?;
        let next = sampler.sample(&logits, &ids, &config);
        ids.push(next);

        if next == tokenizer.eos_id() {
            break;
        }
    }

    Ok(tokenizer.decode(&ids))
}

pub fn chat<B: Backend>(
    model: &TinyLm<B>,
    tokenizer: &TinyTokenizer,
    user_message: &str,
    config: GenerationConfig,
    device: &B::Device,
) -> Result<String> {
    let prompt = build_chat_prompt(user_message);
    let output = generate(model, tokenizer, &prompt, config, device)?;
    Ok(clean_chat_output(&output))
}

pub fn last_token_logits<B: Backend>(
    model: &TinyLm<B>,
    token_ids: &[usize],
    device: &B::Device,
) -> Result<Vec<f32>> {
    if token_ids.is_empty() {
        return Err(NixiaError::InvalidArgument(
            "prompt produced no tokens".to_string(),
        ));
    }

    let seq_len = token_ids.len().min(model.max_seq_len());
    let start = token_ids.len() - seq_len;
    let input = token_ids[start..]
        .iter()
        .map(|&id| (id as i64).elem::<B::IntElem>())
        .collect::<Vec<_>>();

    let input = Tensor::<B, 2, Int>::from_data(TensorData::new(input, [1, seq_len]), device);
    let logits = model.forward(input);
    let logits = logits
        .slice([0..1, seq_len - 1..seq_len, 0..model.vocab_size()])
        .reshape([model.vocab_size()]);

    logits
        .into_data()
        .to_vec::<f32>()
        .map_err(|error| NixiaError::InvalidArgument(error.to_string()))
}
