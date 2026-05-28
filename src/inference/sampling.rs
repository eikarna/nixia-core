#[derive(Clone, Debug)]
pub struct GenerationConfig {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub min_p: f32,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
    pub no_repeat_ngram_size: usize,
    pub seed: u64,
}

impl Default for GenerationConfig {
    fn default() -> Self {
        Self {
            max_new_tokens: 64,
            temperature: 0.8,
            top_k: 30,
            top_p: 0.92,
            min_p: 0.03,
            repetition_penalty: 1.1,
            repetition_window: 64,
            no_repeat_ngram_size: 3,
            seed: 42,
        }
    }
}

#[derive(Clone, Debug)]
pub struct TokenSampler {
    state: u64,
}

impl TokenSampler {
    pub fn new(seed: u64) -> Self {
        Self { state: seed.max(1) }
    }

    pub fn sample(
        &mut self,
        logits: &[f32],
        history: &[usize],
        config: &GenerationConfig,
    ) -> usize {
        if logits.is_empty() {
            return 0;
        }

        let recent_start = history.len().saturating_sub(config.repetition_window);
        let recent = &history[recent_start..];

        let mut scored = logits
            .iter()
            .enumerate()
            .map(|(id, &logit)| {
                let mut value = logit;
                if config.repetition_penalty > 1.0 && recent.contains(&id) {
                    value /= config.repetition_penalty;
                }
                if would_repeat_ngram(history, id, config.no_repeat_ngram_size) {
                    value = f32::NEG_INFINITY;
                }
                (id, value)
            })
            .collect::<Vec<_>>();

        scored.sort_by(|left, right| right.1.total_cmp(&left.1));
        scored.retain(|(_, score)| score.is_finite());

        if scored.is_empty() {
            return logits
                .iter()
                .enumerate()
                .max_by(|left, right| left.1.total_cmp(right.1))
                .map(|(id, _)| id)
                .unwrap_or_default();
        }

        if config.temperature <= 0.0 || config.top_k <= 1 {
            return scored.first().map(|(id, _)| *id).unwrap_or_default();
        }

        let keep = config.top_k.min(scored.len());
        scored.truncate(keep);

        let max_logit = scored[0].1;
        let temperature = config.temperature.max(1.0e-4);
        let mut weighted = Vec::with_capacity(scored.len());
        let mut total = 0.0;

        for (id, logit) in scored {
            let weight = ((logit - max_logit) / temperature).exp();
            total += weight;
            weighted.push((id, weight));
        }

        if total <= 0.0 || !total.is_finite() {
            return weighted.first().map(|(id, _)| *id).unwrap_or_default();
        }

        for (_, weight) in &mut weighted {
            *weight /= total;
        }

        weighted.sort_by(|left, right| right.1.total_cmp(&left.1));
        let best_probability = weighted[0].1;
        weighted.retain(|(_, probability)| *probability >= best_probability * config.min_p);

        let mut cumulative = 0.0;
        let mut filtered = Vec::new();
        for item in weighted {
            cumulative += item.1;
            filtered.push(item);
            if cumulative >= config.top_p.clamp(0.0, 1.0) {
                break;
            }
        }

        let total = filtered
            .iter()
            .map(|(_, probability)| probability)
            .sum::<f32>();
        let mut threshold = self.next_f32() * total;
        for (id, weight) in filtered {
            threshold -= weight;
            if threshold <= 0.0 {
                return id;
            }
        }

        0
    }

    fn next_f32(&mut self) -> f32 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let value = (self.state >> 40) as u32;
        (value as f32) / ((1u32 << 24) as f32)
    }
}

fn would_repeat_ngram(history: &[usize], candidate: usize, ngram_size: usize) -> bool {
    if ngram_size < 2 || history.len() + 1 < ngram_size {
        return false;
    }

    let prefix_len = ngram_size - 1;
    let prefix_start = history.len() - prefix_len;
    let prefix = &history[prefix_start..];

    history
        .windows(ngram_size)
        .any(|window| &window[..prefix_len] == prefix && window[prefix_len] == candidate)
}

#[cfg(test)]
mod tests {
    use super::{GenerationConfig, TokenSampler};

    #[test]
    fn avoids_repeated_ngram_when_possible() {
        let mut sampler = TokenSampler::new(1);
        let config = GenerationConfig {
            temperature: 0.0,
            no_repeat_ngram_size: 2,
            ..GenerationConfig::default()
        };

        let next = sampler.sample(&[0.0, 10.0, 9.0], &[1, 1], &config);
        assert_eq!(next, 2);
    }
}
