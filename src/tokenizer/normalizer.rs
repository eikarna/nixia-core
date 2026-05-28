const MAX_REPEAT: usize = 3;

pub fn normalize_text(text: &str) -> String {
    text.to_lowercase()
        .replace("\r\n", "\n")
        .replace('\n', " <nl> ")
        .split_whitespace()
        .map(normalize_token)
        .filter(|token| !token.is_empty())
        .collect::<Vec<_>>()
        .join(" ")
}

fn normalize_token(token: &str) -> String {
    if token.starts_with("http://") || token.starts_with("https://") {
        return "<url>".to_string();
    }

    if token.starts_with('@') && token.chars().count() > 1 {
        return "<user>".to_string();
    }

    if token.chars().all(|ch| ch.is_ascii_digit()) {
        return "<num>".to_string();
    }

    collapse_repeated_chars(token, MAX_REPEAT)
}

fn collapse_repeated_chars(token: &str, max_repeat: usize) -> String {
    let mut out = String::new();
    let mut last = None;
    let mut count = 0usize;

    for ch in token.chars() {
        if Some(ch) == last {
            count += 1;
        } else {
            last = Some(ch);
            count = 1;
        }

        if count <= max_repeat {
            out.push(ch);
        }
    }

    out
}

#[cfg(test)]
mod tests {
    use super::normalize_text;

    #[test]
    fn normalizes_common_noise() {
        let text = "HALOOO @aku https://x.test 123\nBangettt";
        assert_eq!(
            normalize_text(text),
            "halooo <user> <url> <num> <nl> bangettt"
        );
    }
}
