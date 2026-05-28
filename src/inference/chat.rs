use crate::tokenizer::special;

pub fn build_chat_prompt(user_message: &str) -> String {
    let trimmed = user_message.trim();

    if trimmed.contains(special::CHARACTER) || trimmed.contains(special::USER) {
        return trimmed.to_string();
    }

    format!("{} {} {}", special::USER, trimmed, special::CHARACTER)
}

pub fn clean_chat_output(text: &str) -> String {
    let mut text = text.trim().to_string();

    if let Some(index) = text.rfind(special::CHARACTER) {
        text = text[index + special::CHARACTER.len()..].trim().to_string();
    }

    if let Some(index) = text.find(special::USER) {
        text.truncate(index);
    }

    text.trim().to_string()
}

#[cfg(test)]
mod tests {
    use super::{build_chat_prompt, clean_chat_output};

    #[test]
    fn wraps_plain_user_message() {
        assert_eq!(build_chat_prompt("halo"), "<user> halo <char>");
    }

    #[test]
    fn keeps_only_character_output() {
        assert_eq!(clean_chat_output("<user> halo <char> hai juga"), "hai juga");
    }
}
