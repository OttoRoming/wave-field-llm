//! Character-level tokenizer
//!
//! Implements the same character tokenizer that was used to produce the
//! WikiText-2 benchmark results quoted in the README
//! (Wave Field V3.5 PPL 6.2 vs Standard Transformer PPL 5.9).
//!
//! Every unique UTF-8 character seen in training data gets a token ID.
//! Unknown characters map to a special `<UNK>` token.
//! Special tokens: `<PAD>=0`, `<BOS>=1`, `<EOS>=2`, `<UNK>=3`.
//!
//! # Example
//!
//! ```
//! use wave_field_rs::tokenizer::CharTokenizer;
//!
//! let mut tok = CharTokenizer::new();
//! tok.build_vocab("hello world");
//!
//! let ids = tok.encode("hello");
//! let text = tok.decode(&ids);
//! assert_eq!(text, "hello");
//! ```

use std::collections::HashMap;

/// Special token IDs.
pub const PAD_ID: u32 = 0;
pub const BOS_ID: u32 = 1;
pub const EOS_ID: u32 = 2;
pub const UNK_ID: u32 = 3;
const NUM_SPECIAL: u32 = 4;

/// Character-level tokenizer.
#[derive(Debug, Clone)]
pub struct CharTokenizer {
    /// char → token ID
    char_to_id: HashMap<char, u32>,
    /// token ID → char
    id_to_char: HashMap<u32, char>,
    /// Number of tokens including specials
    pub vocab_size: usize,
}

impl CharTokenizer {
    /// Create an empty tokenizer (before building vocab).
    pub fn new() -> Self {
        Self {
            char_to_id: HashMap::new(),
            id_to_char: HashMap::new(),
            vocab_size: NUM_SPECIAL as usize,
        }
    }

    /// Build vocabulary from `text`.
    ///
    /// Characters are assigned IDs in the order they first appear after the
    /// 4 reserved special tokens.  Call multiple times to extend the vocab.
    pub fn build_vocab(&mut self, text: &str) {
        for ch in text.chars() {
            if !self.char_to_id.contains_key(&ch) {
                let id = self.vocab_size as u32;
                self.char_to_id.insert(ch, id);
                self.id_to_char.insert(id, ch);
                self.vocab_size += 1;
            }
        }
    }

    /// Encode `text` to a sequence of token IDs.
    ///
    /// Unknown characters map to `UNK_ID`.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        text.chars()
            .map(|c| *self.char_to_id.get(&c).unwrap_or(&UNK_ID))
            .collect()
    }

    /// Encode with BOS prepended and EOS appended.
    pub fn encode_with_special(&self, text: &str) -> Vec<u32> {
        let mut ids = vec![BOS_ID];
        ids.extend(self.encode(text));
        ids.push(EOS_ID);
        ids
    }

    /// Decode a sequence of token IDs back to a String.
    ///
    /// Special tokens (PAD, BOS, EOS, UNK) are skipped.
    pub fn decode(&self, ids: &[u32]) -> String {
        ids.iter()
            .filter_map(|&id| {
                if id < NUM_SPECIAL {
                    None
                } else {
                    self.id_to_char.get(&id).copied()
                }
            })
            .collect()
    }

    /// Save vocabulary to a JSON string (for persistence).
    pub fn to_json(&self) -> String {
        // Represent as sorted list of (char, id) pairs
        let mut pairs: Vec<(String, u32)> = self
            .char_to_id
            .iter()
            .map(|(&c, &id)| (c.to_string(), id))
            .collect();
        pairs.sort_by_key(|(_, id)| *id);

        let entries: Vec<String> = pairs
            .iter()
            .map(|(c, id)| format!("{{\"char\":{},\"id\":{}}}", json_escape(c), id))
            .collect();

        format!("{{\"vocab_size\":{},\"chars\":[{}]}}", self.vocab_size, entries.join(","))
    }

    /// Restore a tokenizer previously saved with `to_json`.
    pub fn from_json(json: &str) -> Option<Self> {
        let mut tok = Self::new();

        // Extract vocab_size
        let vs_start = json.find("\"vocab_size\":")?;
        let vs_rest = &json[vs_start + 13..];
        let vs_end = vs_rest.find(|c: char| !c.is_ascii_digit())?;
        tok.vocab_size = vs_rest[..vs_end].parse::<usize>().ok()?;

        // Extract char entries with a simple parser
        // Format: {"char":"x","id":N}
        let mut pos = 0;
        while let Some(rel) = json[pos..].find("\"char\":") {
            let start = pos + rel + 7;
            // skip opening quote
            let rest = &json[start..];
            let (ch, consumed) = parse_json_char(rest)?;
            let after = start + consumed;

            // find "id":N
            let id_start = json[after..].find("\"id\":")?;
            let id_rest = &json[after + id_start + 5..];
            let id_end = id_rest.find(|c: char| !c.is_ascii_digit()).unwrap_or(id_rest.len());
            let id: u32 = id_rest[..id_end].parse().ok()?;

            tok.char_to_id.insert(ch, id);
            tok.id_to_char.insert(id, ch);

            pos = after + id_start + 5 + id_end;
        }

        Some(tok)
    }
}

impl Default for CharTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

// ─── helpers ─────────────────────────────────────────────────────────────────

fn json_escape(s: &str) -> String {
    let mut out = String::from('"');
    for ch in s.chars() {
        match ch {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                out.push_str(&format!("\\u{:04x}", c as u32));
            }
            c => out.push(c),
        }
    }
    out.push('"');
    out
}

/// Parse a JSON-encoded character from the start of `s` (which begins with `"`).
/// Returns `(char, bytes_consumed)`.
fn parse_json_char(s: &str) -> Option<(char, usize)> {
    if !s.starts_with('"') {
        return None;
    }
    let inner = &s[1..];
    if inner.starts_with("\\\"") {
        return Some(('"', 3));
    }
    if inner.starts_with("\\\\") {
        return Some(('\\', 3));
    }
    if inner.starts_with("\\n") {
        return Some(('\n', 3));
    }
    if inner.starts_with("\\r") {
        return Some(('\r', 3));
    }
    if inner.starts_with("\\t") {
        return Some(('\t', 3));
    }
    if inner.starts_with("\\u") {
        let hex = inner.get(2..6)?;
        let code = u32::from_str_radix(hex, 16).ok()?;
        let ch = char::from_u32(code)?;
        return Some((ch, 7)); // "\uXXXX" = 7 bytes: '"' + '\' + 'u' + 4 hex + '"'
    }
    // Plain character
    let ch = inner.chars().next()?;
    let byte_len = ch.len_utf8();
    // 2 = opening '"' + closing '"', plus the UTF-8 bytes of the character itself
    Some((ch, 2 + byte_len))
}

// ─── tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let mut tok = CharTokenizer::new();
        tok.build_vocab("hello world");

        let ids = tok.encode("hello");
        let text = tok.decode(&ids);
        assert_eq!(text, "hello");
    }

    #[test]
    fn test_special_tokens() {
        let mut tok = CharTokenizer::new();
        tok.build_vocab("abc");

        let ids = tok.encode_with_special("abc");
        assert_eq!(ids[0], BOS_ID);
        assert_eq!(*ids.last().unwrap(), EOS_ID);

        // decode strips specials
        let text = tok.decode(&ids);
        assert_eq!(text, "abc");
    }

    #[test]
    fn test_unk() {
        let mut tok = CharTokenizer::new();
        tok.build_vocab("abc");

        let ids = tok.encode("abcX");
        assert_eq!(*ids.last().unwrap(), UNK_ID);
    }

    #[test]
    fn test_vocab_size() {
        let mut tok = CharTokenizer::new();
        tok.build_vocab("abcd");
        // 4 special + 4 unique chars = 8
        assert_eq!(tok.vocab_size, 8);
    }

    #[test]
    fn test_json_roundtrip() {
        let mut tok = CharTokenizer::new();
        tok.build_vocab("hello world!\n");
        let json = tok.to_json();

        let tok2 = CharTokenizer::from_json(&json).expect("parse failed");
        assert_eq!(tok2.vocab_size, tok.vocab_size);

        let ids1 = tok.encode("hello");
        let ids2 = tok2.encode("hello");
        assert_eq!(ids1, ids2);
    }
}
