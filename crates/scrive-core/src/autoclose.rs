//! Auto-close pair rules. The fixed pair set and the character predicates the
//! type/backspace verbs consult; the stateful guards (provenance, quote-parity)
//! live on [`Document`](crate::Document) because they need the buffer to decide.
//!
//! Pair set: `()`, `[]`, `{}`, `""` — the universal bracket set plus string
//! quotes. Every pair char is single-byte ASCII, so a byte offset and a char
//! offset coincide at a pair and no multi-byte handling is needed here.

/// The closing char for an opener, or `None` if `open` isn't a pair opener.
pub(crate) fn opener_close(open: char) -> Option<char> {
    match open {
        '(' => Some(')'),
        '[' => Some(']'),
        '{' => Some('}'),
        '"' => Some('"'),
        _ => None,
    }
}

/// Whether `ch` is a closing char of the pair set. Consulted when the caret
/// sits just before `ch` (to decide whether typing that closer should overtype
/// it rather than insert a duplicate).
pub(crate) fn is_closer(ch: char) -> bool {
    matches!(ch, ')' | ']' | '}' | '"')
}

/// Whether `ch` is a quote. Quotes carry extra guards the bracket pairs don't,
/// since a quote is its own opener and closer and must not auto-close in the
/// middle of a word or against an already-open quote.
pub(crate) fn is_quote(ch: char) -> bool {
    ch == '"'
}

/// Word-char test for the quote guard — delegates to the one owner,
/// [`crate::movement::is_word_char`], so the auto-close and word-motion notions
/// of a word boundary stay identical.
pub(crate) fn is_word_char(ch: char) -> bool {
    crate::movement::is_word_char(ch)
}
