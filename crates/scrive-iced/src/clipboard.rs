//! Clipboard side table and end-of-line export.
//!
//! The OS clipboard carries text only, but whole-line Copy/Cut needs one bit
//! of metadata — *"this was an entire-line copy"* — so paste can splice the
//! line above the caret instead of inserting at it. That bit lives in an
//! in-process, single-entry table keyed by a **hash of the exported
//! (OS-flavor) text**: hashing the exported bytes means an OS clipboard
//! round-trip still matches, and a miss (text copied elsewhere, app restart)
//! degrades to a plain paste, which is always a safe fallback.
//!
//! [`export_eol`] is the one place LF is re-expanded to the OS line flavor: the
//! editor's buffer is LF-only, and Copy re-expands to CRLF on Windows so text
//! pasted into other apps uses the platform's expected line endings.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::sync::Mutex;

/// The last export's `(hash(exported_text), is_entire_line)`. One entry is the
/// honest minimum: only the most recent copy can still be on the OS clipboard
/// *from us*; anything else is external and must degrade to a plain paste.
static LAST_EXPORT: Mutex<Option<(u64, bool)>> = Mutex::new(None);

fn hash_of(text: &str) -> u64 {
    let mut h = DefaultHasher::new();
    text.hash(&mut h);
    h.finish()
}

/// Re-expand LF-only buffer text to the OS clipboard flavor: CRLF on Windows
/// (platform convention), LF elsewhere.
pub(crate) fn export_eol(text: &str) -> String {
    if cfg!(windows) {
        text.replace('\n', "\r\n")
    } else {
        text.to_owned()
    }
}

/// Record a Copy/Cut export's `is_entire_line` bit, keyed by the exported text.
pub(crate) fn record(exported: &str, is_entire_line: bool) {
    *LAST_EXPORT.lock().expect("clipboard table lock") = Some((hash_of(exported), is_entire_line));
}

/// Whether `pasted` (as read back from the OS clipboard) is our last
/// whole-line export. A hash miss is a plain paste.
pub(crate) fn is_entire_line(pasted: &str) -> bool {
    LAST_EXPORT.lock().expect("clipboard table lock").is_some_and(|(h, entire)| entire && h == hash_of(pasted))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip_matches_and_external_text_misses() {
        let exported = export_eol("two\n");
        record(&exported, true);
        assert!(is_entire_line(&exported), "our own export matches");
        assert!(!is_entire_line("something else"), "external text degrades to plain paste");
        // A later non-whole-line copy overwrites the entry.
        record("plain", false);
        assert!(!is_entire_line(&exported), "stale entry no longer matches");
        assert!(!is_entire_line("plain"), "and the new one is not entire-line");
    }

    #[test]
    fn export_eol_is_platform_flavored() {
        let out = export_eol("a\nb\n");
        if cfg!(windows) {
            assert_eq!(out, "a\r\nb\r\n");
        } else {
            assert_eq!(out, "a\nb\n");
        }
    }
}
