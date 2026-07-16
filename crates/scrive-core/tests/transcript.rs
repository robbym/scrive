//! Headless acceptance gate: a scripted editing session driven entirely
//! through the public `Document` API, asserting text + dirty + undo/redo at each
//! step. This is the cross-module acceptance test for the transaction/undo core
//! — it proves the aggregate behaves, not just each module in isolation.
//!
//! Read it top to bottom: it is the transcript. The `examples/transcript.rs`
//! program prints the same shape for eyeball review.

use scrive_core::{Document, EditOp, EolFlavor, GroupingHint, OpClass};

/// Insert `text` at `at` as a mergeable keystroke (one typing run = one undo).
fn key(doc: &mut Document, at: u32, text: &str) {
    doc.edit_grouped(vec![EditOp::insert(at, text)], GroupingHint::mergeable(OpClass::Type))
        .unwrap();
}

#[test]
fn a_full_editing_session() {
    // Open an empty document.
    let mut doc = Document::new("").unwrap();
    assert_eq!(doc.text(), "");
    assert!(!doc.is_dirty());
    assert_eq!(doc.revision().0, 0);

    // Type "hello world" one key at a time — a single mergeable run.
    for (i, ch) in "hello world".chars().enumerate() {
        key(&mut doc, i as u32, &ch.to_string());
    }
    assert_eq!(doc.text(), "hello world");
    assert!(doc.is_dirty());

    // Save. Clean, but the text stays.
    doc.mark_saved();
    assert!(!doc.is_dirty());

    // A discrete multi-range edit: uppercase both words' initials in one undo
    // step (two ranges, one transaction).
    doc.edit(vec![EditOp::new(0..1, "H"), EditOp::new(6..7, "W")]).unwrap();
    assert_eq!(doc.text(), "Hello World");
    assert!(doc.is_dirty());

    // Undo the capitalization — back to the saved state, so it reads clean.
    assert!(doc.undo());
    assert_eq!(doc.text(), "hello world");
    assert!(!doc.is_dirty());

    // Redo it.
    assert!(doc.redo());
    assert_eq!(doc.text(), "Hello World");
    assert!(doc.is_dirty());

    // Undo the whole typing run too (redo was already consumed; undo the caps
    // first, then the run).
    assert!(doc.undo()); // -> "hello world"
    assert!(doc.undo()); // -> "" (the entire run, one unit)
    assert_eq!(doc.text(), "");
    assert!(!doc.undo(), "nothing left to undo");

    // Type a divergent line; the old redo branch is gone.
    key(&mut doc, 0, "bye");
    assert_eq!(doc.text(), "bye");
    assert!(!doc.redo(), "divergent edit cleared the redo branch");

    // Newline handling + CRLF-on-save round trip.
    doc.edit(vec![EditOp::insert(3, "\r\nline2")]).unwrap(); // \r\n normalized in
    assert_eq!(doc.text(), "bye\nline2"); // stored LF-only
    assert_eq!(doc.buffer().line_count(), 2);
    assert_eq!(doc.serialize(EolFlavor::CrLf), "bye\r\nline2");
    assert_eq!(doc.serialize(EolFlavor::Lf), "bye\nline2");
}

#[test]
fn multi_range_undo_restores_every_span() {
    // Multi-cursor undo: N simultaneous edits in one transaction undo together.
    let mut doc = Document::new("a a a a").unwrap();
    doc.edit(vec![
        EditOp::new(0..1, "X"),
        EditOp::new(2..3, "X"),
        EditOp::new(4..5, "X"),
        EditOp::new(6..7, "X"),
    ])
    .unwrap();
    assert_eq!(doc.text(), "X X X X");
    assert!(doc.undo());
    assert_eq!(doc.text(), "a a a a", "all four spans restored by one undo");
    assert!(doc.redo());
    assert_eq!(doc.text(), "X X X X");
}
