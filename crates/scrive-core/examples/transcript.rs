//! A runnable transcript of the headless editing core.
//!
//! Run: `cargo run -p scrive-core --example transcript`
//!
//! Prints a scripted editing session step by step so the transaction and undo
//! semantics can be eyeballed without any UI in the loop: each line shows the
//! revision, the dirty/clean flag, and the full buffer text after one action.

use scrive_core::{Document, EditOp, GroupingHint, OpClass};

fn show(step: &str, doc: &Document) {
    let dirty = if doc.is_dirty() { "dirty" } else { "clean" };
    println!(
        "{step:<28} rev {:<2} [{dirty}]  {:?}",
        doc.revision().0,
        doc.text()
    );
}

fn main() {
    let mut doc = Document::new("").unwrap();
    show("open empty", &doc);

    for (i, ch) in "hello world".chars().enumerate() {
        doc.edit_grouped(
            vec![EditOp::insert(i as u32, ch.to_string())],
            GroupingHint::mergeable(OpClass::Type),
        )
        .unwrap();
    }
    show("type \"hello world\"", &doc);

    doc.mark_saved();
    show("save", &doc);

    doc.edit(vec![EditOp::new(0..1, "H"), EditOp::new(6..7, "W")]).unwrap();
    show("capitalize (2 ranges)", &doc);

    doc.undo();
    show("undo caps", &doc);

    doc.redo();
    show("redo caps", &doc);

    doc.undo();
    doc.undo();
    show("undo caps + typing run", &doc);

    println!("\n(one typing run undoes as a single unit; undo back to the save");
    println!(" point reads clean; a divergent edit would clear the redo branch.)");
}
