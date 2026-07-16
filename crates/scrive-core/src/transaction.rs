//! The atomic multi-range transaction engine.
//!
//! Every change to a [`Buffer`] — typing, paste, undo, redo, programmatic —
//! goes through `apply`: one mutation path, so an inverse can never drift
//! from what was applied — the inverse is derived mechanically from the same
//! normalized batch, never hand-written. The contract, in order:
//!
//! 1. clip each op's range to the pre-edit document (char boundaries) and
//!    normalize its text to LF;
//! 2. stable-sort ops ascending by start, ties in caller order (so two inserts
//!    at one offset apply in the order the caller gave them);
//! 3. overlap is a programmer error → `Err` (debug-panics);
//! 4. capture each op's replaced text *before* mutating, and build both the
//!    forward [`Patch`] and the inverse ops (delta-chained on the sorted list);
//! 5. apply descending so offsets stay valid; drop no-ops; an empty batch is no
//!    transaction at all (the revision does not move).
//!
//! Undo and redo are just `apply` fed the inverse ops — the
//! `Committed::inverse_ops` of one call are the input of the next.

use std::ops::Range;

use crate::buffer::Buffer;
use crate::coords::Bias;
use crate::patch::{Edit, Patch};

/// One replacement: put `text` where `range` currently is. `range` is in
/// pre-edit byte coordinates; an empty `range` with empty `text` is a no-op and
/// is dropped.
///
/// (A `cursor: Option<CursorPolicy>` field will join this once selections exist
/// to position against; it has no consumer yet, so it is not built.)
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct EditOp {
    /// Pre-edit byte range to replace.
    pub range: Range<u32>,
    /// Replacement text (normalized to LF at the boundary).
    pub text: String,
}

impl EditOp {
    /// Replace `range` with `text`.
    #[must_use]
    pub fn new(range: Range<u32>, text: impl Into<String>) -> Self {
        Self { range, text: text.into() }
    }

    /// Insert `text` at `at`.
    #[must_use]
    pub fn insert(at: u32, text: impl Into<String>) -> Self {
        Self { range: at..at, text: text.into() }
    }

    /// Delete `range`.
    #[must_use]
    pub fn delete(range: Range<u32>) -> Self {
        Self { range, text: String::new() }
    }
}

/// The result of a committed transaction.
#[derive(Clone, Debug, Default)]
pub struct Committed {
    patch: Patch,
    inverse: Vec<EditOp>,
    /// The applied (forward) ops — normalized (clipped, CRLF-folded, sorted,
    /// no-ops dropped). Replaying them reproduces this edit, so history stores
    /// them for redo. Present on the value `apply` returns; the `Committed`
    /// handed back to callers keeps only the [`patch`](Self::patch) (the ops
    /// move into history).
    forward: Vec<EditOp>,
}

impl Committed {
    /// The forward position-mapping currency for this transaction.
    #[must_use]
    pub fn patch(&self) -> &Patch {
        &self.patch
    }

    /// A patch-only `Committed` — what [`Document::edit`](crate::Document::edit)
    /// returns to callers after the op batches move into history (the verbs layer
    /// reads only the patch, so carrying the ops here would just be a clone kept
    /// alive for nobody).
    pub(crate) fn from_patch(patch: Patch) -> Self {
        Self { patch, inverse: Vec::new(), forward: Vec::new() }
    }

    /// Whether the transaction changed nothing (all ops were no-ops / the batch
    /// was empty). Derived from the patch (not the inverse), so it stays correct
    /// on the patch-only value returned after the ops move into history.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.patch.is_empty()
    }

    /// Borrow the ops that undo this transaction, in post-edit coordinates.
    /// Feeding them back to `apply` restores the pre-edit buffer (and yields
    /// the redo ops as *its* inverse).
    #[must_use]
    pub(crate) fn inverse_ops(&self) -> &[EditOp] {
        &self.inverse
    }

    /// Borrow the applied (forward) ops — the normalized batch that produced this
    /// edit. Used to classify the edit (e.g. did it touch a bracket char) without
    /// re-cloning the caller's original `ops`.
    #[must_use]
    pub(crate) fn forward_ops(&self) -> &[EditOp] {
        &self.forward
    }

    /// Consume this `Committed`, yielding `(patch, forward, inverse)` so the
    /// history can *own* the two op batches with no clone while the caller keeps
    /// the patch. The one place that needs owned ops (redo replay + undo).
    pub(crate) fn into_ops(self) -> (Patch, Vec<EditOp>, Vec<EditOp>) {
        (self.patch, self.forward, self.inverse)
    }
}

/// A transaction could not be applied.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TransactionError {
    /// Two ops in the batch overlap in the pre-edit document — a programmer
    /// error (the cursor layer must produce disjoint ops).
    #[error("overlapping edit ranges: {first:?} and {second:?}")]
    Overlap {
        /// The earlier (by start) range.
        first: Range<u32>,
        /// The range that overlapped it.
        second: Range<u32>,
    },
    /// The batch would grow the document past the u32 offset space — the
    /// representational bound (offsets are `u32`), not a policy limit. There is
    /// no separate document-size cap, so a large paste is what reaches this.
    #[error("edit would grow the document to {len} bytes, past the u32 offset space")]
    WouldOverflow {
        /// The post-edit length the batch would have produced.
        len: u64,
    },
}

/// The post-edit document length for a normalized (disjoint, clipped) batch.
/// Pure, so the overflow guard is unit-testable without allocating gigabytes
/// (the same trick as the load guard in `buffer.rs`).
fn post_len(current: u32, ops: &[EditOp]) -> u64 {
    let delta: i64 =
        ops.iter().map(|op| op.text.len() as i64 - (op.range.end - op.range.start) as i64).sum();
    let len = current as i64 + delta;
    debug_assert!(len >= 0, "deletes cannot exceed the document");
    len as u64
}

/// Apply a batch of [`EditOp`]s to `buffer` atomically, returning the
/// [`Committed`] result (forward patch + inverse ops + change records).
///
/// See the module docs for the six-step contract. The revision advances by
/// exactly one iff at least one op actually changed text; an empty or all-no-op
/// batch leaves the buffer and revision untouched.
///
/// `pub(crate)`: the public choke point is [`Document::edit`](crate::Document::edit),
/// which is the only way to mutate text without bypassing history.
pub(crate) fn apply(buffer: &mut Buffer, ops: Vec<EditOp>) -> Result<Committed, TransactionError> {
    // (1) clip ranges to the pre-edit doc + normalize text; drop no-ops.
    let mut norm: Vec<EditOp> = Vec::with_capacity(ops.len());
    for op in ops {
        let (a, b) = (op.range.start.min(op.range.end), op.range.start.max(op.range.end));
        let start = buffer.clip_offset(a, Bias::Left);
        let end = buffer.clip_offset(b, Bias::Right);
        let text = if op.text.as_bytes().contains(&b'\r') {
            op.text.replace("\r\n", "\n").replace('\r', "\n")
        } else {
            op.text
        };
        if start == end && text.is_empty() {
            continue; // no-op
        }
        norm.push(EditOp { range: start..end, text });
    }
    if norm.is_empty() {
        return Ok(Committed::default()); // empty batch → no transaction, no bump
    }

    // (2) stable-sort ascending by start; ties keep caller order.
    norm.sort_by_key(|op| op.range.start);

    // (3) reject overlap (touching is allowed).
    for w in norm.windows(2) {
        if w[0].range.end > w[1].range.start {
            return Err(TransactionError::Overlap {
                first: w[0].range.clone(),
                second: w[1].range.clone(),
            });
        }
    }

    // (3b) the representational bound: refuse growth past the u32 offset
    // space BEFORE any mutation, so a failed batch changes nothing. Offsets are
    // u32, so this is the only size gate.
    let len = post_len(buffer.len(), &norm);
    if len >= u32::MAX as u64 {
        return Err(TransactionError::WouldOverflow { len });
    }

    // (4) capture old text, build the forward patch and inverse ops.
    let mut patch = Patch::new();
    let mut inverse = Vec::with_capacity(norm.len());
    let mut delta: i64 = 0;
    for op in &norm {
        let (s, e) = (op.range.start, op.range.end);
        let old_text = buffer.slice(s..e).into_owned();
        let ns = (s as i64 + delta) as u32; // post-edit start
        let ne = ns + op.text.len() as u32; // post-edit end
        patch.push(Edit { old: s..e, new: ns..ne });
        // Inverse: put the old text back over the post-edit range (moves `old_text`).
        inverse.push(EditOp { range: ns..ne, text: old_text });
        delta += op.text.len() as i64 - (e - s) as i64;
    }

    // (5) apply all edits in ONE batched rope pass (the multi-caret path: one spine
    // rebuild sharing every untouched subtree, not N sequential splices). `norm` is
    // sorted ascending and disjoint — exactly `edit_many`'s contract. The `batch`
    // borrows `norm`, so it lives in its own scope and drops before `norm` moves
    // into the returned `Committed` as the forward ops.
    {
        let batch: Vec<(Range<u32>, &str)> =
            norm.iter().map(|op| (op.range.clone(), op.text.as_str())).collect();
        buffer.edit_many(&batch);
    }
    buffer.bump_revision();

    Ok(Committed { patch, inverse, forward: norm })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(s: &str) -> Buffer {
        Buffer::new(s).unwrap()
    }

    #[test]
    fn single_insert() {
        let mut b = buf("ac");
        let c = apply(&mut b, vec![EditOp::insert(1, "b")]).unwrap();
        assert_eq!(b.text(), "abc");
        assert_eq!(b.revision().0, 1);
        assert!(!c.is_empty());
    }

    #[test]
    fn single_delete_and_replace() {
        let mut b = buf("hello");
        apply(&mut b, vec![EditOp::delete(1..3)]).unwrap();
        assert_eq!(b.text(), "hlo");
        apply(&mut b, vec![EditOp::new(0..1, "H")]).unwrap();
        assert_eq!(b.text(), "Hlo");
        assert_eq!(b.revision().0, 2);
    }

    #[test]
    fn multi_range_batch_is_atomic() {
        // Replace both "a"s in "a b a" in one transaction, pre-edit coords.
        let mut b = buf("a b a");
        apply(&mut b, vec![EditOp::new(0..1, "X"), EditOp::new(4..5, "Y")]).unwrap();
        assert_eq!(b.text(), "X b Y");
        assert_eq!(b.revision().0, 1); // one transaction, one bump
    }

    #[test]
    fn same_offset_insertions_apply_in_caller_order() {
        let mut b = buf("");
        apply(&mut b, vec![EditOp::insert(0, "A"), EditOp::insert(0, "B")]).unwrap();
        assert_eq!(b.text(), "AB");
    }

    #[test]
    fn overlap_is_rejected() {
        let mut b = buf("abcdef");
        let err = apply(&mut b, vec![EditOp::new(0..3, "x"), EditOp::new(2..5, "y")]);
        assert!(matches!(err, Err(TransactionError::Overlap { .. })));
        assert_eq!(b.text(), "abcdef"); // unchanged on error
        assert_eq!(b.revision().0, 0);
    }

    #[test]
    fn post_len_overflow_guard_is_pure() {
        // The u32-offset guard is length arithmetic only — testable without
        // allocating gigabytes. apply() refuses when this reaches u32::MAX.
        let ops = vec![EditOp::new(0..3, "xxxxx"), EditOp::delete(5..9)];
        assert_eq!(post_len(20, &ops), 20 + 2 - 4);
        assert_eq!(post_len(0, &[]), 0);
        let huge = vec![EditOp::insert(0, "x".repeat(16))];
        assert!(post_len(u32::MAX - 8, &huge) >= u32::MAX as u64, "this batch would be refused");
    }

    #[test]
    fn empty_and_noop_batches_do_not_move_the_revision() {
        let mut b = buf("abc");
        assert!(apply(&mut b, vec![]).unwrap().is_empty());
        assert!(apply(&mut b, vec![EditOp::new(1..1, "")]).unwrap().is_empty());
        assert_eq!(b.revision().0, 0);
        assert_eq!(b.text(), "abc");
    }

    #[test]
    fn text_is_normalized_to_lf() {
        let mut b = buf("");
        apply(&mut b, vec![EditOp::insert(0, "a\r\nb")]).unwrap();
        assert_eq!(b.text(), "a\nb");
        assert!(!b.text().contains('\r'));
        assert_eq!(b.line_count(), 2);
    }

    #[test]
    fn inverse_round_trips_text_and_revision_parity() {
        // apply → apply(inverse) restores the exact pre-edit text.
        let mut b = buf("the quick brown fox");
        let before = b.text().into_owned();
        let committed =
            apply(&mut b, vec![EditOp::new(4..9, "SLOW"), EditOp::insert(0, ">> ")]).unwrap();
        assert_ne!(b.text(), before);
        let redo = apply(&mut b, committed.inverse_ops().to_vec()).unwrap();
        assert_eq!(b.text(), before, "inverse must restore the pre-edit text");
        assert_eq!(b.revision().0, 2); // forward + undo are both transactions
        // And the inverse-of-the-inverse (redo) reapplies the change.
        apply(&mut b, redo.inverse_ops().to_vec()).unwrap();
        assert_ne!(b.text(), before);
    }

    #[test]
    fn patch_maps_positions_across_the_transaction() {
        // "hello world" → delete "hello " (0..6): a marker at 'w' (offset 6)
        // maps to offset 0.
        let mut b = buf("hello world");
        let c = apply(&mut b, vec![EditOp::delete(0..6)]).unwrap();
        assert_eq!(b.text(), "world");
        assert_eq!(c.patch().map_offset(6, Bias::Left), 0);
        assert_eq!(c.patch().map_offset(8, Bias::Left), 2); // 'r'
    }
}
