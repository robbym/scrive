//! The opt-in incremental change log behind
//! [`Document::drain_changes`](crate::Document::drain_changes).
//!
//! **Contract:** replaying every drained op, in drain order, onto a copy of the
//! text the host last saw yields exactly `Document::text()` — each op's range is
//! in the coordinates of the document *as left by the previous op* (the shape an
//! LSP `didChange` content-change array has). That must hold across forward
//! edits, undo, redo, and merged undo units alike, so the log is fed from the
//! one place all three routes pass through per applied step: `rebase_views`,
//! the single commit-path mover (it holds a `&mut ChangeLog` in its `Views`).
//! Feeding it from `edit_grouped` alone is what let undo/redo go unlogged.
//!
//! **Order within one step.** A step's applied ops are the transaction engine's
//! normalized batch: ascending by start, disjoint (touching allowed), ties in
//! the caller order the batch was applied in. Sequential replay must run them
//! back to front — each op then lands at or before everything already replayed,
//! so its pre-step range is still exact. That is a plain **reversal**, never a
//! re-sort: a stable descending sort keeps tied starts in *ascending* order,
//! which replays `[insert(2,"A"), insert(2,"B")]` as `"BA"`, and replays
//! `[insert(2,"X"), delete(2..4)]` insert-first so the delete eats the new `X`.
//!
//! **Zero overhead when off.** [`ChangeLog::record`] is one branch on the
//! flag before anything else; an unobserved document never clones an op or
//! allocates the backing `Vec` (pinned by
//! `unobserved_edits_never_allocate_a_log`).

use crate::transaction::{Committed, EditOp};

/// The incremental-change log: off by default, and when on, every applied
/// step's ops in sequential-replay order (see the module docs).
#[derive(Debug, Default)]
pub(crate) struct ChangeLog {
    observing: bool,
    ops: Vec<EditOp>,
}

impl ChangeLog {
    /// Turn observation on or off. Off also drops any pending ops *and* their
    /// allocation, so an unobserved log holds no memory.
    pub(crate) fn set_observing(&mut self, on: bool) {
        self.observing = on;
        if !on {
            self.ops = Vec::new();
        }
    }

    /// Append one applied step — forward edit, undo step, or redo step — in
    /// sequential-replay order: the step's ascending, tie-ordered batch
    /// reversed (see the module docs for why reversal, not a sort).
    pub(crate) fn record(&mut self, committed: &Committed) {
        if !self.observing {
            return; // the common path: no clone, no allocation
        }
        self.ops.extend(committed.forward_ops().iter().rev().cloned());
    }

    /// Take everything logged since the last drain, in replay order.
    pub(crate) fn drain(&mut self) -> Vec<EditOp> {
        core::mem::take(&mut self.ops)
    }
}

#[cfg(test)]
mod tests {
    use crate::{Document, EditOp, GroupingHint, OpClass};

    /// Replay drained ops, in drain order, onto the host's copy of the text —
    /// exactly what a language-server client does with a `didChange`
    /// content-change array (each change applies to the result of the last).
    fn replay(mirror: &mut String, ops: &[EditOp]) {
        for op in ops {
            mirror.replace_range(op.range.start as usize..op.range.end as usize, &op.text);
        }
    }

    /// A document observed from load, plus the host's mirror of what it saw.
    fn observed(text: &str) -> (Document, String) {
        let mut d = Document::new(text).unwrap();
        d.observe_changes(true);
        let mirror = d.text().into_owned();
        (d, mirror)
    }

    /// Drain, replay onto the mirror, and require the mirror to equal the
    /// document — the change log's whole contract.
    fn sync(d: &mut Document, mirror: &mut String, at: &str) {
        let ops = d.drain_changes();
        replay(mirror, &ops);
        assert_eq!(*mirror, d.text(), "mirror drifted after {at}; drained {ops:?}");
    }

    #[test]
    fn undo_and_redo_are_logged() {
        // Fails-first: the log was fed only from `edit_grouped`, so undo/redo
        // drained `[]` and the mirror kept the undone " world".
        let (mut d, mut mirror) = observed("hello\n");
        d.edit(vec![EditOp::insert(5, " world")]).unwrap();
        sync(&mut d, &mut mirror, "the edit");

        assert!(d.undo());
        let undo = d.drain_changes();
        assert_eq!(undo, vec![EditOp::delete(5..11)], "undo logs the revert");
        replay(&mut mirror, &undo);
        assert_eq!(mirror, "hello\n");

        assert!(d.redo());
        let redo = d.drain_changes();
        assert_eq!(redo, vec![EditOp::insert(5, " world")], "redo logs the replay");
        replay(&mut mirror, &redo);
        assert_eq!(mirror, "hello world\n");
        assert_eq!(mirror, d.text());
    }

    #[test]
    fn same_offset_inserts_replay_in_applied_order() {
        // Fails-first: a stable descending sort kept the tie in ascending caller
        // order, so replay inserted "A" in front of the already-inserted "B".
        let (mut d, mut mirror) = observed("abcd");
        d.edit(vec![EditOp::insert(2, "A"), EditOp::insert(2, "B")]).unwrap();
        assert_eq!(d.text(), "abABcd");
        sync(&mut d, &mut mirror, "two inserts at one offset");
    }

    #[test]
    fn insert_then_delete_at_one_offset_replays_in_applied_order() {
        // Fails-first: replay ran the insert first, so the delete then removed
        // "Xc" instead of "cd".
        let (mut d, mut mirror) = observed("abcdef");
        d.edit(vec![EditOp::insert(2, "X"), EditOp::delete(2..4)]).unwrap();
        assert_eq!(d.text(), "abXef");
        sync(&mut d, &mut mirror, "an insert and a delete sharing a start");
    }

    #[test]
    fn unobserved_edits_never_allocate_a_log() {
        // The zero-overhead promise: with observation off, no route (edit, undo,
        // redo) clones an op or allocates the log. `drain` hands back the backing
        // Vec itself, so its capacity is the witness.
        let mut d = Document::new("abc\n").unwrap();
        d.edit(vec![EditOp::insert(1, "x"), EditOp::delete(2..3)]).unwrap();
        assert!(d.undo());
        assert!(d.redo());
        assert_eq!(d.drain_changes().capacity(), 0, "a never-observed document allocated a log");

        // Turning observation off releases what it held, and stays silent after.
        d.observe_changes(true);
        d.edit(vec![EditOp::insert(0, "y")]).unwrap();
        d.observe_changes(false);
        d.edit(vec![EditOp::insert(0, "z")]).unwrap();
        assert!(d.undo());
        assert!(d.redo());
        assert_eq!(d.drain_changes().capacity(), 0, "an unobserved document kept or grew a log");
    }

    /// A seeded xorshift — deterministic, no dependency.
    fn xorshift(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    /// A random valid multi-op batch over a `len`-byte document, handed over in
    /// a shuffled caller order. Cut points are drawn mostly from two shared
    /// anchors, so equal starts — several inserts at one offset, an insert
    /// before a delete/replace at that offset — are the common case rather than
    /// a rarity. Pairing *sorted* cut points keeps the batch disjoint (touching
    /// allowed) and puts every empty range before the non-empty one sharing its
    /// start, the only tie order the engine accepts.
    fn random_batch(next: &mut impl FnMut() -> u64, len: u32) -> Vec<EditOp> {
        // CRLF and a two-byte char make the logged ops prove they carry the
        // engine's *normalized* batch (LF text, char-boundary-clipped ranges).
        const PIECES: &[&str] = &["a", "b", "{", "}", "\n", " ", "é", "\r\n"];
        let pos = |next: &mut dyn FnMut() -> u64| (next() % u64::from(len + 1)) as u32;
        let anchors = [pos(next), pos(next)];
        let n = 1 + (next() % 4) as usize;
        let mut cuts: Vec<u32> = (0..2 * n)
            .map(|_| if next().is_multiple_of(3) { pos(next) } else { anchors[(next() % 2) as usize] })
            .collect();
        cuts.sort_unstable();
        let ops: Vec<EditOp> = cuts
            .chunks(2)
            .map(|c| {
                // 0 pieces ⇒ a pure delete (or, on an empty range, a dropped no-op).
                let text: String =
                    (0..next() % 3).map(|_| PIECES[(next() % PIECES.len() as u64) as usize]).collect();
                EditOp::new(c[0]..c[1], text)
            })
            .collect();
        // Shuffle whole runs of equal start (keeping each run's internal order,
        // which is what decides a tie) so the engine's sort does real work.
        let mut runs: Vec<Vec<EditOp>> = Vec::new();
        for op in ops {
            match runs.last_mut() {
                Some(run) if run[0].range.start == op.range.start => run.push(op),
                _ => runs.push(vec![op]),
            }
        }
        for i in (1..runs.len()).rev() {
            runs.swap(i, (next() % (i as u64 + 1)) as usize);
        }
        runs.into_iter().flatten().collect()
    }

    #[test]
    fn replayed_log_equals_text_across_random_batches_and_undo_redo() {
        // The change-log oracle: interleave random multi-op batches (merging and
        // discrete undo units) with undo, redo, and redo-branch selection; after
        // EVERY step the drained log, replayed onto the host's mirror, must equal
        // the live text.
        let mut next = xorshift(0x5EED_C4A6_E10C_0001);
        let (mut d, mut mirror) = observed("fn main() {\n    let s = \"é\";\n}\n");
        let (mut undos, mut redos, mut forks, mut batches) = (0, 0, 0, 0);
        for step in 0..1_500 {
            match next() % 10 {
                0..=1 => {
                    for _ in 0..1 + next() % 3 {
                        undos += usize::from(d.undo());
                    }
                }
                // Redo only succeeds right after an undo, so it gets the
                // heavier weight to keep both routes well exercised.
                2..=4 => {
                    let branches = d.redo_branch_count();
                    if branches > 1 {
                        assert!(d.select_redo_branch((next() % branches as u64) as usize));
                        forks += 1;
                    }
                    redos += usize::from(d.redo());
                }
                _ => {
                    let ops = random_batch(&mut next, d.buffer().len());
                    let hint = match next() % 3 {
                        0 => GroupingHint::discrete(),
                        1 => GroupingHint::mergeable(OpClass::Type),
                        _ => GroupingHint::mergeable(OpClass::Delete),
                    };
                    // Mid-char cut points can clip into an overlap, which the
                    // engine rejects whole — then nothing applied, so nothing
                    // may be logged (the sync below checks that too).
                    batches += usize::from(d.edit_grouped(ops, hint).is_ok_and(|c| !c.is_empty()));
                }
            }
            sync(&mut d, &mut mirror, &format!("step {step}"));
        }
        // Guard against a vacuous walk: every route was really exercised.
        // (This seed: 675 applied batches, 630 undos, 134 redos, 41 forks.)
        assert!(
            batches > 400 && undos > 300 && redos > 80 && forks > 20,
            "walk too thin: {batches} batches / {undos} undos / {redos} redos / {forks} forks",
        );
    }
}
