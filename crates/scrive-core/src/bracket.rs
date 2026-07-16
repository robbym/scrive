//! Bracket matching — the shared structural pass behind bracket-pair
//! colorization, matching-bracket highlight, indent guides, and folding.
//! `()`/`[]`/`{}` pair by a stack automaton (quotes are not nestable), yielding
//! each bracket's nesting depth and matched-partner offset and flagging the
//! unmatched.
//!
//! Matches **all** brackets regardless of string or comment context: this is a
//! purely structural scan with no lexer state, so a `)` inside a string still
//! counts as a bracket. Excluding string/comment brackets would need a scope
//! class threaded through from the highlight layer, which carries only a
//! resolved color, not lexical context.
//!
//! [`Brackets`] is a thin façade over one `SumTree<BracketItem>` (see
//! `bracket_tree`) that stores only PRIMARY facts — each bracket's char and
//! delta-gap position. Every derived fact a caller reads — `open`, `depth`,
//! `partner`, "what pair encloses this" — is recomputed by an O(log) cursor
//! descent, so no cached copy can drift, and an edit is a pure
//! `SumTree::replace` of the edited byte range with
//! the rescanned brackets. Because no partner offset is stored, there is nothing
//! to repair across the edit boundary — the tree is always internally
//! consistent by construction. [`Brackets::match_text`] is the O(n) load-time
//! constructor (and the tests' oracle); [`Brackets::apply_edit`] rides one
//! committed patch through the same tree, on the forward path and on every
//! undo/redo step alike.

use std::ops::Range;

use crate::bracket_tree::{self, BracketItem};
use crate::buffer::Buffer;
use crate::patch::Patch;
use crate::sum_tree::SumTree;

// Op-count canary: how many `enclosing_or_touching` walks ran on this thread. A
// document-scale multi-caret edit must trigger O(1) of them inside
// `expand_folds_touched` (one, for the first edit point), NOT one per caret, so a
// select-all → fold → type sequence stays cheap regardless of caret count.
// Debug/test only; zero-cost in release.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static ENCLOSING_WALKS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Whether `c` is any bracket char (opener or closer), as a byte — the document's
/// reconcile gate ("did this edit add or remove a bracket?").
pub(crate) fn is_bracket_byte(c: u8) -> bool {
    matches!(c, b'(' | b')' | b'[' | b']' | b'{' | b'}')
}

/// One bracket occurrence in the document — fully derived on demand (nothing here
/// is stored; the tree holds only the char and position).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Bracket {
    /// Byte offset of the bracket char.
    pub offset: u32,
    /// Whether it is an opener (`([{`) vs a closer (`)]}`).
    pub open: bool,
    /// Nesting depth (0 = outermost).
    pub depth: u32,
    /// The matched partner's offset, or `None` if unmatched.
    pub partner: Option<u32>,
}

/// Every bracket in a document, riding one augmented `SumTree` that is the sole
/// owner of position. Queries are O(log) descents; an edit is a
/// `SumTree::replace` of the edited byte range.
///
/// `Default` is an EMPTY tree — valid both for the read-only query surface
/// (movement's fold-free helper pairs it with an empty `FoldSet`) and as a base
/// for [`Brackets::apply_edit`], since the tree needs no scaffold to seed: the
/// first edit simply replaces an empty range.
#[derive(Clone, Debug, Default)]
pub struct Brackets {
    tree: SumTree<BracketItem>,
}

impl Brackets {
    /// Match every `()`/`[]`/`{}` bracket in `text` (LF-only — the buffer
    /// contract) — the O(n) from-scratch constructor (document load; the tests'
    /// oracle). Per-edit maintenance goes through [`Brackets::apply_edit`].
    #[must_use]
    pub fn match_text(text: &str) -> Self {
        Self { tree: bracket_tree::tree_from_text(text) }
    }

    /// All brackets, in document order — the O(n) cold path (fold-source scans, the
    /// capture-folds harness, tests). Hot paths use [`Self::in_range`].
    #[must_use]
    pub fn all(&self) -> Vec<Bracket> {
        bracket_tree::derive_all(&self.tree)
    }

    /// The bracket at exactly `offset`, if any. O(log).
    #[must_use]
    pub fn at(&self, offset: u32) -> Option<Bracket> {
        bracket_tree::at(&self.tree, offset)
    }

    /// The matched partner of the OPEN bracket at `offset`, or `None` if it is not a
    /// live opener or is unmatched — the fold hot-path lookup (fold reconcile,
    /// expand-on-hidden-edit, block/inline resolution). Unlike [`Self::at`] it never
    /// resolves `depth`, so it costs O(log + partner fold) with no `ShapeSummary`
    /// allocation — the difference between a fold-heavy keystroke being O(folds·log)
    /// and an allocation storm. Callers that only need "is this opener's pair still
    /// there, and where does it close?" must use this, never `at().partner`.
    #[must_use]
    pub fn foldable_partner(&self, offset: u32) -> Option<u32> {
        bracket_tree::foldable_partner(&self.tree, offset)
    }

    /// The brackets whose offset lies in `range` (half-open: start inclusive, end
    /// exclusive), in order, collected — the cold/slice callers (fold-pair
    /// derivation, tests). An inverted range is empty, not a panic. Per-frame hot
    /// paths iterate [`Self::in_range_iter`] instead, which allocates nothing.
    #[must_use]
    pub fn in_range(&self, range: Range<u32>) -> Vec<Bracket> {
        self.in_range_iter(range).collect()
    }

    /// [`Self::in_range`] as a lazy iterator — the per-visible-row colorization
    /// loop iterates this directly, so a frame's bracket colouring allocates no
    /// `Vec` (one per visible row × every frame otherwise). An inverted range
    /// yields nothing (the count bounds cross), matching [`Self::in_range`].
    pub fn in_range_iter(&self, range: Range<u32>) -> impl Iterator<Item = Bracket> + '_ {
        bracket_tree::in_range(&self.tree, range.start, range.end)
    }

    /// The matched bracket pair to highlight for a caret at `caret`, as
    /// `(bracket_offset, partner_offset)`. The bracket directly *left* of the
    /// caret is preferred (the one you just typed past); the one directly right
    /// is the fallback. `None` if the caret isn't adjacent to a *matched* bracket.
    #[must_use]
    pub fn active_pair(&self, caret: u32) -> Option<(u32, u32)> {
        bracket_tree::active_pair(&self.tree, caret)
    }

    /// The innermost matched pair whose interior contains `caret` (opener offset
    /// `< caret <=` closer offset), as `(open_offset, close_offset)` — the *active*
    /// indent guide. `None` if the caret is inside no matched pair.
    #[must_use]
    pub fn enclosing_pair(&self, caret: u32) -> Option<(u32, u32)> {
        bracket_tree::enclosing_pairs(&self.tree, caret).into_iter().next()
    }

    /// The opener of the innermost pair strictly enclosing `caret`
    /// (`open < caret <= close`) that satisfies `pred`. `None` if none matches.
    #[must_use]
    pub fn innermost_enclosing_where(&self, caret: u32, pred: impl Fn(u32, u32) -> bool) -> Option<u32> {
        bracket_tree::enclosing_pairs(&self.tree, caret)
            .into_iter()
            .find(|&(o, c)| pred(o, c))
            .map(|(o, _)| o)
    }

    /// The innermost matched pair whose CONTENTS contain the whole range —
    /// `open + 1 <= start && end <= close` — as `(open, close)`. The
    /// expand-selection ladder's next rung. `None` outside every pair.
    #[must_use]
    pub fn enclosing_pair_of_range(&self, start: u32, end: u32) -> Option<(u32, u32)> {
        // The enclosers of `start` are innermost-first (closes widen outward), so
        // the first whose closer also reaches `end` is the tightest fit.
        bracket_tree::enclosing_pairs(&self.tree, start).into_iter().find(|&(_, close)| end <= close)
    }

    /// Every matched pair `(open, close)` with `open <= caret <= close + 1` —
    /// the pairs enclosing the caret PLUS the ones it touches at either bracket
    /// (caret AT an opener, or just past a closer) — the fold-at-caret candidate
    /// set. Unordered; the caller picks the innermost.
    #[must_use]
    pub fn enclosing_or_touching(&self, caret: u32) -> Vec<(u32, u32)> {
        #[cfg(any(test, debug_assertions))]
        ENCLOSING_WALKS.with(|c| c.set(c.get() + 1));
        bracket_tree::enclosing_or_touching_pairs(&self.tree, caret)
    }

    /// Splice this structure through one committed `patch` — the incremental twin
    /// of [`Brackets::match_text`]. `buffer` is the POST-edit buffer and `patch`
    /// the committed transaction's old→new byte ranges (`new` in final
    /// coordinates) — exactly what `rebase_views` holds, on the forward path and on
    /// every undo/redo step alike.
    ///
    /// Returns the reconcile window `[seed_lo, re)` in post-edit coordinates: the
    /// re-matched span extended left to the outermost opener enclosing the first
    /// edit (so a fold whose partner the edit deletes is covered), or `0..0` when
    /// no bracket structure changed. `Document::reconcile_folds` re-checks only the
    /// folds in that window.
    ///
    /// # Why the result deep-equals a from-scratch rebuild
    ///
    /// The tree stores only each bracket's char and delta-gap position; `open`,
    /// `depth`, and `partner` are DERIVED per query from the always-current tree.
    /// So an edit need only make the stored primary facts correct: rescan the
    /// edited byte range into brackets, `SumTree::replace` them in, and — because
    /// the encoding is delta-gap — re-anchor just the FIRST suffix bracket's gap
    /// (the rest are relative and do not move). No cross-boundary partner pointer
    /// is stored, so none can be left stale by the splice; the
    /// incremental≡`match_text` oracle enforces the equivalence.
    pub fn apply_edit(&mut self, patch: &Patch, buffer: &Buffer) -> Range<u32> {
        let (tree, region) = bracket_tree::apply_edit(&self.tree, patch, buffer);
        self.tree = tree;
        region
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Point;
    use crate::transaction::{apply, EditOp};

    fn bat(b: &Brackets, o: u32) -> Bracket {
        b.at(o).expect("a bracket at that offset")
    }

    #[test]
    fn matches_nested_pairs_with_depth() {
        let b = Brackets::match_text("(a[b]{c})"); // ( [ ] { } ) at 0 2 4 5 7 8
        assert_eq!((bat(&b, 0).partner, bat(&b, 0).depth), (Some(8), 0));
        assert_eq!((bat(&b, 2).partner, bat(&b, 2).depth), (Some(4), 1));
        assert_eq!((bat(&b, 5).partner, bat(&b, 5).depth), (Some(7), 1));
        assert_eq!((bat(&b, 8).partner, bat(&b, 8).depth), (Some(0), 0));
    }

    #[test]
    fn flags_unmatched_brackets() {
        let b = Brackets::match_text("(]"); // opener never closes, closer has no opener
        assert_eq!(bat(&b, 0).partner, None);
        assert_eq!(bat(&b, 1).partner, None);
    }

    #[test]
    fn active_pair_prefers_the_bracket_left_of_the_caret() {
        let b = Brackets::match_text("(){}"); // '(' 0↔1  ')' 1↔0  '{' 2↔3  '}' 3↔2
        assert_eq!(b.active_pair(0), Some((0, 1))); // no left → right '(' at 0
        assert_eq!(b.active_pair(1), Some((0, 1))); // left '(' at 0 (preferred over ')')
        assert_eq!(b.active_pair(2), Some((1, 0))); // left ')' at 1 (preferred over '{')
        assert_eq!(b.active_pair(3), Some((2, 3))); // left '{' at 2
        assert_eq!(b.active_pair(4), Some((3, 2))); // left '}' at 3
        assert_eq!(Brackets::match_text("x").active_pair(1), None); // none adjacent
    }

    #[test]
    fn enclosing_pair_is_the_innermost_containing_the_caret() {
        let b = Brackets::match_text("(a[b]{c})"); // ( [ ] { } ) at 0 2 4 5 7 8
        assert_eq!(b.enclosing_pair(1), Some((0, 8))); // inside only the outer ()
        assert_eq!(b.enclosing_pair(3), Some((2, 4))); // inside [] (depth 1) beats ()
        assert_eq!(b.enclosing_pair(6), Some((5, 7))); // inside {} (depth 1)
        assert_eq!(b.enclosing_pair(0), None); // at the opener → not yet inside
        assert_eq!(b.enclosing_pair(9), None); // past the closer
    }

    /// The enclosing queries must find the pair even when its opener is FAR to the
    /// left with many sibling pairs in between (the O(brackets)-scan case the tree
    /// descent replaces), and the touching cases (`caret == open`, `caret ==
    /// close + 1`). An `enclosing_*` result independent of the sibling count is the
    /// whole point.
    #[test]
    fn enclosing_walk_matches_a_brute_scan_with_siblings() {
        //           0         1         2
        //           0123456789012345678901234567
        let b = Brackets::match_text("([1][2]{a(b)c}[3])xy");
        // A brute-force oracle: innermost matched pair strictly containing c.
        let brute_enclosing = |caret: u32| {
            b.all()
                .iter()
                .filter_map(|br| br.partner.map(|p| (br.offset, p)).filter(|_| br.open))
                .filter(|&(o, c)| o < caret && caret <= c)
                .min_by_key(|&(o, c)| c - o)
        };
        for caret in 0..=20 {
            assert_eq!(b.enclosing_pair(caret), brute_enclosing(caret), "enclosing_pair({caret})");
        }
        // Deep inside the nested `(b)` at offset 10 — enclosed by (9,11), {7,13),
        // (0,17), reached past the [1] [2] siblings.
        assert_eq!(b.enclosing_pair(10), Some((9, 11)));
        // Just after the `}` at 13: caret 14 is back at the outer ( level.
        assert_eq!(b.enclosing_pair(14), Some((0, 17)));
        // enclosing_or_touching includes the touched pairs at either bracket.
        let mut touch = b.enclosing_or_touching(9); // caret ON the `(` at 9
        touch.sort_unstable();
        assert!(touch.contains(&(9, 11)), "the pair opening at the caret is touched");
        assert!(touch.contains(&(7, 13)) && touch.contains(&(0, 17)), "and its enclosers");
        let after_close = b.enclosing_or_touching(12); // caret just past `)` at 11
        assert!(after_close.contains(&(9, 11)), "the pair closing at caret-1 is touched");
        // enclosing_pair_of_range: the tightest pair whose CONTENTS hold [start,end].
        assert_eq!(b.enclosing_pair_of_range(10, 11), Some((9, 11))); // fits (b)
        assert_eq!(b.enclosing_pair_of_range(9, 12), Some((7, 13))); // spans (b) → next out
    }

    #[test]
    fn active_pair_skips_unmatched_brackets() {
        let b = Brackets::match_text("(]"); // both unmatched
        assert_eq!(b.active_pair(1), None); // left '(' unmatched, right ']' unmatched
        assert_eq!(b.active_pair(2), None);
    }

    #[test]
    fn quotes_are_not_brackets() {
        let b = Brackets::match_text("\"()\""); // quotes ignored; ( ) matched
        assert_eq!(b.all().len(), 2);
        assert_eq!(bat(&b, 1).partner, Some(2));
    }

    #[test]
    fn in_range_boundaries() {
        let b = Brackets::match_text("(a[b]{c})"); // ( [ ] { } ) at 0 2 4 5 7 8
        let offsets = |s: &[Bracket]| s.iter().map(|x| x.offset).collect::<Vec<_>>();
        assert!(b.in_range(0..0).is_empty()); // empty range
        assert!(b.in_range(3..3).is_empty());
        assert_eq!(b.in_range(0..9), b.all()); // full range == all
        assert_eq!(b.in_range(0..u32::MAX), b.all());
        assert_eq!(offsets(&b.in_range(2..5)), vec![2, 4]); // start inclusive…
        assert_eq!(offsets(&b.in_range(2..6)), vec![2, 4, 5]); // …end exclusive
        assert_eq!(offsets(&b.in_range(8..9)), vec![8]); // exactly the last one
        assert_eq!(offsets(&b.in_range(1..2)), Vec::<u32>::new()); // gap between brackets
        #[allow(clippy::reversed_empty_ranges)]
        let inverted = b.in_range(5..2);
        assert!(inverted.is_empty()); // inverted range is empty, not a panic
    }

    // ───────────────────────── incremental engine ─────────────────────────

    /// The oracle assertion: the incrementally-maintained structure must
    /// deep-equal a from-scratch `match_text` of the same text — every derived
    /// Bracket field. With no stored line shapes, `all()` (offset, open, depth,
    /// partner) IS the whole observable state.
    fn assert_oracle(b: &Brackets, buf: &Buffer, ctx: &str) {
        let oracle = Brackets::match_text(&buf.text());
        assert_eq!(b.all(), oracle.all(), "brackets diverged from a scratch rebuild: {ctx}");
    }

    /// Apply `ops` to the buffer and ride the committed patch through
    /// `apply_edit` — the exact call shape `rebase_views` will use.
    fn edit(b: &mut Brackets, buf: &mut Buffer, ops: Vec<EditOp>) -> Range<u32> {
        let committed = apply(buf, ops).expect("test edits are disjoint and in-bounds");
        b.apply_edit(committed.patch(), buf)
    }

    /// Oracle test: ~500 seeded-random edits (biased toward bracket chars,
    /// newlines, mid-pair splits, multi-char inserts/deletes and the occasional
    /// scattered two-edit transaction) over a bracket-heavy document; after every
    /// commit the incremental structure must deep-equal a from-scratch rebuild.
    /// This generator exercises every way an edit can re-pair brackets across the
    /// splice boundary, so any divergence between incremental and scratch matching
    /// surfaces here.
    #[test]
    fn incremental_matches_scratch_after_random_edits() {
        let seed: u64 = 0x5EED_0BAD_F00D_2026;
        let mut s = seed;
        let mut rng = move || {
            // xorshift64 — deterministic, no dev-dependency.
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        let mut buf = Buffer::new(
            "fn main() {\n    let v = vec![1, (2), [3]];\n    if (a[i]) { b(c[d]); }\n}\n",
        )
        .expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        const POOL: &[u8] = b"(){}[](){}[]\n\nab ";
        for i in 0..500 {
            let len = buf.len();
            let kind = rng() % 12;
            let ops = if kind < 5 {
                let at = (rng() % (u64::from(len) + 1)) as u32;
                let n = 1 + (rng() % 4) as usize;
                let text: String =
                    (0..n).map(|_| POOL[(rng() % POOL.len() as u64) as usize] as char).collect();
                vec![EditOp::insert(at, text)]
            } else if kind < 7 {
                let a = (rng() % (u64::from(len) + 1)) as u32;
                let e = (a + 1 + (rng() % 12) as u32).min(len);
                vec![EditOp::delete(a..e)]
            } else if kind < 9 {
                let a = (rng() % (u64::from(len) + 1)) as u32;
                let e = (a + (rng() % 6) as u32).min(len);
                let n = 1 + (rng() % 5) as usize;
                let text: String =
                    (0..n).map(|_| POOL[(rng() % POOL.len() as u64) as usize] as char).collect();
                vec![EditOp::new(a..e, text)]
            } else if kind == 9 {
                let a = (rng() % (u64::from(len) / 2 + 1)) as u32;
                let c = (len / 2 + (rng() % (u64::from(len) / 2 + 1)) as u32).min(len);
                vec![EditOp::insert(a, "("), EditOp::insert(c, ")")]
            } else {
                // Scattered structure-neutral inserts across the whole document —
                // the O(N log) per-edit splice path over a MULTI-edit patch; must
                // still deep-equal a scratch rebuild.
                let n = 2 + (rng() % 3) as usize; // 2..=4 carets
                let mut offs: Vec<u32> =
                    (0..n).map(|_| (rng() % (u64::from(len) + 1)) as u32).collect();
                offs.sort_unstable();
                offs.dedup();
                offs.into_iter().map(|o| EditOp::insert(o, "z")).collect()
            };
            let committed = apply(&mut buf, ops.clone()).expect("generated ops are disjoint");
            b.apply_edit(committed.patch(), &buf);
            let ctx = format!("edit {i} (seed {seed:#x}): {ops:?}");
            assert_oracle(&b, &buf, &ctx);
        }
    }

    // ── re-pairing cases: an edit changes which brackets pair with which. With
    //    partners DERIVED from the always-current tree, each is just an ordinary
    //    edit — the oracle confirms the derived partners follow the text ──

    #[test]
    fn prefix_opener_repoints_into_region() {
        // "{\nx\n}", line 1 → "}": the prefix `{` re-points to the NEW line-1 closer,
        // and the old line-2 `}` goes unmatched.
        let mut buf = Buffer::new("{\nx\n}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::new(2..3, "}")]); // "{\n}\n}"
        assert_oracle(&b, &buf, "line 1 x → }");
        assert_eq!(bat(&b, 0).partner, Some(2));
        assert_eq!(bat(&b, 2).partner, Some(0));
        assert_eq!((bat(&b, 4).partner, bat(&b, 4).depth), (None, 0));
    }

    #[test]
    fn converged_shape_different_identity() {
        // "{\nx\n}", line 1 → "}{": the shape after line 1 is again `{`, but the
        // identity behind it changed — the final `}` pairs the NEW region `{` at 3,
        // and the prefix `{` re-points to the new line-1 `}`.
        let mut buf = Buffer::new("{\nx\n}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::new(2..3, "}{")]); // "{\n}{\n}"
        assert_oracle(&b, &buf, "line 1 x → }{");
        assert_eq!(bat(&b, 0).partner, Some(2));
        assert_eq!(bat(&b, 3).partner, Some(5));
        assert_eq!(bat(&b, 5).partner, Some(3));
    }

    #[test]
    fn converged_shape_crosses_into_the_suffix() {
        // Same identity swap, but with a clean line between the edit and the final
        // `}` — the `}` is a true suffix entry that must re-pair the region opener.
        let mut buf = Buffer::new("{\nx\nq\n}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::new(2..3, "}{")]); // "{\n}{\nq\n}"
        assert_oracle(&b, &buf, "line 1 x → }{ with a clean row before the closer");
        assert_eq!(bat(&b, 0).partner, Some(2));
        assert_eq!(bat(&b, 3).partner, Some(7));
        assert_eq!(bat(&b, 7).partner, Some(3));
    }

    #[test]
    fn crossing_repairs_mixed_seed_and_region_entries() {
        // "{\n(\nb\n)}", line 1 → ")(": `)` pairs the region opener, `}` pairs the
        // seed `{`, and the new line-1 `)` stays unmatched.
        let mut buf = Buffer::new("{\n(\nb\n)}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::new(2..3, ")(")]); // "{\n)(\nb\n)}"
        assert_oracle(&b, &buf, "mixed carried stack");
        assert_eq!((bat(&b, 2).partner, bat(&b, 2).depth), (None, 0)); // unmatched )
        assert_eq!(bat(&b, 3).partner, Some(7)); // region ( ↔ suffix )
        assert_eq!(bat(&b, 7).partner, Some(3));
        assert_eq!(bat(&b, 0).partner, Some(8)); // seed { ↔ suffix }
        assert_eq!(bat(&b, 8).partner, Some(0));
    }

    #[test]
    fn pair_spanning_edit_shifts_the_closer() {
        // An edit strictly inside a multi-line pair's interior (no bracket chars
        // touched): the closer's offset shifts by delta and the opener's DERIVED
        // partner follows automatically (nothing stored to leave stale).
        let mut buf = Buffer::new("{\naaa\nzzz\n}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::new(2..5, "aaaaa")]); // grow "aaa"→"aaaaa"
        assert_oracle(&b, &buf, "interior growth inside a multi-line pair");
        assert_eq!(bat(&b, 0).partner, Some(12)); // was Some(10)
        assert_eq!(bat(&b, 12).partner, Some(0)); // the closer, shifted
    }

    #[test]
    fn seed_leftover_cleared_to_none() {
        // Delete a prefix opener's closer: the opener is now unmatched — with a
        // derived partner there is no stale pointer to clear.
        let mut buf = Buffer::new("{\n}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::delete(2..3)]); // "{\n"
        assert_oracle(&b, &buf, "closer deleted");
        assert_eq!(b.all().len(), 1);
        assert_eq!((bat(&b, 0).partner, bat(&b, 0).open), (None, true));
    }

    // ── edits at scale: correctness deep in a large document (the incremental
    //    splice is local by construction; the oracle proves it stays correct) ──

    #[test]
    fn keystroke_deep_in_2000_lines() {
        let doc = vec!["fn f() { g(a[i], (b)); }"; 2000].join("\n");
        let mut buf = Buffer::new(&doc).expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let target = buf.point_to_offset(Point::new(1000, 3));
        edit(&mut b, &mut buf, vec![EditOp::insert(target, "x")]);
        assert_oracle(&b, &buf, "keystroke at line 1000");
    }

    #[test]
    fn balanced_pair_insert_deep_in_a_document() {
        let doc = vec!["fn f() { g(a[i], (b)); }"; 2000].join("\n");
        let mut buf = Buffer::new(&doc).expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let target = buf.point_to_offset(Point::new(1000, 8));
        edit(&mut b, &mut buf, vec![EditOp::insert(target, "()")]);
        assert_oracle(&b, &buf, "balanced pair at line 1000");
    }

    #[test]
    fn structural_edit_after_many_sibling_blocks() {
        // A newline at depth 0 after N closed sibling blocks — the seed reconcile
        // walks nothing (enclosing is an O(log) descent), but the result must still
        // be oracle-correct.
        const BLOCKS: usize = 20;
        let mut text = String::new();
        for _ in 0..BLOCKS {
            text.push_str("{[[[[[[[[[[]]]]]]]]]]}\n");
        }
        let tail = text.len() as u32;
        text.push_str("tail");
        let mut buf = Buffer::new(&text).expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::insert(tail, "\n")]);
        assert_oracle(&b, &buf, "newline after N closed sibling blocks");
    }

    #[test]
    fn structure_neutral_insert_at_document_start() {
        let mut lines = vec!["alpha", "beta"]; // two bracket-free head rows
        lines.extend(std::iter::repeat_n("x(y[z]) {w}", 1998));
        let doc = lines.join("\n");
        let mut buf = Buffer::new(&doc).expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let region = edit(&mut b, &mut buf, vec![EditOp::insert(0, "q")]);
        assert_eq!(region, 0..0, "a structure-neutral insert reconciles nothing");
        assert_oracle(&b, &buf, "letter at offset 0");
    }

    #[test]
    fn scattered_multicursor_typing_is_structure_neutral() {
        // THE document-scale multi-cursor case: a plain-letter keystroke at N
        // cursors spread top-to-bottom is per-edit O(log) (the delta-gap suffix does
        // not move) and reconciles nothing.
        let doc = vec!["fn f() { g(a[i], (b)); }"; 500].join("\n");
        let mut buf = Buffer::new(&doc).expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let inserts: Vec<EditOp> = (0..500)
            .step_by(50)
            .map(|row| EditOp::insert(buf.point_to_offset(Point::new(row, 0)), "x"))
            .collect();
        let region = edit(&mut b, &mut buf, inserts);
        assert_eq!(region, 0..0, "scattered structure-neutral edits reconcile nothing");
        assert_oracle(&b, &buf, "scattered multi-cursor letter inserts");
    }

    #[test]
    fn unbalanced_opener_at_the_top() {
        // An unbalanced opener changes every downstream depth — but depth is derived,
        // so the tree just gains one bracket. Oracle-equal, and the opener is
        // unmatched.
        let doc = vec!["(x)"; 100].join("\n");
        let mut buf = Buffer::new(&doc).expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        edit(&mut b, &mut buf, vec![EditOp::insert(0, "{")]);
        assert_oracle(&b, &buf, "unbalanced opener at the top");
        assert_eq!((bat(&b, 0).partner, bat(&b, 0).open), (None, true));
    }

    // ─────────────────────────────── edges ───────────────────────────────

    #[test]
    fn empty_document_first_insert_and_full_delete() {
        let mut buf = Buffer::new("").expect("empty loads");
        let mut b = Brackets::match_text(&buf.text());
        // An empty patch is a no-op with an empty reconcile window.
        assert_eq!(b.apply_edit(&Patch::new(), &buf), 0..0);
        edit(&mut b, &mut buf, vec![EditOp::insert(0, "({\n[")]);
        assert_oracle(&b, &buf, "first insert into an empty document");
        let len = buf.len();
        edit(&mut b, &mut buf, vec![EditOp::delete(0..len)]);
        assert_oracle(&b, &buf, "delete everything");
        assert!(b.all().is_empty());
    }

    #[test]
    fn edit_at_eof_appends() {
        let mut buf = Buffer::new("(a\n[b").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let len = buf.len();
        edit(&mut b, &mut buf, vec![EditOp::insert(len, "])")]); // "(a\n[b])"
        assert_oracle(&b, &buf, "append at EOF");
        assert_eq!(bat(&b, 0).partner, Some(6)); // ( … )
        assert_eq!(bat(&b, 3).partner, Some(5)); // [ … ]
    }

    #[test]
    fn whole_document_replace() {
        let mut buf = Buffer::new("(a)\n[b]\n{c}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let len = buf.len();
        edit(&mut b, &mut buf, vec![EditOp::new(0..len, "{new\n(doc)\n}")]);
        assert_oracle(&b, &buf, "whole-document replace");
        let len = buf.len();
        edit(&mut b, &mut buf, vec![EditOp::new(0..len, "[]")]);
        assert_oracle(&b, &buf, "whole-document replace, fewer lines");
    }

    #[test]
    fn edit_ending_exactly_on_a_line_boundary() {
        let mut buf = Buffer::new("(a)\n[b]\n{c}").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let target = buf.point_to_offset(Point::new(1, 0));
        edit(&mut b, &mut buf, vec![EditOp::insert(target, "(q)\n")]);
        assert_oracle(&b, &buf, "insert ending with a newline");
        let a = buf.point_to_offset(Point::new(1, 0));
        let e = buf.point_to_offset(Point::new(2, 0));
        edit(&mut b, &mut buf, vec![EditOp::delete(a..e)]);
        assert_oracle(&b, &buf, "delete ending at a line start");
    }

    #[test]
    fn multi_edit_transaction_with_scattered_inserts() {
        // Two scattered inserts in ONE apply(): the pair they form must resolve.
        let mut buf = Buffer::new("aaa\nbbb\nccc\nddd\neee").expect("fixture loads");
        let mut b = Brackets::match_text(&buf.text());
        let p1 = buf.point_to_offset(Point::new(0, 1));
        let p2 = buf.point_to_offset(Point::new(4, 1));
        edit(&mut b, &mut buf, vec![EditOp::insert(p1, "("), EditOp::insert(p2, ")")]);
        assert_oracle(&b, &buf, "scattered two-insert transaction");
        assert_eq!(bat(&b, 1).partner, Some(18)); // "a(aa\n…\ne)ee"
    }
}
