//! Bracket matching over a `SumTree` of bracket positions — matched pairs,
//! enclosing chains, and depths derived on an O(log) descent, so an edit is a pure
//! splice with no stored pointers to repair.
//!
//! Matching is not an O(1) summary, so each node carries a **shape**: a stack
//! machine's boundary state over its subtree, `(pending_closers, opener_stack)` —
//! the leading unmatched closers (still seeking an opener to their left) and the
//! trailing unmatched openers (LIFO, seeking a closer to their right), each entry
//! tagged with its byte offset (relative to the subtree, resolved to absolute by a
//! query descent).
//!
//! Combining is that automaton run across the seam: a right subtree's pending
//! closers pop the left subtree's opener stack — a pair matches and cancels, a
//! mismatched closer is permanently dropped (an opener above it closes nothing
//! below), an empty stack leaves it pending further left. The combine is
//! associative, so an edit is split-splice-append and the shape recomputes on the
//! O(log) path with no back-to-front crossing bookkeeping.
//!
//! On this monoid three queries ride as cursor descents, no flat scan:
//! [`enclosing_openers`] (the prefix stack left of a caret — the enclosing chain)
//! and [`partner`] (an opener's or closer's match). Both are checked in tests
//! against a from-scratch stack machine.

// A few helpers (`shape_of`, the scratch oracles) are exercised only by tests.
#![allow(dead_code)]

use std::ops::{ControlFlow, Range};

use crate::buffer::Buffer;
use crate::coords::Bias;
use crate::patch::Patch;
use crate::sum_tree::{Dimension, Item, SumTree, Summary};

// Canary counter: how many times `bracket_view` built a full prefix-stack summary
// (the heap-allocating depth resolution). The edit-path fold queries must keep this
// at ZERO — see `foldable_partner`. Debug/test only.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static BRACKET_VIEW_CALLS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// One bracket: its byte gap from the previous bracket, and its char.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BracketItem {
    gap: u32,
    ch: u8,
}

fn is_opener(c: u8) -> bool {
    matches!(c, b'(' | b'[' | b'{')
}

fn is_closer(c: u8) -> bool {
    matches!(c, b')' | b']' | b'}')
}

/// Whether opener `o` is closed by closer `c`.
fn pairs(o: u8, c: u8) -> bool {
    matches!((o, c), (b'(', b')') | (b'[', b']') | (b'{', b'}'))
}

/// The shape of a run of brackets — a stack machine's boundary state, the ONLY
/// thing that composes associatively:
/// - `pending`: leading unmatched closers, in document order, each still wanting a
///   matching opener somewhere to its LEFT (they hit an empty stack in this run).
/// - `stack`: unmatched openers, LIFO (last = top), each wanting a closer to its
///   RIGHT.
///
/// A closer that MISMATCHED an opener already on the stack is permanently
/// unmatched — it is NOT in `pending` (it closes nothing and never retries a
/// deeper opener). That drop rule is exactly what makes the shape associative:
/// because a mismatched closer never revisits a deeper opener, the same root shape
/// falls out no matter where the subtree boundaries are drawn.
///
/// One entry is one unmatched bracket in a [`ShapeSummary`]: its char and its byte
/// offset RELATIVE to that subtree's start (shifted on combine, resolved to
/// absolute by a query descent).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Entry {
    ch: u8,
    off: u32,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct ShapeSummary {
    span: u32,
    count: u32,
    pending: Vec<Entry>,
    stack: Vec<Entry>,
}

impl Summary for ShapeSummary {
    fn add_summary(&mut self, o: &Self) {
        let sl = self.span; // the right subtree begins here, in the combined frame
        self.span += o.span;
        self.count += o.count;
        // The right run's pending closers try the left run's opener stack (LIFO):
        // a match pops it; a mismatch permanently drops the closer (an opener on
        // top closes nothing below it); an empty stack leaves it pending for a run
        // further left. The right's own stack then sits on top. Right offsets shift
        // by `sl` into the combined frame.
        for c in &o.pending {
            let c = Entry { ch: c.ch, off: c.off + sl };
            match self.stack.last() {
                Some(t) if pairs(t.ch, c.ch) => {
                    self.stack.pop();
                }
                Some(_) => {} // mismatch: permanently unmatched, drop
                None => self.pending.push(c),
            }
        }
        self.stack.extend(o.stack.iter().map(|e| Entry { ch: e.ch, off: e.off + sl }));
    }
}

impl Item for BracketItem {
    type Summary = ShapeSummary;
    fn summary(&self) -> ShapeSummary {
        // The bracket sits at the end of its own gap-span (offset == gap in the
        // 1-item frame).
        let e = Entry { ch: self.ch, off: self.gap };
        let (pending, stack) = if is_opener(self.ch) { (vec![], vec![e]) } else { (vec![e], vec![]) };
        ShapeSummary { span: self.gap, count: 1, pending, stack }
    }
}

/// Seek by byte offset (cumulative gap).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ByteDim(u32);
impl Dimension<ShapeSummary> for ByteDim {
    fn add_summary(&mut self, s: &ShapeSummary) {
        self.0 += s.span;
    }
}

/// Seek by bracket index (cumulative count) — the bridge that turns a byte offset
/// into the item there (`k = summary_before(byte).count`, then the `k`-th item).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct CountDim(u32);
impl Dimension<ShapeSummary> for CountDim {
    fn add_summary(&mut self, s: &ShapeSummary) {
        self.0 += s.count;
    }
}

/// The open brackets enclosing byte `offset` — the opener stack of the shape of
/// everything strictly before it, innermost (top of stack) last, with absolute
/// offsets. O(log + depth): the prefix fold visits one root-to-leaf path, then the
/// stack is the enclosing chain — no scan of the brackets to the left, so the cost
/// tracks nesting depth, not document size.
fn enclosing_openers(tree: &SumTree<BracketItem>, offset: u32) -> Vec<Entry> {
    tree.summary_before(&ByteDim(offset)).stack
}

/// The matched partner of the bracket `ch` at byte `offset`, or `None` if it is
/// unmatched. Dispatches on the char — a closer looks left, an opener looks right.
fn partner(tree: &SumTree<BracketItem>, offset: u32, ch: u8) -> Option<u32> {
    if is_opener(ch) {
        opener_partner(tree, offset, ch)
    } else {
        closer_partner(tree, offset, ch)
    }
}

/// A closer's partner: the top of the prefix stack (the shape of everything
/// strictly left of it) — but only if that opener pairs the closer's char; a
/// mismatched top or an empty stack means the closer matches nothing. O(log +
/// depth): one prefix fold, then the stack top. The automaton's rule read off the
/// boundary state.
fn closer_partner(tree: &SumTree<BracketItem>, offset: u32, ch: u8) -> Option<u32> {
    crate::perf::charge(1); // complexity gate: one bracket-partner resolution
    match tree.summary_before(&ByteDim(offset)).stack.last() {
        Some(top) if pairs(top.ch, ch) => Some(top.off),
        _ => None,
    }
}

/// An opener's partner: the first closer in the SUFFIX that reaches the opener's
/// own stack level and pairs it. Folds the suffix's canonical subtree summaries
/// (offsets `> offset`) left-to-right into a running local stack `ls` of the
/// openers seen since — a balanced subtree contributes nothing and is skipped
/// whole; a subtree with unmatched closers hands over its `pending` list from the
/// cached summary (never a leaf scan). A pending closer that finds `ls` empty is
/// down at the opener's level: it pairs the opener (its partner, done) or
/// mismatches and is dropped (a deeper opener never retries — the automaton leaves
/// it permanently unmatched, so the next bottom-reaching closer tries the opener
/// again). Exhausting the suffix means the opener is unmatched. O(log + unmatched
/// brackets crossed).
fn opener_partner(tree: &SumTree<BracketItem>, offset: u32, ch: u8) -> Option<u32> {
    crate::perf::charge(1); // complexity gate: one bracket-partner resolution
    // An unmatched opener is exactly one still on the whole-document stack. Check
    // that O(root depth) set up front so a never-closed opener at the top of a huge
    // file costs O(depth), not an O(document) fold of the entire (matchless) suffix
    // — the per-frame colorization guarantee for the top bracket of an open block.
    if tree.summary().stack.iter().any(|e| e.off == offset) {
        return None;
    }
    let mut ls: Vec<u8> = Vec::new();
    tree.try_suffix_summaries(&ByteDim(offset + 1), &mut |shape: &ShapeSummary, start: &ByteDim| {
        for c in &shape.pending {
            match ls.last() {
                Some(&t) if pairs(t, c.ch) => {
                    ls.pop();
                }
                Some(_) => {} // mismatch above the opener: permanently unmatched, drop
                None => {
                    // Down at the opener's level.
                    if pairs(ch, c.ch) {
                        return ControlFlow::Break(start.0 + c.off);
                    }
                    // Mismatches the opener: drop it, the opener stays open.
                }
            }
        }
        for e in &shape.stack {
            ls.push(e.ch);
        }
        ControlFlow::Continue(())
    })
}

// ───────────────────────── derivation layer ─────────────────────────
// The tree stores only PRIMARY facts (a bracket's char + position, delta-gap
// encoded). Every DERIVED field — `open`, `depth`, `partner` — is recomputed here
// by an O(log) descent, so no cached copy can drift and an edit is a pure
// `SumTree::replace` with nothing to repair afterward. These functions answer the
// same queries as the from-scratch `Brackets` engine used as their test oracle.

use crate::bracket::Bracket;

/// Scan a document's bracket chars into the delta-gap tree — the O(n) load-time
/// constructor (the twin of `Brackets::match_text`). Bracket chars are single-byte
/// ASCII, so the byte index is the offset and non-bracket (incl. multibyte) bytes
/// are simply skipped.
pub(crate) fn tree_from_text(text: &str) -> SumTree<BracketItem> {
    let mut items: Vec<BracketItem> = Vec::new();
    let mut prev = 0u32;
    for (i, b) in text.bytes().enumerate() {
        if is_opener(b) || is_closer(b) {
            let off = i as u32;
            items.push(BracketItem { gap: off - prev, ch: b });
            prev = off;
        }
    }
    SumTree::from_items(items)
}

/// [`tree_from_text`], but skipping brackets the [`SkipContext`] flags as inside a
/// comment / string / char literal — the comment-aware load-time constructor.
pub(crate) fn tree_from_text_with(
    text: &str,
    cfg: &crate::bracket::BracketConfig,
) -> SumTree<BracketItem> {
    let mut items: Vec<BracketItem> = Vec::new();
    let mut prev = 0u32;
    for (off, b, skip) in crate::bracket::SkipContext::new(text.as_bytes(), cfg) {
        if !skip && (is_opener(b) || is_closer(b)) {
            items.push(BracketItem { gap: off - prev, ch: b });
            prev = off;
        }
    }
    SumTree::from_items(items)
}

/// The bracket char at exactly `offset`, or `None` if no bracket sits there — a
/// byte→index bridge (`k` brackets lie before `offset`, so the item there is the
/// `k`-th, kept only if its resolved offset is exactly `offset`). O(log), ZERO heap
/// (the count rides a light dimension, not the Vec-bearing shape summary) — the
/// fold hot paths call this per fold opener, so it must not allocate.
fn item_char_at(tree: &SumTree<BracketItem>, offset: u32) -> Option<u8> {
    let CountDim(k) = tree.measure_before::<ByteDim, CountDim>(&ByteDim(offset));
    let (item, _c, ByteDim(start)) = tree.seek::<CountDim, ByteDim>(&CountDim(k))?;
    (start + item.gap == offset).then_some(item.ch)
}

/// The partner of the OPENER at `offset`, or `None` if `offset` is not a live open
/// bracket or is unmatched — the fold hot path's lookup (`expand_folds_touched`,
/// `reconcile`, `FoldMap::new`). Unlike [`at`]/[`bracket_view`] it never builds the
/// prefix stack for `depth` (a field folds ignore): O(log + partner fold), and its
/// only heap is [`opener_partner`]'s small local stack. This is the difference
/// between a fold-heavy keystroke costing O(folds · log) and O(folds · log · depth)
/// with a `ShapeSummary` Vec cloned at every tree level per opener.
pub(crate) fn foldable_partner(tree: &SumTree<BracketItem>, offset: u32) -> Option<u32> {
    let ch = item_char_at(tree, offset)?;
    is_opener(ch).then(|| opener_partner(tree, offset, ch)).flatten()
}

/// The fully-derived [`Bracket`] for the char `ch` at `offset`. `open`/`depth` read
/// off the prefix stack (the shape strictly left of the bracket): an opener sits at
/// `depth = |stack|` and matches to its right ([`opener_partner`]); a matched
/// closer pops the stack top, taking the pair's depth `|stack| - 1`, and points at
/// it; anything unmatched is depth 0 with no partner. Mirrors [`super::step`].
fn bracket_view(tree: &SumTree<BracketItem>, offset: u32, ch: u8) -> Bracket {
    // Canary: `bracket_view` builds the full prefix stack (a heap `Vec`) for `depth`.
    // Legitimate on the draw path (colorization needs depth); a RED FLAG on the edit
    // path, where the fold queries must use the depth-free `foldable_partner`. A
    // fold-heavy keystroke that trips this is the O(folds · depth) alloc storm.
    #[cfg(any(test, debug_assertions))]
    BRACKET_VIEW_CALLS.with(|c| c.set(c.get() + 1));
    crate::perf::charge(1); // complexity gate: one full bracket resolution (incl. depth)
    let stack = tree.summary_before(&ByteDim(offset)).stack;
    if is_opener(ch) {
        Bracket { offset, open: true, depth: stack.len() as u32, partner: opener_partner(tree, offset, ch) }
    } else if let Some(top) = stack.last().filter(|t| pairs(t.ch, ch)) {
        Bracket { offset, open: false, depth: stack.len() as u32 - 1, partner: Some(top.off) }
    } else {
        Bracket { offset, open: false, depth: 0, partner: None }
    }
}

/// The bracket at exactly `offset`, fully derived, or `None`. O(log).
pub(crate) fn at(tree: &SumTree<BracketItem>, offset: u32) -> Option<Bracket> {
    item_char_at(tree, offset).map(|ch| bracket_view(tree, offset, ch))
}

/// Every bracket with offset in `[start, end)`, fully derived, in order — the
/// windowed per-visible-row colorization query, returned **lazily** so the hot
/// caller iterating it (once per visible row, every frame) allocates no `Vec`.
/// The brackets before `start`/`end` give the two count bounds (read via the
/// alloc-free `measure_before`, not the Vec-bearing shape summary); each
/// in-window bracket is then resolved and derived. O(log + window · log) —
/// window = brackets on the visible rows, tiny per row.
pub(crate) fn in_range(
    tree: &SumTree<BracketItem>,
    start: u32,
    end: u32,
) -> impl Iterator<Item = Bracket> + '_ {
    let CountDim(k_lo) = tree.measure_before::<ByteDim, CountDim>(&ByteDim(start));
    let CountDim(k_hi) = tree.measure_before::<ByteDim, CountDim>(&ByteDim(end));
    (k_lo..k_hi).filter_map(move |k| {
        let (item, _c, ByteDim(s)) = tree.seek::<CountDim, ByteDim>(&CountDim(k))?;
        Some(bracket_view(tree, s + item.gap, item.ch))
    })
}

/// Every bracket in document order, fully derived — the O(n) cold-path `all()`
/// (fold-source scans, the capture-folds harness, tests). One stack sweep over the
/// items, identical to [`Brackets::match_text`]'s automaton, so partners and depths
/// come out in O(n) rather than a descent each.
pub(crate) fn derive_all(tree: &SumTree<BracketItem>) -> Vec<Bracket> {
    let mut out: Vec<Bracket> = Vec::new();
    let mut stack: Vec<(u32, u8, usize)> = Vec::new(); // (offset, ch, out index)
    let mut off = 0u32;
    for item in tree.item_refs() {
        off += item.gap;
        let ch = item.ch;
        if is_opener(ch) {
            let depth = stack.len() as u32;
            stack.push((off, ch, out.len()));
            out.push(Bracket { offset: off, open: true, depth, partner: None });
        } else if let Some(&(ooff, _, oi)) = stack.last().filter(|(_, och, _)| pairs(*och, ch)) {
            stack.pop();
            out[oi].partner = Some(off);
            out.push(Bracket { offset: off, open: false, depth: stack.len() as u32, partner: Some(ooff) });
        } else {
            out.push(Bracket { offset: off, open: false, depth: 0, partner: None });
        }
    }
    out
}

/// The matched pair to highlight for a caret at `caret` — the matched bracket left
/// of it preferred, else the one at it. The tree twin of `Brackets::active_pair`.
pub(crate) fn active_pair(tree: &SumTree<BracketItem>, caret: u32) -> Option<(u32, u32)> {
    let matched = |off: u32| at(tree, off).filter(|b| b.partner.is_some());
    let b = caret
        .checked_sub(1)
        .and_then(matched)
        .or_else(|| matched(caret))?;
    Some((b.offset, b.partner.expect("filtered to matched")))
}

/// The matched pairs strictly enclosing `caret` (`open < caret <= close`),
/// **innermost first** — the prefix stack's matched openers, top down, each with
/// its partner. An opener on the stack at `caret` is unclosed before it, so a
/// matched one closes at/after `caret` by construction. The tree twin of
/// `Brackets::enclosing_pair` / `innermost_enclosing_where` / the enclosing part of
/// the fold walks. O(log + depth · partner-descent).
pub(crate) fn enclosing_pairs(tree: &SumTree<BracketItem>, caret: u32) -> Vec<(u32, u32)> {
    let stack = tree.summary_before(&ByteDim(caret)).stack;
    stack
        .iter()
        .rev()
        .filter_map(|e| opener_partner(tree, e.off, e.ch).map(|c| (e.off, c)))
        .collect()
}

/// Every matched pair enclosing OR touching `caret` (`open <= caret <= close + 1`)
/// — the enclosers plus a pair opened AT the caret or closed just before it. The
/// tree twin of `Brackets::enclosing_or_touching` (the fold-at-caret candidates).
pub(crate) fn enclosing_or_touching_pairs(tree: &SumTree<BracketItem>, caret: u32) -> Vec<(u32, u32)> {
    let mut out = Vec::new();
    // Touch from the left: the caret sits ON an opener.
    if let Some(ch) = item_char_at(tree, caret).filter(|&c| is_opener(c)) {
        if let Some(close) = opener_partner(tree, caret, ch) {
            out.push((caret, close));
        }
    }
    // Touch from the right: the caret sits just past a closer.
    if let Some(prev) = caret.checked_sub(1) {
        if let Some(ch) = item_char_at(tree, prev).filter(|&c| is_closer(c)) {
            if let Some(open) = closer_partner(tree, prev, ch) {
                out.push((open, prev));
            }
        }
    }
    out.extend(enclosing_pairs(tree, caret));
    out
}

// ───────────────────────────── the edit splice ─────────────────────────────
// An edit is a `SumTree::replace` of each edit's covering byte range with the
// brackets rescanned from the post-edit text. There is NO stored partner/depth —
// every such field is derived on query — so a splice has nothing to reconcile
// after it. Because the tree is delta-gap, the suffix past an edit does not move:
// only the FIRST suffix bracket's gap is re-anchored to bridge the delta. Edits in
// one patch are applied left-to-right on the running tree, each a local
// O(log + region) splice — a scattered N-caret keystroke is O(N log), never a
// re-scan of the whole first-to-last covering span.

/// Splice `tree` through `patch` (POST-edit `buffer`), returning the new tree and
/// the reconcile window — the fold openers `reconcile_folds_in` must re-check — or
/// `0..0` when no bracket structure changed (a pure offset shift; nothing can
/// break). The mover in `rebase_views` rides this on the forward path and every
/// undo/redo step alike.
///
/// The window is `[seed_lo, re + 1)`: `seed_lo` is the outermost opener enclosing
/// the first edit (PRE-edit tree), `re` the last edit's end. Every fold whose
/// foldability the edit can change has its opener in this window, because a fold
/// breaks ONLY when its opener becomes unmatched or its interior empties, and that
/// opener is always AT or ENCLOSING the edit — never in the downstream tail:
/// removing a closer orphans the *enclosing* opener it matched (left of the edit,
/// hence `>= seed_lo`), and inserting an opener steals a closer from an *earlier*
/// opener, never a later one. A pair wholly right of `re` keeps its own two
/// brackets and its balance, so it stays foldable (its offsets merely shift). This
/// is exactly why DERIVING partners keeps the window small: there is no stored
/// downstream identity to repair, so no convergence tail to scan — the window is
/// the enclosing chain plus the rescanned span, nothing more.
/// The `+ 1` includes a deletion's collapse point (`re == rs`), where a swallowed
/// fold opener maps.
pub(crate) fn apply_edit(
    tree: &SumTree<BracketItem>,
    patch: &Patch,
    buffer: &Buffer,
    cfg: &crate::bracket::BracketConfig,
) -> (SumTree<BracketItem>, Range<u32>) {
    let edits = patch.edits();
    if edits.is_empty() {
        return (tree.clone(), 0..0);
    }
    // Fast path: a structure-neutral patch (no bracket char inserted, none deleted)
    // leaves the bracket SET and pairing identical — only offsets shift. With MANY
    // edits (document-scale multi-caret typing) the per-edit split/append splices
    // below cost O(edits · log²) node allocations; rebuild in ONE bulk pass instead.
    // But that "no bracket char ⇒ set unchanged" reasoning holds ONLY without
    // comment/string awareness: with it, a typed `/` or `"` reshuffles skip-state
    // and so the bracket set, touching no bracket char — so the fast path is taken
    // only when the config is inactive.
    if !cfg.is_active() && edits.len() > 64 && is_structure_neutral(tree, patch, buffer) {
        return (bulk_shift(tree, patch), 0..0);
    }
    // The regions to rescan+splice, in NEW coords, as (new range, old byte length).
    // Without awareness each edit is its own region. WITH awareness the rescan must
    // see whole LINES — skip-state is line-local, so a `SkipContext` started at a
    // line boundary is correct — so each edit widens to its line boundaries and
    // same/adjacent-line edits merge. Still O(edited lines), never document-scale.
    let regions: Vec<(Range<u32>, u32)> = if cfg.is_active() {
        line_aligned_regions(patch, buffer)
    } else {
        edits.iter().map(|e| (e.new.clone(), e.old.end - e.old.start)).collect()
    };

    // Reconcile's left edge, from the PRE-edit tree: the outermost opener enclosing
    // the first region's start (same byte in old/new coords — the prefix is untouched).
    let rs0 = regions[0].0.start;
    let seed_lo = enclosing_openers(tree, rs0).first().map_or(rs0, |e| e.off);
    let re = regions[regions.len() - 1].0.end;

    let mut cur = tree.clone();
    let mut structural = false;
    for (new_range, old_len) in &regions {
        // On the running tree (regions to the left already applied), this region's
        // new span is `[rs, re_new)`; the old span it replaces ends at `rs + old_len`
        // (the prefix left of `rs` is settled, so `rs` is the same byte in both).
        let rs = new_range.start;
        let re_new = new_range.end;
        let re_old = rs + old_len;
        let delta = i64::from(re_new) - i64::from(re_old);

        // Split by COUNT, not byte: a delta-gap item is located at its span's END,
        // so the strict `summary_before(..).count` bounds are the correct index cuts.
        let k_rs = cur.summary_before(&ByteDim(rs)).count;
        let k_re_old = cur.summary_before(&ByteDim(re_old)).count;
        let before = cur.split_at(&CountDim(k_rs)).0;
        let before_end = before.extent::<ByteDim>().0;
        let removed = k_re_old - k_rs;

        // Rescan the new span's bracket chars (single-byte ASCII; the byte index is
        // the offset), delta-gap encoded from the last placed bracket. When aware,
        // `rs` is a line start, so the `SkipContext` starts fresh and correctly
        // classifies each byte as inside a comment / string / char literal.
        let mut region_items: Vec<BracketItem> = Vec::new();
        let mut prev = before_end;
        if re_new > rs {
            let slice = buffer.slice(rs..re_new);
            if cfg.is_active() {
                for (i, byte, skip) in crate::bracket::SkipContext::new(slice.as_bytes(), cfg) {
                    if !skip && (is_opener(byte) || is_closer(byte)) {
                        let off = rs + i;
                        region_items.push(BracketItem { gap: off - prev, ch: byte });
                        prev = off;
                    }
                }
            } else {
                for (i, b) in slice.bytes().enumerate() {
                    if is_opener(b) || is_closer(b) {
                        let off = rs + i as u32;
                        region_items.push(BracketItem { gap: off - prev, ch: b });
                        prev = off;
                    }
                }
            }
        }
        if removed > 0 || !region_items.is_empty() {
            structural = true;
        }

        let suffix = cur.split_at(&CountDim(k_re_old)).1;
        let mid = before.append(&SumTree::from_items(region_items));
        let fixed_suffix = reanchor_suffix(&suffix, k_re_old, &cur, delta, prev);
        cur = mid.append(&fixed_suffix);
    }
    (cur, if structural { seed_lo..re.saturating_add(1) } else { 0..0 })
}

/// Widen each edit's new-coord span to line boundaries (skip-state is line-local),
/// merging same/adjacent-line spans, as `(new range, old byte length)`. Bounded by
/// the edited lines — a scattered multi-caret patch yields one small region per
/// caret line, never one document-spanning region.
fn line_aligned_regions(patch: &Patch, buffer: &Buffer) -> Vec<(Range<u32>, u32)> {
    use crate::coords::Point;
    let line_start = |off: u32| buffer.point_to_offset(Point::new(buffer.offset_to_point(off).row, 0));
    let next_line_start = |off: u32| {
        let row = buffer.offset_to_point(off).row;
        if row + 1 < buffer.line_count() {
            buffer.point_to_offset(Point::new(row + 1, 0))
        } else {
            buffer.len()
        }
    };
    // (new_start, new_end, net delta), one per edit, merged where line spans touch.
    let mut merged: Vec<(u32, u32, i64)> = Vec::new();
    for e in patch.edits() {
        let ns = line_start(e.new.start);
        let ne = next_line_start(e.new.end);
        let d = i64::from(e.new.end - e.new.start) - i64::from(e.old.end - e.old.start);
        match merged.last_mut() {
            Some(last) if ns <= last.1 => {
                last.1 = last.1.max(ne);
                last.2 += d;
            }
            _ => merged.push((ns, ne, d)),
        }
    }
    // old length = new length − net delta (never negative: the region held the edits).
    merged
        .into_iter()
        .map(|(ns, ne, d)| (ns..ne, (i64::from(ne - ns) - d).max(0) as u32))
        .collect()
}

/// Whether `patch` changes no bracket structure — no bracket char inserted, none
/// deleted — so [`apply_edit`] can take the bulk offset-shift fast path. Cheap: the
/// insert check is one slice scan per edit's (keystroke-sized) new text, and "no
/// bracket deleted" is two O(log) count probes into the pre-edit tree.
fn is_structure_neutral(tree: &SumTree<BracketItem>, patch: &Patch, buffer: &Buffer) -> bool {
    patch.edits().iter().all(|e| {
        let no_insert = e.new.start == e.new.end
            || buffer.slice(e.new.start..e.new.end).bytes().all(|c| !is_opener(c) && !is_closer(c));
        // Brackets in the replaced OLD range (original coords — the pre-edit tree).
        let CountDim(lo) = tree.measure_before::<ByteDim, CountDim>(&ByteDim(e.old.start));
        let CountDim(hi) = tree.measure_before::<ByteDim, CountDim>(&ByteDim(e.old.end));
        no_insert && lo == hi
    })
}

/// Shift every bracket's offset through a structure-neutral `patch` and rebuild the
/// delta-gap tree in ONE balanced `from_items` — O(brackets), a memory-bandwidth
/// pass with no semantic work. The mapped offsets stay strictly ascending (a
/// structure-neutral shift never reorders brackets, and none lies inside an edited
/// range), so the gaps are positive.
fn bulk_shift(tree: &SumTree<BracketItem>, patch: &Patch) -> SumTree<BracketItem> {
    let items = tree.item_refs();
    if items.is_empty() {
        return tree.clone();
    }
    let mut off = 0u32;
    let mut queries: Vec<(u32, Bias)> = Vec::with_capacity(items.len());
    for it in &items {
        off += it.gap;
        queries.push((off, Bias::Right));
    }
    let mut mapped: Vec<u32> = Vec::new();
    patch.map_many(&queries, &mut mapped);
    let mut prev = 0u32;
    let new_items: Vec<BracketItem> = mapped
        .iter()
        .zip(&items)
        .map(|(&o, it)| {
            debug_assert!(o >= prev, "structure-neutral shift keeps brackets ordered");
            let gap = o - prev;
            prev = o;
            BracketItem { gap, ch: it.ch }
        })
        .collect();
    SumTree::from_items(new_items)
}

/// Re-anchor a split-off `suffix`'s first item so its offset becomes
/// `old_offset + delta`, keeping every later gap (they are relative — delta-gap, so
/// the suffix does not move). `k` is the first suffix bracket's overall index; `cur`
/// is the running tree the suffix came from (read only for that bracket's absolute
/// old offset); `last_off` is the last bracket placed before the suffix. O(log).
fn reanchor_suffix(
    suffix: &SumTree<BracketItem>,
    k: u32,
    cur: &SumTree<BracketItem>,
    delta: i64,
    last_off: u32,
) -> SumTree<BracketItem> {
    if suffix.is_empty() {
        return suffix.clone();
    }
    let (item, _c, ByteDim(s)) =
        cur.seek::<CountDim, ByteDim>(&CountDim(k)).expect("suffix non-empty ⇒ a k-th bracket");
    let first_old_off = s + item.gap;
    let new_gap = (i64::from(first_old_off) + delta - i64::from(last_off)) as u32;
    let fixed = BracketItem { gap: new_gap, ch: item.ch };
    suffix.replace(CountDim(0)..CountDim(1), std::iter::once(fixed))
}

/// From-scratch stack machine over `(offset, char)` — the oracle the monoid equals.
fn shape_of(brackets: &[(u32, u8)]) -> (Vec<Entry>, Vec<Entry>) {
    let mut stack: Vec<Entry> = Vec::new();
    let mut pending: Vec<Entry> = Vec::new();
    for &(off, c) in brackets {
        let e = Entry { ch: c, off };
        if is_opener(c) {
            stack.push(e);
        } else {
            match stack.last() {
                Some(t) if pairs(t.ch, c) => {
                    stack.pop();
                }
                Some(_) => {}
                None => pending.push(e),
            }
        }
    }
    (pending, stack)
}

/// From-scratch matched pairs (the same automaton as [`shape_of`], recording the
/// bond) — the oracle [`partner`] equals: for each bracket, its partner offset or
/// `None`. A closer bonds the stack top iff it pairs; a mismatch pops nothing and
/// leaves both unmatched, exactly as [`super::step`] does.
#[cfg(test)]
fn scratch_partners(brackets: &[(u32, u8)]) -> std::collections::HashMap<u32, Option<u32>> {
    let mut stack: Vec<(u32, u8)> = Vec::new();
    let mut out: std::collections::HashMap<u32, Option<u32>> =
        brackets.iter().map(|&(off, _)| (off, None)).collect();
    for &(off, c) in brackets {
        if is_opener(c) {
            stack.push((off, c));
        } else if let Some(&(ooff, och)) = stack.last() {
            if pairs(och, c) {
                stack.pop();
                out.insert(off, Some(ooff));
                out.insert(ooff, Some(off));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sum_tree::SumTree;

    /// Build the tree and the `(offset, char)` oracle input from `(gap, char)` runs.
    fn build(runs: &[(u32, u8)]) -> (SumTree<BracketItem>, Vec<(u32, u8)>) {
        let tree = SumTree::from_items(runs.iter().map(|&(gap, ch)| BracketItem { gap, ch }));
        let mut off = 0;
        let oracle = runs
            .iter()
            .map(|&(gap, ch)| {
                off += gap;
                (off, ch)
            })
            .collect();
        (tree, oracle)
    }

    #[test]
    fn shape_monoid_matches_the_scratch_stack_machine() {
        // Every associativity split (from_items chunks into leaves, folds bottom-up)
        // must land the same root shape — chars AND offsets — as one left-to-right
        // stack machine, at every length so the tree spans multiple internal levels.
        let all = b"()[]{}";
        let mut state = 0xB0A7u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for n in 0..600 {
            let len = (n % 50) + (next() as usize % 60);
            let runs: Vec<(u32, u8)> =
                (0..len).map(|_| (1 + next() % 5, all[next() as usize % all.len()])).collect();
            let (tree, oracle) = build(&runs);
            let s = tree.summary().clone();
            let (pending, stack) = shape_of(&oracle);
            assert_eq!((&s.pending, &s.stack), (&pending, &stack), "len={len}");
            assert_eq!(s.count, len as u32);
        }
        // Offsets are resolved to absolute at the root (gaps summed): "( (" at gaps
        // 3,4 leaves the stack openers at offsets 3 and 7.
        let (tree, _) = build(&[(3, b'('), (4, b'(')]);
        assert_eq!(tree.summary().stack, vec![Entry { ch: b'(', off: 3 }, Entry { ch: b'(', off: 7 }]);
        // "[[{(}})]]": ( and ) match; the rest mismatch
        // the '{' / '(' on the stack, so [ [ { stay open, nothing pending.
        let (tree, _) = build(&[1, 1, 1, 1, 1, 1, 1, 1, 1].iter().zip(b"[[{(}})]]").map(|(&g, &c)| (g, c)).collect::<Vec<_>>());
        assert!(tree.summary().pending.is_empty());
        assert_eq!(tree.summary().stack.iter().map(|e| e.ch).collect::<Vec<_>>(), b"[[{");
    }

    #[test]
    fn enclosing_openers_matches_the_scratch_stack() {
        // The open brackets at every byte offset must equal the stack of a scratch
        // stack machine over the brackets strictly before it — the O(log+depth)
        // enclosing chain, no siblings scan.
        let all = b"()[]{}";
        let mut state = 0x515Eu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..200 {
            let len = 1 + next() as usize % 80;
            let runs: Vec<(u32, u8)> =
                (0..len).map(|_| (1 + next() % 4, all[next() as usize % all.len()])).collect();
            let (tree, oracle) = build(&runs);
            let top = oracle.last().map_or(0, |&(o, _)| o) + 3;
            for x in 0..=top {
                let before: Vec<(u32, u8)> = oracle.iter().copied().filter(|&(o, _)| o < x).collect();
                let (_, want_stack) = shape_of(&before);
                assert_eq!(enclosing_openers(&tree, x), want_stack, "offset {x}");
            }
        }
    }

    #[test]
    fn partner_matches_the_scratch_stack_machine() {
        // Every bracket's partner (opener → right, closer → left) must equal the
        // bond a from-scratch matched-pairs pass records — the O(log)-descent twin
        // of the Vec engine's precomputed `partner` field. The generator is
        // bracket-dense with varied gaps so the tree spans levels and the suffix
        // fold crosses whole balanced subtrees, mismatched closers, and nesting.
        let all = b"()[]{}";
        let mut state = 0x9A27u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..300 {
            let len = 1 + next() as usize % 90;
            let runs: Vec<(u32, u8)> =
                (0..len).map(|_| (1 + next() % 4, all[next() as usize % all.len()])).collect();
            let (tree, oracle) = build(&runs);
            let want = scratch_partners(&oracle);
            for &(off, ch) in &oracle {
                assert_eq!(partner(&tree, off, ch), want[&off], "bracket {ch:?} at {off}");
            }
        }
        // Hand cases pinning the fold's subtle branches.
        // Opener skips a mismatched bottom closer, then bonds the matching one:
        // "( ] )" → '(' pairs the ')', the ']' is unmatched.
        let (tree, _) = build(&[(0, b'('), (2, b']'), (2, b')')]); // offsets 0, 2, 4
        assert_eq!(partner(&tree, 0, b'('), Some(4));
        assert_eq!(partner(&tree, 4, b')'), Some(0));
        assert_eq!(partner(&tree, 2, b']'), None);
        // Nested, partner far to the right across a balanced sibling subtree:
        // "( [ ] { } )" → outer '(' at 0 bonds ')' at 10.
        let (tree, _) =
            build(&[(0, b'('), (2, b'['), (2, b']'), (2, b'{'), (2, b'}'), (2, b')')]);
        assert_eq!(partner(&tree, 0, b'('), Some(10));
        assert_eq!(partner(&tree, 2, b'['), Some(4));
        assert_eq!(partner(&tree, 6, b'{'), Some(8));
        // A closer whose stack top mismatches bonds nothing: "{ ]" → both unmatched.
        let (tree, _) = build(&[(0, b'{'), (1, b']')]);
        assert_eq!(partner(&tree, 0, b'{'), None);
        assert_eq!(partner(&tree, 1, b']'), None);
    }

    /// A seeded-random bracket-dense ASCII document (letters + newlines interleaved
    /// so byte offsets, line structure, and non-bracket gaps all vary).
    fn random_text(next: &mut impl FnMut() -> u32) -> String {
        const POOL: &[u8] = b"()[]{}()[]{}abc \n";
        let len = 1 + next() as usize % 160;
        (0..len).map(|_| POOL[next() as usize % POOL.len()] as char).collect()
    }

    #[test]
    fn derivations_match_the_vec_engine() {
        // Every derived Bracket field — `open`, `depth`, `partner` — recomputed from
        // the tree must deep-equal the Vec engine's precomputed one, at every
        // bracket, over random docs. `derive_all` (the O(n) sweep) AND the per-offset
        // `bracket_view`/`at` descents (the hot `at`/`in_range` path) are both
        // checked against `match_text`, plus `item_char_at`'s bracket/non-bracket
        // discrimination.
        let mut state = 0x1D01u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..400 {
            let text = random_text(&mut next);
            let real = crate::bracket::Brackets::match_text(&text);
            let tree = tree_from_text(&text);
            assert_eq!(derive_all(&tree), real.all(), "derive_all: {text:?}");
            for b in real.all() {
                let ch = text.as_bytes()[b.offset as usize];
                assert_eq!(bracket_view(&tree, b.offset, ch), b, "bracket_view @{}: {text:?}", b.offset);
                assert_eq!(at(&tree, b.offset), Some(b), "at @{}: {text:?}", b.offset);
                assert_eq!(item_char_at(&tree, b.offset), Some(ch));
            }
            // `at` / `item_char_at` at non-bracket offsets is `None`.
            for off in 0..text.len() as u32 {
                if !is_opener(text.as_bytes()[off as usize]) && !is_closer(text.as_bytes()[off as usize]) {
                    assert_eq!(at(&tree, off), None, "non-bracket @{off}: {text:?}");
                    assert_eq!(item_char_at(&tree, off), None);
                }
            }
        }
    }

    #[test]
    fn queries_match_the_vec_engine() {
        // The enclosing / active-pair / touching tuple queries, re-expressed as tree
        // descents, must equal the Vec engine's `walk_enclosing`-based answers at
        // every caret over random docs — including the touching edges and the
        // range-enclosing ladder rung.
        let mut state = 0x7A11u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..300 {
            let text = random_text(&mut next);
            let real = crate::bracket::Brackets::match_text(&text);
            let tree = tree_from_text(&text);
            for caret in 0..=text.len() as u32 {
                assert_eq!(active_pair(&tree, caret), real.active_pair(caret), "active_pair @{caret}: {text:?}");
                assert_eq!(
                    enclosing_pairs(&tree, caret).first().copied(),
                    real.enclosing_pair(caret),
                    "enclosing_pair @{caret}: {text:?}"
                );
                let mut got = enclosing_or_touching_pairs(&tree, caret);
                got.sort_unstable();
                let mut want = real.enclosing_or_touching(caret);
                want.sort_unstable();
                assert_eq!(got, want, "enclosing_or_touching @{caret}: {text:?}");
                // The expand-selection rung: the tightest pair whose CONTENTS hold
                // `[caret, caret+2)` — exercise `enclosing_pair_of_range`.
                let end = (caret + 2).min(text.len() as u32);
                let got_range = enclosing_pairs(&tree, caret).into_iter().find(|&(_, c)| end <= c);
                assert_eq!(
                    got_range,
                    real.enclosing_pair_of_range(caret, end),
                    "enclosing_pair_of_range {caret}..{end}: {text:?}"
                );
            }
        }
    }
}
