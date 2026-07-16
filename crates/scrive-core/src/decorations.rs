//! Tracked-range decorations: the one store, one mover for diagnostics,
//! snippet stops, auto-close regions, and find matches.
//!
//! Diagnostics, snippet placeholders, and find matches are ONE mechanism — a
//! [`TrackedRange`] carrying a [`DecorationKind`] and a [`Stickiness`]. The
//! store holds bare current-revision byte offsets and is moved *eagerly* by
//! [`DecorationStore::apply_patch`] on every commit, so a decoration's offsets
//! are always "now" — there is no cross-revision anchor wrapper to translate
//! through, because the positions are re-based the instant the text changes.
//! The store is a **delta-gap interval `SumTree`** (`DecoItem` = gap + len; the
//! relative-coordinate `DecoSummary` carries a rebasing `max_end`): windowed
//! queries are an O(log n) interval descent, and the eager mover shifts a whole
//! suffix in O(edit-window + log n) by rewriting one seam gap — never the
//! O(store) per-commit rescan a flat `Vec` would force at find scale (10k
//! matches).
//!
//! Stickiness is a named pair of [`Bias`]es, one per boundary marker ("sticks
//! to the previous char?"); the mover threads each endpoint through
//! [`Patch::map_offset`] with that boundary's bias and never inverts — a range
//! may collapse to empty, but its start never passes its end.
//!
//! One file, one owner: this module owns the store and its mover only. The
//! squiggle *drawing* (`squiggle_vertices`, `SquiggleContext`,
//! `default_squiggle_style`) is the iced layer's concern and lives in
//! `scrive-iced/src/squiggle.rs` — nothing here touches pixels or colors. The
//! store's methods live on a standalone [`DecorationStore`] so this module is
//! self-contained; `Document` wires the store in and drives its mover.

use std::ops::Range;
use std::sync::Arc;

use crate::coords::Bias;
use crate::patch::Patch;
use crate::sum_tree::{Dimension, Item, Summary, SumTree};

// Op-count canary: full-store `sort()`s on this thread. A document-scale test
// asserts a plain keystroke with a large decoration set (find matches /
// diagnostics) does NOT re-sort the whole store — the windowed mover keeps
// order without sorting, and `add_sorted_batch` skips an empty batch. Debug/
// test only.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static DECORATION_SORTS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// Visit canary: how many decoration items the read/query paths touch on this
// thread. Incremented once per item visited in `decorations_in` (pushed hits),
// `decoration_range` (whole-store id scan), and `decoitems_to_ranges`/`to_vec`
// (band/whole materialization). The visit-count tests read it to prove a read
// stays sublinear even where the work-meter charges ~0 (a root-summary read like
// `find_count`, or a bounded own-store scan). It is a separate mechanism from the
// `crate::perf` work-meter, which deliberately skips the band materialization the
// windowed mover rides; this canary counts it. Debug/test only.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static DECORATION_VISITS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Bump the visit canary by one item (debug/test only; erased in release).
#[inline]
fn count_visit() {
    #[cfg(any(test, debug_assertions))]
    DECORATION_VISITS.with(|c| c.set(c.get() + 1));
}

/// Diagnostic severity. `Ord` is the render order: squiggles draw in ascending
/// order so the most severe paints last and wins.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub enum Severity {
    /// Faint advisory (unused import, style nudge). Lowest render priority.
    Hint = 0,
    /// Informational note.
    Info = 1,
    /// Something suspect but not fatal.
    Warning = 2,
    /// A hard error. Highest render priority.
    Error = 3,
}

/// The four stickiness modes, stored as what they are: a named pair of
/// [`Bias`]es, one per boundary marker.
///
/// The mode names describe how the range reacts to an insertion *at one of its
/// edges*: `AlwaysGrows` swallows text inserted at either edge, `NeverGrows`
/// rejects both, and the two `GrowsOnly*` modes swallow at exactly one edge.
/// Typing strictly *inside* a range always grows it regardless of mode —
/// stickiness only decides the edges.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Stickiness {
    /// Grows at both edges: `(Left, Right)`. Active snippet placeholder.
    AlwaysGrows,
    /// Grows at neither edge: `(Right, Left)`. Diagnostics, find matches,
    /// inactive snippet stops.
    NeverGrows,
    /// Grows only at its start: `(Left, Left)`. Caret at end-of-line.
    GrowsOnlyBefore,
    /// Grows only at its end: `(Right, Right)`. A trailing caret.
    GrowsOnlyAfter,
}

impl Stickiness {
    /// The `(start_bias, end_bias)` pair — the entire semantic content of the
    /// enum.
    #[must_use]
    pub fn biases(self) -> (Bias, Bias) {
        match self {
            Self::AlwaysGrows => (Bias::Left, Bias::Right),
            Self::NeverGrows => (Bias::Right, Bias::Left),
            Self::GrowsOnlyBefore => (Bias::Left, Bias::Left),
            Self::GrowsOnlyAfter => (Bias::Right, Bias::Right),
        }
    }
}

/// What a tracked range *is*. The variant carries every distinction the render
/// path, a diagnostics panel, and a future hover need directly — no separate
/// owner-id or filter machinery alongside it.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub enum DecorationKind {
    /// A compiler finding. `severity`, `message`, and `code` ride the store so
    /// position and content stay reunited in one place; `code` is the
    /// compiler's diagnostic code.
    Diagnostic {
        /// Render color / render-order key.
        severity: Severity,
        /// Human-readable message (shared, cheap to clone across frames).
        message: Arc<str>,
        /// The compiler's diagnostic code, if any (e.g. `E0425`).
        code: Option<Arc<str>>,
    },
    /// A find/search hit. The find controller re-queries the whole match set on
    /// every edit, so a collapsed hit is dropped rather than kept — its
    /// empty-range policy is [`EmptyPolicy::Drop`].
    FindMatch,
    /// A snippet placeholder stop. Active-ness lives in the range's
    /// [`Stickiness`], not here — the active stop is `AlwaysGrows`, the rest
    /// `NeverGrows`.
    SnippetStop {
        /// Tab-order index of this stop within its session.
        index: u8,
    },
    /// The provenance region of an auto-inserted closing pair.
    AutoClosePair,
}

impl DecorationKind {
    /// Post-commit policy for a range of this kind that has collapsed to empty:
    /// [`FindMatch`](Self::FindMatch) is re-queried so it drops; everything else
    /// keeps (the owner controls its lifetime).
    #[must_use]
    pub fn empty_policy(&self) -> EmptyPolicy {
        match self {
            Self::FindMatch => EmptyPolicy::Drop,
            _ => EmptyPolicy::Keep,
        }
    }
}

/// What the mover does with a range that collapsed to empty on an edit.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum EmptyPolicy {
    /// Keep the (now zero-width) range; its owner controls its lifetime and it
    /// still renders (diagnostics at least one cell wide).
    Keep,
    /// Drop the range on collapse; a wholesale producer re-publishes it.
    Drop,
}

/// A decoration's identity: monotonic, never reused within a store's lifetime.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct DecorationId(u64);

/// One tracked range: a [`DecorationId`], a current-revision byte [`Range`], its
/// [`DecorationKind`], and its [`Stickiness`].
///
/// Because the store is moved eagerly on every commit, `range` is always in
/// "now" coordinates — no cross-revision `Anchor` wrapper is needed.
#[derive(Clone, Debug)]
pub struct TrackedRange {
    /// Stable identity, minted at insertion.
    pub id: DecorationId,
    /// Current-revision byte offsets `[start, end)`.
    pub range: Range<u32>,
    /// What this decoration is (and its content, for diagnostics).
    pub kind: DecorationKind,
    /// How each endpoint reacts to an edit at its boundary.
    pub stickiness: Stickiness,
}

/// One compiler finding as published by the app's debounced compile loop — the
/// unified diagnostic type.
///
/// `span` is byte offsets into the revision-`N` snapshot the compile ran
/// against; it is stamped with that revision at [`DecorationStore::set_diagnostics`]
/// so a stale set is dropped rather than mis-placed. `#[non_exhaustive]` so
/// future fields (related spans, fix-its) don't break construction sites.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct Diagnostic {
    /// Byte span into the revision this diagnostic was computed against.
    pub span: Range<u32>,
    /// Severity — render color and render-order key.
    pub severity: Severity,
    /// Human-readable message.
    pub message: String,
    /// The compiler's diagnostic code, if any.
    pub code: Option<String>,
}

impl Diagnostic {
    /// A diagnostic with no code. Convenience for the common construction site.
    #[must_use]
    pub fn new(span: Range<u32>, severity: Severity, message: impl Into<String>) -> Self {
        Self {
            span,
            severity,
            message: message.into(),
            code: None,
        }
    }
}

/// The result of publishing a diagnostic set.
#[derive(Clone, PartialEq, Eq, Debug)]
pub enum DiagnosticsOutcome {
    /// The set was current: squiggle decorations were replaced wholesale.
    Applied {
        /// How many diagnostics were installed.
        count: usize,
    },
    /// The set's revision was not current — the whole set was dropped (no
    /// patch-forwarding of stale sets); the previous decorations keep riding
    /// edits via stickiness and the app waits for its next debounce.
    Stale {
        /// The store's current revision at drop time.
        current: u64,
    },
}

/// Private delta-gap `SumTree` leaf. `gap` = this range's start minus the previous
/// item's start; `len` = end − start. The absolute start is the prefix sum of gaps
/// (the [`StartDim`] dimension), end = start + len. **Deltas, not absolute
/// offsets** — this is what lets [`DecorationStore::apply_patch`] shift a whole
/// suffix by adjusting ONE seam gap instead of rewriting every item
/// (O(edit-window + log n) per commit, not O(n)).
#[derive(Clone, Debug)]
struct DecoItem {
    gap: u32,
    len: u32,
    id: DecorationId,
    kind: DecorationKind,
    stickiness: Stickiness,
}

/// Interval-tree summary in RELATIVE coordinates. `span` = Σ gaps (the subtree's
/// width); `count`; `max_end` = the greatest item end **relative to the subtree
/// base**, composed by rebasing the right child by the left's `span`. Because the
/// interior gaps and the relative `max_end` are invariant under a uniform shift,
/// a suffix moves in O(1) at the seam — the whole point.
#[derive(Clone, Copy, Debug, Default)]
struct DecoSummary {
    span: u32,
    count: u32,
    max_end: u32,
    /// Number of [`DecorationKind::FindMatch`] items in the subtree — the additive
    /// monoid behind the O(1) `find_count`, the O(log) `find_rank_before`/
    /// `nth_find` order-statistic seek (via `FindCountDim`), and the find lane
    /// of the scrollbar overview. Shift-invariant (independent of `span`), so the
    /// suffix reanchor and the windowed-mover oracle are untouched.
    find_count: u32,
    /// The greatest Diagnostic severity in the subtree, encoded `severity + 1`
    /// (`1..=4`, `0` = no diagnostic) so the empty monoid is a clean `0`. A max,
    /// so it folds a whole subtree in O(1) for the diagnostic lane of the overview
    /// reduce. Shift-invariant like `find_count`.
    sev_max: u8,
}

impl Summary for DecoSummary {
    fn add_summary(&mut self, o: &Self) {
        // Rebase the right subtree's max end by this (left) subtree's span, then
        // take the max — the relative-coordinate interval composition.
        self.max_end = self.max_end.max(self.span + o.max_end);
        self.span += o.span;
        self.count += o.count;
        self.find_count += o.find_count; // additive monoid
        self.sev_max = self.sev_max.max(o.sev_max); // max monoid, position-free
    }
}

impl Item for DecoItem {
    type Summary = DecoSummary;
    fn summary(&self) -> DecoSummary {
        DecoSummary {
            span: self.gap,
            count: 1,
            max_end: self.gap + self.len,
            find_count: u32::from(matches!(self.kind, DecorationKind::FindMatch)),
            sev_max: match self.kind {
                DecorationKind::Diagnostic { severity, .. } => severity as u8 + 1, // 1..=4
                _ => 0,
            },
        }
    }
}

/// Dimension: absolute start position — accumulates each item's `gap` (a prefix
/// sum), so seeking `StartDim(p)` lands at the first item whose start ≥ p.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct StartDim(u32);
impl Dimension<DecoSummary> for StartDim {
    fn add_summary(&mut self, s: &DecoSummary) {
        self.0 += s.span;
    }
}

/// Dimension: item index — accumulates `count`. Splits are by *index* (not byte),
/// because a delta-gap item is located at its start and a byte split would keep an
/// item the edit moves on the wrong side (mirrors the bracket tree).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct CountDim(u32);
impl Dimension<DecoSummary> for CountDim {
    fn add_summary(&mut self, s: &DecoSummary) {
        self.0 += s.count;
    }
}

/// Dimension: cumulative find-match count — accumulates `find_count`. Seeking
/// `FindCountDim(r)` lands directly on the item carrying the r-th find increment
/// (its cumulative find-count first exceeds `r`), which is therefore the r-th
/// [`DecorationKind::FindMatch`] itself — the order-statistic behind `nth_find`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct FindCountDim(u32);
impl Dimension<DecoSummary> for FindCountDim {
    fn add_summary(&mut self, s: &DecoSummary) {
        self.0 += s.find_count;
    }
}

/// Delta-gap encode a `(start, id)`-sorted absolute list, the first gap relative to
/// `base` (0 for a whole-store rebuild; the split point's absolute offset when
/// re-encoding a middle band that `append`s onto a shared prefix). The rebuild
/// handle for the producer mutations and the windowed mover.
fn ranges_to_decoitems(v: Vec<TrackedRange>, base: u32) -> impl Iterator<Item = DecoItem> {
    let mut prev = base;
    v.into_iter().map(move |r| {
        debug_assert!(r.range.start >= prev, "ranges_to_decoitems needs ascending starts");
        let gap = r.range.start - prev;
        prev = r.range.start;
        DecoItem {
            gap,
            len: r.range.end - r.range.start,
            id: r.id,
            kind: r.kind,
            stickiness: r.stickiness,
        }
    })
}

/// A delta-gap subtree's items as absolute tracked ranges, accumulating from `base`
/// (the absolute start where this — possibly split-off — subtree begins).
fn decoitems_to_ranges(tree: &SumTree<DecoItem>, base: u32) -> Vec<TrackedRange> {
    let mut out = Vec::with_capacity(tree.summary().count as usize);
    let mut start = base;
    for it in tree.items() {
        count_visit(); // canary: band / whole-store materialization
        start += it.gap;
        out.push(TrackedRange {
            id: it.id,
            range: start..start + it.len,
            kind: it.kind,
            stickiness: it.stickiness,
        });
    }
    out
}

/// Map every range's endpoints through `patch` with its stickiness bias, never
/// inverting (a collapse pins both ends to the mapped end — [`Patch::map_range`]'s
/// rule), and drop the ranges that collapsed to empty whose kind re-publishes
/// ([`EmptyPolicy::Drop`], i.e. find matches). Shared by the naive and windowed
/// movers so their per-range semantics are identical by construction.
fn remap_ranges(patch: &Patch, v: &mut Vec<TrackedRange>) {
    let mut queries: Vec<(u32, Bias)> = Vec::with_capacity(v.len() * 2);
    for r in v.iter() {
        let (bs, be) = r.stickiness.biases();
        queries.push((r.range.start, bs));
        queries.push((r.range.end, be));
    }
    let mut mapped: Vec<u32> = Vec::new();
    patch.map_many(&queries, &mut mapped);
    for (i, r) in v.iter_mut().enumerate() {
        let (ms, me) = (mapped[2 * i], mapped[2 * i + 1]);
        r.range = ms.min(me)..me;
    }
    v.retain(|r| {
        let collapsed = r.range.start == r.range.end;
        !(collapsed && matches!(r.kind.empty_policy(), EmptyPolicy::Drop))
    });
}

/// Re-anchor a split-off `zone_c` (decorations entirely after the edit) onto a
/// rebuilt head: its first item's start becomes `old_start + delta`, while every
/// later gap and length stays relative and unchanged — so the whole suffix shifts
/// by rewriting ONE gap, O(log n). `k_hi` is that first item's overall index;
/// `orig` is the pre-split tree (read only for the item's old absolute start);
/// `prev_start` is the last start placed before it. Mirrors the bracket tree's
/// `reanchor_suffix`.
fn reanchor(
    zone_c: &SumTree<DecoItem>,
    k_hi: u32,
    orig: &SumTree<DecoItem>,
    delta: i64,
    prev_start: u32,
) -> SumTree<DecoItem> {
    if zone_c.is_empty() {
        return zone_c.clone();
    }
    let (item, _c, StartDim(s)) = orig
        .seek::<CountDim, StartDim>(&CountDim(k_hi))
        .expect("zone_c non-empty ⇒ a k_hi-th item exists");
    let first_old_start = s + item.gap;
    let new_gap = (i64::from(first_old_start) + delta - i64::from(prev_start)) as u32;
    let fixed = DecoItem { gap: new_gap, ..item.clone() };
    zone_c.replace(CountDim(0)..CountDim(1), std::iter::once(fixed))
}

/// The tracked-range decoration store: the one store, one mover.
///
/// Producers (snippet session, auto-close, find, the compile loop) add/remove/
/// replace; the commit path calls [`apply_patch`](Self::apply_patch) exactly once
/// per transaction, *before* change events fire, so every consumer only ever sees
/// post-edit positions.
#[derive(Clone, Debug)]
pub struct DecorationStore {
    /// Delta-gap interval `SumTree`, items in `(start, id)` order. Each node's
    /// relative [`DecoSummary`] carries a rebasing `max_end`, so `decorations_in`
    /// is an O(log n) [`SumTree::filter_visit`] interval descent and `apply_patch`
    /// shifts a suffix in O(log n) at one seam. Producer mutations (add / remove /
    /// diagnostics) still reconstruct → rebuild O(n) — they are not the hot path;
    /// the per-keystroke mover is.
    tree: SumTree<DecoItem>,
    next_id: u64,
}

impl Default for DecorationStore {
    fn default() -> Self {
        Self::new()
    }
}

impl DecorationStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self { tree: SumTree::new(), next_id: 1 }
    }

    /// Number of tracked ranges.
    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.summary().count as usize
    }

    /// Whether the store holds no ranges.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// Insert one tracked range and return its fresh id. Producers: the snippet
    /// session (stops), auto-close, and find.
    ///
    /// The caller supplies current-revision offsets; per-endpoint char-boundary
    /// clipping needs the buffer, so it is the `Document`'s responsibility, not
    /// this standalone store's.
    pub fn add_decoration(
        &mut self,
        range: Range<u32>,
        kind: DecorationKind,
        stickiness: Stickiness,
    ) -> DecorationId {
        let id = self.mint();
        let mut v = self.to_vec();
        v.push(TrackedRange { id, range, kind, stickiness });
        self.set_sorted(v);
        id
    }

    /// Owner-driven removal (session end, auto-close invalidation): returns the
    /// range if `id` was present, else `None`.
    pub fn take_decoration(&mut self, id: DecorationId) -> Option<TrackedRange> {
        let mut v = self.to_vec();
        let i = v.iter().position(|r| r.id == id)?;
        let taken = v.remove(i);
        self.rebuild(v); // removal keeps sort order — rebuild, no re-sort
        Some(taken)
    }

    /// The decoration's current byte range after the mover has ridden it across
    /// edits, or `None` if the id is gone. The snippet session and find read a
    /// stop's or match's live position back through this.
    #[must_use]
    pub fn decoration_range(&self, id: DecorationId) -> Option<Range<u32>> {
        // Delta-gap items store no absolute range; visit in order (accumulating the
        // start) and read off the match. O(n) — a cold path (snippet tab / find
        // navigation), never per-keystroke.
        let mut found = None;
        self.tree.filter_visit::<StartDim, _, _>(
            &|_, _| true,
            &mut |it: &DecoItem, before: &StartDim| {
                // Unpruned whole-store scan: charge and count EVERY item so the
                // complexity gate and visit canary see this O(M) read. This is a
                // cold path (snippet tab / find navigation), never per-keystroke.
                crate::perf::charge(1);
                count_visit();
                if it.id == id {
                    let start = before.0 + it.gap;
                    found = Some(start..start + it.len);
                }
            },
        );
        found
    }

    /// Swap a range's stickiness — the snippet active-stop swap: `AlwaysGrows`
    /// onto the newly active stop, `NeverGrows` onto the leaving one. Returns
    /// `false` if the id is gone.
    pub fn set_decoration_stickiness(&mut self, id: DecorationId, s: Stickiness) -> bool {
        let mut v = self.to_vec();
        match v.iter_mut().find(|r| r.id == id) {
            Some(r) => {
                r.stickiness = s;
                self.rebuild(v); // start order unchanged — rebuild, no re-sort
                true
            }
            None => false,
        }
    }

    /// All ranges intersecting `range`, in ascending `(start, id)` order.
    /// Touching counts, so a zero-width range on the boundary still renders.
    ///
    /// An O(log n + hits) interval descent: a subtree is entered only when it can
    /// hold a touching range — some start ≤ `range.end` and some end ≥
    /// `range.start`, read from the relative `max_end` in its summary — then an
    /// exact per-item touch filter keeps only the real hits. In-order, so the
    /// yield is ascending `(start, id)`.
    pub fn decorations_in(&self, range: Range<u32>) -> impl Iterator<Item = TrackedRange> {
        let (qs, qe) = (range.start, range.end);
        // Interval descent (O(log n + hits)): enter a subtree only if some start ≤ qe
        // (`before ≤ qe`, since every start ≥ the base) AND some end ≥ qs (absolute
        // max end = `before + max_end`), then the exact per-item touch filter.
        // In-order, so ascending `(start, id)`. The range is *computed* from the
        // delta-gaps (nothing absolute is stored to borrow), so the yield is owned.
        let mut out: Vec<TrackedRange> = Vec::new();
        self.tree.filter_visit::<StartDim, _, _>(
            &|before: &StartDim, sum: &DecoSummary| {
                before.0 <= qe && before.0 + sum.max_end >= qs
            },
            &mut |it: &DecoItem, before: &StartDim| {
                let start = before.0 + it.gap;
                let end = start + it.len;
                if start <= qe && end >= qs {
                    // Charge/count per HIT: a whole-store `decorations_in(0..MAX)`
                    // charges O(M), while a windowed query charges only its
                    // handful of hits.
                    crate::perf::charge(1);
                    count_visit();
                    out.push(TrackedRange {
                        id: it.id,
                        range: start..end,
                        kind: it.kind.clone(),
                        stickiness: it.stickiness,
                    });
                }
            },
        );
        out.into_iter()
    }

    /// How many [`DecorationKind::FindMatch`] ranges the store holds — an O(1)
    /// read of the root summary, not a whole-store walk.
    #[must_use]
    pub fn find_count(&self) -> usize {
        self.tree.summary().find_count as usize
    }

    /// How many find matches start strictly before byte offset `off` — the rank
    /// (0-based document-order index) of the first find at or after `off`. O(log n)
    /// prefix fold; find navigation uses it to number the current match.
    #[must_use]
    pub fn find_rank_before(&self, off: u32) -> usize {
        self.tree.summary_before(&StartDim(off)).find_count as usize
    }

    /// The r-th find match in `(start, id)` order, or `None` when `r >=
    /// find_count()`. O(log n): `FindCountDim` seek lands directly on the item
    /// carrying the r-th find increment — since only find matches contribute to
    /// that dimension, the landed item IS the r-th find (a `debug_assert` pins it).
    #[must_use]
    pub fn nth_find(&self, r: usize) -> Option<(DecorationId, Range<u32>)> {
        if r >= self.find_count() {
            return None;
        }
        let (it, _fc, StartDim(base)) =
            self.tree.seek::<FindCountDim, StartDim>(&FindCountDim(r as u32))?;
        debug_assert!(
            matches!(it.kind, DecorationKind::FindMatch),
            "FindCountDim seek must land on the r-th FindMatch (only finds carry the dimension)"
        );
        let start = base + it.gap;
        Some((it.id, start..start + it.len))
    }

    /// Insert an ascending, disjoint `spans` batch confined to a small window into
    /// its index band WITHOUT rematerializing the store — the windowed sibling of
    /// [`add_sorted_batch`](Self::add_sorted_batch) for the per-keystroke find repair. O(band + batch +
    /// log n), mirroring [`take_matching_in`](Self::take_matching_in): only the
    /// existing items whose start falls in `[spans.first().start,
    /// spans.last().start]` sit in the touched band; everything outside is SHARED
    /// (`Arc` clone) and the suffix re-relativizes at one delta-gap seam.
    ///
    /// Ids are minted in span order and exceed every existing id, so a tie at a
    /// shared start places the new item AFTER the existing one — byte-identical to
    /// `set_sorted`. Insertion moves no bytes, so the suffix
    /// `reanchor` runs with `delta = 0`.
    pub fn splice_sorted_batch(
        &mut self,
        spans: &[Range<u32>],
        kind: DecorationKind,
        stickiness: Stickiness,
    ) -> Vec<DecorationId> {
        if spans.is_empty() {
            return Vec::new();
        }
        let mut batch: Vec<TrackedRange> = Vec::with_capacity(spans.len());
        let mut ids = Vec::with_capacity(spans.len());
        for span in spans {
            let id = self.mint();
            ids.push(id);
            batch.push(TrackedRange { id, range: span.clone(), kind: kind.clone(), stickiness });
        }
        // The touched band: existing items whose start ∈ [lo, hi] (every batch
        // start lies here, so the batch merges entirely into it).
        let lo = spans.first().expect("non-empty").start;
        let hi = spans.last().expect("non-empty").start;
        let k_lo = self.tree.summary_before(&StartDim(lo)).count; // start < lo
        let k_hi = self.tree.summary_before(&StartDim(hi.saturating_add(1))).count; // start ≤ hi

        let (left, rest) = self.tree.split_at(&CountDim(k_lo));
        let (middle, zone_c) = rest.split_at(&CountDim(k_hi - k_lo));
        let left_span = left.extent::<StartDim>().0; // absolute start where `middle` begins

        // Merge the batch into the band by (start, id) — both ascending; new ids
        // exceed all existing so ties order new-after-old, matching `set_sorted`.
        let mut merged = decoitems_to_ranges(&middle, left_span);
        merged.extend(batch);
        merged.sort_by(|a, b| a.range.start.cmp(&b.range.start).then(a.id.cmp(&b.id)));
        let middle_new = SumTree::from_items(ranges_to_decoitems(merged, left_span));

        // Positions are unchanged (insertion moves no bytes), so delta = 0: zone_c's
        // first gap just re-relativizes onto the new last band item.
        let head = left.append(&middle_new);
        let prev_start = head.extent::<StartDim>().0;
        let zone_c_new = reanchor(&zone_c, k_hi, &self.tree, 0, prev_start);
        self.tree = head.append(&zone_c_new);
        ids
    }

    /// Per bucket `b` in `0..bounds.len()-1` (ascending offset bounds): the max
    /// Diagnostic severity among the diagnostics *starting* in `[bounds[b],
    /// bounds[b+1])` (encoded `severity + 1`, `1..=4`) and the start offset of the
    /// FIRST severest one there — `(0, 0)` when the bucket holds no diagnostic.
    /// The scrollbar-overview diagnostic lane: O(P + log M) to fold per-bucket max
    /// severity, then O(log M) per non-empty bucket to fetch the winning offset,
    /// so it never scans every diagnostic per frame.
    pub fn diagnostic_overview(&self, bounds: &[u32], out: &mut Vec<(u8, u32)>) {
        out.clear();
        // Phase 1 — per-bucket max severity by summary fold (whole subtrees O(1)).
        let start_bounds: Vec<StartDim> = bounds.iter().map(|&b| StartDim(b)).collect();
        let sev = self.tree.bucketed_reduce(&start_bounds, 0u8, |acc: &mut u8, s: &DecoSummary| {
            *acc = (*acc).max(s.sev_max);
        });
        // Phase 2 — the winner's offset (needed to reproduce the exact draw-y).
        for (b, &max_sev) in sev.iter().enumerate() {
            if max_sev == 0 {
                out.push((0, 0));
            } else {
                let off = self
                    .first_start_with_severity(bounds[b], bounds[b + 1], max_sev)
                    .unwrap_or(bounds[b]);
                out.push((max_sev, off));
            }
        }
    }

    /// Per bucket: the start offset of the first [`DecorationKind::FindMatch`]
    /// starting in `[bounds[b], bounds[b+1])`, or `None`. The overview's find
    /// lane: O(P·log M) via `find_rank_before`/`nth_find`, no whole-store walk.
    pub fn find_overview(&self, bounds: &[u32], out: &mut Vec<Option<u32>>) {
        out.clear();
        for w in bounds.windows(2) {
            let (lo, hi) = (w[0], w[1]);
            let r = self.find_rank_before(lo); // first find with start ≥ lo
            let first = self.nth_find(r).map(|(_, rng)| rng.start).filter(|&s| s < hi);
            out.push(first);
        }
    }

    /// The start offset of the first diagnostic in `[lo, hi)` whose encoded
    /// severity equals `target` — the overview's phase-2 winner lookup. A pruned
    /// [`SumTree::filter_visit`] descent (enter only subtrees whose start-range
    /// overlaps `[lo, hi)` and whose `sev_max >= target`), so O(log M + few); this
    /// is a per-frame read, so it charges neither the work-meter nor the canary.
    fn first_start_with_severity(&self, lo: u32, hi: u32, target: u8) -> Option<u32> {
        let mut found: Option<u32> = None;
        self.tree.filter_visit::<StartDim, _, _>(
            &|before: &StartDim, sum: &DecoSummary| {
                before.0 < hi && before.0 + sum.span >= lo && sum.sev_max >= target
            },
            &mut |it: &DecoItem, before: &StartDim| {
                if found.is_some() {
                    return; // in-order visit: the first hit is THE first severest
                }
                let start = before.0 + it.gap;
                if start >= lo && start < hi {
                    let sev = match it.kind {
                        DecorationKind::Diagnostic { severity, .. } => severity as u8 + 1,
                        _ => 0,
                    };
                    if sev == target {
                        found = Some(start);
                    }
                }
            },
        );
        found
    }

    /// Batch-insert `spans` (ascending, one `kind`/`stickiness`) and return
    /// the minted ids, in span order — ONE sort-and-rebuild for the whole batch
    /// instead of one per span, so installing a 10k-match find set is a single
    /// sort rather than one per match. The find re-scan installs its whole match
    /// set through this.
    pub fn add_sorted_batch(
        &mut self,
        spans: impl IntoIterator<Item = Range<u32>>,
        kind: DecorationKind,
        stickiness: Stickiness,
    ) -> Vec<DecorationId> {
        let mut v = self.to_vec();
        let mut ids = Vec::new();
        for range in spans {
            let id = self.mint();
            v.push(TrackedRange { id, range, kind: kind.clone(), stickiness });
            ids.push(id);
        }
        if !ids.is_empty() {
            self.set_sorted(v); // an empty batch (a find re-scan that found nothing) skips it
        }
        ids
    }

    /// Remove — and return, in store order — every range intersecting `window`
    /// that `pred` accepts. Find's per-commit repair drives this with a small
    /// window, so it is **windowed**: only the decorations that can touch `window`
    /// sit in the index band `[k_lo, k_hi)` (starts in `[wl, qe]`, `wl` the leftmost
    /// touching start), and everything outside is SHARED — a decoration with start
    /// < wl has end < qs so it cannot touch, and start > qe cannot touch either.
    /// Removal keeps positions, so the suffix re-relativizes at one seam
    /// (`reanchor` with delta = 0). **O(band + log n)**, not O(store).
    ///
    /// A whole-store `window` (a find rescan / `clear_autoclose`) makes the band the
    /// whole store — O(store) — but those callers are debounced or early-out, not
    /// the per-keystroke path.
    pub fn take_matching_in(
        &mut self,
        window: Range<u32>,
        mut pred: impl FnMut(&TrackedRange) -> bool,
    ) -> Vec<TrackedRange> {
        let (qs, qe) = (window.start, window.end);
        let Some(wl) = self.decorations_in(qs..qe).map(|r| r.range.start).min() else {
            return Vec::new(); // nothing touches ⇒ nothing removed, tree untouched
        };
        let k_hi = self.tree.summary_before(&StartDim(qe.saturating_add(1))).count; // start ≤ qe
        let k_lo = self.tree.summary_before(&StartDim(wl)).count; // start < wl

        let (left, rest) = self.tree.split_at(&CountDim(k_lo));
        let (middle, zone_c) = rest.split_at(&CountDim(k_hi - k_lo));
        let left_span = left.extent::<StartDim>().0;

        let mut removed = Vec::new();
        let mut survivors = Vec::new();
        for r in decoitems_to_ranges(&middle, left_span) {
            let touches = r.range.start <= qe && r.range.end >= qs;
            if touches && pred(&r) {
                removed.push(r);
            } else {
                survivors.push(r); // ascending order preserved
            }
        }
        if removed.is_empty() {
            return removed; // matched nothing in the band ⇒ leave the tree as-is
        }
        // Positions are unchanged (removal only), so delta = 0: zone_c's first gap
        // just re-relativizes onto the last surviving band item.
        let middle_new = SumTree::from_items(ranges_to_decoitems(survivors, left_span));
        let head = left.append(&middle_new);
        let prev_start = head.extent::<StartDim>().0;
        let zone_c_new = reanchor(&zone_c, k_hi, &self.tree, 0, prev_start);
        self.tree = head.append(&zone_c_new);
        removed
    }

    /// Every tracked range, in ascending `(start, id)` order — the render path's
    /// full-store view before it intersects with the visible rows.
    pub fn iter(&self) -> impl Iterator<Item = TrackedRange> {
        self.to_vec().into_iter()
    }

    /// The eager mover: ride every tracked range across `patch`.
    ///
    /// Each endpoint threads through [`Patch::map_offset`] with its stickiness
    /// bias (start with the start-bias, end with the end-bias); the range never
    /// inverts (a collapse pins both to the mapped end, via
    /// [`Patch::map_range`]). Ranges that collapsed to empty whose kind's
    /// [`empty_policy`](DecorationKind::empty_policy) is [`EmptyPolicy::Drop`]
    /// are removed, and the store is re-sorted (ties can only reorder at a
    /// shared boundary).
    ///
    /// `Patch` already folds a multi-edit transaction into final post-edit
    /// coordinates, so one `map_range` per range applies the whole transaction
    /// at once — no back-to-front bookkeeping here (that lives in the patch).
    pub fn apply_patch(&mut self, patch: &Patch) {
        if patch.is_empty() {
            return;
        }
        // A single edit (the overwhelmingly common keystroke) takes the windowed
        // structural mover: remap only the O(edit-window) decorations the edit can
        // touch and shift the ENTIRE suffix in O(log) by adjusting one delta-gap
        // seam — O(window + log n), not O(n). A multi-edit (multi-caret) patch
        // falls back to the naive whole-store remap: rare with a large decoration
        // set (you don't usually multi-cursor with 10k find matches live), and
        // correctness-first. Both paths are pinned identical by the oracle test
        // `windowed_apply_patch_equals_naive_under_random_single_edits`.
        match patch.edits() {
            [edit] => self.apply_single_edit(patch, edit),
            _ => self.apply_patch_naive(patch),
        }
    }

    /// The naive O(n) mover: materialize the whole store, remap every range, drop
    /// collapsed find matches, re-encode. The reference the windowed path equals,
    /// and the fallback for multi-edit patches.
    fn apply_patch_naive(&mut self, patch: &Patch) {
        let mut v = self.to_vec();
        remap_ranges(patch, &mut v);
        // `map_offset(_, Right)` is NOT monotonic (a start at a replacement's start
        // maps past an interior one — see FoldSet), so starts CAN reorder. Re-sort
        // only when they actually did (the common case rebuilds without the sort).
        if v.windows(2).any(|w| (w[0].range.start, w[0].id) > (w[1].range.start, w[1].id)) {
            self.set_sorted(v);
        } else {
            self.rebuild(v);
        }
    }

    /// The windowed single-edit mover. Partition the store, by *index*, into three
    /// bands relative to the edit `old = os..oe`:
    ///
    /// - **left** — every decoration whose start is `< wl`, where `wl` is the
    ///   leftmost start of any decoration touching `[os, oe]`. Each has `end < os`
    ///   (else it would touch, so its start ≥ wl), so both endpoints are fixed
    ///   points of the edit's map — **shared verbatim**, no work.
    /// - **middle** — starts in `[wl, oe]`: the touched decorations plus any
    ///   straddler reaching in from the left, plus untouched decorations interleaved
    ///   between them (their remap is a no-op). Remapped in absolute coords. `O(window)`.
    /// - **zone_c** — starts `> oe`: entirely after the edit, so a uniform
    ///   `+delta` shift. Delta-gap makes that **one seam-gap adjustment** — every
    ///   interior gap and length is relative and unchanged. `O(log n)`.
    ///
    /// Only the middle is retained for the [`EmptyPolicy::Drop`] collapse rule, and
    /// that is complete: a Drop-kind range collapses only when the edit deletes its
    /// interior, which makes it *touch* the edit — so it is in the middle. The store
    /// never holds a collapsed Drop-kind range (it is never added empty and is
    /// dropped on the edit that collapses it), so left/zone_c need no retain pass.
    fn apply_single_edit(&mut self, patch: &Patch, edit: &crate::patch::Edit) {
        let (os, oe) = (edit.old.start, edit.old.end);
        let delta = (i64::from(edit.new.end) - i64::from(edit.new.start))
            - (i64::from(oe) - i64::from(os));

        // Index band [k_lo, k_hi) = the middle. `wl` extends it left over straddlers.
        let wl = self.decorations_in(os..oe).map(|r| r.range.start).min();
        let k_hi = self.tree.summary_before(&StartDim(oe.saturating_add(1))).count; // start ≤ oe
        let k_lo = match wl {
            Some(wl) => self.tree.summary_before(&StartDim(wl)).count, // start < wl
            None => k_hi, // nothing touches ⇒ empty middle (only a zone_c shift, if any)
        };

        let (left, rest) = self.tree.split_at(&CountDim(k_lo));
        let (middle, zone_c) = rest.split_at(&CountDim(k_hi - k_lo));
        let left_span = left.extent::<StartDim>().0; // absolute start where `middle` begins

        // Remap the middle (absolute), drop collapsed find matches, re-sort — a
        // single edit can reorder starts only inside its own span — delta-gap
        // re-encode with the first gap relative to `left`.
        let mut mv = decoitems_to_ranges(&middle, left_span);
        remap_ranges(patch, &mut mv);
        mv.sort_by(|a, b| a.range.start.cmp(&b.range.start).then(a.id.cmp(&b.id)));
        let middle_new = SumTree::from_items(ranges_to_decoitems(mv, left_span));

        // Reanchor zone_c onto the rebuilt head. `prev_start` is the last start
        // before zone_c; zone_c's first item's start becomes `old_start + delta`.
        let head = left.append(&middle_new);
        let prev_start = head.extent::<StartDim>().0;
        let zone_c_new = reanchor(&zone_c, k_hi, &self.tree, delta, prev_start);
        self.tree = head.append(&zone_c_new);
    }

    /// Publish a diagnostic set against the revision it was computed on.
    ///
    /// 1. **Revision gate**: `revision != current_revision` → return
    ///    [`DiagnosticsOutcome::Stale`] having touched nothing; the previous
    ///    decorations keep riding edits via stickiness, so squiggles stay glued
    ///    to their code until the next set arrives.
    /// 2. **Wholesale replace**: remove every existing
    ///    [`DecorationKind::Diagnostic`] range and install the new set with
    ///    [`Stickiness::NeverGrows`]; severity, message, and code ride the kind
    ///    variant. Non-diagnostic decorations (snippet stops, find matches,
    ///    auto-close) are untouched.
    ///
    /// Span clipping and degenerate-span normalization need the buffer, so they
    /// are the `Document`'s responsibility; this standalone store installs spans
    /// verbatim.
    pub fn set_diagnostics(
        &mut self,
        revision: u64,
        current_revision: u64,
        diags: Vec<Diagnostic>,
    ) -> DiagnosticsOutcome {
        if revision != current_revision {
            return DiagnosticsOutcome::Stale {
                current: current_revision,
            };
        }
        let mut v = self.to_vec();
        v.retain(|r| !matches!(r.kind, DecorationKind::Diagnostic { .. }));
        let count = diags.len();
        for d in diags {
            let id = self.mint();
            v.push(TrackedRange {
                id,
                range: d.span,
                kind: DecorationKind::Diagnostic {
                    severity: d.severity,
                    message: Arc::from(d.message),
                    code: d.code.map(Arc::from),
                },
                stickiness: Stickiness::NeverGrows,
            });
        }
        self.set_sorted(v);
        DiagnosticsOutcome::Applied { count }
    }

    /// Mint the next id and bump the monotonic counter.
    fn mint(&mut self) -> DecorationId {
        let id = DecorationId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Materialize the ranges in store order (accumulating the delta-gaps into
    /// absolute offsets) — the reconstruct handle for the O(n) producer mutations
    /// that reuse the pre-tree Vec logic (add / diagnostics / the naive mover).
    fn to_vec(&self) -> Vec<TrackedRange> {
        decoitems_to_ranges(&self.tree, 0)
    }

    /// Rebuild the tree from `v`, restoring the `(start, id)` order (ids are
    /// unique, so it is total). The sort also repairs the reorder an edit can cause
    /// at a shared boundary; use [`rebuild`](Self::rebuild) when order is already
    /// intact (a removal / a stickiness swap).
    fn set_sorted(&mut self, mut v: Vec<TrackedRange>) {
        #[cfg(any(test, debug_assertions))]
        DECORATION_SORTS.with(|c| c.set(c.get() + 1));
        crate::perf::charge(v.len() as u64); // complexity gate: whole-store sort
        v.sort_by(|a, b| a.range.start.cmp(&b.range.start).then(a.id.cmp(&b.id)));
        self.tree = SumTree::from_items(ranges_to_decoitems(v, 0));
    }

    /// Rebuild the tree from an already-`(start, id)`-ordered `v` (no re-sort).
    fn rebuild(&mut self, v: Vec<TrackedRange>) {
        crate::perf::charge(v.len() as u64); // complexity gate: whole-store pass
        self.tree = SumTree::from_items(ranges_to_decoitems(v, 0));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Bias::{Left, Right};
    use crate::patch::Edit;

    fn store() -> DecorationStore {
        DecorationStore::new()
    }

    #[test]
    fn windowed_query_equals_the_linear_oracle() {
        // A linear filter, kept verbatim as the oracle that the interval-descent
        // decorations_in must match.
        let linear = |s: &DecorationStore, qs: u32, qe: u32| -> Vec<u64> {
            s.iter()
                .filter(|r| r.range.start <= qe && r.range.end >= qs)
                .map(|r| r.id.0)
                .collect()
        };
        // Seeded pseudo-random ranges (incl. zero-width and long spanners).
        let mut rng = 0x9E3779B97F4A7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let mut s = store();
        for _ in 0..300 {
            let a = (next() % 10_000) as u32;
            let len = (next() % 50) as u32 * if next() % 4 == 0 { 40 } else { 1 };
            s.add_decoration(a..a + len, DecorationKind::FindMatch, Stickiness::NeverGrows);
        }
        for _ in 0..200 {
            let qs = (next() % 11_000) as u32;
            let qe = qs + (next() % 400) as u32;
            let fast: Vec<u64> = s.decorations_in(qs..qe).map(|r| r.id.0).collect();
            assert_eq!(fast, linear(&s, qs, qe), "window {qs}..{qe} diverged from the oracle");
        }
        // After removals the bounds must stay exact (the prefix max tracks).
        for _ in 0..50 {
            let victim = DecorationId((next() % 300) + 1);
            let _ = s.take_decoration(victim);
        }
        for _ in 0..100 {
            let qs = (next() % 11_000) as u32;
            let qe = qs + (next() % 400) as u32;
            let fast: Vec<u64> = s.decorations_in(qs..qe).map(|r| r.id.0).collect();
            assert_eq!(fast, linear(&s, qs, qe));
        }
    }

    #[test]
    fn batch_add_and_windowed_take() {
        let mut s = store();
        s.add_decoration(5..8, DecorationKind::SnippetStop { index: 0 }, Stickiness::AlwaysGrows);
        let ids =
            s.add_sorted_batch([0..2, 10..12, 20..22], DecorationKind::FindMatch, Stickiness::NeverGrows);
        assert_eq!(ids.len(), 3);
        // Store order is (start, id) regardless of insertion order.
        let starts: Vec<u32> = s.iter().map(|r| r.range.start).collect();
        assert_eq!(starts, vec![0, 5, 10, 20]);
        // Windowed take removes only matching kinds inside the window…
        let removed = s.take_matching_in(4..15, |r| matches!(r.kind, DecorationKind::FindMatch));
        assert_eq!(removed.len(), 1);
        assert_eq!(removed[0].range, 10..12);
        // …the snippet stop in-window and the match outside both survive.
        let left: Vec<u32> = s.iter().map(|r| r.range.start).collect();
        assert_eq!(left, vec![0, 5, 20]);
        // And queries stay exact after the batch removal.
        assert_eq!(s.decorations_in(0..u32::MAX).count(), 3);
    }

    fn ins(at: u32, len: u32) -> Patch {
        Patch::single(Edit { old: at..at, new: at..at + len })
    }

    fn del(start: u32, end: u32) -> Patch {
        Patch::single(Edit { old: start..end, new: start..start })
    }

    // Apply `patch` to a single `stickiness` decoration over 4..8 and read back
    // its moved range. `AutoClosePair` keeps on collapse, so delete cases don't
    // vanish from under the test.
    fn moved(stickiness: Stickiness, patch: &Patch) -> Range<u32> {
        let mut s = store();
        let id = s.add_decoration(4..8, DecorationKind::AutoClosePair, stickiness);
        s.apply_patch(patch);
        s.decoration_range(id).unwrap()
    }

    #[test]
    fn biases_are_the_four_named_pairs() {
        assert_eq!(Stickiness::AlwaysGrows.biases(), (Left, Right));
        assert_eq!(Stickiness::NeverGrows.biases(), (Right, Left));
        assert_eq!(Stickiness::GrowsOnlyBefore.biases(), (Left, Left));
        assert_eq!(Stickiness::GrowsOnlyAfter.biases(), (Right, Right));
    }

    #[test]
    fn severity_orders_hint_to_error() {
        assert!(Severity::Hint < Severity::Info);
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn only_find_match_drops_on_collapse() {
        assert_eq!(DecorationKind::FindMatch.empty_policy(), EmptyPolicy::Drop);
        assert_eq!(DecorationKind::AutoClosePair.empty_policy(), EmptyPolicy::Keep);
        assert_eq!(
            DecorationKind::SnippetStop { index: 0 }.empty_policy(),
            EmptyPolicy::Keep
        );
        assert_eq!(
            DecorationKind::Diagnostic {
                severity: Severity::Error,
                message: Arc::from("x"),
                code: None,
            }
            .empty_policy(),
            EmptyPolicy::Keep
        );
    }

    #[test]
    fn ids_are_monotonic_and_never_reused() {
        let mut s = store();
        let a = s.add_decoration(0..1, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        let b = s.add_decoration(2..3, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        assert!(a < b);
        // Removing `a` does not free its id: the next add is strictly greater.
        s.take_decoration(a);
        let c = s.add_decoration(0..1, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        assert!(c > b);
    }

    #[test]
    fn take_returns_the_range_then_forgets_it() {
        let mut s = store();
        let id = s.add_decoration(3..5, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        let taken = s.take_decoration(id).unwrap();
        assert_eq!(taken.range, 3..5);
        assert_eq!(s.decoration_range(id), None);
        assert!(s.take_decoration(id).is_none()); // idempotent
    }

    #[test]
    fn set_stickiness_swaps_or_reports_gone() {
        let mut s = store();
        let id = s.add_decoration(0..1, DecorationKind::SnippetStop { index: 0 }, Stickiness::NeverGrows);
        assert!(s.set_decoration_stickiness(id, Stickiness::AlwaysGrows));
        // The swap is observable through the mover: typing at the end now grows.
        s.apply_patch(&ins(1, 2));
        assert_eq!(s.decoration_range(id), Some(0..3));
        // A gone id reports false.
        s.take_decoration(id);
        assert!(!s.set_decoration_stickiness(id, Stickiness::NeverGrows));
    }

    #[test]
    fn decorations_in_counts_touching() {
        let mut s = store();
        let a = s.add_decoration(0..2, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        let b = s.add_decoration(5..5, DecorationKind::AutoClosePair, Stickiness::NeverGrows); // zero-width
        let c = s.add_decoration(8..10, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        // Query 2..8 touches `a` at 2, includes the zero-width `b` at 5, and
        // touches `c` at 8.
        let got: Vec<_> = s.decorations_in(2..8).map(|r| r.id).collect();
        assert_eq!(got, vec![a, b, c]);
        // A query strictly before everything hits nothing.
        assert_eq!(s.decorations_in(20..21).count(), 0);
    }

    #[test]
    fn decorations_in_is_ascending_by_start_then_id() {
        let mut s = store();
        // Insert out of order; the store sorts.
        s.add_decoration(9..9, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        s.add_decoration(1..2, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        s.add_decoration(1..1, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        let starts: Vec<_> = s.decorations_in(0..100).map(|r| r.range.start).collect();
        assert_eq!(starts, vec![1, 1, 9]);
        // The two at start 1 tie-break on ascending id (insertion order here).
        let ids: Vec<_> = s.decorations_in(0..2).map(|r| r.id).collect();
        assert!(ids[0] < ids[1]);
    }

    // --- The stickiness table: 4 modes × 5 edit shapes. ---

    #[test]
    fn table_insert_at_start() {
        let p = ins(4, 2); // insert 2 bytes at the start edge (offset 4)
        assert_eq!(moved(Stickiness::AlwaysGrows, &p), 4..10); // Left start swallows
        assert_eq!(moved(Stickiness::NeverGrows, &p), 6..10); // Right start rejects
        assert_eq!(moved(Stickiness::GrowsOnlyBefore, &p), 4..10);
        assert_eq!(moved(Stickiness::GrowsOnlyAfter, &p), 6..10);
    }

    #[test]
    fn table_insert_at_end() {
        let p = ins(8, 2); // insert 2 bytes at the end edge (offset 8)
        assert_eq!(moved(Stickiness::AlwaysGrows, &p), 4..10); // Right end swallows
        assert_eq!(moved(Stickiness::NeverGrows, &p), 4..8); // Left end rejects
        assert_eq!(moved(Stickiness::GrowsOnlyBefore, &p), 4..8);
        assert_eq!(moved(Stickiness::GrowsOnlyAfter, &p), 4..10);
    }

    #[test]
    fn table_insert_inside_always_grows() {
        let p = ins(6, 2); // insert strictly inside 4..8
        for m in [
            Stickiness::AlwaysGrows,
            Stickiness::NeverGrows,
            Stickiness::GrowsOnlyBefore,
            Stickiness::GrowsOnlyAfter,
        ] {
            assert_eq!(moved(m, &p), 4..10, "{m:?} must grow on interior insert");
        }
    }

    #[test]
    fn table_delete_around_collapses_without_inverting() {
        let p = del(2, 10); // delete a range that fully contains 4..8
        for m in [
            Stickiness::AlwaysGrows,
            Stickiness::NeverGrows,
            Stickiness::GrowsOnlyBefore,
            Stickiness::GrowsOnlyAfter,
        ] {
            let r = moved(m, &p);
            assert!(r.start <= r.end, "{m:?} inverted: {r:?}");
            assert_eq!(r, 2..2, "{m:?} collapses to the mapped end");
        }
    }

    #[test]
    fn table_delete_prefix_pins_start_and_shifts_end() {
        let p = del(4, 6); // delete the first half of 4..8
        for m in [
            Stickiness::AlwaysGrows,
            Stickiness::NeverGrows,
            Stickiness::GrowsOnlyBefore,
            Stickiness::GrowsOnlyAfter,
        ] {
            // A deletion never drags the start boundary right; the end shifts -2.
            assert_eq!(moved(m, &p), 4..6, "{m:?}");
        }
    }

    #[test]
    fn edit_before_a_range_shifts_it_rigidly() {
        let p = ins(0, 3); // insert 3 bytes before everything
        assert_eq!(moved(Stickiness::NeverGrows, &p), 7..11);
    }

    #[test]
    fn apply_patch_drops_collapsed_find_match_keeps_others() {
        let mut s = store();
        let fm = s.add_decoration(4..8, DecorationKind::FindMatch, Stickiness::NeverGrows);
        let diag = s.add_decoration(
            4..8,
            DecorationKind::Diagnostic {
                severity: Severity::Error,
                message: Arc::from("boom"),
                code: None,
            },
            Stickiness::NeverGrows,
        );
        s.apply_patch(&del(2, 10)); // collapse both to 2..2
        assert_eq!(s.decoration_range(fm), None, "find match dropped on collapse");
        assert_eq!(
            s.decoration_range(diag),
            Some(2..2),
            "diagnostic kept (owner lifetime), rendered ≥1 cell later"
        );
    }

    #[test]
    fn empty_patch_leaves_the_store_untouched() {
        let mut s = store();
        let id = s.add_decoration(4..8, DecorationKind::FindMatch, Stickiness::NeverGrows);
        s.apply_patch(&Patch::new());
        assert_eq!(s.decoration_range(id), Some(4..8));
    }

    #[test]
    fn apply_patch_reestablishes_sort_order() {
        let mut s = store();
        // `a` sits before `b`; a big insert at the front of `a` pushes it past
        // `b` only if it grows past it — here they just shift and keep order.
        let a = s.add_decoration(0..1, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        let b = s.add_decoration(5..6, DecorationKind::AutoClosePair, Stickiness::NeverGrows);
        s.apply_patch(&ins(2, 10)); // between a and b
        let order: Vec<_> = s.iter().map(|r| r.id).collect();
        assert_eq!(order, vec![a, b]);
        assert_eq!(s.decoration_range(a), Some(0..1));
        assert_eq!(s.decoration_range(b), Some(15..16));
    }

    // --- Diagnostics ingestion. ---

    #[test]
    fn set_diagnostics_replaces_wholesale() {
        let mut s = store();
        let outcome = s.set_diagnostics(
            0,
            0,
            vec![
                Diagnostic::new(0..3, Severity::Warning, "one"),
                Diagnostic::new(5..8, Severity::Error, "two"),
            ],
        );
        assert_eq!(outcome, DiagnosticsOutcome::Applied { count: 2 });
        assert_eq!(s.len(), 2);
        // A second publish replaces the first set entirely.
        let outcome = s.set_diagnostics(0, 0, vec![Diagnostic::new(1..2, Severity::Hint, "solo")]);
        assert_eq!(outcome, DiagnosticsOutcome::Applied { count: 1 });
        assert_eq!(s.len(), 1);
        let only = s.iter().next().unwrap();
        assert_eq!(only.range, 1..2);
        assert_eq!(only.stickiness, Stickiness::NeverGrows);
    }

    #[test]
    fn set_diagnostics_stale_drops_and_keeps_previous() {
        let mut s = store();
        s.set_diagnostics(0, 0, vec![Diagnostic::new(0..3, Severity::Error, "kept")]);
        // Published against revision 0, but the store is now at revision 2.
        let outcome = s.set_diagnostics(0, 2, vec![Diagnostic::new(0..3, Severity::Info, "stale")]);
        assert_eq!(outcome, DiagnosticsOutcome::Stale { current: 2 });
        // The previous set survived untouched.
        assert_eq!(s.len(), 1);
        let kept = s.iter().next().unwrap().clone();
        match &kept.kind {
            DecorationKind::Diagnostic { message, severity, .. } => {
                assert_eq!(&**message, "kept");
                assert_eq!(*severity, Severity::Error);
            }
            other => panic!("expected a diagnostic, got {other:?}"),
        }
    }

    #[test]
    fn set_diagnostics_leaves_other_kinds_alone() {
        let mut s = store();
        let stop = s.add_decoration(9..9, DecorationKind::SnippetStop { index: 0 }, Stickiness::AlwaysGrows);
        s.set_diagnostics(0, 0, vec![Diagnostic::new(0..3, Severity::Error, "e")]);
        // Re-publishing diagnostics wipes only diagnostics; the snippet stop stays.
        s.set_diagnostics(0, 0, vec![Diagnostic::new(1..2, Severity::Warning, "w")]);
        assert_eq!(s.decoration_range(stop), Some(9..9));
        let diag_count = s
            .iter()
            .filter(|r| matches!(r.kind, DecorationKind::Diagnostic { .. }))
            .count();
        assert_eq!(diag_count, 1);
    }

    #[test]
    fn diagnostic_content_rides_the_kind() {
        let mut s = store();
        s.set_diagnostics(
            3,
            3,
            vec![Diagnostic {
                span: 0..4,
                severity: Severity::Warning,
                message: "unused".to_string(),
                code: Some("W0612".to_string()),
            }],
        );
        let got = s.iter().next().unwrap().clone();
        match &got.kind {
            DecorationKind::Diagnostic { severity, message, code } => {
                assert_eq!(*severity, Severity::Warning);
                assert_eq!(&**message, "unused");
                assert_eq!(code.as_deref(), Some("W0612"));
            }
            other => panic!("expected a diagnostic, got {other:?}"),
        }
    }

    #[test]
    fn diagnostics_ride_edits_between_publishes() {
        // Continuity: a squiggle stays glued to its code as the user types above
        // it, until the next publish re-sets it.
        let mut s = store();
        s.set_diagnostics(0, 0, vec![Diagnostic::new(10..14, Severity::Error, "here")]);
        let id = s.iter().next().unwrap().id;
        s.apply_patch(&ins(0, 5)); // type 5 bytes at the top of the file
        assert_eq!(s.decoration_range(id), Some(15..19));
    }

    #[test]
    fn windowed_apply_patch_equals_naive_under_random_single_edits() {
        // The windowed single-edit mover must produce byte-identical state to the
        // naive whole-store remap for EVERY store shape and single edit —
        // straddlers reaching in from the left, collapses, in-span reorders, an
        // empty middle, an empty zone_c, and edits before / inside / after the
        // decorations. xorshift-seeded, deep-compared over both survival and range.
        let mut rng = 0xD1CE_5EED_u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let kind_of = |t: u64| -> (DecorationKind, Stickiness) {
            match t % 4 {
                0 => (DecorationKind::FindMatch, Stickiness::NeverGrows), // drops on collapse
                1 => (DecorationKind::AutoClosePair, Stickiness::AlwaysGrows),
                2 => (DecorationKind::SnippetStop { index: 0 }, Stickiness::GrowsOnlyBefore),
                _ => (
                    DecorationKind::Diagnostic {
                        severity: Severity::Error,
                        message: Arc::from("x"),
                        code: None,
                    },
                    Stickiness::GrowsOnlyAfter,
                ),
            }
        };
        // Project to (id, start, end): the mover carries kind/stickiness unchanged,
        // so this captures every observable it can affect (survival + position).
        let proj = |s: &DecorationStore| -> Vec<(u64, u32, u32)> {
            s.iter().map(|r| (r.id.0, r.range.start, r.range.end)).collect()
        };
        for trial in 0..4000u32 {
            let mut windowed = store();
            let mut naive = store();
            let n = next() % 40;
            for _ in 0..n {
                let a = (next() % 120) as u32;
                let (kind, stick) = kind_of(next());
                let mut len = (next() % 25) as u32;
                // A Drop-kind (FindMatch) decoration is never zero-width in practice
                // (a needle has length ≥ 1) and the store never holds a *collapsed*
                // one — it is dropped on the edit that collapses it and never added
                // empty. The windowed mover leans on that invariant to skip the
                // untouched bands, so honor it here (a zero-width KEEP decoration —
                // a snippet stop / diagnostic point — is fine and still exercised).
                if matches!(kind, DecorationKind::FindMatch) && len == 0 {
                    len = 1;
                }
                // Identical op sequence ⇒ identical minted ids in both stores.
                windowed.add_decoration(a..a + len, kind.clone(), stick);
                naive.add_decoration(a..a + len, kind, stick);
            }
            // A single edit `old = os..oe` → `new = os..os+ins` (ns == os).
            let os = (next() % 120) as u32;
            let oe = os + (next() % 15) as u32;
            let ins = (next() % 15) as u32;
            let patch = Patch::single(Edit { old: os..oe, new: os..os + ins });
            windowed.apply_patch(&patch); // single edit ⇒ windowed path
            naive.apply_patch_naive(&patch); // the reference
            assert_eq!(
                proj(&windowed),
                proj(&naive),
                "trial {trial}: edit {os}..{oe} → +{ins}"
            );
        }
    }

    #[test]
    fn windowed_apply_patch_is_sublinear_in_store_size() {
        use crate::sum_tree::NODE_ALLOCS;
        // The delta-gap payoff, pinned: a single-edit commit on a store with N find
        // matches must allocate O(edit-window + log N) tree nodes — NOT O(N). The
        // naive whole-store rebuild allocates ~N/B leaves + spine, so it reads ~4×
        // across a 4× store; the windowed mover stays ~flat (only log growth). This
        // cell trips the moment `apply_patch` reverts to a whole-store rebuild.
        let build = |n: u32| -> DecorationStore {
            let mut s = store();
            let spans: Vec<Range<u32>> = (0..n).map(|i| (i * 10)..(i * 10 + 3)).collect();
            s.add_sorted_batch(spans, DecorationKind::FindMatch, Stickiness::NeverGrows);
            s
        };
        let allocs_for = |n: u32| -> f64 {
            let mut s = build(n);
            // Insert near the FRONT: the whole N-match suffix must shift — the case
            // that is O(N) with absolute offsets and O(log N) with delta-gap (one
            // reanchored seam gap moves every later match for free).
            let patch = Patch::single(Edit { old: 5..5, new: 5..6 });
            NODE_ALLOCS.with(|c| c.set(0));
            s.apply_patch(&patch);
            NODE_ALLOCS.with(std::cell::Cell::get) as f64
        };
        let (small, big) = (allocs_for(1000), allocs_for(4000));
        eprintln!("[decorations] apply_patch node allocs {small} -> {big}  ({:.2}x)", big / small);
        assert!(
            big <= small * 2.0,
            "apply_patch allocates superlinearly ({small} -> {big} nodes): the mover \
             reverted to a whole-store rebuild instead of shifting the suffix at one seam"
        );
    }

    #[test]
    fn windowed_take_matching_in_equals_naive() {
        // The windowed removal must take exactly what a naive whole-store scan would
        // (touching ∧ pred), leave the rest in order with positions intact. Compared
        // against a plain Vec model over random stores and windows.
        let mut rng = 0x7A6E_1234_u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for trial in 0..3000u32 {
            let mut s = store();
            let n = next() % 40;
            for _ in 0..n {
                let a = (next() % 120) as u32;
                let len = (next() % 25) as u32;
                let (kind, stick) = match next() % 3 {
                    0 => (DecorationKind::FindMatch, Stickiness::NeverGrows),
                    1 => (DecorationKind::AutoClosePair, Stickiness::AlwaysGrows),
                    _ => (DecorationKind::SnippetStop { index: 0 }, Stickiness::NeverGrows),
                };
                s.add_decoration(a..a + len, kind, stick);
            }
            let before: Vec<(u64, u32, u32, bool)> = s
                .iter()
                .map(|r| {
                    (r.id.0, r.range.start, r.range.end, matches!(r.kind, DecorationKind::FindMatch))
                })
                .collect();
            let qs = (next() % 120) as u32;
            let qe = qs + (next() % 30) as u32;
            let removed: Vec<u64> = s
                .take_matching_in(qs..qe, |r| matches!(r.kind, DecorationKind::FindMatch))
                .iter()
                .map(|r| r.id.0)
                .collect();
            let touched = |st: u32, en: u32| st <= qe && en >= qs;
            let exp_removed: Vec<u64> = before
                .iter()
                .filter(|&&(_, st, en, f)| f && touched(st, en))
                .map(|&(id, ..)| id)
                .collect();
            let exp_surv: Vec<(u64, u32, u32)> = before
                .iter()
                .filter(|&&(_, st, en, f)| !(f && touched(st, en)))
                .map(|&(id, st, en, _)| (id, st, en))
                .collect();
            let got_surv: Vec<(u64, u32, u32)> =
                s.iter().map(|r| (r.id.0, r.range.start, r.range.end)).collect();
            assert_eq!(removed, exp_removed, "trial {trial}: removed for window {qs}..{qe}");
            assert_eq!(got_surv, exp_surv, "trial {trial}: survivors for window {qs}..{qe}");
        }
    }

    #[test]
    fn windowed_take_matching_in_is_sublinear_in_store_size() {
        use crate::sum_tree::NODE_ALLOCS;
        // Find's per-commit repair: removing a small window's matches from an N-match
        // store must be O(window + log N) node allocs, not O(N) (a whole-store scan
        // + rebuild). Same allocation proof as the apply_patch cell.
        let build = |n: u32| -> DecorationStore {
            let mut s = store();
            let spans: Vec<Range<u32>> = (0..n).map(|i| (i * 10)..(i * 10 + 3)).collect();
            s.add_sorted_batch(spans, DecorationKind::FindMatch, Stickiness::NeverGrows);
            s
        };
        let allocs_for = |n: u32| -> f64 {
            let mut s = build(n);
            NODE_ALLOCS.with(|c| c.set(0));
            s.take_matching_in(30..45, |r| matches!(r.kind, DecorationKind::FindMatch));
            NODE_ALLOCS.with(std::cell::Cell::get) as f64
        };
        let (small, big) = (allocs_for(1000), allocs_for(4000));
        eprintln!("[decorations] take_matching_in node allocs {small} -> {big}  ({:.2}x)", big / small);
        assert!(
            big <= small * 2.0,
            "take_matching_in allocates superlinearly ({small} -> {big} nodes): it scanned \
             and rebuilt the whole store instead of splicing the window band"
        );
    }

    // ─── Foundation oracles ─────────────────────────────────────────────────

    fn xorshift(seed: u64) -> impl FnMut() -> u64 {
        let mut rng = seed;
        move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        }
    }

    fn diag(sev: Severity) -> DecorationKind {
        DecorationKind::Diagnostic { severity: sev, message: Arc::from("x"), code: None }
    }

    #[test]
    fn splice_batch_equals_naive() {
        // `splice_sorted_batch` (the windowed find-repair insert) must produce a
        // store byte-identical to the naive `to_vec()` + push-batch + `set_sorted`
        // whole rebuild, for every store shape and every ascending in-window batch
        // — the same mint-order, the same (start,id) placement, the same suffix.
        let mut next = xorshift(0xBEEF_1234);
        let kind_of = |t: u64| -> (DecorationKind, Stickiness) {
            match t % 4 {
                0 => (DecorationKind::FindMatch, Stickiness::NeverGrows),
                1 => (DecorationKind::AutoClosePair, Stickiness::AlwaysGrows),
                2 => (DecorationKind::SnippetStop { index: 0 }, Stickiness::GrowsOnlyBefore),
                _ => (diag(Severity::Warning), Stickiness::NeverGrows),
            }
        };
        let proj = |s: &DecorationStore| -> Vec<(u64, u32, u32)> {
            s.iter().map(|r| (r.id.0, r.range.start, r.range.end)).collect()
        };
        for trial in 0..4000u32 {
            let mut a = store();
            let n = next() % 40;
            for _ in 0..n {
                let start = (next() % 200) as u32;
                let len = (next() % 20) as u32;
                let (kind, stick) = kind_of(next());
                a.add_decoration(start..start + len, kind, stick);
            }
            let mut b = a.clone(); // shares next_id ⇒ identical minted ids

            // A random ascending (possibly duplicate-start) in-window batch.
            let base = (next() % 180) as u32;
            let width = 1 + (next() % 30) as u32;
            let count = next() % 8;
            let mut starts: Vec<u32> =
                (0..count).map(|_| base + (next() % u64::from(width)) as u32).collect();
            starts.sort_unstable();
            let spans: Vec<Range<u32>> =
                starts.iter().map(|&s| s..s + (next() % 5) as u32).collect();

            let ids_a =
                a.splice_sorted_batch(&spans, DecorationKind::FindMatch, Stickiness::NeverGrows);

            // Naive reference: mint the same ids, push, whole-store re-sort.
            let mut v = b.to_vec();
            let mut ids_b = Vec::new();
            for span in &spans {
                let id = b.mint();
                ids_b.push(id);
                v.push(TrackedRange {
                    id,
                    range: span.clone(),
                    kind: DecorationKind::FindMatch,
                    stickiness: Stickiness::NeverGrows,
                });
            }
            if !spans.is_empty() {
                b.set_sorted(v);
            }
            assert_eq!(ids_a, ids_b, "trial {trial}: minted ids diverged");
            assert_eq!(proj(&a), proj(&b), "trial {trial}: batch {spans:?}");
        }
    }

    #[test]
    fn find_count_rank_nth_agree_with_linear() {
        // `find_count` / `find_rank_before(x)` / `nth_find(r)` must equal a linear
        // walk over the FindMatch decorations (in (start,id) order) for every r and
        // a spread of offsets, over dense find/diagnostic/snippet interleavings.
        let mut next = xorshift(0xF19D_5EED);
        for trial in 0..3000u32 {
            let mut s = store();
            let n = next() % 60;
            for _ in 0..n {
                let start = (next() % 300) as u32;
                let len = (next() % 10) as u32;
                let (kind, stick) = match next() % 3 {
                    0 => (DecorationKind::FindMatch, Stickiness::NeverGrows),
                    1 => (diag(Severity::Error), Stickiness::NeverGrows),
                    _ => (DecorationKind::SnippetStop { index: 0 }, Stickiness::AlwaysGrows),
                };
                s.add_decoration(start..start + len, kind, stick);
            }
            let finds: Vec<(u64, u32, u32)> = s
                .iter()
                .filter(|r| matches!(r.kind, DecorationKind::FindMatch))
                .map(|r| (r.id.0, r.range.start, r.range.end))
                .collect();
            assert_eq!(s.find_count(), finds.len(), "trial {trial}: find_count");
            for off in [0u32, 1, 50, 100, 150, 200, 299, 300, u32::MAX] {
                let want = finds.iter().filter(|(_, st, _)| *st < off).count();
                assert_eq!(s.find_rank_before(off), want, "trial {trial}: rank before {off}");
            }
            for (r, want) in finds.iter().enumerate() {
                let (id, rng) = s.nth_find(r).expect("r < find_count");
                assert_eq!((id.0, rng.start, rng.end), *want, "trial {trial}: nth_find({r})");
            }
            assert!(s.nth_find(finds.len()).is_none(), "trial {trial}: nth_find past the end");
        }
    }

    #[test]
    fn overview_reduce_equals_linear_scan() {
        // Both overview lanes must equal a naive per-bucket scan for random stores
        // and random monotonic bounds: diagnostic → (max encoded severity, offset
        // of the first item at that severity); find → the first FindMatch start.
        // Bucket membership is by START ∈ [bounds[b], bounds[b+1]).
        let mut next = xorshift(0x0FF3_1234);
        for trial in 0..2500u32 {
            let mut s = store();
            let n = next() % 60;
            for _ in 0..n {
                let start = (next() % 400) as u32;
                let len = (next() % 8) as u32;
                let (kind, stick) = match next() % 5 {
                    0 | 1 => {
                        let sev = match next() % 4 {
                            0 => Severity::Hint,
                            1 => Severity::Info,
                            2 => Severity::Warning,
                            _ => Severity::Error,
                        };
                        (diag(sev), Stickiness::NeverGrows)
                    }
                    2 | 3 => (DecorationKind::FindMatch, Stickiness::NeverGrows),
                    _ => (DecorationKind::SnippetStop { index: 0 }, Stickiness::AlwaysGrows),
                };
                s.add_decoration(start..start + len, kind, stick);
            }
            // Random ascending bounds (duplicates ⇒ empty buckets; may reach below
            // and above the populated offset range).
            let m = 2 + (next() % 8) as usize;
            let mut bounds: Vec<u32> = (0..m).map(|_| (next() % 450) as u32).collect();
            bounds.sort_unstable();

            let items: Vec<TrackedRange> = s.iter().collect();

            let mut diag_out = Vec::new();
            s.diagnostic_overview(&bounds, &mut diag_out);
            let want_diag: Vec<(u8, u32)> = bounds
                .windows(2)
                .map(|w| {
                    let (lo, hi) = (w[0], w[1]);
                    let mut max_sev = 0u8;
                    let mut off = 0u32;
                    for r in &items {
                        if r.range.start >= lo && r.range.start < hi {
                            if let DecorationKind::Diagnostic { severity, .. } = r.kind {
                                let sev = severity as u8 + 1;
                                if sev > max_sev {
                                    max_sev = sev;
                                    off = r.range.start;
                                }
                            }
                        }
                    }
                    (max_sev, off)
                })
                .collect();
            assert_eq!(diag_out, want_diag, "trial {trial}: diag, bounds {bounds:?}");

            let mut find_out = Vec::new();
            s.find_overview(&bounds, &mut find_out);
            let want_find: Vec<Option<u32>> = bounds
                .windows(2)
                .map(|w| {
                    let (lo, hi) = (w[0], w[1]);
                    items
                        .iter()
                        .find(|r| {
                            matches!(r.kind, DecorationKind::FindMatch)
                                && r.range.start >= lo
                                && r.range.start < hi
                        })
                        .map(|r| r.range.start)
                })
                .collect();
            assert_eq!(find_out, want_find, "trial {trial}: find, bounds {bounds:?}");
        }
    }

    #[test]
    fn splice_sorted_batch_is_sublinear_in_store_size() {
        use crate::sum_tree::NODE_ALLOCS;
        // The windowed find-repair insert must allocate O(window + log N) tree
        // nodes for a tiny in-window batch, NOT O(N) — a whole-store rebuild
        // (`to_vec()` + `set_sorted`) reads ~4× across a 4× store and trips this
        // cell; the windowed splice stays flat (only log growth).
        let build = |n: u32| -> DecorationStore {
            let mut s = store();
            let spans: Vec<Range<u32>> = (0..n).map(|i| (i * 10)..(i * 10 + 3)).collect();
            s.add_sorted_batch(spans, DecorationKind::FindMatch, Stickiness::NeverGrows);
            s
        };
        let allocs_for = |n: u32| -> f64 {
            let mut s = build(n);
            let spans = [35..37, 36..39, 44..46]; // three matches into the local band
            NODE_ALLOCS.with(|c| c.set(0));
            s.splice_sorted_batch(&spans, DecorationKind::FindMatch, Stickiness::NeverGrows);
            NODE_ALLOCS.with(std::cell::Cell::get) as f64
        };
        let (small, big) = (allocs_for(1000), allocs_for(4000));
        eprintln!("[decorations] splice_sorted_batch node allocs {small} -> {big}  ({:.2}x)", big / small);
        assert!(
            big <= small * 2.0,
            "splice_sorted_batch allocates superlinearly ({small} -> {big} nodes): it rebuilt \
             the whole store instead of splicing the window band"
        );
    }
}
