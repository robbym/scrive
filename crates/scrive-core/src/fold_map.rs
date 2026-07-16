//! Code folding — **bracket-anchored**. A fold is a collapsed bracket
//! *pair*, identified by the byte offset of its **opener** (`([{`). That offset
//! is the only thing stored; it rides each committed transaction's [`Patch`] via
//! [`Patch::map_offset`], and the fold's *extent* (which rows are hidden) is
//! re-derived from the live [`Brackets`] pass every frame.
//!
//! Storing only the opener (never a hidden-interior byte range) is what keeps a
//! fold from drifting onto an unrelated span: there is no interior offset to go
//! stale. Reconcile is then trivial — a fold survives iff its opener is still a
//! live, foldable open bracket. Rows are reconstructed each frame, so there is no
//! persisted visual index to desync.
//!
//! **View state, not document state:** folds never enter the buffer or the undo
//! stack; they compose over the buffer as a second display layer the
//! render/movement/hit-test paths consult.

use crate::bracket::Brackets;
use crate::buffer::Buffer;
use crate::coords::{Bias, Point};
use crate::display_map::{BufferRow, DisplayRow};
use crate::offset_set::OffsetSet;
use crate::patch::Patch;
use crate::sum_tree::{Dimension, Item, SumTree, Summary};

// Op-count canary: the number of full `FoldMap::new` builds on this thread. The
// per-keystroke analog of the widget's draw-budget gate — a document-scale test
// asserts a plain keystroke / arrow key does NOT rebuild the whole map (which is
// O(folds)); it shifts the cache in place or reads it. Thread-local so parallel
// tests (which build fresh maps constantly) don't race the count. Debug/test
// only; zero-cost in release.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static FOLD_BUILDS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// Op-count canary: `FoldSet::reconcile` scans (the O(folds) retain) on this
// thread. A test asserts a plain keystroke / Enter (no bracket char changed, so
// no fold can lose its pair) does NOT scan the fold set, while a brace deletion
// does. Debug/test only; zero-cost in release.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static RECONCILE_SCANS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The persistent, document-owned set of active folds. Each entry is the byte
/// offset of one collapsed pair's **opening bracket**, kept sorted and unique.
/// Rides [`Patch`] remap in the commit path (see [`FoldSet::apply_patch`]); the
/// extent is resolved from the [`Brackets`] pass in [`FoldMap::new`], so folds
/// survive edits and undo/redo with no interior bookkeeping to drift.
#[derive(Clone, Default, Debug)]
pub struct FoldSet {
    /// Collapsed pairs' opening-bracket byte offsets, on a delta-gap SumTree
    /// ([`OffsetSet`]) so a text edit shifts the whole set without the flat Vec's
    /// O(folds) rebase; membership / window / first-at are O(log n).
    offsets: OffsetSet,
    /// Bumped on every mutation — the memoization key for the document's cached
    /// [`FoldMap`] (folds are view state, so an edit's buffer revision does NOT
    /// cover a pure fold toggle; this does).
    generation: u64,
}

impl FoldSet {
    /// An empty fold set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Whether nothing is folded.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.offsets.is_empty()
    }

    /// The number of active folds.
    #[must_use]
    pub fn len(&self) -> usize {
        self.offsets.len()
    }

    /// The smallest folded opener at byte `offset` or after, if any — an O(log)
    /// probe into the opener-sorted set (e.g. "the fold headed on this row"),
    /// not a scan of every fold.
    #[must_use]
    pub fn first_at_or_after(&self, offset: u32) -> Option<u32> {
        self.offsets.first_at_or_after(offset)
    }

    /// The mutation counter — the document's `FoldMap` cache invalidates when it
    /// changes (see the struct field).
    #[must_use]
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Note a mutation for the cache key.
    fn bump(&mut self) {
        self.generation = self.generation.wrapping_add(1);
    }

    /// Unfold everything.
    pub fn clear(&mut self) {
        self.offsets = OffsetSet::new();
        self.bump();
    }

    /// Collapse the pair whose opener is at byte offset `open`. No-op returning
    /// `false` if that opener is already folded. The caller vouches that `open` is
    /// a foldable open bracket (see [`crate::Document::fold`]).
    pub fn fold(&mut self, open: u32) -> bool {
        if self.offsets.insert(open) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Unfold the pair whose opener is at `open`. Returns whether one was removed.
    pub fn unfold(&mut self, open: u32) -> bool {
        if self.offsets.remove(open) {
            self.bump();
            true
        } else {
            false
        }
    }

    /// Collapse every opener in `opens` in ONE pass — one set mutation for the
    /// whole batch. A multi-caret `Ctrl+Shift+[` can collapse thousands of pairs at
    /// once; calling [`Self::fold`] per item would be O(folds) each and O(folds²)
    /// overall, so the batch path keeps a document-scale collapse linear. The
    /// caller vouches each offset is a foldable open bracket.
    pub fn fold_all(&mut self, opens: impl IntoIterator<Item = u32>) {
        for open in opens {
            self.offsets.insert(open); // idempotent; dedups
        }
        self.bump();
    }

    /// Unfold every opener in `opens` in ONE pass — the batch mirror of
    /// [`Self::unfold`].
    pub fn unfold_all(&mut self, opens: &[u32]) {
        if opens.is_empty() {
            return;
        }
        self.offsets.remove_all(opens);
        self.bump();
    }

    /// Whether the pair opening at `open` is folded.
    #[must_use]
    pub fn is_folded(&self, open: u32) -> bool {
        self.offsets.contains(open)
    }

    /// Fold the pair at `open` if not folded, else unfold it.
    pub fn toggle(&mut self, open: u32) -> bool {
        if self.is_folded(open) {
            self.unfold(open)
        } else {
            self.fold(open)
        }
    }

    /// Shift every opener through a committed patch. Bias `Right` keeps the
    /// offset tracking its bracket char (an insertion at the bracket pushes both
    /// right together). An edit that deletes the opener maps it into the deletion;
    /// [`FoldSet::reconcile`] then drops it (it is no longer a live bracket).
    pub fn apply_patch(&mut self, patch: &Patch) {
        // Shift every opener through the patch (Bias::Right — the offset tracks its
        // bracket char) on the delta-gap tree; a deleted opener maps into the
        // deletion and reconcile drops it.
        self.offsets.apply_patch(patch);
        // Deliberately does NOT bump the generation: shifting openers through an
        // edit is not a structural change to the fold SET, and the generation is
        // the document's `FoldMap` cache key. The edit path shifts the cached
        // map in lockstep via `FoldMap::apply_patch`, so a rebuild is reserved for
        // real fold add/remove (fold/unfold/reconcile), which do bump.
    }

    /// Drop every fold whose opener is no longer a live *foldable* open bracket.
    /// `is_foldable_opener` answers "is this byte offset an open bracket with a
    /// matched partner forming a foldable range?" (from the live [`Brackets`]
    /// pass — see `Document::reconcile_folds`). Because the extent is
    /// re-derived, there is nothing else to validate or heal.
    pub fn reconcile(&mut self, is_foldable_opener: impl Fn(u32) -> bool) {
        #[cfg(any(test, debug_assertions))]
        RECONCILE_SCANS.with(|c| c.set(c.get() + 1));
        crate::perf::charge(self.offsets.len() as u64); // complexity gate: whole-set scan
        let drop: Vec<u32> =
            self.offsets.offsets().into_iter().filter(|&o| !is_foldable_opener(o)).collect();
        self.offsets.remove_all(&drop);
        self.bump();
    }

    /// The collapsed openers whose offset lies in `range`, ascending — the folds an
    /// edit's re-matched region could directly touch. O(log n + hits).
    #[must_use]
    pub fn openers_in(&self, range: core::ops::Range<u32>) -> Vec<u32> {
        self.offsets.in_range(range.start, range.end)
    }

    /// Windowed reconcile ([`reconcile`](Self::reconcile)'s twin): drop only the
    /// `candidates` (a small, pre-narrowed set) that are no longer foldable. The
    /// common case — a bracket edit that broke no fold — removes nothing and costs
    /// only the O(candidates) predicate; the O(folds) `retain` runs (and is
    /// counted) ONLY when a fold actually broke, which is the unavoidable Vec
    /// compaction. Callers vouch every candidate is currently folded.
    pub fn reconcile_only(&mut self, candidates: &[u32], is_foldable_opener: impl Fn(u32) -> bool) {
        let drop: Vec<u32> = candidates.iter().copied().filter(|&o| !is_foldable_opener(o)).collect();
        if drop.is_empty() {
            return; // no fold broke — O(candidates), no whole-set touch
        }
        #[cfg(any(test, debug_assertions))]
        RECONCILE_SCANS.with(|c| c.set(c.get() + 1));
        crate::perf::charge(self.offsets.len() as u64); // complexity gate: fold-set touch
        self.offsets.remove_all(&drop);
        self.bump();
    }

    /// The collapsed openers (offsets), ascending — for [`FoldMap`] and tests.
    pub fn iter(&self) -> impl Iterator<Item = u32> + '_ {
        self.offsets.offsets().into_iter()
    }
}

/// One resolved block fold: it hides buffer rows `header+1..=last`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct FoldedRegion {
    header: u32,
    last: u32,
}

/// One resolved **inline** fold — a collapsed single-line bracket pair. It hides
/// the bytes *between* the brackets (`open+1..close`) on one still-visible row,
/// rendered `[ … ]`. The delimiters stay; only the interior collapses, so `close`
/// remains a real, addressable buffer position.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct InlineFold {
    /// The buffer row the pair sits on (opener and closer share it).
    pub row: u32,
    /// Byte offset of the opening bracket.
    pub open: u32,
    /// Byte offset of the closing bracket.
    pub close: u32,
}

impl InlineFold {
    /// The LEFT landable edge of the collapsed gap: just after the `[`.
    #[must_use]
    pub fn left_edge(&self) -> u32 {
        crate::row_layout::gap_left_edge(self.open)
    }

    /// The RIGHT landable edge: the closing bracket itself.
    #[must_use]
    pub fn right_edge(&self) -> u32 {
        self.close
    }

    /// Whether caret offset `off` is strictly inside the collapsed gap — the one
    /// caret-boundary rule shared by movement, the document's fold-time caret
    /// pull-out, and the widget's projections.
    #[must_use]
    pub fn hides_caret_at(&self, off: u32) -> bool {
        crate::row_layout::gap_hides_caret(self.open, self.close, off)
    }

    /// Whether the *glyph* at `off` is hidden by this fold — the deliberately
    /// off-by-one sibling of [`Self::hides_caret_at`]: the glyph at `open+1`
    /// hides even though its caret slot stays landable.
    #[must_use]
    pub fn hides_glyph_at(&self, off: u32) -> bool {
        crate::row_layout::gap_hides_glyph(self.open, self.close, off)
    }
}

/// Reduce a laminar set of block regions to its **root** (outermost) members,
/// sorted ascending by header and pairwise disjoint. A region nested inside
/// another hides no extra rows and its header is itself hidden, so only roots
/// drive the row arithmetic. Bracket pairs are naturally laminar (nested or
/// disjoint, never crossing), so this never has to reject a crossing.
fn root_regions(mut regions: Vec<FoldedRegion>) -> Vec<FoldedRegion> {
    regions.sort_by_key(|r| (r.header, core::cmp::Reverse(r.last)));
    let mut roots: Vec<FoldedRegion> = Vec::new();
    for r in regions {
        // `r.header > prev_root.last` ⇒ disjoint (a new root); otherwise `r` sits
        // inside the current root's row span (laminar ⇒ fully contained) — skip.
        if roots.last().is_none_or(|root| r.header > root.last) {
            roots.push(r);
        }
    }
    roots
}

/// Reduce inline folds to their **roots**: drop any inline fold nested inside
/// another collapsed inline fold on the same row — its interior is already hidden
/// in the outer fold's `…` chip, so rendering it separately would double-count the
/// horizontal collapse (and, on the no-span path, slice a backwards range). Bracket
/// pairs are laminar, so an inner fold is fully contained
/// (`outer.open < inner.open && inner.close < outer.close`). The inline analog of
/// [`root_regions`]: sort outer-before-inner, keep a fold unless the last root
/// encloses it (the last root is always the tightest enclosing candidate, since a
/// container and all its contents sort before any disjoint sibling).
/// Delta-gap encode root-reduced, opener-ascending inline folds into the tree.
fn build_inline(folds: Vec<InlineFold>) -> SumTree<InlineItem> {
    let (mut prev_open, mut prev_row) = (0u32, 0u32);
    let items = folds.into_iter().map(move |f| {
        debug_assert!(f.open >= prev_open && f.row >= prev_row, "inline folds ascend");
        let it = InlineItem { open_gap: f.open - prev_open, width: f.close - f.open, row_gap: f.row - prev_row };
        prev_open = f.open;
        prev_row = f.row;
        it
    });
    SumTree::from_items(items)
}

fn root_inline(mut inline: Vec<InlineFold>) -> Vec<InlineFold> {
    inline.sort_by(|a, b| a.open.cmp(&b.open).then(b.close.cmp(&a.close)));
    let mut roots: Vec<InlineFold> = Vec::new();
    for f in inline {
        if roots.last().is_none_or(|r| !(r.row == f.row && r.open < f.open && f.close < r.close)) {
            roots.push(f);
        }
    }
    roots
}

/// A visible display row, resolved back to its buffer row — the render driver.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct VisibleRow {
    /// The row's index in display space.
    pub display_row: DisplayRow,
    /// The buffer row it renders.
    pub buffer_row: BufferRow,
    /// Whether this row is a fold header (draw a chevron + `…` chip).
    pub is_fold_header: bool,
    /// On a header, the last folded row — `Some(last)` ⇒ this fold is collapsed.
    pub last_folded: Option<BufferRow>,
}

/// One resolved block fold on the delta-gap region tree. `vgap` = this region's
/// header row minus the PREVIOUS region's `last` (the visible rows between them;
/// ≥ 1 for every region after the first, since roots are disjoint — 0 only for a
/// fold headed on row 0). `span` = `last − header` (the hidden interior rows, ≥ 1
/// for a block). Absolute header/last are prefix sums, so a line insert shifts a
/// whole suffix by reanchoring ONE seam gap.
#[derive(Clone, Copy, Debug)]
struct Region {
    vgap: u32,
    span: u32,
}

/// Relative region summary. `vgap_sum` accumulates to a header's display row
/// (`Σ vgap = header − hidden_before`), `span_sum` to hidden rows, and their sum to
/// the absolute `last` row — the three dimensions the projections seek by.
#[derive(Clone, Copy, Debug, Default)]
struct RegionSummary {
    vgap_sum: u32,
    span_sum: u32,
    count: u32,
}

impl Summary for RegionSummary {
    fn add_summary(&mut self, o: &Self) {
        self.vgap_sum += o.vgap_sum;
        self.span_sum += o.span_sum;
        self.count += o.count;
    }
}

impl Item for Region {
    type Summary = RegionSummary;
    fn summary(&self) -> RegionSummary {
        RegionSummary { vgap_sum: self.vgap, span_sum: self.span, count: 1 }
    }
}

/// Absolute `last` row — `Σ (vgap + span)`; the cumulative end after region `i` is
/// `last_i`, so `seek(LastDim(r))` finds the first region whose `last > r`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct LastDim(u32);
impl Dimension<RegionSummary> for LastDim {
    fn add_summary(&mut self, s: &RegionSummary) {
        self.0 += s.vgap_sum + s.span_sum;
    }
}

/// A header's display row — `Σ vgap`; the cumulative end after region `i` is
/// `header_display[i]` (strictly increasing), so `summary_before(DispDim(d)).count`
/// = the headers displayed strictly above `d`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct DispDim(u32);
impl Dimension<RegionSummary> for DispDim {
    fn add_summary(&mut self, s: &RegionSummary) {
        self.0 += s.vgap_sum;
    }
}

/// Hidden rows — `Σ span`; accumulated before region `i` it is `hidden_before[i]`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct HiddenDim(u32);
impl Dimension<RegionSummary> for HiddenDim {
    fn add_summary(&mut self, s: &RegionSummary) {
        self.0 += s.span_sum;
    }
}

/// Region index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct RCount(u32);
impl Dimension<RegionSummary> for RCount {
    fn add_summary(&mut self, s: &RegionSummary) {
        self.0 += s.count;
    }
}

/// A located region decoded to absolute rows plus the rows hidden before it — the
/// shared result of the buffer-row → region seek (see [`FoldMap::locate`]).
struct Located {
    header: u32,
    last: u32,
    span: u32,
    hidden_before: u32,
}

/// Delta-gap inline (single-line) fold leaf, on the same contract as [`Region`].
/// `open_gap` = open − previous fold's open, `width` = close − open, `row_gap` =
/// row − previous fold's row. Absolute open/row are prefix sums; close = open +
/// width. So a text edit shifts the whole inline set by reanchoring one seam.
#[derive(Clone, Copy, Debug)]
struct InlineItem {
    open_gap: u32,
    width: u32,
    row_gap: u32,
}

#[derive(Clone, Copy, Debug, Default)]
struct InlineSummary {
    open_span: u32,
    row_span: u32,
    count: u32,
}

impl Summary for InlineSummary {
    fn add_summary(&mut self, o: &Self) {
        self.open_span += o.open_span;
        self.row_span += o.row_span;
        self.count += o.count;
    }
}

impl Item for InlineItem {
    type Summary = InlineSummary;
    fn summary(&self) -> InlineSummary {
        InlineSummary { open_span: self.open_gap, row_span: self.row_gap, count: 1 }
    }
}

/// Combined open-offset / buffer-row dimension: accumulates BOTH so a single seek
/// or visit yields a fold's `open` and `row` together (one dimension slot). Ordered
/// by `(open, row)` — open dominates, and rows rise monotonically with opens, so it
/// is a total order in document order, and `summary_before(OpenRow { open: x, .. })`
/// counts the folds with `open < x`.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct OpenRow {
    open: u32,
    row: u32,
}
impl Dimension<InlineSummary> for OpenRow {
    fn add_summary(&mut self, s: &InlineSummary) {
        self.open += s.open_span;
        self.row += s.row_span;
    }
}

/// Inline fold index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ICount(u32);
impl Dimension<InlineSummary> for ICount {
    fn add_summary(&mut self, s: &InlineSummary) {
        self.0 += s.count;
    }
}

/// A fold-aware row converter — buffer rows ↔ display rows with folded interiors
/// hidden. Resolved from a [`FoldSet`] + the live [`Brackets`] + buffer,
/// **memoized** on the [`Document`](crate::Document). A fold add/remove
/// ([`FoldSet::generation`] bump) rebuilds it from scratch; every text edit shifts
/// it in place through the patch ([`Self::apply_patch`]) — a no-line-change edit
/// only moves inline offsets, and a line insert/remove reanchors the region tree's
/// affected seam — so plain typing over a document-scale fold set never pays an
/// O(folds) rebuild. Manual `PartialEq` (decoded regions) so a test can deep-equal
/// the incrementally-shifted map against a fresh build (the drift oracle).
#[derive(Debug)]
pub struct FoldMap {
    /// Resolved *root* block folds (sorted, disjoint by header) on a delta-gap
    /// [`SumTree`] keyed by header row: a line insert/remove shifts a whole suffix
    /// in O(log n) by reanchoring one seam gap, and every row projection is an
    /// O(log n) cursor descent, not an O(folds) full rebuild.
    regions: SumTree<Region>,
    /// Resolved inline (single-line) folds on a delta-gap [`SumTree`], keyed by
    /// opener offset (hence by row), so a text edit reanchors one seam in O(log n)
    /// rather than an O(inline) rebase, and offset/row queries are O(log n).
    inline: SumTree<InlineItem>,
    buffer_row_count: u32,
}

impl PartialEq for FoldMap {
    fn eq(&self, other: &Self) -> bool {
        self.buffer_row_count == other.buffer_row_count
            && self.decoded_regions() == other.decoded_regions()
            && self.decoded_inline() == other.decoded_inline()
    }
}
impl Eq for FoldMap {}

impl FoldMap {
    /// Resolve the collapsed openers into visible block-fold regions against the
    /// live `brackets` + `buffer`. `O(folds log brackets)`. An opener that is no
    /// longer a matched bracket, or whose pair is single-line, yields no block
    /// region (the latter is a future inline fold, not a row-hider).
    #[must_use]
    pub fn new(folds: &FoldSet, brackets: &Brackets, buffer: &Buffer) -> Self {
        #[cfg(any(test, debug_assertions))]
        FOLD_BUILDS.with(|c| c.set(c.get() + 1));
        let mut regions = Vec::new();
        let mut inline = Vec::new();
        for open in folds.iter() {
            // `foldable_partner`, not `at().partner`: the depth-free lookup, so a
            // rebuild over many folds never allocates a prefix stack per opener.
            let Some(close) = brackets.foldable_partner(open) else { continue };
            let header = buffer.offset_to_point(open).row;
            let last = buffer.offset_to_point(close).row;
            if last > header {
                regions.push(FoldedRegion { header, last }); // multi-line ⇒ block
            } else if crate::row_layout::pair_has_interior(open, close) {
                inline.push(InlineFold { row: header, open, close }); // single-line ⇒ inline
            }
        }
        Self::assemble(regions, inline, buffer.line_count())
    }

    /// Shift this map through a committed edit's `patch` IN PLACE, returning
    /// `false` (→ the caller rebuilds instead) when the edit changed the line
    /// count. A line insertion/removal moves block-region rows and the display
    /// prefix sums, which is not a cheap in-place update; a **no-line-change**
    /// edit (plain typing) leaves every block region, prefix sum, and the buffer
    /// row count unchanged and only moves inline-fold offsets — so this is
    /// O(inline folds), and O(0) for the common block-only "collapse all" case,
    /// far cheaper than the O(folds) rebuild the per-keystroke edit path would
    /// otherwise pay.
    ///
    /// A fold whose *interior* an edit touches is expanded (removed) before this
    /// runs, so a surviving inline fold's interior is intact — only its (and its
    /// `close`'s) offset shifts, exactly as [`FoldSet::apply_patch`] shifts the
    /// opener, keeping the two in lockstep. Fold add/remove bumps the generation
    /// and forces a rebuild, so this never has to reconcile the fold set.
    #[must_use]
    pub fn apply_patch(&mut self, patch: &Patch, buffer: &Buffer) -> bool {
        let new_rows = buffer.line_count();
        let d = i64::from(new_rows) - i64::from(self.buffer_row_count);
        // A line change moves block-region rows. A SINGLE edit shifts every region
        // (and inline fold) at/after the insertion row by `+d` — the region tree in
        // O(log n) via one reanchored seam, inline offsets/rows in the loop below.
        // A MULTI-edit line change is a piecewise row shift (multi-caret, rare) —
        // rebuild. `os` bounds the inline row shift to the folds after the edit.
        if d != 0 {
            let [edit] = patch.edits() else { return false };
            let (os, oe) = (edit.old.start, edit.old.end);
            // A line change overlapping an inline fold's span can RECLASSIFY it
            // (single-line ↔ multi-line block) — a shift can't express that, and
            // `expand` doesn't catch an inline fold's byte interior, so rebuild.
            if self.inline_overlaps(os, oe) {
                return false;
            }
            // The edit's OLD range covered rows `[r_os, r_oe_old]`; content ending
            // after `r_os` shifts by `d`. `r_oe_old = new-end row − d` recovers the
            // old-end row without the old buffer (the net line delta `d` is the
            // newlines the range gained). For an insertion `r_oe_old == r_os`.
            let r_os = buffer.offset_to_point(edit.new.start).row;
            let r_oe_old = (i64::from(buffer.offset_to_point(edit.new.end).row) - d) as u32;
            // Regions that DON'T shift end at/before `r_os` (`last ≤ r_os`): a fold
            // tailed on `r_os` has its opener earlier, so it is before the edit.
            let k = self.regions.summary_before(&LastDim(r_os.saturating_add(1))).count;
            let (before, suffix) = self.regions.split_at(&RCount(k));
            if !suffix.is_empty() {
                let last_prev = before.extent::<LastDim>().0; // last of the region before the seam
                let (first, ..) = suffix.seek::<RCount, LastDim>(&RCount(0)).expect("non-empty");
                let header = last_prev + first.vgap;
                // If the edit's OLD range reached INTO the first shifted region
                // (`header < r_oe_old`), a deletion shrank its span — `expand` misses
                // that (it checks only the collapsed NEW endpoint, which can land on
                // a still-visible row), and a rigid row shift can't express a span
                // change — so rebuild. Later regions have larger headers, so probing
                // the first suffices. (An insertion has `r_oe_old == r_os ≤ header`,
                // so this never trips — surviving folds never straddle an insertion.)
                if header < r_oe_old {
                    return false;
                }
                // Reanchor: the first shifted region's `vgap` absorbs `+d` (its
                // header moves while its predecessor, in `before`, stayed); later
                // gaps/spans are relative and unchanged. The straddle guard makes
                // the new header ≥ r_os ≥ the predecessor's last, so `vgap` stays ≥ 0.
                let new_vgap = i64::from(first.vgap) + d;
                debug_assert!(new_vgap >= 0, "a non-straddling shift keeps vgap ≥ 0");
                let fixed = Region { vgap: new_vgap as u32, span: first.span };
                let suffix = suffix.replace(RCount(0)..RCount(1), std::iter::once(fixed));
                self.regions = before.append(&suffix);
            }
            self.buffer_row_count = new_rows;
        }
        self.shift_inline(patch, d);
        true
    }

    /// Shift the inline-fold tree through `patch`. A single edit whose range
    /// overlaps no surviving inline fold reanchors ONE seam in O(log n) — folds
    /// after the edit move `open += byte_delta`, `row += d`. Otherwise (a straddling
    /// edit that changes a fold's width, or a multi-edit) it rides each fold through
    /// the patch and rebuilds — O(inline), the rare fallback. A line-changing
    /// straddle already forced a full rebuild upstream, so the fallback is `d == 0`.
    fn shift_inline(&mut self, patch: &Patch, d: i64) {
        if let [edit] = patch.edits() {
            let (os, oe) = (edit.old.start, edit.old.end);
            if !self.inline_overlaps(os, oe) {
                let byte_delta = (i64::from(edit.new.end) - i64::from(edit.new.start))
                    - (i64::from(oe) - i64::from(os));
                let k = self.inline.summary_before(&OpenRow { open: os, row: 0 }).count; // open < os
                let (before, suffix) = self.inline.split_at(&ICount(k));
                if !suffix.is_empty() {
                    let (it, ..) = suffix.seek::<ICount, OpenRow>(&ICount(0)).expect("non-empty");
                    let fixed = InlineItem {
                        open_gap: (i64::from(it.open_gap) + byte_delta) as u32,
                        width: it.width,
                        row_gap: (i64::from(it.row_gap) + d) as u32,
                    };
                    let suffix = suffix.replace(ICount(0)..ICount(1), std::iter::once(fixed));
                    self.inline = before.append(&suffix);
                }
                return;
            }
        }
        debug_assert!(d == 0, "the inline fallback is only reached on a no-line-change edit");
        let mut v = self.decoded_inline();
        for f in &mut v {
            f.open = patch.map_offset(f.open, Bias::Right);
            f.close = patch.map_offset(f.close, Bias::Right);
        }
        self.inline = build_inline(v);
    }

    /// Reduce raw folds to roots and build the O(log) search prefixes (the
    /// hidden-row prefix sum and each header's display row). The one owner of the
    /// [`FoldMap`] invariants, shared by [`Self::new`] and the test constructor.
    fn assemble(regions: Vec<FoldedRegion>, inline: Vec<InlineFold>, buffer_row_count: u32) -> Self {
        let regions = root_regions(regions); // sorted, disjoint by header
        // Delta-gap encode: vgap = header − previous region's last (visible rows
        // between them), span = last − header (hidden interior rows).
        let mut prev_last = 0u32;
        let mut items = Vec::with_capacity(regions.len());
        for r in &regions {
            items.push(Region { vgap: r.header - prev_last, span: r.last - r.header });
            prev_last = r.last;
        }
        Self {
            regions: SumTree::from_items(items),
            inline: build_inline(root_inline(inline)),
            buffer_row_count,
        }
    }

    /// Decode the region tree back to absolute `(header, last)` pairs — the O(n)
    /// reduce for `PartialEq` (the drift oracle) and tests.
    fn decoded_regions(&self) -> Vec<(u32, u32)> {
        let mut out = Vec::with_capacity(self.regions.summary().count as usize);
        let mut last = 0u32;
        for item in self.regions.items() {
            let header = last + item.vgap;
            last = header + item.span;
            out.push((header, last));
        }
        out
    }

    /// Decode the inline-fold tree to absolute [`InlineFold`]s, ascending by opener
    /// — the O(n) reduce for `PartialEq`, [`Self::inline_folds`], and tests.
    fn decoded_inline(&self) -> Vec<InlineFold> {
        let mut out = Vec::with_capacity(self.inline.summary().count as usize);
        let (mut open, mut row) = (0u32, 0u32);
        for it in self.inline.items() {
            crate::perf::charge(1); // complexity gate: whole inline-set decode is O(F)
            open += it.open_gap;
            row += it.row_gap;
            out.push(InlineFold { row, open, close: open + it.width });
        }
        out
    }

    /// The last inline fold opening strictly before `off` (the only one that can
    /// hide `off`, since roots are disjoint) — O(log n). The offset-keyed lookup
    /// movement / hit-testing / the caret-eject path use.
    #[must_use]
    pub fn inline_fold_before(&self, off: u32) -> Option<InlineFold> {
        let j = self.inline.summary_before(&OpenRow { open: off, row: 0 }).count; // open < off
        let (it, _c, OpenRow { open: ob, row: rb }) =
            self.inline.seek::<ICount, OpenRow>(&ICount(j.checked_sub(1)?))?;
        let open = ob + it.open_gap;
        Some(InlineFold { row: rb + it.row_gap, open, close: open + it.width })
    }

    /// The root inline fold opening *exactly* at `opener`, or `None` — O(log n)
    /// membership for the collapsed-chip hit-test. Roots are disjoint and
    /// open-sorted, so [`Self::inline_fold_before`]`(opener+1)` returns the last
    /// root with `open ≤ opener`; testing `open == opener` is then exact root
    /// membership. `saturating_add` guards the u32 edge; a nested (non-root)
    /// opener resolves to its enclosing root, whose `open != opener` → `None` —
    /// exactly what a linear `inline_folds().binary_search` membership test yields.
    #[must_use]
    pub fn inline_fold_at(&self, opener: u32) -> Option<InlineFold> {
        self.inline_fold_before(opener.saturating_add(1)).filter(|f| f.open == opener)
    }

    /// Whether any inline fold's `[open, close]` overlaps `[os, oe]` — O(log n). One
    /// probe suffices: the fold with the greatest `open ≤ oe` has the greatest
    /// `close` (disjoint, open-sorted ⇒ close ascending).
    fn inline_overlaps(&self, os: u32, oe: u32) -> bool {
        let j = self.inline.summary_before(&OpenRow { open: oe.saturating_add(1), row: 0 }).count;
        let Some(k) = j.checked_sub(1) else { return false };
        let Some((it, _c, OpenRow { open: ob, .. })) = self.inline.seek::<ICount, OpenRow>(&ICount(k))
        else {
            return false;
        };
        ob + it.open_gap + it.width >= os // close ≥ os
    }

    /// The region whose `[last_prev, last]` brackets buffer row `r` — the first with
    /// `last > r`, or the last region as a fallback (then `r` is beyond every fold) —
    /// decoded to absolute rows plus the hidden rows before it. `None` iff no folds.
    /// The shared O(log n) descent under every buffer-row → display query.
    fn locate(&self, r: u32) -> Option<Located> {
        let (item, LastDim(last_prev), HiddenDim(hidden_before)) =
            self.regions.seek::<LastDim, HiddenDim>(&LastDim(r))?;
        let header = last_prev + item.vgap;
        Some(Located { header, last: header + item.span, span: item.span, hidden_before })
    }

    /// Tail-inclusive [`Self::locate`]: the region with `last >= r` (via
    /// `seek(last > r − 1)`), so a fold's TAIL row (`last == r`) resolves to ITS
    /// fold, not the next one below it. `locate`'s `last > r` is right for the
    /// display projections but off-by-one for containment (`is_folded` /
    /// `fold_containing`), where the hidden tail row must still count as folded.
    fn containing(&self, r: u32) -> Option<Located> {
        let (item, LastDim(last_prev), HiddenDim(hidden_before)) =
            self.regions.seek::<LastDim, HiddenDim>(&LastDim(r.saturating_sub(1)))?;
        let header = last_prev + item.vgap;
        Some(Located { header, last: header + item.span, span: item.span, hidden_before })
    }

    /// An empty (nothing-folded) map — the document's cache seed before its first
    /// real build.
    #[must_use]
    pub(crate) fn empty() -> Self {
        Self::assemble(Vec::new(), Vec::new(), 0)
    }

    /// The resolved inline (single-line) folds, sorted by opener offset — the
    /// render / caret / hit-test consult these to collapse a mid-line span.
    #[must_use]
    pub fn inline_folds(&self) -> Vec<InlineFold> {
        self.decoded_inline()
    }

    /// The inline folds on buffer `row`, ascending by opener — the per-row render
    /// query (`RowLayout`). O(log n + hits): a row's folds are a contiguous window.
    #[must_use]
    pub fn inline_folds_on_row(&self, row: u32) -> Vec<InlineFold> {
        let mut out = Vec::new();
        self.inline.filter_visit::<OpenRow, _, _>(
            &|before: &OpenRow, sum: &InlineSummary| {
                before.row <= row && before.row + sum.row_span >= row
            },
            &mut |it: &InlineItem, before: &OpenRow| {
                let r = before.row + it.row_gap;
                if r == row {
                    let open = before.open + it.open_gap;
                    out.push(InlineFold { row: r, open, close: open + it.width });
                }
            },
        );
        out
    }

    /// Total hidden rows across all root folds (the tree root summary, O(1)).
    fn hidden_total(&self) -> u32 {
        self.regions.summary().span_sum
    }

    /// Number of display rows (buffer rows minus every hidden interior row).
    #[must_use]
    pub fn display_row_count(&self) -> u32 {
        self.buffer_row_count - self.hidden_total()
    }

    /// The last valid display row.
    #[must_use]
    pub fn max_display_row(&self) -> DisplayRow {
        DisplayRow(self.display_row_count().saturating_sub(1))
    }

    /// Buffer row → display row. A hidden interior row clips to its header's
    /// display row, so a collapsed fold's interior renders at the header line.
    #[must_use]
    pub fn to_display_row(&self, row: BufferRow) -> DisplayRow {
        let r = row.0;
        let Some(l) = self.locate(r) else { return DisplayRow(r) };
        if r > l.header && r <= l.last {
            DisplayRow(l.header - l.hidden_before) // interior → clip to the header's display row
        } else if r <= l.header {
            DisplayRow(r - l.hidden_before) // visible, at/before this region's header
        } else {
            DisplayRow(r - l.hidden_before - l.span) // r beyond the last fold (seek fallback)
        }
    }

    /// Display row → buffer row. Always a visible row (a header or an unfolded
    /// line), never a hidden interior.
    #[must_use]
    pub fn to_buffer_row(&self, row: DisplayRow) -> BufferRow {
        let d = row.0;
        // Regions whose header displays strictly above `d` are hidden-above; add
        // their hidden rows back. `summary_before(DispDim(d))` = exactly those
        // (header_display[i] < d), `.span_sum` = their hidden rows (= hidden_before[j]).
        BufferRow(d + self.regions.summary_before(&DispDim(d)).span_sum)
    }

    /// Whether `row` is a hidden interior row of some fold.
    #[must_use]
    pub fn is_folded(&self, row: BufferRow) -> bool {
        let r = row.0;
        self.containing(r).is_some_and(|l| l.header < r && r <= l.last)
    }

    /// If `offset` is HIDDEN inside a collapsed fold, the caret's eject target — a
    /// block's header line end, or an inline gap's left landable edge; `None` when
    /// visible. `O(log folds)` (regions and inline roots are sorted and disjoint),
    /// so a multi-caret fold pulls every hidden caret out in one `O(carets·log
    /// folds)` pass instead of an `O(carets·folds)` [`Self::display_position`]
    /// probe each — the "collapse all" ejection at document scale.
    #[must_use]
    pub fn entry_edge_if_hidden(&self, buffer: &Buffer, offset: u32) -> Option<u32> {
        // Block: is `offset`'s row STRICTLY inside a region's hidden interior?
        // (The tail row `last` rides the header and stays visible, so `< last`.)
        let row = buffer.offset_to_point(offset).row;
        if let Some(l) = self.locate(row) {
            if l.header < row && row < l.last {
                let end = buffer.line_len(l.header);
                return Some(buffer.point_to_offset(Point::new(l.header, end)));
            }
        }
        // Inline: is `offset` strictly inside a collapsed single-line gap? Only the
        // fold opening just before it can hide it (roots are disjoint).
        if let Some(f) = self.inline_fold_before(offset) {
            if f.hides_caret_at(offset) {
                return Some(f.left_edge());
            }
        }
        None
    }

    /// The collapsed root fold whose hidden interior contains `row` (i.e.
    /// `header < row <= last`), as `(header, last)`. `None` if `row` is a header
    /// or not folded. Used by fold-aware horizontal movement to hop the collapsed
    /// gap between the header line and the inline closing tail.
    #[must_use]
    pub fn fold_containing(&self, row: BufferRow) -> Option<(BufferRow, BufferRow)> {
        let r = row.0;
        self.containing(r)
            .filter(|l| l.header < r && r <= l.last)
            .map(|l| (BufferRow(l.header), BufferRow(l.last)))
    }

    /// If `row` is a folded header, the fold's last row (for the chip/chevron and
    /// placeholder). `None` if `row` is not a header of a collapsed fold.
    #[must_use]
    pub fn fold_at_header(&self, row: BufferRow) -> Option<BufferRow> {
        // The region bracketing `row` is the one headed there iff its header == row
        // (a header's own `last > row`, and the prior region's `last < row`).
        self.locate(row.0).filter(|l| l.header == row.0).map(|l| BufferRow(l.last))
    }

    /// If `row` is the *last* (closing) row of a collapsed root fold, that fold's
    /// header row — the display anchor for the fold's inline closing tail.
    /// `None` otherwise. Roots are disjoint, so at most one fold ends on `row`.
    /// The inverse direction of [`Self::fold_at_header`], for placing a caret /
    /// hit-testing a click on the collapsed line's real closing bracket.
    #[must_use]
    pub fn header_of_tail(&self, row: BufferRow) -> Option<BufferRow> {
        // Roots are disjoint ⇒ `last` is strictly increasing. `j` regions end before
        // `row`; the j-th (first with `last >= row`) is tailed on `row` iff its
        // `last == row`, and then its header is `last − span`.
        let r = row.0;
        let j = self.regions.summary_before(&LastDim(r)).count;
        let (item, RCount(_), LastDim(last_prev)) = self.regions.seek::<RCount, LastDim>(&RCount(j))?;
        let last = last_prev + item.vgap + item.span;
        (last == r).then(|| BufferRow(last - item.span))
    }

    /// The display-row window covering fractional rows `[top_rows, bottom_rows)`
    /// from the content top: floor the first visible row, ceil past the last,
    /// clamp to the row count. THE one owner of the render window rule — the
    /// widget derives the fractional rows from pixels ([`Self::display_row_at`]
    /// is its single-row sibling) and never floors/clamps itself.
    /// (`f64` like [`Self::display_row_at`]: exact for every u32 row count,
    /// where `f32` fractional rows degrade past ~2²³.)
    #[must_use]
    pub fn display_window(&self, top_rows: f64, bottom_rows: f64) -> core::ops::Range<u32> {
        let first = top_rows.floor().max(0.0) as u32;
        let last = (bottom_rows.ceil().max(0.0) as u32).min(self.display_row_count());
        first..last.max(first)
    }

    /// The visible display rows in `window` (see [`Self::display_window`]),
    /// each resolved to its buffer row with the fold-header flag — the render
    /// loop's driver. Hidden interior rows are simply not produced.
    pub fn visible_rows(&self, window: core::ops::Range<u32>) -> impl Iterator<Item = VisibleRow> + '_ {
        window.map(move |d| {
            let buffer_row = self.to_buffer_row(DisplayRow(d));
            let last_folded = self.fold_at_header(buffer_row);
            VisibleRow {
                display_row: DisplayRow(d),
                buffer_row,
                is_fold_header: last_folded.is_some(),
                last_folded,
            }
        })
    }

    /// Build a `FoldMap` straight from `(header, last)` block regions — the
    /// row-arithmetic under test, without needing a bracketed buffer.
    #[cfg(test)]
    pub(crate) fn from_rows(regions: impl IntoIterator<Item = (u32, u32)>, row_count: u32) -> Self {
        let regions = regions
            .into_iter()
            .filter(|&(h, l)| l > h)
            .map(|(header, last)| FoldedRegion { header, last })
            .collect();
        Self::assemble(regions, Vec::new(), row_count)
    }

    /// Build a `FoldMap` from `(row, open, close)` inline folds — the inline
    /// row/offset arithmetic under test, without a bracketed buffer.
    #[cfg(test)]
    pub(crate) fn from_inline(
        folds: impl IntoIterator<Item = (u32, u32, u32)>,
        row_count: u32,
    ) -> Self {
        let inline = folds.into_iter().map(|(row, open, close)| InlineFold { row, open, close }).collect();
        Self::assemble(Vec::new(), inline, row_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{Edit, Patch};

    // ── FoldSet: opener-keyed set that rides the patch mover ──

    #[test]
    fn fold_toggle_and_duplicate() {
        let mut f = FoldSet::new();
        assert!(f.fold(10));
        assert!(!f.fold(10), "same opener already folded");
        assert!(f.is_folded(10));
        assert!(f.toggle(10)); // unfolds
        assert!(!f.is_folded(10));
        assert!(f.is_empty());
    }

    #[test]
    fn apply_patch_shifts_openers() {
        let mut f = FoldSet::new();
        f.fold(5);
        f.fold(20);
        // Insert 3 bytes at the top → both openers shift right by 3.
        f.apply_patch(&Patch::single(Edit { old: 0..0, new: 0..3 }));
        assert_eq!(f.iter().collect::<Vec<_>>(), vec![8, 23]);
    }

    #[test]
    fn reconcile_drops_openers_that_are_no_longer_foldable() {
        let mut f = FoldSet::new();
        f.fold(8);
        f.fold(23);
        // Only offset 8 is still a live foldable open bracket.
        f.reconcile(|o| o == 8);
        assert_eq!(f.iter().collect::<Vec<_>>(), vec![8]);
    }

    #[test]
    fn deleting_the_opener_reconciles_away() {
        let mut f = FoldSet::new();
        f.fold(4);
        // Delete the bracket char at [4,5); the opener maps into the deletion.
        f.apply_patch(&Patch::single(Edit { old: 4..5, new: 4..4 }));
        f.reconcile(|_| false); // nothing foldable there anymore
        assert!(f.is_empty());
    }

    // ── FoldMap: row arithmetic over resolved block regions ──

    #[test]
    fn empty_is_identity() {
        let m = FoldMap::from_rows([], 6);
        assert_eq!(m.display_row_count(), 6);
        for r in 0..6 {
            assert_eq!(m.to_display_row(BufferRow(r)), DisplayRow(r));
            assert_eq!(m.to_buffer_row(DisplayRow(r)), BufferRow(r));
            assert!(!m.is_folded(BufferRow(r)));
        }
    }

    #[test]
    fn hides_interior_rows() {
        let m = FoldMap::from_rows([(2, 5)], 10); // hides rows 3,4,5
        assert_eq!(m.display_row_count(), 7);
        assert!(m.is_folded(BufferRow(3)) && m.is_folded(BufferRow(5)));
        assert!(!m.is_folded(BufferRow(2)) && !m.is_folded(BufferRow(6)));
        assert_eq!(m.fold_at_header(BufferRow(2)), Some(BufferRow(5)));
    }

    #[test]
    fn display_and_buffer_row_mapping() {
        let m = FoldMap::from_rows([(2, 5)], 10);
        for r in [3, 4, 5] {
            assert_eq!(m.to_display_row(BufferRow(r)), m.to_display_row(BufferRow(2)));
        }
        assert_eq!(m.to_display_row(BufferRow(6)), DisplayRow(3)); // 6 − 3 hidden
        assert_eq!(m.to_buffer_row(DisplayRow(2)), BufferRow(2)); // the header
        assert_eq!(m.to_buffer_row(DisplayRow(3)), BufferRow(6)); // interior skipped
        assert_eq!(m.to_buffer_row(DisplayRow(6)), BufferRow(9));
    }

    #[test]
    fn roundtrip_visible_rows() {
        let m = FoldMap::from_rows([(2, 5)], 10);
        let all: Vec<_> = m.visible_rows(0..m.display_row_count()).collect();
        assert_eq!(all.iter().map(|v| v.buffer_row.0).collect::<Vec<_>>(), vec![0, 1, 2, 6, 7, 8, 9]);
        let header = all.iter().find(|v| v.is_fold_header).unwrap();
        assert_eq!((header.buffer_row, header.last_folded), (BufferRow(2), Some(BufferRow(5))));
    }

    #[test]
    fn nested_reduces_to_root() {
        let m = FoldMap::from_rows([(1, 8), (3, 5)], 10); // inner (3,5) ⊆ outer (1,8)
        assert_eq!(m.display_row_count(), 3); // rows 2..=8 hidden ⇒ visible 0,1,9
        assert_eq!(m.fold_at_header(BufferRow(1)), Some(BufferRow(8)));
        assert_eq!(m.fold_at_header(BufferRow(3)), None, "inner header is hidden, not a root");
        let visible: Vec<_> = m.visible_rows(0..m.display_row_count()).map(|v| v.buffer_row.0).collect();
        assert_eq!(visible, vec![0, 1, 9]);
    }

    #[test]
    fn three_level_nesting_reduces_to_root() {
        let m = FoldMap::from_rows([(0, 10), (2, 8), (4, 6)], 12);
        assert_eq!(m.display_row_count(), 2); // root (0,10) hides 1..=10 ⇒ visible 0,11
        assert_eq!(m.to_buffer_row(DisplayRow(1)), BufferRow(11));
    }

    #[test]
    fn disjoint_siblings_both_hide() {
        let m = FoldMap::from_rows([(2, 5), (6, 8)], 10);
        assert_eq!(m.display_row_count(), 10 - 3 - 2);
        assert_eq!(m.fold_at_header(BufferRow(2)), Some(BufferRow(5)));
        assert_eq!(m.fold_at_header(BufferRow(6)), Some(BufferRow(8)));
    }

    #[test]
    fn fold_at_eof() {
        let m = FoldMap::from_rows([(3, 5)], 6);
        assert_eq!(m.display_row_count(), 4); // rows 4,5 hidden
        assert_eq!(m.fold_at_header(BufferRow(3)), Some(BufferRow(5)));
    }

    #[test]
    fn header_of_tail_is_the_inverse_of_fold_at_header() {
        let m = FoldMap::from_rows([(2, 5)], 10);
        assert_eq!(m.fold_at_header(BufferRow(2)), Some(BufferRow(5)));
        assert_eq!(m.header_of_tail(BufferRow(5)), Some(BufferRow(2)));
        assert_eq!(m.header_of_tail(BufferRow(2)), None, "the header is not a tail");
        assert_eq!(m.header_of_tail(BufferRow(4)), None, "an interior row is not the last");
    }

    #[test]
    fn fold_containing_finds_the_enclosing_root() {
        let m = FoldMap::from_rows([(2, 5)], 10);
        assert_eq!(m.fold_containing(BufferRow(4)), Some((BufferRow(2), BufferRow(5))));
        assert_eq!(m.fold_containing(BufferRow(2)), None, "the header is visible, not contained");
        assert_eq!(m.fold_containing(BufferRow(6)), None);
    }

    #[test]
    fn inline_shift_is_sublinear_in_fold_count() {
        use crate::sum_tree::NODE_ALLOCS;
        // N inline folds, one per row. A single front insert shifts ALL of them
        // (open += 1) — one reanchored delta-gap seam, O(log N) node allocs, not the
        // O(N) rebuild the flat Vec (or the fallback) would do. A `d == 0` edit keeps
        // it on the inline path only (no block regions, no offset_to_point).
        let allocs_for = |n: u32| -> f64 {
            let mut m = FoldMap::from_inline((0..n).map(|i| (i, i * 10 + 100, i * 10 + 103)), n);
            let buf = Buffer::new(&"\n".repeat((n - 1) as usize)).unwrap(); // line_count == n
            let patch = Patch::single(Edit { old: 5..5, new: 5..6 }); // insert 1 byte, no newline
            NODE_ALLOCS.with(|c| c.set(0));
            let _ = m.apply_patch(&patch, &buf);
            NODE_ALLOCS.with(std::cell::Cell::get) as f64
        };
        let (small, big) = (allocs_for(1000), allocs_for(4000));
        eprintln!("[fold_map] inline shift node allocs {small} -> {big}  ({:.2}x)", big / small);
        assert!(
            big <= small * 2.0,
            "inline shift allocates superlinearly ({small} -> {big} nodes): it rebuilt the \
             whole inline set instead of reanchoring the suffix at one seam"
        );
    }

    #[test]
    fn a_hidden_tail_row_counts_as_folded_with_a_fold_below() {
        // `is_folded`/`fold_containing` use the TAIL-inclusive lookup so a fold's
        // hidden tail row still counts as folded even when a second root sits
        // directly below it. Row 5 is fold (2,5)'s hidden tail; root (6,8) follows.
        // A plain `last > r` lookup would resolve row 5 to (6,8) and report it visible.
        let m = FoldMap::from_rows([(2, 5), (6, 8)], 10);
        assert!(m.is_folded(BufferRow(5)), "a fold's tail row is hidden");
        assert_eq!(m.fold_containing(BufferRow(5)), Some((BufferRow(2), BufferRow(5))));
        // The display projection agrees — row 5 collapses onto the header (row 2).
        assert_eq!(m.to_display_row(BufferRow(5)), m.to_display_row(BufferRow(2)));
        assert!(!m.is_folded(BufferRow(2)), "the header is visible");
        assert!(m.is_folded(BufferRow(7)) && m.is_folded(BufferRow(8)), "second fold hidden");
    }

    // ── inline root reduction: a collapsed inline pair nested in another ──

    #[test]
    fn nested_inline_folds_reduce_to_the_outer_root() {
        // One row: `( … )` at [4,30] with `[ … ]` at [20,28] nested inside it.
        let roots = root_inline(vec![
            InlineFold { row: 0, open: 20, close: 28 },
            InlineFold { row: 0, open: 4, close: 30 },
        ]);
        assert_eq!(roots.iter().map(|f| f.open).collect::<Vec<_>>(), vec![4], "only the outer chip renders");
    }

    #[test]
    fn disjoint_inline_folds_both_survive() {
        let roots = root_inline(vec![
            InlineFold { row: 0, open: 12, close: 18 },
            InlineFold { row: 0, open: 4, close: 8 },
        ]);
        assert_eq!(roots.iter().map(|f| f.open).collect::<Vec<_>>(), vec![4, 12]);
    }

    #[test]
    fn triple_nested_inline_reduces_to_root() {
        // A(0..40) ⊃ B(5..30) ⊃ C(10..20), all on row 0 → only A survives.
        let roots = root_inline(vec![
            InlineFold { row: 0, open: 10, close: 20 },
            InlineFold { row: 0, open: 5, close: 30 },
            InlineFold { row: 0, open: 0, close: 40 },
        ]);
        assert_eq!(roots.iter().map(|f| f.open).collect::<Vec<_>>(), vec![0]);
    }

    // ── the collapsed-chip inline-membership hit-test ───────────────────────────

    #[test]
    fn inline_fold_at_matches_the_linear_membership() {
        // Parity oracle: over random inline-fold sets (disjoint + nested, several
        // rows) the O(log F) offset-keyed `inline_fold_at` descent must agree with the
        // linear `inline_folds()` membership scan — presence bit AND resolved fold —
        // at EVERY candidate offset (root openers, nested/reduced openers, non-openers).
        // This pins the `+1` and the `f.open == opener` filter (drop either and a
        // root/non-opener boundary diverges here). Row bands (open = row*100 + …) keep
        // opener-ascending ⟺ row-ascending, the real-buffer invariant `from_inline`
        // asserts; openers are deduped because a bracket char occupies one offset.
        let mut seed = 0x9E37_79B9_7F4A_7C15u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        for _ in 0..300 {
            let n = (rng() % 9) as usize;
            let mut raw: Vec<(u32, u32, u32)> = Vec::new();
            for _ in 0..n {
                let row = (rng() % 4) as u32;
                let open = row * 100 + (rng() % 80) as u32;
                let width = 1 + (rng() % 18) as u32;
                raw.push((row, open, open + width));
            }
            raw.sort_unstable();
            raw.dedup_by_key(|&mut (_, o, _)| o); // unique openers, as real brackets are
            let m = FoldMap::from_inline(raw.iter().copied(), 4);
            let roots = m.inline_folds();
            for o in 0..=400u32 {
                let linear = roots.iter().find(|f| f.open == o).copied();
                let via_log = m.inline_fold_at(o);
                assert_eq!(
                    via_log, linear,
                    "inline_fold_at({o}) = {via_log:?} but linear membership = {linear:?}\nroots = {roots:?}"
                );
            }
            // The u32 edge: `saturating_add(1)` must not panic or false-match at MAX.
            assert_eq!(m.inline_fold_at(u32::MAX), roots.iter().find(|f| f.open == u32::MAX).copied());
        }
    }

    #[test]
    fn inline_membership_is_fold_count_independent() {
        // `inline_fold_at` is an O(log F) offset-keyed descent, so its work meter
        // must stay flat as the inline-fold count doubles. A membership test that
        // decoded every inline leaf (`inline_folds().binary_search`) would charge F
        // per item in `decoded_inline`; the `sum_tree` seek under `inline_fold_before`
        // is unmetered, so the O(log) path charges ~0. One disjoint fold per row;
        // probe the middle opener (a real opener, so both the seek and a whole-set
        // scan do their full work).
        let meter_for = |f: u32| -> u64 {
            let m = FoldMap::from_inline((0..f).map(|i| (i, i * 10 + 100, i * 10 + 104)), f);
            let mid = (f / 2) * 10 + 100;
            crate::perf::reset();
            let _ = m.inline_fold_at(mid);
            crate::perf::meter()
        };
        let (small, big) = (meter_for(1000), meter_for(2000));
        eprintln!("[fold_map] inline membership meter {small} -> {big}");
        assert!(
            big <= small + small / 4 + 256,
            "inline_fold_at charged {small} -> {big}: membership decoded the whole inline set \
             instead of an O(log F) offset-keyed descent"
        );
    }
}
