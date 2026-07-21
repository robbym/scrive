//! Syntax highlighting — a GUI-free public surface over a private syntect
//! engine. **Syntect is named only inside this module**; nothing from it
//! crosses the `pub` line, so the GUI crate never has to pin syntect's version.
//!
//! The core is **language-agnostic**: the grammar is an app-supplied
//! `.sublime-syntax`, injected as text (`SyntaxDef::from_sublime_syntax`), and
//! the theme an app-supplied `.tmTheme` (`TokenTheme::from_tm_theme`);
//! scrive-core ships neither.
//!
//! # Two entry points
//!
//! [`Highlighter::highlight`] tokenizes the whole document top-to-bottom on each
//! call (state carried line to line) — correct but O(lines), used as the
//! convergence oracle in tests. Production reads go through [`HighlightCache`],
//! the incremental engine: geometric shift on edit ([`HighlightCache::on_commit`]),
//! lazy end-state convergence ([`HighlightCache::tokenize_until`]), and full
//! invalidation on theme change ([`HighlightCache::set_theme`]). Untokenized
//! lines return `None` and render in the default style — never an error or a
//! stall. Each drive tokenizes at most [`HIGHLIGHT_MAX_LINES_PER_CALL`] lines,
//! so no single call can stall a frame however far a cascade wants to run.
//!
//! Retention is **virtualized** so RAM does not grow with the idle sweep:
//! spans + per-line states live only in a window around the viewport
//! ([`HighlightCache::set_window`]); everywhere else, sparse checkpoints
//! ([`HIGHLIGHT_CHECKPOINT_STRIDE`]) keep every row re-derivable. A fully swept
//! document holds `O(window + lines/stride)`, not `O(lines)` — see
//! [`HighlightCache`] for the mechanics.

use core::ops::Range;
use std::io::Cursor;
use std::sync::Arc;

use crate::buffer::Snapshot;
use crate::sum_tree::{Dimension, Item, Summary, SumTree};

use syntect::highlighting::{
    FontStyle, HighlightState, Highlighter as SyntectHighlighter, RangedHighlightIterator, Style,
    Theme, ThemeSet,
};
use syntect::parsing::{ParseState, ScopeStack, SyntaxDefinition, SyntaxReference, SyntaxSet, SyntaxSetBuilder};

/// A GUI-free color; scrive-iced maps it to `iced::Color` at render time.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Rgba {
    /// Red.
    pub r: u8,
    /// Green.
    pub g: u8,
    /// Blue.
    pub b: u8,
    /// Alpha.
    pub a: u8,
}

/// Resolved inline style for one run — the theme is consulted at tokenize time
/// (syntect's highlight iterator already yields resolved styles), so nothing
/// downstream re-touches syntect.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct SpanStyle {
    /// Foreground color.
    pub fg: Rgba,
    /// Bold.
    pub bold: bool,
    /// Italic.
    pub italic: bool,
}

/// One styled run within a single buffer line. `range` is **bytes within the
/// line** (not document offsets), so it survives line-local repair and drops
/// straight onto a display chunk; byte→cell conversion happens only at render.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct HighlightSpan {
    /// Byte range within the line.
    pub range: Range<u32>,
    /// The run's resolved style.
    pub style: SpanStyle,
}

/// Failed to parse an injected `.sublime-syntax` grammar.
#[derive(Debug, thiserror::Error)]
#[error("invalid .sublime-syntax grammar: {0}")]
pub struct SyntaxError(String);

/// Failed to parse a `.tmTheme`.
#[derive(Debug, thiserror::Error)]
#[error("invalid .tmTheme: {0}")]
pub struct ThemeError(String);

/// A parsed grammar. The app supplies a validated `.sublime-syntax` definition;
/// scrive-core ships no grammar of its own and stays language-agnostic.
pub struct SyntaxDef {
    set: SyntaxSet,
    name: String,
}

impl SyntaxDef {
    /// Parse an injected `.sublime-syntax` grammar (LF-only lines, no trailing
    /// newline in the regexes).
    pub fn from_sublime_syntax(s: &str) -> Result<Self, SyntaxError> {
        let syntax =
            SyntaxDefinition::load_from_str(s, false, None).map_err(|e| SyntaxError(e.to_string()))?;
        let name = syntax.name.clone();
        let mut builder = SyntaxSetBuilder::new();
        builder.add(syntax);
        Ok(Self { set: builder.build(), name })
    }

    /// The syntax reference to parse with — the one injected grammar.
    fn reference(&self) -> &SyntaxReference {
        self.set
            .find_syntax_by_name(&self.name)
            .or_else(|| self.set.syntaxes().first())
            .expect("the injected grammar is in its own set")
    }
}

/// Convert a syntect `(Style, byte range)` to a public [`HighlightSpan`] — the
/// one place a resolved style crosses out of syntect.
fn span_from(style: Style, range: Range<usize>) -> HighlightSpan {
    HighlightSpan {
        range: range.start as u32..range.end as u32,
        style: SpanStyle {
            fg: Rgba {
                r: style.foreground.r,
                g: style.foreground.g,
                b: style.foreground.b,
                a: style.foreground.a,
            },
            bold: style.font_style.contains(FontStyle::BOLD),
            italic: style.font_style.contains(FontStyle::ITALIC),
        },
    }
}

/// A parsed highlight theme — opaque over syntect's `Theme`. `Clone` is cheap
/// (a syntect `Theme` is a small style table), so an integrating widget can
/// retain a theme and re-apply it across a document reload or grammar swap.
#[derive(Clone)]
pub struct TokenTheme(Theme);

impl TokenTheme {
    /// Parse a `.tmTheme` (plist).
    pub fn from_tm_theme(s: &str) -> Result<Self, ThemeError> {
        ThemeSet::load_from_reader(&mut Cursor::new(s))
            .map(TokenTheme)
            .map_err(|e| ThemeError(e.to_string()))
    }
}

/// A whole-document highlighter over a grammar + theme. Owns both; syntect stays
/// private behind it.
pub struct Highlighter {
    syntax: SyntaxDef,
    theme: TokenTheme,
}

impl Highlighter {
    /// A highlighter over an injected grammar and theme.
    #[must_use]
    pub fn new(syntax: SyntaxDef, theme: TokenTheme) -> Self {
        Self { syntax, theme }
    }

    /// Tokenize `text` (LF-only) into per-line spans — one `Vec` per line
    /// (including the trailing empty final line), each span a byte range within
    /// its line. State is carried line to line, so multi-line constructs resolve.
    #[must_use]
    pub fn highlight(&self, text: &str) -> Vec<Vec<HighlightSpan>> {
        let syntect = SyntectHighlighter::new(&self.theme.0);
        let mut state = LineState {
            parse: ParseState::new(self.syntax.reference()),
            highlight: HighlightState::new(&syntect, ScopeStack::new()),
        };
        text.split('\n')
            .map(|line| {
                let (spans, next) = tokenize_line(&syntect, &self.syntax.set, &state, line);
                state = next;
                spans
            })
            .collect()
    }
}

/// The per-line carry state — syntect's `(ParseState, HighlightState)`. Its `==`
/// is the convergence probe: equal end states mean identical tokens downstream
/// (both halves are `Clone + PartialEq`). Private: no syntect type crosses the
/// module boundary.
#[derive(Clone, PartialEq)]
struct LineState {
    parse: ParseState,
    highlight: HighlightState,
}

/// Tokenize one line from `start`, returning its spans and its **end** state
/// (the start state for the next line). A parse error falls back to no spans,
/// carrying the start state forward unchanged so the tokenize loop cannot spin.
fn tokenize_line(
    syntect: &SyntectHighlighter<'_>,
    set: &SyntaxSet,
    start: &LineState,
    line: &str,
) -> (Vec<HighlightSpan>, LineState) {
    let mut end = start.clone();
    let ops = end.parse.parse_line(line, set).unwrap_or_default();
    let spans = RangedHighlightIterator::new(&mut end.highlight, &ops, line, syntect)
        .map(|(style, _text, range)| span_from(style, range))
        .collect();
    (spans, end)
}

/// Per-call tokenize budget, expressed as an op count — deterministic and
/// testable, unlike wall-clock. At ~2–20 µs of syntect per line this is
/// roughly 0.5–5 ms per call, so one drive can never stall a frame however
/// far a state-changing cascade wants to run; the idle sweep resumes where
/// the budget stopped.
pub const HIGHLIGHT_MAX_LINES_PER_CALL: u32 = 256;

/// Sparse-checkpoint stride, in rows: outside the retention window the cache
/// keeps one end state per this many tokenized rows (plus each budget-stop
/// resume point), so any row's start state is re-derivable by tokenizing at
/// most a stride forward from the checkpoint above it. Memory per fully-swept
/// document: `lines / stride` states instead of `lines`.
pub const HIGHLIGHT_CHECKPOINT_STRIDE: u32 = 256;

/// Retention slack, in rows, kept on EACH side of the window handed to
/// [`HighlightCache::set_window`] — scrolling within the slack costs nothing;
/// beyond it, evicted rows refill from the nearest checkpoint (budgeted).
pub const HIGHLIGHT_WINDOW_SLACK: u32 = 512;

/// Hard cap on the retention window's total length, in rows. A collapsed
/// mega-fold makes the reported visible range span the fold's hidden interior;
/// capping the window keeps retention bounded there instead of re-growing to
/// O(document). Rows past the cap render in the fallback style until scrolled
/// to (which re-aims the window at them); a fold hiding fewer than ~3k rows
/// still fits entirely.
pub const HIGHLIGHT_MAX_WINDOW_ROWS: u32 = 4096;

/// The retention/paint window for a viewport: `viewport` padded by
/// [`HIGHLIGHT_WINDOW_SLACK`] on each side, its length capped at
/// [`HIGHLIGHT_MAX_WINDOW_ROWS`], and clamped to `[0, n_lines]`. The **one**
/// owner of this formula: [`HighlightCache::set_window`] (what the cache
/// *retains*) and any parallel-highlight worker pool (what it speculatively
/// *paints* and *tokenizes*) must aim at the SAME rows, so both call this
/// instead of re-deriving it — an integrator running the core's speculative
/// tokenizer on worker threads cannot drift from the core's retention rule. The
/// cap is load-bearing: a collapsed mega-fold makes the reported visible range
/// span the fold's hidden interior, and without it retention (or a pool's
/// synchronous speculate) would re-grow to O(document).
#[must_use]
pub fn padded_highlight_window(viewport: Range<u32>, n_lines: u32) -> Range<u32> {
    let start = viewport.start.saturating_sub(HIGHLIGHT_WINDOW_SLACK);
    let end = viewport
        .end
        .saturating_add(HIGHLIGHT_WINDOW_SLACK)
        .min(start.saturating_add(HIGHLIGHT_MAX_WINDOW_ROWS))
        .min(n_lines)
        .max(start);
    start..end
}

/// Disjoint, ascending dirty-line ranges: the set of lines whose highlight
/// state has not yet converged. Storing runs (not individual rows) keeps a
/// commit O(edit + #runs) even when a large file carries a long dirty tail.
/// All operations are front-biased: `tokenize_until` always consumes the first
/// dirty row, and edits splice near the front of whatever tail remains, so the
/// run list stays single-digit in practice (a canary pins that).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
struct DirtyRanges(Vec<Range<u32>>);

impl DirtyRanges {
    /// Every row of an `n`-line document dirty (load / theme change) — ONE
    /// run covering the document (a `Vec` holding a single `Range` element,
    /// which is exactly the point of the range-run representation).
    fn all(n: u32) -> Self {
        Self((n > 0).then_some(0..n).into_iter().collect())
    }

    /// The first dirty row, if any — the tokenize frontier.
    fn first(&self) -> Option<u32> {
        self.0.first().map(|r| r.start)
    }

    /// Remove `row`, which MUST be the current first dirty row (the only
    /// consumption order `tokenize_until` uses). O(1) amortized.
    fn remove_first(&mut self, row: u32) {
        debug_assert_eq!(self.first(), Some(row), "consumption is front-only");
        let r = &mut self.0[0];
        r.start += 1;
        if r.start >= r.end {
            self.0.remove(0);
        }
    }

    /// Mark one row dirty (the cascade step) — merging into a neighbouring
    /// run so the list stays disjoint and non-adjacent. O(log + shift), and
    /// the cascade inserts at/near the front in practice.
    fn insert(&mut self, row: u32) {
        let i = self.0.partition_point(|r| r.end < row);
        if i < self.0.len() {
            let r = &mut self.0[i];
            if r.start <= row && row < r.end {
                return; // already dirty
            }
            if r.end == row {
                r.end += 1; // extend this run right…
                if i + 1 < self.0.len() && self.0[i].end == self.0[i + 1].start {
                    let next_end = self.0[i + 1].end;
                    self.0[i].end = next_end; // …merging into the next run
                    self.0.remove(i + 1);
                }
                return;
            }
            if r.start == row + 1 {
                r.start = row; // extend this run left
                return;
            }
        }
        self.0.insert(i, row..row + 1);
    }

    /// The commit splice: `spans` is the per-edit list of pre-edit line
    /// spans `(pre_start, old_lines, new_lines)`, ascending and disjoint. Shifts
    /// existing runs through the combined splices and marks ONLY each edit's own
    /// post-edit line span dirty — so a scattered multi-caret transaction marks
    /// O(carets) lines, not the whole first-to-last covering range. One pass and
    /// one sort of the O(runs + edits) boundary set. A single covering span
    /// reproduces the classic clip-shift-mark commit splice exactly.
    fn apply_splices(&mut self, spans: &[(u32, u32, u32)]) {
        if spans.is_empty() {
            return;
        }
        // `pref[i]` = accumulated line delta of `spans[..i]`. `shift_at(pre)` is
        // the post-edit displacement of a pre-edit row *not inside* any edit
        // (rows inside an edit are covered by that edit's new span below), which
        // is the delta of all edits entirely above it.
        let mut pref: Vec<i64> = Vec::with_capacity(spans.len() + 1);
        pref.push(0);
        for &(_, o, n) in spans {
            pref.push(pref[pref.len() - 1] + i64::from(n) - i64::from(o));
        }
        let shift_at = |pre: u32| -> i64 {
            let i = spans.partition_point(|s| s.0 + s.1 <= pre);
            pref[i]
        };
        let old_runs = std::mem::take(&mut self.0);
        let mut out: Vec<Range<u32>> = Vec::with_capacity(old_runs.len() + spans.len());
        // Existing runs, shifted. `[a + shift(a), b + shift(b))` is a superset
        // of the shifted gap rows and any spanned edit region (shift is
        // non-decreasing), which is safe: over-marking dirty only costs re-work,
        // under-marking would leave a stale color.
        for r in &old_runs {
            let a = (i64::from(r.start) + shift_at(r.start)) as u32;
            let b = (i64::from(r.end) + shift_at(r.end)) as u32;
            if a < b {
                out.push(a..b);
            }
        }
        // Each edit's own post-edit new span.
        for (j, &(ps, _o, n)) in spans.iter().enumerate() {
            if n > 0 {
                let post_start = (i64::from(ps) + pref[j]) as u32;
                out.push(post_start..post_start + n);
            }
        }
        out.sort_unstable_by_key(|r| r.start);
        // Coalesce touching/overlapping runs back into the ascending-disjoint
        // invariant `splice`/`clear_range` maintain.
        let mut merged: Vec<Range<u32>> = Vec::with_capacity(out.len());
        for r in out {
            match merged.last_mut() {
                Some(last) if last.end >= r.start => last.end = last.end.max(r.end),
                _ => merged.push(r),
            }
        }
        self.0 = merged;
    }

    /// Total dirty rows (sum of run lengths) — the invalidation size a commit
    /// scheduled, which the sweep must eventually walk.
    #[cfg(test)]
    fn total_rows(&self) -> u32 {
        self.0.iter().map(|r| r.end - r.start).sum()
    }

    /// Whether `row` is dirty. Binary search over the ascending disjoint runs.
    fn contains(&self, row: u32) -> bool {
        let i = self.0.partition_point(|r| r.end <= row);
        i < self.0.len() && self.0[i].start <= row
    }

    /// Clear rows `[range)` from the dirty set — the verified-absorb path: those
    /// rows' states are proven, so they leave the frontier without being walked.
    /// Clips each run around the range; the output stays ascending, disjoint,
    /// and non-adjacent (inputs are, and clipping only shrinks), so front-biased
    /// consumption is preserved.
    fn clear_range(&mut self, range: Range<u32>) {
        if range.start >= range.end {
            return;
        }
        let old = std::mem::take(&mut self.0);
        let mut out: Vec<Range<u32>> = Vec::with_capacity(old.len() + 1);
        for r in old {
            let lo = r.start..r.end.min(range.start); // part below the cleared span
            let hi = r.start.max(range.end)..r.end; // part at/above it
            if lo.start < lo.end {
                out.push(lo);
            }
            if hi.start < hi.end {
                out.push(hi);
            }
        }
        self.0 = out;
    }

    /// Number of disjoint runs — the canary probe.
    #[cfg(test)]
    fn runs(&self) -> usize {
        self.0.len()
    }
}

/// The document-owned **incremental, virtualized** highlight cache.
///
/// The *correctness* model is end-state convergence over `DirtyRanges`:
/// a line is re-tokenized only while its computed end state differs from the
/// stored one, so an edit repaints just the lines whose state actually
/// changed. It is driven lazily and budgeted by
/// [`HighlightCache::tokenize_until`]. **Retention** is what keeps memory
/// bounded — two facts make it work:
///
/// - **Spans are draw output** — nothing but visible rows reads them, so they
///   are retained only inside a *dense window* (the viewport ±
///   [`HIGHLIGHT_WINDOW_SLACK`], set via [`HighlightCache::set_window`]),
///   together with per-line end states there (so a keystroke at the caret
///   still converges per-line).
/// - **States exist to be resumed from** — and any row's state is
///   re-derivable by tokenizing forward, so outside the window only sparse
///   *checkpoints* survive: one per [`HIGHLIGHT_CHECKPOINT_STRIDE`] tokenized
///   rows plus each budget-stop resume point. A cold jump refills the window
///   from the nearest checkpoint (≤ a stride of warm-up, budgeted; the rows
///   render in the fallback style until filled).
///
/// Convergence verification is per-line inside the window and
/// checkpoint-grain outside (an out-of-window cascade may run up to one
/// stride past where a fully-dense cache would have stopped — bounded, and
/// such edits are rare: you edit what you see). Memory for a fully swept
/// document: `O(window + lines / stride)` instead of `O(lines)`.
pub struct HighlightCache {
    /// `Arc`-shared so [`HighlightCache::engine`] hands an off-thread worker
    /// the SAME immutable grammar/theme without a deep `SyntaxSet` clone.
    syntax: Arc<SyntaxDef>,
    theme: Arc<TokenTheme>,
    /// Everything the walks mutate — split from the grammar/theme so the
    /// borrow of `theme` inside a live `SyntectHighlighter` stays disjoint
    /// from the retention mutations.
    ret: Retention,
}

/// The virtualized retention state (see [`HighlightCache`]).
struct Retention {
    /// Buffer line count (the commit path keeps it in step).
    n_lines: u32,
    /// Dirty-line runs — the correctness frontier: lines not yet converged.
    invalid: DirtyRanges,
    /// The last window AIM (the rows handed to `set_window`, pre-padding) —
    /// preserved across grammar swaps so `set_syntax` keeps retention aimed at
    /// the viewport instead of silently resetting to the document top.
    aim: Range<u32>,
    /// The dense retention window: rows `[win.start, win.end)`.
    win: Range<u32>,
    /// Spans for window rows (`None` ⇒ fallback / not yet filled).
    win_spans: Vec<Option<Vec<HighlightSpan>>>,
    /// End states for window rows — the per-line convergence probes.
    win_states: Vec<Option<Box<LineState>>>,
    /// Sparse end-state checkpoints, ascending by row — a delta-gap
    /// [`SumTree`]: reads are an O(log) seek and the per-keystroke
    /// [`Checkpoints::shift`] mover rewrites one seam gap rather than the whole
    /// list. Holds stride rows plus budget-stop resume points; splices shift
    /// them like every other line-indexed structure.
    checkpoints: Checkpoints,
}

// Manual (syntect's `Theme`/state aren't all `Debug`, and they'd be noise
// anyway); lets `Document` keep its `#[derive(Debug)]`.
impl core::fmt::Debug for HighlightCache {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("HighlightCache")
            .field("lines", &self.ret.n_lines)
            .field("window", &self.ret.win)
            .field("checkpoints", &self.ret.checkpoints.len())
            .field("dirty", &self.ret.invalid)
            .finish_non_exhaustive()
    }
}

// ======================================================================
// Sparse end-state checkpoints — a delta-gap `SumTree`.
//
// One of the delta-gap `SumTree` structures (shared with decorations, folds,
// and brackets): every read is an O(log) seek, the per-keystroke mover
// (`shift`) shifts a suffix by rewriting ONE seam gap, and only a scattered
// MULTI-edit commit falls back to the correctness-first O(#checkpoints) walk.
// `row_gap` is this checkpoint's row minus the previous one's (a prefix sum
// over `RowDim` recovers the absolute row); `state` is the end state after
// that row — opaque and heavy (a syntect `LineState`), the copy-on-write cost
// the `benches/perf.rs` checkpoint case pins.
// ======================================================================

/// One sparse checkpoint leaf. `row_gap` = this checkpoint's row minus the
/// previous checkpoint's row (delta-gap, so a suffix shift rewrites just one
/// gap); `state` = the end state after that row.
struct CkptItem {
    row_gap: u32,
    state: Box<LineState>,
}

impl Clone for CkptItem {
    fn clone(&self) -> Self {
        Self { row_gap: self.row_gap, state: self.state.clone() }
    }
}

// `LineState` is deliberately not `Debug` (syntect's parse/highlight state is
// noise) — the same reason `HighlightCache`'s `Debug` is manual. `Item` demands
// `Debug`, so show the shape without the state.
impl core::fmt::Debug for CkptItem {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("CkptItem").field("row_gap", &self.row_gap).finish_non_exhaustive()
    }
}

/// Subtree aggregate: `rows` = Σ row_gaps (the [`RowDim`] absolute-row extent),
/// `count` = item count (the [`CkptIx`] index extent). Both additive monoids, so
/// a suffix shift is invariant under a uniform row displacement.
#[derive(Clone, Copy, Debug, Default)]
struct CkptSummary {
    rows: u32,
    count: u32,
}

impl Summary for CkptSummary {
    fn add_summary(&mut self, o: &Self) {
        self.rows += o.rows;
        self.count += o.count;
    }
}

impl Item for CkptItem {
    type Summary = CkptSummary;
    fn summary(&self) -> CkptSummary {
        CkptSummary { rows: self.row_gap, count: 1 }
    }
}

/// Dimension: absolute row — accumulates each item's `row_gap` (a prefix sum),
/// so an item's `RowDim`-end is its absolute checkpoint row.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct RowDim(u32);
impl Dimension<CkptSummary> for RowDim {
    fn add_summary(&mut self, s: &CkptSummary) {
        self.0 += s.rows;
    }
}

/// Dimension: item index — accumulates `count`. Writes split/replace by index
/// (a delta-gap item is located at its row, not a byte offset), mirroring the
/// decoration and bracket trees.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct CkptIx(u32);
impl Dimension<CkptSummary> for CkptIx {
    fn add_summary(&mut self, s: &CkptSummary) {
        self.0 += s.count;
    }
}

/// Re-anchor `suffix`'s first item onto a rebuilt head: its absolute row becomes
/// `old_row + delta` and its predecessor is now at `prev_row`, so its single
/// `row_gap` is rewritten while every later gap stays relative and untouched — a
/// whole-suffix shift in O(log). `k_hi` is that item's index in `orig` (read
/// only for its old absolute row). A port of the decoration tree's `reanchor`
/// (oracle-proven there); the heavy `state` is cloned exactly once.
fn reanchor(
    suffix: &SumTree<CkptItem>,
    k_hi: u32,
    orig: &SumTree<CkptItem>,
    delta: i64,
    prev_row: u32,
) -> SumTree<CkptItem> {
    if suffix.is_empty() {
        return suffix.clone();
    }
    let (item, _c, RowDim(r)) = orig
        .seek::<CkptIx, RowDim>(&CkptIx(k_hi))
        .expect("suffix non-empty ⇒ a k_hi-th item exists");
    let first_old_row = r + item.row_gap;
    let new_gap = (i64::from(first_old_row) + delta - i64::from(prev_row)) as u32;
    let fixed = CkptItem { row_gap: new_gap, state: item.state.clone() };
    suffix.replace(CkptIx(0)..CkptIx(1), std::iter::once(fixed))
}

/// Sparse end-state checkpoints as a delta-gap [`SumTree`]. See the module
/// section header above for the structure and its cost model.
struct Checkpoints {
    tree: SumTree<CkptItem>,
}

impl Checkpoints {
    fn new() -> Self {
        Self { tree: SumTree::new() }
    }

    /// Number of checkpoints — an O(1) root-summary read.
    fn len(&self) -> usize {
        self.tree.summary().count as usize
    }

    #[cfg(test)]
    fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    fn clear(&mut self) {
        self.tree = SumTree::new();
    }

    /// The state stored EXACTLY at `row`, or `None`. O(log). The `stored_state_after`
    /// checkpoint probe.
    fn state_at(&self, row: u32) -> Option<&LineState> {
        let n_le = self.tree.summary_before(&RowDim(row.saturating_add(1))).count; // abs_row <= row
        let n_lt = self.tree.summary_before(&RowDim(row)).count; //                   abs_row < row
        if n_le == n_lt {
            return None; // nothing sits at exactly `row`
        }
        let (item, _ix, RowDim(before)) = self.tree.seek::<CkptIx, RowDim>(&CkptIx(n_lt))?;
        debug_assert_eq!(before + item.row_gap, row, "state_at must land exactly on `row`");
        Some(&*item.state)
    }

    /// The nearest checkpoint at or before `row`: `(abs_row, &state)`, or `None`.
    /// O(log). The warm-up floor.
    fn floor(&self, row: u32) -> Option<(u32, &LineState)> {
        let n_le = self.tree.summary_before(&RowDim(row.saturating_add(1))).count; // abs_row <= row
        let idx = n_le.checked_sub(1)?; // none at or before `row`
        let (item, _ix, RowDim(before)) = self.tree.seek::<CkptIx, RowDim>(&CkptIx(idx))?;
        Some((before + item.row_gap, &*item.state))
    }

    /// Replace the state of the checkpoint at index `i`, keeping its `row_gap`
    /// (its position). O(log).
    fn replace_state_at_index(&mut self, i: u32, state: LineState) {
        let gap = {
            let (item, _ix, _rd) =
                self.tree.seek::<CkptIx, RowDim>(&CkptIx(i)).expect("index in range");
            item.row_gap
        };
        self.tree = self
            .tree
            .replace(CkptIx(i)..CkptIx(i + 1), std::iter::once(CkptItem { row_gap: gap, state: Box::new(state) }));
    }

    /// Insert a new checkpoint at absolute `row` (which must NOT already exist)
    /// at index `i`: mint its delta-gap onto its predecessor and re-relativize the
    /// successor's now-orphaned gap at one seam. Insertion moves no rows, so the
    /// suffix re-anchors with `delta = 0`. O(log).
    fn insert_at(&mut self, i: u32, row: u32, state: LineState) {
        let (left, right) = self.tree.split_at(&CkptIx(i));
        let pred = left.extent::<RowDim>().0; // abs_row of item i-1 (0 if none)
        let mid = SumTree::from_items(std::iter::once(CkptItem {
            row_gap: row - pred,
            state: Box::new(state),
        }));
        // The successor keeps its absolute row; only its predecessor changed to
        // `row`, so it re-anchors with delta 0.
        let right_fixed = reanchor(&right, i, &self.tree, 0, row);
        self.tree = left.append(&mid).append(&right_fixed);
    }

    /// Insert a checkpoint at `row`, or refresh the state if one already sits
    /// there. O(log).
    fn upsert(&mut self, row: u32, state: LineState) {
        let n_le = self.tree.summary_before(&RowDim(row.saturating_add(1))).count; // abs_row <= row
        let n_lt = self.tree.summary_before(&RowDim(row)).count; //                   abs_row < row
        if n_le > n_lt {
            self.replace_state_at_index(n_lt, state); // exact match at index n_lt
        } else {
            self.insert_at(n_lt, row, state);
        }
    }

    /// Refresh the state at `row` ONLY if a checkpoint already sits there (a
    /// drifted off-grid pin a fresh tokenize just re-derived); a no-op otherwise.
    /// O(log).
    fn refresh_if_present(&mut self, row: u32, state: LineState) {
        let n_le = self.tree.summary_before(&RowDim(row.saturating_add(1))).count;
        let n_lt = self.tree.summary_before(&RowDim(row)).count;
        if n_le > n_lt {
            self.replace_state_at_index(n_lt, state);
        }
    }

    /// Insert/refresh a stride-aligned checkpoint at `row`, then drop the
    /// off-grid pins within one stride behind it (drifted pins and old resume
    /// points, redundant once this one lands). THE one owner of the
    /// sparse-checkpoint rule. O(log + pruned).
    fn set_stride(&mut self, row: u32, state: LineState) {
        self.upsert(row, state);
        self.prune_behind(row);
    }

    /// Drop the off-grid checkpoints strictly between `row - stride` and `row`
    /// (exclusive of both an exact-`floor` pin and `row` itself) — redundant once
    /// a stride checkpoint lands at `row`. `row`'s checkpoint keeps its absolute
    /// position; only its `row_gap` re-relativizes onto the surviving predecessor.
    /// O(log + pruned).
    fn prune_behind(&mut self, row: u32) {
        let floor = row.saturating_sub(HIGHLIGHT_CHECKPOINT_STRIDE);
        let k_lo = self.tree.summary_before(&RowDim(floor.saturating_add(1))).count; // abs_row <= floor
        let k_hi = self.tree.summary_before(&RowDim(row)).count; //                     abs_row < row
        if k_hi <= k_lo {
            return; // nothing off-grid behind `row`
        }
        let (left, rest) = self.tree.split_at(&CkptIx(k_lo));
        let (_dropped, right) = rest.split_at(&CkptIx(k_hi - k_lo));
        // `right` starts with `row`'s checkpoint (abs_row == row); it stays put
        // (delta 0), re-gapped onto the last surviving item at/below `floor`.
        let right_fixed = reanchor(&right, k_hi, &self.tree, 0, left.extent::<RowDim>().0);
        self.tree = left.append(&right_fixed);
    }

    /// Shift checkpoints through a commit's per-edit line splices `(pre_start,
    /// old_lines, new_lines)` (ascending, disjoint) — the per-keystroke mover,
    /// run on EVERY commit and every undo step (`on_commit_patch`). A single
    /// edit takes the O(edit + log) windowed path; a scattered multi-edit
    /// commit falls back to the correctness-first O(#checkpoints) walk.
    fn shift(&mut self, spans: &[(u32, u32, u32)]) {
        if spans.is_empty() {
            return;
        }
        if spans.len() == 1 {
            self.shift_single(spans[0]);
        } else {
            self.shift_walk(spans);
        }
    }

    /// The single-edit fast path: drop the (usually 0) checkpoints inside the
    /// replaced rows `[s, s+o)` and shift the suffix `[s+o, ∞)` by the line delta
    /// at ONE seam. O(edit + log). Charges only the checkpoints actually TOUCHED
    /// (dropped band + the one re-anchor seam), so a keystroke far from any
    /// checkpoint is flat in the checkpoint count.
    fn shift_single(&mut self, (s, o, n): (u32, u32, u32)) {
        let d = i64::from(n) - i64::from(o);
        let k_lo = self.tree.summary_before(&RowDim(s)).count; //     abs_row < s
        let k_hi = self.tree.summary_before(&RowDim(s + o)).count; // abs_row < s+o
        let dropped = k_hi - k_lo;
        crate::perf::charge(u64::from(dropped) + 1);
        if dropped == 0 && d == 0 {
            return; // nothing dropped, nothing shifts — tree untouched
        }
        let (left, rest) = self.tree.split_at(&CkptIx(k_lo));
        let (_dropped, suffix) = rest.split_at(&CkptIx(k_hi - k_lo));
        let pred = left.extent::<RowDim>().0;
        let suffix_fixed = reanchor(&suffix, k_hi, &self.tree, d, pred);
        self.tree = left.append(&suffix_fixed);
    }

    /// The multi-edit fallback: decode to absolute `(row, state)`, run the
    /// merge (drop checkpoints inside any edit, shift the rest by the cumulative
    /// delta of all edits entirely above them), rebuild. O(#checkpoints +
    /// edits). Charges per checkpoint walked — the cost the perf gate meters, so
    /// the single-edit seam path stays the norm.
    fn shift_walk(&mut self, spans: &[(u32, u32, u32)]) {
        let old = self.tree.items();
        let mut out: Vec<CkptItem> = Vec::with_capacity(old.len());
        let mut si = 0usize;
        let mut acc = 0i64;
        let mut abs = 0u32; // running absolute row of the current checkpoint
        let mut prev_out = 0u32; // last emitted absolute row (delta-gap re-encode)
        for item in old {
            crate::perf::charge(1); // walks every checkpoint — the O(#checkpoints) cost
            abs += item.row_gap;
            while si < spans.len() && spans[si].0 + spans[si].1 <= abs {
                acc += i64::from(spans[si].2) - i64::from(spans[si].1);
                si += 1;
            }
            if si < spans.len() && abs >= spans[si].0 {
                continue; // inside a replaced span → the stored state is stale
            }
            let new_abs = (i64::from(abs) + acc) as u32;
            out.push(CkptItem { row_gap: new_abs - prev_out, state: item.state });
            prev_out = new_abs;
        }
        self.tree = SumTree::from_items(out);
    }

    /// Build a checkpoint tree from ascending absolute `rows` sharing one `state`
    /// — the perf-cell / oracle seed (no real sweep required).
    #[cfg(test)]
    fn from_rows(rows: &[u32], state: &LineState) -> Self {
        let mut prev = 0u32;
        let items: Vec<CkptItem> = rows
            .iter()
            .map(|&r| {
                let gap = r - prev;
                prev = r;
                CkptItem { row_gap: gap, state: Box::new(state.clone()) }
            })
            .collect();
        Self { tree: SumTree::from_items(items) }
    }

    /// The absolute checkpoint rows, ascending — the oracle handle.
    #[cfg(test)]
    fn rows(&self) -> Vec<u32> {
        let mut abs = 0u32;
        self.tree
            .items()
            .iter()
            .map(|it| {
                abs += it.row_gap;
                abs
            })
            .collect()
    }
}

/// How [`Retention::start_state_for`] answers: the start state is stored, is
/// the fresh document-top state, or must be warmed up by tokenizing forward
/// from the nearest earlier stored state (`WarmupFresh` ⇒ from row 0).
enum StartState {
    Ready(LineState),
    Fresh,
    WarmupStored(u32, LineState),
    WarmupFresh,
}

impl HighlightCache {
    /// A cache for an `n_lines` document over an injected grammar + theme:
    /// every line invalid, nothing retained. The retention window defaults to
    /// the document top (2 × [`HIGHLIGHT_WINDOW_SLACK`] rows) — right for a
    /// freshly opened view; the app re-aims it per viewport report via
    /// [`HighlightCache::set_window`].
    #[must_use]
    pub fn new(syntax: SyntaxDef, theme: TokenTheme, n_lines: u32) -> Self {
        let win = 0..(2 * HIGHLIGHT_WINDOW_SLACK).min(n_lines);
        let len = win.len();
        Self {
            syntax: Arc::new(syntax),
            theme: Arc::new(theme),
            ret: Retention {
                n_lines,
                invalid: DirtyRanges::all(n_lines),
                aim: win.clone(),
                win,
                win_spans: vec![None; len],
                win_states: vec![None; len],
                checkpoints: Checkpoints::new(),
            },
        }
    }

    /// The first dirty row, if any — `None` means every line's state has
    /// converged (spans may still be evicted outside the window; see
    /// [`HighlightCache::pending`], which the sweep polls instead).
    #[must_use]
    pub fn first_dirty(&self) -> Option<u32> {
        self.ret.invalid.first()
    }

    /// The next row a budgeted [`HighlightCache::tokenize_until`] call would
    /// work on: the first dirty row, or — once states have converged — the
    /// first window row whose spans await a refill (a window move over
    /// already-swept rows). `None` = nothing to do (the idle-zero-work probe;
    /// the app's sweep keys its subscription off this).
    #[must_use]
    pub fn pending(&self) -> Option<u32> {
        let dirt = self.ret.invalid.first();
        let gap = self.ret.first_window_gap(dirt.unwrap_or(u32::MAX));
        match (dirt, gap) {
            (Some(d), Some(g)) => Some(d.min(g)),
            (d, g) => d.or(g),
        }
    }

    /// The cached spans for a line, or `None` if the row is outside the
    /// retention window or not yet tokenized/refilled (the caller renders
    /// that line in the default style — never an error, never a stall).
    #[must_use]
    pub fn line_spans(&self, row: u32) -> Option<&[HighlightSpan]> {
        self.ret.win_spans[self.ret.win_idx(row)?].as_deref()
    }

    /// How many lines the cache is sized for — must track the buffer's line
    /// count (the commit path keeps them equal).
    #[must_use]
    pub fn line_count(&self) -> u32 {
        self.ret.n_lines
    }

    /// The last rows handed to [`HighlightCache::set_window`] (pre-padding) —
    /// `Document::set_syntax` re-aims a replacement cache with this so a
    /// grammar swap keeps retention at the viewport instead of silently
    /// resetting to the document top.
    #[must_use]
    pub fn window_aim(&self) -> Range<u32> {
        self.ret.aim.clone()
    }

    /// Aim the retention window at `rows` (the viewport, in buffer rows) —
    /// the cache retains `rows` ± [`HIGHLIGHT_WINDOW_SLACK`] and evicts
    /// outside. Repositioning is O(window); rows entering the window refill
    /// on the next [`HighlightCache::tokenize_until`] (see
    /// [`HighlightCache::pending`]).
    pub fn set_window(&mut self, rows: Range<u32>) {
        let r = &mut self.ret;
        r.aim = rows.clone();
        // The padded, length-capped window through the ONE owner
        // (`padded_highlight_window`), shared with any parallel-highlight pool so
        // the rows the cache retains and the rows a pool paints cannot drift.
        let win = padded_highlight_window(rows, r.n_lines);
        let (start, end) = (win.start, win.end);
        if (start..end) == r.win {
            return;
        }
        let len = (end - start) as usize;
        let mut spans = vec![None; len];
        let mut states = vec![None; len];
        for row in start.max(r.win.start)..end.min(r.win.end) {
            let old_i = (row - r.win.start) as usize;
            let new_i = (row - start) as usize;
            spans[new_i] = r.win_spans[old_i].take();
            states[new_i] = r.win_states[old_i].take();
        }
        r.win = start..end;
        r.win_spans = spans;
        r.win_states = states;
    }

    /// The commit hook: buffer lines `[start, start+old)` became `new` lines.
    /// Splices the window (keeping the **old end state of the last edited
    /// line** as the convergence probe in the new last slot), drops replaced
    /// checkpoints and shifts the rest, and marks `[start, start+new)` dirty.
    /// O(edit + window + #checkpoints); the cascade waits for `tokenize_until`.
    pub fn on_commit(&mut self, start: u32, old: u32, new: u32) {
        // A single edit as one line splice. Multi-edit commits use the per-edit
        // spans directly (see `on_commit_patch`).
        self.on_commit_patch(&[(start, old, new)]);
    }

    /// The commit hook, given the **per-edit** pre-edit line spans `(pre_start,
    /// old_lines, new_lines)` (ascending, disjoint). Checkpoints, the dense
    /// window, and the dirty runs all ride the per-edit spans — NOT the whole
    /// transaction's first-to-last covering range. A scattered multi-caret
    /// transaction has a document-scale covering range but O(carets) tiny spans,
    /// so this keeps the commit (and the sweep it schedules) O(carets + window +
    /// checkpoints), and — critically — leaves the retention window aimed at the
    /// viewport: only the actually-edited in-window rows are invalidated, so the
    /// visible rows recolor on the next tokenize instead of going stale until a
    /// scroll re-aims the window.
    pub fn on_commit_patch(&mut self, spans: &[(u32, u32, u32)]) {
        if spans.is_empty() {
            return;
        }
        let r = &mut self.ret;
        let delta: i64 = spans.iter().map(|&(_, o, n)| i64::from(n) - i64::from(o)).sum();
        // Checkpoints: drop only those inside an actual edit and shift the rest
        // by the per-edit deltas — scattered carets must not wipe every
        // checkpoint between the first and last (a cold-jump refill needs them).
        r.checkpoints.shift(spans);
        // Window: shift through the per-edit spans, staying aimed at the viewport.
        r.shift_window(spans);
        // Dirty runs: mark ONLY each edit's own post-edit line span.
        r.invalid.apply_splices(spans);
        r.n_lines = (r.n_lines as i64 + delta) as u32;
    }

    /// Tokenize toward `target` (inclusive), lazily and budgeted, and return
    /// **how many lines were tokenized** (warm-up included) — the op count
    /// the perf canaries assert on. Two phases share the `max_lines` budget:
    ///
    /// 1. **The dirty walk** — dense convergence semantics: each first-dirty
    ///    line tokenizes from its start state (stored, or warmed up from the
    ///    nearest checkpoint); if its new end state differs from the stored
    ///    one the cascade continues, otherwise it stops (convergence).
    ///    Where no state is stored to compare against (outside the window,
    ///    between checkpoints) the cascade keeps going until one is — at
    ///    most a stride more than a fully-dense cache would have done.
    /// 2. **The window refill** — rows inside the retention window whose
    ///    spans were evicted (a window move over already-swept text) are
    ///    re-derived from the nearest checkpoint. Only rows below the dirty
    ///    frontier are fillable (states past dirt are unverified).
    ///
    /// `line(i)` hands back line i's text — an ACCESSOR, never an all-lines
    /// Vec, so this never materializes the whole document. A budget stop
    /// records a resume checkpoint so the next call continues without re-warming.
    pub fn tokenize_until<S: AsRef<str>>(
        &mut self,
        target: u32,
        max_lines: u32,
        mut line: impl FnMut(u32) -> S,
    ) -> u32 {
        let syntect = SyntectHighlighter::new(&self.theme.0);
        let syntax = &*self.syntax;
        let fresh = || LineState {
            parse: ParseState::new(syntax.reference()),
            highlight: HighlightState::new(&syntect, ScopeStack::new()),
        };
        let ret = &mut self.ret;
        let mut work = 0u32;

        // Phase 1 — the dirty walk.
        'walk: while work < max_lines {
            let Some(first) = ret.invalid.first() else { break };
            if first > target {
                break;
            }
            let started = match ret.start_state_for(first) {
                StartState::Ready(s) => Some(s),
                StartState::Fresh => Some(fresh()),
                StartState::WarmupStored(r, s) => ret
                    .warm_up(Some(r), first, s, &mut work, max_lines, &syntect, &syntax.set, &mut line),
                StartState::WarmupFresh => ret
                    .warm_up(None, first, fresh(), &mut work, max_lines, &syntect, &syntax.set, &mut line),
            };
            let Some(mut state) = started else { break 'walk }; // budget died mid-warm-up
            let mut row = first;
            loop {
                if work >= max_lines {
                    ret.keep_resume_checkpoint(row.checked_sub(1), &state);
                    break 'walk;
                }
                let (spans, end) = tokenize_line(&syntect, &syntax.set, &state, line(row).as_ref());
                // Probe BEFORE retain overwrites; nothing stored ⇒ unverifiable
                // ⇒ keep cascading.
                let converged = ret.stored_state_after(row).is_some_and(|old| *old == end);
                ret.retain(row, spans, &end);
                ret.invalid.remove_first(row);
                work += 1;
                state = end;
                if converged || row + 1 >= ret.n_lines {
                    break;
                }
                ret.invalid.insert(row + 1);
                if row + 1 > target {
                    break; // beyond the caller's viewport — the sweep's job
                }
                row += 1;
            }
        }

        // Phase 2 — the window refill.
        'fill: while work < max_lines {
            let dirt = ret.invalid.first().unwrap_or(u32::MAX);
            let Some(gap) = ret.first_window_gap(dirt) else { break };
            let started = match ret.start_state_for(gap) {
                StartState::Ready(s) => Some(s),
                StartState::Fresh => Some(fresh()),
                StartState::WarmupStored(r, s) => ret
                    .warm_up(Some(r), gap, s, &mut work, max_lines, &syntect, &syntax.set, &mut line),
                StartState::WarmupFresh => ret
                    .warm_up(None, gap, fresh(), &mut work, max_lines, &syntect, &syntax.set, &mut line),
            };
            let Some(mut state) = started else { break 'fill };
            let end_row = ret.win.end.min(ret.n_lines).min(dirt);
            let mut row = gap;
            while row < end_row {
                if work >= max_lines {
                    ret.keep_resume_checkpoint(row.checked_sub(1), &state);
                    break 'fill;
                }
                if ret.win_spans[(row - ret.win.start) as usize].is_some() {
                    if let Some(s) = ret.stored_state_after(row) {
                        state = s.clone(); // hop an already-filled row
                        row += 1;
                        continue;
                    }
                }
                let (spans, end) = tokenize_line(&syntect, &syntax.set, &state, line(row).as_ref());
                ret.retain(row, spans, &end);
                work += 1;
                state = end;
                row += 1;
            }
        }
        work
    }

    /// Swap the theme and invalidate the whole cache: a theme change is a full
    /// invalidation because syntect resolves colors at tokenize time, so every
    /// line's styles change and must be re-tokenized. States and checkpoints
    /// are dropped and every line marked dirty; the **old window spans are kept
    /// as a stale fallback** so the view keeps showing the outgoing theme's
    /// colors (not a flash to default) until the frontier repaints them on the
    /// next `tokenize_until`.
    pub fn set_theme(&mut self, theme: TokenTheme) {
        self.theme = Arc::new(theme);
        for s in &mut self.ret.win_states {
            *s = None;
        }
        self.ret.checkpoints.clear();
        self.ret.invalid = DirtyRanges::all(self.ret.n_lines);
    }

    /// A cheap-to-clone, `Send + Sync` handle to this cache's grammar + theme,
    /// for off-thread bulk tokenization (the parallel/speculative sweep).
    /// Shares the `Arc`-held grammar/theme — no `SyntaxSet` deep clone.
    #[must_use]
    pub fn engine(&self) -> HighlightEngine {
        HighlightEngine { syntax: self.syntax.clone(), theme: self.theme.clone() }
    }

    /// Ingest a segment tokenized off-thread (parallel/speculative sweep). The
    /// caller ([`crate::Document::absorb_highlight`]) guarantees the segment
    /// was computed at the current revision, so `rows`/checkpoints align with
    /// this cache.
    ///
    /// - `verified` — the coordinator chained this segment from row 0, so its
    ///   start state (and thus its whole result) is TRUE. Merge its stride
    ///   checkpoints (through the one [`Checkpoints::set_stride`]
    ///   owner, so the set matches a sync walk), write its window spans+states
    ///   into the current retention window, and **clear `rows` from the dirty
    ///   set** — this replaces the foreground dirty walk for that span.
    /// - `!verified` — viewport speculation from a GUESSED fresh start. Write
    ///   window spans+states ONLY into window rows that are still dirty (never
    ///   clobber verified data), and **never plant a checkpoint**: an
    ///   unverified state outside the window would poison a later warm-up.
    ///   Dirt is untouched, so the frontier re-verifies
    ///   these rows per-line and converges in O(1) if the guess was right; a
    ///   dirty row's state is only ever read as a convergence probe, never as
    ///   a warm-up source (that reads `first_dirty − 1`, which is clean).
    pub(crate) fn absorb(&mut self, seg: SegmentTokens, verified: bool) {
        let r = &mut self.ret;
        let SegmentTokens { rows, checkpoints, win, win_spans, win_states, .. } = seg;
        if verified {
            for (row, state) in checkpoints {
                r.checkpoints.set_stride(row, *state);
            }
            // Evict window rows this segment COVERS but did not SUPPLY spans
            // for: they are about to be marked clean, so any speculative span
            // there would survive as trusted-but-unverified (poison). `None`
            // renders as fallback and is refilled from the (correct)
            // checkpoints by the sync window sweep. Supplied rows are written
            // below, so they show correct colors with no flash.
            let cover = rows.start.max(r.win.start)..rows.end.min(r.win.end);
            for row in cover {
                if !win.contains(&row) {
                    let i = (row - r.win.start) as usize;
                    r.win_spans[i] = None;
                    r.win_states[i] = None;
                }
            }
            for ((row, spans), state) in win.zip(win_spans).zip(win_states) {
                if let Some(i) = r.win_idx(row) {
                    r.win_spans[i] = Some(spans);
                    r.win_states[i] = Some(Box::new(state));
                }
            }
            r.invalid.clear_range(rows);
        } else {
            for ((row, spans), state) in win.zip(win_spans).zip(win_states) {
                // Only still-dirty window rows: never clobber verified spans,
                // and a dirty row's state is a probe, not a warm-up source.
                if r.invalid.contains(row) {
                    if let Some(i) = r.win_idx(row) {
                        r.win_spans[i] = Some(spans);
                        r.win_states[i] = Some(Box::new(state));
                    }
                }
            }
        }
    }

    // Test-only retention probes — the memory canaries count entries, the
    // deterministic op-count analog of an RSS measurement.
    #[cfg(test)]
    fn retained_span_rows(&self) -> usize {
        self.ret.win_spans.iter().filter(|s| s.is_some()).count()
    }
    #[cfg(test)]
    fn window_len(&self) -> usize {
        debug_assert_eq!(self.ret.win_spans.len(), self.ret.win_states.len());
        self.ret.win_spans.len()
    }
    #[cfg(test)]
    fn dirty_row_count(&self) -> u32 {
        self.ret.invalid.total_rows()
    }
    #[cfg(test)]
    fn retained_state_rows(&self) -> usize {
        self.ret.win_states.iter().filter(|s| s.is_some()).count() + self.ret.checkpoints.len()
    }
}

// ======================================================================
// Off-thread bulk tokenization (parallel + speculative sweep).
//
// The retention cache above serves the UI thread synchronously. On a
// multi-million-line document the top-down state chain would make a
// jump-to-bottom render fall back to plain text for ~16-20s of single-core
// work. Instead the app tokenizes SEGMENTS off-thread over an O(1) `Snapshot`
// clone, from GUESSED fresh states, and STITCHES boundaries with the same
// convergence rule the cache uses per line: segment i's true end state equals
// segment i+1's guessed (fresh) start iff everything downstream is provably
// identical. A viewport segment is shown immediately (speculative); the full
// document is verified left-to-right and absorbed. All threading is the APP's;
// this layer is pure and sync, syntect stays sealed.
// ======================================================================

/// A cheap-to-clone, `Send + Sync` handle to a grammar + theme for off-thread
/// bulk tokenization. Nothing syntect crosses its surface; [`SegmentBoundary`]
/// and [`SegmentTokens`] are opaque. Obtain via [`HighlightCache::engine`] (to
/// share the cache's exact grammar/theme) or [`HighlightEngine::new`], and move
/// clones to worker threads.
#[derive(Clone)]
pub struct HighlightEngine {
    syntax: Arc<SyntaxDef>,
    theme: Arc<TokenTheme>,
}

impl HighlightEngine {
    /// An engine over an injected grammar + theme (mirrors [`Highlighter::new`]).
    #[must_use]
    pub fn new(syntax: SyntaxDef, theme: TokenTheme) -> Self {
        Self { syntax: Arc::new(syntax), theme: Arc::new(theme) }
    }

    /// The document-top fresh start state, as a boundary. The coordinator
    /// compares a segment's guessed `Fresh` start against a prior segment's
    /// true end with this: equal => the guess was right, the segment verifies
    /// as-is; unequal => re-run from the true end.
    #[must_use]
    pub fn fresh_boundary(&self) -> SegmentBoundary {
        SegmentBoundary(Box::new(self.fresh_state()))
    }

    fn fresh_state(&self) -> LineState {
        let syntect = SyntectHighlighter::new(&self.theme.0);
        LineState {
            parse: ParseState::new(self.syntax.reference()),
            highlight: HighlightState::new(&syntect, ScopeStack::new()),
        }
    }
}

/// The per-line carry state at a segment edge - opaque, comparable, sendable.
/// Two edges compare equal iff everything downstream is provably identical
/// (the convergence theorem: same state + same text => same tokens).
#[derive(Clone, PartialEq)]
pub struct SegmentBoundary(Box<LineState>);

/// Where a segment's tokenization begins.
pub enum SegmentStart {
    /// From the document-top fresh state - a speculative guess, or genuinely
    /// row 0.
    Fresh,
    /// From a known boundary (a prior segment's verified end state).
    After(SegmentBoundary),
}

/// One segment tokenized off-thread - opaque; ingested by
/// [`crate::Document::absorb_highlight`]. Carries stride checkpoints, the end
/// boundary, and (for a requested sub-range) dense window spans+states.
pub struct SegmentTokens {
    rows: Range<u32>,
    started_fresh: bool,
    /// Stride-aligned end-state checkpoints within `rows` (rows where
    /// `(row + 1) % HIGHLIGHT_CHECKPOINT_STRIDE == 0`) - the SAME rows the
    /// sync walk would pick, so [`HighlightCache::absorb`] merges them cleanly.
    checkpoints: Vec<(u32, Box<LineState>)>,
    /// End state after `rows.end - 1` (equals the start state if `rows` empty).
    end: SegmentBoundary,
    /// The captured window sub-range (subset of `rows`) and its dense data.
    win: Range<u32>,
    win_spans: Vec<Vec<HighlightSpan>>,
    win_states: Vec<LineState>,
    /// How many rows were actually tokenized (a converging re-run splices the
    /// rest from the old segment): `rows.len()` on a full run, less on an
    /// early-stop repair. The op-count the repair-bound canary asserts on, and
    /// a progress signal for the app.
    tokenized: u32,
}

impl SegmentTokens {
    /// The rows this segment covers.
    #[must_use]
    pub fn rows(&self) -> Range<u32> {
        self.rows.clone()
    }

    /// How many rows this segment actually tokenized (`< rows().len()` when a
    /// converging re-run spliced a proven-identical tail from the old segment).
    #[must_use]
    pub fn tokenized_rows(&self) -> u32 {
        self.tokenized
    }

    /// The end-of-segment boundary (the true state after `rows.end - 1` once
    /// this segment is verified) - what the coordinator chains the next
    /// segment's verification against.
    #[must_use]
    pub fn end_boundary(&self) -> &SegmentBoundary {
        &self.end
    }

    /// Whether this segment started from the fresh (guessed) state.
    #[must_use]
    pub fn started_fresh(&self) -> bool {
        self.started_fresh
    }

    fn checkpoint_at(&self, row: u32) -> Option<&LineState> {
        self.checkpoints.binary_search_by_key(&row, |(r, _)| *r).ok().map(|i| &*self.checkpoints[i].1)
    }
}

/// Tokenize `rows` of `snapshot` off-thread - pure, sync, unbudgeted (the app
/// runs it on a worker over an O(1) [`Snapshot`] clone). Records stride
/// checkpoints and, for `spans_for` intersected with `rows`, dense
/// spans+states.
///
/// `converge_against`: when RE-running a segment from a corrected start (a
/// speculative guess proved wrong), pass the old result. At each stride
/// checkpoint the new end state is compared with the old's; on the first match
/// the tail is spliced verbatim from the old (provably identical downstream) -
/// so repair costs O(distance to the construct's close), not O(segment). Pass
/// it on a re-run whose `rows` match the old segment's; **the window is forced
/// to the old segment's** (the passed `spans_for` is ignored when
/// `converge_against` is `Some`), so the caller need not track which window
/// the old segment was tokenized with — the splice stays internally consistent
/// even if the viewport moved since.
#[must_use]
pub fn tokenize_segment(
    engine: &HighlightEngine,
    snapshot: &Snapshot,
    rows: Range<u32>,
    start: SegmentStart,
    spans_for: Option<Range<u32>>,
    converge_against: Option<&SegmentTokens>,
) -> SegmentTokens {
    let syntect = SyntectHighlighter::new(&engine.theme.0);
    let set = &engine.syntax.set;
    let started_fresh = matches!(start, SegmentStart::Fresh);
    let mut state = match start {
        SegmentStart::Fresh => engine.fresh_state(),
        SegmentStart::After(b) => *b.0,
    };

    let n = snapshot.line_count();
    let lo = rows.start.min(n);
    let hi = rows.end.min(n);
    // A converging re-run splices the old segment's window tail, so its window
    // MUST equal the old segment's — enforce that here regardless of the
    // caller's `spans_for`. The coordinator can dispatch a re-run with a moved
    // viewport's `spans_for` while `converge_against` still carries the original
    // window; honoring the moved one would splice mismatched rows. When
    // re-running, the window IS the old segment's window.
    let win = match converge_against {
        Some(old) => old.win.clone(),
        None => spans_for
            .map(|w| w.start.max(lo)..w.end.min(hi))
            .filter(|w| w.start < w.end)
            .unwrap_or(lo..lo),
    };

    let mut checkpoints: Vec<(u32, Box<LineState>)> = Vec::new();
    let mut win_spans: Vec<Vec<HighlightSpan>> = Vec::with_capacity(win.len());
    let mut win_states: Vec<LineState> = Vec::with_capacity(win.len());

    let mut row = lo;
    while row < hi {
        let line = snapshot.line(row);
        let (spans, end) = tokenize_line(&syntect, set, &state, line.as_ref());
        if win.contains(&row) {
            win_spans.push(spans);
            win_states.push(end.clone());
        }
        state = end;
        if (row + 1).is_multiple_of(HIGHLIGHT_CHECKPOINT_STRIDE) {
            checkpoints.push((row, Box::new(state.clone())));
            // Early-stop against the old segment on a corrected re-run: the
            // new state after `row` matching the old's checkpoint there means
            // every later row is provably identical - splice the old tail.
            if let Some(old) = converge_against {
                if old.checkpoint_at(row).is_some_and(|s| *s == state) {
                    debug_assert_eq!(win, old.win, "the window was forced to old.win above");
                    for (r, s) in &old.checkpoints {
                        if *r > row {
                            checkpoints.push((*r, s.clone()));
                        }
                    }
                    for (k, wr) in old.win.clone().enumerate() {
                        if wr > row {
                            win_spans.push(old.win_spans[k].clone());
                            win_states.push(old.win_states[k].clone());
                        }
                    }
                    return SegmentTokens {
                        rows,
                        started_fresh,
                        checkpoints,
                        end: old.end.clone(),
                        win,
                        win_spans,
                        win_states,
                        tokenized: row - lo + 1, // rows lo..=row were run
                    };
                }
            }
        }
        row += 1;
    }

    SegmentTokens {
        rows,
        started_fresh,
        checkpoints,
        end: SegmentBoundary(Box::new(state)),
        win,
        win_spans,
        win_states,
        tokenized: hi - lo,
    }
}

impl Retention {
    /// The window slot for `row`, if it is inside the retention window (and
    /// the document).
    fn win_idx(&self, row: u32) -> Option<usize> {
        (self.win.contains(&row) && row < self.n_lines).then(|| (row - self.win.start) as usize)
    }

    /// Move the dense retention window through the per-edit line splices,
    /// keeping it aimed at the same buffer rows: a row above an edit shifts by
    /// that edit's line delta, a row inside an edit loses its now-stale span but
    /// keeps a convergence probe (the edit's old end state) so a state-neutral
    /// keystroke still converges in O(1), and rows below are untouched. Because
    /// it rides the per-edit spans (not one covering range), a SCATTERED
    /// multi-caret edit whose first-to-last span engulfs the window neither
    /// drains nor repositions it — the visible rows stay in the window and
    /// recolor on the next tokenize, rather than going stale until a scroll
    /// fires `set_window`. Bounded to [`HIGHLIGHT_MAX_WINDOW_ROWS`];
    /// O(window + edits).
    fn shift_window(&mut self, spans: &[(u32, u32, u32)]) {
        if spans.is_empty() {
            return;
        }
        let b = HIGHLIGHT_MAX_WINDOW_ROWS as usize;
        let mut pref: Vec<i64> = Vec::with_capacity(spans.len() + 1);
        pref.push(0);
        for &(_, o, n) in spans {
            pref.push(pref[pref.len() - 1] + i64::from(n) - i64::from(o));
        }
        // Displacement of a pre-edit row NOT inside any edit = delta of edits
        // entirely above it.
        let shift_at = |pre: u32| -> i64 { pref[spans.partition_point(|s| s.0 + s.1 <= pre)] };
        let edited = |pre: u32| -> bool {
            let i = spans.partition_point(|s| s.0 + s.1 <= pre);
            i < spans.len() && pre >= spans[i].0
        };
        // Convergence probes: each edit's OLD end state (of its last old row),
        // read BEFORE the window moves, keyed by the post row it lands on.
        let mut probes: Vec<(u32, Box<LineState>)> = Vec::new();
        for (j, &(ps, o, n)) in spans.iter().enumerate() {
            if n == 0 {
                continue;
            }
            if let Some(st) = self.stored_state_after(ps + o - 1) {
                let post = (i64::from(ps) + pref[j]) as u32 + n - 1;
                probes.push((post, Box::new(st.clone())));
            }
        }
        let old_start = self.win.start;
        let new_start = (i64::from(old_start) + shift_at(old_start)) as u32;
        let old_spans = std::mem::take(&mut self.win_spans);
        let old_states = std::mem::take(&mut self.win_states);
        // Survivors: window rows not inside an edit, at their post positions —
        // `(post row, spans, end state)`.
        type Survivor = (u32, Option<Vec<HighlightSpan>>, Option<Box<LineState>>);
        let mut max_post = new_start;
        let mut survivors: Vec<Survivor> = Vec::with_capacity(old_spans.len());
        for (idx, (sp, st)) in old_spans.into_iter().zip(old_states).enumerate() {
            let pre = old_start + idx as u32;
            if edited(pre) {
                continue; // edited row → span stale (a probe below drives convergence)
            }
            let post = (i64::from(pre) + shift_at(pre)) as u32;
            max_post = max_post.max(post);
            survivors.push((post, sp, st));
        }
        for (r, _) in &probes {
            max_post = max_post.max(*r);
        }
        let len = ((max_post + 1).saturating_sub(new_start) as usize).min(b);
        let mut spans_v: Vec<Option<Vec<HighlightSpan>>> = vec![None; len];
        let mut states_v: Vec<Option<Box<LineState>>> = vec![None; len];
        for (post, sp, st) in survivors {
            if let Some(i) = post.checked_sub(new_start).map(|d| d as usize) {
                if i < len {
                    spans_v[i] = sp;
                    states_v[i] = st;
                }
            }
        }
        for (post, probe) in probes {
            if let Some(i) = post.checked_sub(new_start).map(|d| d as usize) {
                if i < len && states_v[i].is_none() {
                    states_v[i] = Some(probe);
                }
            }
        }
        self.win = new_start..new_start + len as u32;
        self.win_spans = spans_v;
        self.win_states = states_v;
    }

    /// The stored end state after `row`: its window slot, else an exact
    /// checkpoint. Trustworthy only when `first_dirty()` is above `row` — the
    /// invariant that the first dirty line implies every line above it is
    /// valid.
    fn stored_state_after(&self, row: u32) -> Option<&LineState> {
        if let Some(i) = self.win_idx(row) {
            if let Some(s) = &self.win_states[i] {
                return Some(s);
            }
        }
        self.checkpoints.state_at(row)
    }

    /// The nearest stored state at or before `row` (window slot or
    /// checkpoint), for warm-up.
    fn nearest_state_at_or_before(&self, row: u32) -> Option<(u32, &LineState)> {
        let cp = self.checkpoints.floor(row);
        // A window state can only beat the checkpoint in (cp_row, row] — scan
        // down just that far.
        let mut ws = None;
        if !self.win.is_empty() && self.win.start <= row {
            let hi = row.min(self.win.end - 1);
            let lo = cp.map_or(self.win.start, |(r, _)| (r + 1).max(self.win.start));
            let mut r = hi;
            while r >= lo {
                if let Some(s) = &self.win_states[(r - self.win.start) as usize] {
                    ws = Some((r, &**s));
                    break;
                }
                if r == lo {
                    break;
                }
                r -= 1;
            }
        }
        match (cp, ws) {
            (Some(c), Some(w)) => Some(if w.0 >= c.0 { w } else { c }),
            (c, w) => w.or(c),
        }
    }

    /// The start state for tokenizing `row`.
    fn start_state_for(&self, row: u32) -> StartState {
        if row == 0 {
            return StartState::Fresh;
        }
        if let Some(s) = self.stored_state_after(row - 1) {
            return StartState::Ready(s.clone());
        }
        match self.nearest_state_at_or_before(row - 1) {
            Some((r, s)) => StartState::WarmupStored(r, s.clone()),
            None => StartState::WarmupFresh,
        }
    }

    /// Tokenize the clean rows between `from` (the row whose end state
    /// `state` is; `None` ⇒ fresh before row 0) and `to` (exclusive) to
    /// re-derive `to`'s start state, retaining as it goes. `None` if the
    /// budget dies first (a resume checkpoint is recorded).
    #[allow(clippy::too_many_arguments)]
    fn warm_up<S: AsRef<str>>(
        &mut self,
        from: Option<u32>,
        to: u32,
        mut state: LineState,
        work: &mut u32,
        max_lines: u32,
        syntect: &SyntectHighlighter<'_>,
        set: &SyntaxSet,
        line: &mut impl FnMut(u32) -> S,
    ) -> Option<LineState> {
        let mut r = from;
        loop {
            let next = r.map_or(0, |r| r + 1);
            if next >= to {
                return Some(state);
            }
            if *work >= max_lines {
                if let Some(r) = r {
                    self.keep_resume_checkpoint(Some(r), &state);
                }
                return None;
            }
            let (spans, end) = tokenize_line(syntect, set, &state, line(next).as_ref());
            self.retain(next, spans, &end);
            *work += 1;
            r = Some(next);
            state = end;
        }
    }

    /// Retain one tokenized row: spans + state in the window, a checkpoint at
    /// stride rows — and, crucially, a freshly tokenized row REFRESHES any
    /// checkpoint already sitting at it. Checkpoints drift off the stride grid
    /// (edits shift them; budget stops pin arbitrary rows), and a stale one
    /// left below the frontier would poison later warm-ups into permanently
    /// wrong colors. Stride inserts also prune the now-redundant off-grid
    /// checkpoints in the stride behind them, so drift and old resume pins
    /// cannot accumulate.
    fn retain(&mut self, row: u32, spans: Vec<HighlightSpan>, end: &LineState) {
        if let Some(i) = self.win_idx(row) {
            self.win_spans[i] = Some(spans);
            self.win_states[i] = Some(Box::new(end.clone()));
        }
        if (row + 1).is_multiple_of(HIGHLIGHT_CHECKPOINT_STRIDE) {
            self.checkpoints.set_stride(row, end.clone());
        } else {
            // A drifted (off-grid) checkpoint sitting on this freshly tokenized
            // row is refreshed so a stale state can't poison a later warm-up; a
            // no-op when none is there. Do not prune (it is not stride-aligned).
            self.checkpoints.refresh_if_present(row, end.clone());
        }
    }

    /// A budget stop about to abandon the state after `row`: pin it as a
    /// checkpoint if nothing stores it, so the next call resumes without
    /// re-warming.
    fn keep_resume_checkpoint(&mut self, row: Option<u32>, state: &LineState) {
        let Some(row) = row else { return }; // before row 0 — fresh is free
        if self.stored_state_after(row).is_none() {
            self.checkpoints.upsert(row, state.clone());
        }
    }

    /// The first window row below `before` whose spans await a (re)fill.
    fn first_window_gap(&self, before: u32) -> Option<u32> {
        let end = self.win.end.min(self.n_lines).min(before);
        (self.win.start..end).find(|&r| self.win_spans[(r - self.win.start) as usize].is_none())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A one-rule grammar: the word `kw` is a keyword.
    const GRAMMAR: &str = "%YAML 1.2\n\
        ---\n\
        name: Test\n\
        scope: source.test\n\
        contexts:\n\
        \x20 main:\n\
        \x20   - match: '\\bkw\\b'\n\
        \x20     scope: keyword.control.test\n";

    // A minimal theme: default white, keyword red.
    const THEME: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>name</key><string>Test</string>
<key>settings</key><array>
<dict><key>settings</key><dict><key>background</key><string>#000000</string><key>foreground</key><string>#ffffff</string></dict></dict>
<dict><key>scope</key><string>keyword</string><key>settings</key><dict><key>foreground</key><string>#ff0000</string></dict></dict>
</array></dict></plist>"#;

    fn highlighter() -> Highlighter {
        Highlighter::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses"),
            TokenTheme::from_tm_theme(THEME).expect("theme parses"),
        )
    }

    #[test]
    fn keyword_gets_the_keyword_color() {
        let lines = highlighter().highlight("kw x");
        assert_eq!(lines.len(), 1);
        // The `kw` at bytes 0..2 is red; something else is white.
        let red = Rgba { r: 0xff, g: 0, b: 0, a: 0xff };
        let white = Rgba { r: 0xff, g: 0xff, b: 0xff, a: 0xff };
        let kw = lines[0].iter().find(|s| s.range == (0..2)).expect("a span at 0..2");
        assert_eq!(kw.style.fg, red);
        assert!(lines[0].iter().any(|s| s.style.fg == white), "non-keyword text is default white");
    }

    #[test]
    fn per_line_spans_and_a_trailing_empty_line() {
        let lines = highlighter().highlight("kw\nx\n");
        assert_eq!(lines.len(), 3); // "kw", "x", "" (trailing)
        assert!(lines[0].iter().any(|s| s.range == (0..2))); // "kw" on line 0
        assert!(lines[2].is_empty()); // the empty final line has no spans
    }

    fn cache(n: u32) -> HighlightCache {
        HighlightCache::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
            n,
        )
    }

    fn lines(text: &str) -> Vec<&str> {
        text.split('\n').collect()
    }

    /// After tokenizing, every cached line must equal a fresh whole-document
    /// highlight of the same text — the convergence invariant, whatever the
    /// edit path.
    fn assert_matches_full(cache: &HighlightCache, text: &str) {
        let full = highlighter().highlight(text);
        for (row, expected) in full.iter().enumerate() {
            assert_eq!(cache.line_spans(row as u32), Some(expected.as_slice()), "line {row}");
        }
    }

    #[test]
    fn fresh_cache_matches_whole_document_highlight() {
        let text = "kw a\nb kw\nc";
        let ls = lines(text);
        let mut c = cache(ls.len() as u32);
        c.tokenize_until(ls.len() as u32, u32::MAX, |r| ls[r as usize]);
        assert_matches_full(&c, text);
    }

    #[test]
    fn in_place_edit_reconverges() {
        let before = "kw a\nb b\nkw c";
        let ls0 = lines(before);
        let mut c = cache(ls0.len() as u32);
        c.tokenize_until(ls0.len() as u32, u32::MAX, |r| ls0[r as usize]);
        // Edit line 1 in place ("b b" -> "b kw"); old == new == 1 line.
        let after = "kw a\nb kw\nkw c";
        let ls1 = lines(after);
        c.on_commit(1, 1, 1);
        c.tokenize_until(ls1.len() as u32, u32::MAX, |r| ls1[r as usize]);
        assert_matches_full(&c, after);
    }

    #[test]
    fn insert_line_splices_and_reconverges() {
        let before = "kw\nx";
        let ls0 = lines(before);
        let mut c = cache(ls0.len() as u32);
        c.tokenize_until(ls0.len() as u32, u32::MAX, |r| ls0[r as usize]);
        // Insert a line: line 0 ("kw") becomes two lines ("kw", "y").
        let after = "kw\ny\nx";
        let ls1 = lines(after);
        c.on_commit(0, 1, 2);
        c.tokenize_until(ls1.len() as u32, u32::MAX, |r| ls1[r as usize]);
        assert_matches_full(&c, after);
    }

    #[test]
    fn theme_swap_invalidates_all_keeps_stale_colors_then_recolors() {
        // The same minimal theme, but the keyword is blue instead of red.
        const THEME_BLUE: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>name</key><string>Blue</string>
<key>settings</key><array>
<dict><key>settings</key><dict><key>background</key><string>#000000</string><key>foreground</key><string>#ffffff</string></dict></dict>
<dict><key>scope</key><string>keyword</string><key>settings</key><dict><key>foreground</key><string>#0000ff</string></dict></dict>
</array></dict></plist>"#;
        let red = Rgba { r: 0xff, g: 0, b: 0, a: 0xff };
        let blue = Rgba { r: 0, g: 0, b: 0xff, a: 0xff };
        let has = |c: &HighlightCache, row: u32, col: Rgba| {
            c.line_spans(row).unwrap().iter().any(|s| s.style.fg == col)
        };
        // Two keyword lines, both tokenized (red).
        let ls = lines("kw a\nb kw");
        let mut c = cache(ls.len() as u32);
        c.tokenize_until(ls.len() as u32, u32::MAX, |r| ls[r as usize]);
        assert!(has(&c, 0, red) && has(&c, 1, red));
        // Swap the theme: every line goes dirty, but the OLD (red) spans stay as
        // a stale fallback — no flash to default before the repaint.
        c.set_theme(TokenTheme::from_tm_theme(THEME_BLUE).unwrap());
        assert!(has(&c, 0, red) && has(&c, 1, red), "stale colors until repainted");
        // Repaint the whole cache: the keyword is now blue on every line, no red left.
        c.tokenize_until(ls.len() as u32, u32::MAX, |r| ls[r as usize]);
        assert!(has(&c, 0, blue) && has(&c, 1, blue), "both lines recolored");
        assert!(!has(&c, 0, red) && !has(&c, 1, red), "no stale red survives repaint");
    }

    #[test]
    fn convergence_stops_early_below_an_edit() {
        // A grammar with no multi-line state ⇒ every line's end state is equal,
        // so an in-place edit converges at the edited line and never re-tokenizes
        // the (unchanged) line below it. We prove that by leaving the line below
        // deliberately WRONG in `lines` — if it were re-tokenized it would change.
        let ls0 = lines("kw\nkw\nkw");
        let mut c = cache(3);
        c.tokenize_until(3, u32::MAX, |r| ls0[r as usize]);
        c.on_commit(0, 1, 1); // edit line 0 only
        // Pass a lines slice whose line 2 is a sentinel that must NOT be touched.
        let probe_lines = ["kw", "kw", "SHOULD-NOT-RETOKENIZE"];
        c.tokenize_until(3, u32::MAX, |r| probe_lines[r as usize]);
        // Line 2 kept its original "kw" spans (convergence stopped at line 0).
        assert_eq!(c.line_spans(2), Some(highlighter().highlight("kw").pop().unwrap().as_slice()));
    }

    #[test]
    fn perf_canary_state_neutral_keystroke_is_o1_regardless_of_size() {
        // Op-count canary (not timing): the test grammar has no multi-line
        // state, so editing any line converges at that line — a keystroke
        // re-tokenizes ~1 line no matter how big the document is. The "no
        // O(file) stacking" guarantee: the number is independent of the
        // 2000-line size.
        let big = vec!["kw x"; 2000].join("\n");
        let ls = lines(&big);
        let mut c = cache(ls.len() as u32);
        c.tokenize_until(ls.len() as u32, u32::MAX, |r| ls[r as usize]); // full initial pass
        c.on_commit(1000, 1, 1); // a keystroke deep in the document
        let n = c.tokenize_until(ls.len() as u32, u32::MAX, |r| ls[r as usize]);
        assert!(n <= 2, "state-neutral keystroke re-tokenized {n} lines in a 2000-line doc (expected O(1))");
    }

    #[test]
    fn perf_canary_target_bounds_work_to_the_viewport() {
        // Perf canary 1 (the viewport-targeting tooth): tokenizing to a viewport
        // target does ≤ viewport work, never O(file) — even with every line dirty
        // (a fresh cache, or a top-of-file state cascade). 2000 lines dirty,
        // target row 50 → exactly 51 lines tokenized, nothing below.
        let big = vec!["kw x"; 2000].join("\n");
        let ls = lines(&big);
        let mut c = cache(ls.len() as u32); // all 2000 lines dirty
        let n = c.tokenize_until(50, u32::MAX, |r| ls[r as usize]);
        assert_eq!(n, 51, "target must bound tokenization to the viewport (rows 0..=50)");
        assert!(c.line_spans(50).is_some(), "the viewport is tokenized");
        assert!(c.line_spans(51).is_none(), "nothing past the viewport target is tokenized");
    }

    #[test]
    fn perf_canary_budget_caps_a_call_and_resumes() {
        // The per-call budget: one call does at most `max_lines` of work, the
        // frontier records where it stopped, and the next call resumes there —
        // total work conserved.
        let big = vec!["kw x"; 300].join("\n");
        let ls = lines(&big);
        let mut c = cache(ls.len() as u32);
        let n = c.tokenize_until(u32::MAX, 100, |r| ls[r as usize]);
        assert_eq!(n, 100, "the budget caps the call");
        assert_eq!(c.first_dirty(), Some(100), "the frontier records the resume point");
        let n = c.tokenize_until(u32::MAX, u32::MAX, |r| ls[r as usize]);
        assert_eq!(n, 200, "the next call finishes the remainder");
        assert_eq!(c.first_dirty(), None, "converged");
    }

    #[test]
    fn perf_canary_deep_dirty_tail_commit_is_cheap() {
        // A commit must be O(edit + #runs), independent of how long a dirty
        // tail the document carries — storing runs (not individual rows) is
        // what makes that hold. Generous wall-clock bound (the one in-test
        // timing allowed): 1000 commits against a 1M-line dirty tail must be
        // near-instant; a per-commit rebuild of the whole dirty set would take
        // tens of seconds.
        let mut c = cache(1_000_000); // every line dirty, one run
        let t0 = std::time::Instant::now();
        for _ in 0..1_000 {
            c.on_commit(500_000, 1, 1);
        }
        assert!(
            t0.elapsed() < std::time::Duration::from_secs(1),
            "1000 deep-dirty-tail commits took {:?} — the dirty set is being rebuilt per commit",
            t0.elapsed()
        );
        assert!(c.ret.invalid.runs() <= 3, "the run list stays compact: {}", c.ret.invalid.runs());
    }

    #[test]
    fn dirty_runs_stay_small_under_a_typing_run() {
        let mut c = cache(2_000);
        // Converge everything first.
        let big = vec!["kw x"; 2000].join("\n");
        let ls = lines(&big);
        c.tokenize_until(u32::MAX, u32::MAX, |r| ls[r as usize]);
        // A typing run at one spot, plus a couple of scattered edits.
        for _ in 0..50 {
            c.on_commit(700, 1, 1);
        }
        c.on_commit(100, 1, 2);
        c.on_commit(1500, 2, 1);
        assert!(c.ret.invalid.runs() <= 4, "scattered typing produced {} runs", c.ret.invalid.runs());
    }

    // ------------------------------------------------------------------
    // Highlight virtualization: a STATEFUL grammar — an unterminated `"` flips
    // every following line into the `string` context — so checkpoints,
    // warm-ups, and cross-stride convergence are exercised for real (the `kw`
    // grammar above is stateless: every end state is equal, which can never
    // distinguish a right checkpoint from a wrong one).
    // ------------------------------------------------------------------
    const GRAMMAR_STR: &str = "%YAML 1.2\n\
        ---\n\
        name: TestStr\n\
        scope: source.tstr\n\
        contexts:\n\
        \x20 main:\n\
        \x20   - match: '\"'\n\
        \x20     scope: punctuation.definition.string.begin\n\
        \x20     push: string\n\
        \x20   - match: '\\bkw\\b'\n\
        \x20     scope: keyword.control.test\n\
        \x20 string:\n\
        \x20   - match: '\"'\n\
        \x20     scope: punctuation.definition.string.end\n\
        \x20     pop: true\n\
        \x20   - match: '.'\n\
        \x20     scope: string.quoted.test\n";

    // A grammar with an ASYMMETRIC block comment (`/* … */`): unlike the
    // symmetric `"` toggle, a wrong guess about being inside a comment
    // SELF-HEALS at the next `*/` (which returns to ground unconditionally),
    // so a mis-guessed segment re-CONVERGES — the case the stitch's early-stop
    // splice exists for. (`/*` only opens from `main`; `*/` only closes from
    // `comment`; neither nests.)
    const GRAMMAR_CMT: &str = "%YAML 1.2\n\
        ---\n\
        name: TestCmt\n\
        scope: source.tcmt\n\
        contexts:\n\
        \x20 main:\n\
        \x20   - match: '/\\*'\n\
        \x20     scope: punctuation.definition.comment.begin\n\
        \x20     push: comment\n\
        \x20   - match: '\\bkw\\b'\n\
        \x20     scope: keyword.control.test\n\
        \x20 comment:\n\
        \x20   - match: '\\*/'\n\
        \x20     scope: punctuation.definition.comment.end\n\
        \x20     pop: true\n\
        \x20   - match: '.'\n\
        \x20     scope: comment.block.test\n";

    fn engine_cmt() -> HighlightEngine {
        HighlightEngine::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR_CMT).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
        )
    }

    fn cache_cmt(n: u32) -> HighlightCache {
        HighlightCache::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR_CMT).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
            n,
        )
    }

    fn highlighter_cmt() -> Highlighter {
        Highlighter::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR_CMT).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
        )
    }

    fn cache_str(n: u32) -> HighlightCache {
        HighlightCache::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR_STR).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
            n,
        )
    }

    fn highlighter_str() -> Highlighter {
        Highlighter::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR_STR).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
        )
    }

    /// Drive to full convergence (dirty walk + window refill), budgeted like
    /// the real sweep.
    fn drive_all(c: &mut HighlightCache, lines: &[String]) -> u32 {
        let mut work = 0;
        while c.pending().is_some() {
            work += c
                .tokenize_until(u32::MAX, HIGHLIGHT_MAX_LINES_PER_CALL, |r| lines[r as usize].as_str());
        }
        work
    }

    /// The virtualization guarantee: sweeping a 10k-line document with the
    /// window aimed at the top retains a bounded number of rows, not the whole
    /// document — spans and states outside the window are evicted, so RAM does
    /// not grow with the sweep.
    #[test]
    fn virtualized_sweep_retains_bounded_rows() {
        let lines: Vec<String> = (0..10_000).map(|i| format!("kw line {i}")).collect();
        let mut c = cache(10_000);
        c.set_window(0..64);
        drive_all(&mut c, &lines);
        let window_bound = (64 + 2 * HIGHLIGHT_WINDOW_SLACK) as usize;
        assert!(
            c.retained_span_rows() <= window_bound,
            "span retention must be windowed: {} rows kept (bound {window_bound})",
            c.retained_span_rows()
        );
        let state_bound = window_bound + 3 * (10_000 / HIGHLIGHT_CHECKPOINT_STRIDE as usize);
        assert!(
            c.retained_state_rows() <= state_bound,
            "state retention must be window + sparse checkpoints: {} kept (bound {state_bound})",
            c.retained_state_rows()
        );
        // Retention is bounded but the window is still CORRECT…
        let full = highlighter().highlight(&lines.join("\n"));
        assert_eq!(c.line_spans(10), Some(full[10].as_slice()));
        // …and out-of-window rows honestly report nothing (fallback render).
        assert_eq!(c.line_spans(5_000), None);
    }

    /// A window move over already-swept rows refills from the nearest
    /// checkpoint: correct spans (stateful grammar — a wrong warm-up state
    /// would color the string region wrong), bounded work.
    #[test]
    fn window_move_refills_from_checkpoints() {
        let mut lines: Vec<String> = (0..3_000).map(|i| format!("x {i} kw")).collect();
        lines[1_000] = "\"open".into(); // unterminated: rows 1001.. are in-string
        lines[2_000] = "close\"".into(); // …until the string pops here
        let full = highlighter_str().highlight(&lines.join("\n"));
        let mut c = cache_str(3_000);
        c.set_window(0..40);
        drive_all(&mut c, &lines);
        assert!(c.line_spans(1_600).is_none(), "deep rows evicted after the sweep");

        // Cold jump into the middle of the multi-line string.
        c.set_window(1_500..1_540);
        let fill_work = drive_all(&mut c, &lines);
        for r in 1_500..1_540 {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
        assert_eq!(c.line_spans(10), None, "the old window was evicted");
        // ≤ one stride of warm-up + the retained window (+ a little dust).
        let bound = HIGHLIGHT_CHECKPOINT_STRIDE + 40 + 2 * HIGHLIGHT_WINDOW_SLACK + 8;
        assert!(fill_work <= bound, "cold-jump refill did {fill_work} lines (bound {bound})");
    }

    /// A state-neutral edit OUTSIDE the window re-converges at checkpoint
    /// grain: bounded by strides (+ warm-up), never by the document — and
    /// the far-away window's rows stay correct.
    #[test]
    fn out_of_window_edit_reconverges_within_a_stride() {
        let mut lines: Vec<String> = (0..2_000).map(|i| format!("kw {i}")).collect();
        let mut c = cache_str(2_000);
        c.set_window(1_500..1_540);
        drive_all(&mut c, &lines);
        lines[100] = "kw edited".into(); // state-neutral, far above the window
        c.on_commit(100, 1, 1);
        let work = drive_all(&mut c, &lines);
        let bound = 2 * HIGHLIGHT_CHECKPOINT_STRIDE + 2;
        assert!(work <= bound, "out-of-window keystroke re-tokenized {work} lines (bound {bound})");
        let full = highlighter_str().highlight(&lines.join("\n"));
        assert_eq!(c.line_spans(1_520), Some(full[1_520].as_slice()));
    }

    /// An in-window state-neutral keystroke stays O(1) under the STATEFUL
    /// grammar too — per-line convergence probes live in the window.
    #[test]
    fn in_window_keystroke_stays_o1_with_stateful_grammar() {
        let mut lines: Vec<String> = (0..2_000).map(|i| format!("kw {i}")).collect();
        lines[300] = "\"open".into();
        lines[600] = "shut\"".into();
        let mut c = cache_str(2_000);
        c.set_window(900..940);
        drive_all(&mut c, &lines);
        lines[920] = "kw retyped".into();
        c.on_commit(920, 1, 1);
        let work = drive_all(&mut c, &lines);
        assert!(work <= 2, "in-window keystroke re-tokenized {work} lines (expected O(1))");
    }

    /// The randomized battering ram: seeded edits, window moves, and
    /// budget-starved drives in any order — after convergence, every retained
    /// row equals the whole-document oracle, wherever the window lands.
    #[test]
    fn randomized_edits_and_window_moves_match_the_oracle() {
        let mut lines: Vec<String> = (0..1_200)
            .map(|i| match i % 97 {
                13 => "\"".to_string(),
                51 => "shut\" kw".to_string(),
                _ => format!("kw word {i}"),
            })
            .collect();
        let mut c = cache_str(1_200);
        let mut state = 0xDEADBEEF_u64;
        let mut rand = move |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound.max(1)
        };
        for _step in 0..120 {
            match rand(3) {
                0 => {
                    let s = rand(lines.len());
                    let old = (1 + rand(3)).min(lines.len() - s);
                    let n = 1 + rand(3);
                    let repl: Vec<String> = (0..n)
                        .map(|k| match rand(5) {
                            0 => "\"".to_string(),
                            1 => "q\" kw".to_string(),
                            _ => format!("e{k} kw"),
                        })
                        .collect();
                    lines.splice(s..s + old, repl);
                    c.on_commit(s as u32, old as u32, n as u32);
                }
                1 => {
                    let s = rand(lines.len());
                    c.set_window(s as u32..(s + 40).min(lines.len()) as u32);
                }
                _ => {
                    let target = rand(lines.len()) as u32;
                    let budget = 32 + rand(200) as u32;
                    c.tokenize_until(target, budget, |r| lines[r as usize].as_str());
                }
            }
            assert_eq!(c.line_count() as usize, lines.len(), "line count in step");
        }
        // Converge, then check every retained row — and re-aim the window at
        // several places so refill paths get checked against the oracle too.
        let full = highlighter_str().highlight(&lines.join("\n"));
        drive_all(&mut c, &lines);
        let mut checked = 0usize;
        for r in 0..lines.len() as u32 {
            if let Some(spans) = c.line_spans(r) {
                assert_eq!(spans, full[r as usize].as_slice(), "row {r}");
                checked += 1;
            }
        }
        assert!(checked >= 40, "the window retained rows to check ({checked})");
        for &probe in &[0usize, 400, 800, lines.len().saturating_sub(45)] {
            c.set_window(probe as u32..(probe + 40).min(lines.len()) as u32);
            drive_all(&mut c, &lines);
            for (r, expected) in
                full.iter().enumerate().take((probe + 40).min(lines.len())).skip(probe)
            {
                assert_eq!(c.line_spans(r as u32), Some(expected.as_slice()), "probe {probe} row {r}");
            }
        }
    }

    #[test]
    fn scattered_multi_caret_marks_only_edited_lines_dirty() {
        // A scattered multi-caret keystroke folds into ONE covering line range
        // (first caret .. last caret). The dirty set must hold only the EDITED
        // lines (per-edit spans), not the whole covering span — else the sweep
        // re-tokenizes O(document) between the carets.
        let mut lines: Vec<String> = (0..2_000).map(|i| format!("kw {i}")).collect();
        let mut c = cache_str(2_000);
        c.set_window(0..40);
        drive_all(&mut c, &lines);
        assert_eq!(c.dirty_row_count(), 0, "swept clean");
        // Two carets, rows 10 and 1_500, one char each (state-neutral, 1→1).
        lines[10] = "kw a".into();
        lines[1_500] = "kw b".into();
        c.on_commit_patch(&[(10, 1, 1), (1_500, 1, 1)]);
        assert_eq!(
            c.dirty_row_count(),
            2,
            "a scattered 2-caret edit must mark 2 lines dirty, not the ~1491-row covering span"
        );
        // …and the highlight is still correct after the sweep.
        drive_all(&mut c, &lines);
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in 0..40 {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    #[test]
    fn scattered_edit_keeps_the_window_aimed_at_the_viewport() {
        // Select-all-occurrences of a word + delete is a scattered multi-edit
        // whose covering range engulfs the viewport window. The window must
        // stay aimed at the viewport (only the edited rows invalidated), so the
        // VISIBLE rows recolor on the next tokenize — not only after a scroll
        // fires set_window and re-aims it. A splice that rode the covering range
        // would drain or reposition the window, dropping the visible rows to
        // fallback until a scrollwheel tick.
        let mut lines: Vec<String> = (0..2_000).map(|i| format!("kw word {i}")).collect();
        let mut c = cache_str(2_000);
        c.set_window(940..980);
        drive_all(&mut c, &lines);
        assert!(c.line_spans(960).is_some(), "the viewport window is filled");
        // Delete a word at scattered occurrences: rows 10, 960 (in view), 1_500,
        // one transaction, all state-neutral (1→1). Covering range 10..1_500
        // engulfs the window at 940..980.
        for r in [10usize, 960, 1_500] {
            lines[r] = "kw w".into();
        }
        c.on_commit_patch(&[(10, 1, 1), (960, 1, 1), (1_500, 1, 1)]);
        // Tokenize WITHOUT re-aiming the window (no set_window / no scroll).
        drive_all(&mut c, &lines);
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in 940..980u32 {
            assert_eq!(
                c.line_spans(r),
                Some(full[r as usize].as_slice()),
                "viewport row {r} did not recolor without a scroll to re-aim the window",
            );
        }
    }

    /// The multi-edit analogue of the randomized oracle: each step commits a
    /// TRANSACTION of several disjoint edits at once via `on_commit_patch` (the
    /// scattered multi-caret path), and after convergence every retained row
    /// must still equal the whole-document oracle. Guards the per-edit
    /// checkpoint/dirty merge walks against any shift or coalescing bug.
    #[test]
    fn randomized_multi_edit_commits_match_the_oracle() {
        let mut lines: Vec<String> = (0..1_200)
            .map(|i| match i % 97 {
                13 => "\"".to_string(),
                51 => "shut\" kw".to_string(),
                _ => format!("kw word {i}"),
            })
            .collect();
        let mut c = cache_str(1_200);
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut rand = move |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound.max(1)
        };
        for _step in 0..80 {
            if rand(3) == 2 {
                // Sometimes just tokenize a budget, leaving dirt around.
                let target = rand(lines.len()) as u32;
                c.tokenize_until(target, 32 + rand(200) as u32, |r| lines[r as usize].as_str());
                assert_eq!(c.line_count() as usize, lines.len());
                continue;
            }
            // Build up to 4 disjoint edits, ascending with a gap between each.
            let k = 1 + rand(4);
            let mut sites: Vec<(usize, usize, Vec<String>)> = Vec::new();
            let mut cursor = rand(20);
            for _ in 0..k {
                let s = cursor + rand(30);
                if s >= lines.len() {
                    break;
                }
                let old = (1 + rand(3)).min(lines.len() - s);
                let n = 1 + rand(3);
                let repl: Vec<String> = (0..n)
                    .map(|j| match rand(5) {
                        0 => "\"".to_string(),
                        1 => "q\" kw".to_string(),
                        _ => format!("m{j} kw"),
                    })
                    .collect();
                sites.push((s, old, repl));
                cursor = s + old + 1; // a gap ⇒ the next start is strictly after
            }
            if sites.is_empty() {
                continue;
            }
            // Per-edit spans (pre-edit, disjoint) + the covering splice.
            let spans: Vec<(u32, u32, u32)> =
                sites.iter().map(|(s, o, r)| (*s as u32, *o as u32, r.len() as u32)).collect();
            c.on_commit_patch(&spans);
            // Apply to `lines` back-to-front so earlier indices stay valid.
            for (s, o, repl) in sites.iter().rev() {
                lines.splice(*s..*s + *o, repl.clone());
            }
            assert_eq!(c.line_count() as usize, lines.len(), "line count after multi-edit");
        }
        let full = highlighter_str().highlight(&lines.join("\n"));
        drive_all(&mut c, &lines);
        let mut checked = 0usize;
        for r in 0..lines.len() as u32 {
            if let Some(spans) = c.line_spans(r) {
                assert_eq!(spans, full[r as usize].as_slice(), "row {r}");
                checked += 1;
            }
        }
        assert!(checked >= 40, "retained rows to check ({checked})");
        for &probe in &[0usize, 300, 700, lines.len().saturating_sub(45)] {
            c.set_window(probe as u32..(probe + 40).min(lines.len()) as u32);
            drive_all(&mut c, &lines);
            for (r, expected) in
                full.iter().enumerate().take((probe + 40).min(lines.len())).skip(probe)
            {
                assert_eq!(c.line_spans(r as u32), Some(expected.as_slice()), "probe {probe} row {r}");
            }
        }
    }

    /// Checkpoints drift off the stride grid (any line-delta edit shifts them),
    /// and a state-changing cascade crossing an off-stride checkpoint must
    /// REFRESH it — a stale one left below the frontier poisons a later
    /// cold-jump warm-up into permanently wrong highlighting. Fails against a
    /// `retain` that only writes stride-aligned checkpoints.
    #[test]
    fn shifted_checkpoints_are_refreshed_by_a_crossing_cascade() {
        let mut lines: Vec<String> = (0..2_000).map(|i| format!("kw {i}")).collect();
        let mut c = cache_str(2_000);
        c.set_window(0..40);
        drive_all(&mut c, &lines);
        // A single Enter above everything: every checkpoint shifts off-stride.
        lines.insert(10, "kw split".to_string());
        c.on_commit(10, 1, 2);
        drive_all(&mut c, &lines);
        // A state-changing edit near the top: the cascade repaints the tail,
        // crossing every shifted checkpoint.
        lines[12] = "\"open".to_string(); // unterminated → everything below is in-string
        c.on_commit(12, 1, 1);
        drive_all(&mut c, &lines);
        // Cold-jump far below: the refill warms up from whatever checkpoint
        // is nearest — which must hold the POST-cascade (in-string) state.
        c.set_window(1_285..1_325);
        drive_all(&mut c, &lines);
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in 1_285..1_325u32 {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    /// Off-stride and orphaned resume checkpoints must be pruned as fresh
    /// stride checkpoints land, or they accumulate without bound across
    /// edit/cascade cycles — slowly re-growing the RAM virtualization keeps
    /// bounded.
    #[test]
    fn checkpoints_do_not_accumulate_across_edit_cycles() {
        let mut lines: Vec<String> = (0..4_000).map(|i| format!("kw {i}")).collect();
        let mut c = cache_str(4_000);
        c.set_window(0..40);
        drive_all(&mut c, &lines);
        for cycle in 0..16 {
            // Shift the grid…
            lines.insert(5, format!("kw cycle {cycle}"));
            c.on_commit(5, 1, 2);
            // …and flip the whole tail's state, twice (open then close).
            lines[7] = "\"open".to_string();
            c.on_commit(7, 1, 1);
            drive_all(&mut c, &lines);
            lines[7] = "kw closed".to_string();
            c.on_commit(7, 1, 1);
            drive_all(&mut c, &lines);
        }
        let bound = 2 * (c.line_count() / HIGHLIGHT_CHECKPOINT_STRIDE) as usize + 8;
        assert!(
            c.ret.checkpoints.len() <= bound,
            "checkpoints accumulate: {} after 16 cycles (bound {bound})",
            c.ret.checkpoints.len()
        );
    }

    // ── The checkpoint delta-gap `SumTree` ──────────────────────────────────

    /// A fresh (document-top) `LineState` for seeding checkpoint trees without a
    /// real sweep — the `main`-context start state.
    fn fresh_state() -> LineState {
        engine_str().fresh_state()
    }

    /// A distinct `LineState`: the parser left INSIDE a string context (after an
    /// unclosed `"`), so it compares unequal to [`fresh_state`] — lets the oracle
    /// pin the `(row, state)` sequence, not just the rows.
    fn str_state() -> LineState {
        let eng = engine_str();
        let syntect = SyntectHighlighter::new(&eng.theme.0);
        let (_spans, end) = tokenize_line(&syntect, &eng.syntax.set, &eng.fresh_state(), "\"");
        end
    }

    /// A straightforward flat-Vec checkpoint shift — the ORACLE reference the
    /// delta-gap tree's `shift` must match, edit for edit. Drift in a
    /// checkpoint row poisons later warm-ups (a wrong start state → permanently
    /// wrong colors), so this equivalence is load-bearing.
    fn vec_shift_reference(cps: &mut Vec<(u32, Box<LineState>)>, spans: &[(u32, u32, u32)]) {
        if spans.is_empty() {
            return;
        }
        let old = std::mem::take(cps);
        let mut si = 0;
        let mut acc = 0i64;
        for (row, state) in old {
            while si < spans.len() && spans[si].0 + spans[si].1 <= row {
                acc += i64::from(spans[si].2) - i64::from(spans[si].1);
                si += 1;
            }
            if si < spans.len() && row >= spans[si].0 {
                continue; // inside a replaced span → stale
            }
            cps.push(((row as i64 + acc) as u32, state));
        }
    }

    /// ORACLE: the checkpoint tree's `shift` equals the flat-Vec reference under
    /// random single- AND multi-edit patches, after EVERY commit. Single-edit
    /// patches drive `shift_single` (the windowed seam surgery — the real risk);
    /// multi-edit patches drive `shift_walk`. States alternate between two
    /// distinct values so a mismapped payload (not just a wrong row) is caught.
    #[test]
    fn checkpoint_tree_equals_the_vec_reference_under_random_edits() {
        let (st_a, st_b) = (fresh_state(), str_state());
        assert!(st_a != st_b, "seed states must differ for the state check to bite");
        let mut rng = 0x1234_5678u32;
        let mut next = || {
            rng ^= rng << 13;
            rng ^= rng >> 17;
            rng ^= rng << 5;
            rng
        };
        for trial in 0..200 {
            // A fresh random ascending, distinct checkpoint set per trial.
            let mut n_lines = 400u32 + next() % 4_000;
            let k = 4 + next() % 60;
            let mut rows: Vec<u32> = Vec::new();
            let mut r = next() % 40;
            for _ in 0..k {
                if r >= n_lines {
                    break;
                }
                rows.push(r);
                r += 1 + next() % 90;
            }
            let seed = |i: usize| if i.is_multiple_of(2) { st_a.clone() } else { st_b.clone() };
            let init: Vec<(u32, Box<LineState>)> =
                rows.iter().enumerate().map(|(i, &r)| (r, Box::new(seed(i)))).collect();
            let mut tree = {
                let items = init.iter().scan(0u32, |prev, (r, s)| {
                    let gap = r - *prev;
                    *prev = *r;
                    Some(CkptItem { row_gap: gap, state: s.clone() })
                });
                Checkpoints { tree: SumTree::from_items(items) }
            };
            let mut reference = init;

            for step in 0..12 {
                // A valid ascending, disjoint patch of 1..=3 edits (mixes the paths).
                let n_edits = 1 + (next() % 3) as usize;
                let mut spans: Vec<(u32, u32, u32)> = Vec::new();
                let mut cursor = 0u32;
                for _ in 0..n_edits {
                    if cursor >= n_lines {
                        break;
                    }
                    let span = (n_lines - cursor).clamp(1, 120);
                    let s = cursor + next() % span;
                    if s >= n_lines {
                        break;
                    }
                    let o = (next() % 5).min(n_lines - s);
                    let nw = next() % 5;
                    spans.push((s, o, nw));
                    cursor = s + o + 1; // keep replaced spans disjoint + ascending
                }
                if spans.is_empty() {
                    continue;
                }
                let delta: i64 = spans.iter().map(|&(_, o, n)| i64::from(n) - i64::from(o)).sum();

                tree.shift(&spans);
                vec_shift_reference(&mut reference, &spans);
                n_lines = (n_lines as i64 + delta) as u32;

                let got = tree.rows();
                let want: Vec<u32> = reference.iter().map(|(r, _)| *r).collect();
                assert_eq!(got, want, "trial {trial} step {step}: rows diverge for {spans:?}");
                let states_ok = tree
                    .tree
                    .items()
                    .iter()
                    .zip(&reference)
                    .all(|(it, (_, s))| *it.state == **s);
                assert!(states_ok, "trial {trial} step {step}: a checkpoint state was mismapped");
            }
        }
    }

    /// PERF CELL: one 1-line keystroke's checkpoint shift is independent of the
    /// checkpoint count. A flat-Vec walk would charge K (take + rebuild every
    /// commit and undo step); the windowed seam touches O(1). FLIP-VERIFY:
    /// force `shift` down the `shift_walk` fallback (change its dispatch to
    /// `if false`) and this trips — the meter reads ≈K → 2K instead of a flat 1.
    #[test]
    fn checkpoint_shift_is_checkpoint_count_independent() {
        let st = fresh_state();
        let meter = |k: u32| -> u64 {
            // K stride checkpoints; the edit (row 5, 1→1) sits far below the first
            // (row 255), so the windowed shift drops nothing and shifts nothing.
            let rows: Vec<u32> =
                (0..k).map(|i| (i + 1) * HIGHLIGHT_CHECKPOINT_STRIDE - 1).collect();
            let mut cps = Checkpoints::from_rows(&rows, &st);
            crate::perf::reset();
            cps.shift(&[(5, 1, 1)]);
            crate::perf::meter()
        };
        let (small, big) = (meter(1000), meter(2000));
        eprintln!("[perf_gate] checkpoint shift charge       {small:>7} -> {big:>7}");
        assert!(
            big <= small + small / 4 + 256,
            "checkpoint shift charged {small} -> {big}: it walked every checkpoint, not the seam",
        );
    }

    /// A collapsed mega-fold makes the widget report a visible range spanning
    /// the fold's hidden interior — the retention window must CAP its length,
    /// or a fold re-grows retention to O(document).
    #[test]
    fn window_length_is_capped() {
        let lines: Vec<String> = (0..200_000).map(|i| format!("kw {i}")).collect();
        let mut c = cache(200_000);
        c.set_window(0..150_000); // a viewport straddling a huge collapsed fold
        drive_all(&mut c, &lines);
        assert!(
            c.retained_span_rows() <= HIGHLIGHT_MAX_WINDOW_ROWS as usize,
            "the window must be capped: {} rows retained",
            c.retained_span_rows()
        );
    }

    #[test]
    fn in_window_commit_with_a_document_scale_span_stays_bounded() {
        // A scattered multi-caret transaction folds into ONE covering line span
        // (first caret .. last caret), so on_commit's in-window branch sees a
        // document-scale `new` even though the window is small. The dense splice
        // must stay bounded — splicing `vec![None; new]` would regrow `win` to
        // O(new), an O(document) per-keystroke allocation.
        let mut c = cache(200_000);
        c.set_window(100..140); // small window near the top; the first caret is in it
        assert!(c.window_len() <= HIGHLIGHT_MAX_WINDOW_ROWS as usize);
        // The covering splice of an edit spanning rows 100..150_000 (top+bottom
        // carets, one char each → +150k covering lines, say).
        c.on_commit(100, 1, 150_000);
        assert!(
            c.window_len() <= HIGHLIGHT_MAX_WINDOW_ROWS as usize,
            "in-window covering splice regrew the window to {} rows",
            c.window_len()
        );
        // The dirty set and line count still track the (over-wide) splice, so
        // the covered rows are refilled correctly on the sweep.
        assert_eq!(c.line_count(), 200_000 + 150_000 - 1);
    }

    // ------------------------------------------------------------------
    // Off-thread bulk tokenization: parallel sweep + viewport speculation.
    // All under the STATEFUL GRAMMAR_STR so a wrong guess (a cut inside a
    // multi-line string) produces genuinely different colors that the stitch
    // must repair - the whole point.
    // ------------------------------------------------------------------

    fn engine_str() -> HighlightEngine {
        HighlightEngine::new(
            SyntaxDef::from_sublime_syntax(GRAMMAR_STR).unwrap(),
            TokenTheme::from_tm_theme(THEME).unwrap(),
        )
    }

    fn snap(lines: &[String]) -> Snapshot {
        crate::buffer::Buffer::new(&lines.join("\n")).unwrap().snapshot()
    }

    fn drive_all_snap(c: &mut HighlightCache, s: &Snapshot) {
        while c.pending().is_some() {
            c.tokenize_until(u32::MAX, HIGHLIGHT_MAX_LINES_PER_CALL, |r| s.line(r));
        }
    }

    /// Simulate the app's coordinator: tokenize every segment Fresh (as the
    /// worker pool would, order-independently), then stitch left to right -
    /// re-running (with the early-stop `converge_against`) any segment whose
    /// Fresh guess mismatched the true prior end. Returns the verified chain
    /// and how many segments needed a re-run.
    fn parallel_verify(
        engine: &HighlightEngine,
        snapshot: &Snapshot,
        cuts: &[u32],
        spans_for: Option<Range<u32>>,
    ) -> (Vec<SegmentTokens>, usize) {
        let n = snapshot.line_count();
        let mut bounds = vec![0u32];
        bounds.extend(cuts.iter().copied().filter(|&c| c > 0 && c < n));
        bounds.sort_unstable();
        bounds.dedup();
        bounds.push(n);
        // Pass 1 - all Fresh (the parallel workers; order irrelevant).
        let segs: Vec<SegmentTokens> = bounds
            .windows(2)
            .map(|w| {
                tokenize_segment(engine, snapshot, w[0]..w[1], SegmentStart::Fresh, spans_for.clone(), None)
            })
            .collect();
        // Pass 2 - sequential stitch.
        let fresh = engine.fresh_boundary();
        let mut verified: Vec<SegmentTokens> = Vec::with_capacity(segs.len());
        let mut prev_end: Option<SegmentBoundary> = None;
        let mut reruns = 0;
        for (idx, seg) in segs.into_iter().enumerate() {
            let corrected = if idx == 0 {
                seg // Fresh at row 0 is the true start
            } else {
                let true_start = prev_end.clone().unwrap();
                if true_start == fresh {
                    seg // the Fresh guess happened to be right
                } else {
                    reruns += 1;
                    let rows = seg.rows();
                    tokenize_segment(
                        engine,
                        snapshot,
                        rows,
                        SegmentStart::After(true_start),
                        spans_for.clone(),
                        Some(&seg),
                    )
                }
            };
            prev_end = Some(corrected.end_boundary().clone());
            verified.push(corrected);
        }
        (verified, reruns)
    }

    /// THE correctness spine (property test): under random stateful documents
    /// and random segment cuts - including cuts deliberately inside multi-line
    /// strings - the parallel-verified + absorbed cache must equal a
    /// whole-document `Highlighter::highlight` on every row, and its checkpoint
    /// rows must match a sync-swept cache. Fails against any stitch that does
    /// not repair a wrong-guessed boundary.
    #[test]
    fn parallel_stitch_equals_the_whole_document_oracle() {
        let engine = engine_str();
        let mut state = 0x1234_5678_9ABC_DEF0_u64;
        let mut rng = move |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound.max(1)
        };
        for round in 0..24 {
            let n = 200 + rng(1_200);
            let lines: Vec<String> = (0..n)
                .map(|i| match rng(11) {
                    0 => "\"".to_string(),        // open a multi-line string
                    1 => "text\" kw".to_string(), // close it, then a keyword
                    2 => "\" and \" kw".to_string(),
                    _ => format!("kw word {i}"),
                })
                .collect();
            let s = snap(&lines);
            let ln = s.line_count();
            // 1-6 random cuts, some likely inside string regions.
            let cuts: Vec<u32> = (0..1 + rng(6)).map(|_| rng(ln as usize) as u32).collect();
            let win_start = rng(ln as usize) as u32;
            let win = win_start..(win_start + 40).min(ln);

            let (verified, _reruns) = parallel_verify(&engine, &s, &cuts, None);
            let mut c = cache_str(ln);
            c.set_window(win.clone());
            for seg in verified {
                c.absorb(seg, true);
            }
            assert_eq!(c.first_dirty(), None, "round {round}: verified sweep must clear all dirt");
            // Spans-less segments left window gaps - the sync refill (from the
            // absorbed checkpoints) fills them; then every window row is exact.
            drive_all_snap(&mut c, &s);
            let full = highlighter_str().highlight(&lines.join("\n"));
            for r in win.clone() {
                assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "round {round} row {r}");
            }
            // Checkpoint ROWS must match a purely sync sweep (the absorb path
            // may not fabricate off-grid checkpoints the sync path lacks).
            let mut sync = cache_str(ln);
            sync.set_window(win.clone());
            drive_all(&mut sync, &lines);
            let got: Vec<u32> = c.ret.checkpoints.rows();
            let want: Vec<u32> = sync.ret.checkpoints.rows();
            assert_eq!(got, want, "round {round}: checkpoint rows diverge from a sync sweep");
        }
    }

    /// A wrong-guessed boundary (a cut deep inside a block comment) repairs in
    /// O(distance to the comment's close), NOT O(segment): the re-run's
    /// early-stop splices the proven-identical tail, tokenizing only up to one
    /// stride past the closing line. (Uses the self-healing GRAMMAR_CMT — a
    /// symmetric-toggle string never re-converges, so it can't exercise the
    /// splice; that is the degenerate case tested separately.)
    #[test]
    fn wrong_guess_repair_stops_within_a_stride() {
        let engine = engine_cmt();
        // A comment opens at row 100 and closes at row 300; the rest is plain.
        // A cut at 1500 is far PAST the comment, so both the fresh guess and
        // the truth are in the ground state there — to force a genuine
        // mis-guess, cut at 200 (inside the comment).
        let mut lines: Vec<String> = (0..8_000).map(|i| format!("kw word {i}")).collect();
        lines[100] = "/*".to_string();
        lines[300] = "*/ kw".to_string();
        let s = snap(&lines);
        // The tail segment 200..8000, guessed Fresh (ground) — wrong, row 200
        // is inside the comment.
        let seg_rows = 200u32..8_000;
        let guessed = tokenize_segment(&engine, &s, seg_rows.clone(), SegmentStart::Fresh, None, None);
        let truth_prefix = tokenize_segment(&engine, &s, 0..200, SegmentStart::Fresh, None, None);
        let true_start = SegmentStart::After(truth_prefix.end_boundary().clone());
        let repaired = tokenize_segment(&engine, &s, seg_rows, true_start, None, Some(&guessed));
        // Converged: the tail was spliced, so far fewer than 7800 rows ran.
        // The comment closes at 300; convergence is detected at the first
        // stride checkpoint past it, so tokenized ≤ (300 − 200) + one stride.
        let bound = (300 - 200) + HIGHLIGHT_CHECKPOINT_STRIDE;
        assert!(
            repaired.tokenized_rows() <= bound,
            "repair tokenized {} rows (bound {bound}) — early-stop did not fire",
            repaired.tokenized_rows()
        );
        assert!(repaired.tokenized_rows() < 7_800, "a full re-run would tokenize the whole segment");
        // …and it is byte-identical to a full correct re-run.
        let full_rerun =
            tokenize_segment(&engine, &s, 200..8_000, SegmentStart::After(truth_prefix.end_boundary().clone()), None, None);
        assert!(repaired.end_boundary() == full_rerun.end_boundary(), "spliced tail must be correct");
        // End-to-end oracle around the close, where the guess was wrong.
        let (verified, reruns) = parallel_verify(&engine, &s, &[200], None);
        assert_eq!(reruns, 1, "the in-comment cut must trigger exactly one re-run");
        let mut c = cache_cmt(8_000);
        c.set_window(190..600);
        for v in verified {
            c.absorb(v, true);
        }
        drive_all_snap(&mut c, &s);
        let full = highlighter_cmt().highlight(&lines.join("\n"));
        for r in 190..600u32 {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    /// A converging re-run whose `spans_for` DIFFERS from the old segment's
    /// window (the coordinator dispatched the re-run after the viewport moved)
    /// must stay internally consistent — the window is forced to the old
    /// segment's, so the spliced tail lines up. Were the moved `spans_for`
    /// honored instead, the splice would write misaligned spans as clean
    /// (viewport poison) or trip the window-equality assertion.
    #[test]
    fn converging_rerun_ignores_a_moved_spans_for() {
        let engine = engine_cmt();
        let mut lines: Vec<String> = (0..4_000).map(|i| format!("kw word {i}")).collect();
        lines[100] = "/*".to_string();
        lines[300] = "*/ kw".to_string();
        let s = snap(&lines);
        let seg_rows = 200u32..2_000;
        let old_window = Some(210u32..250); // the window when the guess was made
        let guessed =
            tokenize_segment(&engine, &s, seg_rows.clone(), SegmentStart::Fresh, old_window.clone(), None);
        let prefix = tokenize_segment(&engine, &s, 0..200, SegmentStart::Fresh, None, None);
        let after = || SegmentStart::After(prefix.end_boundary().clone());
        // Re-run with a MOVED window (1500..1540 — the viewport scrolled).
        let moved = tokenize_segment(&engine, &s, seg_rows.clone(), after(), Some(1_500..1_540), Some(&guessed));
        // …and a control re-run whose spans_for already matches the old window.
        let matched = tokenize_segment(&engine, &s, seg_rows, after(), old_window, Some(&guessed));
        // The moved re-run behaves as if it reused the old window: same tail,
        // same window range, same end — no panic, no misalignment.
        assert!(moved.end_boundary() == matched.end_boundary(), "same converged end");
        assert_eq!(moved.rows, matched.rows);
        assert_eq!(moved.win, matched.win, "window forced to the old segment's");
        assert_eq!(moved.win_spans.len(), moved.win.len(), "spans line up with the window");
        assert_eq!(moved.win_states.len(), moved.win.len());
        // And absorbing it colours the old window's rows correctly.
        let full = highlighter_cmt().highlight(&lines.join("\n"));
        let mut c = cache_cmt(4_000);
        c.set_window(210..250);
        c.absorb(prefix, true);
        c.absorb(moved, true);
        drive_all_snap(&mut c, &s);
        for r in 210..250u32 {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    /// Degenerate honesty: a document that is one never-closed multi-line
    /// string makes every boundary mismatch (full sequential re-run) — the
    /// result is still byte-identical to the oracle.
    #[test]
    fn one_giant_string_falls_back_to_sequential_but_stays_correct() {
        let engine = engine_str();
        let mut lines: Vec<String> = (0..2_000).map(|i| format!("body {i}")).collect();
        lines[0] = "\"".to_string(); // opened, never closed
        let s = snap(&lines);
        let (verified, reruns) = parallel_verify(&engine, &s, &[400, 800, 1_200, 1_600], None);
        assert_eq!(reruns, 4, "every mid-string boundary mismatches");
        let mut c = cache_str(2_000);
        c.set_window(900..980);
        for v in verified {
            c.absorb(v, true);
        }
        drive_all_snap(&mut c, &s);
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in 900..980u32 {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    /// Verified absorb clears the segment's rows from the frontier and the
    /// spans it carries (spans_for) render immediately - no sync walk needed.
    #[test]
    fn verified_absorb_clears_dirt_and_shows_spans() {
        let engine = engine_str();
        let lines: Vec<String> = (0..2_000).map(|i| format!("kw word {i}")).collect();
        let s = snap(&lines);
        let win = 900u32..940;
        // One verified segment covering the viewport, carrying its spans.
        let seg = tokenize_segment(&engine, &s, 0..2_000, SegmentStart::Fresh, Some(win.clone()), None);
        let mut c = cache_str(2_000);
        c.set_window(win.clone());
        c.absorb(seg, true);
        assert_eq!(c.first_dirty(), None, "verified absorb clears the whole frontier");
        // The requested spans render immediately — no sync walk. (The window
        // pads ±SLACK beyond `spans_for`, so pending() may still report a
        // slack gap; that is the sync refill's cheap job, tested elsewhere.)
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in win {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    /// Speculative absorb (a GUESSED fresh start) shows spans in the window
    /// immediately, but plants NO checkpoints (unverified states would poison
    /// warm-ups) and leaves the rows dirty (the frontier re-verifies them).
    #[test]
    fn speculative_absorb_shows_spans_without_checkpoints_or_clearing_dirt() {
        let engine = engine_str();
        let lines: Vec<String> = (0..2_000).map(|i| format!("kw word {i}")).collect();
        let s = snap(&lines);
        let win = 900u32..940;
        // A viewport-first speculative segment: back off 128 rows, Fresh guess.
        let seg = tokenize_segment(&engine, &s, 772..940, SegmentStart::Fresh, Some(win.clone()), None);
        let mut c = cache_str(2_000);
        c.set_window(win.clone());
        c.absorb(seg, false);
        // Spans are visible now (the guess is right here - no multi-line state).
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in win.clone() {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "speculative row {r}");
        }
        // But nothing is verified: the frontier is untouched and no checkpoint
        // was planted (retained_state_rows counts window states + checkpoints;
        // the window states are the harmless per-line probes).
        assert_eq!(c.first_dirty(), Some(0), "speculation must not clear dirt");
        assert!(c.ret.checkpoints.is_empty(), "speculation must not plant checkpoints");
    }

    /// End to end: speculative colors first, then the frontier's dirty walk
    /// reaches the window and CONVERGES in O(1) against the correct
    /// speculative state (the guess was right) - the whole window consumed by
    /// tokenizing a single row.
    #[test]
    fn speculative_then_frontier_converges_o1_when_the_guess_was_right() {
        let engine = engine_str();
        let lines: Vec<String> = (0..2_000).map(|i| format!("kw word {i}")).collect();
        let s = snap(&lines);
        let win = 900u32..940;
        let seg = tokenize_segment(&engine, &s, 772..940, SegmentStart::Fresh, Some(win.clone()), None);
        let mut c = cache_str(2_000);
        c.set_window(win.clone());
        c.absorb(seg, false);
        // Now drive the frontier from row 0 (as the idle sweep would). When it
        // reaches the window, each row's freshly-tokenized state matches the
        // stored speculative state, so it converges immediately and the whole
        // window's spans are proven correct.
        drive_all_snap(&mut c, &s);
        assert_eq!(c.first_dirty(), None);
        let full = highlighter_str().highlight(&lines.join("\n"));
        for r in win {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r}");
        }
    }

    /// The poison guard: a WRONG speculative span (guessed fresh inside a
    /// multi-line comment) must never survive as clean when a verified
    /// spans-less segment clears its dirt. Verified absorb evicts it; the sync
    /// refill (from the correct checkpoints) repaints it right.
    #[test]
    fn verified_absorb_evicts_wrong_speculative_spans_no_poison() {
        let engine = engine_cmt();
        let mut lines: Vec<String> = (0..4_000).map(|i| format!("kw word {i}")).collect();
        lines[100] = "/*".to_string(); // a comment that stays open…
        lines[3_000] = "*/ kw".to_string(); // …until here
        let s = snap(&lines);
        let win = 1_500u32..1_540; // deep inside the comment
        let full = highlighter_cmt().highlight(&lines.join("\n"));
        let mut c = cache_cmt(4_000);
        c.set_window(win.clone());
        // Speculate from a FRESH (ground) start — WRONG here, the truth is
        // in-comment.
        let spec = tokenize_segment(&engine, &s, 1_400..1_540, SegmentStart::Fresh, Some(win.clone()), None);
        c.absorb(spec, false);
        assert_ne!(
            c.line_spans(1_520),
            Some(full[1_520].as_slice()),
            "the speculative guess must actually be wrong here (else the test proves nothing)"
        );
        // The correct verified sweep (one segment, Fresh at row 0 = true).
        let (verified, _) = parallel_verify(&engine, &s, &[], None);
        for v in verified {
            c.absorb(v, true);
        }
        // The wrong speculative span is GONE — evicted, not left clean.
        assert_eq!(c.line_spans(1_520), None, "wrong speculative spans must be evicted, never trusted");
        // …and the sync refill repaints the window correctly.
        drive_all_snap(&mut c, &s);
        for r in win {
            assert_eq!(c.line_spans(r), Some(full[r as usize].as_slice()), "row {r} corrected");
        }
    }

    #[test]
    fn dirty_ranges_shift_merge_unit() {
        let mut d = DirtyRanges::default();
        d.insert(5);
        d.insert(7);
        d.insert(6); // bridges 5..6 and 7..8 into one run
        assert_eq!(d.0, vec![5..8]);
        d.insert(4); // extends left
        assert_eq!(d.0, vec![4..8]);
        // Splice via a single covering span: rows [5,7) became 3 rows → tail
        // shifts by +1 (apply_splices reproduces the classic commit splice).
        d.apply_splices(&[(5, 2, 3)]);
        assert_eq!(d.0, vec![4..9], "clip + shift + new-block merge into one run");
        // A pure delete ahead of the run shifts it left.
        d.apply_splices(&[(0, 2, 0)]);
        assert_eq!(d.0, vec![2..7]);
        // Front consumption.
        assert_eq!(d.first(), Some(2));
        d.remove_first(2);
        assert_eq!(d.0, vec![3..7]);
    }
}
