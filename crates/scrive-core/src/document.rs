//! The document aggregate: the buffer plus its edit history and every derived
//! view (selections, decorations, highlight cache, brackets, folds, find).
//!
//! [`Document`] is the public choke point for changing text. Every edit goes
//! through [`Document::edit`]/[`Document::edit_grouped`], which run the
//! transaction engine and record the inverse for undo — so a caller can never
//! mutate the buffer behind history's back. The one invariant this module rests
//! on: every position-tracked derived fact rides a committed edit's patch
//! through the single mover `rebase_views`, on both the forward path and
//! undo/redo, so no two copies of one position can drift apart.

use core::ops::Range;

use std::borrow::Cow;

use crate::bracket::Brackets;
use crate::buffer::{Buffer, EolFlavor, LoadError, Revision, Snapshot};
use crate::coords::{Bias, Point};
use crate::decorations::{
    DecorationKind, DecorationStore, Diagnostic, DiagnosticsOutcome, Severity, Stickiness,
    TrackedRange,
};
use crate::display_map::{BufferRow, DisplayRow};
use crate::find::{FindQuery, FindState};
use std::cell::{Ref, RefCell};

use crate::fold_map::{FoldMap, FoldSet};
use crate::highlight::{HighlightCache, HighlightEngine, HighlightSpan, SyntaxDef, TokenTheme};
use crate::history::{GroupingHint, History};
use crate::movement::{self, ColumnDir, Granularity, Motion};
use crate::selection::{Selection, SelectionId, SelectionSet};
use crate::transaction::{apply, Committed, EditOp, TransactionError};

/// A text document: buffer + undo history + selections + every derived view.
#[derive(Debug)]
pub struct Document {
    pub(crate) buffer: Buffer,
    history: History,
    pub(crate) selections: SelectionSet,
    /// The anchored rectangle while in column (box) selection mode. `Some`
    /// only between consecutive Ctrl+Shift+Alt+Arrow presses; any other action
    /// clears it, leaving the box as an ordinary multi-cursor set.
    column: Option<ColumnSelection>,
    /// The incremental syntax-highlight cache, present once the app injects
    /// a grammar+theme via [`Document::set_syntax`]. Spliced on every commit.
    highlight: Option<HighlightCache>,
    /// Matched brackets — spliced incrementally on every edit; drives bracket
    /// colorization and the matching-bracket highlight.
    brackets: Brackets,
    /// Tracked-range decoration store: diagnostic squiggles, find matches, and
    /// snippet stops. Rides every forward edit's patch in the commit path so
    /// positions stay reunited with content (one store, one mover).
    decorations: DecorationStore,
    /// Auto-close provenance is store-as-truth: the `AutoClosePair` decorations
    /// in THIS store ARE the live pairs (one per caret that auto-closed, so
    /// multi-caret), each spanning `[open char start, close char start)`. It is
    /// its OWN [`DecorationStore`], separate from `decorations`, because a pair
    /// is a small (≤ caret-count), self-invalidating, per-gesture set: giving it
    /// a dedicated store keeps every arm/clear/validate O(pairs) instead of
    /// scanning the bulk (10k-find-match / diagnostic) store. There is no
    /// separate handle to keep in sync — the store itself is the truth: the
    /// pairs ride every edit through the one mover (`rebase_views` moves this
    /// store beside `decorations`), `validate_autoclose` drops the ones no caret
    /// still occupies, and emptiness is read straight off the store
    /// (`!self.autoclose.is_empty()`).
    autoclose: DecorationStore,
    /// Active code folds — **view state**, never on the buffer or the undo
    /// stack. Rides the commit-path patch mover (see `rebase_views`) like a
    /// decoration, so folds survive edits and undo/redo with no fold-specific
    /// resync. Queried per frame through a `FoldMap` (never stored).
    folds: FoldSet,
    /// Find view-state: the active query, the active match's decoration id, and
    /// the coverage/cap bookkeeping. The match set itself lives ONLY in
    /// `decorations` (the sorted `FindMatch` set IS the set) and is repaired
    /// eagerly per commit via `rebase_views`. Idle until `set_find_query`.
    find: FindState,
    /// The language's line-comment prefix (e.g. `//`), injected by the app
    /// like the grammar — the core ships no language knowledge. `None`
    /// makes [`Document::toggle_line_comment`] a no-op.
    line_comment: Option<String>,
    /// The expand-selection ladder: each [`Document::expand_selection`]
    /// pushes the pre-expansion set so `shrink_selection` can walk back down.
    /// Transient like `column` — any other gesture or edit clears it.
    expand_stack: Vec<SelectionSet>,
    /// Monotonic reveal-request counter: bumped by actions that move the caret
    /// programmatically — outside the widget's own input path — so the widget
    /// can autoscroll to the new caret. The view compares it to its last-seen
    /// value and reveals once on a change (a generation counter rather than an
    /// `Option<request>`, since a `&Document` renderer can't take or clear a
    /// request at layout). Verbs bump it via [`Document::request_reveal`] ONLY
    /// when they actually changed something — a no-op F8 or bracket jump never
    /// scrolls.
    reveal_seq: u64,
    /// The strategy of the last reveal request.
    reveal_mode: RevealMode,
    /// Memoized [`FoldMap`]. The render path queries it ~20× per frame, so it is
    /// cached rather than rebuilt (a fresh build is O(folds), which at document
    /// scale with everything collapsed would stall scrolling). A fold toggle
    /// ([`FoldSet::generation`]) or a line-count change rebuilds from scratch; a
    /// no-line-change edit shifts the cached map in place via
    /// [`FoldMap::apply_patch`] (driven by [`Document::edit_grouped`]), so plain
    /// typing over a document-scale fold set costs O(edit), not an O(folds)
    /// rebuild per keystroke. Because it is shifted incrementally it could in
    /// principle drift, so the drift oracle
    /// (`fold_map_cache_matches_a_fresh_build_across_changes`) deep-equals it
    /// against a fresh build to keep it honest.
    fold_cache: RefCell<FoldMapCache>,
}

/// The document's memoized [`FoldMap`] plus the `(buffer revision, fold
/// generation)` key it was built at.
#[derive(Debug)]
struct FoldMapCache {
    key: (u64, u64),
    map: FoldMap,
}

/// How a reveal request should autoscroll the view to the newest caret.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum RevealMode {
    /// Fit the newest caret with the margin band, but HOLD the viewport if any
    /// cursor is already on screen — a multi-cursor op (select-all-occurrences,
    /// multi-cursor typing) must not scroll away from cursors the user can
    /// already see. Identical to a plain fit for a single caret.
    Fit,
    /// Always fit the newest caret, even when other cursors are on screen — a
    /// deliberate jump to a just-added cursor (Ctrl+D add-next-occurrence).
    FitForce,
    /// Center the target row (find / diagnostic navigation).
    Center,
}

/// The two corners of a column (box) selection, in DISPLAY space:
/// display *cells*, not byte columns — tabs and collapsed inline folds make
/// the two disagree, and the box is a visual rectangle. Every spanned row
/// resolves the same two cells through the one click inverse
/// ([`FoldMap::hit_row`]), clamping to its own content, so the box stays
/// rectangular over ragged lines and shrinks exactly when the active corner
/// moves back toward the anchor.
#[derive(Copy, Clone, Debug)]
struct ColumnSelection {
    anchor: CellCorner,
    active: CellCorner,
}

/// One box corner: a buffer row (visible when the corner was placed) and a
/// *virtual* display cell — the cell may sit past the row's content.
#[derive(Copy, Clone, Debug)]
struct CellCorner {
    row: u32,
    cell: u32,
}

impl Document {
    /// Load `text` as a new document (see [`Buffer::new`] for the load
    /// contract: size limit, CRLF normalization, EOL-flavor detection). Starts
    /// with a single caret at the top.
    ///
    /// [`Buffer::new`]: crate::Buffer::new
    pub fn new(text: &str) -> Result<Self, LoadError> {
        let buffer = Buffer::new(text)?;
        let brackets = Brackets::match_text(&buffer.text());
        Ok(Self {
            buffer,
            history: History::new(),
            selections: SelectionSet::new(0),
            column: None,
            highlight: None,
            brackets,
            decorations: DecorationStore::new(),
            autoclose: DecorationStore::new(),
            folds: FoldSet::new(),
            find: FindState::new(),
            line_comment: None,
            expand_stack: Vec::new(),
            reveal_seq: 0,
            reveal_mode: RevealMode::Fit,
            // Seed with a never-matching key so the first `fold_map()` builds it.
            fold_cache: RefCell::new(FoldMapCache { key: (u64::MAX, u64::MAX), map: FoldMap::empty() }),
        })
    }

    /// The memoized fold projection — buffer rows ↔ display rows, hidden
    /// interiors, hit-testing. Rebuilt from scratch only when the buffer revision
    /// or the [`FoldSet`] generation changed since the cached build; otherwise the
    /// cached map is returned untouched, so the render path's ~20 per-frame calls
    /// cost O(1) instead of O(folds) each. The one owner every fold query flows
    /// through.
    #[must_use]
    pub fn fold_map(&self) -> Ref<'_, FoldMap> {
        self.ensure_fold_map();
        Ref::map(self.fold_cache.borrow(), |c| &c.map)
    }

    /// Rebuild the fold cache iff its inputs changed, leaving it current. Split
    /// from [`fold_map`](Self::fold_map) so a caller that also needs `&mut` on a
    /// sibling field (e.g. `move_carets` mutating `selections`) can freshen the
    /// cache and then borrow `fold_cache` and that field disjointly — instead of
    /// building a throwaway O(folds) `FoldMap::new` per keystroke.
    fn ensure_fold_map(&self) {
        let key = (self.buffer.revision().0, self.folds.generation());
        if self.fold_cache.borrow().key != key {
            let map = FoldMap::new(&self.folds, &self.brackets, &self.buffer);
            *self.fold_cache.borrow_mut() = FoldMapCache { key, map };
        }
    }

    /// Ask the view to reveal the newest selection: bump the generation
    /// and record the strategy ([`RevealMode`]). The ONE way a core verb
    /// requests autoscroll; call it only after actually changing state.
    fn request_reveal(&mut self, mode: RevealMode) {
        self.reveal_seq += 1;
        self.reveal_mode = mode;
    }

    /// The [`RevealMode`] of the pending reveal request.
    #[must_use]
    pub fn reveal_mode(&self) -> RevealMode {
        self.reveal_mode
    }

    /// Inject the language's line-comment prefix (e.g. `"//"`) — app-supplied
    /// configuration, like [`Document::set_syntax`]; `None` disables
    /// toggle-comment.
    pub fn set_line_comment(&mut self, prefix: Option<&str>) {
        self.line_comment = prefix.map(str::to_owned);
    }

    /// The injected line-comment prefix, if any.
    #[must_use]
    pub fn line_comment(&self) -> Option<&str> {
        self.line_comment.as_deref()
    }

    /// Read-only access to the underlying buffer (text, lines, coordinates,
    /// snapshots).
    #[must_use]
    pub fn buffer(&self) -> &Buffer {
        &self.buffer
    }

    /// The current selection set (carets and ranges).
    #[must_use]
    pub fn selections(&self) -> &SelectionSet {
        &self.selections
    }

    /// Replace the selection set (e.g. a mouse click placing a caret). Offsets
    /// are clamped to the buffer.
    pub fn set_selections(&mut self, selections: SelectionSet) {
        self.reset_transient();
        self.selections = selections;
    }

    /// Move (or, with `extend`, drag) every caret by `motion`. Vertical
    /// motion is fold-aware: the caret steps display rows, skipping folds.
    pub fn move_carets(&mut self, motion: Motion, extend: bool) {
        self.reset_transient();
        // Read the MEMOIZED fold map, not a throwaway `FoldMap::new` per arrow
        // press: a fresh build is O(folds), which janks at document scale with
        // everything collapsed. Freshen the cache, then borrow `fold_cache` and
        // `selections` disjointly.
        self.ensure_fold_map();
        let tab = self.tab_size();
        let cache = self.fold_cache.borrow();
        movement::move_selections(&mut self.selections, &self.buffer, &cache.map, tab, motion, extend);
    }

    /// Add a bare caret at `offset` — the Alt+Click add-caret gesture.
    /// Merges into any selection it touches via the set's merge rule.
    pub fn add_caret(&mut self, offset: u32) {
        self.reset_transient();
        self.selections.add_caret(offset);
    }

    /// Ctrl+D. If the newest selection is an empty caret, select the word
    /// surrounding it; otherwise add the next literal occurrence of the newest
    /// selection's text as a new (and now newest) selection — scanning forward
    /// from its end, wrapping to the document start, skipping already-selected
    /// ranges. No word / no further match ⇒ no change. Selection-only, no edit.
    pub fn add_next_occurrence(&mut self) {
        self.reset_transient();
        let newest = *self.selections.newest();
        if newest.is_empty() {
            // First press: expand the caret to its word. `add_selection` mints a
            // newer id, so the range absorbs the caret and becomes the newest.
            if let Some((ws, we)) = movement::surrounding_word(&self.buffer, newest.head()) {
                if ws != we {
                    self.selections.add_selection(ws, we);
                    // Ctrl+D jumps to the just-added cursor even if others are on
                    // screen — a deliberate FitForce, not the hold-if-visible Fit
                    // that select-all-occurrences uses.
                    self.request_reveal(RevealMode::FitForce);
                }
            }
            return;
        }
        // Read just the needle, not the whole rope: reading the selection alone
        // keeps each press O(needle), never O(document).
        let needle = self.buffer.slice(newest.start()..newest.end()).into_owned();
        let len = needle.len() as u32;
        // Taken-check by binary search over the selection starts (already sorted
        // by start), not an O(carets) linear `any` per candidate.
        let taken: Vec<(u32, u32)> =
            self.selections.all().iter().map(|s| (s.start(), s.end())).collect();
        let is_taken = |s: u32| {
            taken.binary_search_by_key(&s, |&(st, _)| st).is_ok_and(|i| taken[i].1 == s + len)
        };
        // Next literal occurrence at/after the caret, wrapping to the start —
        // windowed over the buffer, never materializing it.
        let from = newest.end();
        let found = scan_from(&self.buffer, &needle, from, self.buffer.len(), &is_taken)
            .or_else(|| scan_from(&self.buffer, &needle, 0, from, &is_taken));
        if let Some((start, end)) = found {
            self.selections.add_selection(start, end);
            // Reveal the newly added cursor (FitForce — jump even if other
            // cursors are already on screen).
            self.request_reveal(RevealMode::FitForce);
        }
    }

    /// Ctrl+Shift+\: move each caret to its matching bracket. Adjacent to
    /// a matched bracket (the [`Brackets::active_pair`] rule — left of the
    /// caret preferred), the caret crosses to the SAME side of the partner, so
    /// a second press returns it. With no adjacent bracket it jumps to the
    /// innermost enclosing pair's closer. No bracket in reach ⇒ the
    /// caret stays. Collapses selections to carets (a plain motion).
    pub fn jump_to_bracket(&mut self) {
        self.reset_transient();
        let targets: Vec<u32> = self
            .selections
            .all()
            .iter()
            .map(|s| {
                let head = s.head();
                if let Some((b, partner)) = self.brackets.active_pair(head) {
                    if head == b + 1 {
                        partner + 1
                    } else {
                        partner
                    }
                } else if let Some((_, close)) = self.brackets.enclosing_pair(head) {
                    close
                } else {
                    head
                }
            })
            .collect();
        let moved =
            targets.iter().zip(self.selections.all()).any(|(&t, s)| t != s.head() || !s.is_empty());
        self.selections = SelectionSet::from_offsets(&targets);
        for &t in &targets {
            self.unfold_to_reveal(t); // a jump lands on a visible position
        }
        if moved {
            self.request_reveal(RevealMode::Fit); // a bracket hop Fit-reveals
        }
    }

    /// Shift+Alt+Right: grow every selection one structural step — caret
    /// → word → bracket contents → bracket pair → outward, ending at the whole
    /// document — pushing the previous set so [`Document::shrink_selection`]
    /// can walk back down. Fully expanded is a no-op (nothing pushed). The
    /// ladder clears on any other gesture or edit. Selection-only.
    pub fn expand_selection(&mut self) {
        self.reset_gesture();
        let before: Vec<(u32, u32)> =
            self.selections.all().iter().map(|s| (s.start(), s.end())).collect();
        let ranges: Vec<(u32, u32)> =
            before.iter().map(|&(s, e)| self.expanded_range(s, e)).collect();
        if ranges == before {
            return;
        }
        let newest_id = self.selections.newest().id;
        let newest = self
            .selections
            .all()
            .iter()
            .position(|s| s.id == newest_id)
            .expect("the newest selection is in its own set");
        self.expand_stack.push(self.selections.clone());
        self.selections = SelectionSet::from_ranges(&ranges, newest);
        self.request_reveal(RevealMode::Fit); // keep the growing head in view
    }

    /// Shift+Alt+Left: step back down the expansion ladder — restore the
    /// set exactly as it was before the matching expand. Empty ladder ⇒ no-op.
    pub fn shrink_selection(&mut self) {
        self.reset_gesture();
        if let Some(prev) = self.expand_stack.pop() {
            self.selections = prev;
            self.request_reveal(RevealMode::Fit);
        }
    }

    /// One structural expansion step for `[start, end]`: an empty caret
    /// grows to its word; otherwise the innermost pair whose contents contain
    /// the range supplies the rung — its contents first, then (already filled)
    /// the pair including its brackets, whose own enclosing pair is one level
    /// out. Outside every pair: the whole document.
    fn expanded_range(&self, start: u32, end: u32) -> (u32, u32) {
        if start == end {
            if let Some((ws, we)) = movement::surrounding_word(&self.buffer, start) {
                if ws != we {
                    return (ws, we);
                }
            }
        }
        if let Some((open, close)) = self.brackets.enclosing_pair_of_range(start, end) {
            let contents = (crate::row_layout::gap_left_edge(open), close);
            if contents != (start, end) {
                return contents;
            }
            return (open, close + 1);
        }
        (0, self.buffer.len())
    }

    /// Ctrl+Alt+↑/↓: add a caret one display row above/below EVERY
    /// existing caret, keeping the current set — the stack-a-column gesture.
    /// Each landing resolves like a click at that caret's visual column (the
    /// `movement::caret_one_display_row` rule → `hit_row`), so tabs, chips,
    /// and collapsed folds behave exactly as plain vertical movement; a caret
    /// already on the first/last display row adds nothing, and a landing on an
    /// existing caret merges via the set's rule. Selection-only. (Landings on
    /// short lines clamp; the visual goal column is not carried across
    /// presses — a deliberate simplification.)
    pub fn add_caret_vertical(&mut self, down: bool) {
        self.reset_transient();
        let folds = FoldMap::new(&self.folds, &self.brackets, &self.buffer);
        let tab = self.tab_size();
        let delta = if down { 1 } else { -1 };
        let heads: Vec<u32> = self.selections.all().iter().map(Selection::head).collect();
        let mut added = false;
        for head in heads {
            if let Some(off) = movement::caret_one_display_row(&self.buffer, &folds, tab, head, delta)
            {
                self.selections.add_caret(off);
                added = true;
            }
        }
        if added {
            self.request_reveal(RevealMode::Fit); // keep the newest caret in view
        }
    }

    /// Ctrl+Shift+L: select EVERY occurrence of the newest selection's
    /// text as a multi-cursor set, seeding from the word under a bare caret
    /// (like Ctrl+D) — the multi-cursor rename: select all, type once.
    /// Single-pass: ONE scan collects every non-overlapping occurrence (the
    /// same `match_indices` rule `find_next_occurrence` uses), skips
    /// already-taken selection ranges, and adds the rest in cyclic order —
    /// forward from the seed, wrapping — so the final `newest` (the reveal
    /// target) is the last occurrence in that order. Selection-only, no edit.
    pub fn select_all_occurrences(&mut self) {
        self.reset_transient();
        let newest = *self.selections.newest();
        if newest.is_empty() {
            let Some((ws, we)) = movement::surrounding_word(&self.buffer, newest.head()) else {
                return;
            };
            if ws == we {
                return;
            }
            self.selections.add_selection(ws, we);
        }
        let newest = *self.selections.newest();
        // Existing selections stay in the set; an occurrence already covered by
        // one merges into it (normalize dedups), so the taken filter only needs
        // to skip re-listing them — an O(occurrences) HashSet probe, not an
        // O(occurrences × selections) `Vec::contains`.
        let existing: Vec<(u32, u32)> =
            self.selections.all().iter().map(|s| (s.start(), s.end())).collect();
        let taken: std::collections::HashSet<(u32, u32)> = existing.iter().copied().collect();
        let occurrences: Vec<(u32, u32)> = {
            let text = self.buffer.text();
            let needle = &text[newest.start() as usize..newest.end() as usize];
            if needle.is_empty() {
                return;
            }
            let len = needle.len() as u32;
            text.match_indices(needle)
                .map(|(i, _)| (i as u32, i as u32 + len))
                .filter(|occ| !taken.contains(occ))
                .collect()
        };
        if occurrences.is_empty() {
            // Every match is already selected — the set is unchanged; just reveal.
            let head = self.selections.newest().head();
            self.unfold_to_reveal(head);
            self.request_reveal(RevealMode::Fit);
            return;
        }
        // Build the whole set in one shot: `from_ranges` runs a SINGLE normalize
        // (O(m log m)); adding occurrences one at a time would re-normalize on
        // every insert (O(m²)). Reveal target = the last occurrence in the
        // forward-cyclic order (`[wrap.., ..wrap]`), the one nearest before the
        // seed, so the viewport reveal lands where the user expects.
        let wrap = occurrences.partition_point(|&(s, _)| s < newest.end());
        let reveal = (wrap + occurrences.len() - 1) % occurrences.len();
        let base = existing.len();
        let mut ranges = existing;
        ranges.extend_from_slice(&occurrences);
        self.selections = SelectionSet::from_ranges(&ranges, base + reveal);
        // The newest occurrence may sit inside a collapsed fold — unfold it
        // and Fit-reveal, so the reveal lands on a visible position rather
        // than a hidden one the widget could not scroll to.
        let head = self.selections.newest().head();
        self.unfold_to_reveal(head);
        self.request_reveal(RevealMode::Fit);
    }

    /// Every live find match becomes a selection (the find bar's
    /// select-all-matches, Alt+Enter) — the multi-cursor "replace all": select
    /// every match, type once. The active match
    /// (or the first) becomes the newest selection, so the caret stays where
    /// find navigation left it. No live matches ⇒ `false`, nothing changes.
    pub fn select_find_matches(&mut self) -> bool {
        let ranges: Vec<(u32, u32)> =
            self.find_matches_in(0..u32::MAX).map(|(r, _)| (r.start, r.end)).collect();
        if ranges.is_empty() {
            return false;
        }
        let newest = (self.active_find_match().unwrap_or(0) as usize).min(ranges.len() - 1);
        self.selections = SelectionSet::from_ranges(&ranges, newest);
        self.reset_transient(); // a jump-class change: seals undo, clears box state
        // The active match must be visible (matches hidden in folds stay as
        // selections — typing into them expands via the edit path).
        let head = self.selections.newest().head();
        self.unfold_to_reveal(head);
        self.request_reveal(RevealMode::Center); // find-family jumps center
        true
    }

    /// Collapse a multi-cursor set (or a single range) to one caret — the primary
    /// (oldest) cursor. The Escape gesture.
    pub fn collapse_selections(&mut self) {
        self.reset_transient();
        self.selections.collapse_to_primary();
    }

    /// Select the whole document — one selection from the start to the end, head
    /// at the end (Ctrl+A). Selection-only, no edit; an empty buffer stays a
    /// single caret at 0.
    pub fn select_all(&mut self) {
        self.reset_transient();
        let end = self.buffer.len();
        self.selections.set_single(Selection::from_anchor(SelectionId(0), 0, end));
    }

    /// Clear transient per-gesture state — the column (box) anchor, the
    /// expand-selection ladder, and the auto-close provenance — and **seal the
    /// open undo group**: every caret move, click, or jump is an undo-group
    /// boundary, so typing after it never merges with the run before it.
    /// Called by every selection-changing verb that isn't a type/backspace (a
    /// plain type keeps the provenance so overtype works, and stays mergeable).
    /// One owner: a new selection verb gets the boundary by calling this, never
    /// by sealing directly.
    fn reset_transient(&mut self) {
        self.expand_stack.clear();
        self.reset_gesture();
    }

    /// [`reset_transient`](Self::reset_transient) minus the expand ladder —
    /// for the expand/shrink verbs themselves, which own that stack but are
    /// still selection gestures (box exit, provenance drop, undo boundary).
    fn reset_gesture(&mut self) {
        self.column = None;
        self.clear_autoclose();
        self.history.seal();
    }

    /// Record fresh auto-close provenance for the given `[open, close)`
    /// pairs — one `AutoClosePair` decoration each, added in ONE batch (a
    /// multi-caret keystroke pairs at every caret; per-item adds would be
    /// quadratic). `AlwaysGrows` so typing at the caret (on the close boundary)
    /// pushes the close along while the open stays put; the store then moves them
    /// through every edit. The pairs live in their OWN [`autoclose`](Self::autoclose)
    /// store (≤ caret-count items), so this batch add is O(pairs), never O(decorations).
    pub(crate) fn add_autoclose_pairs(&mut self, pairs: impl IntoIterator<Item = (u32, u32)>) {
        self.autoclose.add_sorted_batch(
            pairs.into_iter().map(|(open, close)| open..close),
            DecorationKind::AutoClosePair,
            Stickiness::AlwaysGrows,
        );
    }

    /// Every live provenance range `[open_char_start, close_char_start)`, in
    /// ascending order — empty when no pair is active. The pairs are
    /// disjoint, so the ends are ascending too (overtype relies on this).
    /// O(pairs): the own store holds only `AutoClosePair`, so no kind filter is
    /// needed.
    pub(crate) fn autoclose_ranges(&self) -> Vec<Range<u32>> {
        self.autoclose.iter().map(|r| r.range.clone()).collect()
    }

    /// Drop ALL auto-close provenance (its decorations). O(pairs): the own store
    /// holds only the ≤(caret-count) pairs, so this empties it directly — no
    /// whole-`decorations` retain, and no dirty flag to reset.
    pub(crate) fn clear_autoclose(&mut self) {
        if self.autoclose.is_empty() {
            return; // no live pair — nothing to take
        }
        self.autoclose.take_matching_in(0..u32::MAX, |_| true);
    }

    /// Drop each provenance pair no caret still validly occupies — its line
    /// changed, or no caret sits in `(open, close]`. Reads the *rebased* pairs,
    /// so it runs after the commit mover. The carets are sorted, so each pair is
    /// probed in O(log carets); the scan is over the own store (≤ caret-count),
    /// so validate is O(pairs · log carets), independent of `decorations`.
    pub(crate) fn validate_autoclose(&mut self) {
        if self.autoclose.is_empty() {
            return; // no live pair — nothing to validate
        }
        let carets: Vec<u32> = self.selections.all().iter().map(|s| s.head()).collect();
        let Self { autoclose, buffer, .. } = self;
        autoclose.take_matching_in(0..u32::MAX, |r| {
            debug_assert!(
                matches!(r.kind, DecorationKind::AutoClosePair),
                "the autoclose store holds only AutoClosePair provenance",
            );
            let (start, end) = (r.range.start, r.range.end);
            let one_line = buffer.offset_to_point(start).row == buffer.offset_to_point(end).row;
            // A caret validly occupies the pair when it is in (open, close].
            let occupied = one_line && {
                let i = carets.partition_point(|&h| h <= start);
                i < carets.len() && carets[i] <= end
            };
            !occupied // remove the pair no caret occupies
        });
    }

    /// Drag-select from `origin` to `head` at the given `granularity`. Char
    /// is a plain range; Word/Line extend by whole units and keep the origin unit
    /// fully selected even when the drag reverses past it (the tail flips to the
    /// far edge of the origin unit). `head == origin` selects just the origin
    /// unit — the double/triple-click initial selection. Replaces the set.
    pub fn drag_select(&mut self, granularity: Granularity, origin: u32, head: u32) {
        self.reset_transient();
        let (anchor, head) = match granularity {
            Granularity::Char => (origin, head),
            Granularity::Word => {
                extend_by_unit(self.word_unit(origin), self.word_unit(head), origin, head)
            }
            Granularity::Line => {
                extend_by_unit(self.line_unit(origin), self.line_unit(head), origin, head)
            }
        };
        self.selections.set_single(Selection::from_anchor(SelectionId(0), anchor, head));
    }

    /// The word range surrounding `offset`, or a bare caret there in whitespace.
    fn word_unit(&self, offset: u32) -> (u32, u32) {
        movement::surrounding_word(&self.buffer, offset).unwrap_or((offset, offset))
    }

    /// The line range at `offset`, including the trailing newline (or to line
    /// end on the final line). THE whole-line unit — triple-click drag and the
    /// whole-line Copy/Cut/Paste verbs all read this one rule.
    pub(crate) fn line_unit(&self, offset: u32) -> (u32, u32) {
        let row = self.buffer.offset_to_point(offset).row;
        let start = self.buffer.point_to_offset(Point::new(row, 0));
        let end = if row + 1 < self.buffer.line_count() {
            self.buffer.point_to_offset(Point::new(row + 1, 0))
        } else {
            self.buffer.point_to_offset(Point::new(row, self.buffer.line_len(row)))
        };
        (start, end)
    }

    /// Column (box) selection — Ctrl+Shift+Alt+Arrow. The first press anchors
    /// at the primary caret; each press steps the *active* corner by `dir` and
    /// rebuilds one selection per spanned row, from the anchor cell to the active
    /// cell. Stepping the active corner back toward the anchor shrinks the box.
    /// Display-space end to end: corners are display CELLS, so the box stays
    /// visually rectangular across tabs and collapsed inline folds; vertical
    /// steps walk display rows (a collapsed fold is one step, not one per hidden
    /// row); and cells are virtual (unbounded right), so a box can reach past
    /// short lines and still select the full width of longer ones. Selection-only;
    /// any other action exits the mode, leaving the box as a multi-cursor set.
    pub fn column_select(&mut self, dir: ColumnDir) {
        // A box gesture is a selection change, so it is an undo-group boundary
        // and it invalidates the expand ladder — but it must KEEP `self.column`,
        // so it can't go through `reset_transient`.
        self.history.seal();
        self.expand_stack.clear();
        let folds = FoldMap::new(&self.folds, &self.brackets, &self.buffer);
        let tab = self.tab_size();
        let mut col = self.column.unwrap_or_else(|| {
            let corner = self.caret_corner(&folds, tab);
            ColumnSelection { anchor: corner, active: corner }
        });
        col.active = Self::step_corner(&folds, col.active, dir);
        self.column = Some(col);
        self.rebuild_column_box(&folds, tab, col);
    }

    /// Mouse box (column) selection — Shift+Alt+drag. Sets the box from an
    /// `anchor` and `active` corner, each `(visible buffer row, display cell)`;
    /// the widget derives the row from the one row inversion and the cell from
    /// the one virtual-cell rounding rule ([`crate::row_layout::virtual_cell`]).
    /// Cells are *virtual* (unclamped by line): the box clamps each row to its
    /// own content, so a drag past short lines still selects the full width of
    /// longer ones. Same selection-only semantics as `column_select`; any other
    /// action exits the mode. Shares `Document.column`, so a keyboard box then
    /// continues it.
    pub fn column_drag(&mut self, anchor: (u32, u32), active: (u32, u32)) {
        // Same boundary + ladder rule as `column_select`.
        self.history.seal();
        self.expand_stack.clear();
        let folds = FoldMap::new(&self.folds, &self.brackets, &self.buffer);
        let col = ColumnSelection {
            anchor: CellCorner { row: anchor.0, cell: anchor.1 },
            active: CellCorner { row: active.0, cell: active.1 },
        };
        self.column = Some(col);
        self.rebuild_column_box(&folds, self.tab_size(), col);
    }

    /// The primary caret's box corner: its rendered position — the one owner of
    /// display geometry, [`FoldMap::display_position`] — as a `(visible buffer
    /// row, display cell)`
    /// pair. A caret on a collapsed fold's closing tail anchors on the header
    /// row it renders on. (Fold-time ejection keeps carets visible, so the
    /// hidden-offset fallback to the raw buffer point is belt and braces.)
    fn caret_corner(&self, folds: &FoldMap, tab: u32) -> CellCorner {
        let head = self.selections.newest().head();
        match folds.display_position(&self.buffer, head, tab) {
            Some(p) => CellCorner {
                row: folds.to_buffer_row(p.row).0,
                cell: crate::row_layout::virtual_cell(p.x.cells()),
            },
            None => {
                let p = self.buffer.offset_to_point(head);
                let layout = folds.row_layout(&self.buffer, BufferRow(p.row), tab);
                CellCorner { row: p.row, cell: layout.display_cell(p.col) }
            }
        }
    }

    /// Move the selected line-block up or down one line (Alt+↑/↓). Swaps the
    /// block with its neighbour; the selection rides the moved text. A block at
    /// the document edge — or, moving down, against the trailing empty line — is
    /// a no-op. Single selection. Own undo step; never re-indents.
    pub fn move_line(&mut self, down: bool) {
        let sel = *self.selections.newest();
        let p0 = self.buffer.offset_to_point(sel.start());
        let p1 = self.buffer.offset_to_point(sel.end());
        let (r0, r1) = (p0.row, p1.row);
        let line_count = self.buffer.line_count();
        let ends_nl = self.buffer.char_before(self.buffer.len()) == Some('\n');
        // The last row a real line occupies (the trailing empty line, if any, is
        // not movable — moving into it would drop the terminator).
        let last_movable = if ends_nl { line_count.saturating_sub(2) } else { line_count - 1 };
        if (down && r1 >= last_movable) || (!down && r0 == 0) {
            return;
        }
        let (a, b) = if down { (r0, r1 + 1) } else { (r0 - 1, r1) };
        let has_nl = b + 1 < line_count; // row `b` carries a trailing newline
        let region_start = self.buffer.point_to_offset(Point::new(a, 0));
        let region_end = if has_nl {
            self.buffer.point_to_offset(Point::new(b + 1, 0))
        } else {
            self.buffer.len()
        };
        let lines: Vec<Cow<str>> = (a..=b).map(|r| self.buffer.line(r)).collect();
        let reordered: Vec<Cow<str>> = if down {
            // The line below moves to the front; the block shifts down.
            let mut v = vec![lines[lines.len() - 1].clone()];
            v.extend_from_slice(&lines[..lines.len() - 1]);
            v
        } else {
            // The line above moves to the back; the block shifts up.
            let mut v = lines[1..].to_vec();
            v.push(lines[0].clone());
            v
        };
        let mut new_text = reordered.join("\n");
        if has_nl {
            new_text.push('\n');
        }
        let _ = self.edit(vec![EditOp::new(region_start..region_end, &new_text)]);
        // Place the selection on the moved block (one line over).
        let shift = |r: u32| if down { r + 1 } else { r - 1 };
        let ns = self.buffer.point_to_offset(Point::new(shift(r0), p0.col));
        let ne = self.buffer.point_to_offset(Point::new(shift(r1), p1.col));
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), ns, ne));
        self.selections = set;
        // Drop provenance through the one owner so the `AutoClosePair`
        // decoration is removed with it — the store is the single owner of the
        // pair, with nothing else to keep in sync.
        self.clear_autoclose();
    }

    /// Duplicate the selected line-block above (`down = false`) or below
    /// (`down = true`) — Shift+Alt+↑/↓. The caret lands on the copy. Single
    /// selection. Own undo step.
    pub fn copy_line(&mut self, down: bool) {
        let sel = *self.selections.newest();
        let p0 = self.buffer.offset_to_point(sel.start());
        let p1 = self.buffer.offset_to_point(sel.end());
        let (r0, r1) = (p0.row, p1.row);
        let height = r1 - r0 + 1;
        let line_count = self.buffer.line_count();
        let block: String = (r0..=r1).map(|r| self.buffer.line(r)).collect::<Vec<_>>().join("\n");
        let (at, text) = if !down {
            (self.buffer.point_to_offset(Point::new(r0, 0)), format!("{block}\n"))
        } else if r1 + 1 < line_count {
            (self.buffer.point_to_offset(Point::new(r1 + 1, 0)), format!("{block}\n"))
        } else {
            // Duplicating the final line (no trailing newline): prepend one.
            (self.buffer.len(), format!("\n{block}"))
        };
        let _ = self.edit(vec![EditOp::new(at..at, &text)]);
        let (nr0, nr1) = if down { (r0 + height, r1 + height) } else { (r0, r1) };
        let ns = self.buffer.point_to_offset(Point::new(nr0, p0.col));
        let ne = self.buffer.point_to_offset(Point::new(nr1, p1.col));
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), ns, ne));
        self.selections = set;
        // Same as `move_line`: clear provenance through the owner so the store
        // entry can't be orphaned.
        self.clear_autoclose();
    }

    /// Step a box corner one unit in `dir`, in display space: vertical steps
    /// walk display rows — a collapsed fold is hopped in one step — and clamp
    /// to the document; the left cell saturates at 0; the right cell is
    /// unbounded so it can reach past short lines.
    fn step_corner(folds: &FoldMap, c: CellCorner, dir: ColumnDir) -> CellCorner {
        let d = folds.to_display_row(BufferRow(c.row)).index();
        let row_at = |d: u32| folds.to_buffer_row(DisplayRow(d)).0;
        match dir {
            ColumnDir::Up => CellCorner { row: row_at(d.saturating_sub(1)), cell: c.cell },
            ColumnDir::Down => {
                CellCorner { row: row_at((d + 1).min(folds.max_display_row().index())), cell: c.cell }
            }
            ColumnDir::Left => CellCorner { row: c.row, cell: c.cell.saturating_sub(1) },
            ColumnDir::Right => CellCorner { row: c.row, cell: c.cell + 1 },
        }
    }

    /// Install the box `col` as the selection set: one selection per spanned
    /// *display* row (a collapsed fold's hidden rows get none), each corner cell
    /// resolved to its byte offset through the one click inverse
    /// ([`FoldMap::hit_row`]: tab snapping, chip resolution, header gap/tail) —
    /// so the box selects exactly what its rectangle crosses on screen, clamped
    /// to each row's content. The active row's selection is the newest
    /// (autoscroll target).
    fn rebuild_column_box(&mut self, folds: &FoldMap, tab: u32, col: ColumnSelection) {
        let da = folds.to_display_row(BufferRow(col.anchor.row)).index();
        let dv = folds.to_display_row(BufferRow(col.active.row)).index();
        let (d0, d1) = (da.min(dv), da.max(dv));
        let mut ranges = Vec::with_capacity((d1 - d0 + 1) as usize);
        let mut newest = 0;
        for d in d0..=d1 {
            let row = folds.to_buffer_row(DisplayRow(d));
            let anchor_off = folds.hit_row(&self.buffer, row, col.anchor.cell as f32, Bias::Left, tab);
            let head_off = folds.hit_row(&self.buffer, row, col.active.cell as f32, Bias::Left, tab);
            if d == dv {
                newest = ranges.len();
            }
            ranges.push((anchor_off, head_off));
        }
        self.selections = SelectionSet::from_ranges(&ranges, newest);
    }

    /// The full document text (LF-only). Shorthand for `self.buffer().text()`
    /// — a **cold-path** whole-text read: O(document), never call it per-frame
    /// or per-keystroke.
    #[must_use]
    pub fn text(&self) -> Cow<'_, str> {
        self.buffer.text()
    }

    /// The current revision.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.buffer.revision()
    }

    /// An immutable snapshot for a background consumer (the compile thread).
    /// O(1) — a rope clone; the consumer pays for what it reads.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        self.buffer.snapshot()
    }

    /// Apply a batch of edits as one discrete transaction (its own undo step).
    ///
    /// This is the common programmatic path. For keystroke-level edits that
    /// should merge into one undo unit, use [`Document::edit_grouped`].
    /// Returns the [`Committed`] result (forward patch + change records); an
    /// empty/no-op batch changes nothing and records no undo step.
    pub fn edit(&mut self, ops: Vec<EditOp>) -> Result<Committed, TransactionError> {
        self.edit_grouped(ops, GroupingHint::discrete())
    }

    /// Apply a batch of edits with an explicit [`GroupingHint`]: the
    /// verbs layer uses this so a typing run collapses to one undo step.
    pub fn edit_grouped(
        &mut self,
        ops: Vec<EditOp>,
        hint: GroupingHint,
    ) -> Result<Committed, TransactionError> {
        self.column = None; // any text edit exits column-select mode…
        self.expand_stack.clear(); // …and invalidates the expand ladder
        // The (revision, fold generation) the fold cache would need to be at to
        // shift it in place instead of rebuilding — captured before the edit.
        let pre_fold_key = (self.buffer.revision().0, self.folds.generation());
        let committed = apply(&mut self.buffer, ops)?;
        if !committed.is_empty() {
            // Whether the edit added or removed a bracket character — the
            // reconcile gate below, read from the applied (forward) + inverse ops
            // the transaction already owns, so no `ops.clone()` is needed to see
            // the forward text. History records both (by move) at the end.
            let bracket_edit = committed
                .forward_ops()
                .iter()
                .chain(committed.inverse_ops())
                .any(|op| op.text.bytes().any(crate::bracket::is_bracket_byte));
            // Programmatic edits shift carets through the patch; the verbs layer
            // overrides this with precise post-edit caret placement.
            self.selections.rebase(committed.patch());

            let tab = self.tab_size();
            // Rebase every derived view through the patch — the single mover,
            // shared verbatim with undo/redo (see `rebase_views`). The auto-close
            // provenance is an `AutoClosePair` decoration, so it rides here too —
            // no hand-rebasing; the fold reveal rides inside it too.
            let region = rebase_views(
                &mut Views {
                    highlight: &mut self.highlight,
                    brackets: &mut self.brackets,
                    decorations: &mut self.decorations,
                    autoclose: &mut self.autoclose,
                    folds: &mut self.folds,
                    find: &mut self.find,
                },
                &self.buffer,
                tab,
                &committed,
            );

            // Now that the provenance decoration has ridden the patch, drop it if
            // the caret has left the pair — after the rebase, not before.
            self.validate_autoclose();
            // Drop folds the edit invalidated (e.g. a deleted closing brace) so a
            // modification can't leave rows hidden with no chevron — but ONLY when
            // a bracket CHARACTER was added or removed. If none was, the bracket
            // sequence is unchanged (same chars, merely shifted), so the matching
            // automaton produces the identical pairing — no fold can lose its
            // pair, and reconcile would drop nothing. This skips the whole step
            // for plain typing AND for Enter (a line change with no bracket char).
            // When a bracket DID change, `region` (empty ⇒ shift-only, i.e. not
            // structural) bounds the work to the folds the edit's re-matched rows
            // could touch, not all folds. Any fold whose interior an edit touched
            // was already expanded above.
            if bracket_edit && !region.is_empty() {
                self.reconcile_folds_in(region);
            }
            // Keep the memoized FoldMap current without the per-keystroke O(folds)
            // rebuild: if the cache was current before this edit AND the edit
            // added/removed no fold (generation unchanged — expand/reconcile can
            // bump it), shift it through the patch in place. A line change (or a
            // stale cache, or a fold add/remove) invalidates for a rebuild on the
            // next `fold_map()`. The `pre_fold_key` guard makes a mid-edit
            // `fold_map()` rebuild (which would move the key) fall back safely
            // rather than double-apply the patch.
            {
                let mut cache = self.fold_cache.borrow_mut();
                if cache.key == pre_fold_key
                    && self.folds.generation() == pre_fold_key.1
                    && cache.map.apply_patch(committed.patch(), &self.buffer)
                {
                    cache.key = (self.buffer.revision().0, pre_fold_key.1);
                } else {
                    cache.key = (u64::MAX, u64::MAX);
                }
            }
            // Record history LAST, moving the forward + inverse op batches out of
            // `committed` with no clone (rebase_views already used its borrow of
            // the inverse; the fold-cache borrow above has been dropped, freeing
            // `&mut self` for `history`). The returned `Committed` keeps only the
            // patch — the sole part callers (the verbs layer) read.
            let (patch, forward, inverse) = committed.into_ops();
            self.history.record(forward, inverse, hint);
            return Ok(Committed::from_patch(patch));
        }
        Ok(committed)
    }

    /// Undo the most recent undo unit. Returns `false` if there is nothing to
    /// undo.
    ///
    /// Each reverted step threads its patch through `rebase_views` — the exact
    /// mover forward edits use — so highlight, brackets, and every decoration
    /// (diagnostics, find matches, snippet stops) stay consistent with no
    /// undo-specific resync per feature. The fields are destructured so the
    /// per-step callback can borrow the views while history borrows the buffer.
    pub fn undo(&mut self) -> bool {
        self.reset_transient();
        let tab = self.tab_size();
        // Where to land the caret: the START of the reverted region, so undo takes
        // you TO the change (as mainstream editors do) instead of leaving the
        // caret wherever it happened to be. Steps revert newest→oldest, so the LAST
        // callback is the oldest edit — the region's start.
        let mut caret_home: Option<u32> = None;
        let Self {
            history, buffer, selections, highlight, brackets, decorations, autoclose, folds, find, ..
        } = self;
        let mut views = Views { highlight, brackets, decorations, autoclose, folds, find };
        let undone = history.undo(buffer, selections, |committed, buffer| {
            // The one mover — highlight, brackets, decorations, folds (position +
            // fold reveal) all ride it, so undo needs no per-feature resync.
            rebase_views(&mut views, buffer, tab, committed);
            if let Some(e) = committed.patch().edits().first() {
                caret_home = Some(e.new.start);
            }
        });
        // Reverting text can invalidate a fold's bracket pair too — reconcile once
        // the buffer/brackets are settled, same as the forward edit path.
        if undone {
            self.reconcile_folds();
            // Move the caret to the reverted edit (the widget's `moves_caret`
            // autoscroll then follows it into view). Overrides the mid-revert
            // rebase — the user wants to SEE what undo changed, not stay put.
            if let Some(off) = caret_home {
                self.selections = SelectionSet::new(off);
            }
        }
        undone
    }

    /// Redo the most recently undone unit — the [`undo`](Document::undo) mirror,
    /// through the same `rebase_views` mover. Returns `false` if nothing to redo.
    pub fn redo(&mut self) -> bool {
        self.reset_transient();
        let tab = self.tab_size();
        // Symmetric to `undo`: land the caret at the END of the re-applied region,
        // where it would sit had the edit just been made. Steps replay oldest→newest,
        // so the LAST callback is the newest edit — the region's end.
        let mut caret_home: Option<u32> = None;
        let Self {
            history, buffer, selections, highlight, brackets, decorations, autoclose, folds, find, ..
        } = self;
        let mut views = Views { highlight, brackets, decorations, autoclose, folds, find };
        let redone = history.redo(buffer, selections, |committed, buffer| {
            // The same one mover as `undo` (folds ride it, position + fold reveal).
            rebase_views(&mut views, buffer, tab, committed);
            if let Some(e) = committed.patch().edits().last() {
                caret_home = Some(e.new.end);
            }
        });
        if redone {
            self.reconcile_folds();
            if let Some(off) = caret_home {
                self.selections = SelectionSet::new(off);
            }
        }
        redone
    }

    /// Whether the document differs from the last [`Document::mark_saved`].
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.history.is_dirty()
    }

    /// Record the current state as saved (clears the dirty flag).
    pub fn mark_saved(&mut self) {
        self.history.mark_saved();
    }

    /// Whether there is an edit to [`undo`](Document::undo).
    #[must_use]
    pub fn can_undo(&self) -> bool {
        self.history.can_undo()
    }

    /// Whether there is an edit to [`redo`](Document::redo) along the current
    /// branch.
    #[must_use]
    pub fn can_redo(&self) -> bool {
        self.history.can_redo()
    }

    /// How many redo branches diverge from the current position: 1 in the
    /// classic single-line case, ≥2 at a fork where an undo was followed by a
    /// *divergent* edit — the branching undo tree retains the branch you undid
    /// out of rather than discarding it, so it stays reachable.
    #[must_use]
    pub fn redo_branch_count(&self) -> usize {
        self.history.redo_branch_count()
    }

    /// Steer the next [`redo`](Document::redo) to the `index`-th redo branch
    /// (0 = oldest), returning `false` if out of range. The chosen branch also
    /// becomes the default thereafter. Plain redo (without selecting) always
    /// follows the most-recent branch, so single-line undo is unchanged.
    pub fn select_redo_branch(&mut self, index: usize) -> bool {
        self.history.select_redo_branch(index)
    }

    /// Cap undo history at `limit` units on the current line (`None` = the
    /// default, unbounded), dropping older units and any branch that diverged
    /// before the kept window. Opt-in: a host that wants bounded undo memory
    /// calls this; by default nothing is ever pruned.
    pub fn set_undo_limit(&mut self, limit: Option<usize>) {
        self.history.set_max_undo(limit);
    }

    /// How many undo steps are currently reachable — the depth of `undo`
    /// (bounded by [`set_undo_limit`](Document::set_undo_limit) when set).
    #[must_use]
    pub fn undo_depth(&self) -> usize {
        self.history.undo_depth()
    }

    /// Serialize for saving, re-expanding LF to the given flavor.
    #[must_use]
    pub fn serialize(&self, flavor: EolFlavor) -> String {
        self.buffer.serialize(flavor)
    }

    /// Attach a syntax highlighter. The grammar and theme are app-supplied
    /// (scrive-core is language-agnostic). The cache starts sized to the current
    /// line count, every line dirty; call [`Document::tokenize_highlight`] before
    /// reading spans, and it re-splices on every edit. A mid-session grammar
    /// swap carries the previous retention-window aim forward, so the currently
    /// visible rows are re-highlighted immediately even though nothing visible
    /// moved to re-trigger the widget's viewport report.
    pub fn set_syntax(&mut self, syntax: SyntaxDef, theme: TokenTheme) {
        let aim = self.highlight.as_ref().map(HighlightCache::window_aim);
        let mut cache = HighlightCache::new(syntax, theme, self.buffer.line_count());
        if let Some(aim) = aim {
            cache.set_window(aim);
        }
        self.highlight = Some(cache);
    }

    /// Swap the highlight theme, keeping the grammar and cache sizing. The
    /// whole cache invalidates; colors repaint on the next `tokenize_highlight`
    /// (old colors show meanwhile — see [`HighlightCache::set_theme`]). No-op
    /// without a highlighter.
    pub fn set_theme(&mut self, theme: TokenTheme) {
        if let Some(cache) = self.highlight.as_mut() {
            cache.set_theme(theme);
        }
    }

    /// The tracked-range decoration store. Exposed so an app-level owner —
    /// e.g. a snippet session registering one range per tab stop — can
    /// drive it through the store's handle API. The store rides edits
    /// automatically: every transaction patches it before change events fire.
    #[must_use]
    pub fn decorations(&self) -> &DecorationStore {
        &self.decorations
    }

    /// Mutable [`decorations`](Document::decorations) for a decoration owner.
    pub fn decorations_mut(&mut self) -> &mut DecorationStore {
        &mut self.decorations
    }

    /// How many live auto-close provenance pairs the own store holds — the
    /// store-as-truth count, for tests that pin the provenance lifetime.
    #[cfg(test)]
    #[must_use]
    pub(crate) fn autoclose_pair_count(&self) -> usize {
        self.autoclose.len()
    }

    /// The active code folds — read access for building a per-frame
    /// [`FoldMap`].
    #[must_use]
    pub fn folds(&self) -> &FoldSet {
        &self.folds
    }

    /// Mutable access to the fold set (e.g. `clear`). Folds are view state:
    /// mutating them records nothing on the undo stack. Row-based fold/unfold go
    /// through the convenience methods below (they need the buffer too).
    pub fn folds_mut(&mut self) -> &mut FoldSet {
        &mut self.folds
    }

    /// Fold the multi-line bracket pair opening on `header` and closing on
    /// `last`. A fold is keyed by that pair's opening-bracket offset. Nesting is
    /// allowed (folds may sit inside or around others). Returns `false` if no such
    /// foldable pair exists or it is already folded. Records nothing on the undo
    /// stack — folds are view state.
    pub fn fold(&mut self, header: BufferRow, last: BufferRow) -> bool {
        match self.opener_of(header.0, last.0) {
            Some(open) => self.folds.fold(open),
            None => false,
        }
    }

    /// Unfold the fold whose header (opener row) is `header`. Returns whether one
    /// was removed.
    pub fn unfold(&mut self, header: BufferRow) -> bool {
        match self.folded_opener_on(header.0) {
            Some(open) => self.folds.unfold(open),
            None => false,
        }
    }

    /// Fold if `header` has no active fold, else unfold it (the gutter-chevron /
    /// `Ctrl+Shift+[`,`]` action).
    pub fn toggle_fold(&mut self, header: BufferRow, last: BufferRow) -> bool {
        match self.folded_opener_on(header.0) {
            Some(open) => self.folds.unfold(open),
            None => self.fold(header, last),
        }
    }

    /// The opening-bracket offset of the collapsed fold whose opener sits on
    /// buffer row `header`, if any. Openers are offset-sorted, so this is the
    /// first fold at/after the row start that still lies on the row — an O(log)
    /// probe, so a gutter unfold click never scans every fold.
    fn folded_opener_on(&self, header: u32) -> Option<u32> {
        let start = self.buffer.point_to_offset(Point::new(header, 0));
        let end = if header + 1 >= self.buffer.line_count() {
            self.buffer.len() + 1 // sentinel: an opener at the final byte is on the row
        } else {
            self.buffer.point_to_offset(Point::new(header + 1, 0))
        };
        self.folds.first_at_or_after(start).filter(|&o| o < end)
    }

    /// The opening-bracket offset of the foldable pair spanning rows
    /// `(header, last)`, if one exists.
    fn opener_of(&self, header: u32, last: u32) -> Option<u32> {
        // A pair headed on `header` has its opener on that row — window there.
        self.foldable_pairs_in_rows(header..header + 1)
            .into_iter()
            .find(|&(_, _, h, l)| h == header && l == last)
            .map(|(o, ..)| o)
    }

    /// Every foldable bracket pair as `(open_offset, close_offset, header_row,
    /// last_row)`, in document order — the bracket-anchored source for folds. A
    /// pair is foldable when it has a *non-empty* interior: multi-line (a block)
    /// or single-line with something between the brackets (an inline fold).
    fn foldable_pairs(&self) -> Vec<(u32, u32, u32, u32)> {
        self.foldable_pairs_from(&self.brackets.all())
    }

    /// `Document::foldable_pairs` restricted to pairs whose **opening bracket**
    /// lies on a buffer row in `rows` — the windowed twin for the fold mouse
    /// hot paths (gutter hover/click, Ctrl+hover boxes), so they cost
    /// O(brackets in the window) instead of a whole-document scan per
    /// mouse-move on a large document. Since a pair's `header` is
    /// its opener's row, this returns exactly the pairs headed in `rows` (their
    /// closer may lie far below — that offset is carried in the bracket, no
    /// second scan). A pair *enclosing* `rows` from above is intentionally not
    /// returned (its opener is elsewhere); callers that only test `header ==
    /// row` (block_opener_on_row, the gutter chevron) are exact, and the
    /// Ctrl+hover affordance widens `rows` with a slack for near-enclosing
    /// blocks.
    ///
    /// Public because the widget's plain-hover chip hit-test
    /// (`collapsed_chip_at`) needs exactly this: every foldable pair — inline
    /// `[ … ]` and block `{ … }` alike — headed on the pointer's single row, so
    /// it can test that row's chips instead of scanning every fold in the
    /// document (an O(folds) `HeaderLayout` build per frame at scale).
    pub fn foldable_pairs_in_rows(&self, rows: Range<u32>) -> Vec<(u32, u32, u32, u32)> {
        let start = self.buffer.point_to_offset(Point::new(rows.start, 0));
        let end = if rows.end >= self.buffer.line_count() {
            self.buffer.len() + 1 // sentinel: a bracket at the final byte is inside
        } else {
            self.buffer.point_to_offset(Point::new(rows.end, 0))
        };
        self.foldable_pairs_from(&self.brackets.in_range(start..end))
    }

    /// The shared body of `Document::foldable_pairs` and its windowed twin.
    fn foldable_pairs_from(&self, brackets: &[crate::bracket::Bracket]) -> Vec<(u32, u32, u32, u32)> {
        brackets
            .iter()
            .filter_map(|b| {
                let close = b.partner.filter(|_| b.open)?;
                // The shared foldability rule: a non-empty interior.
                crate::row_layout::pair_has_interior(b.offset, close).then(|| {
                    let header = self.buffer.offset_to_point(b.offset).row;
                    let last = self.buffer.offset_to_point(close).row;
                    (b.offset, close, header, last)
                })
            })
            .collect()
    }

    /// Toggle the fold whose opening bracket is at byte offset `opener` (a
    /// gutter-chevron click or `Ctrl+Shift+[`/`]`). Returns whether it changed.
    /// When it *folds*, any caret the fold would hide is ejected to the gap's
    /// entry edge so it never sits on collapsed text: an inline pair
    /// pulls it to `opener+1` (just after `[`), a block to the header line's
    /// end (just before the `…` placeholder).
    pub fn toggle_fold_opener(&mut self, opener: u32) -> bool {
        let changed = self.folds.toggle(opener);
        if changed {
            // A fold toggle is a gesture that can MOVE carets (the ejection
            // below), so it takes the same boundary as every selection verb:
            // seal the undo group, drop the box anchor / expand ladder /
            // auto-close provenance — otherwise a stale expand ladder could
            // restore a caret into the collapsed fold.
            self.reset_transient();
        }
        if changed && self.folds.is_folded(opener) {
            if let Some(close) = self.brackets.foldable_partner(opener) {
                let single_line = self.buffer.offset_to_point(opener).row == self.buffer.offset_to_point(close).row;
                if single_line {
                    // The shared gap rule: a caret strictly inside the
                    // now-hidden interior pulls out to the left landable edge.
                    self.selections.map_each(|s| {
                        if crate::row_layout::gap_hides_caret(opener, close, s.head()) {
                            s.move_to_caret(crate::row_layout::gap_left_edge(opener));
                        }
                    });
                } else {
                    // Block: eject any caret the fold just hid to the gap's
                    // entry edge — the header line's end, just before the `…`
                    // placeholder (the block analog of the inline pull-out
                    // above), so typing can never edit invisible text. "Hidden"
                    // comes from the one owner: `display_position` is `None`
                    // exactly for offsets in a fold's gap.
                    // The one tab-width owner (hoisted above the destructure that
                    // borrows the fields — `self.tab_size()` can't be called after).
                    let tab = self.tab_size();
                    let Self { selections, buffer, folds, brackets, .. } = self;
                    let fold_map = crate::fold_map::FoldMap::new(folds, brackets, buffer);
                    let header = buffer.offset_to_point(opener).row;
                    let entry = buffer.point_to_offset(Point::new(header, buffer.line_len(header)));
                    selections.map_each(|s| {
                        let inside = s.head() > opener && s.head() < close;
                        if inside && fold_map.display_position(buffer, s.head(), tab).is_none() {
                            s.move_to_caret(entry);
                        }
                    });
                }
            }
        }
        changed
    }

    /// The document's tab-stop width — the ONE owner every consumer (core
    /// projections and the widget's pixel layer alike) consults. Fixed at the
    /// [`crate::display_map::default_tab_size`] for now; when it becomes
    /// configurable, the change is one whole-document edit and no
    /// caller-side constant exists to drift.
    #[must_use]
    pub fn tab_size(&self) -> u32 {
        crate::display_map::default_tab_size()
    }

    /// The opener offset of the widest *multi-line* pair whose header is buffer
    /// row `row` — the block a gutter chevron on that row folds.
    #[must_use]
    pub fn block_opener_on_row(&self, row: u32) -> Option<u32> {
        // Windowed to the row's brackets (a pair headed on `row` has its opener
        // on `row`) — O(brackets on the row), not a whole-document scan per
        // gutter click.
        self.foldable_pairs_in_rows(row..row + 1)
            .into_iter()
            .filter(|&(_, _, h, l)| h == row && l > h)
            .max_by_key(|&(_, _, h, l)| l - h)
            .map(|(o, ..)| o)
    }

    /// The opener of the innermost foldable pair enclosing **or touching** the
    /// primary caret whose collapsed state matches `want_folded` — the
    /// `Ctrl+Shift+[` (fold, `false`) / `Ctrl+Shift+]` (unfold, `true`) target.
    /// A caret directly before the opening bracket or after the closing one
    /// counts (the bracket-highlight adjacency). Includes single-line inline
    /// pairs, disambiguated by the caret's byte offset (rows can't tell two
    /// `[..]` apart).
    #[must_use]
    pub fn fold_opener_at_caret(&self, want_folded: bool) -> Option<u32> {
        self.fold_opener_at(self.selections.newest().head(), want_folded)
    }

    /// The innermost foldable pair enclosing or touching `caret` whose collapsed
    /// state matches `want_folded` — the per-caret core of the fold chords.
    /// Costs O(distance to the innermost target), NOT O(leftward siblings): a
    /// touching pair (the innermost possible) is found by an O(log) lookup, and
    /// otherwise the enclosing walk early-stops at the first match — so
    /// [`Self::fold_at_carets`] over N carets stays O(N · local), not O(N²).
    fn fold_opener_at(&self, caret: u32, want_folded: bool) -> Option<u32> {
        let is_target = |open: u32, close: u32| {
            crate::row_layout::pair_has_interior(open, close)
                && self.folds.is_folded(open) == want_folded
        };
        // Touching pairs are the innermost possible (an encloser is strictly
        // larger, since it contains the touch): the caret ON a foldable opener,
        // or just past a foldable closer. Pick the smaller-extent match.
        let touch_left = self.brackets.foldable_partner(caret).map(|close| (caret, close));
        let touch_right = caret
            .checked_sub(1)
            .and_then(|o| self.brackets.at(o))
            .filter(|b| !b.open)
            .and_then(|b| b.partner.map(|open| (open, b.offset)));
        if let Some((open, _)) = [touch_left, touch_right]
            .into_iter()
            .flatten()
            .filter(|&(o, c)| is_target(o, c))
            .min_by_key(|&(o, c)| c - o)
        {
            return Some(open);
        }
        // Otherwise the innermost ENCLOSING foldable pair, early-stopping.
        self.brackets.innermost_enclosing_where(caret, is_target)
    }

    /// Fold (`unfold == false`) or unfold (`unfold == true`) the innermost
    /// foldable block enclosing or touching EACH caret — the multi-cursor
    /// `Ctrl+Shift+[` / `Ctrl+Shift+]`. Distinct target blocks act once (two
    /// carets in one block don't cancel each other), and every opener is resolved
    /// BEFORE any toggle — safe because folding moves no byte offset. Returns
    /// whether anything changed.
    pub fn fold_at_carets(&mut self, unfold: bool) -> bool {
        let mut openers: Vec<u32> = self
            .selections
            .all()
            .iter()
            .filter_map(|s| self.fold_opener_at(s.head(), unfold))
            .collect();
        openers.sort_unstable();
        openers.dedup();
        if openers.is_empty() {
            return false;
        }
        // A fold gesture is a boundary like every selection verb (seal undo, drop
        // the box anchor / expand ladder / auto-close provenance) — ONCE. Then
        // toggle ALL folds in one batch and eject carets in one pass. Per-fold
        // `toggle_fold_opener` is O(folds) each (a sorted-Vec insert plus a
        // FoldMap rebuild), so folding one at a time would make a document-scale
        // "collapse all" O(folds²); the batch keeps it linear.
        self.reset_transient();
        if unfold {
            self.folds.unfold_all(&openers);
        } else {
            self.folds.fold_all(openers.iter().copied());
        }
        self.eject_hidden_carets();
        true
    }

    /// Pull every caret out of a newly-hidden fold gap to the fold's entry edge
    /// — the batched form of [`Self::toggle_fold_opener`]'s ejection: ONE
    /// FoldMap build, each caret probed in O(log folds).
    fn eject_hidden_carets(&mut self) {
        let Self { selections, buffer, folds, brackets, .. } = self;
        if folds.is_empty() {
            return;
        }
        let fold_map = crate::fold_map::FoldMap::new(folds, brackets, buffer);
        selections.map_each(|s| {
            if let Some(entry) = fold_map.entry_edge_if_hidden(buffer, s.head()) {
                s.move_to_caret(entry);
            }
        });
    }

    /// Every *collapsible* bracket pair — foldable (non-empty interior) and not
    /// already collapsed — as `(open, close, header, last)`, document order. The
    /// candidate set for the Ctrl+hover affordance: a pair already folded is
    /// excluded (there is nothing left to collapse). Includes both multi-line
    /// blocks and single-line inline pairs, so one gesture reaches either.
    #[must_use]
    pub fn collapsible_pairs(&self) -> Vec<(u32, u32, u32, u32)> {
        self.foldable_pairs().into_iter().filter(|&(open, ..)| !self.folds.is_folded(open)).collect()
    }

    /// [`Document::collapsible_pairs`] restricted to pairs headed on buffer rows
    /// in `rows` — the windowed query the Ctrl+hover affordance uses per
    /// mouse-move instead of a whole-document scan on a large document.
    /// The caller widens `rows` with a slack so a pair whose header sits just
    /// above the viewport still arms; a block taller than that slack enclosing
    /// the pointer is the accepted miss (fold it from its header chevron or the
    /// `Ctrl+Shift+[` chord instead).
    #[must_use]
    pub fn collapsible_pairs_in_rows(&self, rows: Range<u32>) -> Vec<(u32, u32, u32, u32)> {
        self.foldable_pairs_in_rows(rows)
            .into_iter()
            .filter(|&(open, ..)| !self.folds.is_folded(open))
            .collect()
    }

    /// The opener of the innermost collapsible pair whose interior contains byte
    /// `offset` (`open < offset < close`) — the Ctrl+hover / Ctrl+Click target.
    /// Innermost = smallest byte span, so a pointer inside nested
    /// collapsibles resolves to the tightest one (an inline array over its
    /// enclosing block). `None` when the offset sits in no collapsible interior.
    #[must_use]
    pub fn collapsible_at(&self, offset: u32) -> Option<u32> {
        self.collapsible_pairs()
            .into_iter()
            .filter(|&(open, close, _, _)| open < offset && offset < close)
            .min_by_key(|&(open, close, _, _)| close - open)
            .map(|(o, ..)| o)
    }

    /// Drop collapsed folds whose opener is no longer a live foldable open bracket
    /// — e.g. an edit deleted or unmatched a block's brackets. Called from the
    /// edit / undo / redo paths once brackets are re-matched. Because a fold's
    /// extent is re-derived from the live brackets (never stored), there is nothing
    /// to heal — a fold either still opens a foldable pair or it is dropped. No-op
    /// when nothing is folded.
    fn reconcile_folds(&mut self) {
        if self.folds.is_empty() {
            return;
        }
        // Check each folded opener DIRECTLY (a matched open bracket with a
        // non-empty interior) via an O(log) bracket lookup, instead of
        // building the whole-document `foldable_pairs()`. Reconcile runs on
        // every commit, so this keeps a keystroke while a fold is active off an
        // O(brackets) whole-document scan.
        let brackets = &self.brackets;
        self.folds.reconcile(|o| {
            brackets
                .foldable_partner(o)
                .is_some_and(|close| crate::row_layout::pair_has_interior(o, close))
        });
    }

    /// Test hook: run the whole-set reconcile (the undo/redo path's version) so a
    /// test can assert the edit-path windowed reconcile left nothing for it to do.
    #[cfg(test)]
    pub(crate) fn force_whole_reconcile(&mut self) {
        self.reconcile_folds();
    }

    /// Windowed reconcile for the edit path: drop only folds the edit's re-matched
    /// `region` could have broken. A fold's foldability changes only if its opener
    /// or closer bracket was re-scanned, and every re-scanned bracket lies in the
    /// bracket engine's replayed rows — so an affected fold's `[opener, closer]`
    /// must INTERSECT that span. `region` already extends its left edge down to the
    /// outermost enclosing opener (the seed stack — see `BracketOps::reconcile_lo`),
    /// so every such fold has its OPENER in `region`: a single binary-searched
    /// window, no separate enclosing walk. This is [`reconcile_folds`]'s O(local)
    /// twin — the whole-set version stays on the undo/redo path, which spans many
    /// steps and so has no single region.
    fn reconcile_folds_in(&mut self, region: core::ops::Range<u32>) {
        if self.folds.is_empty() {
            return;
        }
        let candidates: Vec<u32> = self.folds.openers_in(region);
        let brackets = &self.brackets;
        self.folds.reconcile_only(&candidates, |o| {
            brackets
                .foldable_partner(o)
                .is_some_and(|close| crate::row_layout::pair_has_interior(o, close))
        });
    }

    /// Every foldable `(header, last)` row range — the source for gutter chevrons
    /// and fold commands. Derived from the matched-bracket pass: each *multi-line*
    /// matched pair (`{}`/`()`/`[]`) becomes a range from its opener row to its
    /// closer row. At most one range per header row (the widest), sorted by header.
    #[must_use]
    pub fn foldable_ranges(&self) -> Vec<(BufferRow, BufferRow)> {
        let mut ranges: Vec<(BufferRow, BufferRow)> = self
            .brackets
            .all()
            .iter()
            .filter_map(|br| {
                let close = br.partner.filter(|_| br.open)?;
                let header = self.buffer.offset_to_point(br.offset).row;
                let last = self.buffer.offset_to_point(close).row;
                (last > header).then_some((BufferRow(header), BufferRow(last)))
            })
            .collect();
        // Keep the widest range per header row (largest `last`).
        ranges.sort_by_key(|r| (r.0, core::cmp::Reverse(r.1)));
        ranges.dedup_by_key(|r| r.0);
        ranges
    }

    /// [`Document::foldable_ranges`] restricted to headers on buffer rows in
    /// `rows` — the gutter chevrons' per-frame query: two partition points over
    /// the sorted bracket Vec ([`Brackets::in_range`]) instead of a
    /// whole-document scan per frame.
    /// Same widest-per-header dedup; hidden in-window headers are the
    /// caller's to cull (its `visible_y` already does).
    #[must_use]
    pub fn foldable_ranges_in_rows(&self, rows: Range<u32>) -> Vec<(BufferRow, BufferRow)> {
        let start = self.buffer.point_to_offset(Point::new(rows.start, 0));
        let end = if rows.end >= self.buffer.line_count() {
            self.buffer.len() + 1 // sentinel: a bracket at the final byte is inside
        } else {
            self.buffer.point_to_offset(Point::new(rows.end, 0))
        };
        let mut ranges: Vec<(BufferRow, BufferRow)> = self
            .brackets
            .in_range_iter(start..end)
            .filter_map(|br| {
                let close = br.partner.filter(|_| br.open)?;
                let header = self.buffer.offset_to_point(br.offset).row;
                let last = self.buffer.offset_to_point(close).row;
                (last > header).then_some((BufferRow(header), BufferRow(last)))
            })
            .collect();
        ranges.sort_by_key(|r| (r.0, core::cmp::Reverse(r.1)));
        ranges.dedup_by_key(|r| r.0);
        ranges
    }

    /// Bring the highlight cache current up to buffer line `target` (a viewport's
    /// last line). Lazy and incremental — end-state convergence bounds the work
    /// to the edited lines. No-op without a highlighter.
    pub fn tokenize_highlight(&mut self, target: u32) {
        // `self.highlight` and `self.buffer` are disjoint fields, so both
        // borrow. The buffer is handed in as a line ACCESSOR — never an
        // all-lines Vec — and each call is budget-capped; the app's idle sweep
        // resumes at [`Document::highlight_frontier`] until convergence.
        let buffer = &self.buffer;
        let Some(cache) = self.highlight.as_mut() else { return };
        cache.tokenize_until(target, crate::highlight::HIGHLIGHT_MAX_LINES_PER_CALL, |r| {
            buffer.line(r)
        });
    }

    /// The next row highlight work would touch — a dirty row, or a window
    /// row awaiting a refill after [`Document::set_highlight_window`] moved
    /// the retention window (highlight virtualization). `None` when there is
    /// nothing to do — or when no grammar is injected. The app's idle sweep
    /// polls this to drive budgeted [`Document::tokenize_highlight`] calls
    /// and to stop when idle (an idle document does zero highlight work).
    #[must_use]
    pub fn highlight_frontier(&self) -> Option<u32> {
        self.highlight.as_ref().and_then(HighlightCache::pending)
    }

    /// Aim the highlight retention window at `rows` (the visible buffer rows;
    /// the cache pads by [`crate::highlight::HIGHLIGHT_WINDOW_SLACK`] and
    /// evicts outside it). Spans and per-line states are retained only there;
    /// elsewhere sparse checkpoints keep every row re-derivable, so a fully
    /// swept document holds memory proportional to the window, not the line
    /// count. Call on viewport change, then drive
    /// [`Document::tokenize_highlight`] as usual. No-op without a grammar.
    pub fn set_highlight_window(&mut self, rows: Range<u32>) {
        if let Some(cache) = self.highlight.as_mut() {
            cache.set_window(rows);
        }
    }

    /// Cached highlight spans for a buffer row — `None` if untokenized or no
    /// highlighter is attached (the renderer falls back to the plain text color).
    #[must_use]
    pub fn highlight_line_spans(&self, row: u32) -> Option<&[HighlightSpan]> {
        self.highlight.as_ref().and_then(|c| c.line_spans(row))
    }

    /// A `Send + Sync` handle to the highlighter's grammar + theme, for the
    /// app's off-thread parallel/speculative sweep. `None` without a grammar.
    /// Pair with [`Document::snapshot`] (an O(1) rope clone) and
    /// [`crate::tokenize_segment`] on a worker, then feed results back through
    /// [`Document::absorb_highlight`].
    #[must_use]
    pub fn highlight_engine(&self) -> Option<HighlightEngine> {
        self.highlight.as_ref().map(HighlightCache::engine)
    }

    /// Ingest a segment tokenized off-thread (the parallel/speculative sweep).
    /// Returns `false` — absorbing **nothing** — if `revision` no longer
    /// matches the document (an edit landed since the snapshot the segment was
    /// computed from; the app drops the stale result and re-dispatches). On a
    /// match, forwards to `HighlightCache::absorb`:
    ///
    /// - `verified` — the coordinator chained this segment from row 0, so its
    ///   start state is TRUE: its checkpoints merge, its window spans/states are
    ///   written, and its rows leave the dirty frontier (replacing the
    ///   foreground walk).
    /// - `!verified` — viewport speculation from a guessed fresh start: only
    ///   still-dirty window rows get its spans, no checkpoints are planted, and
    ///   the frontier still re-verifies those rows per-line (converging in O(1)
    ///   if the guess was right).
    ///
    /// Already-absorbed work is never lost to a later edit: it lives in the
    /// cache and rides `rebase_views`' `on_commit` splices like every other
    /// derived fact — only *in-flight* results are dropped by the revision
    /// check.
    pub fn absorb_highlight(
        &mut self,
        revision: Revision,
        seg: crate::SegmentTokens,
        verified: bool,
    ) -> bool {
        if revision != self.buffer.revision() {
            return false;
        }
        let Some(cache) = self.highlight.as_mut() else { return false };
        cache.absorb(seg, verified);
        true
    }

    /// The matched brackets — kept current on every edit. Drives
    /// bracket-pair colorization and the matching-bracket highlight.
    #[must_use]
    pub fn brackets(&self) -> &Brackets {
        &self.brackets
    }

    /// Publish a diagnostic set from the app's debounced compile loop.
    ///
    /// The compiler ran against the snapshot at `revision`; if the buffer has
    /// since moved on this returns [`DiagnosticsOutcome::Stale`] having installed
    /// nothing — the previous squiggles keep riding edits via stickiness until
    /// the next publish (no stale set is placed). On a current revision the whole
    /// diagnostic set is replaced wholesale; spans are first clipped to the buffer
    /// length. Zero-width spans survive — the squiggle enforces its one-cell
    /// minimum at draw time.
    pub fn set_diagnostics(&mut self, revision: Revision, diags: Vec<Diagnostic>) -> DiagnosticsOutcome {
        let current = self.buffer.revision();
        if revision != current {
            return DiagnosticsOutcome::Stale { current: current.0 };
        }
        let len = self.buffer.len();
        let clipped: Vec<Diagnostic> = diags
            .into_iter()
            .map(|mut d| {
                let (s, e) = (d.span.start.min(len), d.span.end.min(len));
                d.span = s.min(e)..e; // clip to buffer, never inverted
                d
            })
            .collect();
        self.decorations.set_diagnostics(current.0, current.0, clipped)
    }

    /// Diagnostics overlapping the byte `range`, for squiggle rendering:
    /// `(span, severity, message)`, position and content reunited from the store,
    /// in ascending `(start, id)` order. The renderer clips to its visible rows.
    pub fn diagnostics_in(
        &self,
        range: Range<u32>,
    ) -> impl Iterator<Item = (Range<u32>, Severity, std::sync::Arc<str>)> + '_ {
        // The store yields owned tracked ranges (its delta-gap encoding has no
        // absolute range to borrow), so the message rides out as its shared
        // `Arc<str>` — an owner bump, not a copy.
        self.decorations.decorations_in(range).filter_map(|r| {
            let TrackedRange { range, kind, .. } = r;
            match kind {
                DecorationKind::Diagnostic { severity, message, .. } => {
                    Some((range, severity, message))
                }
                _ => None,
            }
        })
    }

    /// Set (or clear with `None`) the find query, scanning synchronously.
    /// Never scrolls and drops the active match; the app calls [`find_next`] after
    /// if it wants reveal-as-you-type. `now_ms` is the injected clock. This is the
    /// one whole-document find op — every subsequent edit repairs the set
    /// incrementally through the commit mover.
    ///
    /// [`find_next`]: Document::find_next
    pub fn set_find_query(&mut self, query: Option<FindQuery>, now_ms: u64) {
        self.find.set_query(query, &self.buffer, &mut self.decorations, now_ms);
    }

    /// The active find query, or `None` if find is idle.
    #[must_use]
    pub fn find_query(&self) -> Option<&FindQuery> {
        self.find.query()
    }

    /// Restrict find to a byte range — **find in selection** — or clear the
    /// restriction with `None`. Re-scans synchronously: the scope is part of
    /// what matches, not a display filter, so every consumer (navigation, the
    /// N-of-M count, [`replace_all`](Document::replace_all)) honors it without
    /// knowing it exists.
    ///
    /// The scope then rides every edit's patch like any other derived position,
    /// growing when text is typed at its edges and dropping if it collapses. An
    /// empty range clears it rather than pinning zero matches.
    pub fn set_find_scope(&mut self, scope: Option<Range<u32>>, now_ms: u64) {
        self.find.set_scope(scope, &self.buffer, &mut self.decorations, now_ms);
    }

    /// The live find-in-selection scope, if any — for the view to shade it.
    #[must_use]
    pub fn find_scope(&self) -> Option<Range<u32>> {
        self.find.scope()
    }

    /// Why the current find pattern does not parse, if it does not — for the
    /// find bar to show.
    ///
    /// This is not an error path: with the regex option on, every prefix of a
    /// pattern being typed (`(`, `[a-`) is invalid, so find simply reports zero
    /// matches and carries the reason until the pattern parses again.
    #[must_use]
    pub fn find_pattern_error(&self) -> Option<&str> {
        self.find.pattern_error()
    }

    /// Whole-word occurrences of the word under the newest caret WITHIN the
    /// byte range `within` — the passive occurrence wash (the reading aid for
    /// tracing a PID/label through a script). The window keeps this per-frame
    /// query O(viewport): the widget passes its visible byte range; tests pass
    /// `0..u32::MAX` (the scan clamps). Every
    /// match must be a WHOLE word, judged by the same
    /// `movement::surrounding_word` rule that seeds Ctrl+D — checked against
    /// the full buffer, so a window edge can never fake a word boundary. Empty
    /// when the newest selection is non-empty or the caret is off-word. A pure
    /// per-frame query (the `FoldMap` precedent): no tracked state, nothing to
    /// rebase on the undo path.
    #[must_use]
    pub fn caret_word_occurrences(&self, within: Range<u32>) -> Vec<Range<u32>> {
        let newest = self.selections.newest();
        if !newest.is_empty() {
            return Vec::new();
        }
        let Some((ws, we)) = movement::surrounding_word(&self.buffer, newest.head()) else {
            return Vec::new();
        };
        if ws == we {
            return Vec::new();
        }
        let word = self.buffer.slice(ws..we);
        let end = within.end.min(self.buffer.len());
        let base = within.start.min(end);
        let hay = self.buffer.slice(base..end);
        let mut out = Vec::new();
        let mut from = 0usize;
        while let Some(i) = hay[from..].find(&*word) {
            let start = base + (from + i) as u32;
            let match_end = start + word.len() as u32;
            // Whole-word: the match must BE its own surrounding word.
            if movement::surrounding_word(&self.buffer, start) == Some((start, match_end)) {
                out.push(start..match_end);
            }
            from = from + i + word.len();
        }
        out
    }

    /// How many matches the current query has — the *live* count. The store IS
    /// the match set: there is no shadow handle list to disagree with it, so this
    /// is the O(1) root-summary `find_count`, not an O(matches) walk. Empty
    /// matches never persist (`FindMatch` is `EmptyPolicy::Drop`), so the count
    /// equals the rendered `start < end` set.
    #[must_use]
    pub fn find_match_count(&self) -> usize {
        self.decorations.find_count()
    }

    /// The active match's position among the live matches, in document order, if
    /// one is active and still present — consistent with
    /// [`find_match_count`](Document::find_match_count) for an "N of M" display.
    /// O(log M) from the active handle's tracked start: its rank is the number of
    /// finds starting before it, not an O(M) `position` scan.
    #[must_use]
    pub fn active_find_match(&self) -> Option<u32> {
        self.find.active_start().map(|s| self.decorations.find_rank_before(s) as u32)
    }

    /// Find matches overlapping the byte `range`, for highlight rendering:
    /// `(span, is_active)`, in ascending order. Collapsed (empty) matches are
    /// skipped — an edit may zero one before the next re-scan purges it.
    pub fn find_matches_in(&self, range: Range<u32>) -> impl Iterator<Item = (Range<u32>, bool)> + '_ {
        let active = self.find.active_id();
        self.decorations.decorations_in(range).filter_map(move |r| match &r.kind {
            DecorationKind::FindMatch if r.range.start < r.range.end => {
                Some((r.range.clone(), Some(r.id) == active))
            }
            _ => None,
        })
    }

    /// Fill the scrollbar-overview lanes for the ascending byte-offset bucket
    /// `bounds`. The widget passes the `P + 1` boundaries it gets by inverting its
    /// own `round(y)` pixel map, so bucket `b` holds exactly the decorations whose
    /// mark lands on track pixel `b`. Per bucket `b`: `sev_out[b]` = `(encoded max
    /// Diagnostic severity, byte offset of the first severest one)` — `(0, 0)`
    /// when empty; `find_out[b]` = the start offset of the first
    /// [`DecorationKind::FindMatch`], or `None`. Both lanes are the O(P + log M)
    /// summary reduce, never a per-frame whole-store walk.
    /// `overview_reduce_equals_linear_scan` (decorations.rs) is the correctness
    /// authority.
    pub fn overview_marks(
        &self,
        bounds: &[u32],
        sev_out: &mut Vec<(u8, u32)>,
        find_out: &mut Vec<Option<u32>>,
    ) {
        self.decorations.diagnostic_overview(bounds, sev_out);
        self.decorations.find_overview(bounds, find_out);
    }

    /// Refill a **capped** find set, debounced — driven from the widget's
    /// `update()` each event. Returns whether it scanned.
    ///
    /// The match set is repaired eagerly at every commit, so there is no stale
    /// set to rescan; the only remaining job is re-growing a capped set's
    /// coverage after matches inside it died. Anything else is a no-op — idle
    /// documents do zero work here.
    pub fn maybe_rescan_find(&mut self, now_ms: u64) -> bool {
        self.find.maybe_rescan(&self.buffer, &mut self.decorations, now_ms)
    }

    /// Select the next find match from the caret, wrapping: sets the selection to
    /// the match (head at end), seals the undo group, and returns the match range
    /// for the widget to reveal (autoscroll). `None` when there are no matches.
    pub fn find_next(&mut self, now_ms: u64) -> Option<Range<u32>> {
        self.step_find(true, now_ms)
    }

    /// Select the previous find match from the caret, wrapping — the
    /// [`find_next`](Document::find_next) mirror.
    pub fn find_prev(&mut self, now_ms: u64) -> Option<Range<u32>> {
        self.step_find(false, now_ms)
    }

    /// Replace the active find match with `replacement`, then advance to the
    /// next match, wrapping — the Replace verb. Returns the match left selected,
    /// or `None` when there is nowhere to go.
    ///
    /// Replaces only when the newest selection IS the active match — the state
    /// [`find_next`](Document::find_next) leaves behind. Otherwise it merely
    /// navigates, so the first press selects and the second replaces: text is
    /// never overwritten before it has been shown. One transaction ⇒ one undo
    /// step, and the match set needs no resync — it rides the commit through the
    /// shared view-rebase mover like any other edit.
    /// `preserve_case` (the replace bar's `AB` toggle) re-cases the replacement
    /// to the match it lands on — an ALL-CAPS match takes an upper-cased
    /// replacement, a Capitalized match a capitalized one.
    pub fn replace_next(
        &mut self,
        replacement: &str,
        preserve_case: bool,
        now_ms: u64,
    ) -> Option<Range<u32>> {
        self.find.maybe_rescan(&self.buffer, &mut self.decorations, now_ms);
        let extent = {
            let sel = self.selections.newest();
            sel.start()..sel.end()
        };
        // Armed only when the selection IS the active match. An empty selection
        // can never be one (`FindMatch` is `EmptyPolicy::Drop`), but the guard
        // states it rather than relying on that.
        let armed = !extent.is_empty()
            && self.find.active_range(&self.decorations).is_some_and(|r| r == extent);
        if armed {
            // The replacement is resolved through the one owner of that rule
            // (regex templating + preserve-case), bounded to this match.
            let text = self.find.query().cloned().map_or_else(
                || replacement.to_string(),
                |q| {
                    crate::find::replacements(
                        &self.buffer,
                        &q,
                        extent.clone(),
                        replacement,
                        preserve_case,
                    )
                    .into_iter()
                    .next()
                    .map_or_else(|| replacement.to_string(), |(_, t)| t)
                },
            );
            if self.edit(vec![EditOp::new(extent, text)]).is_err() {
                return None; // a rejected transaction must not read as a replace
            }
        }
        self.step_find(true, now_ms)
    }

    /// Replace **every** match of the current query with `replacement` in one
    /// transaction — the Replace All verb. Returns how many were replaced.
    ///
    /// **One transaction ⇒ one undo step.** Matches are disjoint by
    /// construction, so the batch satisfies the engine's no-overlap rule for
    /// free — [`TransactionError::Overlap`] cannot fire here.
    ///
    /// Scans the document itself rather than reading the live match set: that
    /// set is capped at [`FIND_MATCH_CAP`] and is only a *prefix of the
    /// document*, so an "all" built from it would silently stop at the cap and
    /// leave the tail untouched. The scan is therefore whole-document — the same
    /// class as [`set_find_query`](Document::set_find_query), and legitimate for
    /// the same reason: a discrete user action, never a keystroke.
    ///
    /// Cost, honestly: one [`EditOp`] per match, each owning its own copy of
    /// `replacement`, so a query with millions of hits allocates proportionally
    /// before it commits. Bounded by the match count and paid once per press.
    ///
    /// `preserve_case` (the replace bar's `AB` toggle) re-cases each replacement
    /// to the match it lands on — an ALL-CAPS match takes an upper-cased
    /// replacement, a Capitalized match a capitalized one.
    ///
    /// [`FIND_MATCH_CAP`]: crate::find::FIND_MATCH_CAP
    pub fn replace_all(&mut self, replacement: &str, preserve_case: bool) -> usize {
        let Some(query) = self.find.query().cloned() else {
            return 0; // find is idle
        };
        // Scoped find replaces only inside its range — the scan bound is the one
        // place that has to know, so every other path inherits it.
        let within = self.find.scope().unwrap_or(0..self.buffer.len());
        let plan =
            crate::find::replacements(&self.buffer, &query, within, replacement, preserve_case);
        if plan.is_empty() {
            return 0; // no matches ⇒ no transaction, no undo step
        }
        let n = plan.len();
        let ops = plan.into_iter().map(|(r, t)| EditOp::new(r, t)).collect();
        if self.edit(ops).is_err() {
            return 0;
        }
        n
    }

    /// Shared find-navigation body.
    fn step_find(&mut self, forward: bool, now_ms: u64) -> Option<Range<u32>> {
        // The set is always current at every commit; the one freshness concern
        // left is a capped set with room to refill, which the debounced refill
        // handles opportunistically before navigating.
        self.find.maybe_rescan(&self.buffer, &mut self.decorations, now_ms);
        let sel = self.selections.newest();
        let (head, extent) = (sel.head(), sel.start()..sel.end());
        let found = if forward {
            self.find.find_next(head, extent, &self.decorations)
        } else {
            self.find.find_prev(head, extent, &self.decorations)
        };
        if let Some(range) = &found {
            // Set the selection to the match, head at its end.
            self.selections
                .set_single(Selection::from_anchor(SelectionId(0), range.start, range.end));
            self.reset_transient(); // clears gesture state + seals (a jump is a boundary)
            // A match inside a collapsed fold unfolds its chain first — the
            // reveal below must land on a VISIBLE position.
            self.unfold_to_reveal(range.start);
            self.unfold_to_reveal(range.end);
            self.request_reveal(RevealMode::Center); // find navigation centers
        }
        found
    }

    /// F8 / Shift+F8: select the next/previous diagnostic from the
    /// newest selection's start, wrapping — the compile-loop navigation:
    /// jump, read the message (hover), fix, F8 again. Selects the
    /// diagnostic's span, expands any fold hiding it, and bumps the reveal
    /// generation (a jump-class action, like find navigation). `None` — and
    /// no movement — when the document has no live diagnostics.
    pub fn next_diagnostic(&mut self, forward: bool) -> Option<Range<u32>> {
        // Walk in (start, end) lexicographic order from the CURRENT selection's
        // span, so two diagnostics sharing a start offset are both reachable —
        // a plain `start >` comparison would loop forever between them.
        let sel = self.selections.newest();
        let anchor = (sel.start(), sel.end());
        let mut all: Vec<Range<u32>> = self.diagnostics_in(0..u32::MAX).map(|(r, ..)| r).collect();
        all.sort_by_key(|r| (r.start, r.end));
        let target = if forward {
            all.iter().find(|r| (r.start, r.end) > anchor).or_else(|| all.first())
        } else {
            all.iter().rev().find(|r| (r.start, r.end) < anchor).or_else(|| all.last())
        }?
        .clone();
        self.selections
            .set_single(Selection::from_anchor(SelectionId(0), target.start, target.end));
        self.reset_transient(); // a jump is an undo-group boundary
        self.unfold_to_reveal(target.start);
        self.unfold_to_reveal(target.end);
        self.request_reveal(RevealMode::Center); // diagnostic jumps center
        Some(target)
    }

    /// Unfold every collapsed fold hiding `offset`, innermost-first, until the
    /// offset renders: a jump-class caret placement — find navigation, bracket
    /// jump, select-all-matches — must land on a VISIBLE position. The jump twin
    /// of the edit path's `expand_folds_touched`. Visibility is judged by the one
    /// owner ([`FoldMap::display_position`]); already-visible offsets (a
    /// collapsed tail, a chip edge) unfold nothing.
    fn unfold_to_reveal(&mut self, offset: u32) -> bool {
        let tab = self.tab_size();
        let mut any = false;
        loop {
            let fm = FoldMap::new(&self.folds, &self.brackets, &self.buffer);
            // "Visible" for a JUMP is stricter than "renders somewhere":
            // `display_position` clips a chip-hidden column to the chip's
            // center, so an offset inside a collapsed INLINE fold still gets a
            // position — but the text itself is hidden, and a jump target must
            // be readable, so the gap rule is checked too.
            let renders = fm.display_position(&self.buffer, offset, tab).is_some();
            // Only the inline fold opening just before `offset` can hide it (roots
            // are disjoint) — an O(log) tree probe, not a full-set scan.
            let chip_hidden = fm.inline_fold_before(offset).is_some_and(|f| f.hides_caret_at(offset));
            if renders && !chip_hidden {
                return any;
            }
            // Peel the innermost collapsed pair containing the offset, then
            // re-check — an outer unfold can expose a collapsed inner fold.
            let target = self
                .folds
                .iter()
                .filter_map(|open| self.brackets.foldable_partner(open).map(|c| (open, c)))
                .filter(|&(open, close)| open < offset && offset < close)
                .min_by_key(|&(open, close)| close - open);
            let Some((open, _)) = target else {
                return any; // hidden for a non-fold reason — nothing to peel
            };
            self.folds.unfold(open);
            any = true;
        }
    }

    /// The reveal-request generation. The view autoscrolls to the newest
    /// selection whenever this changes — the bridge that lets app-driven find
    /// navigation scroll a match into view without the widget's input path.
    #[must_use]
    pub fn reveal_seq(&self) -> u64 {
        self.reveal_seq
    }
}

/// Expand every fold whose **hidden** content the committed edit touched:
/// an edit must never change text the user cannot see, so the fold opens to
/// reveal it. Edits in a fold's *visible* parts — a block header's line, its
/// closing tail, just outside an inline pair's brackets — leave it folded; an
/// edit that swallows a fold's brackets entirely is dropped by
/// `reconcile_folds` instead. Shared by the commit path and undo/redo (a free
/// fn for the same borrow reason as `rebase_views`), so undoing into a folded
/// region reveals the reverted text identically.
fn expand_folds_touched(
    folds: &mut FoldSet,
    brackets: &Brackets,
    buffer: &Buffer,
    tab: u32,
    committed: &Committed,
) {
    if folds.is_empty() || committed.patch().is_empty() {
        return;
    }
    // Only a fold that ENCLOSES or TOUCHES an edit endpoint can hide that edit.
    // Their openers lie in `[lo, last]`: `last` (the rightmost endpoint) bounds
    // them on the right since an enclosing opener is `<=` the point, and `lo` —
    // the outermost folded pair enclosing/touching the FIRST endpoint — bounds
    // them on the left (an enclosing fold can open far to the left, but no farther
    // than the first point's outermost encloser). One windowed enclosing walk
    // finds `lo`; the fold set is then a binary-searched slice, so the whole scan
    // is O(folds in the edit span), NOT O(carets · folds-before-each) — a
    // leftward enclosing walk per caret would make select-all → fold → type scale
    // with that product. The `FoldMap` built from just these folds is identical to
    // the full map for these offsets (a non-enclosing fold cannot hide them), so
    // the reveal decision is unchanged.
    let mut pts: Vec<u32> = Vec::with_capacity(committed.patch().edits().len() * 2);
    for e in committed.patch().edits() {
        pts.push(e.new.start);
        pts.push(e.new.end);
    }
    pts.sort_unstable();
    pts.dedup();
    let (first, last) = (pts[0], *pts.last().expect("edits() is non-empty here"));
    let lo = brackets
        .enclosing_or_touching(first)
        .into_iter()
        .filter(|&(o, _)| folds.is_folded(o))
        .map(|(o, _)| o)
        .min()
        .unwrap_or(first);
    // The folded pairs in the window that actually enclose/touch a point —
    // exactly the enclosing/touching set, found by one point probe per windowed
    // fold (`∃ p in [o, c+1]`) rather than a per-caret enclosing walk.
    let candidates: Vec<(u32, u32)> = folds
        .openers_in(lo..last.saturating_add(1))
        .iter()
        .filter_map(|&o| {
            let c = brackets.foldable_partner(o)?;
            let touches =
                pts.partition_point(|&p| p < o) < pts.partition_point(|&p| p <= c.saturating_add(1));
            touches.then_some((o, c))
        })
        .collect();
    if candidates.is_empty() {
        return;
    }
    let mut sub = crate::fold_map::FoldSet::new();
    for &(open, _) in &candidates {
        sub.fold(open);
    }
    let fold_map = crate::fold_map::FoldMap::new(&sub, brackets, buffer);
    // The block test — is any edit endpoint hidden in a collapsed gap? — reads only
    // the point and the map, never the candidate, so all block candidates share one
    // answer; compute it once (O(points · log folds)) instead of re-scanning every
    // edit for every candidate, which would be O(carets²) even when nothing is
    // hidden.
    let any_hidden = pts.iter().any(|&p| fold_map.display_position(buffer, p, tab).is_none());
    // Inline candidates test their own bracket span against the edit STARTS (only a
    // start reveals an inline pair): "∃ start in (opener, close]", binary-searched.
    let mut starts: Vec<u32> = committed.patch().edits().iter().map(|e| e.new.start).collect();
    starts.sort_unstable();
    let touched: Vec<u32> = candidates
        .iter()
        .filter(|&&(opener, close)| {
            let single = buffer.offset_to_point(opener).row == buffer.offset_to_point(close).row;
            if single {
                // Content edited/inserted anywhere inside the pair reveals it; at
                // `[` itself or past `]` doesn't — a start in (opener, close].
                starts.partition_point(|&s| s <= close) > starts.partition_point(|&s| s <= opener)
            } else {
                // Block: `display_position` is `None` exactly in the gap; an edit
                // straddling INTO the gap has an endpoint there, one swallowing the
                // whole fold broke the pair. Candidate-independent (see above).
                any_hidden
            }
        })
        .map(|&(opener, _)| opener)
        .collect();
    for opener in touched {
        folds.unfold(opener);
    }
}

/// The derived views that ride a committed patch — bundled so `rebase_views`
/// and its callers name the set once instead of threading five loose `&mut`s
/// (and so undo/redo can build it from the destructured `Document` fields). Every
/// position-tracked fact the editor shows lives here; adding one is a field here
/// plus one line in `rebase_views`, and it is then consistent on all edit paths.
struct Views<'a> {
    highlight: &'a mut Option<HighlightCache>,
    brackets: &'a mut Brackets,
    decorations: &'a mut DecorationStore,
    /// The auto-close provenance store — a SEPARATE [`DecorationStore`] moved
    /// beside `decorations` so a forward edit's pairs rebase for free (and undo/redo
    /// inherit the move, though `reset_transient` empties it there first).
    autoclose: &'a mut DecorationStore,
    folds: &'a mut FoldSet,
    find: &'a mut FindState,
}

/// The single position-mover for a committed edit — rebase every derived view
/// through its patch. Taking [`Views`] (not `&mut Document`) is what lets both
/// the commit path and undo/redo run the *same* update: undo/redo destructure
/// `self` so this can borrow the views while `history` borrows the buffer. A new
/// derived fact is added to `Views` and moved here once, staying consistent
/// across both paths with no per-feature resync on the undo path.
/// Returns whether the bracket structure changed (not a pure offset shift) — the
/// caller uses it to skip [`Document::reconcile_folds`] on a structure-neutral
/// edit, whose folds provably can't have lost their pair (see the reconcile call
/// site).
fn rebase_views(
    views: &mut Views,
    buffer: &Buffer,
    tab: u32,
    committed: &Committed,
) -> core::ops::Range<u32> {
    // Highlight cache: splice the transaction's per-edit line spans (built
    // below). The size invariant (new_size = old_size − old_count + new_count)
    // keeps each edit's `old_lines` provably in-bounds. Only the actually-edited
    // lines are invalidated — NOT the first-to-last covering range — so a
    // scattered multi-caret edit doesn't over-invalidate the lines between.
    if let Some(cache) = views.highlight.as_mut() {
        let edits = committed.patch().edits();
        if !edits.is_empty() {
            // Per-edit pre-edit line spans `(pre_start, old_lines, new_lines)`,
            // ascending and coalesced disjoint, so the highlight commit
            // invalidates only the actually-edited lines — not the whole
            // first-to-last covering range (which would over-invalidate the
            // lines between scattered multi-caret edits). `old_lines` comes from
            // each edit's replaced text (its inverse op); `new_lines` from the
            // post-edit buffer.
            let mut spans: Vec<(u32, u32, u32)> = Vec::with_capacity(edits.len());
            let mut acc: i64 = 0;
            for (e, inv) in edits.iter().zip(committed.inverse_ops()) {
                let post_sr = buffer.offset_to_point(e.new.start).row;
                let post_er = buffer.offset_to_point(e.new.end).row;
                let new_lines = post_er - post_sr + 1;
                let old_lines = inv.text.bytes().filter(|&b| b == b'\n').count() as u32 + 1;
                let pre_start = (i64::from(post_sr) - acc) as u32;
                acc += i64::from(new_lines) - i64::from(old_lines);
                match spans.last_mut() {
                    // Same-line / touching edits share a pre-edit line — coalesce
                    // so the span list stays disjoint (the merge walks need it).
                    Some(last) if pre_start < last.0 + last.1 => {
                        let merged_end = (last.0 + last.1).max(pre_start + old_lines);
                        let combined_delta = (i64::from(last.2) - i64::from(last.1))
                            + (i64::from(new_lines) - i64::from(old_lines));
                        last.1 = merged_end - last.0;
                        last.2 = (i64::from(last.1) + combined_delta) as u32;
                    }
                    _ => spans.push((pre_start, old_lines, new_lines)),
                }
            }
            // Checkpoints, dense window, and dirty runs all ride the per-edit
            // spans — the window stays aimed at the viewport (only edited rows
            // invalidated), so a scattered multi-caret edit's wide covering range
            // never drains or repositions it.
            debug_assert_eq!(
                spans.iter().map(|&(_, o, n)| i64::from(n) - i64::from(o)).sum::<i64>(),
                i64::from(buffer.line_count()) - i64::from(cache.line_count()),
                "per-edit line deltas must sum to the buffer's line-count change",
            );
            cache.on_commit_patch(&spans);
        }
    }
    // Brackets: splice through the patch (the incremental engine; `match_text`
    // remains the load-time constructor and the tests' oracle). It returns the
    // reconcile window — the re-matched byte span, or `0..0` when only offsets
    // shifted (no structure changed) — the reconcile-skip signal below.
    let reconcile_region = views.brackets.apply_edit(committed.patch(), buffer);
    // Decorations: the one eager mover for ALL bulk kinds — diagnostics, find
    // matches, snippet stops — ride here, so undo/redo need no per-decoration
    // handling.
    views.decorations.apply_patch(committed.patch());
    // Auto-close provenance rides the SAME mover on its own store (per-range
    // independence makes the split byte-identical to keeping the pairs in
    // `decorations`). On forward edits this rebases the live pairs for free; on
    // undo/redo `reset_transient`→`clear_autoclose` has emptied it first, so this
    // is a no-op there — keeping it in the one mover means no edit path can ever
    // leave a pair stranded at a stale offset.
    views.autoclose.apply_patch(committed.patch());
    // The find repair rides the same commit hook, AFTER the store move (it needs
    // post-patch positions) — the match set is re-verified in a window around
    // each edit, so it is always current and undo/redo inherit the repair with no
    // find-specific resync.
    views.find.on_commit(committed.patch(), buffer, views.decorations);
    // Folds ride the same mover: shift on edits above, drop when the interior
    // is deleted — so folds survive edits and undo/redo with zero fold-specific
    // code (they travel the undo/redo closure forward and backward like a
    // decoration).
    views.folds.apply_patch(committed.patch());
    // Fold reveal rides the one mover too (after the position shift above): an
    // edit into HIDDEN text expands its fold — an edit must never alter text the
    // user cannot see. Threaded here, not hand-called per edit path, so undo/redo
    // inherit it and no edit path can forget it. (`reconcile_folds` — dropping
    // folds whose pair an edit broke — stays a once-per-transaction step on each
    // path: it can't run per-step inside history's callback. Order is immaterial:
    // expand skips broken-pair folds, the only ones reconcile drops.)
    expand_folds_touched(views.folds, views.brackets, buffer, tab, committed);
    // The re-matched byte region (POST-edit) — the reconcile window (see
    // `reconcile_folds_in`). Empty (`0..0`) on a shift-only edit, which re-scanned
    // nothing, so `is_empty()` is exactly the "structure unchanged" signal.
    reconcile_region
}

/// Extend a unit-granular drag: keep the origin unit fully selected, with the
/// head at the far edge of the unit containing `head`. A forward drag anchors at
/// the origin unit's start; a backward drag anchors at its end (the tail flips),
/// so the origin word/line is never partially deselected.
fn extend_by_unit(origin_unit: (u32, u32), head_unit: (u32, u32), origin: u32, head: u32) -> (u32, u32) {
    let (os, oe) = origin_unit;
    let (hs, he) = head_unit;
    if head >= origin {
        (os, he.max(oe))
    } else {
        (oe, hs.min(os))
    }
}

/// First untaken literal (case-sensitive, non-overlapping like
/// `str::match_indices`) occurrence of `needle` whose global start lies in
/// `[lo, hi)`, or `None`. Windowed over the buffer's ranged reads with a
/// `k−1`-byte tail overlap so a match straddling a window seam is caught, and
/// **never materializes the document** (so each Ctrl+D press is O(scanned),
/// never O(document)). Byte-identical to a whole-text `match_indices` scan for
/// non-self-overlapping needles — the Ctrl+D case, where selections are
/// word-shaped and never self-overlap; a self-overlapping needle may differ in
/// match phase near a seam. Pinned by
/// `windowed_next_occurrence_equals_whole_text_scan`.
fn scan_from(
    buffer: &Buffer,
    needle: &str,
    lo: u32,
    hi: u32,
    is_taken: &impl Fn(u32) -> bool,
) -> Option<(u32, u32)> {
    let k = needle.len() as u64;
    if k == 0 || lo >= hi {
        return None;
    }
    let len = buffer.len();
    let window = u64::from(crate::buffer::SCAN_WINDOW).max(k * 2); // ≥ 2k ⇒ always advances
    let mut pos = lo;
    while u64::from(pos) + k <= u64::from(len) {
        let win_end = buffer
            .clip_offset((u64::from(pos) + window).min(u64::from(len)) as u32, Bias::Right);
        let slice = buffer.slice(pos..win_end);
        let mut last_end: Option<u32> = None;
        for (i, _) in slice.match_indices(needle) {
            let start = pos + i as u32;
            if u64::from(start) >= u64::from(hi) {
                return None; // matches only grow — nothing left in [lo, hi)
            }
            last_end = Some(start + k as u32);
            if !is_taken(start) {
                return Some((start, start + k as u32));
            }
        }
        if win_end >= len {
            break;
        }
        // Resume through the one owner of the seam rule ([`Buffer::scan_resume`])
        // — shared verbatim with find's `scan_buffer`.
        pos = buffer.scan_resume(pos, win_end, k as u32, last_end);
    }
    None
}

/// The next untaken literal occurrence of `needle` in `text` at or after byte
/// `from`, wrapping to the start; skips ranges already in `taken` (every
/// occurrence shares `needle.len()`). `None` if `needle` is empty or every
/// occurrence is already selected. Non-overlapping (`match_indices`) — the
/// whole-text `#[cfg(test)]` oracle that [`scan_from`]'s windowed scan is
/// checked against (`windowed_next_occurrence_equals_whole_text_scan`).
#[cfg(test)]
fn find_next_occurrence(text: &str, needle: &str, from: u32, taken: &[(u32, u32)]) -> Option<(u32, u32)> {
    if needle.is_empty() {
        return None;
    }
    let len = needle.len() as u32;
    let is_taken = |start: u32| taken.iter().any(|&(s, e)| s == start && e == start + len);
    let mut wrapped: Option<u32> = None;
    for (idx, _) in text.match_indices(needle) {
        let start = idx as u32;
        if is_taken(start) {
            continue;
        }
        if start >= from {
            return Some((start, start + len));
        }
        if wrapped.is_none() {
            wrapped = Some(start); // first candidate before `from`, used on wrap
        }
    }
    wrapped.map(|start| (start, start + len))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::history::{GroupingHint, OpClass};

    fn doc(s: &str) -> Document {
        Document::new(s).unwrap()
    }

    fn typ(op: OpClass) -> GroupingHint {
        GroupingHint::mergeable(op)
    }

    /// The incremental brackets must deep-equal a from-scratch match of the
    /// live text — the bracket oracle, at Document level.
    fn assert_brackets_fresh(d: &Document) {
        let fresh = crate::bracket::Brackets::match_text(&d.text());
        assert_eq!(d.brackets().all(), fresh.all(), "incremental brackets diverged from scratch");
    }

    #[test]
    fn brackets_ride_edits_and_undo_redo_through_the_one_mover() {
        // A random walk of edits, undos, and redos: after EVERY step the
        // incremental bracket structure must equal a from-scratch match —
        // proving the one-mover claim (rebase_views runs the splice on the
        // forward path AND per undo/redo step; no bracket-specific resync
        // exists anywhere).
        let mut d = doc("fn a() {\n    x[0] = (1 + 2);\n}\n");
        let mut rng = 0xC0FFEEu64;
        let mut next = move |m: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % m
        };
        let inserts = ["{", "}", "(", ")", "[", "]", "\n", "x", "{}", "([)", "\n}\n"];
        for i in 0..200 {
            match next(4) {
                0 | 1 => {
                    let at = next(d.buffer().len() as u64 + 1) as u32;
                    let at = d.buffer().clip_offset(at, crate::Bias::Left);
                    let text = inserts[next(inserts.len() as u64) as usize];
                    let _ = d.edit(vec![EditOp::insert(at, text)]);
                }
                2 => {
                    let a = next(d.buffer().len() as u64 + 1) as u32;
                    let a = d.buffer().clip_offset(a, crate::Bias::Left);
                    let b = (a + 1 + next(4) as u32).min(d.buffer().len());
                    let b = d.buffer().clip_offset(b, crate::Bias::Right);
                    if a < b {
                        let _ = d.edit(vec![EditOp::delete(a..b)]);
                    }
                }
                _ => {
                    if next(2) == 0 {
                        d.undo();
                    } else {
                        d.redo();
                    }
                }
            }
            assert_brackets_fresh(&d);
            let _ = i;
        }
    }

    #[test]
    fn edit_undo_redo_round_trip() {
        let mut d = doc("hello");
        d.edit(vec![EditOp::new(0..5, "world")]).unwrap();
        assert_eq!(d.text(), "world");
        assert!(d.undo());
        assert_eq!(d.text(), "hello");
        assert!(d.redo());
        assert_eq!(d.text(), "world");
        // Nothing left to redo.
        assert!(d.undo());
        assert!(!d.undo()); // stack empty
        assert_eq!(d.text(), "hello");
    }

    #[test]
    fn a_new_edit_invalidates_redo() {
        let mut d = doc("");
        d.edit(vec![EditOp::insert(0, "a")]).unwrap();
        d.edit(vec![EditOp::insert(1, "b")]).unwrap(); // "ab"
        d.undo(); // "a"
        d.edit(vec![EditOp::insert(1, "X")]).unwrap(); // "aX" — diverges
        assert_eq!(d.text(), "aX");
        assert!(!d.redo(), "redo must be cleared by the divergent edit");
        assert_eq!(d.text(), "aX");
    }

    #[test]
    fn a_retained_branch_is_reachable_from_the_fork() {
        // Branching undo: the branch `a_new_edit_invalidates_redo` discards
        // is not gone — it is RETAINED as a sibling. Plain redo still does nothing
        // at the tip of the new branch, but undoing back to the fork exposes both
        // branches and either is selectable. A two-stack history could not do this
        // (a divergent edit would clear the redo stack); the branching tree keeps
        // both reachable.
        let mut d = doc("");
        d.edit(vec![EditOp::insert(0, "a")]).unwrap();
        d.edit(vec![EditOp::insert(1, "b")]).unwrap(); // "ab"
        d.undo(); // "a"
        d.edit(vec![EditOp::insert(1, "X")]).unwrap(); // "aX" — diverges, "b" kept
        assert!(!d.redo(), "plain redo does nothing at the tip of the new branch");
        // Undo back to the fork exposes BOTH branches.
        assert!(d.undo());
        assert_eq!(d.text(), "a");
        assert_eq!(d.redo_branch_count(), 2, "the 'b' and 'X' branches both survive");
        assert!(d.can_redo());
        // Steer to the older, retained branch (index 0 = 'b') and redo into it.
        assert!(d.select_redo_branch(0));
        assert!(d.redo());
        assert_eq!(d.text(), "ab", "the retained 'b' branch is fully reachable");
        // The newer branch is still there too.
        assert!(d.undo());
        assert!(d.select_redo_branch(1));
        assert!(d.redo());
        assert_eq!(d.text(), "aX");
    }

    #[test]
    fn opt_in_undo_limit_bounds_reach_and_keeps_recent() {
        // Opt-in pruning: with a limit of 3, older units drop while the most
        // recent 3 stay reachable and the text is untouched. The default (no
        // limit) prunes nothing — every other undo test above relies on that.
        let mut d = doc("");
        d.set_undo_limit(Some(3));
        for (i, ch) in "abcde".chars().enumerate() {
            d.edit(vec![EditOp::insert(i as u32, ch.to_string())]).unwrap();
        }
        assert_eq!(d.text(), "abcde");
        assert_eq!(d.undo_depth(), 3, "only the last 3 units are retained");
        assert!(d.undo()); // "abcd"
        assert!(d.undo()); // "abc"
        assert!(d.undo()); // "ab"
        assert_eq!(d.text(), "ab");
        assert!(!d.undo(), "the pruned units ('a','b' inserts) are gone");
        assert!(!d.can_undo());
        // Redo still runs forward from the pruned base.
        assert!(d.redo());
        assert_eq!(d.text(), "abc");
    }

    #[test]
    fn pruning_preserves_branches_within_the_window() {
        // The arena rebuild must remap children/preferred_child — a fork inside
        // the kept window survives pruning with both branches still selectable.
        let mut d = doc("");
        d.edit(vec![EditOp::insert(0, "a")]).unwrap(); // "a"
        d.edit(vec![EditOp::insert(1, "b")]).unwrap(); // "ab"
        d.undo(); // back to "a"; the "b" branch is retained
        d.edit(vec![EditOp::insert(1, "X")]).unwrap(); // "aX"; "a" is now a fork
        // Limit 1 makes the fork the new base, keeping BOTH of its child branches.
        d.set_undo_limit(Some(1));
        assert_eq!(d.undo_depth(), 1);
        assert!(d.undo()); // back to the fork ("a")
        assert_eq!(d.text(), "a");
        assert_eq!(d.redo_branch_count(), 2, "both branches survived the rebuild");
        assert!(d.select_redo_branch(0));
        assert!(d.redo());
        assert_eq!(d.text(), "ab", "the retained branch is still reachable after a prune");
    }

    #[test]
    fn typing_run_merges_into_one_undo_step() {
        // Five keystrokes with mergeable Type hints → one undo removes them all.
        let mut d = doc("");
        for (i, ch) in "hello".chars().enumerate() {
            d.edit_grouped(vec![EditOp::insert(i as u32, ch.to_string())], typ(OpClass::Type))
                .unwrap();
        }
        assert_eq!(d.text(), "hello");
        assert!(d.undo());
        assert_eq!(d.text(), "", "the whole typing run undoes as one unit");
        assert!(d.redo());
        assert_eq!(d.text(), "hello", "and redoes as one unit");
    }

    #[test]
    fn fold_toggle_seals_undo_and_clears_the_expand_ladder() {
        // A fold toggle moves carets (ejection), so it is a gesture boundary:
        // typing must not merge across it, and the expand ladder must clear so a
        // later shrink cannot restore a caret INTO the collapsed fold.
        let mut d = doc("m {\n  word\n}");
        d.set_selections(SelectionSet::new(7)); // inside "word"
        d.type_char('x');
        d.expand_selection(); // pushes the pre-expansion set on the ladder
        assert!(d.toggle_fold_opener(2));
        d.shrink_selection(); // ladder cleared by the toggle → no-op…
        let head = d.selections().newest().head();
        let fm = crate::fold_map::FoldMap::new(d.folds(), d.brackets(), d.buffer());
        assert!(
            fm.display_position(d.buffer(), head, d.tab_size()).is_some(),
            "…so the caret cannot be restored into the collapsed fold"
        );
        // And the toggle sealed the typing group: the later run undoes alone.
        d.type_char('y');
        assert!(d.undo());
        assert!(d.text().contains('x'), "undo reverted only the post-toggle typing");
    }

    #[test]
    fn find_navigation_expands_a_collapsed_inline_fold() {
        // display_position clips a chip-hidden column to the chip center, so an
        // offset inside a collapsed inline fold still reports a position;
        // unfold_to_reveal must expand the fold anyway so a find jump lands on
        // visible text, not inside the chip.
        let mut d = doc("x = [1, 2, 3]\nafter");
        assert!(d.toggle_fold_opener(4)); // the `[`
        d.set_find_query(Some(FindQuery { text: "2".into(), case_sensitive: false, ..Default::default() }), 0);
        d.find_next(0).expect("the match exists");
        assert!(!d.folds().is_folded(4), "the inline fold expanded to show the match");
    }

    #[test]
    fn editing_hidden_text_expands_only_the_enclosing_fold() {
        // expand_folds_touched, windowed to the enclosing folds: an edit into
        // a collapsed block's hidden interior expands THAT block (never edit text
        // the user can't see) and leaves every other fold collapsed — the walk
        // visits only the folds enclosing the edit, not all of them.
        let text = "a {\n  keep\n}\nb {\n  body\n}\nc\n";
        let mut d = doc(text);
        let opens: Vec<u32> = text.match_indices('{').map(|(i, _)| i as u32).collect();
        // `keep` is the first block, `edited` the second. Both openers sit BEFORE
        // the edit, so their offsets don't shift — the checks stay valid.
        let (keep, edited) = (opens[0], opens[1]);
        assert!(d.toggle_fold_opener(keep));
        assert!(d.toggle_fold_opener(edited));
        assert!(d.folds().is_folded(keep) && d.folds().is_folded(edited));
        // Edit inside the SECOND block's hidden interior (its "body" row).
        let inside = text.find("body").unwrap() as u32 + 2;
        d.edit(vec![EditOp::insert(inside, "X")]).unwrap();
        assert!(!d.folds().is_folded(edited), "editing inside the second block expands it");
        assert!(d.folds().is_folded(keep), "the untouched first block stays folded");
    }

    #[test]
    fn next_diagnostic_advances_past_a_shared_start_offset() {
        use crate::{Diagnostic, Severity};
        // Two diagnostics share a start offset: the (start, end) lexicographic
        // walk must reach both and then wrap, never cycle between them.
        let mut d = doc("abcdef");
        let rev = d.revision();
        d.set_diagnostics(
            rev,
            vec![
                Diagnostic::new(1..3, Severity::Error, "a"),
                Diagnostic::new(1..5, Severity::Warning, "b"),
            ],
        );
        assert_eq!(d.next_diagnostic(true), Some(1..3));
        assert_eq!(d.next_diagnostic(true), Some(1..5), "the same-start sibling is reachable");
        assert_eq!(d.next_diagnostic(true), Some(1..3), "and the walk wraps");
    }

    #[test]
    fn reveal_classes_and_no_op_verbs() {
        use crate::{Diagnostic, Severity};
        // Find-family jumps request Center; bracket hops request Fit; verbs
        // that change nothing request nothing — a no-op F8 must not autoscroll
        // the viewport back to the caret.
        let mut d = doc("f( ab )");
        let seq0 = d.reveal_seq();
        d.next_diagnostic(true); // no diagnostics → no request
        assert_eq!(d.reveal_seq(), seq0, "a no-op F8 requests no reveal");
        d.set_selections(SelectionSet::new(3));
        d.jump_to_bracket(); // no adjacent bracket? offset 3 is inside ( ) → moves
        assert!(d.reveal_seq() > seq0, "a real bracket hop requests a reveal");
        assert_eq!(d.reveal_mode(), RevealMode::Fit, "…of the Fit class");
        let rev = d.revision();
        d.set_diagnostics(rev, vec![Diagnostic::new(1..2, Severity::Error, "e")]);
        let seq1 = d.reveal_seq();
        d.next_diagnostic(true);
        assert!(d.reveal_seq() > seq1);
        assert_eq!(d.reveal_mode(), RevealMode::Center, "a diagnostic jump centers");
    }

    #[test]
    fn ctrl_d_force_reveals_while_select_all_holds() {
        // Ctrl+D jumps to the just-added cursor (FitForce); Ctrl+Shift+L reveals
        // with Fit (holds the viewport if a cursor is already visible).
        let mut d = doc("foo bar foo baz foo");
        d.set_selections(SelectionSet::new(0));
        d.add_next_occurrence(); // expand to "foo"
        d.add_next_occurrence(); // add the next "foo" — a new cursor to jump to
        assert_eq!(d.reveal_mode(), RevealMode::FitForce, "Ctrl+D force-reveals the new cursor");
        let mut e = doc("foo bar foo baz foo");
        e.set_selections(SelectionSet::new(0));
        e.select_all_occurrences();
        assert_eq!(e.reveal_mode(), RevealMode::Fit, "Ctrl+Shift+L holds if a cursor is visible");
    }

    #[test]
    fn diagnostic_navigation_wraps_selects_and_reveals() {
        use crate::{Diagnostic, Severity};
        let mut d = doc("aa bb\ncc\ndd");
        let rev = d.revision();
        d.set_diagnostics(
            rev,
            vec![
                Diagnostic::new(3..5, Severity::Warning, "w"),
                Diagnostic::new(9..11, Severity::Error, "e"),
            ],
        );
        assert_eq!(d.next_diagnostic(true), Some(3..5));
        assert_eq!(d.next_diagnostic(true), Some(9..11));
        assert_eq!(d.next_diagnostic(true), Some(3..5), "wraps to the first");
        assert_eq!(d.next_diagnostic(false), Some(9..11), "prev wraps to the last");
        let s = d.selections().newest();
        assert_eq!((s.start(), s.end()), (9, 11), "the diagnostic's span is selected");
        // No diagnostics → None, caret untouched.
        let mut e = doc("x");
        e.set_selections(SelectionSet::new(1));
        assert!(e.next_diagnostic(true).is_none());
        assert_eq!(e.selections().newest().head(), 1);
        // A diagnostic hidden in a collapsed fold is revealed on arrival.
        let mut f = doc("aa\nfn {\nbb\n}\ncc");
        assert!(f.toggle_fold_opener(6));
        let rev = f.revision();
        f.set_diagnostics(rev, vec![Diagnostic::new(8..10, Severity::Error, "hidden")]);
        assert_eq!(f.next_diagnostic(true), Some(8..10));
        assert!(!f.folds().is_folded(6), "the fold expanded to show the diagnostic");
    }

    #[test]
    fn caret_word_occurrences_are_whole_word_only() {
        let mut d = doc("foo foobar foo\nfoo");
        d.set_selections(SelectionSet::new(1)); // inside the first "foo"
        assert_eq!(
            d.caret_word_occurrences(0..u32::MAX),
            vec![0..3, 11..14, 15..18],
            "the foobar prefix is not a whole-word match"
        );
        // The window bounds the scan (the widget passes its viewport)…
        assert_eq!(d.caret_word_occurrences(4..15), vec![11..14], "only in-window matches");
        // …a non-empty selection produces no occurrence wash…
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 0, 3));
        d.set_selections(set);
        assert!(d.caret_word_occurrences(0..u32::MAX).is_empty());
        // …and neither does a caret in open whitespace.
        let mut w = doc("a  b");
        w.set_selections(SelectionSet::new(2));
        assert!(w.caret_word_occurrences(0..u32::MAX).is_empty());
    }

    #[test]
    fn find_navigation_expands_a_collapsed_fold_to_reveal_the_match() {
        // A match hidden inside a collapsed block: find_next must unfold it —
        // the caret can never land on an invisible position.
        let mut d = doc("aa\nfn {\nneedle\n}\nbb");
        let opener = 6; // the `{` (pair 6..15, hides rows 2..=3)
        assert!(d.toggle_fold_opener(opener));
        d.set_find_query(Some(FindQuery { text: "needle".into(), case_sensitive: false, ..Default::default() }), 0);
        let m = d.find_next(0).expect("the match exists");
        assert!(!d.folds().is_folded(opener), "the fold expanded to reveal the match");
        let fm = crate::fold_map::FoldMap::new(d.folds(), d.brackets(), d.buffer());
        assert!(
            fm.display_position(d.buffer(), m.end, d.tab_size()).is_some(),
            "the match head renders"
        );
        // A match on VISIBLE ground (the header line) leaves folds alone.
        let mut v = doc("aa\nfn {\nx\n}\nbb");
        assert!(v.toggle_fold_opener(6));
        v.set_find_query(Some(FindQuery { text: "fn".into(), case_sensitive: false, ..Default::default() }), 0);
        v.find_next(0).expect("match on the header");
        assert!(v.folds().is_folded(6), "a visible match keeps the fold collapsed");
    }

    #[test]
    fn jump_to_bracket_crosses_and_returns() {
        // `f( ab )` — ( at 1, ) at 6. Right of `(`: cross to the same side of
        // the partner (after `)`), and a second press returns.
        let mut d = doc("f( ab )");
        d.set_selections(SelectionSet::new(2));
        d.jump_to_bracket();
        assert_eq!(d.selections().newest().head(), 7);
        d.jump_to_bracket();
        assert_eq!(d.selections().newest().head(), 2);
        // No adjacent bracket: jump to the enclosing pair's closer.
        d.set_selections(SelectionSet::new(4)); // inside "ab"
        d.jump_to_bracket();
        assert_eq!(d.selections().newest().head(), 6, "lands before `)`");
        // No bracket in reach: the caret stays put.
        let mut e = doc("plain");
        e.set_selections(SelectionSet::new(3));
        e.jump_to_bracket();
        assert_eq!(e.selections().newest().head(), 3);
    }

    #[test]
    fn expand_selection_climbs_the_bracket_ladder_and_shrinks_back() {
        // `m { a(bb) }` — offsets: m0 ␠1 {2 ␠3 a4 (5 b6 b7 )8 ␠9 }10.
        let mut d = doc("m { a(bb) }");
        d.set_selections(SelectionSet::new(7)); // caret in "bb"
        let sel = |d: &Document| (d.selections().newest().start(), d.selections().newest().end());
        d.expand_selection();
        assert_eq!(sel(&d), (6, 8), "word first");
        d.expand_selection();
        assert_eq!(sel(&d), (5, 9), "word == () contents, so the pair incl. brackets");
        d.expand_selection();
        assert_eq!(sel(&d), (3, 10), "the brace contents");
        d.expand_selection();
        assert_eq!(sel(&d), (2, 11), "the brace pair incl. brackets");
        d.expand_selection();
        assert_eq!(sel(&d), (0, 11), "the whole document");
        d.expand_selection();
        assert_eq!(sel(&d), (0, 11), "fully expanded is a no-op");
        // Shrink walks back down the exact ladder.
        d.shrink_selection();
        assert_eq!(sel(&d), (2, 11));
        d.shrink_selection();
        assert_eq!(sel(&d), (3, 10));
        // Any other gesture clears the ladder — shrink becomes a no-op.
        d.move_carets(Motion::Right, false);
        let caret = sel(&d);
        d.shrink_selection();
        assert_eq!(sel(&d), caret, "the ladder cleared on the caret move");
    }

    #[test]
    fn add_caret_vertical_stacks_and_stops_at_edges() {
        // Stack carets down a column: every caret gains a neighbour, landings
        // on existing carets merge, and typing edits the whole column.
        let mut d = doc("aaa\nbbb\nccc");
        d.set_selections(SelectionSet::new(5)); // row 1, col 1
        d.add_caret_vertical(false); // above → row 0
        d.add_caret_vertical(true); // below both → rows 1 (merges) and 2
        assert_eq!(d.selections().len(), 3);
        d.type_char('X');
        assert_eq!(d.text(), "aXaa\nbXbb\ncXcc");
        // On the top display row, Up adds nothing (no clamp-to-doc-start caret).
        let mut e = doc("aa\nbb");
        e.set_selections(SelectionSet::new(1));
        e.add_caret_vertical(false);
        assert_eq!(e.selections().len(), 1);
    }

    #[test]
    fn add_caret_vertical_skips_a_collapsed_fold() {
        let mut d = doc("aa\nfn {\nbb\ncc\n}\ndd");
        assert!(d.toggle_fold_opener(6)); // hides rows 2..=4
        d.set_selections(SelectionSet::new(16)); // "dd", display row 2
        d.add_caret_vertical(false);
        let rows: Vec<u32> =
            d.selections().all().iter().map(|s| d.buffer().offset_to_point(s.head()).row).collect();
        assert_eq!(rows, vec![1, 5], "the new caret lands on the fold header, not a hidden row");
    }

    #[test]
    fn select_all_occurrences_takes_every_match_from_a_word_seed() {
        // Bare caret in "foo": seed the word, then take ALL occurrences —
        // typing then replaces every one (the multi-cursor rename).
        let mut d = doc("foo bar foo baz foo");
        d.set_selections(SelectionSet::new(1));
        d.select_all_occurrences();
        assert_eq!(d.selections().len(), 3);
        d.type_char('X');
        assert_eq!(d.text(), "X bar X baz X");
        // No word under the caret ⇒ no change.
        let mut e = doc("   ");
        e.set_selections(SelectionSet::new(1));
        e.select_all_occurrences();
        assert_eq!(e.selections().len(), 1);
        assert!(e.selections().newest().is_empty());
    }

    #[test]
    fn select_find_matches_turns_matches_into_selections() {
        let mut d = doc("foo bar foo baz foo");
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        d.find_next(0); // activate the first match (caret there)
        assert!(d.select_find_matches());
        assert_eq!(d.selections().len(), 3);
        assert_eq!(d.selections().newest().start(), 0, "the ACTIVE match is newest");
        // No query ⇒ no live matches ⇒ no-op.
        let mut e = doc("abc");
        assert!(!e.select_find_matches());
        assert_eq!(e.selections().len(), 1);
    }

    #[test]
    fn replace_all_swaps_every_match_in_one_undo_step() {
        // The load-bearing claim: replace-all is ONE transaction, so ONE undo
        // puts the document back byte-for-byte.
        let original = "foo bar foo baz foo";
        let mut d = doc(original);
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        assert_eq!(d.replace_all("QUX", false), 3);
        assert_eq!(d.text(), "QUX bar QUX baz QUX");
        assert!(d.undo());
        assert_eq!(d.text(), original, "replace-all must undo as ONE step");
        assert!(!d.undo(), "…with no second step hiding behind it");
        // The match set rode the commit through the shared mover — undo included.
        assert_eq!(d.find_match_count(), 3);
    }

    #[test]
    fn replace_all_is_not_capped_by_the_live_match_set() {
        // The live set is a capped PREFIX of the document, so a replace-all
        // built from it would silently leave everything past the cap untouched.
        // `replace_all` runs its own uncapped scan; this pins that it must.
        let n = crate::find::FIND_MATCH_CAP + 500;
        let mut d = doc(&"a".repeat(n));
        d.set_find_query(Some(FindQuery { text: "a".into(), case_sensitive: false, ..Default::default() }), 0);
        assert_eq!(d.find_match_count(), crate::find::FIND_MATCH_CAP, "the live set caps");
        assert_eq!(d.replace_all("b", false), n, "replace-all must NOT cap");
        assert_eq!(d.text(), "b".repeat(n), "every match past the cap replaced too");
    }

    #[test]
    fn replace_next_selects_before_it_replaces() {
        let mut d = doc("foo bar foo");
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        // Nothing selected yet: the first press only NAVIGATES — text is never
        // overwritten before it has been shown.
        assert_eq!(d.replace_next("X", false, 0), Some(0..3));
        assert_eq!(d.text(), "foo bar foo", "the first press must not replace");
        // Now the selection IS the active match: replace it, then advance. The
        // returned range is POST-edit — "foo bar foo" − "foo" + "X" ⇒ the
        // surviving match slid 8..11 → 6..9.
        assert_eq!(d.replace_next("X", false, 0), Some(6..9));
        assert_eq!(d.text(), "X bar foo");
        // …and one undo takes the replacement back.
        assert!(d.undo());
        assert_eq!(d.text(), "foo bar foo");
    }

    #[test]
    fn find_scope_restricts_matches_and_replace_all() {
        let mut d = doc("foo foo | foo foo");
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        assert_eq!(d.find_match_count(), 4);
        // Scope to the left half: the two matches beyond it stop existing.
        d.set_find_scope(Some(0..7), 0);
        assert_eq!(d.find_match_count(), 2, "only the in-scope matches survive");
        // …and replace-all inherits that without knowing the scope exists.
        assert_eq!(d.replace_all("X", false), 2, "replace-all must honor the scope");
        assert_eq!(d.text(), "X X | foo foo", "the out-of-scope matches are untouched");
        // Clearing the scope brings the rest back.
        d.set_find_scope(None, 0);
        assert_eq!(d.find_match_count(), 2, "the two outside are findable again");
    }

    #[test]
    fn the_find_scope_rides_edits_and_drops_when_it_collapses() {
        let mut d = doc("aaa foo bbb");
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        d.set_find_scope(Some(4..7), 0); // exactly "foo"
        assert_eq!(d.find_match_count(), 1);
        // An insert BEFORE the scope slides it, keeping it over the same text.
        d.edit(vec![EditOp::insert(0, "XY")]).unwrap();
        assert_eq!(d.find_scope(), Some(6..9));
        assert_eq!(d.find_match_count(), 1, "the scope still covers the match");
        // Deleting the scope's whole content collapses it ⇒ it drops, rather
        // than pinning zero matches forever.
        d.edit(vec![EditOp::delete(6..9)]).unwrap();
        assert_eq!(d.find_scope(), None, "a collapsed scope must clear");
    }

    #[test]
    fn an_edit_outside_the_scope_creates_no_matches() {
        let mut d = doc("foo | ...");
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        d.set_find_scope(Some(0..3), 0);
        assert_eq!(d.find_match_count(), 1);
        // Type a matching word far outside the scope: the repair window clamps
        // to the scope, so this must NOT become a match.
        d.edit(vec![EditOp::insert(6, "foo")]).unwrap();
        assert_eq!(d.find_match_count(), 1, "out-of-scope text must not match");
    }

    /// The oracle for the scoped repair: after random edits, the incrementally
    /// repaired set must be byte-identical to what a from-scratch scan of the
    /// (patch-ridden) scope produces. "foo" cannot overlap itself, so the
    /// documented self-overlap phase relaxation does not apply — these must
    /// agree exactly.
    #[test]
    fn scoped_repair_equals_a_fresh_scoped_scan_under_random_edits() {
        let mut seed = 0x5EED_u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let base = "foo bar foo baz\nfoo foo bar\nbarfoo foo\n".repeat(3);
        let n = base.len() as u32;
        for trial in 0..300 {
            let mut d = doc(&base);
            let a = (rng() as u32) % n;
            let b = (rng() as u32) % n;
            let (lo, hi) = (a.min(b), a.max(b));
            let lo = d.buffer().clip_offset(lo, Bias::Right);
            let hi = d.buffer().clip_offset(hi, Bias::Left);
            if lo >= hi {
                continue;
            }
            d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
            d.set_find_scope(Some(lo..hi), 0);
            for _ in 0..4 {
                let at = (rng() as u32) % (d.buffer().len() + 1);
                let at = d.buffer().clip_offset(at, Bias::Left);
                let ins = ["foo", "x", "", "\nfoo "][(rng() % 4) as usize];
                d.edit(vec![EditOp::insert(at, ins)]).unwrap();
            }
            let incremental: Vec<_> = d.find_matches_in(0..u32::MAX).map(|(r, _)| r).collect();
            // Force a from-scratch rescan of the SAME (now patch-ridden) scope:
            // dropping the query and re-setting it re-scans, and never touches
            // the scope.
            let scope = d.find_scope();
            let q = d.find_query().cloned().unwrap();
            d.set_find_query(None, 0);
            d.set_find_query(Some(q), 0);
            let fresh: Vec<_> = d.find_matches_in(0..u32::MAX).map(|(r, _)| r).collect();
            assert_eq!(incremental, fresh, "trial {trial}, scope {scope:?}");
            // …and nothing ever escaped the scope.
            if let Some(s) = scope {
                assert!(
                    incremental.iter().all(|m| m.start >= s.start && m.end <= s.end),
                    "trial {trial}: a match escaped the scope {s:?}: {incremental:?}"
                );
            }
        }
    }

    /// The oracle for the LINE-SCOPED repair. After random edits the
    /// incrementally repaired set must be byte-identical to a from-scratch scan
    /// — the same bar the literal path is held to.
    ///
    /// This is the proof that a line window is *sufficient* for a
    /// variable-length pattern, where the literal path's "a match starts at most
    /// k−1 bytes left of a changed byte" argument does not hold at all. `.*` is
    /// the case that makes the point: one edit anywhere rewrites a match
    /// spanning its entire line.
    #[test]
    fn line_scoped_repair_equals_a_fresh_scan_under_random_edits() {
        let mut seed = 0x00C0_FFEE_u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let base = "foo bar foo baz\nfoo foo bar\nbarfoo foo\n\nqux foofoo\n".repeat(3);
        let queries = [
            FindQuery { text: "foo".into(), whole_word: true, ..Default::default() },
            FindQuery { text: "f.o".into(), regex: true, ..Default::default() },
            FindQuery { text: r"\w+o".into(), regex: true, ..Default::default() },
            // The one a fixed-k window can never handle.
            FindQuery { text: ".*".into(), regex: true, ..Default::default() },
            FindQuery { text: r"\bba\w+".into(), regex: true, ..Default::default() },
            FindQuery {
                text: "foo|bar".into(),
                whole_word: true,
                regex: true,
                ..Default::default()
            },
        ];
        for (qi, q) in queries.iter().enumerate() {
            for trial in 0..40 {
                let mut d = doc(&base);
                d.set_find_query(Some(q.clone()), 0);
                for _ in 0..5 {
                    let len = d.buffer().len();
                    // Mix inserts and deletions — a deletion can JOIN two lines,
                    // which is the line-window repair's sharpest case.
                    if rng() % 3 == 0 && len > 4 {
                        let a = d.buffer().clip_offset((rng() as u32) % len, Bias::Left);
                        let b = d
                            .buffer()
                            .clip_offset((a + 1 + (rng() as u32) % 6).min(len), Bias::Right);
                        if a < b {
                            d.edit(vec![EditOp::delete(a..b)]).unwrap();
                        }
                    } else {
                        let at = d.buffer().clip_offset((rng() as u32) % (len + 1), Bias::Left);
                        let ins = ["foo", "x", "\n", " ", "\nfoo bar"][(rng() % 5) as usize];
                        d.edit(vec![EditOp::insert(at, ins)]).unwrap();
                    }
                }
                let incremental: Vec<_> = d.find_matches_in(0..u32::MAX).map(|(r, _)| r).collect();
                // Force a from-scratch rescan and compare.
                d.set_find_query(None, 0);
                d.set_find_query(Some(q.clone()), 0);
                let fresh: Vec<_> = d.find_matches_in(0..u32::MAX).map(|(r, _)| r).collect();
                assert_eq!(incremental, fresh, "query {qi} ({:?}), trial {trial}", q.text);
            }
        }
    }

    #[test]
    fn regex_replace_all_is_one_undo_step() {
        let original = "a1 b22 c333";
        let mut d = doc(original);
        d.set_find_query(
            Some(FindQuery { text: r"\d+".into(), regex: true, ..Default::default() }),
            0,
        );
        assert_eq!(d.find_match_count(), 3);
        assert_eq!(d.replace_all("#", false), 3);
        assert_eq!(d.text(), "a# b# c#");
        assert!(d.undo());
        assert_eq!(d.text(), original, "a regex replace-all undoes as ONE step");
    }

    #[test]
    fn a_regex_replacement_expands_capture_groups() {
        // Regex mode makes the replacement a TEMPLATE, per VS Code.
        let mut d = doc("fn one()\nfn two()");
        d.set_find_query(
            Some(FindQuery { text: r"fn (\w+)\(\)".into(), regex: true, ..Default::default() }),
            0,
        );
        assert_eq!(d.replace_all("let $1 = 1;", false), 2);
        assert_eq!(d.text(), "let one = 1;\nlet two = 1;");
        assert!(d.undo());
        assert_eq!(d.text(), "fn one()\nfn two()", "still ONE undo step");

        // …and a LITERAL query does not expand: nothing captured anything, so a
        // typed `$1` must land as a literal `$1`.
        let mut e = doc("aaa");
        e.set_find_query(Some(FindQuery { text: "aaa".into(), ..Default::default() }), 0);
        assert_eq!(e.replace_all("$1", false), 1);
        assert_eq!(e.text(), "$1");
    }

    #[test]
    fn replace_next_expands_captures_for_the_active_match_only() {
        let mut d = doc("fn one()\nfn two()");
        d.set_find_query(
            Some(FindQuery { text: r"fn (\w+)\(\)".into(), regex: true, ..Default::default() }),
            0,
        );
        d.replace_next("x", false, 0); // first press: selects only
        d.replace_next("[$1]", false, 0); // second: replaces the active match
        assert_eq!(d.text(), "[one]\nfn two()");
    }

    #[test]
    fn preserve_case_recases_replacements_to_their_match() {
        // A literal rename that keeps each occurrence's casing: FOO→BAR,
        // Foo→Bar, foo→bar, all from one replacement "bar".
        let mut d = doc("FOO Foo foo fOo");
        d.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        assert_eq!(d.replace_all("bar", true), 4);
        // FOO⇒upper, Foo⇒capitalized, foo⇒lower, fOo⇒mixed (left as typed).
        assert_eq!(d.text(), "BAR Bar bar bar");

        // Off, the same replacement lands verbatim.
        let mut e = doc("FOO Foo foo");
        e.set_find_query(Some(FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() }), 0);
        assert_eq!(e.replace_all("bar", false), 3);
        assert_eq!(e.text(), "bar bar bar");
    }

    #[test]
    fn preserve_case_applies_after_regex_expansion() {
        // The re-casing keys off the whole match, applied to the expanded
        // template — so `$1`-driven text is re-cased too.
        let mut d = doc("GETVALUE getvalue");
        d.set_find_query(
            Some(FindQuery { text: "(get)(value)".into(), regex: true, ..Default::default() }),
            0,
        );
        // Braced refs — `$2_$1` would read `$2_` as a group named "2_" (the
        // greedy unbraced-ref rule), so delimit with `${…}`.
        assert_eq!(d.replace_all("${2}_${1}", true), 2);
        // GETVALUE (all upper) ⇒ VALUE_GET; getvalue (all lower) ⇒ value_get.
        assert_eq!(d.text(), "VALUE_GET value_get");
    }

    #[test]
    fn an_unfinished_regex_matches_nothing_and_reports_why() {
        let mut d = doc("foo (bar)");
        d.set_find_query(
            Some(FindQuery { text: "(".into(), regex: true, ..Default::default() }),
            0,
        );
        assert_eq!(d.find_match_count(), 0, "an unfinished pattern matches nothing");
        assert!(d.find_pattern_error().is_some(), "…and says why");
        // Editing while the pattern is broken must not panic or strand state.
        d.edit(vec![EditOp::insert(0, "x")]).unwrap();
        assert_eq!(d.find_match_count(), 0);
        // Finishing the pattern resumes matching, and clears the error.
        d.set_find_query(
            Some(FindQuery { text: r"\(bar\)".into(), regex: true, ..Default::default() }),
            0,
        );
        assert!(d.find_pattern_error().is_none());
        assert_eq!(d.find_match_count(), 1);
    }

    #[test]
    fn replace_all_without_a_match_commits_nothing() {
        let mut d = doc("foo");
        assert_eq!(d.replace_all("x", false), 0, "no query ⇒ no-op");
        d.set_find_query(Some(FindQuery { text: "zzz".into(), case_sensitive: false, ..Default::default() }), 0);
        assert_eq!(d.replace_all("x", false), 0, "no matches ⇒ no-op");
        assert_eq!(d.text(), "foo");
        assert!(!d.is_dirty(), "a no-op replace must not open an undo step");
        assert!(!d.undo());
    }

    #[test]
    fn caret_moves_seal_the_typing_group() {
        // A caret move, click, or box gesture closes the open typing run —
        // type → move → type undoes as TWO steps, not one.
        let mut d = doc("");
        d.type_char('f');
        d.type_char('o');
        d.move_carets(Motion::Left, false); // arrow-key boundary
        d.type_char('x'); // "fxo"
        assert!(d.undo());
        assert_eq!(d.text(), "fo", "undo reverts only the run typed after the move");

        // A mouse click (set_selections) is the same boundary…
        let mut c = doc("");
        c.type_char('a');
        c.type_char('b');
        c.set_selections(SelectionSet::new(0));
        c.type_char('z'); // "zab"
        assert!(c.undo());
        assert_eq!(c.text(), "ab", "the click sealed the run before it");

        // …and so is a column-box gesture (which seals WITHOUT reset_transient,
        // since it must keep the box anchor).
        let mut b = doc("one\ntwo");
        b.set_selections(SelectionSet::new(0));
        b.type_char('z'); // "zone\ntwo"
        b.column_drag((0, 2), (1, 2));
        b.type_char('q');
        assert!(b.undo());
        assert_eq!(b.text(), "zone\ntwo", "the box gesture sealed the run before it");
    }

    /// A line-duplication verb must drop auto-close provenance through the one
    /// owner (`clear_autoclose`), so its `AutoClosePair` decoration (which has
    /// `EmptyPolicy::Keep`, so the store never self-drops it) is removed with it
    /// rather than orphaned in the store with a lost id.
    #[test]
    fn line_duplication_drops_autoclose_provenance() {
        // Provenance lives in its own auto-close store.
        let ac_count = |d: &Document| d.autoclose_pair_count();
        let mut d = doc("x");
        d.move_carets(Motion::Right, false); // caret after 'x' (offset 1)
        d.type_char('('); // auto-close → "x()", provenance armed over the pair
        assert_eq!(d.text(), "x()");
        assert_eq!(ac_count(&d), 1, "typing '(' arms one AutoClosePair decoration");
        // Duplicate the line above: the caret stays inside the (one-line) pair,
        // so `validate_autoclose` keeps the provenance — and copy_line must clear
        // the decoration rather than leave the rebased one orphaned.
        d.copy_line(false);
        assert_eq!(d.text(), "x()\nx()");
        assert_eq!(
            ac_count(&d),
            0,
            "copy_line must clear the provenance decoration, not orphan it in the store",
        );
    }

    /// Provenance in its OWN store must be untouched by a document-scale
    /// diagnostic set living in the bulk `decorations` store: arm a pair, publish
    /// far diagnostics, type inside the pair — the pair survives, overtype still
    /// works, and the diagnostics ride on independently. The two stores stay
    /// fully decoupled.
    #[test]
    fn type_bracket_then_far_diagnostics_pair_survives() {
        use crate::{Diagnostic, Severity};
        let mut d = doc("hello\nworld\n");
        let end = d.text().len() as u32; // 12 — EOL of the empty final line
        d.set_selections(SelectionSet::new(end));
        d.type_char('('); // "hello\nworld\n()" — provenance armed, caret between ( )
        assert_eq!(d.autoclose_pair_count(), 1);
        // Diagnostics FAR from the pair (line 0), in the SEPARATE bulk store.
        d.set_diagnostics(
            d.revision(),
            vec![
                Diagnostic::new(0..1, Severity::Error, "e0"),
                Diagnostic::new(2..3, Severity::Warning, "w0"),
            ],
        );
        // A plain char inside the pair — the pair must survive (caret stays in it),
        // undisturbed by the diagnostics in the other store.
        d.type_char('x');
        assert!(d.text().ends_with("(x)"));
        assert_eq!(d.autoclose_pair_count(), 1, "far diagnostics don't disturb the pair");
        // The diagnostics still live in the bulk store, rebased past the edit.
        assert_eq!(d.diagnostics_in(0..u32::MAX).count(), 2, "the diagnostics ride on");
        // Overtype still works — provenance is intact and readable from its store.
        d.type_char(')');
        assert!(d.text().ends_with("(x)"), "overtype consumes the tracked close, no doubled )");
        assert_eq!(d.autoclose_pair_count(), 0, "overtype consumed the pair");
    }

    /// Oracle: the own auto-close store must MOVE identically to keeping the pairs
    /// in the unified `decorations` store (per-range independence makes the split
    /// byte-identical). A shadow set of `AutoClosePair` decorations is planted in
    /// the bulk store (interleaved with diagnostics, a realistic mixed store) at
    /// the same ranges; then plain chars are typed *inside* every pair (carets
    /// stay in, so `validate_autoclose` reaps nothing), and after each commit the
    /// own store's ranges must equal the bulk store's `AutoClosePair` ranges —
    /// through BOTH the multi-edit naive mover and the single-edit windowed mover.
    /// A wiring slip (a missed rebase site, a wrong patch, a bias drift) breaks
    /// it.
    #[test]
    fn autoclose_store_moves_identically_to_unified_store() {
        use crate::decorations::{DecorationKind, Stickiness};
        use crate::{Diagnostic, Severity};
        // Run one scenario: arm a pair at each caret, mirror the pairs into the
        // bulk store, then type `steps` plain chars at all carets, asserting the two
        // stores stay range-identical after every commit.
        let scenario = |text: &str, carets: &[u32], steps: usize| {
            let mut d = doc(text);
            d.set_selections(SelectionSet::from_offsets(carets));
            d.type_char('('); // arm one one-line pair per caret
            let pairs = d.autoclose_ranges();
            assert_eq!(pairs.len(), carets.len(), "one pair per caret armed");
            // Shadow the pairs into the bulk store (the "unified" reference) and
            // interleave a diagnostic so the mover sees a realistic mixed store.
            for r in &pairs {
                d.decorations_mut().add_decoration(
                    r.clone(),
                    DecorationKind::AutoClosePair,
                    Stickiness::AlwaysGrows,
                );
            }
            d.set_diagnostics(d.revision(), vec![Diagnostic::new(0..1, Severity::Warning, "d")]);
            let unified = |d: &Document| -> Vec<Range<u32>> {
                d.decorations()
                    .iter()
                    .filter(|r| matches!(r.kind, DecorationKind::AutoClosePair))
                    .map(|r| r.range.clone())
                    .collect()
            };
            assert_eq!(d.autoclose_ranges(), unified(&d), "shadow starts equal");
            for step in 0..steps {
                d.type_char('z'); // grows every pair; carets stay inside
                assert_eq!(
                    d.autoclose_ranges(),
                    unified(&d),
                    "own store diverged from the unified store at step {step}",
                );
            }
        };
        // Multi-caret typing keeps all pairs occupied → the multi-edit naive mover.
        scenario("aa\nbb\ncc\ndd\nee\n", &[2, 5, 8, 11, 14], 8);
        // Single caret → the single-edit windowed mover (one pair, kept occupied).
        scenario("solo line here\n", &[14], 10);
    }

    #[test]
    fn different_classes_do_not_merge() {
        let mut d = doc("ab");
        d.edit_grouped(vec![EditOp::insert(2, "c")], typ(OpClass::Type)).unwrap(); // "abc"
        d.edit_grouped(vec![EditOp::delete(0..1)], typ(OpClass::Delete)).unwrap(); // "bc"
        assert_eq!(d.text(), "bc");
        assert!(d.undo()); // undo the delete only
        assert_eq!(d.text(), "abc");
        assert!(d.undo()); // undo the type
        assert_eq!(d.text(), "ab");
    }

    #[test]
    fn dirty_tracking_survives_undo_and_save() {
        let mut d = doc("x");
        assert!(!d.is_dirty());
        d.edit(vec![EditOp::insert(1, "y")]).unwrap(); // "xy"
        assert!(d.is_dirty());
        d.undo(); // back to "x" == the saved (initial) state
        assert!(!d.is_dirty(), "undo back to the save point reads clean");
        d.redo(); // "xy"
        assert!(d.is_dirty());
        d.mark_saved();
        assert!(!d.is_dirty());
    }

    #[test]
    fn ctrl_d_selects_word_then_adds_occurrences_then_stops() {
        let mut d = doc("foo bar foo baz foo"); // "foo" at 0, 8, 16
        // First press: the caret at 0 expands to the surrounding word.
        d.add_next_occurrence();
        assert_eq!(d.selections().len(), 1);
        let s = d.selections().newest();
        assert_eq!((s.start(), s.end()), (0, 3));
        // Next presses add each following "foo" as a new (newest) selection.
        d.add_next_occurrence();
        assert_eq!(d.selections().len(), 2);
        assert_eq!((d.selections().newest().start(), d.selections().newest().end()), (8, 11));
        d.add_next_occurrence();
        assert_eq!(d.selections().len(), 3);
        assert_eq!(d.selections().newest().start(), 16);
        // All three found; the next press wraps, finds only taken ranges → no-op.
        d.add_next_occurrence();
        assert_eq!(d.selections().len(), 3, "no untaken occurrence remains");
    }

    #[test]
    fn windowed_next_occurrence_equals_whole_text_scan() {
        // The windowed `scan_from` (Ctrl+D's engine, never materializes the
        // rope) must match `find_next_occurrence` (the whole-text `match_indices`
        // oracle) for non-self-overlapping needles, across random texts, seeds,
        // and taken sets — including the wrap path.
        let mut state: u64 = 0x243F_6A88_85A3_08D3;
        let mut next = |n: u32| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % n
        };
        // Alphabet {a,b,c,space}; needles below have no proper prefix == suffix.
        let needles = ["ab", "ba", "abc", "cab"];
        for _ in 0..300 {
            let n = 12 + next(60); // text length
            let text: String = (0..n)
                .map(|_| b"abc "[next(4) as usize] as char)
                .collect();
            let d = doc(&text);
            let needle = needles[next(needles.len() as u32) as usize];
            let nlen = needle.len() as u32;
            // A random subset of the actual occurrences becomes "taken".
            let all: Vec<(u32, u32)> = text
                .match_indices(needle)
                .map(|(i, _)| (i as u32, i as u32 + nlen))
                .collect();
            let taken: Vec<(u32, u32)> = all.iter().copied().filter(|_| next(2) == 0).collect();
            let is_taken = |s: u32| taken.iter().any(|&(st, en)| st == s && en == s + nlen);
            let from = next(n + 1);
            let got = scan_from(d.buffer(), needle, from, d.buffer().len(), &is_taken)
                .or_else(|| scan_from(d.buffer(), needle, 0, from, &is_taken));
            let want = find_next_occurrence(&text, needle, from, &taken);
            assert_eq!(got, want, "text {text:?} needle {needle:?} from {from} taken {taken:?}");
        }
    }

    #[test]
    fn column_select_grows_widens_and_shrinks_back() {
        use crate::movement::ColumnDir::{Down, Left, Right, Up};
        // Three aligned rows; caret at (0, 1).
        let mut d = doc("abcd\nabcd\nabcd");
        d.set_selections(SelectionSet::new(1));
        // Down twice → a caret per row at column 1 (empty box, 3 rows).
        d.column_select(Down);
        d.column_select(Down);
        assert_eq!(d.selections().len(), 3);
        assert!(d.selections().all().iter().all(|s| s.is_empty() && s.start() % 5 == 1));
        // Right twice → each row selects columns 1..3.
        d.column_select(Right);
        d.column_select(Right);
        assert_eq!(d.selections().len(), 3);
        assert!(d.selections().all().iter().all(|s| s.end() - s.start() == 2));
        // Up twice → the box shrinks back to the anchor row only.
        d.column_select(Up);
        d.column_select(Up);
        assert_eq!(d.selections().len(), 1, "shrinks back toward the anchor");
        // Left twice → back to a bare caret at the anchor.
        d.column_select(Left);
        d.column_select(Left);
        assert_eq!(d.selections().len(), 1);
        assert!(d.selections().newest().is_empty());
    }

    #[test]
    fn column_select_clamps_each_row_to_its_length() {
        use crate::movement::ColumnDir::{Down, Right};
        // A long row over a short one; a wide box clamps the short row.
        let mut d = doc("abcdef\nab"); // line 0 len 6, line 1 len 2
        d.set_selections(SelectionSet::new(0)); // caret at (0,0)
        d.column_select(Down); // box rows 0..1, col 0..0
        for _ in 0..4 {
            d.column_select(Right); // active col → 4
        }
        let rows = d.selections().all();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].start(), rows[0].end()), (0, 4)); // "abcd" on line 0
        // Line 1 ("ab", offsets 7..9) clamps to its end: 7..9.
        assert_eq!((rows[1].start(), rows[1].end()), (7, 9));
    }

    #[test]
    fn column_drag_builds_a_box_from_two_corners() {
        let mut d = doc("abcd\nabcd\nabcd");
        d.column_drag((0, 1), (2, 3)); // 3 rows, cols 1..3
        assert_eq!(d.selections().len(), 3);
        assert!(d.selections().all().iter().all(|s| s.end() - s.start() == 2));
        // Virtual columns past a short line clamp per row.
        let mut r = doc("abcdef\nab"); // line0 len 6, line1 len 2
        r.column_drag((0, 0), (1, 5)); // active col 5 > line1 len
        let rows = r.selections().all();
        assert_eq!((rows[0].start(), rows[0].end()), (0, 5)); // line0 → 5
        assert_eq!((rows[1].start(), rows[1].end()), (7, 9)); // line1 clamps to its end
    }

    #[test]
    fn column_box_is_display_cells_across_tabs() {
        // Row 0 leads with a tab (4 cells); row 1 is plain. A box over cells
        // 4..6 must select the same VISUAL slice — the two chars after the tab
        // — not reuse the cells as byte columns (which lands at row 0's EOL).
        let mut d = doc("\tabcd\nwxyz"); // row 1 starts at offset 6
        d.column_drag((0, 4), (1, 6));
        let rows = d.selections().all();
        assert_eq!(rows.len(), 2);
        assert_eq!((rows[0].start(), rows[0].end()), (1, 3), "cells 4..6 are bytes 1..3 after the tab");
        assert_eq!((rows[1].start(), rows[1].end()), (10, 10), "row 1 ends at cell 4; both corners clamp");
    }

    #[test]
    fn column_box_resolves_through_a_collapsed_inline_fold() {
        // `f([a, b]) x` with the [..] pair collapsed: display cells right of
        // the chip sit one left of their byte columns. A box edge at cell 9
        // must land on `x` (byte 10), not the byte-9 space.
        let mut d = doc("f([a, b]) x");
        assert!(d.toggle_fold_opener(2)); // the `[`
        d.column_drag((0, 6), (0, 9));
        let s = d.selections().newest();
        assert_eq!((s.start(), s.end()), (7, 10), "cells 6..9 are `]) `, one left of the byte columns");
    }

    #[test]
    fn column_box_spans_visible_rows_only() {
        // A box dragged across a collapsed block fold selects only what is on
        // screen: the rows hidden inside the fold get no selection.
        let mut d = doc("aa\nfn {\nbb\ncc\n}\ndd");
        assert!(d.toggle_fold_opener(6)); // the `{` — hides rows 2..=4
        d.column_drag((0, 0), (5, 1));
        let rows: Vec<u32> =
            d.selections().all().iter().map(|s| d.buffer().offset_to_point(s.start()).row).collect();
        assert_eq!(rows, vec![0, 1, 5], "one selection per VISIBLE row");
    }

    #[test]
    fn column_select_steps_display_rows_over_a_fold() {
        use crate::movement::ColumnDir::Up;
        // Caret below a collapsed fold; growing the box upward hops the hidden
        // rows in ONE step — corners walk display rows, not buffer rows.
        let mut d = doc("aa\nfn {\nbb\ncc\n}\ndd");
        assert!(d.toggle_fold_opener(6));
        d.set_selections(SelectionSet::new(16)); // caret at (5, 0) — on "dd"
        d.column_select(Up);
        let rows: Vec<u32> =
            d.selections().all().iter().map(|s| d.buffer().offset_to_point(s.start()).row).collect();
        assert_eq!(rows, vec![1, 5], "the step lands on the fold header, skipping hidden rows");
    }

    #[test]
    fn column_select_anchors_at_the_display_cell_after_a_tab() {
        use crate::movement::ColumnDir::Down;
        // A caret just after row 0's leading tab renders at cell 4; the box
        // column is that CELL, so on the plain row below it lands at byte 4
        // (visually aligned), not byte 1.
        let mut d = doc("\tab\nwwwwww"); // row 1 starts at offset 4
        d.set_selections(SelectionSet::new(1)); // (0, 1): just after the tab
        d.column_select(Down);
        assert_eq!(d.selections().newest().head(), 8, "cell 4 on row 1 is byte 4 (offset 8)");
    }

    #[test]
    fn any_action_exits_column_mode() {
        use crate::movement::ColumnDir::Down;
        let mut d = doc("abcd\nabcd\nabcd");
        d.set_selections(SelectionSet::new(0));
        d.column_select(Down); // 2-row box
        assert_eq!(d.selections().len(), 2);
        d.move_carets(Motion::Right, false); // a plain move exits the mode…
        // …so the next column_select re-anchors from the current caret, not the
        // stale box (a fresh 2-row box, not a 3-row one).
        d.column_select(Down);
        assert_eq!(d.selections().len(), 2, "re-anchored, not continuing the old box");
    }

    fn range(d: &Document) -> (u32, u32) {
        (d.selections().newest().start(), d.selections().newest().end())
    }

    #[test]
    fn drag_select_word_granularity_keeps_the_origin_word() {
        let mut d = doc("foo bar baz\nqux"); // foo 0..3, bar 4..7, baz 8..11
        // Double-click "bar" (head == origin) selects just the word.
        d.drag_select(Granularity::Word, 5, 5);
        assert_eq!(range(&d), (4, 7));
        // Drag right into "baz": whole words, "bar".."baz".
        d.drag_select(Granularity::Word, 5, 9);
        assert_eq!(range(&d), (4, 11));
        // Drag back-left into "foo": origin word "bar" stays fully selected.
        d.drag_select(Granularity::Word, 5, 1);
        assert_eq!(range(&d), (0, 7));
        // A double-click surrounded by whitespace is a bare caret.
        let mut w = doc("a  b");
        w.drag_select(Granularity::Word, 2, 2);
        assert!(w.selections().newest().is_empty());
    }

    #[test]
    fn drag_select_line_and_char_granularity() {
        let mut d = doc("aa\nbb\ncc"); // line0 0..2/\n2, line1 3..5/\n5, line2 6..8
        d.drag_select(Granularity::Line, 1, 1); // triple-click line 0 → incl \n
        assert_eq!(range(&d), (0, 3));
        d.drag_select(Granularity::Line, 1, 4); // drag into line 1
        assert_eq!(range(&d), (0, 6));
        d.drag_select(Granularity::Line, 7, 7); // last line → to its end
        assert_eq!(range(&d), (6, 8));
        d.drag_select(Granularity::Char, 2, 8); // char = a plain range
        assert_eq!(range(&d), (2, 8));
    }

    fn head_row(d: &Document) -> u32 {
        d.buffer().offset_to_point(d.selections().newest().head()).row
    }

    #[test]
    fn move_line_swaps_with_neighbour_and_rides() {
        let mut d = doc("aaa\nbbb\nccc");
        d.set_selections(SelectionSet::new(4)); // "bbb" (row 1)
        d.move_line(true); // down
        assert_eq!(d.text(), "aaa\nccc\nbbb");
        assert_eq!(head_row(&d), 2); // the caret rode down with the line
        d.move_line(false); // up → back
        assert_eq!(d.text(), "aaa\nbbb\nccc");
        assert_eq!(head_row(&d), 1);
    }

    #[test]
    fn move_line_is_a_noop_at_the_edges() {
        let mut d = doc("aaa\nbbb");
        d.set_selections(SelectionSet::new(0));
        d.move_line(false); // up at the top
        assert_eq!(d.text(), "aaa\nbbb");
        d.set_selections(SelectionSet::new(4)); // last content line
        d.move_line(true); // down at the bottom
        assert_eq!(d.text(), "aaa\nbbb");
    }

    #[test]
    fn move_line_down_respects_the_trailing_empty_line() {
        let mut d = doc("aaa\nbbb\n"); // trailing \n → an empty final row
        d.set_selections(SelectionSet::new(4)); // "bbb", the last *content* line
        d.move_line(true); // must not move into/past the empty final line
        assert_eq!(d.text(), "aaa\nbbb\n");
    }

    #[test]
    fn copy_line_duplicates_below_and_above() {
        let mut d = doc("aaa\nbbb");
        d.set_selections(SelectionSet::new(0));
        d.copy_line(true); // down → caret on the lower copy
        assert_eq!(d.text(), "aaa\naaa\nbbb");
        assert_eq!(head_row(&d), 1);
        let mut d2 = doc("aaa\nbbb");
        d2.set_selections(SelectionSet::new(0));
        d2.copy_line(false); // up → caret on the upper copy
        assert_eq!(d2.text(), "aaa\naaa\nbbb");
        assert_eq!(head_row(&d2), 0);
    }

    #[test]
    fn copy_line_duplicates_the_final_line() {
        let mut d = doc("aaa\nbbb"); // no trailing \n
        d.set_selections(SelectionSet::new(4)); // "bbb"
        d.copy_line(true);
        assert_eq!(d.text(), "aaa\nbbb\nbbb");
        assert_eq!(head_row(&d), 2);
    }

    #[test]
    fn add_caret_adds_a_second_cursor() {
        let mut d = doc("abcdef");
        assert_eq!(d.selections().len(), 1);
        d.add_caret(3);
        assert_eq!(d.selections().len(), 2);
    }

    #[test]
    fn collapse_returns_to_the_primary_caret() {
        let mut d = doc("foo foo foo");
        d.add_next_occurrence(); // select "foo"
        d.add_next_occurrence(); // add the next
        assert!(d.selections().len() >= 2);
        d.collapse_selections();
        assert_eq!(d.selections().len(), 1);
        assert!(d.selections().newest().is_empty());
    }

    #[test]
    fn find_next_occurrence_scans_forward_then_wraps() {
        // "aXaYa": 'a' at 0, 2, 4. From 1, skipping the (0,1) already-taken one.
        assert_eq!(super::find_next_occurrence("aXaYa", "a", 1, &[(0, 1)]), Some((2, 3)));
        // From past the last match, wrap to the first untaken.
        assert_eq!(super::find_next_occurrence("aXaYa", "a", 5, &[(2, 3), (4, 5)]), Some((0, 1)));
        // Every occurrence taken → None.
        assert_eq!(super::find_next_occurrence("aa", "a", 0, &[(0, 1), (1, 2)]), None);
    }

    #[test]
    fn dirty_after_undo_then_divergent_edit() {
        // Silent-data-loss guard: save, undo, retype something different → the
        // dirty flag must NOT read clean just because the stack depth matches.
        let mut d = doc("abc");
        d.edit(vec![EditOp::delete(2..3)]).unwrap(); // "ab"
        d.mark_saved();
        d.undo(); // "abc"
        assert!(d.is_dirty(), "undone away from the save point → dirty");
        d.edit(vec![EditOp::delete(0..1)]).unwrap(); // "bc" — diverges from "ab"
        assert!(d.is_dirty(), "divergent edit is still dirty, never falsely clean");
    }

    #[test]
    fn highlight_cache_splice_tracks_line_count_across_edits() {
        // Tiny grammar/theme via the same app-injection path scratch uses.
        const G: &str = "%YAML 1.2\n---\nname: T\nscope: source.t\ncontexts:\n  main:\n    - match: '\\w+'\n      scope: keyword.t\n";
        const TH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>settings</key><array><dict><key>settings</key><dict><key>foreground</key><string>#FFFFFF</string></dict></dict></array></dict></plist>"#;
        let mut d = doc("ab\ncd");
        d.set_syntax(
            crate::SyntaxDef::from_sublime_syntax(G).unwrap(),
            crate::TokenTheme::from_tm_theme(TH).unwrap(),
        );
        d.tokenize_highlight(d.buffer().line_count());
        assert!(d.highlight_line_spans(1).is_some());
        // Enter inserts a line: the cache must grow (splice old=1 -> new=2).
        d.set_selections(SelectionSet::new(2)); // end of "ab"
        d.enter();
        d.tokenize_highlight(d.buffer().line_count());
        assert_eq!(d.buffer().line_count(), 3);
        assert!(d.highlight_line_spans(2).is_some(), "the new last line is tokenized");
        // Backspace merges it back: the cache must shrink (splice old=2 -> new=1).
        d.backspace();
        d.tokenize_highlight(d.buffer().line_count());
        assert_eq!(d.buffer().line_count(), 2);
        assert!(d.highlight_line_spans(1).is_some());
        assert!(d.highlight_line_spans(2).is_none(), "no spans past the buffer");
    }

    #[test]
    fn multi_op_transaction_highlight_equals_a_fresh_document() {
        // A multi-caret transaction commits ALL its edits at once; rebase_views
        // must derive each edit's line span (old lines from the inverse text,
        // new lines from the post buffer) and coalesce same-line edits, then
        // invalidate exactly those lines. The swept highlight must equal a fresh
        // document on the identical text — a coordinate or coalescing error would
        // surface as a stale (un-re-tokenized) row here.
        const G: &str = "%YAML 1.2\n---\nname: T\nscope: source.t\ncontexts:\n  main:\n    - match: '\\w+'\n      scope: keyword.t\n";
        const TH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>settings</key><array><dict><key>settings</key><dict><key>foreground</key><string>#FFFFFF</string></dict></dict></array></dict></plist>"#;
        let make = |text: &str| -> Document {
            let mut d = doc(text);
            d.set_syntax(
                crate::SyntaxDef::from_sublime_syntax(G).unwrap(),
                crate::TokenTheme::from_tm_theme(TH).unwrap(),
            );
            d.tokenize_highlight(u32::MAX);
            d
        };
        // Lines: "aa bb cc"(0-7) \n(8) "dd ee ff"(9-16) \n(17) "gg hh ii"(18-25)
        //        \n(26) "jj kk ll"(27-34) \n(35) "mm nn oo"(36-43).
        let mut d = make("aa bb cc\ndd ee ff\ngg hh ii\njj kk ll\nmm nn oo");
        // One transaction: an edit on line 0, TWO edits on line 1 (must coalesce
        // to one line span), and a newline-bearing insert (a +1 line delta) —
        // all disjoint and ascending.
        d.edit_grouped(
            vec![
                EditOp::new(0..2, "XYZ"),     // line 0
                EditOp::new(9..11, "Q"),      // line 1, col 0
                EditOp::new(15..17, "RR"),    // line 1, col 6 (same line ⇒ coalesce)
                EditOp::insert(27, "NEW\n"),  // start of line 3 ⇒ inserts a line
            ],
            typ(OpClass::Type),
        )
        .unwrap();
        d.tokenize_highlight(u32::MAX);

        let text = d.text();
        let f = make(&text);
        assert_eq!(d.buffer().line_count(), f.buffer().line_count(), "line count");
        for r in 0..d.buffer().line_count() {
            assert_eq!(
                d.highlight_line_spans(r),
                f.highlight_line_spans(r),
                "row {r} live (per-edit invalidation) vs fresh"
            );
        }
    }

    /// A mid-session grammar swap must keep the retention window aimed at the
    /// viewport: the widget's deduped viewport report never re-fires when
    /// nothing visible moved, so the swapped cache must retain the currently
    /// visible rows rather than reset to the top and leave them fallback-styled.
    #[test]
    fn set_syntax_preserves_the_highlight_window_aim() {
        const G: &str = "%YAML 1.2\n---\nname: T\nscope: source.t\ncontexts:\n  main:\n    - match: '\\w+'\n      scope: keyword.t\n";
        const TH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>settings</key><array><dict><key>settings</key><dict><key>foreground</key><string>#FFFFFF</string></dict></dict></array></dict></plist>"#;
        let syntax = || crate::SyntaxDef::from_sublime_syntax(G).unwrap();
        let theme = || crate::TokenTheme::from_tm_theme(TH).unwrap();
        let mut d = doc(&"word\n".repeat(3_000));
        d.set_syntax(syntax(), theme());
        // The app scrolled deep: the window is aimed well past the default.
        d.set_highlight_window(2_000..2_040);
        // Grammar swap mid-session (re-highlight when the host swaps the language).
        d.set_syntax(syntax(), theme());
        while d.highlight_frontier().is_some() {
            d.tokenize_highlight(u32::MAX);
        }
        assert!(
            d.highlight_line_spans(2_020).is_some(),
            "the swapped cache must still retain the viewport rows"
        );
    }

    // Off-thread parallel/speculative highlight ingest. A stateless grammar
    // suffices at the Document seam — the stitch correctness lives in
    // highlight.rs's stateful oracle; here we check the seam's contract:
    // revision gating, dirt clearing, and no-checkpoint speculation.
    const HL_G: &str = "%YAML 1.2\n---\nname: T\nscope: source.t\ncontexts:\n  main:\n    - match: '\\bkw\\b'\n      scope: keyword.t\n";
    const HL_TH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>settings</key><array><dict><key>settings</key><dict><key>foreground</key><string>#FFFFFF</string></dict></dict></array></dict></plist>"#;

    fn doc_with_syntax(n: usize) -> Document {
        let mut d = doc(&"kw word\n".repeat(n));
        d.set_syntax(
            crate::SyntaxDef::from_sublime_syntax(HL_G).unwrap(),
            crate::TokenTheme::from_tm_theme(HL_TH).unwrap(),
        );
        d
    }

    #[test]
    fn absorb_highlight_drops_a_stale_revision() {
        let mut d = doc_with_syntax(2_000);
        let engine = d.highlight_engine().unwrap();
        let snap = d.snapshot();
        let stale_rev = d.revision();
        let seg = crate::tokenize_segment(&engine, &snap, 0..2_001, crate::SegmentStart::Fresh, None, None);
        // An edit lands after the snapshot was taken…
        d.set_selections(SelectionSet::new(0));
        d.type_char('x');
        assert_ne!(d.revision(), stale_rev);
        // …so the in-flight segment is dropped, absorbing nothing.
        assert!(!d.absorb_highlight(stale_rev, seg, true), "stale segment must be dropped");
        assert_eq!(d.highlight_frontier(), Some(0), "the frontier is untouched by a dropped absorb");
    }

    #[test]
    fn absorb_highlight_verified_clears_the_frontier() {
        let mut d = doc_with_syntax(5_000);
        let engine = d.highlight_engine().unwrap();
        let snap = d.snapshot();
        let rev = d.revision();
        let n = d.buffer().line_count();
        // A single verified segment over the whole document, carrying spans for
        // the whole range so no window gap remains either — then the frontier
        // (dirt ∨ window gap) is fully quiet, proving dirt was cleared.
        let seg =
            crate::tokenize_segment(&engine, &snap, 0..n, crate::SegmentStart::Fresh, Some(0..n), None);
        assert!(d.absorb_highlight(rev, seg, true));
        assert_eq!(d.highlight_frontier(), None, "verified absorb clears every dirty row");
    }

    #[test]
    fn absorb_highlight_speculative_shows_spans_but_keeps_dirt() {
        let mut d = doc_with_syntax(2_000);
        d.set_highlight_window(900..940);
        let engine = d.highlight_engine().unwrap();
        let snap = d.snapshot();
        let rev = d.revision();
        // A viewport-first speculative segment (Fresh guess, spans for the win).
        let seg = crate::tokenize_segment(
            &engine,
            &snap,
            772..940,
            crate::SegmentStart::Fresh,
            Some(900..940),
            None,
        );
        assert!(d.absorb_highlight(rev, seg, false));
        // Spans render immediately…
        assert!(d.highlight_line_spans(920).is_some(), "speculative spans are visible");
        // …but nothing is verified: the frontier still starts at row 0.
        assert_eq!(d.highlight_frontier(), Some(0), "speculation keeps the frontier");
        // Driving the sync frontier still converges to the correct colors.
        while d.highlight_frontier().is_some() {
            d.tokenize_highlight(u32::MAX);
        }
        assert!(d.highlight_line_spans(920).is_some());
    }

    #[test]
    fn diagnostics_install_then_ride_a_forward_edit() {
        use crate::{Diagnostic, DiagnosticsOutcome, Severity};
        let mut d = doc("let x = 1;");
        let rev = d.revision();
        let out = d.set_diagnostics(rev, vec![Diagnostic::new(4..5, Severity::Error, "unused")]);
        assert!(matches!(out, DiagnosticsOutcome::Applied { count: 1 }));
        let got: Vec<_> = d.diagnostics_in(0..100).collect();
        assert_eq!((got.len(), got[0].0.clone(), got[0].1), (1, 4..5, Severity::Error));
        // Insert 4 chars at the top → the squiggle (NeverGrows) rides to 8..9.
        d.edit(vec![EditOp::insert(0, "abcd")]).unwrap();
        assert_eq!(d.diagnostics_in(0..100).next().unwrap().0, 8..9, "rode via apply_patch");
    }

    #[test]
    fn stale_diagnostic_set_is_dropped_and_the_prior_kept() {
        use crate::{Diagnostic, DiagnosticsOutcome, Severity};
        let mut d = doc("abcdef");
        let rev = d.revision();
        d.set_diagnostics(rev, vec![Diagnostic::new(0..1, Severity::Warning, "w")]);
        d.edit(vec![EditOp::insert(0, "z")]).unwrap(); // revision advances
        let out = d.set_diagnostics(rev, vec![Diagnostic::new(3..4, Severity::Error, "e")]);
        assert!(matches!(out, DiagnosticsOutcome::Stale { .. }), "old-revision set dropped");
        let got: Vec<_> = d.diagnostics_in(0..100).collect();
        assert_eq!((got.len(), got[0].0.clone(), got[0].1), (1, 1..2, Severity::Warning));
    }

    #[test]
    fn diagnostic_spans_clip_to_the_buffer_length() {
        use crate::{Diagnostic, Severity};
        let mut d = doc("abc"); // len 3
        let rev = d.revision();
        d.set_diagnostics(rev, vec![Diagnostic::new(2..99, Severity::Error, "past eof")]);
        assert_eq!(d.diagnostics_in(0..100).next().unwrap().0, 2..3, "clipped to buffer len");
    }

    #[test]
    fn rend04_diagnostic_render_row_shifts_when_lines_inserted_above() {
        // A diagnostic deep in the document rides the code as whole lines are
        // inserted above it — its RENDER row (offset → point) shifts by the
        // inserted line count, still glued to the same text.
        use crate::{Diagnostic, Point, Severity};
        let mut d = doc("L0\nL1\nL2\nL3\nL4\nL5\nL6\nL7\nL8\nL9");
        let row5 = d.buffer().point_to_offset(Point { row: 5, col: 0 });
        d.set_diagnostics(d.revision(), vec![Diagnostic::new(row5..row5 + 2, Severity::Error, "e")]);
        let start = d.diagnostics_in(0..u32::MAX).next().unwrap().0.start;
        assert_eq!(d.buffer().offset_to_point(start).row, 5);
        // Insert three lines at the very top.
        d.edit(vec![EditOp::insert(0, "a\nb\nc\n")]).unwrap();
        let span = d.diagnostics_in(0..u32::MAX).next().unwrap().0;
        assert_eq!(d.buffer().offset_to_point(span.start).row, 8, "5 + 3 inserted lines");
        assert_eq!(&d.text()[span.start as usize..span.end as usize], "L5", "still glued to its code");
    }

    /// The windowed fold queries (the fold-gutter mouse hot paths) must agree
    /// with the whole-document ones on every row / window — else a hover/click
    /// near a fold would mis-decide. A windowed query that drops a headed pair or
    /// mis-clamps the byte range diverges here. Uses nested + inline + adjacent
    /// blocks so the partition-point windowing is exercised at boundaries.
    #[test]
    fn windowed_fold_queries_match_the_whole_document() {
        let mut d = doc(
            "top\n\
             a {\n  b [1, 2]\n  c {\n    d\n  }\n}\n\
             mid\n\
             e (x,\n  y)\n\
             f {\n  g\n}\n\
             end",
        );
        let n = d.buffer().line_count();
        // block_opener_on_row(r) windowed == the whole-document oracle.
        for row in 0..n {
            let want = d
                .foldable_pairs()
                .into_iter()
                .filter(|&(_, _, h, l)| h == row && l > h)
                .max_by_key(|&(_, _, h, l)| l - h)
                .map(|(o, ..)| o);
            assert_eq!(d.block_opener_on_row(row), want, "block_opener_on_row row {row}");
        }
        // collapsible_pairs_in_rows(0..n) == collapsible_pairs() (whole doc).
        assert_eq!(d.collapsible_pairs_in_rows(0..n), d.collapsible_pairs());
        // A sub-window returns exactly the whole-doc pairs headed inside it.
        for (a, b) in [(0u32, 3u32), (2, 6), (3, 4), (7, n)] {
            let want: Vec<_> =
                d.collapsible_pairs().into_iter().filter(|&(_, _, h, _)| a <= h && h < b).collect();
            assert_eq!(d.collapsible_pairs_in_rows(a..b), want, "collapsible_pairs_in_rows {a}..{b}");
        }
        // foldable_ranges_in_rows(r..r+1) headers agree with foldable_ranges().
        for row in 0..n {
            let windowed: Vec<_> =
                d.foldable_ranges_in_rows(row..row + 1).into_iter().filter(|(h, _)| h.0 == row).collect();
            let whole: Vec<_> =
                d.foldable_ranges().into_iter().filter(|(h, _)| h.0 == row).collect();
            assert_eq!(windowed, whole, "foldable_ranges row {row}");
        }
        // And it still holds after a fold collapses (folded pairs drop out).
        let opener = d.block_opener_on_row(1).unwrap();
        assert!(d.toggle_fold_opener(opener));
        assert_eq!(d.collapsible_pairs_in_rows(0..n), d.collapsible_pairs());
    }

    #[test]
    fn fold_survives_undo_redo() {
        // A fold rides the commit patch exactly like a decoration: an edit
        // above shifts it, undo restores its original rows, redo re-shifts — with
        // no fold-specific undo handling. The block is bracketed so it stays a
        // valid foldable range across the reconcile that runs on every edit.
        use crate::{fold_map::FoldMap, BufferRow};
        let mut d = doc("L0\nL1\nL2\nL3\nL4\nblk {\n  x\n  y\n}\nL9");
        assert_eq!(d.foldable_ranges().iter().map(|(h, l)| (h.0, l.0)).collect::<Vec<_>>(), vec![(5, 8)]);
        assert!(d.fold(BufferRow(5), BufferRow(8)));
        let header = |d: &Document| FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(5));
        let header8 = |d: &Document| FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(8));
        assert_eq!(header(&d), Some(BufferRow(8)));
        // Insert three lines at the top → fold shifts to header 8, last 11.
        d.edit(vec![EditOp::insert(0, "a\nb\nc\n")]).unwrap();
        assert_eq!(header8(&d), Some(BufferRow(11)));
        // Undo → fold back at 5..8. Redo → shifted again.
        assert!(d.undo());
        assert_eq!(header(&d), Some(BufferRow(8)));
        assert!(d.redo());
        assert_eq!(header8(&d), Some(BufferRow(11)));
    }

    #[test]
    fn pasting_after_the_collapsed_line_keeps_the_outer_fold() {
        use crate::{fold_map::FoldMap, BufferRow, SelectionSet};
        // Outer block (0,3) around a nested inner block (1,2).
        let mut d = doc("{\n\t{\n\t}\n}\n");
        assert_eq!(d.foldable_ranges().iter().map(|(h, l)| (h.0, l.0)).collect::<Vec<_>>(), vec![(0, 3), (1, 2)]);
        assert!(d.fold(BufferRow(0), BufferRow(3))); // fold the outer bracket
        // Copy the whole collapsed block `{…}` (offsets [0,9)) and paste it at the
        // end of the collapsed line (offset 9, just after the `}` on row 3).
        let block = d.buffer().text()[0..9].to_string();
        d.set_selections(SelectionSet::new(9));
        d.insert_text(&block);
        // The outer block's `{`…`}` is still on rows 0..3, so the fold must STAY on
        // (0,3) — not drop, and not grow onto a bogus range.
        assert_eq!(d.folds().len(), 1, "the outer fold survives the paste");
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(0)), Some(BufferRow(3)));
    }

    #[test]
    fn editing_after_the_brace_keeps_a_folded_block() {
        use crate::{fold_map::FoldMap, BufferRow, SelectionSet};
        let mut d = doc("{\n}");
        assert!(d.fold(BufferRow(0), BufferRow(1)));
        // Type arbitrary text after the `}` — the `{`…`}` block is intact.
        d.set_selections(SelectionSet::new(d.buffer().len()));
        d.insert_text("abc");
        assert_eq!(d.buffer().text(), "{\n}abc");
        assert_eq!(d.folds().len(), 1, "fold survives typing after the brace");
        // …and delete it back.
        let end = d.buffer().len();
        d.edit(vec![EditOp::delete(end - 3..end)]).unwrap();
        assert_eq!(d.buffer().text(), "{\n}");
        assert_eq!(d.folds().len(), 1, "fold survives deleting after the brace");
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(0)), Some(BufferRow(1)));
    }

    #[test]
    fn pressing_return_after_a_folded_eof_block_keeps_it_folded() {
        use crate::{fold_map::FoldMap, BufferRow, SelectionSet};
        // A fold created at EOF: `}` is the last line, so its interior has no
        // trailing `\n`.
        let mut d = doc("{\n}");
        assert!(d.fold(BufferRow(0), BufferRow(1)));
        // Caret at the very end (after `}`); press Enter (insert `\n`). The block
        // is untouched — a blank line is added after it — so the fold must STAY.
        d.set_selections(SelectionSet::new(d.buffer().len()));
        d.insert_text("\n");
        assert_eq!(d.buffer().text(), "{\n}\n");
        assert_eq!(d.folds().len(), 1, "the intact block stays folded");
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(0)), Some(BufferRow(1)));
    }

    #[test]
    fn cutting_trailing_content_keeps_an_intact_fold() {
        use crate::{fold_map::FoldMap, BufferRow};
        // rows: 0 "{" 1 "}" 2 "X"
        let mut d = doc("{\n}\nX");
        assert!(d.fold(BufferRow(0), BufferRow(1))); // fold the set
        // Select from after X (offset 5) to the end of the `}` (offset 3) and cut:
        // deletes the `}` line's trailing `\n` + `X`. The `{`…`}` block is intact,
        // so the fold must STAY.
        let a = d.buffer().point_to_offset(crate::Point::new(1, 1)); // 3
        let b = d.buffer().point_to_offset(crate::Point::new(2, 1)); // 5
        d.edit(vec![EditOp::delete(a..b)]).unwrap();
        assert_eq!(d.buffer().text(), "{\n}", "only the trailing content was removed");
        assert_eq!(d.folds().len(), 1, "the intact block stays folded");
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(0)), Some(BufferRow(1)));
    }

    #[test]
    fn cutting_across_a_folds_boundary_drops_it_not_drifts_it() {
        use crate::{fold_map::FoldMap, BufferRow};
        // Three sibling blocks. rows: 0 "{" 1 "}" 2 "{" 3 "}" 4 "{" 5 "}" 6 ""
        let mut d = doc("{\n}\n{\n}\n{\n}\n");
        assert!(d.fold(BufferRow(2), BufferRow(3))); // fold the MIDDLE set
        // Select from the end of the first `}` (offset 3) to the end of the middle
        // `}` (offset 7) and cut it — a delete that crosses the fold's interior
        // start. The whole middle block is removed.
        let a = d.buffer().point_to_offset(crate::Point::new(1, 1)); // 3
        let b = d.buffer().point_to_offset(crate::Point::new(3, 1)); // 7
        d.edit(vec![EditOp::delete(a..b)]).unwrap();
        // The fold must DROP — not drift onto the first set (which is itself a
        // valid foldable range, so reconcile can't catch the drift).
        assert!(d.folds().is_empty(), "fold dropped, not drifted onto the first block");
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).fold_at_header(BufferRow(0)), None, "first set not folded");
    }

    #[test]
    fn typing_at_many_carets_over_folds_stays_linear() {
        use crate::bracket::ENCLOSING_WALKS;
        use crate::row_layout::DISPLAY_POSITION_PROBES;
        use crate::{Motion, SelectionSet};
        // Select every `fn pid` occurrence, fold each block, then type.
        // `expand_folds_touched` is windowed, so per commit it does ONE enclosing
        // walk (for the first edit point) and O(edit points) display-position
        // probes, independent of caret count — never O(carets) leftward enclosing
        // walks plus an O(carets²) per-candidate edit scan. Gates both: an
        // un-windowed implementation trips these by a factor of N.
        let mut src = String::new();
        for i in 0..400 {
            src.push_str(&format!(
                "// block {i}\nfn pid_{i}(pid: u8) -> u8 {{\n    switch pid {{\n        0x0D => {{ return 88; }}\n    }}\n}}\n\n"
            ));
        }
        let mut d = doc(&src);
        let first = d.buffer().text().find("fn pid").unwrap() as u32;
        d.set_selections(SelectionSet::from_ranges(&[(first, first + 6)], 0));
        d.select_all_occurrences();
        let carets = d.selections().all().len();
        assert!(carets >= 400, "select-all found every block ({carets} carets)");
        d.move_carets(Motion::LineEnd, false);
        d.fold_at_carets(false);
        d.move_carets(Motion::LineEnd, false);

        // Measure ONLY the commit for the typed character.
        ENCLOSING_WALKS.with(|c| c.set(0));
        DISPLAY_POSITION_PROBES.with(|c| c.set(0));
        d.insert_text("a");
        let walks = ENCLOSING_WALKS.with(std::cell::Cell::get);
        let probes = DISPLAY_POSITION_PROBES.with(std::cell::Cell::get);
        assert!(walks <= 2, "one enclosing walk per commit, not O(carets): {walks} for {carets} carets");
        assert!(
            probes <= 4 * carets as u64,
            "display-position probes are O(edit points), not O(carets²): {probes} for {carets} carets"
        );
    }

    #[test]
    fn folded_keystroke_never_resolves_bracket_depth() {
        use crate::bracket_tree::BRACKET_VIEW_CALLS;
        use crate::SelectionSet;
        // Fold every function block, then type. The edit-path fold queries —
        // expand-on-hidden-edit, reconcile, the FoldMap rebuild — must resolve
        // each fold's partner via the depth-free `foldable_partner`, NEVER
        // `bracket_view`, whose per-opener prefix-stack Vec (allocated at every
        // tree level) would turn a fold-heavy keystroke into an O(folds · depth)
        // allocation storm. `bracket_view` is a DRAW-path (colorization)
        // primitive only; a single fold-edit commit must leave the
        // depth-resolution canary at zero regardless of how many folds are open.
        // Resolving partners through `at().partner` → `bracket_view` would read
        // ~2·folds; the depth-free path pins it to zero.
        let mut src = String::new();
        for i in 0..300 {
            src.push_str(&format!("fn f_{i}() {{\n    body\n}}\n"));
        }
        let mut d = doc(&src);
        // The block `{` of each function (the `()` params are single-line, not folds).
        let openers: Vec<u32> = d
            .brackets()
            .all()
            .iter()
            .filter(|b| b.open && d.buffer().char_at(b.offset) == Some('{'))
            .map(|b| b.offset)
            .collect();
        assert_eq!(openers.len(), 300, "one foldable block opener each");
        for &o in &openers {
            assert!(d.toggle_fold_opener(o));
        }
        // A caret just after each (visible) header `{` — the "hit END, type" position.
        let carets: Vec<(u32, u32)> = openers.iter().map(|&o| (o + 1, o + 1)).collect();
        d.set_selections(SelectionSet::from_ranges(&carets, 0));
        let _ = d.fold_map(); // warm the map so the measured commit is steady-state

        BRACKET_VIEW_CALLS.with(|c| c.set(0));
        d.insert_text("a");
        let _ = d.fold_map(); // the per-keystroke refresh rides the same lookups
        let depth_resolves = BRACKET_VIEW_CALLS.with(std::cell::Cell::get);
        assert_eq!(
            depth_resolves, 0,
            "a fold-heavy keystroke resolved bracket depth {depth_resolves} times over 300 folds \
             — the edit path must use foldable_partner, not at()/bracket_view"
        );
    }

    #[test]
    fn single_line_pair_folds_inline() {
        use crate::{fold_map::FoldMap, SelectionSet};
        // one line: `[` at 4, `]` at 12.
        let mut d = doc("x = [1, 2, 3]");
        d.set_selections(SelectionSet::new(6)); // caret inside the array
        let opener = d.fold_opener_at_caret(false).expect("an enclosing foldable pair");
        assert_eq!(opener, 4);
        assert!(d.toggle_fold_opener(opener));
        let m = FoldMap::new(d.folds(), d.brackets(), d.buffer());
        assert_eq!(m.display_row_count(), 1, "an inline fold hides no rows");
        assert_eq!(m.inline_folds().len(), 1);
        assert_eq!((m.inline_folds()[0].row, m.inline_folds()[0].open, m.inline_folds()[0].close), (0, 4, 12));
        // Unfold via the same opener.
        assert!(d.toggle_fold_opener(4));
        assert!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).inline_folds().is_empty());
    }

    #[test]
    fn fold_at_carets_acts_on_every_caret() {
        use crate::{fold_map::FoldMap, SelectionSet};
        // Two inline blocks: {x} at 1..3 and {y} at 6..8.
        let mut d = doc("a{x}\nb{y}");
        d.set_selections(SelectionSet::from_offsets(&[2, 7])); // a caret inside each
        assert!(d.fold_at_carets(false), "Ctrl+Shift+[ folds at every caret");
        let m = FoldMap::new(d.folds(), d.brackets(), d.buffer());
        assert_eq!(m.inline_folds().len(), 2, "both blocks collapsed, not just the primary");
        // …and unfold at every caret.
        assert!(d.fold_at_carets(true), "Ctrl+Shift+] unfolds at every caret");
        assert!(
            FoldMap::new(d.folds(), d.brackets(), d.buffer()).inline_folds().is_empty(),
            "both blocks expanded",
        );
    }

    #[test]
    fn fold_map_cache_matches_a_fresh_build_across_changes() {
        // The memoized fold_map() must DEEP-EQUAL a from-scratch build. It is
        // shifted in place on a no-line-change edit, so the drift risk is an
        // inline fold's offsets diverging from a fresh resolve — this exercises
        // both a block and an inline fold across folds, plain typing (in-place
        // shift, cumulative), and a newline (line change → rebuild fallback).
        fn consistent(d: &Document) -> bool {
            *d.fold_map() == crate::fold_map::FoldMap::new(d.folds(), d.brackets(), d.buffer())
        }
        let mut d = doc("a {\nx\n}\nb = [1, 2, 3]\nc\n");
        assert!(consistent(&d), "empty folds");
        // Fold a block AND an inline pair (the inline offsets are the shift risk).
        let block = d.text().find('{').unwrap() as u32;
        let inline = d.text().find('[').unwrap() as u32;
        d.toggle_fold_opener(block);
        d.toggle_fold_opener(inline);
        assert!(consistent(&d), "after folds — rebuilt on the generation");
        // A plain char at offset 0 (visible, before both folds) shifts the inline
        // fold's offsets; the cache is shifted in place, not rebuilt.
        d.edit(vec![EditOp::insert(0, "Z")]).unwrap();
        assert!(consistent(&d), "after a no-line-change edit — shifted in place");
        d.edit(vec![EditOp::insert(0, "Y")]).unwrap();
        assert!(consistent(&d), "after a second in-place shift (cumulative)");
        // A newline changes the line count → the cache falls back to a rebuild.
        d.edit(vec![EditOp::insert(0, "\n")]).unwrap();
        assert!(consistent(&d), "after a line change — rebuilt");
        d.edit(vec![EditOp::insert(0, "W")]).unwrap();
        assert!(consistent(&d), "after an edit following a rebuild");
    }

    #[test]
    fn deleting_into_a_folded_interior_rebuilds_not_drifts() {
        // A deletion whose OLD range reached a folded block's hidden interior
        // (the newline ending the header line) is NOT caught by `expand` (which
        // probes only the collapsed new endpoint, landing on a visible row), so
        // the incremental mover must detect the span change and rebuild rather
        // than rigidly shift — a rigid shift would diverge from a fresh build, and
        // underflow `vgap` for a fold headed on row 0.
        let consistent = |d: &Document| {
            *d.fold_map() == crate::fold_map::FoldMap::new(d.folds(), d.brackets(), d.buffer())
        };
        for prefix in ["xx\n", ""] {
            // With "xx\n" the fold heads on row 1; with "" it heads on row 0 (vgap 0,
            // the underflow variant).
            let mut d = doc(&format!("{prefix}fn foo() {{\n  body\n}}\nafter\n"));
            let open = d.text().find('{').unwrap() as u32;
            d.toggle_fold_opener(open);
            let _ = d.fold_map();
            assert!(consistent(&d), "{prefix:?}: folded");
            let nl = open + d.text()[open as usize..].find('\n').unwrap() as u32;
            d.edit(vec![EditOp::delete(nl..nl + 1)]).unwrap(); // delete the header newline
            assert!(consistent(&d), "{prefix:?}: after deleting the header-terminating newline");
        }
    }

    #[test]
    fn fold_map_cache_matches_fresh_build_under_random_edits() {
        // The drift oracle as a random walk: the incrementally-shifted fold cache
        // must DEEP-EQUAL a from-scratch build after EVERY edit — line inserts and
        // deletes at arbitrary offsets (the new O(log) block-region reanchor + the
        // inline row/offset shift), multi-line inserts, and edits that land in a
        // fold interior (expand → generation rebuild). All paths must stay consistent.
        let consistent = |d: &Document| {
            *d.fold_map() == crate::fold_map::FoldMap::new(d.folds(), d.brackets(), d.buffer())
        };
        let mut state = 0xF01D_5EEDu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut d = doc(&"fn f() {\n  x = [1, 2]\n  y\n}\n".repeat(6));
        for open in d.collapsible_pairs().iter().map(|&(o, ..)| o).collect::<Vec<_>>() {
            d.toggle_fold_opener(open);
        }
        assert!(consistent(&d), "after collapse-all");
        for step in 0..1500 {
            let len = d.buffer().len();
            if len == 0 {
                break;
            }
            let at = next() % (len + 1);
            let _ = match next() % 6 {
                0 => d.edit(vec![EditOp::insert(at, "\n")]),       // single-line insert
                1 => d.edit(vec![EditOp::insert(at, "z")]),        // plain char, no line change
                2 => d.edit(vec![EditOp::insert(at, "ab\ncd")]),   // multi-line insert
                3 => d.edit(vec![EditOp::insert(at, "  ")]),       // whitespace, no line change
                _ if at < len => {
                    let end = (at + 1 + next() % 5).min(len); // delete (may cross a newline)
                    d.edit(vec![EditOp::delete(at..end)])
                }
                _ => d.edit(vec![EditOp::insert(at, "q")]),
            };
            assert!(consistent(&d), "step {step}: cache drifted from a fresh build (at {at})");
        }
    }

    #[test]
    fn keystroke_and_arrow_do_not_rebuild_the_fold_map_at_scale() {
        // The per-keystroke fold GATE (the analog of the widget draw-budget gate):
        // with a document-scale set collapsed, a plain typed character and an
        // arrow key must NOT rebuild the whole FoldMap (O(folds)) — the cache is
        // shifted in place / read, never rebuilt. A per-keystroke `FoldMap::new`
        // (a common source of typing lag) would trip this.
        use crate::fold_map::FOLD_BUILDS;
        let mut text = String::new();
        for i in 0..300 {
            text.push_str(&format!("fn f{i}() {{\n    body\n}}\n"));
        }
        let mut d = doc(&text);
        let opens: Vec<u32> = d.collapsible_pairs().into_iter().map(|(o, ..)| o).collect();
        assert!(opens.len() >= 300, "every block is foldable");
        d.set_selections(SelectionSet::from_offsets(&opens));
        d.fold_at_carets(false); // collapse everything
        d.set_selections(SelectionSet::new(0)); // caret at the visible top
        let _ = d.fold_map(); // warm the cache

        // A plain keystroke at a visible position, then a render-style map read.
        let base = FOLD_BUILDS.with(std::cell::Cell::get);
        d.type_char('x');
        let _ = d.fold_map();
        assert_eq!(
            FOLD_BUILDS.with(std::cell::Cell::get) - base,
            0,
            "a keystroke over {} folds must not rebuild the FoldMap",
            opens.len()
        );

        // An arrow key (movement reads the memoized map, not a throwaway build).
        let base = FOLD_BUILDS.with(std::cell::Cell::get);
        d.move_carets(Motion::Down, false);
        let _ = d.fold_map();
        assert_eq!(
            FOLD_BUILDS.with(std::cell::Cell::get) - base,
            0,
            "an arrow key over {} folds must not rebuild the FoldMap",
            opens.len()
        );

        // Adding a NEWLINE at the start changes the line count — the
        // "insert lines at the top with everything folded" case. The incremental
        // region-tree reanchor shifts every fold's rows in O(log) at one seam, so
        // it must NOT rebuild. A line-change fallback that rebuilt the whole
        // O(folds) FoldMap every keystroke would trip this.
        d.set_selections(SelectionSet::new(0));
        let base = FOLD_BUILDS.with(std::cell::Cell::get);
        d.edit(vec![EditOp::insert(0, "\n")]).unwrap();
        let _ = d.fold_map();
        assert_eq!(
            FOLD_BUILDS.with(std::cell::Cell::get) - base,
            0,
            "a newline at the top over {} folds must not rebuild the FoldMap",
            opens.len()
        );
    }

    #[test]
    fn keystroke_with_document_scale_decorations_does_not_resort_the_store() {
        // The per-keystroke DECORATION gate: typing with a document-scale
        // decoration set present must NOT re-sort the whole store — apply_patch
        // shifts in place (a monotonic patch keeps order) and the autoclose scans
        // hit only the (here empty) own auto-close store. An O(D log D) full-store
        // sort every commit would trip this.
        use crate::decorations::DECORATION_SORTS;
        use crate::{Diagnostic, Severity};
        let mut d = doc(&"let x = 1;\n".repeat(2000)); // 22k bytes
        let rev = d.revision();
        let diags: Vec<Diagnostic> = (0..2000u32)
            .map(|i| Diagnostic::new(i * 10..i * 10 + 3, Severity::Warning, "w"))
            .collect();
        d.set_diagnostics(rev, diags); // publishes (sorts once, here — not per keystroke)
        d.set_selections(SelectionSet::new(0));

        let base = DECORATION_SORTS.with(std::cell::Cell::get);
        d.type_char('z');
        assert_eq!(
            DECORATION_SORTS.with(std::cell::Cell::get) - base,
            0,
            "a keystroke with 2000 diagnostics must not re-sort the decoration store"
        );
    }

    #[test]
    fn reconcile_scans_the_fold_set_only_when_a_fold_actually_breaks() {
        // reconcile drops folds whose pair an edit broke. Two tightenings are gated
        // here. (1) A non-bracket edit — a letter, or an Enter (a line change with
        // no bracket char) — leaves every pairing intact, so it must not touch the
        // fold set at all. (2) Even a BRACKET edit that breaks no existing fold must
        // not scan (retain) the whole fold set: the windowed reconcile checks only
        // the folds its re-matched region covers and mutates nothing when none
        // broke. Only an edit that actually breaks a fold pays the O(folds) removal.
        use crate::fold_map::RECONCILE_SCANS;
        let mut d = doc("fn f() {\n  body\n}\ntail\n");
        let open = d.text().find('{').unwrap() as u32;
        assert!(d.toggle_fold_opener(open));
        d.set_selections(SelectionSet::new(0));

        // "a"/"\n": no bracket char → reconcile never runs. "{": a bracket edit
        // that re-pairs but breaks no fold (the block's own pair stays matched) →
        // the windowed reconcile inspects the fold and drops nothing, so still no
        // whole-set scan.
        for text in ["a", "\n", "{"] {
            let base = RECONCILE_SCANS.with(std::cell::Cell::get);
            d.edit(vec![EditOp::insert(0, text)]).unwrap();
            assert_eq!(
                RECONCILE_SCANS.with(std::cell::Cell::get) - base,
                0,
                "inserting {text:?} breaks no fold, so it must not scan the fold set"
            );
        }

        // Deleting the block's `}` breaks its pair → the fold must drop → one scan.
        let close = d.text().rfind('}').unwrap() as u32;
        let base = RECONCILE_SCANS.with(std::cell::Cell::get);
        d.edit(vec![EditOp::delete(close..close + 1)]).unwrap();
        assert!(
            RECONCILE_SCANS.with(std::cell::Cell::get) - base >= 1,
            "deleting a brace that breaks a fold must reconcile the fold set"
        );
        assert!(d.folds().is_empty(), "the broken fold was dropped");
    }

    #[test]
    fn windowed_reconcile_matches_the_whole_set_reconcile_under_random_edits() {
        use crate::SelectionSet;
        // The edit-path windowed reconcile must drop EXACTLY the folds the whole-set
        // reconcile would. Oracle: after each random bracket-heavy edit, an extra
        // whole-set reconcile must be a no-op (find nothing more to drop). A window
        // that misses a broken fold — e.g. an enclosing fold whose partner an edit
        // deletes far from its opener, so the post-edit brackets no longer flag it
        // as enclosing — diverges here immediately. A naive window that omits the
        // seed-stack (enclosing) left edge would fail this.
        let mut src = String::new();
        for i in 0..40 {
            src.push_str(&format!("block{i} {{\n  a\n  b\n}}\n"));
        }
        let mut d = doc(&src);
        for open in d.collapsible_pairs().iter().map(|&(o, ..)| o).collect::<Vec<_>>() {
            d.toggle_fold_opener(open);
        }
        // Deterministic LCG — no rand crate, no Date::now (both banned in tests).
        let mut state: u32 = 0x1234_5678;
        let mut rng = || {
            state = state.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            state
        };
        for _ in 0..300 {
            let len = d.buffer().len();
            if len == 0 {
                break;
            }
            let pos = rng() % len;
            match rng() % 5 {
                0 => {
                    d.set_selections(SelectionSet::new(pos));
                    d.insert_text("{");
                }
                1 => {
                    d.set_selections(SelectionSet::new(pos));
                    d.insert_text("}");
                }
                2 if len > 1 => {
                    // Delete one byte, char-snapped (the doc is ASCII, so pos is a
                    // boundary), biased toward braces to break pairs often.
                    let s = pos.min(len - 1);
                    d.edit(vec![EditOp::delete(s..s + 1)]).unwrap();
                }
                _ => {
                    d.set_selections(SelectionSet::new(pos));
                    d.insert_text("x");
                }
            }
            let before: Vec<u32> = d.folds().iter().collect();
            d.force_whole_reconcile();
            let after: Vec<u32> = d.folds().iter().collect();
            assert_eq!(before, after, "windowed reconcile missed a fold the whole-set pass caught");
            // Every surviving fold is a genuine foldable pair — no orphan left behind.
            for o in after {
                let foldable = d
                    .brackets()
                    .at(o)
                    .filter(|b| b.open)
                    .and_then(|b| b.partner)
                    .is_some_and(|c| crate::row_layout::pair_has_interior(o, c));
                assert!(foldable, "a surviving fold at {o} is not a foldable pair");
            }
        }
    }

    #[test]
    fn fold_at_carets_ejects_carets_hidden_by_a_block() {
        use crate::SelectionSet;
        let mut d = doc("a {\nhidden\n}\nb"); // block: { at 2, } at 11 (rows 0..=2)
        let inside = d.buffer().point_to_offset(Point::new(1, 3)); // inside "hidden"
        d.set_selections(SelectionSet::new(inside));
        assert!(d.fold_at_carets(false), "the block folds");
        // The caret sat on a now-hidden interior row — the batched ejection must
        // pull it to the visible header row, never leave it on invisible text.
        let row = d.buffer().offset_to_point(d.selections().newest().head()).row;
        assert_eq!(row, 0, "caret ejected to the header row");
    }

    #[test]
    fn horizontal_movement_hops_an_inline_fold() {
        use crate::{Motion, SelectionSet};
        let mut d = doc("x = [1, 2, 3]"); // `[` at 4, `]` at 12
        d.set_selections(SelectionSet::new(6)); // inside the array
        let op = d.fold_opener_at_caret(false).unwrap();
        assert!(d.toggle_fold_opener(op));
        assert_eq!(d.selections().newest().head(), 5, "caret snapped just after `[`");
        d.move_carets(Motion::Right, false);
        assert_eq!(d.selections().newest().head(), 12, "Right hops the collapsed interior to `]`");
        d.move_carets(Motion::Left, false);
        assert_eq!(d.selections().newest().head(), 5, "Left hops back to just after `[`");
    }

    #[test]
    fn collapsible_at_prefers_the_innermost() {
        // A block `{ … }` containing an inline `[ … ]` — nested collapsibles.
        let d = doc("fn m() {\n    let x = [1, 2, 3];\n}");
        let mut pairs = d.collapsible_pairs();
        pairs.sort();
        assert_eq!(pairs, vec![(7, 32, 0, 2), (21, 29, 1, 1)]);
        // Inside the array → the tighter inline pair wins over its enclosing block.
        assert_eq!(d.collapsible_at(24), Some(21));
        // Inside the block but left of the array → the block.
        assert_eq!(d.collapsible_at(12), Some(7));
        // On the `fn` header, before the `{` → nothing collapsible.
        assert_eq!(d.collapsible_at(0), None);
    }

    #[test]
    fn collapsible_excludes_the_already_folded() {
        let mut d = doc("fn m() {\n    let x = [1, 2, 3];\n}");
        // Fold the block; it drops out of the candidate set (nothing left to
        // collapse), but the inline array inside stays reachable.
        assert!(d.toggle_fold_opener(7));
        let openers: Vec<u32> = d.collapsible_pairs().iter().map(|&(o, ..)| o).collect();
        assert_eq!(openers, vec![21]);
        assert_eq!(d.collapsible_at(12), None, "the folded block is no longer a target");
        assert_eq!(d.collapsible_at(24), Some(21), "the array inside it still is");
    }

    #[test]
    fn nested_inline_folds_reduce_to_the_outer() {
        use crate::fold_map::FoldMap;
        // `(` at 10, `[` at 14, `]` at 22, `)` at 23 — the params contain the array.
        let mut d = doc("outer_call(a, [1, 2, 3])");
        assert!(d.toggle_fold_opener(14)); // collapse the array first
        assert!(d.toggle_fold_opener(10)); // then the outer_call params around it
        let inline = |d: &Document| {
            FoldMap::new(d.folds(), d.brackets(), d.buffer()).inline_folds().iter().map(|f| f.open).collect::<Vec<_>>()
        };
        // Only the outer `( … )` chip renders; the array is hidden inside it.
        assert_eq!(inline(&d), vec![10]);
        // Expanding the params reveals the array, still collapsed (state preserved).
        assert!(d.toggle_fold_opener(10));
        assert_eq!(inline(&d), vec![14]);
    }

    #[test]
    fn horizontal_movement_hops_the_collapsed_gap() {
        use crate::{BufferRow, Motion, Point, SelectionSet};
        // rows: 0 "m {", 1 "  a", 2 "  b", 3 "}", 4 "after"
        let mut d = doc("m {\n  a\n  b\n}\nafter");
        assert!(d.fold(BufferRow(0), BufferRow(3))); // collapse; the tail is row 3 `}`
        let caret = |d: &Document| d.buffer().offset_to_point(d.selections().newest().head());
        // Caret at the end of the header line "m {" (row 0, col 3).
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(0, 3))));
        d.move_carets(Motion::Right, false);
        assert_eq!(caret(&d), Point::new(3, 0), "Right hops the hidden interior to the closing bracket");
        d.move_carets(Motion::Left, false);
        assert_eq!(caret(&d), Point::new(0, 3), "Left hops back to the header end");
    }

    #[test]
    fn vertical_move_skips_folds() {
        use crate::{BufferRow, Motion, Point, SelectionSet};
        let mut d = doc("L0\n{\nL2\nL3\n}\nL5");
        assert!(d.fold(BufferRow(1), BufferRow(4))); // { on row 1 … } on row 4 — hide rows 2,3,4
        let row = |d: &Document| d.buffer().offset_to_point(d.selections().newest().head()).row;
        // Caret on the header row L1; Down lands on L5 (the fold interior is skipped).
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(1, 0))));
        d.move_carets(Motion::Down, false);
        assert_eq!(row(&d), 5);
        // Up from L5 lands back on the header L1, never inside the fold.
        d.move_carets(Motion::Up, false);
        assert_eq!(row(&d), 1);
    }

    #[test]
    fn typing_into_an_inline_folds_gap_expands_it() {
        use crate::SelectionSet;
        // `x = [1, 2, 3]` — fold the pair; the caret is pulled to just after
        // `[` (offset 5). Typing there would insert a CHIP-HIDDEN character,
        // so the edit expands the fold to reveal itself.
        let mut d = doc("x = [1, 2, 3]");
        d.set_selections(SelectionSet::new(6));
        assert!(d.toggle_fold_opener(4));
        assert_eq!(d.selections().newest().head(), 5, "caret ejected to the left edge");
        d.type_char('9');
        assert!(!d.folds().is_folded(4), "editing hidden text expands the fold");
        assert_eq!(d.text(), "x = [91, 2, 3]");
    }

    #[test]
    fn typing_at_a_collapsed_headers_end_keeps_it_folded() {
        use crate::{Point, SelectionSet};
        // Rows: 0 "m {", 1 "  a", 2 "}" — fold, caret ejected to the header's
        // end (a VISIBLE position, just before the `…`): typing there edits
        // the header line in plain sight, so the fold stays collapsed.
        let mut d = doc("m {\n  a\n}");
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(1, 0))));
        let opener = 2;
        assert!(d.toggle_fold_opener(opener));
        d.type_char('x'); // header line becomes "m {x"... still one visible edit
        assert!(d.folds().is_folded(opener), "a visible-part edit leaves the fold alone");
        assert!(d.text().starts_with("m {x\n"));
    }

    #[test]
    fn undo_into_a_folded_region_expands_it() {
        use crate::{Point, SelectionSet};
        // Type inside the (unfolded) block, then fold, then undo: the revert
        // touches now-hidden text, so the fold expands to show it — the same
        // reveal rule as forward edits, through the same seam.
        let mut d = doc("m {\n  a\n}");
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(1, 3))));
        d.type_char('b'); // "  ab" on row 1
        let opener = 2;
        assert!(d.toggle_fold_opener(opener));
        assert!(d.folds().is_folded(opener));
        assert!(d.undo());
        assert!(!d.folds().is_folded(opener), "undoing hidden text reveals it");
        assert!(d.text().contains("\n  a\n"), "the typed char was reverted");
    }

    #[test]
    fn folding_a_block_ejects_a_hidden_caret_to_the_header_end() {
        use crate::{Point, SelectionSet};
        // Rows: 0 "m {", 1 "  a", 2 "  }" — the `{` at offset 2.
        let mut d = doc("m {\n  a\n  }");
        let opener = 2;
        // A caret on an interior row is ejected to the header line's end…
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(1, 2))));
        assert!(d.toggle_fold_opener(opener));
        assert_eq!(
            d.selections().newest().head(),
            d.buffer().point_to_offset(Point::new(0, 3)),
            "the hidden caret ejects to just before the placeholder"
        );
        assert!(d.toggle_fold_opener(opener)); // unfold
        // …a caret in the tail row's leading whitespace (also hidden) ejects…
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(2, 1))));
        assert!(d.toggle_fold_opener(opener));
        assert_eq!(d.selections().newest().head(), d.buffer().point_to_offset(Point::new(0, 3)));
        assert!(d.toggle_fold_opener(opener)); // unfold
        // …but a caret ON the visible tail (the `}`) stays exactly put.
        let on_tail = d.buffer().point_to_offset(Point::new(2, 2));
        d.set_selections(SelectionSet::new(on_tail));
        assert!(d.toggle_fold_opener(opener));
        assert_eq!(d.selections().newest().head(), on_tail, "the visible tail is not ejected");
    }

    #[test]
    fn fold_chord_works_when_the_caret_touches_a_bracket() {
        use crate::SelectionSet;
        // `x = [1, 2, 3]` — `[` at 4, `]` at 12.
        let mut d = doc("x = [1, 2, 3]");
        // Caret just BEFORE the `[` (offset 4): touching the opener folds.
        d.set_selections(SelectionSet::new(4));
        assert_eq!(d.fold_opener_at_caret(false), Some(4), "touching the opener from the left");
        // Caret just AFTER the `]` (offset 13): touching the closer folds.
        d.set_selections(SelectionSet::new(13));
        assert_eq!(d.fold_opener_at_caret(false), Some(4), "touching the closer from the right");
        // …and once collapsed, the same touching positions unfold it.
        assert!(d.toggle_fold_opener(4));
        assert_eq!(d.fold_opener_at_caret(true), Some(4), "touching unfolds too");
        // Away from the pair entirely: nothing to fold.
        assert!(d.toggle_fold_opener(4));
        d.set_selections(SelectionSet::new(1));
        assert_eq!(d.fold_opener_at_caret(false), None);
    }

    #[test]
    fn fold_chord_at_a_shared_boundary_picks_the_touched_inner_pair() {
        use crate::SelectionSet;
        // `{a [b] c}` — outer `{` 0 / `}` 8, inner `[` 3 / `]` 5. A caret just
        // after the inner `]` (offset 6) is inside the block AND touching the
        // inline pair — the tighter, touched pair wins.
        let d = {
            let mut d = doc("{a [b] c}");
            d.set_selections(SelectionSet::new(6));
            d
        };
        assert_eq!(d.fold_opener_at_caret(false), Some(3), "the touched inline pair beats the enclosing block");
    }

    #[test]
    fn vertical_goal_is_a_display_cell_across_tab_stops() {
        use crate::{Motion, Point, SelectionSet};
        // Row 0: a tab then "ab" — the caret after the tab sits at CELL 4.
        // Row 1: plain "xxxxxxxxxx". The vertical goal is the visual column, so
        // it lands at cell 4 = col 4, not the byte column (col 1) far to the left.
        let mut d = doc("\tab\nxxxxxxxxxx");
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(0, 1))));
        d.move_carets(Motion::Down, false);
        assert_eq!(
            d.buffer().offset_to_point(d.selections().newest().head()),
            Point::new(1, 4),
            "the goal is the visual column, not the byte column"
        );
        // And back up: cell 4 re-resolves to just after the tab (col 1).
        d.move_carets(Motion::Up, false);
        assert_eq!(d.buffer().offset_to_point(d.selections().newest().head()), Point::new(0, 1));
    }

    #[test]
    fn vertical_move_through_a_collapsed_tail_stays_visual() {
        use crate::{BufferRow, Motion, Point, SelectionSet};
        // Rows: 0 "m {", 1 "a", 2 "} t", 3 "abcdefgh"; fold rows 0..=2. The
        // collapsed line reads `m { … } t` — the tail renders at cells 7…10.
        let mut d = doc("m {\na\n} t\nabcdefgh");
        assert!(d.fold(BufferRow(0), BufferRow(2)));
        // Caret at the tail's end (row 2, col 3) — display cell 10.
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(2, 3))));
        d.move_carets(Motion::Down, false);
        assert_eq!(
            d.buffer().offset_to_point(d.selections().newest().head()),
            Point::new(3, 8),
            "Down from the far-right tail lands at the row's visual end, not byte col 3"
        );
        // Up from cell 8 lands back ON the visible tail (row 2), not the header text.
        d.move_carets(Motion::Up, false);
        assert_eq!(
            d.buffer().offset_to_point(d.selections().newest().head()).row,
            2,
            "a goal over the placeholder tail resolves onto the tail"
        );
    }

    #[test]
    fn vertical_landing_snaps_out_of_a_collapsed_inline_gap() {
        use crate::{Motion, Point, SelectionSet};
        // Row 0 plain; row 1 has `[123456]` collapsed — its chip spans cells
        // 2..5. A goal cell on the chip snaps to the landable left edge,
        // never a hidden interior slot.
        let mut d = doc("aaaaaaaaaa\nx[123456]y");
        let opener = d.buffer().text().find('[').unwrap() as u32;
        assert!(d.toggle_fold_opener(opener));
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(0, 3))));
        d.move_carets(Motion::Down, false);
        assert_eq!(
            d.selections().newest().head(),
            opener + 1,
            "a landing on the chip snaps to just after the opening bracket"
        );
    }

    #[test]
    fn home_and_end_operate_on_the_collapsed_display_line() {
        use crate::{BufferRow, Motion, Point, SelectionSet};
        // Rows: 0 "L0", 1 "  m {", 2 "a", 3 "  }" — the collapsed display line
        // reads `  m { … }`, with the tail `}` a real position on row 3.
        let mut d = doc("L0\n  m {\na\n  }");
        assert!(d.fold(BufferRow(1), BufferRow(3)));
        let head = |d: &Document| d.selections().newest().head();
        // End on the collapsed HEADER goes past the visible tail — the last
        // row's end — not to the header text's own end mid-placeholder.
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(1, 2))));
        d.move_carets(Motion::LineEnd, false);
        assert_eq!(head(&d), d.buffer().point_to_offset(Point::new(3, 3)), "End lands after the tail brace");
        // Home from the TAIL goes to the display line's start: the header's
        // first non-whitespace…
        d.move_carets(Motion::LineStart, false);
        assert_eq!(head(&d), d.buffer().point_to_offset(Point::new(1, 2)), "Home jumps to the header's smart start");
        // …and a second Home (now on the header) toggles to column 0.
        d.move_carets(Motion::LineStart, false);
        assert_eq!(head(&d), d.buffer().point_to_offset(Point::new(1, 0)));
        // Unfolded, End is the plain line end again.
        assert!(d.unfold(BufferRow(1)));
        d.set_selections(SelectionSet::new(d.buffer().point_to_offset(Point::new(1, 2))));
        d.move_carets(Motion::LineEnd, false);
        assert_eq!(head(&d), d.buffer().point_to_offset(Point::new(1, 5)), "plain End at the header's own end");
    }

    #[test]
    fn foldable_ranges_from_multiline_brackets() {
        // Rows: 0 `mod m {`, 1 `fn f() {`, 2 `if x {`, 3 `a`, 4 `}`, 5 `}`,
        // 6 `g()` (single-line), 7 `}`.
        let d = doc("mod m {\n    fn f() {\n        if x {\n            a\n        }\n    }\n    g()\n}");
        let pairs: Vec<(u32, u32)> = d.foldable_ranges().iter().map(|(h, l)| (h.0, l.0)).collect();
        // Every multi-line pair; the single-line `q()` is not foldable.
        assert_eq!(pairs, vec![(0, 7), (1, 5), (2, 4)]);
    }

    #[test]
    fn fold_dropped_when_its_bracket_pair_is_broken() {
        use crate::{fold_map::FoldMap, BufferRow};
        // 0 "m {", 1 "  a", 2 "  b", 3 "}", 4 "after"
        let mut d = doc("m {\n  a\n  b\n}\nafter");
        assert_eq!(d.foldable_ranges().iter().map(|(h, l)| (h.0, l.0)).collect::<Vec<_>>(), vec![(0, 3)]);
        assert!(d.fold(BufferRow(0), BufferRow(3)));
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).display_row_count(), 2); // header + "after"

        // Delete only the `}` line (row 3). Its interior rows 1,2 are untouched, so
        // the fold rides the patch and would survive — orphaning rows 1,2 behind a
        // header that no longer draws a chevron. reconcile_folds must drop it.
        let s = d.buffer().point_to_offset(crate::Point::new(3, 0));
        let e = d.buffer().point_to_offset(crate::Point::new(4, 0));
        d.edit(vec![EditOp::delete(s..e)]).unwrap();
        assert!(d.foldable_ranges().is_empty(), "no closing brace ⇒ nothing foldable");
        assert!(d.folds().is_empty(), "invalidated fold dropped, not left orphaning hidden rows");
        assert_eq!(
            FoldMap::new(d.folds(), d.brackets(), d.buffer()).display_row_count(),
            d.buffer().line_count(),
            "every row is visible again"
        );
    }

    #[test]
    fn valid_fold_survives_edit_above() {
        use crate::{fold_map::FoldMap, BufferRow, SelectionSet};
        let mut d = doc("m {\n  a\n  b\n}\nafter");
        assert!(d.fold(BufferRow(0), BufferRow(3)));
        // Insert a line above the block: the fold stays valid (it still matches a
        // foldable range, now shifted down one row) — reconcile must NOT drop it.
        d.set_selections(SelectionSet::new(0));
        d.insert_text("top\n");
        assert_eq!(d.foldable_ranges().iter().map(|(h, l)| (h.0, l.0)).collect::<Vec<_>>(), vec![(1, 4)]);
        assert_eq!(d.folds().len(), 1, "the still-valid fold is kept");
        let m = FoldMap::new(d.folds(), d.brackets(), d.buffer());
        assert_eq!(m.fold_at_header(BufferRow(1)), Some(BufferRow(4)));
        assert!(m.is_folded(BufferRow(2)) && m.is_folded(BufferRow(3)) && m.is_folded(BufferRow(4)));
    }

    #[test]
    fn breaking_one_block_keeps_a_sibling_fold() {
        use crate::{fold_map::FoldMap, BufferRow};
        // Two sibling blocks: A rows 0-2, B rows 3-5, then "after".
        // 0 "a {", 1 "  x", 2 "}", 3 "b {", 4 "  y", 5 "}", 6 "after"
        let mut d = doc("a {\n  x\n}\nb {\n  y\n}\nafter");
        assert_eq!(d.foldable_ranges().iter().map(|(h, l)| (h.0, l.0)).collect::<Vec<_>>(), vec![(0, 2), (3, 5)]);
        assert!(d.fold(BufferRow(0), BufferRow(2)));
        assert!(d.fold(BufferRow(3), BufferRow(5)));
        assert_eq!(d.folds().len(), 2);
        // Delete block B's `}` (row 5). B is no longer foldable; A is untouched —
        // only B's fold should drop.
        let s = d.buffer().point_to_offset(crate::Point::new(5, 0));
        let e = d.buffer().point_to_offset(crate::Point::new(6, 0));
        d.edit(vec![EditOp::delete(s..e)]).unwrap();
        assert_eq!(d.folds().len(), 1, "only the broken block's fold drops");
        let m = FoldMap::new(d.folds(), d.brackets(), d.buffer());
        assert_eq!(m.fold_at_header(BufferRow(0)), Some(BufferRow(2)), "sibling A still collapsed");
        assert_eq!(m.fold_at_header(BufferRow(3)), None, "B is no longer folded");
    }

    #[test]
    fn fold_drops_when_its_interior_is_deleted() {
        use crate::{fold_map::FoldMap, BufferRow};
        let mut d = doc("L0\n{\nL2\nL3\n}\nL5");
        assert!(d.fold(BufferRow(1), BufferRow(4))); // hide rows 2,3,4 (incl. the `}`)
        // Delete rows 2..=4 (their whole byte span, incl. the closing `}`).
        let start = d.buffer().point_to_offset(crate::Point::new(2, 0));
        let end = d.buffer().point_to_offset(crate::Point::new(5, 0));
        d.edit(vec![EditOp::delete(start..end)]).unwrap();
        assert!(d.folds().is_empty(), "fold dropped when its interior text is deleted");
        assert_eq!(FoldMap::new(d.folds(), d.brackets(), d.buffer()).display_row_count(), d.buffer().line_count());
    }

    #[test]
    fn intel05_diagnostic_keeps_byte_offsets_on_a_multibyte_line() {
        // The core stores and returns diagnostics as BYTE spans, never
        // pre-converted to cells — so the render does the byte→cell mapping
        // exactly once (display_map::expand). "aé=x": é is 2 bytes / 1 cell, so
        // "x" is byte 4 but cell 3.
        use crate::{Diagnostic, Severity};
        let mut d = doc("aé=x");
        d.set_diagnostics(d.revision(), vec![Diagnostic::new(4..5, Severity::Error, "e")]);
        let span = d.diagnostics_in(0..100).next().unwrap().0;
        assert_eq!(span, 4..5, "diagnostics_in returns raw byte offsets, not cells");
        // The single byte→cell mapping places "x" at cell 3: the multibyte é
        // shifted the column by one, not two.
        assert_eq!(crate::display_map::expand(&d.buffer().line(0), span.start, 4), 3);
    }

    #[test]
    fn undo_resyncs_brackets_and_highlight_to_the_reverted_text() {
        const G: &str = "%YAML 1.2\n---\nname: T\nscope: source.t\ncontexts:\n  main:\n    - match: '\\w+'\n      scope: keyword.t\n";
        const TH: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict><key>settings</key><array><dict><key>settings</key><dict><key>foreground</key><string>#FFFFFF</string></dict></dict></array></dict></plist>"#;
        let mut d = doc("()");
        d.set_syntax(
            crate::SyntaxDef::from_sublime_syntax(G).unwrap(),
            crate::TokenTheme::from_tm_theme(TH).unwrap(),
        );
        d.tokenize_highlight(d.buffer().line_count());
        d.set_selections(SelectionSet::new(1)); // between ( and )
        d.edit(vec![EditOp::insert(1, "[]\n")]).unwrap(); // "([]\n)" — 2 lines, 4 brackets
        d.tokenize_highlight(d.buffer().line_count());
        assert_eq!((d.buffer().line_count(), d.brackets().all().len()), (2, 4));
        assert!(d.undo(), "there was something to undo");
        assert_eq!(d.buffer().text(), "()");
        // The resync ran: brackets re-matched, highlight cache back to one line.
        assert_eq!((d.buffer().line_count(), d.brackets().all().len()), (1, 2));
        d.tokenize_highlight(d.buffer().line_count());
        assert!(d.highlight_line_spans(0).is_some());
        assert!(d.highlight_line_spans(1).is_none(), "no phantom line after undo");
    }

    #[test]
    fn select_all_spans_the_whole_buffer_with_head_at_end() {
        let mut d = doc("abc\ndef"); // 7 bytes
        d.select_all();
        assert_eq!(d.selections().len(), 1);
        let s = d.selections().newest();
        assert_eq!((s.start(), s.end(), s.head()), (0, 7, 7));
    }

    fn query(text: &str) -> crate::FindQuery {
        crate::FindQuery { text: text.into(), case_sensitive: true, ..Default::default() }
    }

    #[test]
    fn find_matches_ride_an_edit_above_them() {
        // The handle-tracked match set remaps across an edit above it.
        let mut d = doc("xx target"); // "target" at 3..9
        d.set_find_query(Some(query("target")), 0);
        assert_eq!(d.find_matches_in(0..100).collect::<Vec<_>>(), vec![(3..9, false)]);
        d.set_selections(SelectionSet::new(0));
        d.edit(vec![EditOp::insert(0, "ab")]).unwrap(); // 2 chars above the match
        // Rode via apply_patch to 5..11 (the repair window near the edit then
        // re-verifies it in place).
        assert_eq!(d.find_matches_in(0..100).collect::<Vec<_>>(), vec![(5..11, false)]);
    }

    #[test]
    fn find_next_prev_start_from_the_caret_and_wrap() {
        // Nearest-from-caret ordering and wrap-around.
        let mut d = doc("foo bar foo baz foo"); // "foo" at 0..3, 8..11, 16..19
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!((d.find_match_count(), d.active_find_match()), (3, None));
        d.set_selections(SelectionSet::new(5)); // between match 0 and 1
        assert_eq!(d.find_next(0), Some(8..11)); // first start ≥ 5
        assert_eq!(d.active_find_match(), Some(1));
        assert_eq!(d.find_next(0), Some(16..19)); // repeated press cycles
        assert_eq!(d.find_next(0), Some(0..3), "wraps to first");
        assert_eq!(d.find_prev(0), Some(16..19), "prev from first wraps to last");
        d.set_selections(SelectionSet::new(0));
        assert_eq!(d.find_prev(0), Some(16..19), "prev before all wraps to last");
    }

    #[test]
    fn active_match_survives_a_rescan_at_the_same_start() {
        // The active match survives an edit that adds matches elsewhere: the
        // commit-path repair makes the new match visible IMMEDIATELY and the
        // untouched active survives by decoration-id stability —
        // `maybe_rescan_find` is a no-op on an uncapped set.
        let mut d = doc("foo foo foo");
        d.set_find_query(Some(query("foo")), 0);
        d.set_selections(SelectionSet::new(4)); // the middle match's start
        d.find_next(0);
        assert_eq!(d.active_find_match(), Some(1));
        d.set_selections(SelectionSet::new(11));
        d.edit(vec![EditOp::insert(11, " foo")]).unwrap(); // a 4th match, below
        assert_eq!(
            (d.find_match_count(), d.active_find_match()),
            (4, Some(1)),
            "current at the commit, active untouched"
        );
        assert!(!d.maybe_rescan_find(1000), "nothing stale to rescan (uncapped)");
        assert_eq!((d.find_match_count(), d.active_find_match()), (4, Some(1)));
    }

    #[test]
    fn find_matches_ride_undo_and_redo_through_the_shared_mover() {
        // Find highlights ride the reverse patch on undo/redo exactly like any
        // forward edit — through the shared decoration mover, with no
        // find-specific undo code — so they stay at their correct positions.
        let mut d = doc("foo bar");
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(d.find_matches_in(0..100).next().unwrap().0, 0..3);
        d.set_selections(SelectionSet::new(0));
        d.edit(vec![EditOp::insert(0, "xy")]).unwrap(); // "xyfoo bar" → match rides to 2..5
        assert_eq!(d.find_matches_in(0..100).next().unwrap().0, 2..5);
        d.undo(); // back to "foo bar" → match must ride back to 0..3
        assert_eq!(d.buffer().text(), "foo bar");
        assert_eq!(d.find_matches_in(0..100).next().unwrap().0, 0..3, "rode the undo");
        d.redo(); // → 2..5 again
        assert_eq!(d.find_matches_in(0..100).next().unwrap().0, 2..5, "rode the redo");
    }

    #[test]
    fn find_count_reflects_live_matches_after_a_collapse_delete() {
        // A collapse-delete drops a FindMatch decoration in `apply_patch`; the
        // count must follow the live store set. There is no shadow handle list to
        // over-report — the store IS the set.
        let mut d = doc("foo foo"); // matches 0..3 and 4..7
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(d.find_match_count(), 2);
        d.edit(vec![EditOp::delete(0..4)]).unwrap(); // "foo": collapses match 0
        assert_eq!(d.buffer().text(), "foo");
        assert_eq!(d.find_match_count(), 1, "collapsed match not counted");
        assert_eq!(d.find_matches_in(0..100).count(), 1, "count agrees with render");
    }

    #[test]
    fn find_navigation_requests_a_reveal() {
        let mut d = doc("foo bar foo");
        d.set_find_query(Some(query("foo")), 0);
        let before = d.reveal_seq();
        assert!(d.find_next(0).is_some());
        assert!(d.reveal_seq() > before, "navigation bumps the reveal request");
    }

    #[test]
    fn clearing_the_query_drops_every_match() {
        let mut d = doc("foo foo");
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(d.find_match_count(), 2);
        d.set_find_query(None, 0);
        assert_eq!((d.find_match_count(), d.find_query().is_none()), (0, true));
        assert_eq!(d.find_matches_in(0..100).count(), 0);
    }

    #[test]
    fn edits_repair_the_match_set_immediately() {
        // The eager per-commit repair makes the match set current AT the commit,
        // so `maybe_rescan_find` narrows to the capped-refill path — a no-op on
        // an uncapped set, before or after any debounce window.
        let mut d = doc("foo");
        d.set_find_query(Some(query("foo")), 0);
        d.edit(vec![EditOp::insert(0, "foo ")]).unwrap(); // "foo foo"
        assert_eq!(d.find_match_count(), 2, "current immediately — no debounce");
        assert!(!d.maybe_rescan_find(50), "nothing to rescan");
        assert!(!d.maybe_rescan_find(1_000), "uncapped: never rescans");
        assert_eq!(d.find_match_count(), 2);
    }

    // --- Find-match repair battery. ---

    /// The live `FindMatch` decoration ids, in document order.
    fn match_ids(d: &Document) -> Vec<crate::decorations::DecorationId> {
        d.decorations
            .decorations_in(0..u32::MAX)
            .filter(|r| matches!(r.kind, DecorationKind::FindMatch))
            .map(|r| r.id)
            .collect()
    }

    /// The live match ranges, in document order.
    fn match_ranges(d: &Document) -> Vec<Range<u32>> {
        d.find_matches_in(0..u32::MAX).map(|(r, _)| r).collect()
    }

    /// A seeded xorshift for the randomized repair oracles.
    fn xorshift(mut state: u64) -> impl FnMut() -> u64 {
        move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        }
    }

    #[test]
    fn repair_preserves_far_match_ids() {
        // An edit FAR from every match must not re-mint the match decorations —
        // the repair touches only matches near the edit, so untouched ones keep
        // their ids rather than being wholesale-replaced.
        let mut d = doc("foo bar foo bar baz padding padding foo");
        d.set_find_query(Some(query("foo")), 0);
        let before = match_ids(&d);
        assert_eq!(before.len(), 3);
        // Inside the first "padding" — farther than a needle length from all.
        d.edit(vec![EditOp::insert(22, "zz")]).unwrap();
        assert!(!d.maybe_rescan_find(1_000), "always current: nothing to rescan");
        assert_eq!(match_ids(&d), before, "untouched matches keep their ids");
        assert_eq!(match_ranges(&d), vec![0..3, 8..11, 38..41]);
    }

    #[test]
    fn repair_insert_creates_match_in_window() {
        let mut d = doc("fo bar");
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(d.find_match_count(), 0);
        d.edit(vec![EditOp::insert(2, "o")]).unwrap(); // "foo bar"
        assert_eq!(match_ranges(&d), vec![0..3], "current at the commit");
    }

    #[test]
    fn repair_delete_joins_halves_creates_match() {
        // Deleting the interloper joins "fo" + "o" into a fresh match.
        let mut d = doc("foXo bar");
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(d.find_match_count(), 0);
        d.edit(vec![EditOp::delete(2..3)]).unwrap(); // "foo bar"
        assert_eq!(match_ranges(&d), vec![0..3]);
    }

    #[test]
    fn repair_destroys_straddling_match() {
        // A deletion straddling the match's tail leaves "fo" — no match.
        let mut d = doc("afoob");
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(match_ranges(&d), vec![1..4]);
        d.edit(vec![EditOp::delete(3..5)]).unwrap(); // "afo"
        assert_eq!(d.find_match_count(), 0);
    }

    #[test]
    fn repair_interior_insert_invalidates_match() {
        // NeverGrows grows on a strictly-interior insert: the grown range no
        // longer equals the needle, so the repair must remove it.
        let mut d = doc("foo");
        d.set_find_query(Some(query("foo")), 0);
        assert_eq!(d.find_match_count(), 1);
        d.edit(vec![EditOp::insert(1, "x")]).unwrap(); // "fxoo"
        assert_eq!(d.find_match_count(), 0);
    }

    #[test]
    fn repair_multi_edit_transaction() {
        // ONE apply() with two scattered edits — one patch, two repair windows.
        let mut d = doc("fo bar fo baz");
        d.set_find_query(Some(query("foo")), 0);
        d.edit(vec![EditOp::insert(2, "o"), EditOp::insert(9, "o")]).unwrap();
        assert_eq!(d.buffer().text(), "foo bar foo baz");
        assert_eq!(match_ranges(&d), vec![0..3, 8..11]);
    }

    #[test]
    fn repair_equals_full_scan_property() {
        // The repair oracle: for a NON-self-overlapping needle, after EVERY
        // commit the live match set is byte-identical to a fresh `scan` of the
        // live text. Driven through Document::edit so the rebase_views wiring is
        // what's under test.
        let fq = query("ab");
        let mut next = xorshift(0x243F_6A88_85A3_08D3);
        let alphabet = ['a', 'b', ' '];
        let mut d = doc(&"ab ba a b ab ".repeat(8));
        d.set_find_query(Some(fq.clone()), 0);
        for step in 0..300 {
            let len = d.buffer().len();
            match next() % 3 {
                0 => {
                    let at = (next() % u64::from(len + 1)) as u32;
                    let n = 1 + next() % 3;
                    let s: String = (0..n).map(|_| alphabet[(next() % 3) as usize]).collect();
                    d.edit(vec![EditOp::insert(at, s)]).unwrap();
                }
                1 if len > 0 => {
                    let a = (next() % u64::from(len)) as u32;
                    let b = (a + 1 + (next() % 4) as u32).min(len);
                    d.edit(vec![EditOp::delete(a..b)]).unwrap();
                }
                _ => {
                    // Two scattered inserts as ONE transaction.
                    let a = (next() % u64::from(len + 1)) as u32;
                    let b = (next() % u64::from(len + 1)) as u32;
                    let (a, b) = (a.min(b), a.max(b));
                    if a == b {
                        d.edit(vec![EditOp::insert(a, "ab")]).unwrap();
                    } else {
                        d.edit(vec![EditOp::insert(a, "ba"), EditOp::insert(b, "ab")]).unwrap();
                    }
                }
            }
            let (oracle, _) = crate::find::scan(&d.buffer().text(), &fq);
            assert_eq!(match_ranges(&d), oracle, "step {step} diverged from the oracle");
        }
    }

    #[test]
    fn repair_self_overlapping_needle_is_maximal_and_valid() {
        // Documented relaxation: a self-overlapping needle ("aa") repairs to a
        // MAXIMAL valid non-overlapping set — every match equals the needle, no
        // two overlap, and no further non-overlapping placement exists — though
        // the phase near an edit may differ from a from-scratch greedy scan until
        // the next full scan.
        let mut next = xorshift(0x9E37_79B9_7F4A_7C15);
        let alphabet = ['a', ' '];
        let mut d = doc(&"aaa a aaaa aa ".repeat(6));
        d.set_find_query(Some(query("aa")), 0);
        for step in 0..300 {
            let len = d.buffer().len();
            if next().is_multiple_of(2) || len == 0 {
                let at = (next() % u64::from(len + 1)) as u32;
                let n = 1 + next() % 3;
                let s: String = (0..n).map(|_| alphabet[(next() % 2) as usize]).collect();
                d.edit(vec![EditOp::insert(at, s)]).unwrap();
            } else {
                let a = (next() % u64::from(len)) as u32;
                let b = (a + 1 + (next() % 3) as u32).min(len);
                d.edit(vec![EditOp::delete(a..b)]).unwrap();
            }
            let live = match_ranges(&d);
            let text = d.buffer().text();
            let bytes = text.as_bytes();
            for (i, r) in live.iter().enumerate() {
                assert_eq!(
                    &bytes[r.start as usize..r.end as usize],
                    b"aa",
                    "step {step}: stale match {r:?}"
                );
                if i > 0 {
                    assert!(live[i - 1].end <= r.start, "step {step}: overlapping matches");
                }
            }
            for p in 0..bytes.len().saturating_sub(1) {
                if &bytes[p..p + 2] == b"aa" {
                    let p = p as u32;
                    assert!(
                        live.iter().any(|r| r.start < p + 2 && p < r.end),
                        "step {step}: placement at {p} misses every match (not maximal)"
                    );
                }
            }
        }
    }

    #[test]
    fn active_survives_untouched() {
        // Active-match stability by id: an edit near OTHER matches leaves the
        // active one's decoration (and so its N-of-M slot) alone.
        let mut d = doc("foo bar foo");
        d.set_find_query(Some(query("foo")), 0);
        d.find_next(0); // activates 0..3
        let id = d.find.active_id().expect("a match is active");
        d.edit(vec![EditOp::insert(7, "x")]).unwrap(); // within k−1 of the 2nd match
        assert_eq!(d.find.active_id(), Some(id), "untouched active keeps its id");
        assert_eq!((d.find_match_count(), d.active_find_match()), (2, Some(0)));
    }

    #[test]
    fn active_transfers_same_start() {
        // A repair window removes and re-creates the active match at the same
        // start — the active transfers to the new id.
        let mut d = doc("foo bar");
        d.set_find_query(Some(query("foo")), 0);
        d.find_next(0); // activates 0..3
        let old = d.find.active_id().expect("a match is active");
        d.edit(vec![EditOp::insert(4, "x")]).unwrap(); // "foo xbar": in-window, text intact
        let new = d.find.active_id().expect("active transferred");
        assert_ne!(new, old, "the decoration was re-minted by the repair");
        assert_eq!(d.active_find_match(), Some(0), "still the first match");
        assert_eq!(match_ranges(&d), vec![0..3]);
    }

    #[test]
    fn active_cleared_when_destroyed() {
        let mut d = doc("foo bar foo");
        d.set_find_query(Some(query("foo")), 0);
        d.find_next(0); // activates 0..3
        d.edit(vec![EditOp::delete(0..2)]).unwrap(); // "o bar foo": destroys it
        assert_eq!(d.find.active_id(), None, "destroyed active clears");
        assert_eq!((d.find_match_count(), d.active_find_match()), (1, None));
    }

    #[test]
    fn count_and_order_consistent_after_repair() {
        let mut d = doc("foo foo foo");
        d.set_find_query(Some(query("foo")), 0);
        d.set_selections(SelectionSet::new(4));
        d.find_next(0); // activates the match at 4..7
        d.edit(vec![EditOp::insert(0, "foo ")]).unwrap(); // a NEW first match
        let all: Vec<(Range<u32>, bool)> = d.find_matches_in(0..u32::MAX).collect();
        let ranges: Vec<Range<u32>> = all.iter().map(|(r, _)| r.clone()).collect();
        assert_eq!(ranges, vec![0..3, 4..7, 8..11, 12..15]);
        assert_eq!(d.find_match_count(), all.len());
        // The active rode from 4..7 to 8..11; N-of-M agrees with the iteration.
        assert_eq!(d.active_find_match(), Some(2));
        assert_eq!(all.iter().position(|(_, active)| *active), Some(2));
    }

    #[test]
    fn active_start_tracks_the_decoration_under_random_edits() {
        // Anti-drift pin: with a match active, after EVERY commit the tracked
        // `active_start` must equal the active decoration's live start (or both
        // be None). It is the single rebased position kept in lockstep with
        // `active` across the ~5 mutation sites; if the per-commit rebase is
        // dropped (or the transfer forgets to re-set it), an edit *before* the
        // active match slides its decoration but leaves `active_start` stale, and
        // this trips. (An edit exactly *at* the match start always routes through
        // the window remove→transfer, which resets `active_start` — so this pins
        // the delta-rebase, and `count_and_order_consistent_after_repair` pins the
        // ordinal that rebase feeds.)
        let base = "foo foo bar foo baz foo qux foo end";
        let mut rng = 0xDEAD_BEEF_0000_1111u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for trial in 0..400u32 {
            let mut d = doc(base);
            d.set_find_query(Some(query("foo")), 0);
            // Activate a MID-document match (offset > 0), so "edit before it"
            // trials exist to exercise the delta-rebase.
            d.set_selections(SelectionSet::new(10));
            d.find_next(0);
            for _step in 0..14 {
                if d.find.active_id().is_none() {
                    // Re-activate so the invariant keeps getting exercised.
                    d.set_selections(SelectionSet::new(1));
                    d.find_next(0);
                }
                let len = d.buffer().len();
                // Sometimes aim the edit at the active match's start (the
                // Bias::Right boundary), else a random position.
                let at = match d.find.active_start() {
                    Some(s) if next() % 3 == 0 => s,
                    _ => (next() % u64::from(len + 1)) as u32,
                };
                match next() % 3 {
                    0 => {
                        let _ = d.edit(vec![EditOp::insert(at, "z")]);
                    }
                    1 => {
                        let end = (at + 1 + (next() % 3) as u32).min(len);
                        if end > at {
                            let _ = d.edit(vec![EditOp::delete(at..end)]);
                        }
                    }
                    _ => {
                        // Multi-caret insert at two disjoint carets.
                        let a = (next() % u64::from(len + 1)) as u32;
                        let b = (next() % u64::from(len + 1)) as u32;
                        let (lo, hi) = (a.min(b), a.max(b));
                        if lo != hi {
                            d.set_selections(SelectionSet::from_ranges(&[(lo, lo), (hi, hi)], 0));
                            d.insert_text("q");
                        }
                    }
                }
                let want = d
                    .find
                    .active_id()
                    .and_then(|id| d.decorations.decoration_range(id))
                    .map(|r| r.start);
                assert_eq!(
                    d.find.active_start(),
                    want,
                    "trial {trial}: active_start drifted from its decoration's live start",
                );
            }
        }
    }

    #[test]
    fn capped_scan_keeps_a_document_prefix() {
        // > FIND_MATCH_CAP occurrences: the fresh scan keeps the first 10k
        // (a document prefix) and reports capped.
        let text = "x ".repeat(crate::FIND_MATCH_CAP + 50);
        let mut d = doc(&text);
        d.set_find_query(Some(query("x")), 0);
        assert!(d.find.capped());
        let live = match_ranges(&d);
        assert_eq!(live.len(), crate::FIND_MATCH_CAP);
        assert_eq!(live.last().cloned(), Some(19_998..19_999), "prefix, not a sample");
    }

    #[test]
    fn typing_while_capped_does_not_rescan_the_match_set() {
        // While a >cap find is active, the per-keystroke cap bookkeeping must be
        // O(1): a `decorations_in(0..u32::MAX).filter(is_find).collect()` would
        // charge ~FIND_MATCH_CAP work units on EVERY keystroke while capped.
        // Reading the count from the root summary (`find_count`, O(1)) and
        // trimming via ordinal `nth_find` (O(log)) means a steady capped
        // keystroke touches no whole-store scan.
        let text = "x ".repeat(crate::FIND_MATCH_CAP + 50);
        let mut d = doc(&text);
        d.set_find_query(Some(query("x")), 0);
        assert!(d.find.capped());
        assert_eq!(d.find_match_count(), crate::FIND_MATCH_CAP);
        d.set_selections(crate::SelectionSet::new(0));
        crate::perf::reset();
        d.type_char('z'); // adds no `x` ⇒ the set stays exactly at the cap (no trim)
        let work = crate::perf::meter();
        assert_eq!(d.find_match_count(), crate::FIND_MATCH_CAP, "still capped");
        assert!(
            work < 2_000,
            "capped keystroke charged {work} work units (≈FIND_MATCH_CAP ⇒ the O(M) cap scan survived)"
        );
    }

    #[test]
    fn capped_trims_tail_and_coverage() {
        // Exactly at the cap (uncapped) — one edit adds matches mid-document,
        // the repair pushes past the cap, and the TAIL is trimmed so the set
        // stays a prefix of the document.
        let text = "x ".repeat(crate::FIND_MATCH_CAP);
        let mut d = doc(&text);
        d.set_find_query(Some(query("x")), 0);
        assert!(!d.find.capped());
        d.edit(vec![EditOp::insert(1_000, "xxx")]).unwrap();
        assert!(d.find.capped(), "over the cap after the repair");
        let live = match_ranges(&d);
        assert_eq!(live.len(), crate::FIND_MATCH_CAP, "trimmed back to the cap");
        assert!(live.windows(2).all(|w| w[0].end <= w[1].start), "still non-overlapping");
        // The new matches near the edit are IN; the trim came off the tail.
        assert_eq!(live.iter().filter(|r| (1_000..1_004).contains(&r.start)).count(), 4);
        assert_eq!(live.last().map(|r| r.start), Some(19_995), "tail trimmed");
    }

    #[test]
    fn capped_refill_is_debounced() {
        let text = "x ".repeat(crate::FIND_MATCH_CAP + 50);
        let mut d = doc(&text);
        d.set_find_query(Some(query("x")), 0); // capped; scan stamp at 0 ms
        // Kill 5 matches inside the covered prefix → room below the cap.
        d.edit(vec![EditOp::delete(0..10)]).unwrap();
        assert_eq!(d.find_match_count(), crate::FIND_MATCH_CAP - 5);
        assert!(!d.maybe_rescan_find(50), "inside the debounce window");
        assert_eq!(d.find_match_count(), crate::FIND_MATCH_CAP - 5);
        assert!(d.maybe_rescan_find(100), "past the window → refill");
        assert_eq!(d.find_match_count(), crate::FIND_MATCH_CAP, "coverage re-grew");
        assert!(!d.maybe_rescan_find(300), "idle after the refill: no re-scan");
    }

    #[test]
    fn undo_redo_match_set_equals_fresh_scan() {
        // Undo/redo flow through the same rebase_views mover, so the repaired
        // set must equal the oracle after every reverted/replayed step too.
        let fq = query("ab");
        let mut d = doc("ab ab ba ab");
        d.set_find_query(Some(fq.clone()), 0);
        let check = |d: &Document, at: &str| {
            let (oracle, _) = crate::find::scan(&d.buffer().text(), &fq);
            assert_eq!(match_ranges(d), oracle, "diverged after {at}");
        };
        d.edit(vec![EditOp::insert(2, "ab")]).unwrap();
        check(&d, "edit 1");
        d.edit(vec![EditOp::delete(0..3)]).unwrap();
        check(&d, "edit 2");
        d.edit(vec![EditOp::insert(5, "b a")]).unwrap();
        check(&d, "edit 3");
        assert!(d.undo());
        check(&d, "undo 1");
        assert!(d.undo());
        check(&d, "undo 2");
        assert!(d.undo());
        check(&d, "undo 3");
        assert!(d.redo());
        check(&d, "redo 1");
        assert!(d.redo());
        check(&d, "redo 2");
    }

    #[test]
    fn case_fold_repair() {
        // The windowed repair honors the query's fold toggle.
        let fq = crate::FindQuery { text: "foo".into(), case_sensitive: false, ..Default::default() };
        let mut d = doc("FO bar");
        d.set_find_query(Some(fq), 0);
        assert_eq!(d.find_match_count(), 0);
        d.edit(vec![EditOp::insert(2, "O")]).unwrap(); // "FOO bar"
        assert_eq!(match_ranges(&d), vec![0..3]);
        d.edit(vec![EditOp::insert(0, "f")]).unwrap(); // "fFOO bar"
        assert_eq!(match_ranges(&d), vec![1..4], "re-anchored, still folded");
    }
}
