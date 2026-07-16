//! The display map: buffer coordinates ↔ *display* coordinates, tab expansion
//! only.
//!
//! This module owns both coordinate spaces and the sole converter between them.
//! Buffer space is [`Point`] (row + **byte** column); display space is
//! [`DisplayPoint`] (row + **cell** column, one cell = one Unicode scalar's
//! monospace advance, a tab = `1..=tab_size` cells). The seam is a concrete
//! type — [`TabMap`] — not a trait: the one other implementor,
//! [`crate::fold_map::FoldMap`], *composes over* this surface rather than
//! sharing a `dyn` bound, and word wrap is rejected (it belongs in a rich-text
//! editor, not a code editor), so no further implementor is expected.
//!
//! What freezes the callable set against the widget crate is not a `dyn` bound
//! but the `pub(crate)` constructor on [`DisplayPoint`]/[`DisplayRow`]: every
//! display coordinate in existence came out of a [`TabMap`] method, so no code
//! in `scrive-iced` can fabricate one — it must go through
//! [`TabMap::to_display`]/[`TabMap::clip`]. Cross-space arithmetic is a compile
//! error, which *is* the enforcement of "no widget code does row/column math
//! across spaces."
//!
//! The `TabMap` itself is **row-preserving**: `DisplayRow(r)` ⇔ buffer row
//! `r`, rows never move, only columns stretch. A caret can never rest inside a
//! tab expansion — the bias snap in [`collapse`] forbids it. Folding sits as a
//! layer above ([`crate::fold_map::FoldMap`] hides rows / collapses inline
//! spans over this base) without changing any signature here; word wrap is
//! rejected, not deferred.

use std::ops::Range;

use crate::buffer::Buffer;
use crate::coords::{Bias, Point};
use crate::patch::Patch;

/// A display row, 0-based. In the tab-only map: `DisplayRow(r)` ⇔ buffer row
/// `r`.
///
/// A distinct newtype because the `FoldMap` layer breaks that 1:1 — a folded
/// row hides buffer rows beneath it — so every site that would otherwise assume
/// display row equals buffer row is a compile error to revisit. The field is
/// `pub(crate)`: consumer crates can *read* a display row
/// ([`DisplayRow::index`]) but can never **mint** one from arithmetic — every
/// display row in existence came out of a `FoldMap`/`TabMap` method, so "anchor
/// something at a buffer row × line height" is a compile error rather than
/// something to catch by eye. This keeps popups and other row-anchored chrome
/// correctly offset when a fold sits above them.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct DisplayRow(pub(crate) u32);

impl DisplayRow {
    /// The 0-based row index — read-only; pair with the pixel projection's
    /// one `row → y` map, never with hand arithmetic.
    #[must_use]
    pub const fn index(self) -> u32 {
        self.0
    }
}

/// A buffer row, 0-based — the gutter's *printed* line number.
///
/// A **different** newtype from [`DisplayRow`] so "which number do I print for
/// this display row" can never be fed a display row by accident.
/// [`TabMap::row_info`] returns `Some` for every row here; the `Option` in its
/// signature lets a caller cope with a display row that has no printed line
/// number.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct BufferRow(pub u32);

/// A visual position: display row + **cell** column (one cell = one Unicode
/// scalar's monospace advance; a tab occupies `1..=tab_size` cells).
///
/// Fields are private and the constructor is `pub(crate)`: no code in the
/// *widget* crate can fabricate a display point — it must route through
/// [`TabMap::to_display`]/[`TabMap::clip`] (the crate boundary is the
/// enforcement line). Ordered row-major, so display points compare in
/// visual document order for range and decoration math.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Debug)]
pub struct DisplayPoint {
    row: DisplayRow,
    col: u32,
}

impl DisplayPoint {
    /// A display point at `(row, col)`. `pub(crate)`: the widget crate can never
    /// mint one directly.
    pub(crate) fn new(row: DisplayRow, col: u32) -> Self {
        Self { row, col }
    }

    /// The visual origin, `(row 0, col 0)`.
    #[must_use]
    pub fn zero() -> Self {
        Self { row: DisplayRow(0), col: 0 }
    }

    /// The display row.
    #[must_use]
    pub fn row(self) -> DisplayRow {
        self.row
    }

    /// The cell column.
    #[must_use]
    pub fn col(self) -> u32 {
        self.col
    }
}

/// One render run of a display row, split at every tab boundary so the renderer
/// can draw tab invisibles and overlay highlight spans onto each run.
#[derive(Clone, Debug)]
pub struct DisplayChunk<'a> {
    /// The run's source text. For `is_tab` this is the literal `"\t"`; the
    /// caller draws `cells` blanks (plus an optional invisible glyph) and never
    /// paints these bytes.
    pub text: &'a str,
    /// Byte range **within the buffer line** — so a per-line `HighlightSpan`
    /// (also byte-within-line) overlays this run with no coordinate hop.
    pub buffer_bytes: Range<u32>,
    /// Width in cells (a tab: [`tab_width`] at the run's start cell).
    pub cells: u32,
    /// Whether this run is a single tab (the renderer draws it as blanks).
    pub is_tab: bool,
}

/// Lazy per-row splitter yielding one [`DisplayChunk`] per run, tabs isolated.
/// Constructed by [`chunks`] over a line the CALLER holds — the caller
/// materializes the row (one [`Buffer::line`] `Cow`) and the runs borrow from
/// it, so no iterator owns text it also lends out.
#[derive(Clone, Debug)]
pub struct DisplayChunks<'a> {
    line: &'a str,
    tab_size: u32,
    byte: u32,
    cell: u32,
}

/// Split one row's text into [`DisplayChunk`] runs at every tab boundary.
#[must_use]
pub fn chunks(line: &str, tab_size: u32) -> DisplayChunks<'_> {
    DisplayChunks { line, tab_size, byte: 0, cell: 0 }
}

impl<'a> Iterator for DisplayChunks<'a> {
    type Item = DisplayChunk<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let bytes = self.line.as_bytes();
        let len = bytes.len() as u32;
        if self.byte >= len {
            return None;
        }
        let start = self.byte;
        if bytes[start as usize] == b'\t' {
            // A tab is its own run, so the renderer can size the gap exactly.
            let cells = tab_width(self.cell, self.tab_size);
            self.byte = start + 1;
            self.cell += cells;
            return Some(DisplayChunk {
                text: &self.line[start as usize..start as usize + 1],
                buffer_bytes: start..start + 1,
                cells,
                is_tab: true,
            });
        }
        // A maximal run of non-tab scalars; each advances exactly one cell.
        let mut cells = 0u32;
        for ch in self.line[start as usize..].chars() {
            if ch == '\t' {
                break;
            }
            self.byte += ch.len_utf8() as u32;
            cells += 1;
        }
        self.cell += cells;
        Some(DisplayChunk {
            text: &self.line[start as usize..self.byte as usize],
            buffer_bytes: start..self.byte,
            cells,
            is_tab: false,
        })
    }
}

/// A display-space edit region: what [`TabMap::sync`] would return if a
/// display-space invalidation reader ever appeared. There is none today — the
/// `FoldMap` re-derives per frame and word wrap is rejected — so `sync` returns
/// `()` and downstream caches (line cache, paragraph cache) invalidate off the
/// buffer-space edit they already consume.
///
/// If it is ever produced, `old` ranges the pre-sync snapshot and `new` the
/// post-sync one, read downstream as a *region to invalidate* rather than a
/// description of the change. It stays row-preserving: within one line `old`
/// and `new` share their row and differ only in cell columns; a line
/// insert/delete shifts whole rows.
#[derive(Clone, Debug)]
pub struct DisplayEdit {
    /// Pre-sync display-space range.
    pub old: Range<DisplayPoint>,
    /// Post-sync display-space range.
    pub new: Range<DisplayPoint>,
}

/// Expansion width of a tab beginning at 0-based **cell** column `c`: advances
/// to the next multiple of `tab_size`, always `1..=tab_size`, never zero.
#[must_use]
pub fn tab_width(c: u32, tab_size: u32) -> u32 {
    tab_size - (c % tab_size)
}

/// Byte column → cell column, a per-line scan from column 0.
///
/// Each `\t` advances [`tab_width`]; every other Unicode **scalar** advances one
/// cell (scalars, not bytes — a stray multibyte char shifts columns, never
/// corrupts them). `byte_col` past the line length simply expands the whole
/// line. `O(line)`; at typical short lines this is single-digit µs, so no
/// sum-tree or cursor structure is needed.
#[must_use]
pub fn expand(line: &str, byte_col: u32, tab_size: u32) -> u32 {
    let mut cell = 0u32;
    let mut byte = 0u32;
    for ch in line.chars() {
        if byte >= byte_col {
            break; // count only the cells strictly before byte_col
        }
        cell += if ch == '\t' { tab_width(cell, tab_size) } else { 1 };
        byte += ch.len_utf8() as u32;
    }
    cell
}

/// Indentation depth (in guide levels) for one row — the number of indent
/// guides drawn to its left.
///
/// A content row's level is `ceil(indent / indent_size)`, where `own` is its
/// leading-whitespace width in display cells. A blank row (`own = None`) has no
/// indent of its own, so its level is interpolated from the nearest non-blank
/// rows `above` / `below` (their widths in cells): the blank inherits the
/// *deeper* neighbour's ladder, biased so a dedent still shows the outgoing
/// level. This is the convention for brace-delimited languages, where
/// whitespace does not open or close a block. A row at the top/bottom edge of
/// the file (one neighbour missing) is level 0. The guides drawn are at levels
/// `1..=level`, i.e. cells `0, indent_size, …, (level-1)*indent_size`.
#[must_use]
pub fn indent_guide_level(
    own: Option<u32>,
    above: Option<u32>,
    below: Option<u32>,
    indent_size: u32,
) -> u32 {
    match own {
        Some(indent) => indent.div_ceil(indent_size),
        None => match (above, below) {
            (None, _) | (_, None) => 0,
            (Some(a), Some(b)) if a < b => 1 + a / indent_size,
            (Some(a), Some(b)) if a == b => b.div_ceil(indent_size),
            (Some(_), Some(b)) => 1 + b / indent_size, // a > b: dedenting, brace-delimited
        },
    }
}

/// Which indent guide is *active* (highlighted) for the caret's line.
///
/// Returns `(indent_level, start_row, end_row)` inclusive, or `None` when no
/// guide is active (the caret sits at level 0 with no deeper neighbour).
/// `level_at(row)` yields a row's indent level (content = `ceil`, blank =
/// interpolated, matching [`indent_guide_level`]); rows are 0-based,
/// `line_count` the total.
///
/// The scope special-cases are the whole point: on a line that *opens* a deeper
/// scope (the next line is more indented) the **first child level** is
/// highlighted — the opener's own level `+ 1`, i.e. one tab stop in from the
/// opener, regardless of how deep the body actually sits — and symmetrically on
/// a line that *closes* one (the line above is more indented). Off a brace line
/// it is just the caret line's level. The chosen level is then extended over
/// the contiguous run of rows at least that deep.
///
/// Pinning the highlight to `initial + 1` (not the body's real level) is
/// deliberate: an over-indented body (opener at col 4, body at col 16) still
/// lights the guide one stop inside the opener — cell 4 — rather than the
/// body's own deep guide. For uniformly-indented code the jump *is* one level,
/// so `initial + 1` already equals the body level.
///
/// `rows` bounds the outward walk: the guide only *paints* inside the caller's
/// viewport, so the extent search never needs to leave that clamp (the widget
/// passes its window ± slack; tests pass `0..line_count` for the unbounded
/// behaviour). A caret row outside `rows` yields `None` — an off-screen caret's
/// scope highlight is not painted anyway.
#[must_use]
pub fn active_indent_guide(
    caret_row: u32,
    line_count: u32,
    rows: Range<u32>,
    level_at: impl Fn(u32) -> u32,
) -> Option<(u32, u32, u32)> {
    let lo = rows.start;
    let hi = rows.end.min(line_count);
    if caret_row >= line_count || caret_row < lo || caret_row >= hi {
        return None;
    }
    let initial = level_at(caret_row);
    let down = (caret_row + 1 < line_count).then(|| level_at(caret_row + 1));
    let up = caret_row.checked_sub(1).map(&level_at);

    let (indent, mut start, mut end, go_up, go_down) = match (down, up) {
        // Opener of a deeper scope: activate the first child level (opener + 1).
        (Some(d), _) if d > initial => (initial + 1, caret_row + 1, caret_row + 1, false, true),
        // Closer of a deeper scope: same, anchored on the line above.
        (_, Some(u)) if u > initial => (initial + 1, caret_row - 1, caret_row - 1, true, false),
        _ if initial == 0 => return None,
        _ => (initial, caret_row, caret_row, true, true),
    };

    if go_up {
        while start > lo && level_at(start - 1) >= indent {
            start -= 1;
        }
    }
    if go_down {
        while end + 1 < hi && level_at(end + 1) >= indent {
            end += 1;
        }
    }
    Some((indent, start, end))
}

/// Cell column → byte column, with the mid-tab bias snap.
///
/// A cell landing strictly inside a tab is not a valid caret slot: `Bias::Left`
/// yields the byte of the `\t`, `Bias::Right` the byte just after it.
/// Only a tab has width > 1, so only a tab can trigger the snap. A `cell_col`
/// past end-of-line clamps to the line's byte length. The result is always a
/// UTF-8 char boundary.
#[must_use]
pub fn collapse(line: &str, cell_col: u32, tab_size: u32, bias: Bias) -> u32 {
    let mut cell = 0u32;
    let mut byte = 0u32;
    for ch in line.chars() {
        if cell >= cell_col {
            return byte; // exact hit at a boundary
        }
        let w = if ch == '\t' { tab_width(cell, tab_size) } else { 1 };
        if cell + w > cell_col {
            // Target falls inside this char's span — only a tab reaches here.
            return match bias {
                Bias::Left => byte,
                Bias::Right => byte + 1,
            };
        }
        cell += w;
        byte += ch.len_utf8() as u32;
    }
    byte // past EOL → line byte length
}

/// Default tab stop width — 4, the common indent width; a change is one
/// whole-document edit through [`TabMap::sync`].
#[must_use]
pub const fn default_tab_size() -> u32 {
    4
}

/// The coordinates seam — a **concrete** type, not a trait.
///
/// Its **entire** durable state is `tab_size`; it holds no persisted visual
/// indices, so there is nothing derived that could drift out of sync with the
/// text. Built per frame from the document's `tab_size` plus a borrow of the
/// buffer's line index, so every conversion reads live text and the map is
/// current by construction.
///
/// Every method that can land on a lossy position (mid-tab, or inside a fold at
/// the layer above) threads [`Bias`] from the start, so no call site needs
/// retrofitting when a new lossy case appears.
pub struct TabMap<'a> {
    tab_size: u32,
    lines: &'a Buffer,
}

impl<'a> TabMap<'a> {
    /// A map over `lines` with the given tab stop width. `tab_size` is expected
    /// to be ≥ 1 (see [`default_tab_size`]).
    #[must_use]
    pub fn new(lines: &'a Buffer, tab_size: u32) -> Self {
        Self { tab_size, lines }
    }

    /// Buffer → display. `point.col` is a char-boundary byte column (the buffer
    /// guarantees this); the result column is cells. Total: every valid buffer
    /// point maps to exactly one display point, so `bias` is irrelevant here.
    #[must_use]
    pub fn to_display(&self, point: Point, _bias: Bias) -> DisplayPoint {
        DisplayPoint::new(
            DisplayRow(point.row),
            expand(&self.line(point.row), point.col, self.tab_size),
        )
    }

    /// Display → buffer, with the bias snap: if `point.col` falls strictly
    /// inside a tab expansion it is not a caret slot — `Left` yields the byte of
    /// the `\t`, `Right` the byte after it.
    #[must_use]
    pub fn to_buffer(&self, point: DisplayPoint, bias: Bias) -> Point {
        Point {
            row: point.row().0,
            col: collapse(&self.line(point.row().0), point.col(), self.tab_size, bias),
        }
    }

    /// Snap any externally-sourced display position (mouse hit, diagnostic span,
    /// find match) to the nearest valid caret slot — the guard every such
    /// position passes before use. Steps: [`collapse`] by `bias` → clamp + char
    /// snap in buffer space → re-[`expand`]. Idempotent.
    #[must_use]
    pub fn clip(&self, point: DisplayPoint, bias: Bias) -> DisplayPoint {
        let buffer_point = self.lines.clip_point(self.to_buffer(point, bias), bias);
        self.to_display(buffer_point, bias)
    }

    /// The bottom-right valid position: the last row and its cell length.
    #[must_use]
    pub fn max_point(&self) -> DisplayPoint {
        let row = self.max_row();
        DisplayPoint::new(row, self.line_len(row))
    }

    /// Number of display rows in this map — the buffer line count. (Hiding
    /// rows is the `FoldMap` layer's job, above this one.)
    #[must_use]
    pub fn row_count(&self) -> u32 {
        self.lines.line_count()
    }

    /// Cell width of a display row (the whole line expanded).
    #[must_use]
    pub fn line_len(&self, row: DisplayRow) -> u32 {
        expand(&self.line(row.0), self.lines.line_len(row.0), self.tab_size)
    }

    /// Gutter protocol: the buffer line number to print. Always `Some` in this
    /// map; the `Option` lets a caller handle a display row that has no printed
    /// line number.
    #[must_use]
    pub fn row_info(&self, row: DisplayRow) -> Option<BufferRow> {
        Some(BufferRow(row.0))
    }

    /// Consume the transaction's buffer-space edit patch and splice any internal
    /// derived state. Returns `()`: downstream caches (line cache, paragraph
    /// cache) invalidate off the buffer-space `LineSplice` they already consume,
    /// so no display-space edit vector is emitted — widen to `Vec<DisplayEdit>`
    /// only if a display-space reader ever appears. The map holds no derived
    /// state, so this is a no-op; the borrow of live buffer text keeps every
    /// conversion current by construction.
    pub fn sync(&mut self, _buffer_patch: &Patch) {}

    /// The last display row. `row_count − 1`, saturating.
    #[must_use]
    pub fn max_row(&self) -> DisplayRow {
        DisplayRow(self.row_count().saturating_sub(1))
    }

    /// The text of buffer row `row`, excluding its `\n` (out-of-range → `""`).
    /// A [`std::borrow::Cow`]: the rope backing lends a borrowed slice where it
    /// can and allocates only when a line spans internal chunks.
    fn line(&self, row: u32) -> std::borrow::Cow<'a, str> {
        self.lines.line(row)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(s: &str) -> Buffer {
        Buffer::new(s).unwrap()
    }

    /// A dumb per-char oracle for [`expand`]: independent of the implementation
    /// under test, so the two cannot share a mistake and mask a disagreement.
    fn oracle_expand(line: &str, byte_col: u32, tab_size: u32) -> u32 {
        let mut cell = 0u32;
        let mut byte = 0u32;
        for ch in line.chars() {
            if byte >= byte_col {
                break;
            }
            cell += if ch == '\t' {
                tab_size - (cell % tab_size)
            } else {
                1
            };
            byte += ch.len_utf8() as u32;
        }
        cell
    }

    #[test]
    fn indent_guide_level_content_rows() {
        // Content row: ceil(indent / size). size = 4.
        assert_eq!(indent_guide_level(Some(0), None, None, 4), 0);
        assert_eq!(indent_guide_level(Some(4), None, None, 4), 1);
        assert_eq!(indent_guide_level(Some(8), None, None, 4), 2);
        assert_eq!(indent_guide_level(Some(12), None, None, 4), 3);
        // Off-grid indent rounds up (a 6-cell indent shows 2 guides).
        assert_eq!(indent_guide_level(Some(6), None, None, 4), 2);
    }

    #[test]
    fn indent_guide_level_blank_rows() {
        // File edge (a neighbour missing) ⇒ level 0.
        assert_eq!(indent_guide_level(None, None, Some(4), 4), 0);
        assert_eq!(indent_guide_level(None, Some(4), None, 4), 0);
        // Equal neighbours ⇒ carry that level straight through the blank.
        assert_eq!(indent_guide_level(None, Some(12), Some(12), 4), 3);
        // Indenting in (above < below): keep the shallower outer ladder + 1.
        assert_eq!(indent_guide_level(None, Some(4), Some(8), 4), 2);
        // Dedenting out (above > below, brace-delimited): follow the deeper
        // outgoing block so the guide doesn't snap in early.
        assert_eq!(indent_guide_level(None, Some(12), Some(4), 4), 2);
    }

    #[test]
    fn active_indent_guide_scope_cases() {
        // mod m {            0
        //     fn f() {       1
        //         if x {     2
        //             a();   3
        //                    3  (blank, interpolated)
        //             b();   3
        //         }          2
        //     }              1
        // }                  0
        let levels = [0u32, 1, 2, 3, 3, 3, 2, 1, 0];
        let n = levels.len() as u32;
        let at = |r: u32| levels[r as usize];

        // Caret on a body line ⇒ its own level, extended over the run.
        assert_eq!(active_indent_guide(3, n, 0..n, at), Some((3, 3, 5)));
        // Caret on the top opener `mod m {` (level 0) ⇒ child level 1 active
        // across the whole body, NOT nothing.
        assert_eq!(active_indent_guide(0, n, 0..n, at), Some((1, 1, 7)));
        // Caret on an inner opener `fn f() {` ⇒ child level 2 over its body.
        assert_eq!(active_indent_guide(1, n, 0..n, at), Some((2, 2, 6)));
        // Caret on the final closer `}` (level 0) ⇒ end-of-scope, level 1 body.
        assert_eq!(active_indent_guide(8, n, 0..n, at), Some((1, 1, 7)));
        // Caret on an inner closer `}` (level 2), body above is level 3.
        assert_eq!(active_indent_guide(6, n, 0..n, at), Some((3, 3, 5)));
    }

    #[test]
    fn active_indent_guide_over_indented_body() {
        // mod main {            0  col0
        //     fn f() {          1  col4
        //                 l00   4  col16 (body jumps 3 levels — over-indented)
        //                 l01   4  col16
        //     }                 1  col4
        // }                     0  col0
        let levels = [0u32, 1, 4, 4, 1, 0];
        let n = levels.len() as u32;
        let at = |r: u32| levels[r as usize];

        // Caret on the opener `fn f() {` ⇒ the FIRST child level (2, cell 4) —
        // one stop in from `fn f()` — regardless of the body's real depth (4).
        assert_eq!(active_indent_guide(1, n, 0..n, at), Some((2, 2, 3)));
        // Caret on a body line ⇒ that line's own level (4, cell 12).
        assert_eq!(active_indent_guide(2, n, 0..n, at), Some((4, 2, 3)));
        // Caret on `mod main {` ⇒ its first child level 1, cell 0.
        assert_eq!(active_indent_guide(0, n, 0..n, at), Some((1, 1, 4)));
    }

    #[test]
    fn active_indent_guide_none_at_flat_level_zero() {
        // A flat, unindented document has no active guide anywhere.
        let levels = [0u32, 0, 0];
        assert_eq!(active_indent_guide(1, 3, 0..3, |r| levels[r as usize]), None);
    }

    #[test]
    fn tab_width_advances_to_next_stop() {
        // Width is 1..=tab_size, never zero; hits tab_size exactly at a stop.
        assert_eq!(tab_width(0, 4), 4);
        assert_eq!(tab_width(1, 4), 3);
        assert_eq!(tab_width(3, 4), 1);
        assert_eq!(tab_width(4, 4), 4); // back to a full step at the next stop
        assert_eq!(tab_width(7, 4), 1);
    }

    #[test]
    fn expand_no_tabs_is_scalar_count() {
        assert_eq!(expand("hello", 5, 4), 5);
        assert_eq!(expand("hello", 3, 4), 3);
        assert_eq!(expand("hello", 0, 4), 0);
        // Past EOL expands the whole line.
        assert_eq!(expand("hi", 99, 4), 2);
    }

    #[test]
    fn expand_counts_scalars_not_bytes() {
        // "é" is 2 bytes but one cell; a stray multibyte char shifts, not
        // corrupts, columns.
        let line = "aé b"; // bytes: a(1) é(2) space(1) b(1) = 5
        assert_eq!(line.len(), 5);
        assert_eq!(expand(line, 5, 4), 4); // 4 scalars
        assert_eq!(expand(line, 3, 4), 2); // after "aé" → 2 cells
    }

    #[test]
    fn expand_stretches_at_tabs() {
        // "\t" at col 0 → 4 cells; "a\t" → a(1) then tab to next stop (3) = 4.
        assert_eq!(expand("\t", 1, 4), 4);
        assert_eq!(expand("a\t", 2, 4), 4);
        assert_eq!(expand("ab\tc", 4, 4), 5); // ab(2) tab→4 c→5
        assert_eq!(expand("\t\t", 2, 4), 8); // two full tab stops
    }

    #[test]
    fn expand_matches_oracle() {
        for line in ["", "a", "\t", "a\tb\tc", "\t\ta", "aé\tb", "abcd\te"] {
            for byte_col in 0..=line.len() as u32 + 1 {
                assert_eq!(
                    expand(line, byte_col, 4),
                    oracle_expand(line, byte_col, 4),
                    "line {line:?} byte_col {byte_col}"
                );
            }
        }
    }

    #[test]
    fn collapse_hits_boundaries_exactly() {
        // No tabs: cell == byte on a boundary.
        assert_eq!(collapse("hello", 0, 4, Bias::Left), 0);
        assert_eq!(collapse("hello", 3, 4, Bias::Left), 3);
        assert_eq!(collapse("hello", 5, 4, Bias::Left), 5);
        // Past EOL clamps to the byte length.
        assert_eq!(collapse("hi", 9, 4, Bias::Right), 2);
    }

    #[test]
    fn collapse_snaps_mid_tab_by_bias() {
        // "\t" spans cells 0..4; landing at 1/2/3 is inside the expansion.
        for c in 1..4 {
            assert_eq!(collapse("\t", c, 4, Bias::Left), 0); // to the '\t' byte
            assert_eq!(collapse("\t", c, 4, Bias::Right), 1); // just after it
        }
        // Cell 0 and cell 4 are boundaries — no snap.
        assert_eq!(collapse("\t", 0, 4, Bias::Left), 0);
        assert_eq!(collapse("\tx", 4, 4, Bias::Left), 1); // after the tab, on 'x'
    }

    #[test]
    fn collapse_lands_on_char_boundaries_after_multibyte() {
        // "é" occupies bytes 0..2, one cell. Collapsing cell 1 lands on byte 2,
        // the boundary after 'é'.
        let line = "é\t";
        assert_eq!(collapse(line, 1, 4, Bias::Left), 2);
        // The tab starts at cell 1 (width 3) → mid-tab snaps to its byte (2) or
        // just after (3).
        assert_eq!(collapse(line, 2, 4, Bias::Left), 2);
        assert_eq!(collapse(line, 2, 4, Bias::Right), 3);
    }

    #[test]
    fn expand_collapse_round_trip_on_boundaries() {
        // expand ∘ collapse is identity on valid (boundary) cell columns.
        let line = "a\tbé\tc";
        let mut byte = 0u32;
        for ch in line.chars() {
            let cell = expand(line, byte, 4);
            assert_eq!(collapse(line, cell, 4, Bias::Left), byte, "byte {byte}");
            assert_eq!(collapse(line, cell, 4, Bias::Right), byte, "byte {byte}");
            byte += ch.len_utf8() as u32;
        }
    }

    #[test]
    fn default_tab_size_is_four() {
        assert_eq!(default_tab_size(), 4);
    }

    #[test]
    fn display_point_accessors() {
        let p = DisplayPoint::new(DisplayRow(2), 7);
        assert_eq!(p.row(), DisplayRow(2));
        assert_eq!(p.col(), 7);
        assert_eq!(DisplayPoint::zero(), DisplayPoint::new(DisplayRow(0), 0));
    }

    #[test]
    fn display_point_orders_row_major() {
        assert!(DisplayPoint::new(DisplayRow(0), 9) < DisplayPoint::new(DisplayRow(1), 0));
        assert!(DisplayPoint::new(DisplayRow(1), 2) < DisplayPoint::new(DisplayRow(1), 3));
    }

    #[test]
    fn to_display_expands_the_column() {
        let b = buf("a\tb\nx");
        let m = TabMap::new(&b, 4);
        // "b" sits at byte 2, cell 4 (a=1, tab→4).
        let d = m.to_display(Point::new(0, 2), Bias::Left);
        assert_eq!(d.row(), DisplayRow(0));
        assert_eq!(d.col(), 4);
        // Row is preserved: buffer row 1 → display row 1.
        assert_eq!(m.to_display(Point::new(1, 1), Bias::Left).row(), DisplayRow(1));
    }

    #[test]
    fn to_buffer_snaps_out_of_the_tab() {
        let b = buf("\tx");
        let m = TabMap::new(&b, 4);
        // Cell 2 is inside the tab expansion.
        let mid = DisplayPoint::new(DisplayRow(0), 2);
        assert_eq!(m.to_buffer(mid, Bias::Left), Point::new(0, 0));
        assert_eq!(m.to_buffer(mid, Bias::Right), Point::new(0, 1));
    }

    #[test]
    fn clip_pulls_a_caret_out_of_a_tab() {
        let b = buf("\tab");
        let m = TabMap::new(&b, 4);
        let inside = DisplayPoint::new(DisplayRow(0), 2);
        // Left snaps to the tab's start cell (0); Right to just after it (4).
        assert_eq!(m.clip(inside, Bias::Left), DisplayPoint::new(DisplayRow(0), 0));
        assert_eq!(m.clip(inside, Bias::Right), DisplayPoint::new(DisplayRow(0), 4));
    }

    #[test]
    fn clip_is_idempotent() {
        let b = buf("a\tbc\td");
        let m = TabMap::new(&b, 4);
        for col in 0..12 {
            let once = m.clip(DisplayPoint::new(DisplayRow(0), col), Bias::Left);
            let twice = m.clip(once, Bias::Left);
            assert_eq!(once, twice, "col {col}");
        }
    }

    #[test]
    fn clip_clamps_past_end_of_line() {
        let b = buf("hi");
        let m = TabMap::new(&b, 4);
        let far = DisplayPoint::new(DisplayRow(0), 99);
        assert_eq!(m.clip(far, Bias::Left), DisplayPoint::new(DisplayRow(0), 2));
    }

    #[test]
    fn row_count_and_max_row_track_lines() {
        let b = buf("a\nb\nc");
        let m = TabMap::new(&b, 4);
        assert_eq!(m.row_count(), 3);
        assert_eq!(m.max_row(), DisplayRow(2));
        // An empty document is one row; max_row saturates at 0.
        let e = buf("");
        let me = TabMap::new(&e, 4);
        assert_eq!(me.row_count(), 1);
        assert_eq!(me.max_row(), DisplayRow(0));
    }

    #[test]
    fn line_len_and_max_point_are_cells() {
        let b = buf("a\tb\nlong");
        let m = TabMap::new(&b, 4);
        assert_eq!(m.line_len(DisplayRow(0)), 5); // a(1) tab→4 b→5
        assert_eq!(m.line_len(DisplayRow(1)), 4);
        assert_eq!(m.max_point(), DisplayPoint::new(DisplayRow(1), 4));
    }

    #[test]
    fn row_info_is_identity_in_v1() {
        let b = buf("x\ny");
        let m = TabMap::new(&b, 4);
        assert_eq!(m.row_info(DisplayRow(0)), Some(BufferRow(0)));
        assert_eq!(m.row_info(DisplayRow(1)), Some(BufferRow(1)));
    }

    #[test]
    fn chunks_split_at_every_tab() {
        // The caller owns the line; the runs borrow from it.
        let b = buf("ab\tc");
        let line = b.line(0);
        let runs: Vec<_> = chunks(&line, 4).collect();
        assert_eq!(runs.len(), 3);
        // "ab": bytes 0..2, 2 cells, not a tab.
        assert_eq!(runs[0].text, "ab");
        assert_eq!(runs[0].buffer_bytes, 0..2);
        assert_eq!(runs[0].cells, 2);
        assert!(!runs[0].is_tab);
        // "\t": byte 2..3, width 2 (from cell 2 to the next stop at 4).
        assert_eq!(runs[1].text, "\t");
        assert_eq!(runs[1].buffer_bytes, 2..3);
        assert_eq!(runs[1].cells, 2);
        assert!(runs[1].is_tab);
        // "c": byte 3..4, 1 cell.
        assert_eq!(runs[2].text, "c");
        assert_eq!(runs[2].buffer_bytes, 3..4);
        assert_eq!(runs[2].cells, 1);
    }

    #[test]
    fn chunks_of_empty_and_tab_only_lines() {
        let b = buf("\n\t");
        // Empty first line → no runs.
        assert_eq!(chunks(&b.line(0), 4).count(), 0);
        // Tab-only line → a single tab run of full width.
        let line = b.line(1);
        let runs: Vec<_> = chunks(&line, 4).collect();
        assert_eq!(runs.len(), 1);
        assert!(runs[0].is_tab);
        assert_eq!(runs[0].cells, 4);
    }

    #[test]
    fn chunk_cells_sum_to_line_len() {
        let b = buf("x\t\tyz\tw");
        let m = TabMap::new(&b, 4);
        let total: u32 = chunks(&b.line(0), 4).map(|c| c.cells).sum();
        assert_eq!(total, m.line_len(DisplayRow(0)));
    }

    #[test]
    fn sync_is_a_noop_that_leaves_conversions_current() {
        let b = buf("a\tb");
        let m0 = TabMap::new(&b, 4);
        let before = m0.line_len(DisplayRow(0));
        let mut m = TabMap::new(&b, 4);
        m.sync(&Patch::new());
        assert_eq!(m.line_len(DisplayRow(0)), before);
    }
}
