//! Per-row horizontal projection — the **one owner** of byte-column ↔
//! display-cell math on rows with collapsed inline folds, and of the collapsed
//! block header's `head … tail` one-line layout.
//!
//! Every consumer that needs to place something horizontally — render, caret
//! placement, hit testing, selection, and movement — consults the functions
//! here rather than recomputing the inline-collapse shift, its inverse, the
//! chip-center formula, the caret/glyph boundary predicates, or the header
//! width that feeds a collapsed block's inline tail. Because each of those
//! facts has exactly one definition, a change to the chip layout or a boundary
//! rule is a one-site edit and every site stays in agreement by construction.
//!
//! Everything here is in **cells and columns** — GUI-free. The widget's only
//! remaining job is `x = origin + cell × advance` (and its inverse).

use std::borrow::Cow;

use crate::buffer::Buffer;
use crate::coords::{Bias, Point};
use crate::display_map::{self, BufferRow, DisplayRow};
use crate::fold_map::{FoldMap, InlineFold};

// Op-count canary: counts `display_position` probes on this thread so a test
// can assert `expand_folds_touched` calls it O(edit points) per commit — once
// per point, for the hidden-gap check — and never O(candidates · edits), which
// would make a document-scale multi-caret edit over a folded document cost the
// product of the two. Debug/test only; zero-cost in release.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static DISPLAY_POSITION_PROBES: std::cell::Cell<u64> =
        const { std::cell::Cell::new(0) };
}

/// Display cells a collapsed inline fold's `…` chip occupies. The chip
/// starts one cell past the opening bracket; the interior collapses to this.
pub const INLINE_CHIP_CELLS: u32 = 3;

/// Display cells between a collapsed block header's end and its inline closing
/// tail — the ` … ` placeholder gap.
pub const FOLD_PLACEHOLDER_CELLS: u32 = 4;

/// Where a caret at some byte column renders horizontally: a whole display
/// cell, or — for a column hidden inside a collapsed inline fold — the center
/// of that fold's `…` chip. The chip center is the **only** fractional cell in
/// the system; fencing it in this enum keeps `f32` out of the inverse maps.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum CaretCell {
    /// An exact display cell.
    Cell(u32),
    /// The fractional cell at a collapsed chip's visual center.
    ChipCenter(f32),
}

impl CaretCell {
    /// The (possibly fractional) display-cell value, for pixel projection.
    #[must_use]
    pub fn cells(self) -> f32 {
        match self {
            Self::Cell(c) => c as f32,
            Self::ChipCenter(c) => c,
        }
    }
}

/// Where a buffer offset renders: its display row plus its horizontal
/// [`CaretCell`]. THE owner of "where does offset O show on screen" — a caret,
/// selection endpoint, squiggle bound, popup anchor, and autoscroll target all
/// read the same value (see [`FoldMap::display_position`]).
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct DisplayPosition {
    /// The visible display row (a hidden closing tail resolves to its header's).
    pub row: DisplayRow,
    /// The horizontal position on that row.
    pub x: CaretCell,
}

/// One collapsed inline fold's `…` chip on a row, in display cells, as named
/// fields the widget destructures to render and hit-test the chip.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Chip {
    /// The chip's first display cell (one past the opening bracket).
    pub cell: u32,
    /// The chip's visual center, in fractional display cells.
    pub center: f32,
    /// Byte column of the opening bracket on its line.
    pub open_col: u32,
    /// Byte column of the closing bracket on its line.
    pub close_col: u32,
}

/// One visible glyph of a collapsed block's inline closing tail, resolved to
/// the display cell it occupies on the *header* row.
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct TailGlyph {
    /// Byte column on the fold's *last* buffer row.
    pub col: u32,
    /// Display cell on the header's display row.
    pub cell: u32,
    /// The character itself.
    pub ch: char,
}

/// Which region of a collapsed block header's display line a cell falls in.
#[derive(Copy, Clone, PartialEq, Debug)]
pub enum HeaderHit {
    /// On the header's own text — resolve against the header row's [`RowLayout`].
    Head,
    /// In the ` … ` placeholder gap — clamps to the header line's end.
    Gap,
    /// On the inline closing tail: the byte column on the fold's *last* row.
    Tail(u32),
}

/// The caret slot just after a pair's opening bracket — the LEFT landable edge
/// of a collapsed inline gap. The one place `open + 1` is written.
#[must_use]
pub fn gap_left_edge(open: u32) -> u32 {
    open + 1
}

/// Whether caret offset `off` is strictly inside the collapsed gap of the pair
/// `(open, close)`. THE caret-boundary rule: `open+1` and `close` are
/// landable; strictly between them hides. (Its off-by-one sibling is
/// [`gap_hides_glyph`] — a *glyph* at `open+1` hides even though the caret slot
/// there is landable. The two variants exist as exactly these two functions.)
#[must_use]
pub fn gap_hides_caret(open: u32, close: u32, off: u32) -> bool {
    off > gap_left_edge(open) && off < close
}

/// Whether the glyph at offset `off` is hidden by the collapsed pair
/// `(open, close)`: everything strictly between the brackets.
#[must_use]
pub fn gap_hides_glyph(open: u32, close: u32, off: u32) -> bool {
    off > open && off < close
}

/// Byte column where a line's visible content starts (its leading whitespace
/// length) — where a collapsed block's closing tail begins on its last row.
/// The one `trim_start` rule, shared by movement and the widget.
#[must_use]
pub fn tail_start_col(line: &str) -> u32 {
    (line.len() - line.trim_start().len()) as u32
}

/// Whether a bracket pair `(open, close)` has a *non-empty interior* — the one
/// foldability rule: there must be something between the brackets to
/// hide. Shared by the document's foldable-pair enumeration and the fold map's
/// inline/block resolution.
#[must_use]
pub fn pair_has_interior(open: u32, close: u32) -> bool {
    close > open + 1
}

/// Round a fractional display cell to the nearest cell boundary, floored at 0
/// — the quantization for a column (box) selection's *virtual* corner cells,
/// which may lie past any line's content and so can't resolve through
/// [`RowLayout::hit`] yet. The same round-half-up rule `hit` applies over real
/// content, so a box edge and a click at the same pixel agree on the boundary.
#[must_use]
pub fn virtual_cell(cell: f32) -> u32 {
    cell.round().max(0.0) as u32
}

/// One collapsed inline fold on a row, with its bracket cells precomputed.
#[derive(Copy, Clone, Debug)]
struct InlineSpan {
    /// Tab-expanded (pre-collapse) cell of the opening bracket.
    open_cell: u32,
    /// Tab-expanded (pre-collapse) cell of the closing bracket.
    close_cell: u32,
    fold: InlineFold,
}

/// A visible buffer row's horizontal projection: byte column ↔ display cell,
/// with tab expansion *and* the horizontal collapse of every root inline fold
/// on the row. Built per use by [`FoldMap::row_layout`]; holds only the row's
/// text (a [`Cow`] — borrowed straight off the backing when the row is stored
/// contiguously, owned only when it spans a chunk boundary) and copies of the
/// row's folds, so it carries no derived state that could drift out of sync
/// with the document. Like [`FoldMap`], it is cheap to rebuild and never
/// stored.
pub struct RowLayout<'a> {
    line: Cow<'a, str>,
    /// Byte offset of the row's first character (column 0), for offset-space
    /// boundary predicates.
    row_start: u32,
    tab: u32,
    /// This row's root inline folds, sorted by opening cell.
    spans: Vec<InlineSpan>,
}

impl<'a> RowLayout<'a> {
    fn new(fold_map: &FoldMap, buffer: &'a Buffer, row: BufferRow, tab: u32) -> Self {
        let line = buffer.line(row.0);
        let row_start = buffer.point_to_offset(Point::new(row.0, 0));
        // This row's inline folds — an O(log n + hits) windowed descent into the
        // fold tree, not an O(all inline folds) scan/materialize per row per frame.
        let mut spans: Vec<InlineSpan> = fold_map
            .inline_folds_on_row(row.0)
            .into_iter()
            .map(|fold| InlineSpan {
                open_cell: display_map::expand(&line, fold.open - row_start, tab),
                close_cell: display_map::expand(&line, fold.close - row_start, tab),
                fold,
            })
            .collect();
        spans.sort_by_key(|s| s.open_cell);
        Self { line, row_start, tab, spans }
    }

    /// Whether the row has no collapsed inline folds (the identity projection).
    #[must_use]
    pub fn is_plain(&self) -> bool {
        self.spans.is_empty()
    }

    /// Byte offset of the row's column 0 — for turning a chip's byte columns back
    /// into document offsets (e.g. to test a chip against the selection set).
    #[must_use]
    pub fn row_start(&self) -> u32 {
        self.row_start
    }

    /// Collapse shift at pre-collapse cell `cell`: every fold whose closing
    /// bracket sits at/before it has hidden `close−open−1` cells behind an
    /// [`INLINE_CHIP_CELLS`]-wide chip.
    fn shift_at(&self, cell: u32) -> i32 {
        self.spans
            .iter()
            .filter(|s| cell >= s.close_cell)
            .map(|s| (s.close_cell as i32 - s.open_cell as i32 - 1) - INLINE_CHIP_CELLS as i32)
            .sum()
    }

    /// Pre-collapse (tab-expanded) cell → post-collapse display cell.
    fn cell_of(&self, raw_cell: u32) -> u32 {
        (raw_cell as i32 - self.shift_at(raw_cell)).max(0) as u32
    }

    /// Byte column → display cell (tab-expanded, inline-collapsed). Total and
    /// monotone; a column hidden inside a chip maps into the chip's span (use
    /// [`Self::caret_cell`] for caret placement, which clips to the center).
    #[must_use]
    pub fn display_cell(&self, col: u32) -> u32 {
        self.cell_of(display_map::expand(&self.line, col, self.tab))
    }

    /// Caret placement for a byte column: its display cell, or the chip center
    /// when the column is hidden inside a collapsed inline fold's gap.
    #[must_use]
    pub fn caret_cell(&self, col: u32) -> CaretCell {
        let off = self.row_start + col;
        match self.spans.iter().find(|s| s.fold.hides_caret_at(off)) {
            Some(s) => CaretCell::ChipCenter(
                self.cell_of(s.open_cell + 1) as f32 + INLINE_CHIP_CELLS as f32 / 2.0,
            ),
            None => CaretCell::Cell(self.display_cell(col)),
        }
    }

    /// Whether the glyph at byte column `col` is hidden inside a collapsed
    /// inline fold (the deliberately-different sibling of the caret rule: the
    /// glyph at `open+1` hides while its caret slot stays landable).
    #[must_use]
    pub fn glyph_hidden(&self, col: u32) -> bool {
        let off = self.row_start + col;
        self.spans.iter().any(|s| s.fold.hides_glyph_at(off))
    }

    /// Inverse projection: a (fractional, unrounded) display cell → the byte
    /// column a click there lands on. Rounding policy lives HERE, not at call
    /// sites. A cell on a chip resolves to just after the opening bracket;
    /// past-EOL clamps; mid-tab snaps by `bias`.
    #[must_use]
    pub fn hit(&self, cell: f32, bias: Bias) -> u32 {
        let dc = cell.round().max(0.0) as u32;
        let mut extra = 0i32;
        for s in &self.spans {
            // Compare in DISPLAY space: prior collapsed folds shift this
            // fold's bracket cells left before the click can be tested.
            let d_open = self.cell_of(s.open_cell);
            let d_chip_end = d_open + 1 + INLINE_CHIP_CELLS;
            if dc >= d_chip_end {
                extra += (s.close_cell as i32 - s.open_cell as i32 - 1) - INLINE_CHIP_CELLS as i32;
            } else if dc > d_open {
                return s.fold.left_edge() - self.row_start; // on the chip → just after `[`
            }
        }
        let raw_cell = (dc as i32 + extra).max(0) as u32;
        display_map::collapse(&self.line, raw_cell, self.tab, bias)
    }

    /// The row's rendered display width in cells (tab-expanded, collapsed).
    /// A collapsed block's inline tail begins [`FOLD_PLACEHOLDER_CELLS`] past
    /// this — see [`HeaderLayout::tail_cell`].
    #[must_use]
    pub fn width(&self) -> u32 {
        self.display_cell(self.line.len() as u32)
    }

    /// The row's collapsed chips, in display order.
    pub fn chips(&self) -> impl Iterator<Item = Chip> + '_ {
        self.spans.iter().map(|s| {
            let cell = self.cell_of(s.open_cell + 1);
            Chip {
                cell,
                center: cell as f32 + INLINE_CHIP_CELLS as f32 / 2.0,
                open_col: s.fold.open - self.row_start,
                close_col: s.fold.close - self.row_start,
            }
        })
    }
}

/// A collapsed block fold's one-line display layout: the (inline-fold-
/// aware) header text, the ` … ` placeholder gap, then the fold's real closing
/// tail — `fn main() { … }`. The single source for the placeholder render,
/// caret placement on the tail, selection washes, hit-testing, the hover-chip
/// rect, and the preview anchor, so they agree to the pixel by construction.
pub struct HeaderLayout<'a> {
    /// The header row's own horizontal projection.
    head: RowLayout<'a>,
    /// The fold's last buffer row — the line the tail glyphs come from.
    last: BufferRow,
    tail_line: Cow<'a, str>,
    /// Byte column where the visible tail starts on `last` (its leading ws).
    tail_lead: u32,
    tab: u32,
}

impl<'a> HeaderLayout<'a> {
    /// The header row's projection (for hits resolving to [`HeaderHit::Head`]).
    #[must_use]
    pub fn head(&self) -> &RowLayout<'a> {
        &self.head
    }

    /// Rendered display width of the header text — where the placeholder gap
    /// begins. Inline-fold aware: a collapsed inline fold before the block's
    /// opener shrinks it.
    #[must_use]
    pub fn head_cells(&self) -> u32 {
        self.head.width()
    }

    /// The fractional cell at the center of the ` … ` gap, where the chip and
    /// its `…` glyph both center.
    #[must_use]
    pub fn gap_center(&self) -> f32 {
        self.head_cells() as f32 + FOLD_PLACEHOLDER_CELLS as f32 / 2.0
    }

    /// The display cell where the inline closing tail begins.
    #[must_use]
    pub fn tail_cell(&self) -> u32 {
        self.head_cells() + FOLD_PLACEHOLDER_CELLS
    }

    /// The fold's last buffer row (the tail's real home).
    #[must_use]
    pub fn last_row(&self) -> BufferRow {
        self.last
    }

    /// Byte column on the last row where the visible tail starts.
    #[must_use]
    pub fn tail_start_col(&self) -> u32 {
        self.tail_lead
    }

    /// Tab-expanded cell of the tail's first visible column on its own row.
    fn lead_cells(&self) -> u32 {
        display_map::expand(&self.tail_line, self.tail_lead, self.tab)
    }

    /// Byte column on the *last* row → display cell on the header row, using
    /// full-line tab stops (the caret/hit/selection convention). `None` for a
    /// column in the leading whitespace before the visible tail.
    #[must_use]
    pub fn tail_col_cell(&self, col: u32) -> Option<u32> {
        (col >= self.tail_lead)
            .then(|| self.tail_cell() + display_map::expand(&self.tail_line, col, self.tab) - self.lead_cells())
    }

    /// The tail's rendered width in cells.
    #[must_use]
    pub fn tail_cells(&self) -> u32 {
        display_map::expand(&self.tail_line, self.tail_line.len() as u32, self.tab) - self.lead_cells()
    }

    /// The collapsed line's total rendered width (head + gap + tail), in cells.
    #[must_use]
    pub fn width(&self) -> u32 {
        self.tail_cell() + self.tail_cells()
    }

    /// The visible tail glyphs at their header-row display cells.
    pub fn tail_glyphs(&self) -> impl Iterator<Item = TailGlyph> + '_ {
        self.tail_line[self.tail_lead as usize..].char_indices().map(move |(i, ch)| {
            let col = self.tail_lead + i as u32;
            TailGlyph {
                col,
                cell: self.tail_col_cell(col).expect("tail glyph is at/after the lead"),
                ch,
            }
        })
    }

    /// Resolve a (fractional) display cell on the collapsed header line to the
    /// region it falls in, with the ±half-cell boundaries the hit-test has
    /// always used: at/after the tail (minus half a cell) → the tail column;
    /// past the header text (plus half a cell) → the gap; else the head.
    #[must_use]
    pub fn hit(&self, cell: f32, bias: Bias) -> HeaderHit {
        if cell >= self.tail_cell() as f32 - 0.5 {
            let cell_in_tail = (cell - self.tail_cell() as f32).round().max(0.0) as u32;
            let col = display_map::collapse(&self.tail_line, self.lead_cells() + cell_in_tail, self.tab, bias);
            HeaderHit::Tail(col)
        } else if cell > self.head_cells() as f32 + 0.5 {
            HeaderHit::Gap
        } else {
            HeaderHit::Head
        }
    }
}

impl FoldMap {
    /// The horizontal projection of visible buffer `row` — built per use,
    /// like the `FoldMap` itself; never store it.
    #[must_use]
    pub fn row_layout<'a>(&self, buffer: &'a Buffer, row: BufferRow, tab: u32) -> RowLayout<'a> {
        RowLayout::new(self, buffer, row, tab)
    }

    /// The `head … tail` one-line layout of `row`, iff it is a collapsed block
    /// fold's header.
    #[must_use]
    pub fn header_layout<'a>(&self, buffer: &'a Buffer, row: BufferRow, tab: u32) -> Option<HeaderLayout<'a>> {
        let last = self.fold_at_header(row)?;
        let head = self.row_layout(buffer, row, tab);
        let tail_line = buffer.line(last.0);
        let tail_lead = tail_start_col(&tail_line);
        Some(HeaderLayout { head, last, tail_line, tail_lead, tab })
    }

    /// THE owner of "where does buffer `offset` render": its display row and
    /// horizontal cell. Follows a collapsed block's closing tail to the header
    /// row; clips a column hidden in an inline fold to its chip center. `None`
    /// iff the offset is genuinely hidden — inside a block fold's gap, or on
    /// the last row before the visible tail.
    #[must_use]
    pub fn display_position(&self, buffer: &Buffer, offset: u32, tab: u32) -> Option<DisplayPosition> {
        #[cfg(any(test, debug_assertions))]
        DISPLAY_POSITION_PROBES.with(|c| c.set(c.get() + 1));
        crate::perf::charge(1); // complexity gate: one display-map probe
        let p = buffer.offset_to_point(offset);
        let row = BufferRow(p.row);
        if !self.is_folded(row) {
            let layout = self.row_layout(buffer, row, tab);
            return Some(DisplayPosition { row: self.to_display_row(row), x: layout.caret_cell(p.col) });
        }
        // Hidden row: only a collapsed fold's closing tail is representable —
        // it rides the header's display line.
        let hdr = self.header_of_tail(row)?;
        let layout = self.header_layout(buffer, hdr, tab)?;
        let cell = layout.tail_col_cell(p.col)?;
        Some(DisplayPosition { row: self.to_display_row(hdr), x: CaretCell::Cell(cell) })
    }

    /// Inverse of [`Self::display_position`] on one visible row: a (fractional,
    /// unrounded) display cell → the byte offset a click there lands on,
    /// resolving a collapsed header's gap (→ header line end) and tail
    /// (→ the last row's column) before the plain row projection.
    #[must_use]
    pub fn hit_row(&self, buffer: &Buffer, row: BufferRow, cell: f32, bias: Bias, tab: u32) -> u32 {
        if let Some(layout) = self.header_layout(buffer, row, tab) {
            match layout.hit(cell, bias) {
                HeaderHit::Tail(col) => return buffer.point_to_offset(Point::new(layout.last_row().0, col)),
                HeaderHit::Gap => return buffer.point_to_offset(Point::new(row.0, buffer.line_len(row.0))),
                HeaderHit::Head => {}
            }
        }
        let layout = self.row_layout(buffer, row, tab);
        buffer.point_to_offset(Point::new(row.0, layout.hit(cell, bias)))
    }

    /// THE pixel-y inversion policy, in row units: a (fractional) count of
    /// display rows from the content top → the display row it falls on,
    /// floored and clamped to the valid range. Every y-driven hit (clicks,
    /// hover, gutter, boxes) routes through this one rule.
    ///
    /// `f64`, deliberately: `f32` holds fractional rows exactly only below
    /// ~2²³ rows — past that, hits land on the wrong line and (via the
    /// widget's px↔row maps) rendered rows visibly skip. `f64` is exact for
    /// every u32-addressable document.
    #[must_use]
    pub fn display_row_at(&self, rows_from_top: f64) -> DisplayRow {
        DisplayRow((rows_from_top.floor().max(0.0) as u32).min(self.display_row_count().saturating_sub(1)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::document::Document;

    /// A document with the inline pair at `inline_at` and/or the block pair at
    /// `block_at` collapsed (each the byte offset of an opening bracket).
    fn doc_with_folds(text: &str, fold_openers: &[u32]) -> Document {
        let mut doc = Document::new(text).expect("test doc fits");
        for &o in fold_openers {
            assert!(doc.toggle_fold_opener(o), "opener {o} must be foldable");
        }
        doc
    }

    fn fold_map(doc: &Document) -> FoldMap {
        FoldMap::new(doc.folds(), doc.brackets(), doc.buffer())
    }

    // ── RowLayout: the caret boundary rule — open+1 lands on the chip's left
    //    edge (a cell), never its center ──

    #[test]
    fn caret_cell_lands_on_chip_edge_not_center() {
        // `a[bcdef]g` — fold the [..] pair (opener at byte 1).
        let doc = doc_with_folds("a[bcdef]g", &[1]);
        let fm = fold_map(&doc);
        let rl = fm.row_layout(doc.buffer(), BufferRow(0), 4);
        // col 2 == open+1: the LEFT landable edge — a cell, not the center.
        assert_eq!(rl.caret_cell(2), CaretCell::Cell(2), "open+1 is landable at the chip's left edge");
        // col 3 (strict interior) clips to the chip center: cell 2 + 1.5.
        assert_eq!(rl.caret_cell(3), CaretCell::ChipCenter(2.0 + INLINE_CHIP_CELLS as f32 / 2.0));
        // col 7 == close: the RIGHT landable edge.
        assert_eq!(rl.caret_cell(7), CaretCell::Cell(rl.display_cell(7)));
        // Glyphs: open+1's glyph hides even though its caret slot is landable.
        assert!(rl.glyph_hidden(2));
        assert!(!rl.glyph_hidden(7), "the closing bracket stays visible");
        assert!(!rl.glyph_hidden(1), "the opening bracket stays visible");
    }

    // ── RowLayout: the inverse round-trips on multi-chip rows — the hit-test
    //    compares in display space, so a later chip resolves correctly ──

    #[test]
    fn hit_round_trips_display_cell_on_multi_chip_rows() {
        // Two inline pairs on one row, a tab up front, a multibyte char after.
        let text = "\tf([aa], [bb]) é";
        let open1 = text.find("[aa").unwrap() as u32;
        let open2 = text.find("[bb").unwrap() as u32;
        let doc = doc_with_folds(text, &[open1, open2]);
        let fm = fold_map(&doc);
        let rl = fm.row_layout(doc.buffer(), BufferRow(0), 4);
        assert_eq!(rl.chips().count(), 2);
        // Every landable char-boundary column round-trips through the inverse.
        let line = doc.buffer().line(0);
        for (i, _) in line.char_indices() {
            let col = i as u32;
            if rl.glyph_hidden(col) {
                continue; // interior columns resolve to the chip instead
            }
            assert_eq!(rl.hit(rl.display_cell(col) as f32, Bias::Left), col, "round-trip col {col}");
        }
        // Every cell strictly on a chip resolves to just after its `[`. The
        // second chip is the discriminating case: it resolves correctly only
        // because the compare is done in display space, not buffer space.
        for chip in rl.chips().collect::<Vec<_>>() {
            for dc in chip.cell..chip.cell + INLINE_CHIP_CELLS {
                assert_eq!(rl.hit(dc as f32, Bias::Left), chip.open_col + 1, "chip cell {dc}");
            }
        }
    }

    // ── HeaderLayout: an inline fold preceding a block fold shrinks the head,
    //    so the tail is placed against the collapsed width, not the raw one ──

    #[test]
    fn header_layout_shrinks_with_preceding_inline_fold() {
        let text = "\tcall([a, b, c]) {\n\tbody\n\t}\n";
        let inline_open = text.find('[').unwrap() as u32;
        let block_open = text.find('{').unwrap() as u32;
        let doc = doc_with_folds(text, &[inline_open, block_open]);
        let fm = fold_map(&doc);
        let hl = fm.header_layout(doc.buffer(), BufferRow(0), 4).expect("row 0 is a collapsed header");
        let raw = display_map::expand(&doc.buffer().line(0), doc.buffer().line(0).len() as u32, 4);
        // The rendered head shrank: the [..] interior collapsed to a chip.
        assert!(hl.head_cells() < raw, "head {} must be < raw {raw}", hl.head_cells());
        assert_eq!(hl.head_cells(), fm.row_layout(doc.buffer(), BufferRow(0), 4).width());
        assert_eq!(hl.tail_cell(), hl.head_cells() + FOLD_PLACEHOLDER_CELLS);
        // The tail's `}` cell agrees between the glyph list and the col map.
        let g: Vec<TailGlyph> = hl.tail_glyphs().collect();
        assert_eq!(g.len(), 1);
        assert_eq!(g[0].ch, '}');
        assert_eq!(Some(g[0].cell), hl.tail_col_cell(g[0].col));
    }

    #[test]
    fn tail_glyph_cells_match_tail_col_cell_with_tab_in_tail() {
        // A tab INSIDE the visible tail: full-line tab stops, not trimmed-line.
        let text = "f() {\nbody\n}\tx\n";
        let block_open = text.find('{').unwrap() as u32;
        let doc = doc_with_folds(text, &[block_open]);
        let fm = fold_map(&doc);
        let hl = fm.header_layout(doc.buffer(), BufferRow(0), 4).expect("collapsed header");
        for g in hl.tail_glyphs() {
            assert_eq!(Some(g.cell), hl.tail_col_cell(g.col), "glyph at col {} agrees with the col map", g.col);
        }
        // And the hit resolution lands back on each glyph's column.
        for g in hl.tail_glyphs() {
            assert_eq!(hl.hit(g.cell as f32, Bias::Left), HeaderHit::Tail(g.col));
        }
    }

    // ── display_position: the one offset→(row, x) owner — an offset below a
    //    fold resolves onto its visible display-space row ──

    #[test]
    fn display_position_follows_tail_and_hides_gap() {
        let text = "a {\nhidden\n} tail\nafter\n";
        let block_open = text.find('{').unwrap() as u32;
        let doc = doc_with_folds(text, &[block_open]);
        let fm = fold_map(&doc);
        let buffer = doc.buffer();
        // An offset inside the fold's gap is unrepresentable.
        let hidden = buffer.point_to_offset(Point::new(1, 2));
        assert_eq!(fm.display_position(buffer, hidden, 4), None);
        // The tail `}` rides the header's display row at the tail cell.
        let tail = buffer.point_to_offset(Point::new(2, 0));
        let p = fm.display_position(buffer, tail, 4).expect("tail is visible");
        assert_eq!(p.row, DisplayRow(0));
        let hl = fm.header_layout(buffer, BufferRow(0), 4).unwrap();
        assert_eq!(p.x, CaretCell::Cell(hl.tail_cell()));
        // A row below the fold is shifted up by the hidden count.
        let after = buffer.point_to_offset(Point::new(3, 0));
        let p = fm.display_position(buffer, after, 4).expect("visible");
        assert_eq!(p.row, DisplayRow(1), "rows 1..=2 hidden ⇒ row 3 displays at 1");
    }

    #[test]
    fn hit_row_resolves_head_gap_and_tail() {
        let text = "ab {\nhidden\n}\n";
        let block_open = text.find('{').unwrap() as u32;
        let doc = doc_with_folds(text, &[block_open]);
        let fm = fold_map(&doc);
        let buffer = doc.buffer();
        let hl = fm.header_layout(buffer, BufferRow(0), 4).unwrap();
        // Head: cell 0 → offset 0.
        assert_eq!(fm.hit_row(buffer, BufferRow(0), 0.0, Bias::Left, 4), 0);
        // Gap: between head end and tail → clamps to the header line's end.
        let gap_cell = hl.head_cells() as f32 + FOLD_PLACEHOLDER_CELLS as f32 / 2.0;
        assert_eq!(fm.hit_row(buffer, BufferRow(0), gap_cell, Bias::Left, 4), buffer.line_len(0));
        // Tail: the tail cell → the `}` on the last row.
        let tail_off = buffer.point_to_offset(Point::new(2, 0));
        assert_eq!(fm.hit_row(buffer, BufferRow(0), hl.tail_cell() as f32, Bias::Left, 4), tail_off);
    }

    #[test]
    fn display_row_at_floors_and_clamps() {
        let doc = doc_with_folds("a\nb\nc\n", &[]);
        let fm = fold_map(&doc);
        assert_eq!(fm.display_row_at(-2.0), DisplayRow(0));
        assert_eq!(fm.display_row_at(0.9), DisplayRow(0));
        assert_eq!(fm.display_row_at(1.0), DisplayRow(1));
        assert_eq!(fm.display_row_at(99.0), fm.max_display_row());
    }

    // ── plain rows: the projection is the identity over tab expansion ──

    #[test]
    fn plain_row_layout_is_tab_expansion() {
        let doc = doc_with_folds("\tx = 1\n", &[]);
        let fm = fold_map(&doc);
        let rl = fm.row_layout(doc.buffer(), BufferRow(0), 4);
        assert!(rl.is_plain());
        assert_eq!(rl.display_cell(0), 0);
        assert_eq!(rl.display_cell(1), 4, "tab expands to the stop");
        assert_eq!(rl.width(), 4 + "x = 1".len() as u32);
        assert_eq!(rl.hit(4.0, Bias::Left), 1);
    }
}
