//! Caret movement — the motions and the two combinators that apply them to a
//! [`SelectionSet`].
//!
//! Motions are pure functions of the buffer and a position; the combinator
//! [`move_selections`] maps one over every selection, either *moving* (collapse
//! to a caret) or *extending* (drag the head, anchor fixed). Two details worth
//! knowing: a plain horizontal move over a non-empty selection collapses to its
//! near edge (not one char further), and vertical motion carries a **goal
//! column** so travelling across short lines returns to the original column —
//! the affinity editors are judged on.
//!
//! Horizontal positions here are byte columns, but the vertical **goal is a
//! display cell**: the
//! caret keeps its *visual* column through tab stops, collapsed inline folds,
//! and collapsed block placeholders, resolved on landing by the same inverse
//! projection a mouse click uses.

use crate::buffer::Buffer;
use crate::coords::{Bias, Point};
use crate::display_map::{self, BufferRow, DisplayRow};
use crate::fold_map::FoldMap;
use crate::selection::SelectionSet;

/// A caret motion.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Motion {
    /// One character left (wraps to the end of the previous line).
    Left,
    /// One character right (wraps to the start of the next line).
    Right,
    /// One line up, honoring the goal column.
    Up,
    /// One line down, honoring the goal column.
    Down,
    /// Up by a page — the given number of viewport rows (from the widget's
    /// layout; the core has no viewport). Honors the goal column; past the top
    /// lands at the document start.
    PageUp(u32),
    /// Down by a page — the given number of viewport rows. Past the bottom lands
    /// at the document end.
    PageDown(u32),
    /// To the start of the word left of the caret.
    WordLeft,
    /// To the start/end of the word right of the caret.
    WordRight,
    /// Smart line start: toggles between the first non-whitespace column and
    /// column 0.
    LineStart,
    /// To the end of the line.
    LineEnd,
    /// To the start of the document.
    DocStart,
    /// To the end of the document.
    DocEnd,
}

/// Drag/selection granularity — the unit a click-drag extends by: single
/// characters, whole words, or whole lines.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum Granularity {
    /// Character (single-click drag).
    Char,
    /// Word (double-click drag).
    Word,
    /// Line, including the trailing newline (triple-click drag).
    Line,
}

/// A direction for column (box) selection — Ctrl+Shift+Alt+Arrow. Up/
/// Down move the active row; Left/Right move the active (virtual) column.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum ColumnDir {
    /// Extend/shrink the box one row up.
    Up,
    /// Extend/shrink the box one row down.
    Down,
    /// Move the active column one cell left.
    Left,
    /// Move the active column one cell right (may go past a line's end).
    Right,
}

/// Apply `motion` to every selection in `set`. With `extend`, each selection's
/// head moves and its anchor stays (grow/shrink); without, each collapses to a
/// caret at the motion target. Re-merges afterward (a move can make two
/// selections coincide). `tab` is the document's tab-stop width — vertical
/// motion keeps a *visual* goal column, so it needs the display projection.
pub fn move_selections(set: &mut SelectionSet, buffer: &Buffer, folds: &FoldMap, tab: u32, motion: Motion, extend: bool) {
    set.map_each(|s| {
        let (target, goal) = motion_target(buffer, folds, tab, s.head(), s.goal, motion);
        if extend {
            s.set_head(target);
            s.goal = goal;
        } else if !s.is_empty() && matches!(motion, Motion::Left | Motion::Right) {
            // Plain horizontal move collapses to the near edge of the selection.
            let edge = if motion == Motion::Left { s.start() } else { s.end() };
            s.move_to_caret(edge);
        } else {
            s.move_to_caret(target);
            s.goal = goal;
        }
    });
}

/// Compute a motion's target offset and the goal *display cell* it should
/// leave behind (`Some` only for vertical motions, which preserve it;
/// everything else clears it by returning `None`).
fn motion_target(buffer: &Buffer, folds: &FoldMap, tab: u32, head: u32, goal: Option<u32>, motion: Motion) -> (u32, Option<u32>) {
    match motion {
        Motion::Left => (char_left(buffer, folds, head), None),
        Motion::Right => (char_right(buffer, folds, head), None),
        Motion::Up => vertical_by(buffer, folds, tab, head, goal, -1),
        Motion::Down => vertical_by(buffer, folds, tab, head, goal, 1),
        Motion::PageUp(rows) => vertical_by(buffer, folds, tab, head, goal, -(rows as i32)),
        Motion::PageDown(rows) => vertical_by(buffer, folds, tab, head, goal, rows as i32),
        Motion::WordLeft => (skip_fold_left(buffer, folds, word_left(buffer, head)), None),
        Motion::WordRight => (skip_fold_right(buffer, folds, word_right(buffer, head)), None),
        Motion::LineStart => (line_start_smart_folded(buffer, folds, head), None),
        Motion::LineEnd => (line_end_folded(buffer, folds, head), None),
        Motion::DocStart => (0, None),
        Motion::DocEnd => (buffer.len(), None),
    }
}

/// Previous character boundary over the whole text (crosses `\n`), hopping any
/// collapsed fold whose gap it would land in.
fn char_left(buffer: &Buffer, folds: &FoldMap, offset: u32) -> u32 {
    if offset == 0 {
        return 0;
    }
    let prev = buffer.char_before(offset).expect("offset > 0");
    skip_fold_left(buffer, folds, offset - prev.len_utf8() as u32)
}

/// Next character boundary over the whole text (crosses `\n`), hopping any
/// collapsed fold whose gap it would land in.
fn char_right(buffer: &Buffer, folds: &FoldMap, offset: u32) -> u32 {
    if offset >= buffer.len() {
        return buffer.len();
    }
    let next = buffer.char_at(offset).expect("offset < len");
    skip_fold_right(buffer, folds, offset + next.len_utf8() as u32)
}

/// The collapsed fold whose *hidden* interior `off` falls in — `Some((header,
/// last))` only when `off` is genuinely hidden (not on the header line and not in
/// the visible closing tail, i.e. row `last` at/after its leading whitespace).
/// `None` for any visible position, so movement over unfolded text is untouched.
fn hidden_fold(buffer: &Buffer, folds: &FoldMap, off: u32) -> Option<(u32, u32)> {
    let p = buffer.offset_to_point(off);
    let (header, last) = folds.fold_containing(BufferRow(p.row))?;
    // The closing tail (row `last`, col ≥ its leading whitespace) is visible —
    // a collapsed block renders as `head … tail`, so positions on the tail from
    // its indent onward are on screen and never count as hidden.
    if p.row == last.0 && p.col >= crate::row_layout::tail_start_col(&buffer.line(last.0)) {
        return None;
    }
    Some((header.0, last.0))
}

/// The inline (single-line) fold whose collapsed interior `off` falls strictly
/// inside ([`InlineFold::hides_caret_at`]): the edges `open+1` and `close` stay
/// landable, everything between hides.
fn inline_hidden(folds: &FoldMap, off: u32) -> Option<crate::fold_map::InlineFold> {
    // The only fold that can hide `off` is the last one opening before it (roots
    // are disjoint) — an O(log) tree lookup, not an O(inline folds) scan on every
    // caret motion × N carets.
    folds.inline_fold_before(off).filter(|f| f.hides_caret_at(off))
}

/// Snap a left-moving target out of a collapsed gap to the gap's *entry* edge —
/// a block's header-line end, or an inline fold's left landable edge. A no-op
/// for visible positions.
fn skip_fold_left(buffer: &Buffer, folds: &FoldMap, off: u32) -> u32 {
    if let Some((header, _)) = hidden_fold(buffer, folds, off) {
        return buffer.point_to_offset(Point::new(header, buffer.line_len(header)));
    }
    inline_hidden(folds, off).map_or(off, |f| f.left_edge())
}

/// Snap a right-moving target out of a collapsed gap to the gap's *exit* edge —
/// a block's closing-tail first column, or an inline fold's right landable
/// edge. A no-op for visible positions.
fn skip_fold_right(buffer: &Buffer, folds: &FoldMap, off: u32) -> u32 {
    if let Some((_, last)) = hidden_fold(buffer, folds, off) {
        return buffer.point_to_offset(Point::new(last, crate::row_layout::tail_start_col(&buffer.line(last))));
    }
    inline_hidden(folds, off).map_or(off, |f| f.right_edge())
}

/// Vertical motion by a signed row `delta` (±1 for Up/Down, ±page for the Page
/// moves), honoring a *visual* goal column — a display cell, not a byte column:
/// the caret keeps its on-screen column through tab stops, collapsed
/// inline folds, and a collapsed block's one-line placeholder. Stepping happens
/// in DISPLAY rows (a folded interior is not a display row, so the caret can
/// never land inside a fold), and the landing resolves through the standard
/// inverse projection ([`FoldMap::hit_row`]) — exactly like a click at the goal
/// cell: a goal in a collapsed header's gap clamps to the header's end, one
/// over the tail lands on the tail's real offset, one on a chip snaps to its
/// landable left edge, one mid-tab snaps by bias, and past-EOL clamps.
/// Overshooting the top lands at the document start, the bottom at the
/// document end — so a single-row move at an edge lands on the nearest document
/// end, and page moves that overshoot collapse to those same ends.
fn vertical_by(buffer: &Buffer, folds: &FoldMap, tab: u32, offset: u32, goal: Option<u32>, delta: i32) -> (u32, Option<u32>) {
    // The caret's display position (fold/tab aware). A caret can only rest on
    // a visible slot, but fall back to the raw expansion defensively.
    let (row, cell) = match folds.display_position(buffer, offset, tab) {
        Some(p) => (p.row, p.x.cells()),
        None => {
            let p = buffer.offset_to_point(offset);
            (folds.to_display_row(BufferRow(p.row)), display_map::expand(&buffer.line(p.row), p.col, tab) as f32)
        }
    };
    let goal_cell = goal.unwrap_or_else(|| cell.round() as u32);
    let target = row.0 as i32 + delta;
    if target < 0 {
        return (0, Some(goal_cell)); // past the top → document start
    }
    let last_display = folds.display_row_count().saturating_sub(1) as i32;
    if target > last_display {
        return (buffer.len(), Some(goal_cell)); // past the bottom → document end
    }
    let new_row = folds.to_buffer_row(DisplayRow(target as u32));
    let off = folds.hit_row(buffer, new_row, goal_cell as f32, Bias::Left, tab);
    (off, Some(goal_cell))
}

/// The offset one display row above (`delta = -1`) / below (`+1`) `offset`, at
/// its visual column — `None` when `offset` is already on the first/last
/// display row (nothing to add there). The landing for add-cursor-above/below:
/// one call into [`vertical_by`], so tabs, chips, and collapsed headers
/// resolve exactly as plain vertical movement does.
pub(crate) fn caret_one_display_row(
    buffer: &Buffer,
    folds: &FoldMap,
    tab: u32,
    offset: u32,
    delta: i32,
) -> Option<u32> {
    let (off, _) = vertical_by(buffer, folds, tab, offset, None, delta);
    // vertical_by clamps an over-the-edge step to the doc start/end — which
    // stays on the source's display row; reject that instead of adding there.
    let row_of = |o: u32| folds.to_display_row(BufferRow(buffer.offset_to_point(o).row));
    (row_of(off) != row_of(offset)).then_some(off)
}

/// The end of the line the caret is on (excludes the `\n`).
fn line_end(buffer: &Buffer, offset: u32) -> u32 {
    let row = buffer.offset_to_point(offset).row;
    buffer.point_to_offset(Point::new(row, buffer.line_len(row)))
}

/// End of the caret's DISPLAY line: on a collapsed block's header, the
/// visible line reads `fn main() { … }` — End goes past the closing tail
/// (the *last* row's end), not to the header text's own end mid-placeholder.
/// Everywhere else (including the tail itself, whose buffer row IS the last
/// row) this is the plain line end.
fn line_end_folded(buffer: &Buffer, folds: &FoldMap, offset: u32) -> u32 {
    let row = buffer.offset_to_point(offset).row;
    if let Some(last) = folds.fold_at_header(BufferRow(row)) {
        return buffer.point_to_offset(Point::new(last.0, buffer.line_len(last.0)));
    }
    line_end(buffer, offset)
}

/// Smart Home: the first non-whitespace column, unless already there, in which
/// case column 0.
fn line_start_smart(buffer: &Buffer, offset: u32) -> u32 {
    let p = buffer.offset_to_point(offset);
    line_start_smart_on(buffer, p.row, p.col)
}

/// [`line_start_smart`] against an explicit `(row, col)` — the col decides the
/// toggle, the row supplies the line (they differ for a collapsed tail, whose
/// display line is the header's).
fn line_start_smart_on(buffer: &Buffer, row: u32, col: u32) -> u32 {
    let line = buffer.line(row);
    let indent = crate::row_layout::tail_start_col(&line);
    let line_start = buffer.point_to_offset(Point::new(row, 0));
    if col == indent {
        line_start // toggle back to column 0
    } else {
        line_start + indent
    }
}

/// Start of the caret's DISPLAY line: from a collapsed fold's closing
/// tail (which rides the header's display line), Home goes to the *header's*
/// smart start — the visible line's beginning — not the hidden last row's own
/// indent. Everywhere else this is the plain smart Home.
fn line_start_smart_folded(buffer: &Buffer, folds: &FoldMap, offset: u32) -> u32 {
    let p = buffer.offset_to_point(offset);
    if let Some(hdr) = folds.header_of_tail(BufferRow(p.row)) {
        // A tail col can never equal the header's indent-col compare target
        // meaningfully — pass a sentinel col that always lands on the header's
        // first non-whitespace (the second press, now ON the header, toggles).
        return line_start_smart_on(buffer, hdr.0, u32::MAX);
    }
    line_start_smart(buffer, offset)
}

/// Character class for word-boundary detection.
/// Ordered `Whitespace < Punct < Word` so `max(kind(prev), kind(next))` lets a
/// word char win at a boundary — the tie-break [`surrounding_word`] needs.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord)]
enum CharKind {
    Whitespace,
    Punct,
    Word,
}

/// Whether `c` is a word character — alphanumeric or `_`. The ONE owner of the
/// word-boundary rule: word motion ([`char_kind`]) and the auto-close
/// quote guard ([`crate::autoclose::is_word_char`], which delegates here) share
/// it, so they cannot disagree about where a word ends. (Completion's word rule
/// is deliberately *wider* — it admits `-` for kebab identifiers — and stays
/// distinct: see [`crate::intel::providers::is_completion_word_char`].)
pub(crate) fn is_word_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_'
}

fn char_kind(c: char) -> CharKind {
    if c.is_whitespace() {
        CharKind::Whitespace
    } else if is_word_char(c) {
        CharKind::Word
    } else {
        CharKind::Punct
    }
}

/// Word-left: skip whitespace, then consume a run of one non-whitespace kind.
fn word_left(buffer: &Buffer, offset: u32) -> u32 {
    // Single-char reads: the scan never materializes the whole line's text.
    let prev = |o: usize| buffer.char_before(o as u32);
    let mut o = offset as usize;
    // Cross AT MOST one line boundary: at a line start, step to the previous
    // line's end, then find that line's word. A caret at a line start has `\n`
    // immediately behind it.
    if prev(o) == Some('\n') {
        o -= 1; // → previous line end
        if o == 0 || prev(o) == Some('\n') {
            return o as u32; // empty previous line (or doc start) → stop there
        }
    }
    // Same line: skip trailing whitespace, then consume one character-kind run —
    // never crossing another `\n`.
    while let Some(c) = prev(o).filter(|&c| c != '\n' && c.is_whitespace()) {
        o -= c.len_utf8();
    }
    if let Some(c) = prev(o).filter(|&c| c != '\n') {
        let kind = char_kind(c);
        while let Some(c) = prev(o).filter(|&c| c != '\n' && char_kind(c) == kind) {
            o -= c.len_utf8();
        }
    }
    o as u32
}

/// Word-right: skip whitespace, then consume a run of one non-whitespace kind.
fn word_right(buffer: &Buffer, offset: u32) -> u32 {
    let len = buffer.len() as usize;
    let next = |o: usize| buffer.char_at(o as u32);
    let mut o = offset as usize;
    // Cross AT MOST one line boundary: at a line end, step to the next line's
    // start, then find that line's word. A caret at a line end has `\n`
    // immediately ahead of it.
    if next(o) == Some('\n') {
        o += 1; // → next line start
        if o == len || next(o) == Some('\n') {
            return o as u32; // empty next line (or doc end) → stop there
        }
    }
    // Same line: skip leading whitespace, then consume one kind-run to its end.
    while let Some(c) = next(o).filter(|&c| c != '\n' && c.is_whitespace()) {
        o += c.len_utf8();
    }
    if let Some(c) = next(o).filter(|&c| c != '\n') {
        let kind = char_kind(c);
        while o < len {
            match next(o).filter(|&c| c != '\n' && char_kind(c) == kind) {
                Some(c) => o += c.len_utf8(),
                None => break,
            }
        }
    }
    o as u32
}

/// Word-delete target *left* of `caret` (Ctrl+Backspace), matching the
/// word-deletion behaviour of mainstream editors:
///
/// 1. The newline is a hard stop: at column 0 (char before the caret is `\n`)
///    return `caret − 1`, deleting *only* the newline, and the scan otherwise
///    never crosses a `\n`.
/// 2. **Whitespace heuristic**: if **two or more** non-newline whitespace chars
///    sit immediately before the caret, delete only that whitespace run — never
///    the preceding word too. (One or zero → fall through to word deletion,
///    which does eat the trailing single space.)
/// 3. Otherwise delete back to the previous word start: the ≤1 whitespace plus
///    one character-kind run, never crossing `\n`.
pub(crate) fn word_delete_left(buffer: &Buffer, caret: u32) -> u32 {
    let mut o = caret as usize;
    let prev = |o: usize| buffer.char_before(o as u32);
    if prev(o) == Some('\n') {
        return (o - 1) as u32; // (1) at column 0, delete just the newline
    }
    // Measure the non-newline whitespace run immediately before the caret.
    let (mut ws, mut ws_chars) = (o, 0u32);
    while let Some(c) = prev(ws).filter(|&c| c != '\n' && c.is_whitespace()) {
        ws -= c.len_utf8();
        ws_chars += 1;
    }
    if ws_chars >= 2 {
        return ws as u32; // (2) whitespace heuristic: delete only the run
    }
    // (3) delete the ≤1 whitespace plus one kind-run, never crossing \n.
    o = ws;
    if let Some(c) = prev(o).filter(|&c| c != '\n') {
        let kind = char_kind(c);
        while let Some(c) = prev(o).filter(|&c| c != '\n' && char_kind(c) == kind) {
            o -= c.len_utf8();
        }
    }
    o as u32
}

/// Word-delete target *right* of `caret` (Ctrl+Delete) — the
/// [`word_delete_left`] mirror: at end of line (`\n` ahead) delete only the
/// newline; the whitespace heuristic deletes a run of 2+ leading whitespace by
/// itself; otherwise delete forward to the next word end.
pub(crate) fn word_delete_right(buffer: &Buffer, caret: u32) -> u32 {
    let len = buffer.len() as usize;
    let mut o = caret as usize;
    let next = |o: usize| buffer.char_at(o as u32);
    if next(o) == Some('\n') {
        return (o + 1) as u32; // at EOL, delete just the newline
    }
    let (mut ws, mut ws_chars) = (o, 0u32);
    while let Some(c) = next(ws).filter(|&c| c != '\n' && c.is_whitespace()) {
        ws += c.len_utf8();
        ws_chars += 1;
    }
    if ws_chars >= 2 {
        return ws as u32; // whitespace heuristic
    }
    o = ws;
    if let Some(c) = next(o).filter(|&c| c != '\n') {
        let kind = char_kind(c);
        while o < len {
            match next(o).filter(|&c| c != '\n' && char_kind(c) == kind) {
                Some(c) => o += c.len_utf8(),
                None => break,
            }
        }
    }
    o as u32
}

/// The word range surrounding `offset`: the run of one non-whitespace
/// [`CharKind`] touching the caret. Kind is `max(kind(prev), kind(next))` so a
/// caret at a word/punctuation boundary prefers the word; the run never
/// crosses a `\n` (whitespace, so already a different kind). Returns `None` when
/// the caret sits in whitespace or the buffer is empty — Ctrl+D then has nothing
/// to select.
pub(crate) fn surrounding_word(buffer: &Buffer, offset: u32) -> Option<(u32, u32)> {
    let prev = |o: usize| buffer.char_before(o as u32);
    let next = |o: usize| buffer.char_at(o as u32);
    let o = offset as usize;
    let kind = match (prev(o).map(char_kind), next(o).map(char_kind)) {
        (Some(a), Some(b)) => a.max(b),
        (Some(a), None) => a,
        (None, Some(b)) => b,
        (None, None) => return None,
    };
    if kind == CharKind::Whitespace {
        return None;
    }
    let mut start = o;
    while let Some(c) = prev(start) {
        if char_kind(c) == kind {
            start -= c.len_utf8();
        } else {
            break;
        }
    }
    let mut end = o;
    while let Some(c) = next(end) {
        if char_kind(c) == kind {
            end += c.len_utf8();
        } else {
            break;
        }
    }
    Some((start as u32, end as u32))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Point;

    fn buf(s: &str) -> Buffer {
        Buffer::new(s).unwrap()
    }

    /// `move_selections` with no folds (identity) — the movement tests don't
    /// exercise folding; the fold-aware vertical path is covered in `fold_map` /
    /// `document` tests.
    fn mv(set: &mut SelectionSet, b: &Buffer, motion: Motion, extend: bool) {
        use crate::fold_map::{FoldMap, FoldSet};
        move_selections(
            set,
            b,
            &FoldMap::new(&FoldSet::new(), &crate::bracket::Brackets::default(), b),
            display_map::default_tab_size(),
            motion,
            extend,
        );
    }

    /// Offset of `(row, col)`.
    fn at(b: &Buffer, row: u32, col: u32) -> u32 {
        b.point_to_offset(Point::new(row, col))
    }

    fn head_point(set: &SelectionSet, b: &Buffer) -> Point {
        b.offset_to_point(set.all()[0].head())
    }

    #[test]
    fn left_right_wrap_across_lines() {
        let b = buf("ab\ncd");
        let mut set = SelectionSet::new(at(&b, 1, 0)); // start of "cd"
        mv(&mut set, &b, Motion::Left, false);
        assert_eq!(head_point(&set, &b), Point::new(0, 2)); // end of "ab"
        mv(&mut set, &b, Motion::Right, false);
        assert_eq!(head_point(&set, &b), Point::new(1, 0)); // back to start of "cd"
    }

    #[test]
    fn inline_hidden_binary_search_picks_the_right_fold() {
        use crate::document::Document;
        // Three inline pairs on three rows, all collapsed. `inline_hidden` binary-
        // searches the opener-sorted set: an interior offset resolves to *its own*
        // fold, and the edges / gaps between folds resolve to None — the
        // partition_point boundary, not a linear find. Offsets: row0 `[`=1 `]`=5,
        // row1 `[`=9 `]`=13, row2 `[`=17 `]`=21.
        let text = "a[bcd]e\nf[ghi]j\nk[lmn]o\n";
        let mut doc = Document::new(text).unwrap();
        for open in [1u32, 9, 17] {
            assert!(doc.toggle_fold_opener(open), "pair at {open} folds");
        }
        let fm = doc.fold_map();
        // Interior of the second fold resolves to *that* fold (open == 9), not the
        // first or third — the off-by-one a linear `.find` can't get wrong but a
        // partition_point can.
        assert_eq!(inline_hidden(&fm, 11).map(|f| f.open), Some(9));
        assert_eq!(inline_hidden(&fm, 3).map(|f| f.open), Some(1));
        assert_eq!(inline_hidden(&fm, 19).map(|f| f.open), Some(17));
        // Before all folds, in a gap after a closer, and at a row start before its
        // fold all resolve to nothing.
        assert_eq!(inline_hidden(&fm, 0), None);
        assert_eq!(inline_hidden(&fm, 6), None);
        assert_eq!(inline_hidden(&fm, 8), None);
    }

    #[test]
    fn plain_horizontal_move_collapses_selection_to_edge() {
        let b = buf("hello");
        let mut set = SelectionSet::new(0);
        set.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 1, 4));
        mv(&mut set, &b, Motion::Left, false);
        assert!(set.all()[0].is_empty());
        assert_eq!(set.all()[0].head(), 1); // near (left) edge, not 0
    }

    #[test]
    fn vertical_move_preserves_goal_column_over_short_lines() {
        // Column 5 on line 0, down through a 2-char line, down to a long line:
        // the caret returns to column 5.
        let b = buf("0123456789\nab\nlongenough");
        let mut set = SelectionSet::new(at(&b, 0, 5));
        mv(&mut set, &b, Motion::Down, false);
        assert_eq!(head_point(&set, &b), Point::new(1, 2)); // clamped to "ab"
        mv(&mut set, &b, Motion::Down, false);
        assert_eq!(head_point(&set, &b), Point::new(2, 5)); // goal 5 restored
    }

    #[test]
    fn page_moves_jump_by_rows_and_clamp_to_the_document_ends() {
        let b = Buffer::new("l0\nl1\nl2\nl3\nl4").unwrap(); // 5 rows
        let mut set = SelectionSet::new(0); // caret at (0,0)
        mv(&mut set, &b, Motion::PageDown(2), false);
        assert_eq!(b.offset_to_point(set.newest().head()).row, 2);
        mv(&mut set, &b, Motion::PageDown(10), false); // overshoot ↓
        assert_eq!(set.newest().head(), b.len(), "clamps to the document end");
        mv(&mut set, &b, Motion::PageUp(10), false); // overshoot ↑
        assert_eq!(set.newest().head(), 0, "clamps to the document start");
    }

    #[test]
    fn word_delete_targets_stop_at_newlines() {
        let b = Buffer::new("foo bar\nbaz").unwrap(); // f0..r6 \n7 b8 a9 z10, len 11
        assert_eq!(word_delete_left(&b, 11), 8); // deletes "baz"
        assert_eq!(word_delete_left(&b, 8), 7); // at col 0, only the newline
        assert_eq!(word_delete_right(&b, 0), 3); // deletes "foo"
        assert_eq!(word_delete_right(&b, 7), 8); // at EOL, only the newline
    }

    #[test]
    fn word_delete_eats_only_a_multi_space_run() {
        // 2+ whitespace before/after the caret → delete ONLY the whitespace run.
        let b = Buffer::new("foo   bar").unwrap(); // f0 o1 o2 sp3 sp4 sp5 b6 a7 r8
        assert_eq!(word_delete_left(&b, 6), 3); // "foo   |bar" → deletes "   ", not the word
        assert_eq!(word_delete_right(&b, 3), 6); // "foo|   bar" → deletes "   " forward
        // Exactly one whitespace → fall through to word delete (eats the space too).
        let b1 = Buffer::new("foo bar").unwrap();
        assert_eq!(word_delete_left(&b1, 4), 0); // "foo |bar" → deletes "foo "
        assert_eq!(word_delete_right(&b1, 3), 7); // "foo| bar" → deletes " bar"
        // Trailing whitespace run at EOL: delete just the run, keep the word.
        let b2 = Buffer::new("foo  ").unwrap();
        assert_eq!(word_delete_left(&b2, 5), 3); // "foo  |" → deletes "  "
    }

    #[test]
    fn word_motion_stops_at_boundaries() {
        let b = buf("foo bar_baz  qux");
        let mut set = SelectionSet::new(0);
        mv(&mut set, &b, Motion::WordRight, false);
        assert_eq!(set.all()[0].head(), 3); // after "foo"
        mv(&mut set, &b, Motion::WordRight, false);
        assert_eq!(set.all()[0].head(), 11); // after "bar_baz" (one word run)
        mv(&mut set, &b, Motion::WordLeft, false);
        assert_eq!(set.all()[0].head(), 4); // back to start of "bar_baz"
    }

    #[test]
    fn word_motion_crosses_one_line_boundary_then_finds_the_word() {
        // At a line edge, word motion steps ONE line and finds that line's
        // word. "foo\nbar" — f0 o1 o2 \n3 b4 a5 r6.
        let b = buf("foo\nbar");
        assert_eq!(word_left(&b, 4), 0); // "bar" start → "foo" start (prev line's word)
        assert_eq!(word_right(&b, 3), 7); // "foo" end → "bar" end (next line's word)
    }

    #[test]
    fn word_motion_stops_on_a_blank_line() {
        // A blank line is crossed at most once: motion STOPS on it (never skips
        // straight to the far word). "foo\n\nbar" — f0 o1 o2 \n3 \n4 b5 a6 r7.
        let b = buf("foo\n\nbar");
        assert_eq!(word_left(&b, 5), 4); // "bar" start → the blank line, not "foo"
        assert_eq!(word_right(&b, 3), 4); // "foo" end → the blank line, not "bar"
    }

    #[test]
    fn smart_home_toggles() {
        let b = buf("    indented");
        let mut set = SelectionSet::new(at(&b, 0, 12)); // end of line
        mv(&mut set, &b, Motion::LineStart, false);
        assert_eq!(head_point(&set, &b), Point::new(0, 4)); // first non-ws
        mv(&mut set, &b, Motion::LineStart, false);
        assert_eq!(head_point(&set, &b), Point::new(0, 0)); // toggle to col 0
        mv(&mut set, &b, Motion::LineStart, false);
        assert_eq!(head_point(&set, &b), Point::new(0, 4)); // and back
    }

    #[test]
    fn extend_moves_head_keeps_anchor() {
        let b = buf("hello world");
        let mut set = SelectionSet::new(0);
        mv(&mut set, &b, Motion::WordRight, true); // extend over "hello"
        mv(&mut set, &b, Motion::WordRight, true); // and " world"
        let s = &set.all()[0];
        assert_eq!((s.start(), s.end()), (0, 11));
        assert!(!s.is_empty());
    }

    #[test]
    fn surrounding_word_selects_the_word_at_the_caret() {
        let b = buf("foo bar_baz  qux");
        assert_eq!(surrounding_word(&b, 6), Some((4, 11))); // inside "bar_baz"
        assert_eq!(surrounding_word(&b, 4), Some((4, 11))); // at its start boundary
        assert_eq!(surrounding_word(&b, 11), Some((4, 11))); // at its end boundary
        assert_eq!(surrounding_word(&b, 12), None); // in whitespace → no word
    }

    #[test]
    fn surrounding_word_stops_at_newline_and_picks_punct() {
        let b = buf("ab\n++");
        assert_eq!(surrounding_word(&b, 2), Some((0, 2))); // "ab", never crossing \n
        assert_eq!(surrounding_word(&b, 4), Some((3, 5))); // the "++" punctuation run
    }

    #[test]
    fn multi_cursor_moves_and_remerges() {
        let b = buf("a b c");
        let mut set = SelectionSet::new(0);
        set.add_caret(2); // carets at 0 and 2
        set.add_caret(4); // and 4
        assert_eq!(set.len(), 3);
        // Move all left: 0→0, 2→1, 4→3 — still distinct.
        mv(&mut set, &b, Motion::Left, false);
        assert_eq!(set.len(), 3);
        // Home collapses every caret to column 0 → they merge into one.
        mv(&mut set, &b, Motion::LineStart, false);
        assert_eq!(set.len(), 1);
    }
}
