//! Completion-popup geometry: the pure placement, windowing, and sizing math for
//! the completion popup. The rendering itself lives in the editor widget (it
//! reuses the widget's fill/text helpers); this module is the testable geometry
//! it drives. Every function is recomputed each frame against the live caret, so
//! the popup tracks the caret rather than freezing at the position it opened.

use iced::{Point, Rectangle, Size};
use scrive_core::PopupList;

/// Max rows shown at once before the list windows to keep `selected` in view.
pub const POPUP_MAX_VISIBLE: usize = 12;
/// Min popup width, in cells.
pub const POPUP_MIN_CH: usize = 20;
/// Max popup width, in cells.
pub const POPUP_MAX_CH: usize = 60;
/// Inner padding, in pixels.
pub const POPUP_PAD: f32 = 4.0;
/// Max lines the selected item's doc block occupies below the list.
pub const POPUP_DOC_MAX_LINES: usize = 4;

/// The first visible filtered index that keeps `selected` in view — stateless,
/// recomputed each frame (no popup scrollbar in v1). Below the visible cap the
/// window is the whole list; past it, `selected` rides the bottom edge.
#[must_use]
pub fn window_start(selected: usize, filtered: usize) -> usize {
    if filtered <= POPUP_MAX_VISIBLE {
        0
    } else {
        selected.saturating_sub(POPUP_MAX_VISIBLE - 1).min(filtered - POPUP_MAX_VISIBLE)
    }
}

/// The screen y of visible row `vis` in a popup whose box top is `origin_y` —
/// the forward half of the row map. Its inverse is [`row_at`]; keeping the
/// pair adjacent is what stops the draw path and the click path from drifting.
#[must_use]
pub fn row_y(origin_y: f32, vis: usize, line_h: f32) -> f32 {
    origin_y + POPUP_PAD + vis as f32 * line_h
}

/// The visible row index under screen `y` in a popup whose box top is
/// `origin_y` — the exact inverse of [`row_y`]. `None` off the `visible` rows
/// (in the padding, or the doc block below).
#[must_use]
pub fn row_at(origin_y: f32, y: f32, line_h: f32, visible: usize) -> Option<usize> {
    let row = ((y - origin_y - POPUP_PAD) / line_h).floor();
    (row >= 0.0 && (row as usize) < visible).then_some(row as usize)
}

/// The selected item's doc string, if any.
#[must_use]
pub fn selected_doc(list: &PopupList) -> Option<&str> {
    list.filtered
        .get(list.selected as usize)
        .and_then(|&i| list.items[i as usize].doc.as_deref())
}

/// The width of the popup body, in cells: the widest visible `label` + `detail`,
/// clamped to `[POPUP_MIN_CH, POPUP_MAX_CH]`.
#[must_use]
pub fn width_cells(list: &PopupList) -> usize {
    let n = list.filtered.len();
    let visible = n.min(POPUP_MAX_VISIBLE);
    let start = window_start(list.selected as usize, n);
    let widest = list.filtered[start..start + visible]
        .iter()
        .map(|&i| {
            let it = &list.items[i as usize];
            it.label.chars().count() + it.detail.as_ref().map_or(0, |d| d.chars().count() + 2)
        })
        .max()
        .unwrap_or(0);
    widest.clamp(POPUP_MIN_CH, POPUP_MAX_CH)
}

/// The number of doc-block lines below the list (0 when the selected item has no
/// doc), capped at `POPUP_DOC_MAX_LINES`.
#[must_use]
pub fn doc_lines(list: &PopupList, width_cells: usize) -> usize {
    selected_doc(list).map_or(0, |d| d.chars().count().div_ceil(width_cells.max(1)).clamp(1, POPUP_DOC_MAX_LINES))
}

/// The popup's pixel size for `list` at `advance` × `line_h` metrics.
#[must_use]
pub fn extent(list: &PopupList, advance: f32, line_h: f32) -> Size {
    let cells = width_cells(list);
    let visible = list.filtered.len().min(POPUP_MAX_VISIBLE);
    let rows = visible + doc_lines(list, cells);
    Size::new(cells as f32 * advance + 2.0 * POPUP_PAD, rows as f32 * line_h + 2.0 * POPUP_PAD)
}

/// The popup's top-left: anchored at `anchor_x` (the completion word's start),
/// below the caret line (`caret_bottom`), flipping above (`caret_top − height`)
/// when there isn't room below; `x` clamped so the box stays within `bounds`.
#[must_use]
pub fn place(anchor_x: f32, caret_top: f32, caret_bottom: f32, size: Size, bounds: Rectangle) -> Point {
    let x = anchor_x.clamp(bounds.x, (bounds.x + bounds.width - size.width).max(bounds.x));
    let y = if caret_bottom + size.height <= bounds.y + bounds.height {
        caret_bottom
    } else {
        (caret_top - size.height).max(bounds.y)
    };
    Point::new(x, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn window_keeps_selected_visible() {
        // Short lists never window.
        assert_eq!(window_start(0, 5), 0);
        assert_eq!(window_start(4, 5), 0);
        // A long list: selected rides the bottom past the cap, clamped to the end.
        assert_eq!(window_start(0, 30), 0);
        assert_eq!(window_start(11, 30), 0); // still fits in [0, 12)
        assert_eq!(window_start(12, 30), 1); // scroll one
        assert_eq!(window_start(29, 30), 18); // last row: 30 - 12
    }

    #[test]
    fn place_below_then_flips_above_when_tight() {
        let bounds = Rectangle { x: 0.0, y: 0.0, width: 400.0, height: 300.0 };
        let size = Size::new(120.0, 80.0);
        // Caret near the top: plenty of room below → placed at caret_bottom.
        assert_eq!(place(50.0, 20.0, 40.0, size, bounds).y, 40.0);
        // Caret near the bottom: no room below → flips above (caret_top - height).
        assert_eq!(place(50.0, 280.0, 300.0, size, bounds).y, 200.0);
    }

    #[test]
    fn place_clamps_x_into_bounds() {
        let bounds = Rectangle { x: 0.0, y: 0.0, width: 400.0, height: 300.0 };
        let size = Size::new(120.0, 80.0);
        // Anchored past the right edge → clamped so the box stays in view.
        assert_eq!(place(350.0, 20.0, 40.0, size, bounds).x, 280.0); // 400 - 120
        // Negative anchor → clamped to the left edge.
        assert_eq!(place(-10.0, 20.0, 40.0, size, bounds).x, 0.0);
    }
}
