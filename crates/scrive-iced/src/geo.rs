//! Per-frame pixel projection — the widget half of the px ↔ cell/row seam.
//!
//! [`Geo`] is the **one owner** of every px ↔ cell/row conversion in the
//! widget: the text-area origin (`bounds.x + gutter + TEXT_PAD`, scrolled by
//! `−scroll_x`), its exact inverse, and the row ↔ y maps anchored to the
//! [`ScrollAnchor`]. Folding the forward affix, the `top` derivation, and the
//! geometry helpers into one value — rather than repeating the arithmetic at
//! each call site — means the projection is stated once and every caller reads
//! it back through named methods, so two same-typed `f32`s can never be
//! transposed into a silently-wrong position. The frame is constructed once
//! per `draw`/`update`/`mouse_interaction` pass from live state and never
//! persisted, so it cannot drift from the state it was built from.
//!
//! `Geo` does **only** pixel scaling. All cell/row *policy* — rounding,
//! clamping, chip inversion, fold-tail resolution — lives core-side
//! (`FoldMap::display_row_at`, `RowLayout::hit`, …): the widget converts px to
//! fractional cells/rows and hands them to the core's owners.

use iced::Rectangle;
use scrive_core::DisplayRow;

/// Padding between the gutter's right edge and the first text cell.
pub(crate) const TEXT_PAD: f32 = 6.0;

/// Corner radius of the collapsed-fold `…` pill — paired with
/// [`Geo::chip_pill`] so the painted pill and any overlay on it round alike.
pub(crate) const CHIP_PILL_RADIUS: f32 = 3.0;

/// The vertical scroll position of truth: a **display-row anchor** plus the
/// sub-row pixel offset of the viewport's top edge below that row's top. Scroll
/// position is expressed in line units rather than a flat pixel offset.
///
/// A flat pixel offset loses row-level precision past 2²⁴ px (~840k rows at
/// 20 px), at which point rendered rows snap and skip as they round to the
/// nearest exactly-representable pixel. The anchor keeps all position math in
/// integer row space plus a bounded `[0, line_h)` float, exact for every
/// u32-addressable document — no wider float in state, no magnitude cliff.
///
/// `row` is a raw display-row *ordinal* (fold-aware: it renumbers as folds
/// open and close), clamped each layout pass; it is state, not a minted
/// [`DisplayRow`].
#[derive(Copy, Clone, Debug, PartialEq)]
pub(crate) struct ScrollAnchor {
    /// The first display row the viewport's top edge touches.
    pub(crate) row: u32,
    /// Pixels the top edge sits below that row's top; canonically in
    /// `[0, line_h)` — [`Self::from_rows`] produces canonical anchors, and
    /// transient wheel deltas re-canonicalize through it.
    pub(crate) offset_px: f32,
}

impl ScrollAnchor {
    /// The anchor at the very top of the document.
    pub(crate) const TOP: Self = Self { row: 0, offset_px: 0.0 };

    /// This position in fractional display-row units (`f64`: exact to well
    /// past any u32 row count — the transport every consumer computes in).
    pub(crate) fn rows(self, line_h: f32) -> f64 {
        f64::from(self.row) + f64::from(self.offset_px) / f64::from(line_h)
    }

    /// The canonical anchor at a fractional row position: floor the row,
    /// carry the remainder as sub-row pixels. Clamps negatives to
    /// [`Self::TOP`]; the caller owns the bottom clamp (it needs the row
    /// count and viewport).
    pub(crate) fn from_rows(rows: f64, line_h: f32) -> Self {
        let rows = rows.max(0.0);
        let row = rows.floor().min(f64::from(u32::MAX)) as u32;
        Self { row, offset_px: ((rows - f64::from(row)) * f64::from(line_h)) as f32 }
    }
}

/// One frame's screen geometry: bounds, gutter width, cell metrics, scroll.
///
/// Vertical position flows through the [`ScrollAnchor`] in `f64` row units;
/// only final *screen-space* values (small by construction) come out as
/// `f32`. There is deliberately no absolute content-space `top()` in `f32`:
/// `bounds.y − scroll_px` is exactly the catastrophic cancellation the anchor
/// exists to prevent. (Horizontal stays `f32`: a line would need
/// ~2M cells before x math degrades, and text shaping gives out long before
/// that.)
#[derive(Copy, Clone, Debug)]
pub(crate) struct Geo {
    bounds: Rectangle,
    gutter: f32,
    advance: f32,
    line_h: f32,
    scroll_x: f32,
    /// The anchor's position in f64 row units, derived once per frame.
    scroll_rows: f64,
}

impl Geo {
    /// A frame's projection. Built once at the top of a widget pass; never
    /// stored across frames.
    pub(crate) fn new(bounds: Rectangle, gutter: f32, advance: f32, line_h: f32, scroll_x: f32, scroll: ScrollAnchor) -> Self {
        Self { bounds, gutter, advance, line_h, scroll_x, scroll_rows: scroll.rows(line_h) }
    }

    /// Display cell → screen x: the ONE forward affix
    /// (`bounds.x + gutter + TEXT_PAD + cell·advance − scroll_x`).
    pub(crate) fn cell_x(&self, cell: f32) -> f32 {
        self.bounds.x + self.gutter + TEXT_PAD + cell * self.advance - self.scroll_x
    }

    /// Screen x → (fractional, unrounded) display cell — the exact inverse of
    /// [`Self::cell_x`]. Rounding/snapping policy is the core's, not ours.
    pub(crate) fn x_cell(&self, x: f32) -> f32 {
        (x - self.bounds.x - self.gutter - TEXT_PAD + self.scroll_x) / self.advance
    }

    /// The text area's *unscrolled* left edge (`bounds.x + gutter + TEXT_PAD`)
    /// — for full-row washes (line highlight) that ignore horizontal scroll.
    pub(crate) fn text_left(&self) -> f32 {
        self.bounds.x + self.gutter + TEXT_PAD
    }

    /// Display row → its top screen y: the ONE forward map — the row's
    /// distance from the scroll anchor in row units (exact f64 integer math)
    /// scaled to pixels, so a deep row's position is exact before the one
    /// screen-space rounding to `f32`. Takes a [`DisplayRow`] — which the
    /// widget can only obtain from a `FoldMap` method, never mint from
    /// arithmetic — so "anchor at a buffer row" does not typecheck.
    pub(crate) fn row_y(&self, row: DisplayRow) -> f32 {
        (f64::from(self.bounds.y)
            + (f64::from(row.index()) - self.scroll_rows) * f64::from(self.line_h)) as f32
    }

    /// Screen y → (fractional, unclamped) display rows from the content top —
    /// feed it to [`scrive_core::FoldMap::display_row_at`], which owns the
    /// floor/clamp policy. `f64` end to end: row counts past ~2²³ don't fit
    /// an `f32`'s mantissa, and a half-row error is a wrong hit.
    pub(crate) fn rows_from_top(&self, y: f32) -> f64 {
        self.scroll_rows + f64::from(y - self.bounds.y) / f64::from(self.line_h)
    }

    /// The code area's left edge in screen px (`bounds.x + gutter`) — the
    /// boundary between the gutter and the horizontally-scrolled text area. The
    /// ONE owner of that boundary; [`Self::in_gutter`] is defined against it, so
    /// clicks, the I-beam/finger cursor split, hover arming, and the code clip
    /// can never disagree about where the gutter ends.
    pub(crate) fn code_left(&self) -> f32 {
        self.bounds.x + self.gutter
    }

    /// Whether screen x falls in the gutter (left of the text area).
    pub(crate) fn in_gutter(&self, x: f32) -> bool {
        x < self.code_left()
    }

    /// The widget bounds this frame was built from.
    pub(crate) fn bounds(&self) -> Rectangle {
        self.bounds
    }

    /// The gutter width (px).
    pub(crate) fn gutter(&self) -> f32 {
        self.gutter
    }

    /// One monospace cell's advance (px).
    pub(crate) fn advance(&self) -> f32 {
        self.advance
    }

    /// One display row's height (px).
    pub(crate) fn line_h(&self) -> f32 {
        self.line_h
    }

    /// The rounded `…` pill behind a collapsed chip, from its center x on a
    /// row starting at `row_top` — the ONE pill formula, so the painted pill
    /// and any overlay on it are sized against the same line height.
    pub(crate) fn chip_pill(&self, center_x: f32, row_top: f32) -> Rectangle {
        Rectangle {
            x: center_x - self.advance * 1.2,
            y: row_top + 2.0,
            width: self.advance * 2.4,
            height: self.line_h - 4.0,
        }
    }

    /// The rounded halo box around an inline bracket span `[x0, x1]` (screen
    /// x of opener/closer) on the row at `row_top` — the ONE halo formula, so
    /// every drawn bracket halo has identical padding and rounding.
    pub(crate) fn inline_halo(&self, x0: f32, x1: f32, row_top: f32) -> Rectangle {
        Rectangle {
            x: x0 - 2.0,
            y: row_top + 1.0,
            width: (x1 - x0) + self.advance + 4.0,
            height: self.line_h - 2.0,
        }
    }
}
