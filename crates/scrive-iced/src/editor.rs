//! The editor widget: a direct `iced::advanced::Widget`.
//!
//! It renders a [`scrive_core::Document`] — gutter with line numbers, text,
//! N carets, and selection highlights — and translates raw key/mouse input
//! into semantic [`Action`]s the application applies to its `Document` (the
//! controlled-widget pattern: the widget borrows `&Document` for drawing and
//! never mutates it behind the app's back). Implementing `Widget` directly
//! (not `canvas::Program`) is what lets it participate in iced's focus/operation
//! protocol — only the low-level widget API exposes the `operate()` hook that
//! protocol needs.
//!
//! **Font and size are configurable** ([`Editor::font`], [`Editor::text_size`]);
//! the cell advance is **measured** from the renderer for that font (see
//! [`crate::metrics`]), never hardcoded — so the caret tracks the
//! glyphs for whatever monospace face the app picks, in any renderer.
//!
//! The widget owns its **scroll offset**: the wheel moves it
//! directly, and autoscroll-to-caret is a *deferred* request — set only after a
//! caret-moving action and resolved at layout — so the view never chases the
//! caret while the user is scrolling by hand.

use iced::advanced::clipboard::Kind as ClipboardKind;
use iced::advanced::text::{Alignment, LineHeight, Renderer as _, Shaping, Text, Wrapping};
use iced::advanced::widget::operation::Focusable;
use iced::advanced::widget::{self, Operation, Widget};
use iced::advanced::{layout, mouse, renderer, Clipboard, Layout, Shell};
use iced::alignment::Vertical;
use iced::keyboard::key::Named;
use iced::keyboard::{Event as Keyboard, Key, Modifiers};
use iced::{border, window, Color, Element, Event, Font, Length, Pixels, Point, Rectangle, Shadow, Size, Vector};

use std::cell::Ref;
use std::ops::Range;
use std::time::{Duration, Instant};

use scrive_core::{
    display_map, BufferRow, ColumnDir, CompletionKind, Document, FoldMap, Granularity,
    HighlightSpan, HoverInfo, Motion, Point as BufPoint, PopupList, RevealMode, RowLayout, Severity,
    SignatureInfo, HOVER_IDLE_DELAY_MS,
};

use crate::geo::{Geo, ScrollAnchor, CHIP_PILL_RADIUS, TEXT_PAD};
use crate::popup;

use crate::metrics::Metrics;

/// The two scrollbar-overview lanes filled per frame by
/// [`Document::overview_marks`]: per track-pixel bucket, the diagnostic lane's
/// `(encoded severity, winner offset)` and the find lane's first-match offset.
/// Named so the retained per-thread scratch stays under `clippy::type_complexity`.
type OverviewLanes = (Vec<(u8, u32)>, Vec<Option<u32>>);

/// Default font size in logical pixels.
const DEFAULT_SIZE: f32 = 14.0;
/// Row height as a multiple of the font size (the editor convention).
const LINE_HEIGHT_RATIO: f32 = 1.43;
/// Padding on each side of the gutter's line numbers.
const GUTTER_PAD: f32 = 8.0;
/// Gap between the line numbers and the fold-chevron column — breathing
/// room so the chevron doesn't crowd the numbers.
const FOLD_GAP: f32 = 7.0;
/// Ctrl+hover collapse affordance: the dashed box drawn around a collapsible
/// while Ctrl is held. A warm yellow (the fold-chip / find-match accent) at three
/// intensities — a dim outline on every collapsible in the pointer's nest, a
/// bright outline plus a faint wash on the innermost (the Ctrl+Click target).
const ARM_DASH: Color = Color::from_rgba8(0xfc, 0xe5, 0x66, 0.38);
/// The innermost (Ctrl+Click target) box outline — brighter than the nest.
const ARM_DASH_ACTIVE: Color = Color::from_rgba8(0xfc, 0xe5, 0x66, 0.85);
/// The faint interior wash under the innermost box.
const ARM_FILL: Color = Color::from_rgba8(0xfc, 0xe5, 0x66, 0.07);
/// Dash run, gap, and stroke width (px) of the collapse box.
const ARM_DASH_LEN: f32 = 4.0;
/// Gap between dashes (px).
const ARM_DASH_GAP: f32 = 3.0;
/// Stroke width of the dashed collapse box (px).
const ARM_STROKE: f32 = 1.0;
/// Corner radius (px) of the collapse box — the straight dashed edges are inset by
/// this and the corners are rounded with a short arc.
const ARM_RADIUS: f32 = 4.0;
/// Plain-hover expand affordance: the wash over a collapsed fold's `…`
/// pill while the pointer rests on the chip — quiet hover feedback that a
/// click expands it. A neutral foreground tint (theme-derived) so it reads as
/// button-hover, not selection. Perception-calibrated; provisional until judged
/// on the running app.
const CHIP_HOVER_A: f32 = 0.10;
/// Feature gate for the Ctrl+hover *collapse* affordance — the dashed discovery
/// boxes, the Ctrl+Click-to-collapse gesture, its finger cursor, and the redraws
/// that flash the boxes as Ctrl/the pointer move. Disabled — folding stays on the
/// gutter chevrons and `Ctrl+Shift+[`/`]`. This does **not** touch the plain-hover
/// *expand* affordance (the wash + finger cursor + click on a collapsed `…` chip),
/// which is intentionally always on. Flip to `true` to bring the collapse boxes back.
const SHOW_CTRL_COLLAPSE_AFFORDANCE: bool = false;

/// Tab size for display expansion — *derived* from the core's one owner, never
/// a hand-copied mirror (a change there propagates here at compile time).
const TAB: u32 = display_map::default_tab_size();
/// Rows moved per wheel notch.
const WHEEL_ROWS: f32 = 3.0;
/// Columns scrolled per horizontal wheel-line notch (trackpads send pixels).
const WHEEL_COLS: f32 = 3.0;
/// Caret bar width, in logical pixels. The caret is centered on the insertion
/// point (drawn at `x − CARET_WIDTH/2`), so it straddles the character boundary
/// rather than sitting to its right — the offset is derived from the width, not
/// a calibration constant. (Field-confirmed at 2 px: the caret needed to move
/// exactly `width/2` left.)
const CARET_WIDTH: f32 = 2.0;
/// Caret blink half-period, in milliseconds: the caret shows for
/// one interval and hides for the next while the editor is focused.
const BLINK_MS: u128 = 500;
/// Current-line highlight alpha over the foreground: a faint full-row tint under
/// a single bare cursor. Perception-calibrated; provisional until judged on the
/// running app.
const LINE_HIGHLIGHT_A: f32 = 0.05;
/// Scrollbar overlay width (px), on the right edge, shown only on overflow.
/// Exposed so app chrome (e.g. a floating find bar) can inset itself clear of
/// the scrollbar lane.
pub const SCROLLBAR_WIDTH: f32 = 12.0;
/// Minimum scrollbar thumb height (px), so a huge file still has a grabbable thumb.
const SCROLLBAR_MIN_THUMB: f32 = 24.0;
/// Height (px) of a scrollbar overview marker tick (diagnostics / find matches).
const SCROLLBAR_MARK_H: f32 = 2.0;
/// Scrollbar slider fill — a neutral gray, theme-agnostic like the other chrome.
const SCROLLBAR_THUMB: Color = Color::from_rgba8(0x79, 0x79, 0x79, 0.4);
/// Slider while dragging — brighter than the idle fill.
/// (A distinct hover state needs cursor-over-thumb tracking and is not yet drawn.)
const SCROLLBAR_THUMB_ACTIVE: Color = Color::from_rgba8(0xbf, 0xbf, 0xbf, 0.4);
/// Extra rows below the fold to keep tokenized, so a small scroll doesn't
/// reveal an uncolored line before the next viewport report catches up.
const VIEWPORT_TOKENIZE_MARGIN: u32 = 8;
/// Bracket-pair colorization depth colors, cycled by `depth mod N`. The
/// mainstream-editor default palette — gold / orchid / blue; deeper levels wrap
/// back to gold, so the cycle is exactly 3. A theme palette may override it.
const BRACKET_DEPTH: [Color; 3] = [
    Color::from_rgb8(0xFF, 0xD7, 0x00), // gold
    Color::from_rgb8(0xDA, 0x70, 0xD6), // orchid
    Color::from_rgb8(0x17, 0x9F, 0xFF), // blue
];
/// An unmatched bracket's color — a strong red so a dangling delimiter stands out.
const UNMATCHED_BRACKET: Color = Color::from_rgba8(0xFF, 0x12, 0x12, 0.8);
/// Indentation-guide line width (px).
const GUIDE_WIDTH: f32 = 1.0;
/// Indent-guide alpha over the foreground: idle (dim) and the active guide (the
/// guide at the caret's own indent level, drawn brighter). Perception-calibrated;
/// provisional until judged on the running app.
const GUIDE_IDLE_A: f32 = 0.14;
/// Alpha for the active indent guide.
const GUIDE_ACTIVE_A: f32 = 0.42;
/// Shared floating-popup surface — a neutral dark panel color used by the hover,
/// signature-help box, and completion popup so the three read as one widget
/// family (rendered via [`fill_panel`]).
const POPUP_SURFACE: Color = Color::from_rgb8(0x25, 0x25, 0x26);
/// The popups' hairline edge — `#CCCCCC` at 20%, a neutral widget border.
const POPUP_BORDER: Color = Color::from_rgba8(0xCC, 0xCC, 0xCC, 0.2);
/// Neutral selection fill for the completion popup's focused row — a coloured
/// accent would clash with the kind-coloured labels.
const POPUP_SELECT: Color = Color::from_rgb8(0x37, 0x37, 0x3d);
/// Inline-code pill background in hover markdown — a subtle darker wash behind
/// `` `code` `` spans.
const CODE_PILL_BG: Color = Color::from_rgba8(0x0a, 0x0a, 0x0a, 0.4);
/// Max visible wrapped lines in the hover before it scrolls (rather than
/// truncate).
const HOVER_MAX_VISIBLE: usize = 12;
/// Width (px) of the hover's auto scrollbar — thinner than the editor's.
const HOVER_SB_W: f32 = 6.0;
/// Find-match highlight wash: an amber fill behind every match, brighter
/// for the active one — a warm yellow at two alphas (a theme may override).
const FIND_MATCH: Color = Color::from_rgba8(0xfc, 0xe5, 0x66, 0.22);
/// The active find match's brighter wash.
const FIND_MATCH_ACTIVE: Color = Color::from_rgba8(0xfc, 0xe5, 0x66, 0.48);
/// Rows of context the indent-guide walks may consult beyond the visible
/// window: blank-line level interpolation and the active guide's extent only
/// need NEARBY context for visual continuity, and clamping the walk to this
/// slack keeps a blank-heavy file from scanning to the document edges each frame.
const INDENT_NEIGHBOR_SLACK: u32 = 256;

/// Rows above the visible window the fold-affordance queries (Ctrl+hover boxes)
/// look for a foldable pair's header: a pair headed this far above the viewport
/// still arms; a block taller than this enclosing the pointer is the accepted
/// miss (fold it from its header chevron instead). Bounds the per-mouse-move
/// `collapsible_pairs_in_rows` scan so it stays viewport-proportional.
const FOLD_QUERY_SLACK: u32 = 2048;

/// How many lines of a block the Ctrl+hover box measures for its width: a
/// block can span the whole document, so the width scan is capped here rather
/// than walking every line — the box reads the same, and the per-frame cost
/// stays proportional to the viewport, not the document.
const FOLD_BOX_SCAN_ROWS: u32 = 512;

/// Word-under-caret occurrence wash (perception-calibrated, provisional until
/// judged on the running app): the find wash's neutral, dimmer cousin — a
/// text-color tint visible enough to trace a symbol, quiet enough to read past.
/// Suppressed while find is live (its amber wash owns the screen then).
const OCCURRENCE_MATCH: Color = Color::from_rgba8(0xf7, 0xf1, 0xff, 0.09);
/// Diagnostic squiggle geometry (field-calibrated): the wave's half-period
/// (zero-to-zero), amplitude, and stroke, in px.
const SQUIGGLE_HALF_PERIOD: f32 = 2.0;
/// Squiggle wave amplitude (px).
const SQUIGGLE_AMPLITUDE: f32 = 1.5;
/// Squiggle stroke thickness (px).
const SQUIGGLE_STROKE: f32 = 1.0;

/// The default row height for a given font size.
fn default_line_height(size: f32) -> f32 {
    (size * LINE_HEIGHT_RATIO).round()
}

/// Whether the caret is in the *solid* half of the blink cycle after `elapsed`
/// milliseconds: on for `[0, 500)`, off for `[500, 1000)`, and so on. Pure, so
/// the phase is unit-testable without a clock.
fn blink_on(elapsed_ms: u128) -> bool {
    (elapsed_ms / BLINK_MS).is_multiple_of(2)
}

/// A semantic editor action produced by the widget for the app to apply to its
/// [`Document`] — an edit verb or a caret motion.
#[derive(Clone, Debug, PartialEq)]
pub enum Action {
    /// Insert a typed character (replacing any selection).
    Type(char),
    /// Backspace.
    Backspace,
    /// Forward delete.
    Delete,
    /// Delete the word before the caret (Ctrl+Backspace).
    DeleteWordBack,
    /// Delete the word after the caret (Ctrl+Delete).
    DeleteWordForward,
    /// Split the line (Enter).
    Enter,
    /// Indent / insert-to-tab-stop (Tab).
    Tab,
    /// Outdent to the previous tab stop (Shift+Tab).
    Outdent,
    /// Toggle the language's line comment on the spanned lines (Ctrl+/).
    ToggleComment,
    /// Delete each selection's whole-line block (Ctrl+Shift+K).
    DeleteLine,
    /// Open a fresh indent-carrying line below (Ctrl+Enter) or above
    /// (Ctrl+Shift+Enter) the caret's line, caret landing on it.
    InsertLine {
        /// Below if true, above if false.
        down: bool,
    },
    /// Cut's edit half (the copy half runs in the widget): delete each
    /// selection, an empty caret expanding to its whole line.
    Cut,
    /// Insert clipboard text at the carets (Paste).
    Paste {
        /// The pasted text (the core normalizes CRLF to LF).
        text: String,
        /// Whether this is our own whole-line copy (recognized from a side
        /// table) — pasted above the caret's line, caret staying put.
        entire_line: bool,
    },
    /// Move (or, with `extend`, drag) every caret.
    Move {
        /// Which motion.
        motion: Motion,
        /// Extend the selection instead of collapsing.
        extend: bool,
    },
    /// Place a single caret at a byte offset (mouse click).
    PlaceCaret(u32),
    /// Drag-select from `origin` to `head` at a granularity — the click-drag
    /// family. `head == origin` is the double/triple-click initial selection;
    /// Word/Line drags extend by whole units keeping the origin unit selected.
    DragSelect {
        /// Character / word / line units.
        granularity: Granularity,
        /// The fixed origin (where the gesture began).
        origin: u32,
        /// The moving end (the cursor).
        head: u32,
    },
    /// Add a caret at a byte offset (Alt+Click), keeping existing selections.
    AddCaret(u32),
    /// Add the next occurrence of the selection as a new caret (Ctrl+D).
    AddNextOccurrence,
    /// Select every occurrence of the selection (Ctrl+Shift+L).
    SelectAllOccurrences,
    /// Add a caret one display row above/below every caret (Ctrl+Alt+↑/↓).
    AddCaretVertical {
        /// Below if true, above if false.
        down: bool,
    },
    /// Jump each caret to its matching bracket (Ctrl+Shift+\).
    JumpToBracket,
    /// Grow each selection one structural step (Shift+Alt+Right).
    ExpandSelection,
    /// Walk back down the expansion ladder (Shift+Alt+Left).
    ShrinkSelection,
    /// Collapse a multi-cursor set (or a range) to the primary caret (Escape).
    Collapse,
    /// Select the whole document (Ctrl+A).
    SelectAll,
    /// Undo the last edit (Ctrl+Z).
    Undo,
    /// Redo the last undone edit (Ctrl+Shift+Z / Ctrl+Y).
    Redo,
    /// Grow/shrink the column (box) selection (Ctrl+Shift+Alt+Arrow).
    ColumnSelect(ColumnDir),
    /// Mouse box (column) selection drag (Shift+Alt+drag): corners as
    /// `(visible buffer row, virtual display cell)` — the core resolves the
    /// cells per spanned display row and clamps each to its line.
    ColumnDrag {
        /// The fixed corner `(row, display cell)`.
        anchor: (u32, u32),
        /// The moving corner `(row, display cell)`.
        active: (u32, u32),
    },
    /// Move the selected line-block up/down (Alt+↑/↓).
    MoveLine {
        /// Down if true, up if false.
        down: bool,
    },
    /// Duplicate the selected line-block above/below (Shift+Alt+↑/↓).
    CopyLine {
        /// Below if true, above if false.
        down: bool,
    },
    /// The visible buffer-row range changed (scroll / resize / autoscroll),
    /// margin included. The app aims the highlight retention window here
    /// (`Document::set_highlight_window`) and tokenizes highlights down to its
    /// end — the view reports its viewport to the model so highlighting only
    /// ever runs for what is on screen (see `update`). Not an edit: it never
    /// moves a caret or touches history.
    ViewportChanged(Range<u32>),
    /// Move the completion popup's selection up / down. Published (and the
    /// key captured) only while the popup is open; the app drives the controller.
    PopupUp,
    /// See [`PopupUp`](Action::PopupUp).
    PopupDown,
    /// Accept the popup's selected item (Enter / Tab).
    PopupAccept,
    /// Accept the popup row at this filtered-list index (a mouse click on it).
    PopupClickAccept(u32),
    /// Dismiss the popup (Escape) — sticky until a word boundary.
    PopupDismiss,
    /// Advance / retreat the active snippet tab stop (Tab / Shift+Tab).
    /// Captured only while a session is active; the app drives it.
    SnippetTab,
    /// See [`SnippetTab`](Action::SnippetTab).
    SnippetTabPrev,
    /// Cancel the snippet session (Escape).
    SnippetCancel,
    /// Close the signature-help box (Escape — after the popup, before the
    /// snippet session, in dismissal order).
    SignatureClose,
    /// Jump to the next/previous diagnostic (F8 / Shift+F8).
    NextDiagnostic {
        /// Forward if true (F8), backward if false (Shift+F8).
        forward: bool,
    },
    /// The pointer rested over byte `offset` long enough for a hover query.
    /// The app resolves the word + queries its provider.
    HoverQuery(u32),
    /// Close the hover popup (pointer left the word).
    HoverDismiss,
    /// Toggle the fold whose pair opens at byte offset `opener` — a
    /// gutter-chevron click or `Ctrl+Shift+[`/`]`. The app calls
    /// `Document::toggle_fold_opener`; not an edit (folds are view state, never on
    /// the undo stack). Keyed by the opener so a single-line row with two `[..]`
    /// pairs stays unambiguous.
    ToggleFold {
        /// Byte offset of the fold's opening bracket.
        opener: u32,
    },
    /// Fold (`unfold == false`) or unfold (`unfold == true`) the block at EVERY
    /// caret — `Ctrl+Shift+[` / `Ctrl+Shift+]`. The app calls
    /// `Document::fold_at_carets`; not an edit (folds are view state). Distinct
    /// from [`Action::ToggleFold`], which acts on one gutter-clicked opener.
    FoldAtCarets {
        /// `true` = unfold (`Ctrl+Shift+]`), `false` = fold (`Ctrl+Shift+[`).
        unfold: bool,
    },
}

impl Action {
    /// Whether applying this action can move the primary caret to a possibly
    /// off-screen spot (and so should trigger autoscroll). Clicks (`PlaceCaret`,
    /// `AddCaret`) already land in view, and `Collapse` keeps an existing caret.
    /// The find / multi-cursor jump verbs (`AddNextOccurrence`, `NextDiagnostic`,
    /// …) are excluded: they reveal through the core's `request_reveal`,
    /// which bumps the reveal generation ONLY when something changed — so the
    /// reveal is the single source, and a no-op Ctrl+D never yanks the viewport
    /// the way a spurious `moves_caret` autoscroll would.
    fn moves_caret(&self) -> bool {
        // `DragSelect` autoscrolls: for the double/triple-click case the head is
        // in view (reveal holds), but a real drag past the edge must scroll.
        // `SelectAll` keeps the view (Ctrl+A shouldn't jump to end-of-document).
        !matches!(
            self,
            Action::PlaceCaret(_)
                | Action::AddCaret(_)
                | Action::Collapse
                | Action::SelectAll
                | Action::ViewportChanged(_)
                | Action::PopupUp
                | Action::PopupDown
                | Action::PopupAccept
                | Action::PopupClickAccept(_)
                | Action::PopupDismiss
                | Action::SnippetTab
                | Action::SnippetTabPrev
                | Action::SnippetCancel
                | Action::SignatureClose
                | Action::HoverQuery(_)
                | Action::HoverDismiss
                | Action::ToggleFold { .. }
                | Action::FoldAtCarets { .. }
                // These verbs reveal through the core's request_reveal —
                // bumped only when something actually changed, so a no-op F8 /
                // bracket jump / edge add-caret / Ctrl+D never yanks the viewport
                // back. Ctrl+D's real add still reveals (FitForce) via that bump.
                | Action::NextDiagnostic { .. }
                | Action::JumpToBracket
                | Action::ExpandSelection
                | Action::ShrinkSelection
                | Action::AddCaretVertical { .. }
                | Action::SelectAllOccurrences
                | Action::AddNextOccurrence
        )
    }
}

/// Keyboard focus plus the caret-blink clock (mirrors iced `text_input`'s
/// `Focus`). Absent ⇒ unfocused (no caret).
#[derive(Debug, Clone, Copy)]
struct Focus {
    /// When the caret last became solid — set on focus and reset on caret
    /// activity, so the caret is always solid right after the user acts.
    updated_at: Instant,
    /// The most recent frame time, advanced on every `RedrawRequested`. The
    /// blink phase is computed from `now − updated_at` using only timestamps
    /// iced hands us (never `Instant::now()` inside `draw`), so a frame paints
    /// the same caret regardless of when `draw` runs.
    now: Instant,
}

/// An in-progress left-button drag: the unit it extends by and the origin the
/// selection stays anchored to.
#[derive(Copy, Clone, Debug)]
struct Drag {
    granularity: Granularity,
    origin: u32,
}

/// Per-widget state held in the iced widget tree.
#[derive(Debug)]
struct State {
    /// Keyboard focus + blink clock, via `operation::focusable`. Starts focused
    /// so the single-editor scratch window (and headless capture) type/paint
    /// immediately; focus *changes* go through the operation protocol.
    focus: Option<Focus>,
    /// Vertical scroll position (the widget owns this), as a display-row
    /// [`ScrollAnchor`] — a scroll position in line units rather than a flat
    /// pixel `f32`. A flat `f32` offset quantizes into visible multi-pixel row
    /// steps once content passes ~2²⁴ px (a many-megabyte document); the anchor
    /// (an integer row index plus a bounded sub-row offset) keeps position math
    /// exact at any document size. Consumers compute in `f64` row units via
    /// [`ScrollAnchor::rows`].
    scroll: ScrollAnchor,
    /// Horizontal scroll offset in pixels. Long lines don't wrap, so
    /// the code area scrolls left by this much; the gutter stays fixed.
    scroll_x: f32,
    /// A pending autoscroll-to-caret, resolved at the next layout.
    autoscroll: bool,
    /// Latest keyboard modifiers, tracked from `ModifiersChanged` — a mouse
    /// event carries none, so Alt+Click reads this (as iced `text_input` does).
    modifiers: Modifiers,
    /// The active left-button drag (granularity + origin), while in progress;
    /// `None` otherwise. Set on press, extended on move, cleared on release.
    drag: Option<Drag>,
    /// The anchor `(row, cell_column)` of an in-progress Shift+Alt box (column)
    /// drag; `None` otherwise. Takes precedence over `drag` while set.
    column_drag_anchor: Option<(u32, u32)>,
    /// While the scrollbar thumb is being dragged: the grab offset within the
    /// thumb (`cursor.y - thumb_y` at press), so the thumb tracks the cursor
    /// from where it was grabbed. `None` when not dragging the scrollbar.
    scrollbar_grab: Option<f32>,
    /// The horizontal analog of `scrollbar_grab`: the grab offset within the
    /// bottom scrollbar thumb while dragging it. `None` otherwise.
    hscrollbar_grab: Option<f32>,
    /// The last visible row range reported to the app (the tokenize target +
    /// highlight retention window), to dedupe `Action::ViewportChanged` so it
    /// fires only when the visible range actually moves. `MAX..MAX` until the
    /// first report.
    last_reported_viewport: Range<u32>,
    /// The previous left-button click, for double/triple-click detection (iced
    /// `mouse::Click` — consecutive within 6 px and 300 ms).
    last_click: Option<mouse::Click>,
    /// Cached measured metrics and the font they were measured for; re-measured
    /// when the configured font or size changes.
    metrics: Metrics,
    measured_font: Option<Font>,
    /// The last `Document::reveal_seq` this view acted on — a bump means a
    /// jump-class action (find navigation) moved the caret outside the input
    /// path, so the next layout should autoscroll to it.
    last_reveal_seq: u64,
    /// Hover idle timer: the last pointer position over the text, whether a
    /// re-arm is pending (set on move, resolved to `hover_at` on the next frame
    /// so it stamps an accurate `now`), and the instant a hover query should
    /// fire. Cleared/reset on every move so a moving pointer never queries.
    hover_pos: Option<Point>,
    hover_rearm: bool,
    hover_at: Option<Instant>,
    /// Vertical scroll offset (px) of the open hover popup's content, for hovers
    /// taller than `HOVER_MAX_VISIBLE`. Reset to 0 when the hovered word changes.
    hover_scroll: f32,
    /// Whether the pointer is over the gutter: the expanded (chevron-down) fold
    /// controls show only while it is, so an unfolded block reveals its control
    /// on hover — as mainstream editors do. Tracked so a move on/off the gutter
    /// triggers exactly one redraw.
    gutter_hover: bool,
    /// The buffer row under the pointer while it is over the gutter — drives
    /// the hovered fold chevron's brightening (disambiguating adjacent
    /// chevrons). Tracked so a row-to-row move inside the gutter repaints
    /// exactly once; `None` off the gutter.
    gutter_hover_row: Option<u32>,
    /// The collapsed fold (its opening-bracket offset) whose `…` chip the pointer
    /// is resting on — drives the widget-drawn hover preview of its hidden
    /// content. Set when the hover idle timer fires over a chip; cleared on any move.
    fold_preview: Option<u32>,
    /// The collapsed fold whose chip the (non-Ctrl) pointer is currently over —
    /// drives the immediate plain-hover expand highlight. Tracked so a move
    /// on/off a chip repaints exactly once; `None` when off any chip or Ctrl is held.
    hover_chip: Option<u32>,
}

impl Default for State {
    fn default() -> Self {
        let now = Instant::now();
        Self {
            focus: Some(Focus { updated_at: now, now }),
            scroll: ScrollAnchor::TOP,
            scroll_x: 0.0,
            autoscroll: false,
            modifiers: Modifiers::default(),
            drag: None,
            column_drag_anchor: None,
            scrollbar_grab: None,
            hscrollbar_grab: None,
            last_reported_viewport: u32::MAX..u32::MAX,
            last_click: None,
            metrics: Metrics { advance: DEFAULT_SIZE * 0.6, line_height: default_line_height(DEFAULT_SIZE), size: DEFAULT_SIZE },
            measured_font: None,
            last_reveal_seq: 0,
            hover_pos: None,
            hover_rearm: false,
            hover_at: None,
            hover_scroll: 0.0,
            gutter_hover: false,
            gutter_hover_row: None,
            fold_preview: None,
            hover_chip: None,
        }
    }
}

impl State {
    /// Whether the editor holds keyboard focus.
    fn is_focused(&self) -> bool {
        self.focus.is_some()
    }

    /// Take focus (or refocus), restarting the blink at solid-on.
    fn focus(&mut self) {
        let now = Instant::now();
        self.focus = Some(Focus { updated_at: now, now });
    }

    /// Drop focus; the caret stops being painted.
    fn unfocus(&mut self) {
        self.focus = None;
    }

    /// Restart the blink at solid-on for caret activity (typing/moving), so the
    /// caret is visible immediately after the user acts. No-op if unfocused.
    fn ping(&mut self) {
        if let Some(focus) = &mut self.focus {
            let now = Instant::now();
            focus.updated_at = now;
            focus.now = now;
        }
    }

    /// Whether to paint the caret this frame: focused *and* in the solid half of
    /// the blink cycle.
    fn caret_on(&self) -> bool {
        self.focus
            .is_some_and(|f| blink_on(f.now.saturating_duration_since(f.updated_at).as_millis()))
    }
}

impl Focusable for State {
    fn is_focused(&self) -> bool {
        State::is_focused(self)
    }

    fn focus(&mut self) {
        State::focus(self);
    }

    fn unfocus(&mut self) {
        State::unfocus(self);
    }
}

/// The editor widget. Borrows the document to draw; publishes [`Action`]s.
#[allow(missing_debug_implementations)]
pub struct Editor<'a, Message> {
    id: Option<widget::Id>,
    doc: &'a Document,
    on_action: Box<dyn Fn(Action) -> Message + 'a>,
    /// The open completion popup to render over the editor, if any (app-supplied
    /// from its completion controller). `None` = no popup.
    popup: Option<&'a PopupList>,
    /// Whether a snippet tab-stop session is active — then the editor
    /// captures Tab / Shift+Tab / Escape to drive it (the app owns the session).
    snippet_active: bool,
    /// The active signature-help box, if any (app-supplied). Rendered above
    /// the caret; Escape closes it (after the popup, before the snippet session).
    signature: Option<&'a SignatureInfo>,
    /// The open hover popup, if any (app-supplied) — a markdown box anchored
    /// at the hovered word.
    hover: Option<&'a HoverInfo>,
    font: Font,
    size: f32,
    line_height: f32,
}

impl<'a, Message> Editor<'a, Message> {
    /// An editor rendering `doc`; `on_action(action)` is published for each
    /// input the widget interprets. Defaults to [`Font::MONOSPACE`] at 14 px.
    pub fn new(doc: &'a Document, on_action: impl Fn(Action) -> Message + 'a) -> Self {
        Self {
            id: None,
            doc,
            on_action: Box::new(on_action),
            popup: None,
            snippet_active: false,
            signature: None,
            hover: None,
            font: Font::MONOSPACE,
            size: DEFAULT_SIZE,
            line_height: default_line_height(DEFAULT_SIZE),
        }
    }

    /// Supply the open completion popup to render over the editor. The app
    /// holds the completion controller and passes its `PopupList` here each
    /// frame; `None` when no popup is open.
    #[must_use]
    pub fn popup(mut self, popup: Option<&'a PopupList>) -> Self {
        self.popup = popup;
        self
    }

    /// Tell the editor a snippet session is active, so it captures
    /// Tab / Shift+Tab / Escape to drive it (the app owns the session).
    #[must_use]
    pub fn snippet_active(mut self, active: bool) -> Self {
        self.snippet_active = active;
        self
    }

    /// Supply the active signature-help box to render above the caret.
    #[must_use]
    pub fn signature(mut self, signature: Option<&'a SignatureInfo>) -> Self {
        self.signature = signature;
        self
    }

    /// Supply the open hover popup to render at the hovered word.
    #[must_use]
    pub fn hover(mut self, hover: Option<&'a HoverInfo>) -> Self {
        self.hover = hover;
        self
    }

    /// Give the editor a widget [`Id`](widget::Id) so the application can move
    /// keyboard focus to it through iced's focus operations (`focus`,
    /// `focus_next`, …) — the addressing half of the focus protocol a multi-pane
    /// host needs. Without an id the editor can be unfocused by a focus
    /// elsewhere but never *targeted* by one (a `None` id never matches).
    #[must_use]
    pub fn id(mut self, id: impl Into<widget::Id>) -> Self {
        self.id = Some(id.into());
        self
    }

    /// Set the (monospace) font. The cell advance is re-measured for it.
    #[must_use]
    pub fn font(mut self, font: Font) -> Self {
        self.font = font;
        self
    }

    /// Set the font size in logical pixels; the row height follows unless later
    /// overridden with [`Editor::line_height`].
    #[must_use]
    pub fn text_size(mut self, size: f32) -> Self {
        self.size = size;
        self.line_height = default_line_height(size);
        self
    }

    /// Override the row height in logical pixels.
    #[must_use]
    pub fn line_height(mut self, line_height: f32) -> Self {
        self.line_height = line_height;
        self
    }

    /// Re-measure metrics into `state` if the configured font/size changed.
    fn ensure_metrics(&self, state: &mut State) {
        let stale = state.measured_font != Some(self.font)
            || state.metrics.size != self.size
            || state.metrics.line_height != self.line_height;
        if stale {
            state.metrics = Metrics::measure(self.font, self.size, self.line_height);
            state.measured_font = Some(self.font);
        }
    }

    /// Width reserved for the gutter: left pad, the line-number digits, a gap, a
    /// one-cell fold-chevron column, and right pad.
    ///
    /// Layout (offsets from the widget's left edge):
    /// `| GUTTER_PAD | digits | FOLD_GAP | chevron(advance) | GUTTER_PAD |`
    fn gutter_width(&self, advance: f32) -> f32 {
        let digits = digit_count(self.doc.buffer().line_count()).max(2);
        2.0 * GUTTER_PAD + digits as f32 * advance + FOLD_GAP + advance
    }

    /// The x-offset (from the widget's left) where line numbers right-align — the
    /// right edge of the digit column, before the fold gap.
    fn gutter_number_right(&self, advance: f32) -> f32 {
        let digits = digit_count(self.doc.buffer().line_count()).max(2);
        GUTTER_PAD + digits as f32 * advance
    }

    /// The x-offset where the fold chevron is left-drawn — one `FOLD_GAP` past the
    /// numbers, so it sits in its own padded column just before the code.
    fn gutter_chevron_x(&self, advance: f32) -> f32 {
        self.gutter_number_right(advance) + FOLD_GAP
    }

    /// The screen x of buffer `offset` — the one fold-aware horizontal
    /// projection. Tab expansion, inline-fold collapse, chip-center clipping,
    /// and collapsed-tail following all come from the core's one owner
    /// ([`FoldMap::display_position`]); the pixel affix is [`Geo::cell_x`]'s.
    /// An offset genuinely hidden inside a fold (callers never pass one —
    /// carets, selection endpoints on visible rows, and popup anchors are
    /// visible by construction) falls back to the text origin.
    fn offset_screen_x(&self, fold_map: &FoldMap, geo: &Geo, offset: u32) -> f32 {
        let cells = fold_map
            .display_position(self.doc.buffer(), offset, TAB)
            .map_or(0.0, |p| p.x.cells());
        geo.cell_x(cells)
    }

    /// A floating panel's display-space anchor at buffer `offset`:
    /// `(x, row_top, row_bottom)` — the one place the completion / hover /
    /// signature popups (and any future panel) get their anchor. The row is
    /// display-space by construction, so a panel anchors at the display row and
    /// cannot sit too low below a fold; each panel keeps its own flip/clamp
    /// choice on top.
    fn popup_anchor(&self, fold_map: &FoldMap, geo: &Geo, offset: u32) -> (f32, f32, f32) {
        let top = match fold_map.display_position(self.doc.buffer(), offset, TAB) {
            Some(p) => geo.row_y(p.row),
            // Hidden offset (callers don't pass one): clip to its fold's header row.
            None => {
                let row = self.doc.buffer().offset_to_point(offset).row;
                geo.row_y(fold_map.to_display_row(BufferRow(row)))
            }
        };
        (self.offset_screen_x(fold_map, geo, offset), top, top + geo.line_h())
    }

    /// The document's memoized fold-aware row converter — O(1) unless the
    /// folds or buffer changed since the last build (see [`Document::fold_map`]),
    /// so the render path's many per-frame calls don't each rebuild it.
    fn fold_map(&self) -> Ref<'_, FoldMap> {
        self.doc.fold_map()
    }

    /// Whether any selection's head currently renders inside the vertical
    /// viewport `[top_rows, top_rows + viewport_rows)` (display rows) — the
    /// "hold the viewport on a Fit reveal if a cursor is already visible" test.
    /// Bounded to the on-screen selections via the sorted-set window, so a
    /// document-scale multi-cursor set costs O(visible + log carets), not
    /// O(carets), on the reveal path.
    fn any_caret_on_screen(
        &self,
        buffer: &scrive_core::Buffer,
        fold_map: &FoldMap,
        top_rows: f64,
        viewport_rows: f64,
    ) -> bool {
        let first_buf = fold_map.to_buffer_row(fold_map.display_row_at(top_rows)).0;
        let last_buf = fold_map.to_buffer_row(fold_map.display_row_at(top_rows + viewport_rows)).0;
        let vis_start = buffer.point_to_offset(BufPoint { row: first_buf, col: 0 });
        let vis_end = buffer.point_to_offset(BufPoint { row: last_buf, col: buffer.line_len(last_buf) });
        let sels = self.doc.selections().all();
        // The partially-scrolled top row still counts as visible (floor it).
        let band = top_rows.floor()..(top_rows + viewport_rows);
        sels[visible_selection_span(sels, vis_start, vis_end)].iter().any(|s| {
            fold_map
                .display_position(buffer, s.head(), TAB)
                .is_some_and(|p| band.contains(&f64::from(p.row.index())))
        })
    }

    /// This pass's pixel projection — built once per
    /// `draw`/`update`/`mouse_interaction` from live state, like [`Self::fold_map`];
    /// never stored.
    fn geo(&self, state: &State, bounds: Rectangle) -> Geo {
        let Metrics { advance, line_height, .. } = state.metrics;
        Geo::new(bounds, self.gutter_width(advance), advance, line_height, state.scroll_x, state.scroll)
    }

    /// The screen rectangle of one *unfolded* collapsible's Ctrl+hover box — the
    /// single source all three surfaces (render, cursor, Ctrl+Click) measure against,
    /// so what you see boxed is exactly what the finger and click act on. A block
    /// bounds its `header … last` extent snug to its own indent and widest line; an
    /// inline pair bounds its bracket span. `None` for an all-blank block or a pair
    /// whose header is itself hidden inside an outer collapsed fold. (Collapsed folds
    /// use [`collapsed_chip_rect`](Self::collapsed_chip_rect) instead.)
    fn collapsible_box_rect(&self, fold_map: &FoldMap, geo: &Geo, open: u32, close: u32, header: u32, last: u32) -> Option<Rectangle> {
        let buffer = self.doc.buffer();
        let code_left = geo.code_left();
        if fold_map.is_folded(BufferRow(header)) {
            return None; // the pair's line is hidden inside an outer collapsed fold
        }
        let y = geo.row_y(fold_map.to_display_row(BufferRow(header)));
        if last > header {
            // Block: the whole `header … last` region, snug to its indent / widest
            // line — the RENDERED width (fold-aware), not the raw one, so a
            // collapsed inline fold inside the block doesn't inflate the box.
            // The width scan is bounded (a block can span the whole document): a
            // hover box sized by its first `FOLD_BOX_SCAN_ROWS` lines reads the
            // same, and the scan stays proportional to the viewport, not the
            // document.
            let (mut lo, mut hi) = (u32::MAX, 0u32);
            for r in header..=last.min(header.saturating_add(FOLD_BOX_SCAN_ROWS)) {
                let l = buffer.line(r);
                if l.trim().is_empty() {
                    continue;
                }
                lo = lo.min(line_indent_cells(&l));
                hi = hi.max(fold_map.row_layout(buffer, BufferRow(r), TAB).width());
            }
            if lo == u32::MAX {
                return None;
            }
            let x0 = (geo.cell_x(lo as f32) - 3.0).max(code_left + 1.0);
            let x1 = geo.cell_x(hi as f32) + 3.0;
            let yl = geo.row_y(fold_map.to_display_row(BufferRow(last)));
            Some(Rectangle { x: x0, y: y - 1.0, width: (x1 - x0).max(geo.advance()), height: (yl + geo.line_h()) - y + 1.0 })
        } else {
            // Inline: the bracket span on the one row (shared cell projection).
            let xo = self.offset_screen_x(fold_map, geo, open);
            let xc = self.offset_screen_x(fold_map, geo, close);
            Some(geo.inline_halo(xo.min(xc), xo.max(xc), y))
        }
    }

    /// Every *un*folded collapsible whose Ctrl+hover box actually contains pixel
    /// `pos`, as `(open, close, rect)` — the Ctrl gesture collapses, so only
    /// things you can still collapse arm (a collapsed fold expands on a plain click
    /// instead, [`collapsed_chip_at`](Self::collapsed_chip_at)). Keying on pixel
    /// containment — not byte range — keeps the finger and click confined to the
    /// boxes you can see. The caller picks the innermost by `close - open`.
    fn armed_boxes(&self, geo: &Geo, pos: Point) -> Vec<(u32, u32, Rectangle)> {
        let fold_map = self.fold_map();
        let display = fold_map.display_row_at(geo.rows_from_top(pos.y));
        let pr = fold_map.to_buffer_row(display).0;
        // Windowed: a pair spanning `pr` is headed at a row <= pr; query only
        // headers within FOLD_QUERY_SLACK above pr (not a whole-document
        // `collapsible_pairs()` scan per Ctrl+hover mouse-move). A block taller
        // than the slack enclosing `pr` is the accepted miss.
        self.doc
            .collapsible_pairs_in_rows(pr.saturating_sub(FOLD_QUERY_SLACK)..pr + 1)
            .into_iter()
            .filter(|&(_, _, header, last)| pr >= header && pr <= last) // cheap row pre-filter
            .filter_map(|(open, close, header, last)| {
                let rect = self.collapsible_box_rect(&fold_map, geo, open, close, header, last)?;
                rect.contains(pos).then_some((open, close, rect))
            })
            .collect()
    }


    /// The largest valid scroll position, in f64 display-row units: total
    /// display rows (folded interiors removed) minus the rows the viewport
    /// shows. Row space, not pixels — the [`ScrollAnchor`] model's one clamp.
    fn max_scroll_rows(&self, viewport_h: f32, line_height: f32) -> f64 {
        (f64::from(self.fold_map().display_row_count())
            - f64::from(viewport_h) / f64::from(line_height))
        .max(0.0)
    }

    /// The widest **visible** display row's pixel width — the horizontal
    /// analog of the total row count, fold- and tab-exact through the fold-map
    /// owners (a collapsed inline fold shrinks its row; a collapsed block header
    /// spans its full `head … tail` placeholder). Viewport-max (**field-gated**):
    /// horizontal scroll only ever needs to reach content on rows you can see, so
    /// the h-scroll range adapts as you scroll vertically — mainstream editors
    /// behave the same — and no per-frame pass walks the whole document (a scan of
    /// only the visible rows, never every line, per frame / layout / wheel tick).
    /// If the adaptive feel fails the field gate, the fallback is a document-owned
    /// line-width index (exact global max, 4 B/line).
    fn max_line_px(&self, advance: f32, line_h: f32, bounds: Rectangle, scroll_rows: f64) -> f32 {
        let buffer = self.doc.buffer();
        let fold_map = self.fold_map();
        let window = fold_map
            .display_window(scroll_rows, scroll_rows + f64::from(bounds.height) / f64::from(line_h));
        fold_map
            .visible_rows(window)
            .map(|vr| match fold_map.header_layout(buffer, vr.buffer_row, TAB) {
                Some(hl) => hl.width(),
                None => fold_map.row_layout(buffer, vr.buffer_row, TAB).width(),
            })
            .max()
            .unwrap_or(0) as f32
            * advance
    }

    /// The code-area width (between the gutter and the vertical scrollbar) for a
    /// given viewport — the horizontal viewport that `scroll_x` scrolls within.
    fn code_area_width(&self, bounds: Rectangle, advance: f32, line_h: f32, scroll_rows: f64) -> f32 {
        let vsb = if self.scrollbar(bounds, line_h, scroll_rows).is_some() { SCROLLBAR_WIDTH } else { 0.0 };
        (bounds.width - self.gutter_width(advance) - TEXT_PAD - vsb).max(0.0)
    }

    /// Max horizontal scroll: how far the widest line overhangs the code area (a
    /// trailing `TEXT_PAD` so the last glyph isn't flush against the edge).
    fn max_scroll_x(&self, bounds: Rectangle, advance: f32, line_h: f32, scroll_rows: f64) -> f32 {
        (self.max_line_px(advance, line_h, bounds, scroll_rows) + TEXT_PAD
            - self.code_area_width(bounds, advance, line_h, scroll_rows))
        .max(0.0)
    }

    /// The last *buffer* row at least partially visible, plus a small margin,
    /// clamped to the buffer — the tokenize target reported to the app. The
    /// visible extent is measured in display rows (fold-aware), then mapped back
    /// to a buffer row.
    fn last_visible_row(&self, bounds: Rectangle, scroll_rows: f64, line_height: f32) -> u32 {
        let fold_map = self.fold_map();
        let rows = (scroll_rows + f64::from(bounds.height) / f64::from(line_height)).ceil()
            + f64::from(VIEWPORT_TOKENIZE_MARGIN);
        fold_map.to_buffer_row(fold_map.display_row_at(rows)).0
    }

    /// Right-edge scrollbar geometry for the current scroll, or `None` when
    /// the content fits (no overflow → no scrollbar). Shared by `draw` (thumb)
    /// and `update` (hit-test + drag) so both agree on where the thumb is.
    /// Row-space throughout (the [`ScrollAnchor`] model): the thumb is a
    /// ratio of rows, never an absolute pixel accumulation.
    fn scrollbar(&self, bounds: Rectangle, line_h: f32, scroll_rows: f64) -> Option<Scrollbar> {
        let total_rows = f64::from(self.fold_map().display_row_count());
        let viewport_rows = f64::from(bounds.height) / f64::from(line_h);
        let max_scroll_rows = total_rows - viewport_rows;
        if max_scroll_rows <= 0.0 {
            return None;
        }
        // The min-thumb clamp can exceed the track when the viewport is under
        // SCROLLBAR_MIN_THUMB tall; cap at the track height so `travel` (used by
        // both the thumb_y map and its inverse) never goes negative.
        // Ratios in f64 (row counts are huge); px results are small.
        let thumb_h = ((viewport_rows / total_rows) as f32 * bounds.height)
            .max(SCROLLBAR_MIN_THUMB)
            .min(bounds.height);
        let thumb_y =
            bounds.y + (scroll_rows / max_scroll_rows) as f32 * (bounds.height - thumb_h);
        Some(Scrollbar {
            x: bounds.x + bounds.width - SCROLLBAR_WIDTH,
            track_top: bounds.y,
            track_h: bounds.height,
            thumb_y,
            thumb_h,
            max_scroll_rows,
        })
    }

    /// Bottom-edge horizontal scrollbar geometry, or `None` when lines fit. The
    /// track spans the code area (gutter edge → vertical-scrollbar edge), so the
    /// two bars meet at the corner without overlapping. The mirror of
    /// `scrollbar()` on the X axis.
    fn hscrollbar(&self, bounds: Rectangle, advance: f32, line_h: f32, scroll_x: f32, scroll_rows: f64) -> Option<HScrollbar> {
        let max_scroll = self.max_scroll_x(bounds, advance, line_h, scroll_rows);
        if max_scroll <= 0.0 {
            return None;
        }
        let track_left = bounds.x + self.gutter_width(advance);
        let content_right = code_area_right(bounds, self.scrollbar(bounds, line_h, scroll_rows).as_ref());
        let track_w = (content_right - track_left).max(0.0);
        // thumb / track = viewport / content; clamp to a grabbable min, capped at
        // the track so `travel` never goes negative.
        let code_area_w = self.code_area_width(bounds, advance, line_h, scroll_rows);
        let content_w = code_area_w + max_scroll;
        let thumb_w = (track_w * code_area_w / content_w).max(SCROLLBAR_MIN_THUMB).min(track_w);
        let thumb_x = track_left + (scroll_x / max_scroll) * (track_w - thumb_w);
        Some(HScrollbar {
            y: bounds.y + bounds.height - SCROLLBAR_WIDTH,
            track_left,
            track_w,
            thumb_x,
            thumb_w,
            max_scroll,
        })
    }
}

/// The scrollbar thumb's placement within its track. The placement fields
/// (`x`, `track_top`, `track_h`, `thumb_y`, `thumb_h`) are pixels in the
/// widget's coordinate space; `max_scroll_rows` is the scroll range in f64
/// display-row units (the [`ScrollAnchor`] currency). The inverse map
/// (thumb top → scroll rows, re-anchored by the caller via
/// [`ScrollAnchor::from_rows`]) lives here so `update`'s drag stays in sync
/// with `draw`'s thumb.
struct Scrollbar {
    /// Left edge of the track band (thumb + markers share this x).
    x: f32,
    track_top: f32,
    track_h: f32,
    thumb_y: f32,
    thumb_h: f32,
    /// The scroll range in f64 display-row units (the [`ScrollAnchor`]
    /// currency) — a huge document's range doesn't fit `f32` exactly, and the
    /// drag inverse multiplies by it.
    max_scroll_rows: f64,
}

impl Scrollbar {
    /// Whether `x` falls in the scrollbar's horizontal band.
    fn contains_x(&self, x: f32) -> bool {
        x >= self.x
    }

    /// Whether `y` lands on the thumb (vs. the bare track).
    fn thumb_contains_y(&self, y: f32) -> bool {
        y >= self.thumb_y && y < self.thumb_y + self.thumb_h
    }

    /// The scroll position (f64 row units) that puts the thumb's top at
    /// `top`, clamped to range — the inverse of `scrollbar()`'s `thumb_y`
    /// map. The caller re-anchors via [`ScrollAnchor::from_rows`].
    fn scroll_rows_for_thumb_top(&self, top: f32) -> f64 {
        let travel = self.track_h - self.thumb_h;
        if travel <= 0.0 {
            return 0.0;
        }
        (f64::from((top - self.track_top) / travel) * self.max_scroll_rows)
            .clamp(0.0, self.max_scroll_rows)
    }
}

/// Bottom-edge horizontal scrollbar thumb placement — the X-axis mirror of
/// [`Scrollbar`], with the same inverse-map discipline so drag and draw agree.
struct HScrollbar {
    /// Top edge of the bottom band.
    y: f32,
    track_left: f32,
    track_w: f32,
    thumb_x: f32,
    thumb_w: f32,
    max_scroll: f32,
}

impl HScrollbar {
    /// Whether `y` falls in the bottom scrollbar band.
    fn contains_y(&self, y: f32) -> bool {
        y >= self.y
    }

    /// Whether `x` lands on the thumb (vs. the bare track).
    fn thumb_contains_x(&self, x: f32) -> bool {
        x >= self.thumb_x && x < self.thumb_x + self.thumb_w
    }

    /// The `scroll_x` that puts the thumb's left at `left`, clamped to range —
    /// the inverse of `hscrollbar()`'s `thumb_x` map.
    fn scroll_for_thumb_left(&self, left: f32) -> f32 {
        let travel = self.track_w - self.thumb_w;
        if travel <= 0.0 {
            return 0.0;
        }
        ((left - self.track_left) / travel * self.max_scroll).clamp(0.0, self.max_scroll)
    }
}

/// Number of decimal digits in `n` (min 1).
fn digit_count(n: u32) -> u32 {
    let mut n = n.max(1);
    let mut d = 0;
    while n > 0 {
        d += 1;
        n /= 10;
    }
    d
}

/// Default `Fit` autoscroll margin, in lines: every edit/caret-move reveal keeps
/// the caret this many lines clear of the viewport edges — the comfortable
/// margin mainstream editors keep so the caret is never flush against an edge.
#[must_use]
pub fn default_autoscroll_margin() -> u32 {
    3
}

/// The `Fit` reveal: the scroll offset that brings the target band
/// `[target_top, target_bottom)` plus a `margin` on each side inside a
/// `viewport`-tall/wide window, moving minimally. The margin collapses when the
/// banded target nearly fills the window; if both banded edges are outside
/// (target taller than the window) it does nothing — never jitter. A
/// visible-with-margin target never scrolls, so an edit never moves the user's
/// scrollbar unnecessarily.
fn reveal_fit(scroll: f64, target_top: f64, target_bottom: f64, viewport: f64, margin: f64) -> f64 {
    let m = margin.min(((viewport - (target_bottom - target_top)) / 2.0).max(0.0));
    let top = (target_top - m).max(0.0);
    let bottom = target_bottom + m;
    match (top < scroll, bottom > scroll + viewport) {
        (true, false) => top,
        (false, true) => bottom - viewport,
        _ => scroll, // inside the band, or taller than the viewport
    }
}

/// The code area's right edge in screen px: the vertical scrollbar's left edge,
/// or the widget's right edge when lines fit and there is no scrollbar. The ONE
/// owner — the `draw` content clip and the h-scrollbar track must end at the
/// same x, or the bottom bar and the text would disagree on where the code area
/// stops.
fn code_area_right(bounds: Rectangle, vscrollbar: Option<&Scrollbar>) -> f32 {
    vscrollbar.map_or(bounds.x + bounds.width, |s| s.x)
}

/// The `[lo, hi)` slice of a sorted, disjoint selection set whose selections
/// intersect the visible byte range `[vis_start, vis_end]` — the window every
/// per-frame selection wash and caret pass iterates, so a document-scale
/// multi-cursor set costs O(visible) per frame, not O(carets). Selections are
/// sorted by start and disjoint, so their ends are ascending too; both bounds
/// are binary searches. A selection straddling the window (e.g. Ctrl+A) is
/// included. Excluded selections lie entirely before or after the range and
/// draw nothing, so the slice is exactly the drawing set.
fn visible_selection_span(sels: &[scrive_core::Selection], vis_start: u32, vis_end: u32) -> core::ops::Range<usize> {
    let lo = sels.partition_point(|s| s.end() < vis_start);
    let hi = sels.partition_point(|s| s.start() <= vis_end);
    lo..hi
}

/// Whether one selection in the sorted, non-overlapping `sels` covers the byte
/// range `[start, end]`. `O(log n)`: since selections don't overlap, the last
/// one starting at/before `start` is the only candidate that can reach `end`.
fn range_covered_by(sels: &[scrive_core::Selection], start: u32, end: u32) -> bool {
    let i = sels.partition_point(|s| s.start() <= start);
    i > 0 && sels[i - 1].end() >= end
}

impl<Message> Widget<Message, iced::Theme, iced::Renderer> for Editor<'_, Message> {
    fn tag(&self) -> widget::tree::Tag {
        widget::tree::Tag::of::<State>()
    }

    fn state(&self) -> widget::tree::State {
        widget::tree::State::new(State::default())
    }

    fn operate(
        &mut self,
        tree: &mut widget::Tree,
        layout: Layout<'_>,
        _renderer: &iced::Renderer,
        operation: &mut dyn Operation,
    ) {
        let state = tree.state.downcast_mut::<State>();
        operation.focusable(self.id.as_ref(), layout.bounds(), state);
    }

    fn size(&self) -> Size<Length> {
        Size::new(Length::Fill, Length::Fill)
    }

    fn layout(
        &mut self,
        tree: &mut widget::Tree,
        _renderer: &iced::Renderer,
        limits: &layout::Limits,
    ) -> layout::Node {
        let size = limits.max();
        let state = tree.state.downcast_mut::<State>();
        self.ensure_metrics(state);
        let (advance, line_h) = (state.metrics.advance, state.metrics.line_height);
        // Viewport rect for the scroll math (origin-independent — it uses widths
        // and the vertical-overflow test only).
        let vp = Rectangle { x: 0.0, y: 0.0, width: size.width, height: size.height };
        // A jump-class reveal request (find navigation) moved the caret
        // outside the widget's input path — pick it up as a one-shot autoscroll.
        // Jump-class reveals CENTER the target; the widget's own edits/moves get
        // `Fit` with the margin band.
        let jump = self.doc.reveal_seq() != state.last_reveal_seq;
        if jump {
            state.last_reveal_seq = self.doc.reveal_seq();
            state.autoscroll = true;
        }
        if state.autoscroll {
            let buffer = self.doc.buffer();
            let head = self.doc.selections().newest().head();
            let fold_map = self.fold_map();
            // Both reveal axes read the one fold-aware projection: the row
            // in display space, the cell inline-collapse-aware — a caret past a
            // collapsed chip or riding a fold's tail reveals where it renders.
            // A HIDDEN caret (inside a fold) reveals nothing — jump verbs
            // unfold their targets first; anything else holds the viewport
            // rather than scrolling to a fake row 0.
            if let Some(p) = fold_map.display_position(buffer, head, TAB) {
                // Row units end to end (the ScrollAnchor model): the caret's
                // display row is exact, the viewport is a small row count.
                let caret_row = f64::from(p.row.index());
                let viewport_rows = f64::from(size.height) / f64::from(line_h);
                let cur = state.scroll.rows(line_h);
                // The reveal mode is meaningful only on a jump (an app reveal
                // request). The widget's own autoscroll (typing, click, drag)
                // carries none, so it uses `Fit` — hold the viewport if a cursor
                // is already on screen.
                let mode = if jump { self.doc.reveal_mode() } else { RevealMode::Fit };
                let fit = || {
                    reveal_fit(
                        cur,
                        caret_row,
                        caret_row + 1.0,
                        viewport_rows,
                        f64::from(default_autoscroll_margin()),
                    )
                };
                let rows = match mode {
                    // Find/diagnostic jumps center the target row.
                    RevealMode::Center => caret_row + 0.5 - viewport_rows / 2.0,
                    // Ctrl+D: jump to the just-added cursor even if others show.
                    RevealMode::FitForce => fit(),
                    // Fit: HOLD the viewport if any cursor is already on screen —
                    // a multi-cursor op (select-all-occurrences, multi-cursor
                    // typing) must not scroll to the off-screen newest caret when
                    // the user can already see one (as mainstream editors do). A
                    // lone off-screen caret has none on screen and falls through
                    // to fit it, so single-cursor typing/moves are unchanged.
                    RevealMode::Fit if self.any_caret_on_screen(buffer, &fold_map, cur, viewport_rows) => cur,
                    RevealMode::Fit => fit(),
                };
                state.scroll = ScrollAnchor::from_rows(rows, line_h);
                // Horizontal (both reveal classes): keep cells [col−1,
                // col+2] inside the code area, minimally; a band wider than
                // the viewport is the reveal_fit no-op case.
                let cell = p.x.cells();
                let code_w = self.code_area_width(vp, advance, line_h, state.scroll.rows(line_h));
                state.scroll_x = reveal_fit(
                    f64::from(state.scroll_x),
                    f64::from((cell - 1.0).max(0.0) * advance),
                    f64::from((cell + 2.0) * advance),
                    f64::from(code_w),
                    0.0,
                ) as f32;
            }
            state.autoscroll = false;
        }
        // Re-canonicalize + clamp: from_rows floors the row and bounds the
        // sub-row offset, so wheel deltas and stale anchors normalize here.
        state.scroll = ScrollAnchor::from_rows(
            state.scroll.rows(line_h).min(self.max_scroll_rows(size.height, line_h)),
            line_h,
        );
        state.scroll_x =
            state.scroll_x.clamp(0.0, self.max_scroll_x(vp, advance, line_h, state.scroll.rows(line_h)));
        layout::Node::new(size)
    }

    fn draw(
        &self,
        tree: &widget::Tree,
        renderer: &mut iced::Renderer,
        theme: &iced::Theme,
        _style: &renderer::Style,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
    ) {
        use iced::advanced::Renderer as _; // start_layer / end_layer
        // Draw-budget gate: zero the per-frame row counter. Every windowed
        // wash/squiggle pass bumps it; the assert at the end of the code layer
        // trips if any pass walked the document instead of the viewport.
        draw_budget::reset();
        let state = tree.state.downcast_ref::<State>();
        let (advance, line_h) = (state.metrics.advance, state.metrics.line_height);
        let bounds = layout.bounds();
        let palette = theme.extended_palette();
        let geo = self.geo(state, bounds);
        let buffer = self.doc.buffer();
        // Whether the pointer is over the gutter — the expanded (chevron-down)
        // fold controls show only while it is, as mainstream editors do.
        let gutter_hover = cursor.position_over(bounds).is_some_and(|p| geo.in_gutter(p.x));
        let scroll_x = state.scroll_x;

        // The vertical scrollbar (if any), computed once and reused for both the
        // content clip and the thumb so they can never disagree. The code area —
        // where all horizontally-scrolled content lives — runs from the gutter's
        // right edge to the scrollbar's left edge (or the widget edge when there
        // is no scrollbar).
        let sb = self.scrollbar(bounds, line_h, state.scroll.rows(line_h));
        let content_right = code_area_right(bounds, sb.as_ref());
        let code_left = geo.code_left();
        let code_clip = Rectangle {
            x: code_left,
            y: bounds.y,
            width: (content_right - code_left).max(0.0),
            height: bounds.height,
        };

        // Background (fixed, does not scroll). The gutter shares the editor
        // background — no separate strip — so the line-number margin is seamless.
        fill(renderer, bounds, palette.background.base.color);

        let text_color = palette.background.base.text;
        let dim = palette.background.strong.color;
        // A neutral gray selection (theme-derived), not the accent — as
        // mainstream editors render selection, and it stays legible under
        // syntax colors.
        let selection_color = palette.background.strong.color;

        // Folds: rows hidden inside a fold are not produced. `first`/`last`
        // are DISPLAY rows (the visible window); the render iterates display rows
        // and maps each back to its buffer row.
        let fold_map = self.fold_map();
        // The visible display-row window — floor/ceil/clamp policy lives with
        // the fold map; the widget only supplies fractional rows from pixels.
        let window = fold_map
            .display_window(geo.rows_from_top(bounds.y), geo.rows_from_top(bounds.y + bounds.height));
        // The display Y of a buffer row, or `None` if it's hidden inside a fold or
        // scrolled out of the visible display window — the one place every overlay
        // routes its vertical position through.
        let visible_y = |row: u32| -> Option<f32> {
            let br = BufferRow(row);
            if fold_map.is_folded(br) {
                return None;
            }
            let dr = fold_map.to_display_row(br);
            window.contains(&dr.index()).then(|| geo.row_y(dr))
        };
        // Screen (x, top-y) of a buffer offset — the core's one fold-aware
        // projection (tab expansion, inline collapse, chip-center clipping,
        // collapsed-tail following), culled to the visible display window. `None`
        // if the offset is genuinely hidden (in a fold's gap) or scrolled out.
        let offset_xy = |off: u32| -> Option<(f32, f32)> {
            let p = fold_map.display_position(buffer, off, TAB)?;
            window.contains(&p.row.index()).then(|| (geo.cell_x(p.x.cells()), geo.row_y(p.row)))
        };

        // Current-line highlight: with a single bare cursor, tint the caret's
        // row across the *text area only* — not the line-number gutter, matching
        // mainstream editors' line highlight. Under selection/find/text.
        let selections = self.doc.selections();
        if selections.len() == 1 && selections.newest().is_empty() {
            let row = buffer.offset_to_point(selections.newest().head()).row;
            if let Some(y) = visible_y(row) {
                let color = Color { a: LINE_HIGHLIGHT_A, ..text_color };
                // Start where a selection does — the text origin (gutter + pad),
                // not the raw gutter edge; deliberately UNscrolled (the tint spans
                // the whole text area regardless of horizontal scroll).
                let x = geo.text_left();
                fill(renderer, Rectangle { x, y, width: (content_right - x).max(0.0), height: line_h }, color);
            }
        }

        // Line numbers (fixed — the gutter never scrolls horizontally). The
        // caret's row gets the brighter editor foreground while the rest stay
        // dim, so the active line's number stands out.
        let active_row = buffer.offset_to_point(selections.newest().head()).row;
        for vr in fold_map.visible_rows(window.clone()) {
            let row = vr.buffer_row.0;
            let y = geo.row_y(vr.display_row);
            let color = if row == active_row { text_color } else { dim };
            // Real buffer line numbers at display Y ⇒ continuous with a gap across
            // each fold, matching mainstream editors' gutter.
            self.draw_line(renderer, (row + 1).to_string(), Point::new(bounds.x + self.gutter_number_right(advance), y), color, Alignment::Right, bounds);
        }

        // Fold chevrons in the gutter on foldable header rows (dim), drawn in the
        // bundled Codicon font: chevron-right collapsed (always shown),
        // chevron-down expanded (only while the pointer is over the gutter, as
        // mainstream editors default). Centered in the padded fold column, clear
        // of the code. Clicking one toggles the fold (ButtonPressed). The chevron
        // on the pointer's own row brightens to the foreground (the active-line-
        // number treatment) so adjacent chevrons disambiguate what a click hits.
        let hover_chevron_row = cursor
            .position_over(bounds)
            .filter(|p| geo.in_gutter(p.x))
            .map(|p| fold_map.to_buffer_row(fold_map.display_row_at(geo.rows_from_top(p.y))).0);
        // Only headers on visible buffer rows can paint a chevron — a windowed
        // bracket query, never a whole-document scan per frame. Hidden in-window
        // headers still hit the `visible_y` cull below.
        let first_buf_row = fold_map.to_buffer_row(fold_map.display_row_at(f64::from(window.start))).0;
        let last_buf_row = fold_map.to_buffer_row(fold_map.display_row_at(f64::from(window.end))).0;
        for &(header, _) in &self.doc.foldable_ranges_in_rows(first_buf_row..last_buf_row + 1) {
            let Some(y) = visible_y(header.0) else { continue };
            let folded = fold_map.fold_at_header(header).is_some();
            if !folded && !gutter_hover {
                continue;
            }
            let glyph = if folded { crate::icon::CHEVRON_RIGHT } else { crate::icon::CHEVRON_DOWN };
            let color = if hover_chevron_row == Some(header.0) { text_color } else { dim };
            let origin = Point::new(bounds.x + self.gutter_chevron_x(advance), y);
            self.draw_icon(renderer, glyph, origin, advance, color, bounds);
        }

        // Everything in the code area scrolls horizontally by `scroll_x` and
        // clips to `code_clip`, so glyphs and fills alike stay off the gutter
        // (left) and out from under the vertical scrollbar (right). `fill_quad`
        // can't self-clip, so the whole block rides one clip layer.
        renderer.start_layer(code_clip);

        // The visible byte range (fold-aware) bounds every windowed wash below.
        // `first`/`last` are display rows; the range spans the visible buffer rows
        // (matches inside a fold are drawn by draw_selection, which culls them).
        let line_count = buffer.line_count();
        let vis_start = buffer
            .point_to_offset(BufPoint { row: fold_map.to_buffer_row(fold_map.display_row_at(f64::from(window.start))).0, col: 0 });
        let vis_end = if window.end >= fold_map.display_row_count() {
            buffer.len()
        } else {
            buffer.point_to_offset(BufPoint { row: fold_map.to_buffer_row(fold_map.display_row_at(f64::from(window.end))).0, col: 0 })
        };

        // Selection highlights (under the text) — only the selections that
        // intersect the visible byte range. The set is sorted and disjoint, so
        // ends are ascending too: two binary searches bound the visible slice,
        // so a document-scale multi-cursor set costs O(visible), not O(carets),
        // per frame (the off-screen ones draw nothing anyway). A selection
        // straddling the window (e.g. Ctrl+A) still lands in the slice and is drawn.
        let sels = self.doc.selections().all();
        for sel in &sels[visible_selection_span(sels, vis_start, vis_end)] {
            if !sel.is_empty() {
                self.draw_selection(renderer, &geo, sel.start(), sel.end(), selection_color);
            }
        }
        // Word-under-caret occurrence wash — highlight every occurrence of the
        // word under the caret, under the find washes and only while find is idle.
        // A per-frame pure query, like the FoldMap (no tracked state to go stale).
        if self.doc.find_query().is_none() {
            // The visible window bounds the scan, so it stays O(viewport).
            for span in self.doc.caret_word_occurrences(vis_start..vis_end) {
                self.draw_selection(renderer, &geo, span.start, span.end, OCCURRENCE_MATCH);
            }
        }
        for (span, is_active) in self.doc.find_matches_in(vis_start..vis_end) {
            let color = if is_active { FIND_MATCH_ACTIVE } else { FIND_MATCH };
            self.draw_selection(renderer, &geo, span.start, span.end, color);
        }

        // Indentation guides (on by default): monochrome 1 px vertical lines at
        // each indent stop of a line's *own* leading whitespace, spaced by TAB
        // display cells — no bracket involvement. A blank line takes its level
        // from its nearest non-blank neighbours (below, then above), so guides
        // run unbroken through blank gaps. scrive has no separate indent-size
        // knob, so TAB serves as both the tab expansion width and the level spacing.
        let indent_size = TAB;
        // A row's leading-whitespace width in cells, or None for a blank line.
        let indent_col = |row: u32| -> Option<u32> {
            let l = buffer.line(row);
            (!l.trim().is_empty()).then(|| line_indent_cells(&l))
        };
        // The guide walks (blank-line neighbor interpolation + the active-guide
        // extent) clamp to the window ± slack: the guides only paint visibly, and
        // a few hundred rows of context gives perceptually identical continuity
        // while keeping a blank-heavy file from scanning to the document edges.
        let guide_lo = first_buf_row.saturating_sub(INDENT_NEIGHBOR_SLACK);
        let guide_hi = (last_buf_row.saturating_add(INDENT_NEIGHBOR_SLACK)).min(line_count);
        // Indent level of a row: content = ceil(indent/size); a blank line takes
        // the level interpolated from its nearest non-blank neighbours so the
        // guides run unbroken through blank rows.
        let level_at = |row: u32| -> u32 {
            match indent_col(row) {
                own @ Some(_) => display_map::indent_guide_level(own, None, None, indent_size),
                None => {
                    let above = (guide_lo..row).rev().find_map(&indent_col);
                    let below = (row + 1..guide_hi).find_map(&indent_col);
                    display_map::indent_guide_level(None, above, below, indent_size)
                }
            }
        };
        // Active guide: pick one indent level to highlight over a line-range
        // around the caret, with scope-aware handling on opening/closing brace
        // lines (the child level is highlighted, not the caret line's own). A
        // caret far off-screen paints no visible active guide, so the clamp also
        // skips the computation entirely (noted for the field gate).
        let caret_row = buffer.offset_to_point(self.doc.selections().newest().head()).row;
        let active =
            display_map::active_indent_guide(caret_row, line_count, guide_lo..guide_hi, level_at);
        for vr in fold_map.visible_rows(window.clone()) {
            let row = vr.buffer_row.0;
            let level = level_at(row);
            let y = geo.row_y(vr.display_row);
            // Guides at levels 1..=level ⇒ cells 0, TAB, …, (level-1)*TAB. Since
            // level = ceil(indent/TAB), the deepest guide sits strictly left of a
            // content line's first glyph, so no content-overlap suppression is
            // needed.
            for lvl in 1..=level {
                let cell = (lvl - 1) * indent_size;
                let gx = geo.cell_x(cell as f32);
                let is_active = active.is_some_and(|(ind, s, e)| lvl == ind && row >= s && row <= e);
                let alpha = if is_active { GUIDE_ACTIVE_A } else { GUIDE_IDLE_A };
                fill(renderer, Rectangle { x: gx, y, width: GUIDE_WIDTH, height: line_h }, Color { a: alpha, ..text_color });
            }
        }

        // Text (line numbers are drawn fixed, above the layer). Iterates DISPLAY
        // rows: a fold's hidden interior is simply not produced, and a folded
        // header gets a trailing `…` placeholder chip.
        for vr in fold_map.visible_rows(window.clone()) {
            let row = vr.buffer_row.0;
            let y = geo.row_y(vr.display_row);
            let line = buffer.line(row);
            let origin = Point::new(geo.cell_x(0.0), y);
            let spans = self.doc.highlight_line_spans(row);
            // Rows with an inline (single-line) fold draw collapsed — the interior
            // between the brackets hides behind a `…` chip, the rest shifts left.
            let row_layout = fold_map.row_layout(buffer, BufferRow(row), TAB);
            if !row_layout.is_plain() {
                self.draw_row_inline(renderer, &line, spans, origin, &geo, &row_layout, dim, text_color, code_clip);
            } else {
                match spans {
                    Some(spans) if !spans.is_empty() => {
                        self.draw_spans(renderer, &line, spans, origin, advance, code_clip);
                    }
                    _ if !line.is_empty() => {
                        self.draw_line(renderer, expand_tabs(&line), origin, text_color, Alignment::Left, code_clip);
                    }
                    _ => {}
                }
            }
            if vr.is_fold_header {
                // The collapsed placeholder reads `fn main() { … }` — the `…`
                // chip in the gap, then the fold's REAL, cursor-addressable
                // closing tail. Every cell comes from the core's one
                // `HeaderLayout` owner, so the painted glyphs, caret placement,
                // hit-testing, and selection washes agree by construction.
                let hl = fold_map
                    .header_layout(buffer, BufferRow(row), TAB)
                    .expect("is_fold_header rows have a header layout");
                let mid = geo.cell_x(hl.gap_center());
                // Where the selection runs through the fold, the wash now spans the
                // gap (see `draw_wash_row`), so drop the idle pill — else it islands
                // as a dark box in the wash, exactly like the inline chip. The `…`
                // renders at full glyph color there so it stays visible. The test is
                // the same straddle `draw_wash_row` uses: a selection covering the
                // header's line end into the folded interior below.
                let hdr_end = buffer.point_to_offset(BufPoint { row, col: buffer.line_len(row) });
                let dots = if self.range_selected(hdr_end, hdr_end + 1) {
                    text_color
                } else {
                    fill_rounded(renderer, geo.chip_pill(mid, y), POPUP_SELECT, CHIP_PILL_RADIUS);
                    dim
                };
                self.draw_line(renderer, "…".to_string(), Point::new(mid, y), dots, Alignment::Center, code_clip);
                // The closing tail in its REAL colors — each bracket in its
                // pair-depth color (the closing bracket is on a hidden row, so
                // the bracket-colorization pass below never reaches it).
                let last_start = buffer.point_to_offset(BufPoint { row: hl.last_row().0, col: 0 });
                for g in hl.tail_glyphs() {
                    let off = last_start + g.col;
                    let color = self.doc.brackets().at(off).map_or(
                        text_color,
                        |b| {
                            if b.partner.is_none() {
                                UNMATCHED_BRACKET
                            } else {
                                BRACKET_DEPTH[b.depth as usize % BRACKET_DEPTH.len()]
                            }
                        },
                    );
                    let x = geo.cell_x(g.cell as f32);
                    self.draw_line(renderer, g.ch.to_string(), Point::new(x, y), color, Alignment::Left, code_clip);
                }
            }
        }

        // Bracket-pair colorization: over-paint each visible bracket in its
        // nesting-depth color (unmatched ⇒ red), on top of the syntax spans.
        // Per-visible-row windowed queries — never the whole bracket set per
        // frame; folds make the visible buffer rows non-contiguous, so per-row
        // is simplest-correct.
        for vr in fold_map.visible_rows(window.clone()) {
            let row = vr.buffer_row.0;
            let Some(y) = visible_y(row) else { continue };
            let row_start = buffer.point_to_offset(BufPoint { row, col: 0 });
            let row_end = if row + 1 < line_count {
                buffer.point_to_offset(BufPoint { row: row + 1, col: 0 })
            } else {
                buffer.len() + 1 // sentinel: a bracket at the final byte is inside
            };
            let row_layout = fold_map.row_layout(buffer, BufferRow(row), TAB);
            for br in self.doc.brackets().in_range_iter(row_start..row_end) {
                let col = br.offset - row_start;
                let Some(ch) = buffer.char_at(br.offset) else { continue };
                // On an inline-folded row, shift the bracket to its display
                // cell; one hidden inside a collapsed interior isn't drawn.
                if row_layout.glyph_hidden(col) {
                    continue;
                }
                let x = geo.cell_x(row_layout.display_cell(col) as f32);
                let color = if br.partner.is_none() {
                    UNMATCHED_BRACKET
                } else {
                    BRACKET_DEPTH[br.depth as usize % BRACKET_DEPTH.len()]
                };
                self.draw_line(renderer, ch.to_string(), Point::new(x, y), color, Alignment::Left, code_clip);
            }
        }

        // Matching-bracket highlight: when the primary caret is
        // adjacent to a matched bracket, box that bracket and its partner. The
        // outline is a translucent foreground tint — theme-derived, so it stays
        // legible over any depth color.
        let match_box = Color { a: 0.45, ..text_color };
        if let Some((a, b)) = self.doc.brackets().active_pair(self.doc.selections().newest().head()) {
            for off in [a, b] {
                // Route through `offset_xy` so a matched bracket on a collapsed
                // fold's closing tail (the `}` inline on the header line) gets boxed
                // too, not just the visible opening bracket.
                let Some((x, y)) = offset_xy(off) else { continue };
                fill_border(renderer, Rectangle { x, y, width: advance, height: line_h }, match_box, 1.0);
            }
        }

        // Diagnostic squiggles: a wavy underline per diagnostic, in ascending
        // severity so the most severe paints last (Ord on Severity). Row-culled
        // to the viewport; each row's run is at least one cell wide; x sits in
        // the code area (caret_x never enters the gutter). The store query is
        // binary-search-bounded to the visible byte range, so this is
        // O(visible diags). Iterate the VISIBLE display rows (O(viewport)) and
        // draw the diagnostics covering each — never a diagnostic's raw
        // `sp.row..=ep.row` buffer span, which a document-spanning diagnostic
        // over a folded file would blow up to O(document) per frame. `diags` is
        // already windowed to the visible bytes and sorted by ascending severity,
        // so the most severe over-paints last on any shared row. One collect
        // (carrying `sev` only to sort by it) derives each span's points + color
        // once here rather than in a second `Vec`.
        let mut diags: Vec<(u32, u32, u32, u32, Severity, Color)> = self
            .doc
            .diagnostics_in(vis_start..vis_end)
            .map(|(span, sev, _msg)| {
                let (sp, ep) = (buffer.offset_to_point(span.start), buffer.offset_to_point(span.end));
                (span.start, span.end, sp.row, ep.row, sev, severity_color(sev))
            })
            .collect();
        diags.sort_by_key(|&(.., sev, _)| sev);
        for vr in fold_map.visible_rows(window.clone()) {
            draw_budget::bump_rows(1);
            let row = vr.buffer_row.0;
            let Some(row_y) = visible_y(row) else { continue };
            for &(start, end, sp_row, ep_row, _sev, color) in &diags {
                if row < sp_row || row > ep_row {
                    continue;
                }
                let row_start = if row == sp_row { start } else { buffer.point_to_offset(BufPoint { row, col: 0 }) };
                let row_end = if row == ep_row { end } else { buffer.point_to_offset(BufPoint { row, col: buffer.line_len(row) }) };
                // A boundary row a multi-line span doesn't actually cover (its
                // end landing at column 0, or an empty interior line) has zero
                // width here — skip it so the min-one-cell rule below doesn't
                // manufacture a phantom squiggle. A genuinely zero-width
                // diagnostic (a point) still gets its one cell.
                if row_start == row_end && start != end {
                    continue;
                }
                // Fold-aware endpoints: a diagnostic spanning a collapsed
                // inline fold underlines the shifted glyphs, not the raw columns.
                let x0 = self.offset_screen_x(&fold_map, &geo, row_start);
                let x1 = self.offset_screen_x(&fold_map, &geo, row_end).max(x0 + advance);
                let baseline = row_y + line_h - SQUIGGLE_AMPLITUDE - 0.5;
                squiggle_spans(x0, x1, baseline, |rect| fill(renderer, rect, color));
            }
        }
        // Draw-budget gate: every wash/squiggle pass above is windowed to the
        // viewport, so the rows visited must stay proportional to the visible
        // window. A pass that walked the whole document would blow past this and
        // trip here in tests/dev builds, catching it before it could lock up the
        // field build.
        #[cfg(debug_assertions)]
        {
            let vp_rows = u64::from(window.end.saturating_sub(window.start));
            debug_assert!(
                draw_budget::rows() <= 1024 * (vp_rows + 1),
                "draw visited {} rows for a ~{vp_rows}-row viewport — a per-frame path went O(document) (draw budget)",
                draw_budget::rows(),
            );
        }

        // Ctrl+hover collapse affordance: while Ctrl is held, bound every
        // collapsible under the pointer with a dashed box — dim on the enclosing
        // nest, bright + washed on the innermost (the Ctrl+Click target). Inline
        // spans and blocks share the one indication; the gutter chevron stays the
        // block shortcut. Drawn inside the code-clip layer so it scrolls with the
        // text and stops at the gutter / scrollbar edges.
        if SHOW_CTRL_COLLAPSE_AFFORDANCE && state.modifiers.command() {
            if let Some(pos) = cursor.position_over(bounds) {
                // Discovery layer: while Ctrl is held and the pointer is over the
                // editor, EVERY visible collapsible shows a dim dashed box ("here's
                // what folds"); the innermost box the pointer is actually over is
                // bright + washed (the Ctrl+Click target), drawn last so it wins.
                let active = self
                    .armed_boxes(&geo, pos)
                    .into_iter()
                    .min_by_key(|&(o, c, _)| c - o)
                    .map(|(o, ..)| o);
                let first_buf = fold_map.to_buffer_row(fold_map.display_row_at(f64::from(window.start))).0;
                let last_buf = fold_map.to_buffer_row(fold_map.display_row_at(f64::from(window.end.saturating_sub(1)))).0;
                let mut active_rect = None;
                // Windowed to the visible rows (± slack for near-enclosing
                // blocks) — never a whole-document `collapsible_pairs()` scan per
                // frame while Ctrl is held.
                let query = first_buf.saturating_sub(FOLD_QUERY_SLACK)..last_buf + 1;
                for (open, close, header, last_row) in self.doc.collapsible_pairs_in_rows(query) {
                    if last_row < first_buf || header > last_buf {
                        continue; // wholly off-screen
                    }
                    let Some(rect) = self.collapsible_box_rect(&fold_map, &geo, open, close, header, last_row) else {
                        continue;
                    };
                    if active == Some(open) {
                        active_rect = Some(rect); // deferred so it paints over the nest
                    } else {
                        stroke_dashed_rounded_rect(renderer, rect, ARM_DASH, ARM_RADIUS, ARM_DASH_LEN, ARM_DASH_GAP, ARM_STROKE);
                    }
                }
                if let Some(rect) = active_rect {
                    fill_rounded(renderer, rect, ARM_FILL, ARM_RADIUS);
                    stroke_dashed_rounded_rect(renderer, rect, ARM_DASH_ACTIVE, ARM_RADIUS, ARM_DASH_LEN, ARM_DASH_GAP, ARM_STROKE);
                }
            }
        }

        // Plain-hover expand affordance: the `…` pill of the collapsed
        // fold under the (non-Ctrl) pointer brightens — quiet feedback that a
        // click expands it. Just the pill (the thing that *is* the hidden
        // content), no border. The click/finger target stays the whole chip
        // span (`collapsed_chip_rect`); only the visual is this tight.
        if !state.modifiers.command() {
            if let Some(opener) = state.hover_chip {
                if let Some(pill) = self.chip_pill_rect(&fold_map, &geo, opener) {
                    fill_rounded(renderer, pill, Color { a: CHIP_HOVER_A, ..text_color }, CHIP_PILL_RADIUS);
                }
            }
        }

        // Carets (over the text), clipped to the viewport — only in the solid
        // half of the blink cycle, and only when focused. Centered on the
        // insertion point (x − width/2), text-colored. A caret on a collapsed
        // fold's closing row rides the header line via `offset_xy`.
        if state.caret_on() {
            // Only carets whose selection intersects the visible byte range — the
            // rest project off-screen and are culled by offset_xy anyway. Bounds
            // the per-frame cost of a document-scale multi-cursor set to O(visible).
            let sels = self.doc.selections().all();
            for sel in &sels[visible_selection_span(sels, vis_start, vis_end)] {
                if let Some((x, y)) = offset_xy(sel.head()) {
                    let r = Rectangle { x: x - CARET_WIDTH / 2.0, y: y + 1.0, width: CARET_WIDTH, height: line_h - 2.0 };
                    fill(renderer, r, text_color);
                }
            }
        }

        // Close the horizontally-scrolled code-area clip layer.
        renderer.end_layer();

        // Overlays (scrollbars + completion popup) ride their OWN layer, pushed
        // *after* the content layer. wgpu renders layers in push order, so a base-
        // layer overlay would sit UNDER the content where they overlap (the
        // bottom hscrollbar row, the popup over code) and show through
        // translucently. (tiny_skia composites in draw order and hides that, so a
        // headless capture can't catch it — this must be checked on the wgpu build.)
        renderer.start_layer(bounds);

        // Scrollbar overlay + overview markers on the right edge (only on
        // overflow). Drawn last, over everything (reusing `sb` from the top so
        // the clip and the thumb agree). Markers (find matches + diagnostics)
        // map line → track position; the translucent thumb shows them through.
        if let Some(sb) = sb {
            // The track maps DISPLAY rows (fold-aware), so a marker sits where its
            // line actually shows; one inside a fold clips to the header.
            let total = fold_map.display_row_count().max(1) as f32;
            let mark_span = bounds.height - SCROLLBAR_MARK_H;
            let mark_y = |row: u32| bounds.y + (fold_map.to_display_row(BufferRow(row)).index() as f32 / total) * mark_span;
            // Overview markers ride separate ruler lanes, like mainstream
            // editors: the band splits into thirds, find matches take the center
            // lane and diagnostics the right lane. Distinct lanes mean a line
            // that is both a hit and an error shows both, instead of the
            // diagnostic overpainting the find tick.
            let lane_w = SCROLLBAR_WIDTH / 3.0;
            // Bucket the marks by TRACK PIXEL without scanning the store:
            // `overview_marks` folds each lane per pixel in O(P + log M), so no
            // per-frame pass walks every diagnostic or find match. The P+1 bucket
            // bounds are the INVERSE of the `round(y)` pixel map above — for track
            // pixel `p`, its band's lower edge is `y = p - 0.5`; invert that to the
            // first display row landing at/below it (`ceil` of the row-space
            // threshold), then to a buffer row, then to a byte offset. A mark at
            // offset `o` then falls in bucket `p` exactly when
            // `round(mark_y(row_of(o))) == p`, so the bucket IS the pixel: the
            // reduce picks the per-pixel winner, and the draw recovers its exact
            // float-y by re-projecting the winner's own offset.
            // `overview_reduce_equals_linear_scan` is the correctness authority; a
            // diagnostic-heavy `--capture` is the visual gate (float-quantization
            // at a pixel edge can nudge a boundary mark by one pixel — acceptable,
            // pending a human judging it on the running app).
            let denom = mark_span.max(1.0);
            let max_disp = fold_map.max_display_row().index();
            let px_to_offset = |p: f32| -> u32 {
                let t = ((p - 0.5 - bounds.y) / denom * total).ceil();
                if t <= 0.0 {
                    return 0; // at/above the first display row
                }
                let d = t as u32;
                if d > max_disp {
                    // Below the last row → tail sentinel. `len()+1` (not `len()`) so
                    // the last bucket is `[.., len+1)` and a diagnostic clipped to
                    // exactly `buffer.len()` (past-EOF, EmptyPolicy::Keep) still falls
                    // in a bucket and paints.
                    return buffer.len().saturating_add(1);
                }
                // `display_row_at(d)` reconstructs `DisplayRow(d)` (floor of a whole
                // number is itself) — the widget-crate-visible route to a DisplayRow
                // (its field is `pub(crate)`), the same idiom the fold-preview uses.
                let brow = fold_map.to_buffer_row(fold_map.display_row_at(f64::from(d))).0;
                buffer.point_to_offset(BufPoint { row: brow, col: 0 })
            };
            let p_lo = bounds.y.floor() as i32;
            let p_hi = (bounds.y + mark_span).ceil() as i32;
            // Pixels p_lo..=p_hi need one extra boundary (p_hi + 1), whose offset is
            // the past-tail sentinel (`len+1`) — so the last bucket captures the tail.
            let bucket_bounds: Vec<u32> =
                (p_lo..=p_hi + 1).map(|p| px_to_offset(p as f32)).collect();
            // Retained per-thread scratch — the two lane vectors are P-sized and
            // stable frame to frame, so clearing and refilling avoids a fresh,
            // growing, reallocating pair every frame the scrollbar exists. `draw`
            // takes `&self`, so this can't live in `State`.
            thread_local! {
                static OVERVIEW_LANES: std::cell::RefCell<OverviewLanes> =
                    const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
            }
            OVERVIEW_LANES.with(|cell| {
                let (sev_marks, find_marks) = &mut *cell.borrow_mut();
                self.doc.overview_marks(&bucket_bounds, sev_marks, find_marks);
                // Find lane (center): the first match in each pixel, at its exact y.
                for start in find_marks.iter().flatten() {
                    let y = mark_y(buffer.offset_to_point(*start).row);
                    fill(renderer, Rectangle { x: sb.x + lane_w, y, width: lane_w, height: SCROLLBAR_MARK_H }, FIND_MATCH_ACTIVE);
                }
                // Diagnostic lane (right): the severest mark in each pixel, at its
                // exact y. Encoded severity `0` (empty) and `1` (Hint, deliberately
                // given no overview color) are both skipped.
                for &(enc, off) in sev_marks.iter() {
                    if enc < 2 {
                        continue;
                    }
                    let sev = match enc {
                        2 => Severity::Info,
                        3 => Severity::Warning,
                        _ => Severity::Error, // encoded 4
                    };
                    let y = mark_y(buffer.offset_to_point(off).row);
                    fill(renderer, Rectangle { x: sb.x + 2.0 * lane_w, y, width: lane_w, height: SCROLLBAR_MARK_H }, severity_color(sev));
                }
            });
            let thumb = if state.scrollbar_grab.is_some() { SCROLLBAR_THUMB_ACTIVE } else { SCROLLBAR_THUMB };
            fill(renderer, Rectangle { x: sb.x, y: sb.thumb_y, width: SCROLLBAR_WIDTH, height: sb.thumb_h }, thumb);
        }

        // Horizontal scrollbar thumb (bottom edge, only on horizontal overflow) —
        // the same translucent thumb as the vertical bar, no overview markers. Its
        // track stops at the vertical bar, so the two meet at the corner cleanly.
        if let Some(hb) = self.hscrollbar(bounds, advance, line_h, scroll_x, state.scroll.rows(line_h)) {
            let thumb = if state.hscrollbar_grab.is_some() { SCROLLBAR_THUMB_ACTIVE } else { SCROLLBAR_THUMB };
            fill(renderer, Rectangle { x: hb.thumb_x, y: hb.y, width: hb.thumb_w, height: SCROLLBAR_WIDTH }, thumb);
        }

        // Hover popup — a markdown box at the hovered word.
        if let Some(info) = self.hover {
            self.draw_hover(renderer, info, &geo, text_color, state.hover_scroll);
        }

        // Signature-help box — above the caret, under the completion popup.
        if let Some(sig) = self.signature {
            self.draw_signature(renderer, sig, &geo, text_color);
        }

        // Fold hover preview — the collapsed fold's hidden content in a
        // floating panel, anchored under its `…` chip. In the overlay layer so it
        // rides over the code like the other popups.
        if let Some(opener) = state.fold_preview {
            self.draw_fold_preview(renderer, &geo, opener, text_color);
        }

        // Completion popup — render it over everything, clipped to
        // the widget (it may legitimately cover the gutter).
        if let Some(list) = self.popup.filter(|l| !l.filtered.is_empty()) {
            self.draw_popup(renderer, list, &geo, text_color);
        }

        // Close the overlay layer.
        renderer.end_layer();
    }

    fn update(
        &mut self,
        tree: &mut widget::Tree,
        event: &Event,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _renderer: &iced::Renderer,
        clipboard: &mut dyn Clipboard,
        shell: &mut Shell<'_, Message>,
        _viewport: &Rectangle,
    ) {
        let bounds = layout.bounds();
        let state = tree.state.downcast_mut::<State>();
        self.ensure_metrics(state);
        let (advance, line_h) = (state.metrics.advance, state.metrics.line_height);

        // Report the visible range to the app — the view reports its viewport to
        // the model: highlighting runs only down to what is on screen,
        // re-tokenizes newly revealed lines on scroll, and aims the retention
        // window at these rows. Checked on every event so scroll, resize, and
        // autoscroll are all caught; deduped so it publishes only when the range
        // actually moves.
        let last_visible = self.last_visible_row(bounds, state.scroll.rows(line_h), line_h);
        let first_visible = {
            let fm = self.fold_map();
            fm.to_buffer_row(fm.display_row_at(state.scroll.rows(line_h))).0
        }
        .saturating_sub(VIEWPORT_TOKENIZE_MARGIN);
        let visible = first_visible..last_visible + 1;
        if visible != state.last_reported_viewport {
            state.last_reported_viewport = visible.clone();
            shell.publish((self.on_action)(Action::ViewportChanged(visible)));
        }

        match event {
            Event::Mouse(mouse::Event::ButtonPressed(mouse::Button::Left)) => {
                let Some(pos) = cursor.position_over(bounds) else {
                    state.unfocus();
                    return;
                };
                state.focus();
                state.fold_preview = None; // a click dismisses any open fold preview
                let geo = self.geo(state, bounds);
                // A click inside the open completion popup (on top of everything)
                // accepts the row it lands on — never a caret or the scrollbar.
                // A click in the popup off the rows is swallowed.
                if let Some(list) = self.popup.filter(|l| !l.filtered.is_empty()) {
                    let (origin, sz) = self.popup_layout(list, &geo);
                    if (Rectangle { x: origin.x, y: origin.y, width: sz.width, height: sz.height }).contains(pos) {
                        let n = list.filtered.len();
                        let visible = n.min(popup::POPUP_MAX_VISIBLE);
                        // The inverse of draw_popup's row map — the same pair in
                        // popup.rs, so the click can't drift off the painted rows.
                        if let Some(row) = popup::row_at(origin.y, pos.y, line_h, visible) {
                            let idx = popup::window_start(list.selected as usize, n) + row;
                            shell.publish((self.on_action)(Action::PopupClickAccept(idx as u32)));
                        }
                        shell.capture_event();
                        return;
                    }
                }
                // A press in the scrollbar band drives the scrollbar, never a
                // caret: on the thumb it starts a drag from the grab point; on
                // the bare track it jumps the thumb to center on the click and
                // then drags from there.
                if let Some(sb) = self.scrollbar(bounds, line_h, state.scroll.rows(line_h)) {
                    if sb.contains_x(pos.x) {
                        let grab = if sb.thumb_contains_y(pos.y) {
                            pos.y - sb.thumb_y
                        } else {
                            let half = sb.thumb_h / 2.0;
                            state.scroll = ScrollAnchor::from_rows(sb.scroll_rows_for_thumb_top(pos.y - half), line_h);
                            half
                        };
                        state.scrollbar_grab = Some(grab);
                        shell.request_redraw();
                        shell.capture_event();
                        return;
                    }
                }
                // Same for the bottom horizontal bar (checked after the vertical
                // one, which owns the corner).
                if let Some(hb) = self.hscrollbar(bounds, advance, line_h, state.scroll_x, state.scroll.rows(line_h)) {
                    if hb.contains_y(pos.y) {
                        let grab = if hb.thumb_contains_x(pos.x) {
                            pos.x - hb.thumb_x
                        } else {
                            let half = hb.thumb_w / 2.0;
                            state.scroll_x = hb.scroll_for_thumb_left(pos.x - half);
                            half
                        };
                        state.hscrollbar_grab = Some(grab);
                        shell.request_redraw();
                        shell.capture_event();
                        return;
                    }
                }
                // A click in the gutter on a foldable header row toggles that fold
                // — before caret placement; the gutter never places a caret.
                if geo.in_gutter(pos.x) {
                    let fold_map = self.fold_map();
                    let display = fold_map.display_row_at(geo.rows_from_top(pos.y));
                    let row = fold_map.to_buffer_row(display).0;
                    if let Some(opener) = self.doc.block_opener_on_row(row) {
                        shell.publish((self.on_action)(Action::ToggleFold { opener }));
                        shell.capture_event();
                        return;
                    }
                }
                // A plain click on a collapsed fold's chip expands it — inline `[ … ]`
                // or a block header's `{ … }`. Before caret placement so it
                // reads as toggling the collapsed span. (Ctrl is the collapse gesture,
                // below; a modified click falls through to selection.)
                if !state.modifiers.command() && !state.modifiers.shift() && !state.modifiers.alt() {
                    if let Some(opener) = self.collapsed_chip_at(&geo, pos) {
                        shell.publish((self.on_action)(Action::ToggleFold { opener }));
                        shell.capture_event();
                        return;
                    }
                }
                // Ctrl+Click a collapsible (the Ctrl+hover affordance): collapse the
                // innermost foldable pair under the pointer. No modifier overlap — Alt
                // adds carets, Shift extends; plain Ctrl is otherwise just a caret.
                if SHOW_CTRL_COLLAPSE_AFFORDANCE && state.modifiers.command() && !state.modifiers.shift() && !state.modifiers.alt() {
                    let target = self
                        .armed_boxes(&geo, pos)
                        .into_iter()
                        .min_by_key(|&(o, c, _)| c - o)
                        .map(|(o, ..)| o);
                    if let Some(opener) = target {
                        shell.publish((self.on_action)(Action::ToggleFold { opener }));
                        shell.capture_event();
                        return;
                    }
                }
                let offset = self.hit_test(&geo, pos);
                // Alt+Click adds a caret. Otherwise the click count (iced
                // `mouse::Click`) selects a caret / word / line; a single click
                // also arms a drag-select anchored here (extended on CursorMoved).
                if state.modifiers.alt() && state.modifiers.shift() {
                    // Shift+Alt+drag → mouse box (column) selection. The anchor is
                    // a virtual cell point; the drag moves the active corner.
                    let anchor = self.hit_cell(&geo, pos);
                    state.column_drag_anchor = Some(anchor);
                    shell.publish((self.on_action)(Action::ColumnDrag { anchor, active: anchor }));
                } else if state.modifiers.alt() {
                    shell.publish((self.on_action)(Action::AddCaret(offset)));
                } else if state.modifiers.shift() {
                    // Shift+click extends the primary selection to the click,
                    // keeping its far end (`tail`) as the origin; a following drag
                    // keeps extending (by character) from there.
                    let origin = self.doc.selections().newest().tail();
                    state.drag = Some(Drag { granularity: Granularity::Char, origin });
                    shell.publish((self.on_action)(Action::DragSelect {
                        granularity: Granularity::Char,
                        origin,
                        head: offset,
                    }));
                } else {
                    // Click count picks the drag granularity: a single click just
                    // places a caret, a double/triple selects the word/line.
                    let click = mouse::Click::new(pos, mouse::Button::Left, state.last_click);
                    state.last_click = Some(click);
                    let granularity = match click.kind() {
                        mouse::click::Kind::Single => Granularity::Char,
                        mouse::click::Kind::Double => Granularity::Word,
                        mouse::click::Kind::Triple => Granularity::Line,
                    };
                    state.drag = Some(Drag { granularity, origin: offset });
                    let action = if granularity == Granularity::Char {
                        Action::PlaceCaret(offset)
                    } else {
                        Action::DragSelect { granularity, origin: offset, head: offset }
                    };
                    shell.publish((self.on_action)(action));
                }
                shell.capture_event();
            }
            Event::Mouse(mouse::Event::CursorMoved { .. }) => {
                // One frame's geometry for the whole arm, built before any scroll
                // mutation below so every gutter check and hit-test shares it.
                let geo = self.geo(state, bounds);
                // Gutter-hover fold chevrons: repaint once when the pointer
                // crosses on/off the gutter so the ▾ controls appear/disappear.
                let over_gutter = cursor.position_over(bounds).is_some_and(|p| geo.in_gutter(p.x));
                if over_gutter != state.gutter_hover {
                    state.gutter_hover = over_gutter;
                    shell.request_redraw();
                }
                // While Ctrl is held the collapse boxes track the pointer — repaint on
                // every move (enter / retarget / leave) so the discovery boxes appear,
                // the active one follows, and both clear when the pointer leaves.
                if SHOW_CTRL_COLLAPSE_AFFORDANCE && state.modifiers.command() {
                    shell.request_redraw();
                }
                // Any pointer move dismisses an open fold preview; it re-arms after
                // the idle delay if the pointer is still resting on a chip.
                if state.fold_preview.take().is_some() {
                    shell.request_redraw();
                }
                // Track the collapsed chip under the (non-Ctrl) pointer for the
                // immediate plain-hover expand highlight — repaint only when it changes.
                let chip = (!state.modifiers.command())
                    .then(|| cursor.position_over(bounds).and_then(|p| self.collapsed_chip_at(&geo, p)))
                    .flatten();
                if chip != state.hover_chip {
                    state.hover_chip = chip;
                    shell.request_redraw();
                }
                // Track the gutter row under the pointer so the hovered fold
                // chevron's brightening follows row-to-row moves WITHIN the
                // gutter (crossing the gutter edge repaints above) — repaint
                // only when the row changes.
                let gutter_row = cursor.position_over(bounds).filter(|p| geo.in_gutter(p.x)).map(|p| {
                    let fm = self.fold_map();
                    fm.to_buffer_row(fm.display_row_at(geo.rows_from_top(p.y))).0
                });
                if gutter_row != state.gutter_hover_row {
                    state.gutter_hover_row = gutter_row;
                    shell.request_redraw();
                }
                // Extend an armed drag to the cursor. Use the raw position (not
                // `position_over`) so a drag past the viewport edge keeps
                // selecting — `hit_test` clamps to the buffer.
                if let Some(grab) = state.scrollbar_grab {
                    // A scrollbar-thumb drag takes precedence over everything:
                    // the thumb top follows the cursor, keeping the grab point.
                    if let (Some(sb), Some(pos)) =
                        (self.scrollbar(bounds, line_h, state.scroll.rows(line_h)), cursor.position())
                    {
                        state.scroll = ScrollAnchor::from_rows(sb.scroll_rows_for_thumb_top(pos.y - grab), line_h);
                        shell.request_redraw();
                    }
                    shell.capture_event();
                } else if let Some(grab) = state.hscrollbar_grab {
                    // Bottom-bar thumb drag — the X-axis mirror of the above.
                    if let (Some(hb), Some(pos)) = (
                        self.hscrollbar(bounds, advance, line_h, state.scroll_x, state.scroll.rows(line_h)),
                        cursor.position(),
                    ) {
                        state.scroll_x = hb.scroll_for_thumb_left(pos.x - grab);
                        shell.request_redraw();
                    }
                    shell.capture_event();
                } else if let (Some(anchor), Some(pos)) = (state.column_drag_anchor, cursor.position()) {
                    // A box drag takes precedence over a normal drag.
                    let active = self.hit_cell(&geo, pos);
                    state.autoscroll = true;
                    shell.publish((self.on_action)(Action::ColumnDrag { anchor, active }));
                    shell.capture_event();
                } else if let (Some(drag), Some(pos)) = (state.drag, cursor.position()) {
                    let head = self.hit_test(&geo, pos);
                    state.autoscroll = true;
                    shell.publish((self.on_action)(Action::DragSelect {
                        granularity: drag.granularity,
                        origin: drag.origin,
                        head,
                    }));
                    shell.capture_event();
                } else if let Some(pos) =
                    cursor.position_over(bounds).filter(|p| !geo.in_gutter(p.x))
                {
                    // A plain move (no drag) over the CODE AREA: drive the hover
                    // idle timer. Restricted to the code area so a move over
                    // the gutter/chevrons never arms a hover (else a column-0 word
                    // on an unindented line would pop up). If the pointer is still
                    // over the open hover's word, keep it; otherwise dismiss and
                    // re-arm the timer for the new position (accurate `now` stamped
                    // on the next RedrawRequested).
                    let off = self.hit_test(&geo, pos);
                    // Keep the hover open while the pointer is over its word OR over
                    // the hover box itself — so it can be moved into and scrolled.
                    let still_in = self.hover.is_some_and(|h| {
                        (off >= h.range.start && off < h.range.end)
                            || self.hover_layout(h, &geo).rect.contains(pos)
                    });
                    if !still_in {
                        if self.hover.is_some() {
                            shell.publish((self.on_action)(Action::HoverDismiss));
                        }
                        state.hover_pos = Some(pos);
                        state.hover_rearm = true;
                        state.hover_scroll = 0.0; // a fresh hover starts un-scrolled
                        shell.request_redraw();
                    }
                } else {
                    // Pointer over the gutter or off the widget — cancel any
                    // pending / open hover.
                    state.hover_pos = None;
                    state.hover_at = None;
                    state.hover_scroll = 0.0;
                    if self.hover.is_some() {
                        shell.publish((self.on_action)(Action::HoverDismiss));
                    }
                }
            }
            Event::Mouse(mouse::Event::ButtonReleased(mouse::Button::Left)) => {
                // End any in-progress drag; capture the release only if we were
                // dragging, so a plain click's release still propagates.
                let ended_drag = state.drag.take().is_some();
                let ended_box = state.column_drag_anchor.take().is_some();
                let ended_sb = state.scrollbar_grab.take().is_some();
                let ended_hsb = state.hscrollbar_grab.take().is_some();
                if ended_drag || ended_box || ended_sb || ended_hsb {
                    shell.capture_event();
                }
            }
            Event::Mouse(mouse::Event::WheelScrolled { delta }) => {
                // Wheel over the open popup scrolls the list (moves the
                // selection), captured so it never scrolls the editor.
                let geo = self.geo(state, bounds);
                if let (Some(list), Some(pos)) =
                    (self.popup.filter(|l| !l.filtered.is_empty()), cursor.position_over(bounds))
                {
                    let (origin, sz) = self.popup_layout(list, &geo);
                    if (Rectangle { x: origin.x, y: origin.y, width: sz.width, height: sz.height }).contains(pos) {
                        let up = matches!(delta, mouse::ScrollDelta::Lines { y, .. } | mouse::ScrollDelta::Pixels { y, .. } if *y > 0.0);
                        shell.publish((self.on_action)(if up { Action::PopupUp } else { Action::PopupDown }));
                        shell.capture_event();
                        return;
                    }
                }
                // Wheel over an overflowing hover scrolls its content (widget
                // state, snapped to whole lines), captured so it never reaches
                // the editor beneath.
                if let (Some(info), Some(pos)) = (self.hover, cursor.position_over(bounds)) {
                    let l = self.hover_layout(info, &geo);
                    if l.overflow && l.rect.contains(pos) {
                        let dy = match delta {
                            mouse::ScrollDelta::Lines { y, .. } => y * line_h,
                            mouse::ScrollDelta::Pixels { y, .. } => *y,
                        };
                        let raw = (state.hover_scroll - dy).clamp(0.0, l.max_scroll);
                        state.hover_scroll = (raw / line_h).round() * line_h;
                        shell.request_redraw();
                        shell.capture_event();
                        return;
                    }
                }
                // A thumb drag owns the scroll position; a stray wheel tick
                // would jump the thumb only to be snapped back on the next
                // CursorMoved. Let the drag win.
                if state.scrollbar_grab.is_some()
                    || state.hscrollbar_grab.is_some()
                    || cursor.position_over(bounds).is_none()
                {
                    return;
                }
                let (dx, dy) = match delta {
                    mouse::ScrollDelta::Lines { x, y } => (x * advance * WHEEL_COLS, y * line_h * WHEEL_ROWS),
                    mouse::ScrollDelta::Pixels { x, y } => (*x, *y),
                };
                // Shift+wheel scrolls horizontally, as mainstream editors do: a
                // vertical mouse wheel becomes an X delta.
                let (dx, dy) = if state.modifiers.shift() { (dy, 0.0) } else { (dx, dy) };
                // Predominant-axis lock: the larger-magnitude delta wins, so a
                // trackpad flick doesn't drift diagonally.
                if dx.abs() > dy.abs() {
                    let max_x = self.max_scroll_x(bounds, advance, line_h, state.scroll.rows(line_h));
                    state.scroll_x = (state.scroll_x - dx).clamp(0.0, max_x);
                } else {
                    // Row-space wheel step: shift by dy/line_h rows, then
                    // re-anchor (from_rows clamps the top; max_scroll_rows the
                    // bottom) — no absolute pixel value ever materializes.
                    let max_rows = self.max_scroll_rows(bounds.height, line_h);
                    state.scroll = ScrollAnchor::from_rows(
                        (state.scroll.rows(line_h) - f64::from(dy) / f64::from(line_h))
                            .min(max_rows),
                        line_h,
                    );
                }
                shell.request_redraw();
                shell.capture_event();
            }
            Event::Keyboard(Keyboard::KeyPressed { key, text, modifiers, physical_key, .. }) => {
                if !state.is_focused() {
                    return;
                }
                // Numpad-with-NumLock quirk (iced/winit on Windows): a numpad
                // digit reports a *navigation* logical key (Numpad2 → ArrowDown,
                // Numpad8 → ArrowUp, NumpadDecimal → Delete, …) but carries the
                // real glyph in `text`. Prefer the text so numpad keys type
                // instead of moving the caret. NumLock off ⇒ `text` is None ⇒ this
                // is skipped and the navigation key stands. Modifier chords fall
                // through so Ctrl/Alt+numpad still reach their handlers.
                if !modifiers.command() && !modifiers.control() && !modifiers.alt() {
                    if let Some(ch) = numpad_text(physical_key, text.as_deref()) {
                        state.autoscroll = true;
                        state.ping();
                        shell.publish((self.on_action)(Action::Type(ch)));
                        shell.capture_event();
                        return;
                    }
                }
                // While the completion popup is open, its navigation keys are
                // captured here and drive the controller (in the app); every
                // other key falls through and re-enters as typing.
                if self.popup.is_some() {
                    // Only PLAIN navigation keys belong to the popup — modified
                    // chords (Ctrl+Enter insert-line, Ctrl+Alt add-caret, the
                    // Ctrl+Shift+Alt column arms) pass through to their own arms
                    // rather than being swallowed by the popup.
                    let plain = !modifiers.command() && !modifiers.alt();
                    let popup_action = match key {
                        Key::Named(Named::ArrowUp) if plain => Some(Action::PopupUp),
                        Key::Named(Named::ArrowDown) if plain => Some(Action::PopupDown),
                        Key::Named(Named::Enter | Named::Tab) if plain => Some(Action::PopupAccept),
                        Key::Named(Named::Escape) => Some(Action::PopupDismiss),
                        _ => None,
                    };
                    if let Some(action) = popup_action {
                        state.ping();
                        shell.publish((self.on_action)(action));
                        shell.capture_event();
                        return;
                    }
                }
                // Escape closes the signature box next (after the popup, before
                // the snippet session, in dismissal order).
                if self.signature.is_some() && matches!(key, Key::Named(Named::Escape)) {
                    state.ping();
                    shell.publish((self.on_action)(Action::SignatureClose));
                    shell.capture_event();
                    return;
                }
                // While a snippet session is active, Tab / Shift+Tab / Escape
                // drive it; everything else types/moves normally and the
                // app cancels the session when the caret escapes every stop.
                if self.snippet_active {
                    let snip = match key {
                        Key::Named(Named::Tab) if modifiers.shift() => Some(Action::SnippetTabPrev),
                        Key::Named(Named::Tab) => Some(Action::SnippetTab),
                        Key::Named(Named::Escape) => Some(Action::SnippetCancel),
                        _ => None,
                    };
                    if let Some(action) = snip {
                        state.ping();
                        shell.publish((self.on_action)(action));
                        shell.capture_event();
                        return;
                    }
                }
                // Ctrl+C/X/V need the clipboard and the document, which
                // `interpret_key` has no access to — handle them here first.
                if modifiers.command() && !modifiers.alt() {
                    if let Key::Character(c) = key {
                        match c.as_str() {
                            "c" => {
                                self.write_clipboard(clipboard);
                                shell.capture_event();
                                return;
                            }
                            "x" => {
                                self.write_clipboard(clipboard);
                                state.autoscroll = true;
                                state.ping();
                                shell.publish((self.on_action)(Action::Cut));
                                shell.capture_event();
                                return;
                            }
                            "v" => {
                                if let Some(pasted) = clipboard.read(ClipboardKind::Standard) {
                                    state.autoscroll = true;
                                    state.ping();
                                    // Whole-line copies paste above the caret's
                                    // line — a side table remembers ours.
                                    let entire_line = crate::clipboard::is_entire_line(&pasted);
                                    shell.publish((self.on_action)(Action::Paste {
                                        text: pasted,
                                        entire_line,
                                    }));
                                }
                                shell.capture_event();
                                return;
                            }
                            _ => {}
                        }
                    }
                }
                // Ctrl+Shift+[ / ] fold / unfold the caret's enclosing block;
                // on US layouts Shift makes the logical char `{` / `}`, so accept
                // both. Needs the doc + fold state, so it's handled here (like
                // Ctrl+C/V), not in the layout-free `interpret_key`.
                if modifiers.command() && !modifiers.alt() {
                    if let Key::Character(c) = key {
                        let unfold = matches!(c.as_str(), "]" | "}");
                        if unfold || matches!(c.as_str(), "[" | "{") {
                            // Fold/unfold the block at EVERY caret (the core
                            // resolves the openers and no-ops if none apply).
                            shell.publish((self.on_action)(Action::FoldAtCarets { unfold }));
                            shell.capture_event();
                            return;
                        }
                    }
                }
                // PageUp/PageDown move by a page of *viewport* rows, which
                // `interpret_key` (layout-free) can't know — resolve the row count
                // from the current bounds here.
                if let Key::Named(named @ (Named::PageUp | Named::PageDown)) = key {
                    let page = ((bounds.height / line_h).floor() as u32).max(1);
                    let motion = if *named == Named::PageUp {
                        Motion::PageUp(page)
                    } else {
                        Motion::PageDown(page)
                    };
                    state.autoscroll = true;
                    state.ping();
                    shell.publish((self.on_action)(Action::Move { motion, extend: modifiers.shift() }));
                    shell.capture_event();
                    return;
                }
                if let Some(action) = interpret_key(key, text.as_deref(), *modifiers) {
                    if action.moves_caret() {
                        state.autoscroll = true;
                    }
                    // Any accepted input restarts the blink so the caret stays
                    // solid while the user is working.
                    state.ping();
                    shell.publish((self.on_action)(action));
                    shell.capture_event();
                }
            }
            Event::Keyboard(Keyboard::ModifiersChanged(mods)) => {
                // Arming/disarming the collapse affordance (Ctrl held) repaints so the
                // dashed box appears/clears without waiting for a mouse move.
                if SHOW_CTRL_COLLAPSE_AFFORDANCE && mods.command() != state.modifiers.command() {
                    shell.request_redraw();
                }
                state.modifiers = *mods;
            }
            // Losing window focus ends every in-progress mouse gesture. If the
            // OS steals the pointer mid-drag (a UAC/consent dialog, Win+L, task
            // view, a global hotkey) the button-up is delivered elsewhere and
            // the ButtonReleased arm never runs — without this, a stale grab
            // would drive scroll/selection from a button-less cursor on the
            // next move. (CursorLeft is deliberately NOT a trigger: a drag past
            // the viewport edge must keep selecting.)
            Event::Window(window::Event::Unfocused) => {
                state.drag = None;
                state.column_drag_anchor = None;
                state.scrollbar_grab = None;
                state.hscrollbar_grab = None;
            }
            // Drive the caret blink: advance the clock and schedule the next
            // toggle. Mirrors iced `text_input` — the chain self-sustains from
            // each rendered frame while focused and idles when unfocused.
            Event::Window(window::Event::RedrawRequested(now)) => {
                if let Some(focus) = &mut state.focus {
                    focus.now = *now;
                    let elapsed = now.saturating_duration_since(focus.updated_at).as_millis();
                    let until = BLINK_MS - elapsed % BLINK_MS;
                    shell.request_redraw_at(*now + Duration::from_millis(until as u64));
                }
                // Hover idle timer: a move armed a re-arm — stamp the target
                // instant here (accurate `now`) and schedule a redraw for it. When
                // that redraw arrives with no intervening move, fire the query.
                if state.is_focused() {
                    if state.hover_rearm {
                        state.hover_rearm = false;
                        let at = *now + Duration::from_millis(HOVER_IDLE_DELAY_MS);
                        state.hover_at = Some(at);
                        shell.request_redraw_at(at);
                    }
                    if state.hover_at.is_some_and(|at| *now >= at) {
                        state.hover_at = None;
                        if let Some(pos) = state.hover_pos {
                            // Resting on a collapsed fold's `…` chip previews its
                            // hidden content — a widget-drawn panel. Otherwise
                            // the pointer's word drives the app hover query.
                            let geo = self.geo(state, bounds);
                            if let Some(opener) = self.collapsed_chip_at(&geo, pos) {
                                if state.fold_preview != Some(opener) {
                                    state.fold_preview = Some(opener);
                                    shell.request_redraw();
                                }
                                if self.hover.is_some() {
                                    shell.publish((self.on_action)(Action::HoverDismiss));
                                }
                            } else {
                                let off = self.hit_test(&geo, pos);
                                shell.publish((self.on_action)(Action::HoverQuery(off)));
                            }
                        }
                    }
                } else {
                    state.hover_at = None;
                    state.hover_rearm = false;
                }
            }
            _ => {}
        }
    }

    fn mouse_interaction(
        &self,
        tree: &widget::Tree,
        layout: Layout<'_>,
        cursor: mouse::Cursor,
        _viewport: &Rectangle,
        _renderer: &iced::Renderer,
    ) -> mouse::Interaction {
        let state = tree.state.downcast_ref::<State>();
        let advance = state.metrics.advance;
        let bounds = layout.bounds();
        // A thumb drag (either bar) holds the arrow for its whole duration, even
        // as the cursor drifts off the band (the grab tracks the raw cursor).
        if state.scrollbar_grab.is_some() || state.hscrollbar_grab.is_some() {
            return mouse::Interaction::default();
        }
        let line_h = state.metrics.line_height;
        let geo = self.geo(state, bounds);
        // Ctrl held over a collapsible → the finger cursor (Ctrl+Click folds it).
        if SHOW_CTRL_COLLAPSE_AFFORDANCE && state.modifiers.command() {
            if let Some(p) = cursor.position_over(bounds).filter(|p| !geo.in_gutter(p.x)) {
                if !self.armed_boxes(&geo, p).is_empty() {
                    return mouse::Interaction::Pointer;
                }
            }
        }
        // Plain hover over a collapsed fold's chip → the finger (a plain click
        // expands it).
        if !state.modifiers.command() {
            if let Some(p) = cursor.position_over(bounds).filter(|p| !geo.in_gutter(p.x)) {
                if self.collapsed_chip_at(&geo, p).is_some() {
                    return mouse::Interaction::Pointer;
                }
            }
        }
        match cursor.position_over(bounds) {
            // Over the gutter: a pointer (finger) on a foldable header row — the
            // clickable fold chevron — otherwise the default arrow.
            Some(p) if geo.in_gutter(p.x) => {
                let fold_map = self.fold_map();
                let display = fold_map.display_row_at(geo.rows_from_top(p.y));
                let row = fold_map.to_buffer_row(display).0;
                // Windowed to the hovered row — never a whole-document
                // `foldable_ranges()` scan per mouse-move, so the gutter cursor
                // stays responsive on a large document.
                if self.doc.foldable_ranges_in_rows(row..row + 1).iter().any(|(h, _)| h.0 == row) {
                    mouse::Interaction::Pointer
                } else {
                    mouse::Interaction::default()
                }
            }
            // The scrollbar bands (right / bottom, only on overflow) keep the
            // arrow; the text area past the gutter shows the I-beam.
            Some(p)
                if !geo.in_gutter(p.x)
                    && !self.scrollbar(bounds, line_h, state.scroll.rows(line_h)).is_some_and(|sb| sb.contains_x(p.x))
                    && !self
                        .hscrollbar(bounds, advance, line_h, state.scroll_x, state.scroll.rows(line_h))
                        .is_some_and(|hb| hb.contains_y(p.y)) =>
            {
                mouse::Interaction::Text
            }
            _ => mouse::Interaction::default(),
        }
    }
}

/// The per-frame draw-budget gate — the structural guarantee that no per-frame
/// draw pass costs more than the viewport. Every windowed draw pass bumps the
/// rows it visits; [`draw`](Editor::draw) asserts the frame total stays
/// viewport-proportional, so any pass that walks the whole document trips a
/// `debug_assert` in tests and dev builds rather than locking up the field build.
/// The invariant is not "binary-search your lookups" — it is "**no per-frame work
/// proportional to anything but the viewport**." Zero-cost in release.
#[cfg(any(test, debug_assertions))]
pub(crate) mod draw_budget {
    use std::sync::atomic::{AtomicU64, Ordering};
    static ROWS: AtomicU64 = AtomicU64::new(0);
    /// Zero the counter at the top of a frame.
    pub(crate) fn reset() {
        ROWS.store(0, Ordering::Relaxed);
    }
    /// Record `n` rows visited by a windowed draw pass this frame.
    pub(crate) fn bump_rows(n: u64) {
        ROWS.fetch_add(n, Ordering::Relaxed);
    }
    /// The rows visited so far this frame.
    pub(crate) fn rows() -> u64 {
        ROWS.load(Ordering::Relaxed)
    }
}
#[cfg(not(any(test, debug_assertions)))]
pub(crate) mod draw_budget {
    #[inline(always)]
    pub(crate) fn reset() {}
    #[inline(always)]
    pub(crate) fn bump_rows(_n: u64) {}
}

impl<Message> Editor<'_, Message> {
    /// The completion popup's top-left and size for the current caret/metrics —
    /// shared by `draw_popup` and mouse hit-testing so they agree on where it is.
    fn popup_layout(&self, list: &PopupList, geo: &Geo) -> (Point, Size) {
        let fold_map = self.fold_map();
        // The anchor (the completed word's start) shares the caret's row, so the
        // shared display-space anchor covers both x and y.
        let (anchor_x, row_top, row_bottom) = self.popup_anchor(&fold_map, geo, list.anchor);
        let size = popup::extent(list, geo.advance(), geo.line_h());
        let origin = popup::place(anchor_x, row_top, row_bottom, size, geo.bounds());
        (origin, size)
    }

    /// Geometry + wrapped content for the hover popup — one source of truth,
    /// shared by `draw_hover` and the wheel-scroll handler. Parses the markdown,
    /// word-wraps each source line at the box width, caps the visible height at
    /// `HOVER_MAX_VISIBLE`, and places the box above the word (flips below when
    /// tight).
    fn hover_layout(&self, info: &HoverInfo, geo: &Geo) -> HoverLayout {
        let (bounds, advance, line_h) = (geo.bounds(), geo.advance(), geo.line_h());
        let pad_x = 8.0_f32; // hovers breathe more on the x axis
        let pad_y = popup::POPUP_PAD;
        let md_lines: Vec<Vec<(String, MdStyle)>> = info.markdown.lines().map(parse_md_runs).collect();
        let vis_len = |runs: &[(String, MdStyle)]| runs.iter().map(|(t, _)| t.chars().count()).sum::<usize>();
        let cells = md_lines.iter().map(|r| vis_len(r)).max().unwrap_or(0).clamp(popup::POPUP_MIN_CH, popup::POPUP_MAX_CH);
        let lines: Vec<Vec<(String, MdStyle)>> = md_lines.iter().flat_map(|r| wrap_runs(r, cells)).collect();
        let total = lines.len().max(1);
        let visible = total.min(HOVER_MAX_VISIBLE);
        let overflow = total > visible;

        let inner_w = cells as f32 * advance;
        let sb_w = if overflow { HOVER_SB_W } else { 0.0 };
        let width = inner_w + 2.0 * pad_x + sb_w;
        let height = visible as f32 * line_h + 2.0 * pad_y;

        let fold_map = self.fold_map();
        // The shared display-space anchor; the hover keeps its own
        // above-first flip below.
        let (word_x, word_top, word_bottom) = self.popup_anchor(&fold_map, geo, info.range.start);
        let y = if word_top - height >= bounds.y { word_top - height } else { word_bottom };
        let x = word_x.clamp(bounds.x, (bounds.x + bounds.width - width).max(bounds.x));

        HoverLayout {
            rect: Rectangle { x, y, width, height },
            lines,
            visible,
            max_scroll: (total - visible) as f32 * line_h,
            pad_x,
            pad_y,
            overflow,
        }
    }

    /// Render the hover popup — a rich-markdown box anchored at the hovered
    /// word (above it, flips below when tight). Wraps at the box width; when the
    /// content is taller than `HOVER_MAX_VISIBLE` it scrolls (wheel-driven
    /// `hover_scroll`, snapped to whole lines) with an auto scrollbar.
    fn draw_hover(
        &self,
        renderer: &mut iced::Renderer,
        info: &HoverInfo,
        geo: &Geo,
        text_color: Color,
        hover_scroll: f32,
    ) {
        let (advance, line_h) = (geo.advance(), geo.line_h());
        let l = self.hover_layout(info, geo);
        fill_panel(renderer, l.rect, POPUP_SURFACE, POPUP_BORDER, 8.0);

        let scroll = hover_scroll.clamp(0.0, l.max_scroll);
        let first = (scroll / line_h).round() as usize;
        // Each visible line's runs, left-to-right (monospace ⇒ cell = char): bold
        // in a bold weight, inline `code` over a rounded pill, plain as-is.
        for (vi, line) in l.lines.iter().skip(first).take(l.visible).enumerate() {
            let ly = l.rect.y + l.pad_y + vi as f32 * line_h;
            let mut col = 0usize;
            for (run, st) in line {
                let rx = l.rect.x + l.pad_x + col as f32 * advance;
                let n = run.chars().count();
                if *st == MdStyle::Code {
                    let pill = Rectangle { x: rx - 2.0, y: ly + 1.0, width: n as f32 * advance + 4.0, height: line_h - 2.0 };
                    fill_rounded(renderer, pill, CODE_PILL_BG, 3.0);
                }
                let font = if *st == MdStyle::Bold {
                    Font { weight: iced::font::Weight::Bold, ..self.font }
                } else {
                    self.font
                };
                self.draw_run(renderer, run.clone(), Point::new(rx, ly), text_color, font, l.rect);
                col += n;
            }
        }

        // Auto scrollbar on the right edge when the content overflows.
        if l.overflow {
            let track_h = l.visible as f32 * line_h;
            let thumb_h = (l.visible as f32 / l.lines.len() as f32 * track_h).max(HOVER_SB_W);
            let frac = if l.max_scroll > 0.0 { scroll / l.max_scroll } else { 0.0 };
            let thumb_y = l.rect.y + l.pad_y + frac * (track_h - thumb_h);
            let sbx = l.rect.x + l.rect.width - HOVER_SB_W - 1.0;
            fill_rounded(renderer, Rectangle { x: sbx, y: thumb_y, width: HOVER_SB_W, height: thumb_h }, SCROLLBAR_THUMB, HOVER_SB_W / 2.0);
        }
    }

    /// Render the signature-help box — a one-line box above the caret
    /// showing the signature with the active parameter highlighted, and its doc
    /// below. Reuses the popup surface/border; flips below the caret if there is
    /// no room above.
    #[allow(clippy::too_many_arguments)]
    fn draw_signature(
        &self,
        renderer: &mut iced::Renderer,
        sig: &SignatureInfo,
        geo: &Geo,
        text_color: Color,
    ) {
        let (bounds, advance, line_h) = (geo.bounds(), geo.advance(), geo.line_h());
        let fold_map = self.fold_map();
        let head = self.doc.selections().newest().head();
        // The shared display-space anchor; the signature keeps its own
        // above-first flip below.
        let (caret_x, caret_top, caret_bottom) = self.popup_anchor(&fold_map, geo, head);

        let dim = Color { a: 0.7, ..text_color };
        let active = Color::from_rgb8(0x66, 0xd9, 0xef); // cyan — the active param

        let label_cells = sig.label.chars().count();
        let doc_cells = sig.doc.as_ref().map_or(0, |d| d.chars().count());
        let cells = label_cells.max(doc_cells).clamp(popup::POPUP_MIN_CH, popup::POPUP_MAX_CH);
        let width = cells as f32 * advance + 2.0 * popup::POPUP_PAD;
        let rows = if sig.doc.is_some() { 2 } else { 1 };
        let height = rows as f32 * line_h + 2.0 * popup::POPUP_PAD;
        // Above the caret; flip below when there's no room above.
        let y = if caret_top - height >= bounds.y { caret_top - height } else { caret_bottom };
        let x = caret_x.clamp(bounds.x, (bounds.x + bounds.width - width).max(bounds.x));

        let rect = Rectangle { x, y, width, height };
        fill_panel(renderer, rect, POPUP_SURFACE, POPUP_BORDER, 8.0);

        let tx = x + popup::POPUP_PAD;
        let ty = y + popup::POPUP_PAD;
        // The label dim, then the active parameter's substring highlighted over it.
        self.draw_line(renderer, sig.label.clone(), Point::new(tx, ty), dim, Alignment::Left, rect);
        if let Some(range) = sig.active_param() {
            let (s, e) = (range.start as usize, range.end as usize);
            if let Some(sub) = sig.label.get(s..e) {
                let prefix = sig.label[..s].chars().count();
                let px = tx + prefix as f32 * advance;
                self.draw_line(renderer, sub.to_string(), Point::new(px, ty), active, Alignment::Left, rect);
            }
        }
        if let Some(doc) = &sig.doc {
            let dy = ty + line_h;
            self.draw_line(renderer, doc.clone(), Point::new(tx, dy), Color { a: 0.5, ..text_color }, Alignment::Left, rect);
        }
    }

    /// Render the completion popup — the box, the windowed rows (label
    /// left / detail right, the selected row highlighted), and the selected
    /// item's wrapped doc block below. Placement flips above the caret when
    /// there's no room below.
    fn draw_popup(&self, renderer: &mut iced::Renderer, list: &PopupList, geo: &Geo, text_color: Color) {
        let line_h = geo.line_h();
        let (origin, size) = self.popup_layout(list, geo);

        let dim = Color { a: 0.6, ..text_color };

        // Elevated neutral surface (shared with the hover / signature box).
        // Opaque, so the editor text can't bleed through; drawn first, the rows +
        // selection paint on top.
        let rect = Rectangle { x: origin.x, y: origin.y, width: size.width, height: size.height };
        fill_panel(renderer, rect, POPUP_SURFACE, POPUP_BORDER, 8.0);

        let n = list.filtered.len();
        let visible = n.min(popup::POPUP_MAX_VISIBLE);
        let start = popup::window_start(list.selected as usize, n);
        let row_x = origin.x + popup::POPUP_PAD;
        let right = origin.x + size.width - popup::POPUP_PAD;
        for (vis, &fi) in list.filtered[start..start + visible].iter().enumerate() {
            let item = &list.items[fi as usize];
            let y = popup::row_y(origin.y, vis, line_h);
            if start + vis == list.selected as usize {
                fill(renderer, Rectangle { x: origin.x, y, width: size.width, height: line_h }, POPUP_SELECT);
            }
            self.draw_line(renderer, item.label.clone(), Point::new(row_x, y), kind_color(item.kind), Alignment::Left, rect);
            if let Some(detail) = &item.detail {
                self.draw_line(renderer, detail.clone(), Point::new(right, y), dim, Alignment::Right, rect);
            }
        }

        // Doc block below the list (wrapped plain text), if any.
        let doc_lines = popup::doc_lines(list, popup::width_cells(list));
        if doc_lines > 0 {
            if let Some(doc) = popup::selected_doc(list) {
                let y = popup::row_y(origin.y, visible, line_h); // one past the last row
                fill(renderer, Rectangle { x: origin.x, y, width: size.width, height: 1.0 }, POPUP_BORDER);
                renderer.fill_text(
                    Text {
                        content: doc.to_string(),
                        bounds: Size::new(size.width - 2.0 * popup::POPUP_PAD, doc_lines as f32 * line_h),
                        size: Pixels(self.size),
                        line_height: LineHeight::Absolute(Pixels(line_h)),
                        font: self.font,
                        align_x: Alignment::Left,
                        align_y: Vertical::Top,
                        shaping: Shaping::Basic,
                        wrapping: Wrapping::Word,
                    },
                    Point::new(row_x, y + 1.0),
                    dim,
                    rect,
                );
            }
        }
    }

    /// Whether one of the current selections covers the byte range `[start, end]`
    /// — used to drop a collapsed inline fold's idle chip pill where the selection
    /// wash already backs it. `end` is compared in the wash's offset→cell sense
    /// (its right edge sits at `offset_screen_x(end)`).
    fn range_selected(&self, start: u32, end: u32) -> bool {
        range_covered_by(self.doc.selections().all(), start, end)
    }

    fn draw_selection(&self, renderer: &mut iced::Renderer, geo: &Geo, start: u32, end: u32, color: Color) {
        let buffer = self.doc.buffer();
        let fold_map = self.fold_map();
        let (a, b) = (buffer.offset_to_point(start), buffer.offset_to_point(end));
        // Single-row selection — the common case (a bare caret, a word
        // occurrence, a find match): wash it directly, no window walk.
        if a.row == b.row {
            self.draw_wash_row(renderer, &fold_map, geo, a.row, a, b, start, end, color);
            return;
        }
        // Multi-row: iterate the VISIBLE DISPLAY rows, never the selection's
        // raw buffer-row span. A document-spanning selection (Ctrl+A) over a
        // folded file stays O(viewport): walking every buffer row start→end would
        // be O(document) per frame — a huge fold at the viewport top makes even
        // the visible *buffer*-row span O(document), so only the display-row
        // window is safe. Each visible row washes its own line; a collapsed
        // header additionally washes the `}` tail
        // riding its display line (reached through the header's `last_folded`, so
        // a tail below the window is still drawn without visiting the rows between).
        let bounds = geo.bounds();
        let window = fold_map
            .display_window(geo.rows_from_top(bounds.y), geo.rows_from_top(bounds.y + bounds.height));
        let sel = a.row..=b.row;
        for vr in fold_map.visible_rows(window) {
            if sel.contains(&vr.buffer_row.0) {
                self.draw_wash_row(renderer, &fold_map, geo, vr.buffer_row.0, a, b, start, end, color);
            }
            if let Some(tail) = vr.last_folded {
                if sel.contains(&tail.0) {
                    self.draw_fold_tail_wash(renderer, &fold_map, geo, tail.0, a, b, color);
                }
            }
        }
    }

    /// Wash one visible buffer `row` of a selection: a hidden (folded) row defers
    /// to [`draw_fold_tail_wash`] (its `}` tail rides a header line); otherwise
    /// fill the selected span, endpoints via the one `offset_screen_x` projection.
    #[allow(clippy::too_many_arguments)] // a renderer-side wash genuinely needs all of them
    fn draw_wash_row(&self, renderer: &mut iced::Renderer, fold_map: &FoldMap, geo: &Geo, row: u32, a: BufPoint, b: BufPoint, start: u32, end: u32, color: Color) {
        // The choke point every washed row flows through — count it so the draw
        // budget trips if a caller ever washes O(document) rows.
        draw_budget::bump_rows(1);
        if fold_map.is_folded(BufferRow(row)) {
            self.draw_fold_tail_wash(renderer, fold_map, geo, row, a, b, color);
            return;
        }
        let (bounds, advance, line_h) = (geo.bounds(), geo.advance(), geo.line_h());
        let buffer = self.doc.buffer();
        let y = geo.row_y(fold_map.to_display_row(BufferRow(row)));
        if y + line_h < bounds.y || y > bounds.y + bounds.height {
            return;
        }
        let x0 = if row == a.row { self.offset_screen_x(fold_map, geo, start) } else { geo.cell_x(0.0) };
        let x1 = if row == b.row {
            self.offset_screen_x(fold_map, geo, end)
        } else if let Some(hl) = fold_map.header_layout(buffer, BufferRow(row), TAB) {
            // A collapsed block header the selection runs THROUGH (it continues into
            // the folded interior below): wash straight across the ` … ` placeholder
            // gap so the collapsed `head … tail` line has no unwashed hole. The tail
            // wash resumes exactly at `tail_cell`, so the two are contiguous.
            geo.cell_x(hl.tail_cell() as f32)
        } else {
            let line_end = buffer.point_to_offset(scrive_core::Point::new(row, buffer.line_len(row)));
            self.offset_screen_x(fold_map, geo, line_end) + advance * 0.5
        };
        fill(renderer, Rectangle { x: x0, y, width: (x1 - x0).max(1.0), height: line_h }, color);
    }

    /// Wash the selected part of a collapsed fold's closing (tail) row, which
    /// renders inline on its header's display line. `tail_row` is the fold's
    /// last buffer row; a no-op when it is not a hidden tail or its header is
    /// scrolled off. Split out so both the windowed loop and the single-row path
    /// reuse the exact same tail projection.
    #[allow(clippy::too_many_arguments)] // a renderer-side wash genuinely needs all of them
    fn draw_fold_tail_wash(&self, renderer: &mut iced::Renderer, fold_map: &FoldMap, geo: &Geo, tail_row: u32, a: BufPoint, b: BufPoint, color: Color) {
        draw_budget::bump_rows(1); // choke point — counted for the draw budget
        let (bounds, advance, line_h) = (geo.bounds(), geo.advance(), geo.line_h());
        let buffer = self.doc.buffer();
        // Hidden interior rows show nothing — only a collapsed fold's closing
        // (tail) row rides a header line.
        let Some(hdr) = fold_map.header_of_tail(BufferRow(tail_row)) else { return };
        let Some(hl) = fold_map.header_layout(buffer, hdr, TAB) else { return };
        let lead = hl.tail_start_col();
        let sel_a = if tail_row == a.row { a.col } else { 0 };
        let sel_b = if tail_row == b.row { b.col } else { buffer.line_len(tail_row) };
        let s_col = sel_a.max(lead); // clamp to the visible tail
        let y = geo.row_y(fold_map.to_display_row(hdr));
        let onscreen = y + line_h >= bounds.y && y <= bounds.y + bounds.height;
        if s_col <= sel_b && onscreen {
            // Both endpoints through the one tail projection — the wash sits exactly
            // under the tail glyphs/caret, including when a collapsed inline fold
            // precedes the block on the header.
            let (Some(c0), Some(c1)) = (hl.tail_col_cell(s_col), hl.tail_col_cell(sel_b)) else {
                return;
            };
            let x0 = geo.cell_x(c0 as f32);
            let x1 = geo.cell_x(c1 as f32) + if tail_row == b.row { 0.0 } else { advance * 0.5 };
            fill(renderer, Rectangle { x: x0, y, width: (x1 - x0).max(1.0), height: line_h }, color);
        }
    }

    /// Copy the selections to the system clipboard: the core's one payload
    /// rule ([`Document::clipboard_payload`] — non-empty selections joined, or
    /// whole lines for empty carets), re-expanded to the OS EOL flavor and
    /// recorded in a side table so paste can recognize a whole-line copy.
    fn write_clipboard(&self, clipboard: &mut dyn Clipboard) {
        let (text, entire_line) = self.doc.clipboard_payload();
        if !text.is_empty() {
            let exported = crate::clipboard::export_eol(&text);
            crate::clipboard::record(&exported, entire_line);
            clipboard.write(ClipboardKind::Standard, exported);
        }
    }

    /// The `(visible buffer row, display cell)` under `pos` for box (column)
    /// selection: the row via the ONE row inversion, the cell via the ONE
    /// virtual-cell rounding rule ([`scrive_core::virtual_cell`]) — UNclamped
    /// by the line, so a drag past a short line still reaches the full width
    /// of longer ones. The core resolves each spanned row's cells to bytes
    /// (`Document::column_drag`).
    fn hit_cell(&self, geo: &Geo, pos: Point) -> (u32, u32) {
        let fold_map = self.fold_map();
        // The click lands on a *display* row; map it back to a buffer row (a
        // folded interior is unreachable — clicks resolve to visible rows).
        let row = fold_map.to_buffer_row(fold_map.display_row_at(geo.rows_from_top(pos.y))).0;
        (row, scrive_core::virtual_cell(geo.x_cell(pos.x)))
    }

    /// Pixel → buffer offset: the y half resolves through the ONE row-inversion
    /// policy ([`FoldMap::display_row_at`]), the x half through the ONE
    /// cell-inversion owner ([`FoldMap::hit_row`]: collapsed header gap/tail
    /// resolution, chip clicks, tab snapping). The widget's only contribution is
    /// px → (fractional row, fractional cell) via [`Geo`].
    fn hit_test(&self, geo: &Geo, pos: Point) -> u32 {
        let fold_map = self.fold_map();
        let row = fold_map.to_buffer_row(fold_map.display_row_at(geo.rows_from_top(pos.y)));
        fold_map.hit_row(self.doc.buffer(), row, geo.x_cell(pos.x), scrive_core::Bias::Left, TAB)
    }

    /// Draw a row that has inline (single-line) folds: its highlight spans with a
    /// per-fold horizontal collapse — the bytes between each pair's brackets hide,
    /// later cells shift left, and a `…` chip fills the gap.
    /// The delimiters keep their span color; the closer stays a real position.
    /// All cell math comes from the core's one [`RowLayout`] owner.
    #[allow(clippy::too_many_arguments)]
    fn draw_row_inline(
        &self,
        renderer: &mut iced::Renderer,
        line: &str,
        spans: Option<&[HighlightSpan]>,
        origin: Point,
        geo: &Geo,
        row_layout: &RowLayout<'_>,
        dim: Color,
        text_color: Color,
        clip: Rectangle,
    ) {
        let advance = geo.advance();
        let seg = |renderer: &mut iced::Renderer, text: &str, start_col: u32, color: Color| {
            if text.is_empty() || row_layout.glyph_hidden(start_col) {
                return;
            }
            let x = origin.x + row_layout.display_cell(start_col) as f32 * advance;
            self.draw_line(renderer, expand_tabs(text), Point::new(x, origin.y), color, Alignment::Left, clip);
        };
        match spans {
            Some(spans) if !spans.is_empty() => {
                for s in spans {
                    let fg = s.style.fg;
                    let text = &line[s.range.start as usize..s.range.end as usize];
                    seg(renderer, text, s.range.start, Color::from_rgb8(fg.r, fg.g, fg.b));
                }
            }
            _ => {
                let mut cursor = 0u32;
                for chip in row_layout.chips() {
                    seg(renderer, &line[cursor as usize..=chip.open_col as usize], cursor, text_color);
                    cursor = chip.close_col;
                }
                seg(renderer, &line[cursor as usize..], cursor, text_color);
            }
        }
        // The `…` chip between each pair's brackets, centered on the collapsed gap.
        // Where the selection wash already backs the chip, the idle pill is
        // suppressed: it is `POPUP_SELECT` (darker than the wash), so on a selected
        // row it would island as a dark box ringed by a lighter-wash margin — the
        // "space to the left of the `…`" artifact. The `…` still renders on the
        // wash, exactly like any other selected glyph.
        let row_start = row_layout.row_start();
        for chip in row_layout.chips() {
            let mid = origin.x + chip.center * advance;
            // On a selected row the wash already backs the chip, so the idle pill
            // (`POPUP_SELECT`, darker than the wash) is dropped — else it islands as
            // a dark box ringed by a lighter-wash margin (the "space to the left of
            // the `…`" artifact). Without the pill's contrast the `dim` dots would
            // vanish into the wash, so they render at full glyph color there.
            let dots = if self.range_selected(row_start + chip.open_col + 1, row_start + chip.close_col) {
                text_color
            } else {
                fill_rounded(renderer, geo.chip_pill(mid, origin.y), POPUP_SELECT, CHIP_PILL_RADIUS);
                dim
            };
            self.draw_line(renderer, "…".to_string(), Point::new(mid, origin.y), dots, Alignment::Center, clip);
        }
    }

    /// The screen rect of a collapsed fold's `…` chip — an inline `[ … ]` or a block
    /// header's `{ … }`. The single source for the plain-hover expand affordance:
    /// the highlight, the finger, the click, and the hover preview all measure against
    /// it, so they agree. `None` if `opener` isn't collapsed or its line is itself
    /// hidden inside an outer collapsed fold.
    fn collapsed_chip_rect(&self, fold_map: &FoldMap, geo: &Geo, opener: u32) -> Option<Rectangle> {
        if !self.doc.folds().is_folded(opener) {
            return None;
        }
        let buffer = self.doc.buffer();
        let b = self.doc.brackets().at(opener).filter(|b| b.open)?;
        let close = b.partner?;
        let header = buffer.offset_to_point(opener).row;
        let last = buffer.offset_to_point(close).row;
        if fold_map.is_folded(BufferRow(header)) {
            return None;
        }
        let code_left = geo.code_left();
        let y = geo.row_y(fold_map.to_display_row(BufferRow(header)));
        if last > header {
            // Block: the `{ … }` placeholder — the header's `{` through the closing
            // `}`. End at the CLOSER's cell, NOT `hl.width()` (the whole tail line):
            // text typed after the `}` on the last line is outside the collapsed
            // region and must not join the fold's hover / expand target.
            let hl = fold_map.header_layout(buffer, BufferRow(header), TAB)?;
            let x0 = geo.cell_x((hl.head_cells() as f32 - 1.0).max(0.0)) - 2.0;
            let last_start = buffer.point_to_offset(BufPoint { row: last, col: 0 });
            let close_cell = hl.tail_col_cell(close - last_start).unwrap_or_else(|| hl.tail_cell());
            let x1 = geo.cell_x(close_cell as f32 + 1.0) + 2.0;
            Some(Rectangle {
                x: x0.max(code_left + 1.0),
                y: y + 1.0,
                width: (x1 - x0).max(geo.advance()),
                height: geo.line_h() - 2.0,
            })
        } else {
            // Inline: the `[ … ]` bracket span. Only a *root* inline fold has a chip;
            // one nested inside another collapsed inline pair is hidden in that pair's
            // chip (reduced away by `FoldMap`), so it has no rect of its own.
            // `inline_fold_at` is an O(log F) offset-keyed descent, so a hover /
            // preview frame costs O(log F), never an O(F) decode of every inline fold.
            fold_map.inline_fold_at(opener)?;
            let xo = self.offset_screen_x(fold_map, geo, opener);
            let xc = self.offset_screen_x(fold_map, geo, close);
            Some(geo.inline_halo(xo.min(xc), xo.max(xc), y))
        }
    }

    /// The screen rect of a collapsed fold's `…` pill — the exact box the text
    /// pass paints (a block's, centered in the placeholder gap; an inline
    /// chip's, at its center), for the plain-hover highlight. All cells come
    /// from the same owners the painter reads (`gap_center` / `chips` /
    /// [`Geo::chip_pill`]). `None` when the fold has no on-screen pill (not
    /// collapsed, hidden inside an outer fold, or nested away).
    fn chip_pill_rect(&self, fold_map: &FoldMap, geo: &Geo, opener: u32) -> Option<Rectangle> {
        if !self.doc.folds().is_folded(opener) {
            return None;
        }
        let buffer = self.doc.buffer();
        let b = self.doc.brackets().at(opener).filter(|b| b.open)?;
        let close = b.partner?;
        let header = buffer.offset_to_point(opener).row;
        let last = buffer.offset_to_point(close).row;
        if fold_map.is_folded(BufferRow(header)) {
            return None;
        }
        let y = geo.row_y(fold_map.to_display_row(BufferRow(header)));
        let mid = if last > header {
            geo.cell_x(fold_map.header_layout(buffer, BufferRow(header), TAB)?.gap_center())
        } else {
            let row_start = buffer.point_to_offset(BufPoint { row: header, col: 0 });
            let layout = fold_map.row_layout(buffer, BufferRow(header), TAB);
            let chip = layout.chips().find(|c| c.open_col == opener - row_start)?;
            geo.cell_x(chip.center)
        };
        Some(geo.chip_pill(mid, y))
    }

    /// The collapsed fold whose `…` chip is under `pos` — the plain-hover expand
    /// target. Chips are disjoint on screen, so the first match wins.
    fn collapsed_chip_at(&self, geo: &Geo, pos: Point) -> Option<u32> {
        let fold_map = self.fold_map();
        // A `…` chip renders only on its fold's header (opener) row, so only a
        // pair headed on the pointer's display row can sit under `pos`. Resolve
        // that one buffer row and test just the pairs headed there
        // (O(brackets on the row)) — never a scan of every fold in the document,
        // which would build one `HeaderLayout` (rope reads) per fold on every
        // `mouse_interaction` frame and stall a collapsed large file. Chips are
        // disjoint on screen, so the first hit wins regardless of order.
        let row = fold_map.to_buffer_row(fold_map.display_row_at(geo.rows_from_top(pos.y))).0;
        self.doc
            .foldable_pairs_in_rows(row..row + 1)
            .into_iter()
            .map(|(open, ..)| open)
            .find(|&o| self.collapsed_chip_rect(&fold_map, geo, o).is_some_and(|r| r.contains(pos)))
    }

    /// Draw the hover preview of a collapsed fold's hidden content: a floating
    /// panel, anchored under the collapsed line's chip, showing the folded interior
    /// (a block's hidden rows, or an inline pair's own expanded line) in the
    /// editor's own syntax colors — a folded-region hover like mainstream editors'.
    /// `opener` is the
    /// collapsed pair's opening-bracket offset; a no-op if it is not a live
    /// collapsed fold (e.g. expanded since the preview armed) or its header is
    /// scrolled/folded out of view.
    fn draw_fold_preview(&self, renderer: &mut iced::Renderer, geo: &Geo, opener: u32, text_color: Color) {
        let (bounds, advance, line_h) = (geo.bounds(), geo.advance(), geo.line_h());
        if !self.doc.folds().is_folded(opener) {
            return; // expanded since the preview armed — nothing hidden
        }
        let buffer = self.doc.buffer();
        let Some(b) = self.doc.brackets().at(opener).filter(|b| b.open) else { return };
        let Some(close) = b.partner else { return };
        let header = buffer.offset_to_point(opener).row;
        let last = buffer.offset_to_point(close).row;
        let fold_map = self.fold_map();
        if fold_map.is_folded(BufferRow(header)) {
            return; // the collapsed line is itself hidden inside an outer fold
        }
        // Rows to show: a block's hidden interior (`header+1..=last`), or the inline
        // pair's own row expanded. Only the first `MAX_ROWS` are ever read, so
        // `take` them — the interior of a document-scale collapsed block must not
        // materialize an O(hidden-rows) Vec on every repaint of the preview.
        const MAX_ROWS: usize = 18;
        let total = if last > header { (last - header) as usize } else { 1 };
        let rows: Vec<u32> = if last > header {
            (header + 1..=last).take(MAX_ROWS).collect()
        } else {
            vec![header]
        };
        let shown = rows.len(); // == total.min(MAX_ROWS)
        let more = total - shown;
        // Strip the shared leading indent so the preview reads compact.
        let mut min_cells = u32::MAX;
        let mut max_cells = 0u32;
        for &r in &rows[..shown] {
            let l = buffer.line(r);
            if l.trim().is_empty() {
                continue;
            }
            min_cells = min_cells.min(line_indent_cells(&l));
            max_cells = max_cells.max(display_map::expand(&l, l.len() as u32, TAB));
        }
        let min_cells = if min_cells == u32::MAX { 0 } else { min_cells };
        let content_cells = max_cells.saturating_sub(min_cells);
        let pad = 8.0_f32;
        let code_w = (bounds.width - geo.gutter() - SCROLLBAR_WIDTH).max(advance * 8.0);
        let width = (content_cells as f32 * advance + 2.0 * pad).clamp(advance * 8.0, code_w);
        let height = (shown + usize::from(more > 0)) as f32 * line_h + 2.0 * pad;
        // Anchor at the `…` chip being previewed — the SAME rect the hover
        // wash, finger cursor, and expand click measure against, so the panel
        // opens under the thing the pointer is on (a mid-line inline chip
        // included, not the row's end). Flip above when it would overflow the
        // bottom, and clamp horizontally.
        let Some(chip) = self.collapsed_chip_rect(&fold_map, geo, opener) else {
            return; // no chip on screen (nested away) ⇒ nothing to anchor to
        };
        let code_left = geo.code_left();
        let x = chip.x.clamp(code_left, (bounds.x + bounds.width - width - 4.0).max(code_left));
        let line_top = geo.row_y(fold_map.to_display_row(BufferRow(header)));
        let below = line_top + line_h + 2.0;
        let y = if below + height <= bounds.y + bounds.height { below } else { (line_top - height - 2.0).max(bounds.y) };
        let rect = Rectangle { x, y, width, height };
        fill_panel(renderer, rect, POPUP_SURFACE, POPUP_BORDER, 4.0);
        // Content lines, shifted left by the stripped indent, in their syntax colors.
        let sx = rect.x + pad - min_cells as f32 * advance;
        for (i, &r) in rows[..shown].iter().enumerate() {
            let ry = rect.y + pad + i as f32 * line_h;
            match self.doc.highlight_line_spans(r) {
                Some(spans) if !spans.is_empty() => self.draw_spans(renderer, &buffer.line(r), spans, Point::new(sx, ry), advance, rect),
                _ => {
                    let line = buffer.line(r);
                    let l = line.trim_start();
                    if !l.is_empty() {
                        self.draw_line(renderer, expand_tabs(l), Point::new(rect.x + pad, ry), text_color, Alignment::Left, rect);
                    }
                }
            }
        }
        if more > 0 {
            let ry = rect.y + pad + shown as f32 * line_h;
            let msg = format!("… {more} more line{}", if more == 1 { "" } else { "s" });
            self.draw_line(renderer, msg, Point::new(rect.x + pad, ry), Color { a: 0.6, ..text_color }, Alignment::Left, rect);
        }
    }

    /// Draw one line as its colored highlight spans (each a byte range within the
    /// line, positioned by its start cell). Bold/italic deferred; fg only.
    fn draw_spans(
        &self,
        renderer: &mut iced::Renderer,
        line: &str,
        spans: &[HighlightSpan],
        origin: Point,
        advance: f32,
        clip: Rectangle,
    ) {
        for span in spans {
            let text = &line[span.range.start as usize..span.range.end as usize];
            if text.is_empty() {
                continue;
            }
            let cell = display_map::expand(line, span.range.start, TAB);
            let x = origin.x + cell as f32 * advance;
            let fg = span.style.fg;
            self.draw_line(
                renderer,
                expand_tabs(text),
                Point::new(x, origin.y),
                Color::from_rgb8(fg.r, fg.g, fg.b),
                Alignment::Left,
                clip,
            );
        }
    }

    /// Draw one line of text at `position` using the configured font/size.
    fn draw_line(
        &self,
        renderer: &mut iced::Renderer,
        content: String,
        position: Point,
        color: Color,
        align: Alignment,
        clip: Rectangle,
    ) {
        renderer.fill_text(
            Text {
                content,
                bounds: Size::new(f32::INFINITY, self.line_height),
                size: Pixels(self.size),
                line_height: LineHeight::Absolute(Pixels(self.line_height)),
                font: self.font,
                align_x: align,
                align_y: Vertical::Top,
                shaping: Shaping::Basic,
                wrapping: Wrapping::None,
            },
            position,
            color,
            clip,
        );
    }

    /// Draw a single [`CODICON`](crate::CODICON) glyph (a fold chevron),
    /// horizontally centered in the `width`-wide column whose top-left is `origin`
    /// and vertically centered in the row (same vertical rhythm as
    /// [`Self::draw_line`]). The host must have loaded
    /// [`CODICON_FONT`](crate::CODICON_FONT); if it didn't, iced renders a
    /// missing-glyph box rather than the chevron.
    fn draw_icon(
        &self,
        renderer: &mut iced::Renderer,
        glyph: char,
        origin: Point,
        width: f32,
        color: Color,
        clip: Rectangle,
    ) {
        renderer.fill_text(
            Text {
                content: glyph.to_string(),
                bounds: Size::new(f32::INFINITY, self.line_height),
                size: Pixels(self.size),
                line_height: LineHeight::Absolute(Pixels(self.line_height)),
                font: crate::CODICON,
                align_x: Alignment::Center,
                align_y: Vertical::Top,
                shaping: Shaping::Basic,
                wrapping: Wrapping::None,
            },
            Point::new(origin.x + width / 2.0, origin.y),
            color,
            clip,
        );
    }

    /// Left-aligned, unwrapped run in an explicit `font` — the hover's bold /
    /// code / plain markdown segments (`draw_line` fixed to `self.font`).
    fn draw_run(
        &self,
        renderer: &mut iced::Renderer,
        content: String,
        position: Point,
        color: Color,
        font: Font,
        clip: Rectangle,
    ) {
        renderer.fill_text(
            Text {
                content,
                bounds: Size::new(f32::INFINITY, self.line_height),
                size: Pixels(self.size),
                line_height: LineHeight::Absolute(Pixels(self.line_height)),
                font,
                align_x: Alignment::Left,
                align_y: Vertical::Top,
                shaping: Shaping::Basic,
                wrapping: Wrapping::None,
            },
            position,
            color,
            clip,
        );
    }
}

/// Expand tabs to spaces for display (the caret math uses the display map, so
/// they agree).
fn expand_tabs(line: &str) -> String {
    if !line.contains('\t') {
        return line.to_owned();
    }
    let mut out = String::with_capacity(line.len());
    let mut cell = 0u32;
    for ch in line.chars() {
        if ch == '\t' {
            let w = display_map::tab_width(cell, TAB);
            for _ in 0..w {
                out.push(' ');
            }
            cell += w;
        } else {
            out.push(ch);
            cell += 1;
        }
    }
    out
}

/// Map a key event to an [`Action`], or `None` if it isn't an editor input.
/// The glyph a numpad *character* key carries in `text`, or `None` for a
/// non-numpad key, a numpad key with no text (NumLock off), or control text
/// (NumpadEnter → `\r`, left for the `Named::Enter` path). Works around the
/// NumLock-on quirk where numpad keys report a navigation logical key.
fn numpad_text(physical: &iced::keyboard::key::Physical, text: Option<&str>) -> Option<char> {
    use iced::keyboard::key::{Code, Physical};
    let is_numpad = matches!(
        physical,
        Physical::Code(
            Code::Numpad0
                | Code::Numpad1
                | Code::Numpad2
                | Code::Numpad3
                | Code::Numpad4
                | Code::Numpad5
                | Code::Numpad6
                | Code::Numpad7
                | Code::Numpad8
                | Code::Numpad9
                | Code::NumpadDecimal
                | Code::NumpadComma
                | Code::NumpadAdd
                | Code::NumpadSubtract
                | Code::NumpadMultiply
                | Code::NumpadDivide
                | Code::NumpadEqual
        )
    );
    if !is_numpad {
        return None;
    }
    let ch = text?.chars().next()?;
    (!ch.is_control()).then_some(ch)
}

fn interpret_key(key: &Key, text: Option<&str>, mods: Modifiers) -> Option<Action> {
    let extend = mods.shift();
    let mv = |motion| Some(Action::Move { motion, extend });
    // Column (box) selection needs all three of Ctrl+Shift+Alt — checked before
    // the Ctrl+Arrow word-motions below, which Ctrl+Shift+Alt+Left would else
    // match on `control()` alone.
    let column = mods.control() && mods.shift() && mods.alt();
    match key {
        Key::Named(Named::ArrowUp) if column => Some(Action::ColumnSelect(ColumnDir::Up)),
        Key::Named(Named::ArrowDown) if column => Some(Action::ColumnSelect(ColumnDir::Down)),
        Key::Named(Named::ArrowLeft) if column => Some(Action::ColumnSelect(ColumnDir::Left)),
        Key::Named(Named::ArrowRight) if column => Some(Action::ColumnSelect(ColumnDir::Right)),
        // Ctrl+Alt+↑/↓ add a caret above/below every caret — after the
        // three-modifier column arms, before the plain arrows.
        Key::Named(Named::ArrowUp) if mods.control() && mods.alt() && !mods.shift() => {
            Some(Action::AddCaretVertical { down: false })
        }
        Key::Named(Named::ArrowDown) if mods.control() && mods.alt() && !mods.shift() => {
            Some(Action::AddCaretVertical { down: true })
        }
        // Shift+Alt+←/→ shrink/expand the selection structurally — the
        // standard bindings; without Ctrl, which is column-select above.
        Key::Named(Named::ArrowRight) if mods.alt() && !mods.control() && mods.shift() => {
            Some(Action::ExpandSelection)
        }
        Key::Named(Named::ArrowLeft) if mods.alt() && !mods.control() && mods.shift() => {
            Some(Action::ShrinkSelection)
        }
        // Alt+↑/↓ move the line, Shift+Alt+↑/↓ copy it (both without Ctrl, which
        // is column-select above). Checked before the plain arrow motions.
        Key::Named(Named::ArrowUp) if mods.alt() && !mods.control() && !mods.shift() => {
            Some(Action::MoveLine { down: false })
        }
        Key::Named(Named::ArrowDown) if mods.alt() && !mods.control() && !mods.shift() => {
            Some(Action::MoveLine { down: true })
        }
        Key::Named(Named::ArrowUp) if mods.alt() && !mods.control() && mods.shift() => {
            Some(Action::CopyLine { down: false })
        }
        Key::Named(Named::ArrowDown) if mods.alt() && !mods.control() && mods.shift() => {
            Some(Action::CopyLine { down: true })
        }
        Key::Named(Named::ArrowLeft) if mods.control() => mv(Motion::WordLeft),
        Key::Named(Named::ArrowRight) if mods.control() => mv(Motion::WordRight),
        Key::Named(Named::ArrowLeft) => mv(Motion::Left),
        Key::Named(Named::ArrowRight) => mv(Motion::Right),
        Key::Named(Named::ArrowUp) => mv(Motion::Up),
        Key::Named(Named::ArrowDown) => mv(Motion::Down),
        // Ctrl+Home/End jump to the document ends (Shift extends via `extend`);
        // checked before the plain Home/End smart-line motions below.
        Key::Named(Named::Home) if mods.control() => mv(Motion::DocStart),
        Key::Named(Named::End) if mods.control() => mv(Motion::DocEnd),
        Key::Named(Named::Home) => mv(Motion::LineStart),
        Key::Named(Named::End) => mv(Motion::LineEnd),
        // Ctrl+Backspace/Delete delete by word; before the plain arms.
        Key::Named(Named::Backspace) if mods.control() => Some(Action::DeleteWordBack),
        Key::Named(Named::Delete) if mods.control() => Some(Action::DeleteWordForward),
        Key::Named(Named::Backspace) => Some(Action::Backspace),
        Key::Named(Named::Delete) => Some(Action::Delete),
        // Ctrl+Enter opens a line below, Ctrl+Shift+Enter above — without
        // splitting the current line; before the plain Enter arm. Alt+Enter
        // is the app's find-bar chord (select all matches), so the editor
        // deliberately ignores it — no newline alongside the app gesture.
        Key::Named(Named::Enter) if mods.control() && !mods.alt() => {
            Some(Action::InsertLine { down: !mods.shift() })
        }
        Key::Named(Named::Enter) if !mods.alt() => Some(Action::Enter),
        Key::Named(Named::Enter) => None,
        Key::Named(Named::Tab) if mods.shift() => Some(Action::Outdent),
        Key::Named(Named::Tab) => Some(Action::Tab),
        Key::Named(Named::Space) => Some(Action::Type(' ')),
        Key::Named(Named::Escape) => Some(Action::Collapse),
        // F8 / Shift+F8 jump to the next/previous diagnostic.
        Key::Named(Named::F8) => Some(Action::NextDiagnostic { forward: !mods.shift() }),
        // Ctrl+D add-next-occurrence. The `!alt` guard keeps Ctrl+Alt
        // (AltGr) from ever triggering the gesture.
        Key::Character(c) if mods.control() && !mods.alt() && c.as_str() == "d" => {
            Some(Action::AddNextOccurrence)
        }
        // Ctrl+Shift+L select-all-occurrences (Shift makes the logical "L").
        Key::Character(c)
            if mods.control() && mods.shift() && !mods.alt() && c.as_str().eq_ignore_ascii_case("l") =>
        {
            Some(Action::SelectAllOccurrences)
        }
        // Ctrl+A select-all (same `!alt` guard).
        Key::Character(c) if mods.control() && !mods.alt() && c.as_str() == "a" => {
            Some(Action::SelectAll)
        }
        // Ctrl+/ toggle line comment (same `!alt` AltGr guard).
        Key::Character(c) if mods.control() && !mods.alt() && c.as_str() == "/" => {
            Some(Action::ToggleComment)
        }
        // Ctrl+Shift+K delete line (Shift makes the logical char "K").
        Key::Character(c)
            if mods.control() && mods.shift() && !mods.alt() && c.as_str().eq_ignore_ascii_case("k") =>
        {
            Some(Action::DeleteLine)
        }
        // Ctrl+Shift+\ jump to matching bracket — on US layouts Shift makes
        // the logical char `|`, so accept both (like the fold chords).
        Key::Character(c) if mods.control() && !mods.alt() && matches!(c.as_str(), "\\" | "|") => {
            Some(Action::JumpToBracket)
        }
        // Undo / redo: Ctrl+Z undoes, Ctrl+Shift+Z and Ctrl+Y redo. Shift makes
        // the logical char "Z", so fold case on the z arm.
        Key::Character(c) if mods.control() && !mods.alt() && c.as_str().eq_ignore_ascii_case("z") => {
            Some(if mods.shift() { Action::Redo } else { Action::Undo })
        }
        Key::Character(c) if mods.control() && !mods.alt() && c.as_str().eq_ignore_ascii_case("y") => {
            Some(Action::Redo)
        }
        _ => {
            if (mods.control() && !mods.alt()) || mods.command() {
                return None;
            }
            let ch = text?.chars().next()?;
            (!ch.is_control()).then_some(Action::Type(ch))
        }
    }
}

/// The display-cell width of a line's leading whitespace (its indentation),
/// with tabs expanded — the column an indent guide one level in sits at.
fn line_indent_cells(line: &str) -> u32 {
    let ws = (line.len() - line.trim_start().len()) as u32;
    display_map::expand(line, ws, TAB)
}

/// Completion-item label color by kind — one accent per kind, so the
/// popup's labels are colored by what they complete to (a theme may override).
fn kind_color(kind: CompletionKind) -> Color {
    match kind {
        CompletionKind::Keyword | CompletionKind::Construct => Color::from_rgb8(0xff, 0x61, 0x8d), // pink
        CompletionKind::Type => Color::from_rgb8(0x66, 0xd9, 0xef), // cyan
        CompletionKind::Param | CompletionKind::Field => Color::from_rgb8(0xfd, 0x97, 0x1f), // orange
        CompletionKind::Value | CompletionKind::Event => Color::from_rgb8(0xa6, 0xe2, 0x2e), // green
        CompletionKind::Symbol | CompletionKind::Method => Color::from_rgb8(0xae, 0x81, 0xff), // purple
    }
}

fn severity_color(sev: Severity) -> Color {
    match sev {
        Severity::Error => Color::from_rgb8(0xff, 0x61, 0x69),   // red
        Severity::Warning => Color::from_rgb8(0xfc, 0xe5, 0x66), // yellow
        Severity::Info => Color::from_rgb8(0x5a, 0xd4, 0xe6),    // blue
        Severity::Hint => Color::from_rgb8(0x94, 0x8a, 0xe3),    // muted purple
    }
}

/// Rasterize a diagnostic squiggle from `x0` to `x1` at vertical `baseline` into
/// 1-px vertical spans. The wave is a sine anchored at `x0`
/// (half-period [`SQUIGGLE_HALF_PERIOD`], amplitude [`SQUIGGLE_AMPLITUDE`]); each
/// span covers one horizontal pixel and is tall enough to bridge the wave's rise
/// to the next sample, so the [`SQUIGGLE_STROKE`]-thick line stays continuous
/// through the steep zero-crossings. Pure — the geometry is unit-tested without
/// a renderer, then the widget paints the emitted spans with `fill`.
fn squiggle_spans(x0: f32, x1: f32, baseline: f32, mut emit: impl FnMut(Rectangle)) {
    use std::f32::consts::PI;
    let wave = |x: f32| SQUIGGLE_AMPLITUDE * (PI * (x - x0) / SQUIGGLE_HALF_PERIOD).sin();
    let mut x = x0;
    while x < x1 {
        let next = (x + 1.0).min(x1);
        let (ya, yb) = (baseline + wave(x), baseline + wave(next));
        let top = ya.min(yb) - SQUIGGLE_STROKE / 2.0;
        let height = (ya - yb).abs() + SQUIGGLE_STROKE;
        emit(Rectangle { x, y: top, width: next - x, height });
        x = next;
    }
}

/// Fill a rectangle with a flat color.
fn fill(renderer: &mut iced::Renderer, bounds: Rectangle, color: Color) {
    use iced::advanced::Renderer as _;
    renderer.fill_quad(
        renderer::Quad { bounds, border: border::rounded(0), shadow: Shadow::default(), snap: true },
        color,
    );
}

/// Inline markdown styling for a hover run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum MdStyle {
    Plain,
    Bold,
    Code,
}

/// Split one markdown line into styled runs, consuming the `**bold**` and
/// `` `code` `` markers. A minimal inline subset (no nesting) — enough for the
/// hover's spec-derived docs; unmatched markers just toggle back at line end.
fn parse_md_runs(line: &str) -> Vec<(String, MdStyle)> {
    let mut runs = Vec::new();
    let mut cur = String::new();
    let mut style = MdStyle::Plain;
    let mut push = |cur: &mut String, style: MdStyle| {
        if !cur.is_empty() {
            runs.push((std::mem::take(cur), style));
        }
    };
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' if chars.peek() == Some(&'*') => {
                chars.next(); // second '*'
                push(&mut cur, style);
                style = if style == MdStyle::Bold { MdStyle::Plain } else { MdStyle::Bold };
            }
            '`' => {
                push(&mut cur, style);
                style = if style == MdStyle::Code { MdStyle::Plain } else { MdStyle::Code };
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        runs.push((cur, style));
    }
    runs
}

/// Coalesce a per-char styled sequence into `(String, style)` segments.
fn coalesce_chars(chars: Vec<(char, MdStyle)>) -> Vec<(String, MdStyle)> {
    let mut segs: Vec<(String, MdStyle)> = Vec::new();
    for (c, s) in chars {
        match segs.last_mut() {
            Some((t, st)) if *st == s => t.push(c),
            _ => segs.push((c.to_string(), s)),
        }
    }
    segs
}

/// Word-wrap one source line's styled runs into visual lines that each fit in
/// `max_cells` (monospace ⇒ 1 cell = 1 char). Breaks at spaces (dropped at the
/// break); a word longer than the width hard-breaks. Per-char style rides along.
fn wrap_runs(runs: &[(String, MdStyle)], max_cells: usize) -> Vec<Vec<(String, MdStyle)>> {
    let max = max_cells.max(1);
    let flat: Vec<(char, MdStyle)> =
        runs.iter().flat_map(|(t, s)| t.chars().map(move |c| (c, *s))).collect();
    let mut lines: Vec<Vec<(char, MdStyle)>> = Vec::new();
    let mut cur: Vec<(char, MdStyle)> = Vec::new();
    let trim = |line: &mut Vec<(char, MdStyle)>| {
        while line.last().map(|&(c, _)| c) == Some(' ') {
            line.pop();
        }
    };
    let mut i = 0;
    while i < flat.len() {
        if flat[i].0 == ' ' {
            if cur.len() < max {
                cur.push(flat[i]); // spaces only count while the line has room
            }
            i += 1;
            continue;
        }
        let start = i;
        while i < flat.len() && flat[i].0 != ' ' {
            i += 1;
        }
        let word = &flat[start..i];
        if word.len() > max {
            // Hard-break an over-long word across lines.
            for &ch in word {
                if cur.len() >= max {
                    lines.push(std::mem::take(&mut cur));
                }
                cur.push(ch);
            }
        } else {
            if cur.len() + word.len() > max {
                trim(&mut cur);
                lines.push(std::mem::take(&mut cur));
            }
            cur.extend_from_slice(word);
        }
    }
    trim(&mut cur);
    lines.push(cur);
    lines.into_iter().map(coalesce_chars).collect()
}

/// Geometry + wrapped content for the hover popup — computed once and shared by
/// `draw_hover` and the wheel-scroll handler (one source of truth for the box).
struct HoverLayout {
    /// The whole popup box.
    rect: Rectangle,
    /// Word-wrapped visual lines (styled runs).
    lines: Vec<Vec<(String, MdStyle)>>,
    /// How many lines are visible at once (≤ `HOVER_MAX_VISIBLE`).
    visible: usize,
    /// Max vertical scroll offset in px (`(total − visible) · line_h`), 0 if it fits.
    max_scroll: f32,
    /// Inner padding (x, y).
    pad_x: f32,
    pad_y: f32,
    /// Whether the content is taller than the box (⇒ scrollbar + wheel scroll).
    overflow: bool,
}

/// A filled rounded rect with no border or shadow — the inline-code pill.
fn fill_rounded(renderer: &mut iced::Renderer, bounds: Rectangle, color: Color, radius: f32) {
    use iced::advanced::Renderer as _;
    renderer.fill_quad(
        renderer::Quad { bounds, border: border::rounded(radius), shadow: Shadow::default(), snap: true },
        color,
    );
}

/// Draw a floating panel: a rounded quad with a colored 1-px border and a soft
/// drop shadow — the shared hover/popup surface.
fn fill_panel(renderer: &mut iced::Renderer, bounds: Rectangle, fill: Color, border_color: Color, radius: f32) {
    use iced::advanced::Renderer as _;
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: border::rounded(radius).color(border_color).width(1.0),
            shadow: Shadow {
                color: Color::from_rgba8(0, 0, 0, 0.36),
                offset: Vector::new(0.0, 2.0),
                blur_radius: 8.0,
            },
            snap: true,
        },
        fill,
    );
}

/// Draw a rectangle outline: a transparent-fill quad with a `width`-px colored
/// border (the matching-bracket box).
fn fill_border(renderer: &mut iced::Renderer, bounds: Rectangle, color: Color, width: f32) {
    use iced::advanced::Renderer as _;
    renderer.fill_quad(
        renderer::Quad {
            bounds,
            border: border::rounded(2).color(color).width(width),
            shadow: Shadow::default(),
            snap: true,
        },
        Color::TRANSPARENT,
    );
}

/// Stroke a rounded rectangle's perimeter with a dashed line — the Ctrl+hover
/// collapse box. iced quad borders are solid-only, so the straight edges are
/// laid down as short filled spans (inset by `radius` at each end) and the four
/// corners are traced as short arcs of 1-px dots. `dash`/`gap`/`w` are the run,
/// the gap, and the stroke width; `radius` the corner radius.
#[allow(clippy::too_many_arguments)] // a dashed rounded stroke needs all of them
fn stroke_dashed_rounded_rect(renderer: &mut iced::Renderer, rect: Rectangle, color: Color, radius: f32, dash: f32, gap: f32, w: f32) {
    if rect.width <= 0.0 || rect.height <= 0.0 {
        return;
    }
    let (x, y, ww, hh) = (rect.x, rect.y, rect.width, rect.height);
    let r = radius.min(ww * 0.5).min(hh * 0.5).max(0.0);
    let step = (dash + gap).max(0.1);
    // Top & bottom edges — horizontal dashes, inset by `r` so the corners are free.
    let mut sx = x + r;
    while sx < x + ww - r {
        let dw = dash.min(x + ww - r - sx);
        fill(renderer, Rectangle { x: sx, y, width: dw, height: w }, color);
        fill(renderer, Rectangle { x: sx, y: y + hh - w, width: dw, height: w }, color);
        sx += step;
    }
    // Left & right edges — vertical dashes, likewise inset.
    let mut sy = y + r;
    while sy < y + hh - r {
        let dh = dash.min(y + hh - r - sy);
        fill(renderer, Rectangle { x, y: sy, width: w, height: dh }, color);
        fill(renderer, Rectangle { x: x + ww - w, y: sy, width: w, height: dh }, color);
        sy += step;
    }
    // Rounded corners — trace each quarter-arc with 1-px dots (centre, start→end).
    if r > 0.5 {
        use core::f32::consts::{FRAC_PI_2, PI};
        let half = w * 0.5;
        let arcs = [
            (x + ww - r, y + r, -FRAC_PI_2, 0.0),    // top-right
            (x + ww - r, y + hh - r, 0.0, FRAC_PI_2), // bottom-right
            (x + r, y + hh - r, FRAC_PI_2, PI),       // bottom-left
            (x + r, y + r, PI, PI + FRAC_PI_2),       // top-left
        ];
        for (cx, cy, a0, a1) in arcs {
            let steps = (r * (a1 - a0).abs()).ceil().max(1.0) as u32;
            for k in 0..=steps {
                let a = a0 + (a1 - a0) * (k as f32 / steps as f32);
                fill(renderer, Rectangle { x: cx + r * a.cos() - half, y: cy + r * a.sin() - half, width: w, height: w }, color);
            }
        }
    }
}

impl<'a, Message: 'a> From<Editor<'a, Message>>
    for Element<'a, Message, iced::Theme, iced::Renderer>
{
    fn from(editor: Editor<'a, Message>) -> Self {
        Element::new(editor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iced::advanced::widget::operation;

    #[test]
    fn digit_count_is_right() {
        assert_eq!(digit_count(0), 1);
        assert_eq!(digit_count(9), 1);
        assert_eq!(digit_count(10), 2);
        assert_eq!(digit_count(1000), 4);
    }

    #[test]
    fn visible_selection_span_matches_a_brute_force_filter() {
        use scrive_core::SelectionSet;
        // A sorted, disjoint set: carets and ranges. from_ranges normalizes.
        let ranges = [(0u32, 3u32), (10, 10), (20, 25), (40, 40), (55, 60), (80, 80)];
        let set = SelectionSet::from_ranges(&ranges, 0);
        let sels = set.all();
        for vis_start in 0..90u32 {
            for vis_end in vis_start..90u32 {
                let span = visible_selection_span(sels, vis_start, vis_end);
                // A selection intersects the window iff end ≥ start_of_window and
                // start ≤ end_of_window; the intersecting set is contiguous.
                let brute: Vec<usize> = (0..sels.len())
                    .filter(|&i| sels[i].end() >= vis_start && sels[i].start() <= vis_end)
                    .collect();
                let got: Vec<usize> = span.clone().collect();
                assert_eq!(got, brute, "window [{vis_start}, {vis_end}]");
            }
        }
    }

    #[test]
    fn range_covered_by_matches_a_brute_force_scan() {
        use scrive_core::SelectionSet;
        // Carets and ranges, sorted + disjoint (from_ranges normalizes).
        let ranges = [(0u32, 3u32), (10, 10), (20, 25), (40, 40), (55, 60)];
        let set = SelectionSet::from_ranges(&ranges, 0);
        let sels = set.all();
        for start in 0..65u32 {
            for end in start..65u32 {
                let got = range_covered_by(sels, start, end);
                // Covered iff some single selection spans the whole range.
                let brute = sels.iter().any(|s| s.start() <= start && s.end() >= end);
                assert_eq!(got, brute, "range [{start}, {end}]");
            }
        }
    }

    #[test]
    fn numpad_text_prefers_glyph_over_nav() {
        use iced::keyboard::key::{Code, Physical};
        // NumLock on: a numpad key carries its glyph in `text` → type it, even
        // though its logical key is a navigation key.
        assert_eq!(numpad_text(&Physical::Code(Code::Numpad2), Some("2")), Some('2'));
        assert_eq!(numpad_text(&Physical::Code(Code::NumpadDecimal), Some(".")), Some('.'));
        assert_eq!(numpad_text(&Physical::Code(Code::NumpadAdd), Some("+")), Some('+'));
        // NumLock off: no text ⇒ None ⇒ the navigation key stands.
        assert_eq!(numpad_text(&Physical::Code(Code::Numpad2), None), None);
        // A main-row digit is not a numpad key ⇒ never intercepted here.
        assert_eq!(numpad_text(&Physical::Code(Code::Digit2), Some("2")), None);
        // NumpadEnter is excluded ⇒ left for the Named::Enter → newline path.
        assert_eq!(numpad_text(&Physical::Code(Code::NumpadEnter), Some("\r")), None);
    }

    #[test]
    fn wrap_runs_wraps_at_word_boundaries() {
        use MdStyle::*;
        let text = |line: &Vec<(String, MdStyle)>| line.iter().map(|(t, _)| t.as_str()).collect::<String>();
        // Greedy word wrap at width 8.
        let w = wrap_runs(&[("hello world foo".to_string(), Plain)], 8);
        assert_eq!(w.iter().map(text).collect::<Vec<_>>(), vec!["hello", "world", "foo"]);
        // A word longer than the width hard-breaks.
        let h = wrap_runs(&[("verylongword".to_string(), Plain)], 4);
        assert_eq!(h.iter().map(text).collect::<Vec<_>>(), vec!["very", "long", "word"]);
        // Short content stays one line and keeps its styles.
        let s = wrap_runs(&[("a".to_string(), Bold), (" b".to_string(), Plain)], 40);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0], vec![("a".to_string(), Bold), (" b".to_string(), Plain)]);
    }

    #[test]
    fn parse_md_runs_splits_bold_and_code() {
        use MdStyle::*;
        let runs = parse_md_runs("**name** `id { … }` — handle it.");
        assert_eq!(
            runs,
            vec![
                ("name".to_string(), Bold),
                (" ".to_string(), Plain),
                ("id { … }".to_string(), Code),
                (" — handle it.".to_string(), Plain),
            ]
        );
        // Plain text with no markers is a single Plain run.
        assert_eq!(parse_md_runs("just words"), vec![("just words".to_string(), Plain)]);
        // A single '*' is literal (only '**' toggles bold).
        assert_eq!(parse_md_runs("a * b"), vec![("a * b".to_string(), Plain)]);
    }

    #[test]
    fn reveal_fit_scrolls_with_margin_and_holds() {
        // Margin 0 is plain minimal-reveal: scroll just enough to bring the target in.
        assert_eq!(reveal_fit(100.0, 40.0, 60.0, 300.0, 0.0), 40.0); // above → up
        assert_eq!(reveal_fit(0.0, 600.0, 620.0, 300.0, 0.0), 320.0); // below → down
        assert_eq!(reveal_fit(100.0, 200.0, 220.0, 300.0, 0.0), 100.0); // visible → hold
        // Margin band (3 lines × 20 px = 60): a caret 2 lines from the
        // bottom edge scrolls until 3 clear lines separate it from the edge…
        assert_eq!(reveal_fit(0.0, 240.0, 260.0, 300.0, 60.0), 20.0);
        // …a caret already ≥ margin clear of both edges holds…
        assert_eq!(reveal_fit(0.0, 100.0, 120.0, 300.0, 60.0), 0.0);
        // …and the band clamps at the document top (no negative scroll).
        assert_eq!(reveal_fit(100.0, 40.0, 60.0, 300.0, 60.0), 0.0);
        // The margin collapses when the target nearly fills the window:
        // target 80 tall in a 100 window → m = 10, not 60.
        assert_eq!(reveal_fit(0.0, 200.0, 280.0, 100.0, 60.0), 190.0);
        // Target (with band) taller than the window: never jitter — hold.
        assert_eq!(reveal_fit(50.0, 0.0, 400.0, 300.0, 0.0), 50.0);
    }

    #[test]
    fn squiggle_spans_trace_the_wave_within_amplitude() {
        let collect = |x0, x1, baseline| {
            let mut v = Vec::new();
            squiggle_spans(x0, x1, baseline, |r| v.push(r));
            v
        };
        let spans = collect(0.0, 8.0, 10.0);
        assert!(!spans.is_empty());
        // Contiguous 1-px columns covering [0, 8).
        assert!(spans.iter().all(|r| (r.width - 1.0).abs() < 1e-3));
        assert!((spans[0].x - 0.0).abs() < 1e-3);
        // The stroke stays within the amplitude band [-amp-stroke/2, +amp+stroke/2].
        let top = spans.iter().map(|r| r.y).fold(f32::MAX, f32::min);
        let bot = spans.iter().map(|r| r.y + r.height).fold(f32::MIN, f32::max);
        assert!((top - (10.0 - SQUIGGLE_AMPLITUDE - SQUIGGLE_STROKE / 2.0)).abs() < 0.2, "reaches −amp");
        assert!((bot - (10.0 + SQUIGGLE_AMPLITUDE + SQUIGGLE_STROKE / 2.0)).abs() < 0.2, "reaches +amp");
        // Half-period 2 px: the wave peaks (+amp) one half-period-quarter in, at x≈1.
        assert!(collect(5.0, 5.0, 10.0).is_empty(), "zero width → no spans");
    }

    #[test]
    fn scrollbar_thumb_map_round_trips_and_clamps() {
        // track [0,300), 100-px thumb, 700 rows of scroll → thumb travels 200 px.
        let sb = Scrollbar { x: 288.0, track_top: 0.0, track_h: 300.0, thumb_h: 100.0, thumb_y: 100.0, max_scroll_rows: 700.0 };
        // The forward map (draw) and inverse (drag) agree at the midpoint.
        assert_eq!(sb.scroll_rows_for_thumb_top(100.0), 350.0);
        // Dragging past either end clamps to the scroll range, not past it.
        assert_eq!(sb.scroll_rows_for_thumb_top(-50.0), 0.0);
        assert_eq!(sb.scroll_rows_for_thumb_top(500.0), 700.0);
        // Hit-testing: the band is [x, ∞); the thumb is [thumb_y, +thumb_h).
        assert!(sb.contains_x(290.0) && !sb.contains_x(287.0));
        assert!(sb.thumb_contains_y(150.0) && !sb.thumb_contains_y(200.0) && !sb.thumb_contains_y(99.0));
    }

    #[test]
    fn hscrollbar_thumb_map_round_trips_and_clamps() {
        // The X-axis mirror: track [0,300), 100-px thumb, 700 px of scroll.
        let hb = HScrollbar { y: 288.0, track_left: 0.0, track_w: 300.0, thumb_w: 100.0, thumb_x: 100.0, max_scroll: 700.0 };
        assert_eq!(hb.scroll_for_thumb_left(100.0), 350.0);
        assert_eq!(hb.scroll_for_thumb_left(-50.0), 0.0);
        assert_eq!(hb.scroll_for_thumb_left(500.0), 700.0);
        // Band is [y, ∞) (bottom edge); thumb is [thumb_x, +thumb_w).
        assert!(hb.contains_y(290.0) && !hb.contains_y(287.0));
        assert!(hb.thumb_contains_x(150.0) && !hb.thumb_contains_x(200.0) && !hb.thumb_contains_x(99.0));
    }

    #[test]
    fn severity_order_is_ascending_so_error_paints_last() {
        // The draw loop sorts by this Ord; Error must sort highest (paint last).
        assert!(Severity::Hint < Severity::Info);
        assert!(Severity::Info < Severity::Warning);
        assert!(Severity::Warning < Severity::Error);
    }

    #[test]
    fn interpret_typing_and_chords() {
        let none = Modifiers::default();
        assert_eq!(interpret_key(&Key::Character("a".into()), Some("a"), none), Some(Action::Type('a')));
        // An unbound Ctrl chord is swallowed (no Type), unlike Ctrl+A/D below.
        assert_eq!(interpret_key(&Key::Character("q".into()), Some("q"), Modifiers::CTRL), None);
        assert_eq!(
            interpret_key(&Key::Character("a".into()), Some("a"), Modifiers::CTRL),
            Some(Action::SelectAll)
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowRight), None, Modifiers::SHIFT),
            Some(Action::Move { motion: Motion::Right, extend: true })
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowLeft), None, Modifiers::CTRL),
            Some(Action::Move { motion: Motion::WordLeft, extend: false })
        );
    }

    #[test]
    fn ctrl_backspace_delete_are_word_deletes() {
        assert_eq!(
            interpret_key(&Key::Named(Named::Backspace), None, Modifiers::CTRL),
            Some(Action::DeleteWordBack)
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::Delete), None, Modifiers::CTRL),
            Some(Action::DeleteWordForward)
        );
        // Plain Backspace/Delete are unchanged.
        assert_eq!(
            interpret_key(&Key::Named(Named::Backspace), None, Modifiers::default()),
            Some(Action::Backspace)
        );
    }

    #[test]
    fn ctrl_home_end_jump_to_document_ends() {
        assert_eq!(
            interpret_key(&Key::Named(Named::Home), None, Modifiers::CTRL),
            Some(Action::Move { motion: Motion::DocStart, extend: false })
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::End), None, Modifiers::CTRL | Modifiers::SHIFT),
            Some(Action::Move { motion: Motion::DocEnd, extend: true })
        );
        // Plain Home is still the smart line start.
        assert_eq!(
            interpret_key(&Key::Named(Named::Home), None, Modifiers::default()),
            Some(Action::Move { motion: Motion::LineStart, extend: false })
        );
    }

    #[test]
    fn undo_redo_chords_interpret() {
        assert_eq!(
            interpret_key(&Key::Character("z".into()), Some("z"), Modifiers::CTRL),
            Some(Action::Undo)
        );
        assert_eq!(
            interpret_key(&Key::Character("Z".into()), Some("Z"), Modifiers::CTRL | Modifiers::SHIFT),
            Some(Action::Redo) // Ctrl+Shift+Z, case-folded z
        );
        assert_eq!(
            interpret_key(&Key::Character("y".into()), Some("y"), Modifiers::CTRL),
            Some(Action::Redo)
        );
    }

    #[test]
    fn ctrl_d_and_escape_interpret() {
        assert_eq!(
            interpret_key(&Key::Character("d".into()), Some("d"), Modifiers::CTRL),
            Some(Action::AddNextOccurrence)
        );
        // Ctrl+Alt (AltGr) must NOT trigger the gesture — the `!alt` guard. (What
        // it resolves to instead is platform-dependent, since `command()` folds
        // onto Ctrl off macOS, so assert only that it isn't the gesture.)
        assert_ne!(
            interpret_key(&Key::Character("d".into()), Some("d"), Modifiers::CTRL | Modifiers::ALT),
            Some(Action::AddNextOccurrence)
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::Escape), None, Modifiers::default()),
            Some(Action::Collapse)
        );
        // Tab indents; Shift+Tab outdents.
        assert_eq!(interpret_key(&Key::Named(Named::Tab), None, Modifiers::default()), Some(Action::Tab));
        assert_eq!(
            interpret_key(&Key::Named(Named::Tab), None, Modifiers::SHIFT),
            Some(Action::Outdent)
        );
    }

    #[test]
    fn ctrl_slash_is_toggle_comment() {
        let ctrl = Modifiers::CTRL;
        assert_eq!(
            interpret_key(&Key::Character("/".into()), Some("/"), ctrl),
            Some(Action::ToggleComment)
        );
        // Plain "/" stays typing; Ctrl+Alt (AltGr) never triggers it.
        assert_eq!(
            interpret_key(&Key::Character("/".into()), Some("/"), Modifiers::empty()),
            Some(Action::Type('/'))
        );
        assert_eq!(interpret_key(&Key::Character("/".into()), Some("/"), ctrl | Modifiers::ALT), None);
    }

    #[test]
    fn ctrl_shift_alt_arrow_is_column_select() {
        let csa = Modifiers::CTRL | Modifiers::SHIFT | Modifiers::ALT;
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowDown), None, csa),
            Some(Action::ColumnSelect(ColumnDir::Down))
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowRight), None, csa),
            Some(Action::ColumnSelect(ColumnDir::Right))
        );
        // Without Alt it stays word-motion extend, not column select.
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowLeft), None, Modifiers::CTRL | Modifiers::SHIFT),
            Some(Action::Move { motion: Motion::WordLeft, extend: true })
        );
    }

    #[test]
    fn alt_arrow_moves_and_copies_lines() {
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowDown), None, Modifiers::ALT),
            Some(Action::MoveLine { down: true })
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowUp), None, Modifiers::ALT),
            Some(Action::MoveLine { down: false })
        );
        assert_eq!(
            interpret_key(&Key::Named(Named::ArrowDown), None, Modifiers::ALT | Modifiers::SHIFT),
            Some(Action::CopyLine { down: true })
        );
        // Ctrl+Shift+Alt+Down stays column-select, not copy-line.
        assert_eq!(
            interpret_key(
                &Key::Named(Named::ArrowDown),
                None,
                Modifiers::CTRL | Modifiers::SHIFT | Modifiers::ALT
            ),
            Some(Action::ColumnSelect(ColumnDir::Down))
        );
    }

    #[test]
    fn clipboard_payload_feeds_write_clipboard() {
        // The payload rule itself lives (and is tested) in the core; here we
        // pin that the widget consults it — mixed sets keep the joined form.
        use scrive_core::{Document, Selection, SelectionId, SelectionSet};
        let mut doc = Document::new("foo\nbar\nbaz").unwrap();
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 0, 3)); // "foo"
        set.add_selection(4, 7); // "bar"
        set.add_caret(9); // an empty caret contributes nothing when mixed
        doc.set_selections(set);
        assert_eq!(doc.clipboard_payload(), ("foo\nbar".to_string(), false));
    }

    #[test]
    fn gesture_autoscroll_policy() {
        assert!(!Action::AddNextOccurrence.moves_caret()); // reveals via request_reveal (FitForce)
        assert!(!Action::AddCaret(0).moves_caret()); // click lands in view
        assert!(!Action::Collapse.moves_caret()); // keeps an existing caret
        assert!(!Action::PlaceCaret(0).moves_caret());
        assert!(Action::Type('x').moves_caret());
        // Core-revealed verbs (request_reveal on success only): the widget
        // never blind-Fits for them, so a no-op press cannot move the view.
        assert!(!Action::NextDiagnostic { forward: true }.moves_caret());
        assert!(!Action::JumpToBracket.moves_caret());
        assert!(!Action::ExpandSelection.moves_caret());
        assert!(!Action::AddCaretVertical { down: true }.moves_caret());
        assert!(!Action::SelectAllOccurrences.moves_caret());
        // A drag reveals its head — a drag past the viewport edge must scroll.
        assert!(Action::DragSelect { granularity: Granularity::Char, origin: 0, head: 5 }
            .moves_caret());
        // A viewport report is not an edit and must never autoscroll (that would
        // feed back: report → scroll → report).
        assert!(!Action::ViewportChanged(0..42).moves_caret());
    }

    #[test]
    fn last_visible_row_tracks_scroll_and_clamps() {
        use scrive_core::Document;
        // 100 lines; line height 10 px; 200 px viewport (20 rows); margin 8.
        let text = vec!["x"; 100].join("\n");
        let doc = Document::new(&text).unwrap();
        let ed = Editor::new(&doc, |_: Action| ());
        let bounds = Rectangle { x: 0.0, y: 0.0, width: 300.0, height: 200.0 };
        // scroll 0: last px 200 → row 20 + margin 8 = 28.
        assert_eq!(ed.last_visible_row(bounds, 0.0, 10.0), 28);
        // scrolled 50 rows (500 px): last row 70 + 8 = 78 (row units — the
        // ScrollAnchor currency).
        assert_eq!(ed.last_visible_row(bounds, 50.0, 10.0), 78);
        // scrolled past the end: clamps to the last row (line_count - 1 = 99).
        assert_eq!(ed.last_visible_row(bounds, 500.0, 10.0), 99);
    }

    #[test]
    fn blink_phase_toggles_each_interval() {
        assert!(blink_on(0)); // just focused → solid
        assert!(blink_on(499)); // end of the on-half → still solid
        assert!(!blink_on(500)); // off-half begins
        assert!(!blink_on(999));
        assert!(blink_on(1000)); // back on
        assert!(blink_on(1499));
        assert!(!blink_on(1500));
    }

    #[test]
    fn focus_shows_solid_caret_unfocus_hides_it() {
        let mut st = State::default(); // starts focused
        assert!(st.is_focused());
        assert!(st.caret_on()); // elapsed ≈ 0 → solid
        st.unfocus();
        assert!(!st.is_focused());
        assert!(!st.caret_on()); // no caret when unfocused
        st.focus();
        assert!(st.caret_on()); // refocus → solid again
    }

    #[test]
    fn focus_operation_targets_by_id() {
        // The editor participates in iced's focus protocol via `State:
        // Focusable`; drive the real `focus` operation against it.
        let id = widget::Id::from("editor");
        let other = widget::Id::from("other");
        let bounds = Rectangle::new(Point::ORIGIN, Size::new(1.0, 1.0));
        let mut st = State::default();

        // A focus aimed elsewhere unfocuses us…
        operation::focusable::focus::<()>(other).focusable(Some(&id), bounds, &mut st);
        assert!(!st.is_focused());
        // …one aimed at our id focuses us…
        operation::focusable::focus::<()>(id.clone()).focusable(Some(&id), bounds, &mut st);
        assert!(st.is_focused());
        // …and an id-less editor can never be targeted (None ≠ Some(target)).
        operation::focusable::focus::<()>(id).focusable(None, bounds, &mut st);
        assert!(!st.is_focused());
    }

    // ── Fold-geometry tests: the display projections stay correct at the widget
    //    level, now that `Geo` makes the layout paths renderer-free ──

    /// Rows `x / a { / b / c / } / word here`, block collapsed: buffer rows
    /// 2..=4 hide, so buffer row 5 renders at display row 2.
    fn folded_doc() -> Document {
        let text = "x\na {\nb\nc\n}\nword here\n";
        let mut doc = Document::new(text).unwrap();
        let opener = text.find('{').unwrap() as u32;
        assert!(doc.toggle_fold_opener(opener));
        doc
    }

    /// Round-number frame: gutter 50, advance 10, line_h 10, unscrolled.
    fn test_geo() -> Geo {
        Geo::new(
            Rectangle { x: 0.0, y: 0.0, width: 800.0, height: 600.0 },
            50.0,
            10.0,
            10.0,
            0.0,
            ScrollAnchor::TOP,
        )
    }

    #[test]
    fn collapsed_chip_at_windows_to_the_hovered_row() {
        // The plain-hover chip hit-test resolves the pointer's display row
        // and tests only the pairs headed there — never a scan of every fold in
        // the document (which would build a `HeaderLayout` per fold each frame
        // and stall a collapsed large file). A point over the collapsed `a { … }`
        // header's chip returns its opener; a point on the fold-less row above
        // returns None (that row's windowed query is empty).
        let doc = folded_doc(); // block collapsed; header at buffer row 1
        let ed = Editor::new(&doc, |_: Action| ());
        let geo = test_geo();
        let fm = ed.fold_map();
        let opener = doc.buffer().text().find('{').unwrap() as u32;
        // Hover the exact chip rect the painter uses.
        let rect = ed.collapsed_chip_rect(&fm, &geo, opener).expect("collapsed header has a chip");
        let center = Point::new(rect.x + rect.width / 2.0, rect.y + rect.height / 2.0);
        assert_eq!(ed.collapsed_chip_at(&geo, center), Some(opener), "hover over the chip finds it");
        // A point far right of the chip on the same row is off it.
        assert_eq!(ed.collapsed_chip_at(&geo, Point::new(rect.x + rect.width + 200.0, center.y)), None);
        // A point on display row 0 ("x", no fold) resolves a fold-less row.
        assert_eq!(ed.collapsed_chip_at(&geo, Point::new(center.x, 5.0)), None);
    }

    #[test]
    fn block_fold_hover_target_excludes_text_after_the_closer() {
        // Text typed after a folded block's closing `}` (on the tail line) must
        // not join its hover / expand hit region — the block rect ends at the
        // closer, not the whole tail line's width. The target covers only `{ … }`.
        let text = "x\na {\nb\nc\n} trailing text after brace\n";
        let mut doc = Document::new(text).unwrap();
        let opener = text.find('{').unwrap() as u32;
        assert!(doc.toggle_fold_opener(opener)); // block: `{` row 1 … `}` row 4
        let ed = Editor::new(&doc, |_: Action| ());
        let geo = test_geo();
        let fm = ed.fold_map();
        let rect = ed.collapsed_chip_rect(&fm, &geo, opener).expect("collapsed header has a chip");
        let cy = rect.y + rect.height / 2.0;
        // The `…` chip is still the expand target.
        assert_eq!(
            ed.collapsed_chip_at(&geo, Point::new(rect.x + rect.width / 2.0, cy)),
            Some(opener),
        );
        // The box ends at the closer: `a {`(3) + ` … `(4) + `}`(1) ≈ 8 cells, far
        // short of the ~34-cell full line — so a point out over the trailing text
        // must not hit the fold.
        assert!(
            rect.width < 12.0 * geo.advance(),
            "hover box stops at the closer, not the trailing text ({}px)",
            rect.width
        );
        assert_eq!(
            ed.collapsed_chip_at(&geo, Point::new(geo.cell_x(16.0), cy)),
            None,
            "text after the closing brace is not part of the fold's hover area"
        );
    }

    #[test]
    fn draw_budget_ctrl_a_over_folded_doc_stays_windowed() {
        // A document-spanning selection over a fully-folded large file must wash
        // only the VISIBLE display rows, never every buffer row. Fold a big doc,
        // select all, render one headless frame, and assert the wash visited
        // O(viewport) rows — washing `for row in a.row..=b.row` would visit all N
        // rows (thousands here) and freeze; the display-window walk visits a few
        // dozen. A change that washes buffer rows through `draw_wash_row` trips
        // this and the in-`draw` debug assert.
        use iced::advanced::{clipboard, mouse, renderer};
        use iced::{Font, Pixels, Point as IPoint, Size};
        use iced_runtime::user_interface::{Cache, UserInterface};

        const N: usize = 3000;
        let mut text = String::with_capacity(N * 20);
        for i in 0..N {
            text.push_str("fn f");
            text.push_str(&i.to_string());
            text.push_str("() {\n    body\n}\n");
        }
        let mut doc = Document::new(&text).expect("doc fits");
        let opens: Vec<u32> = doc.collapsible_pairs().into_iter().map(|(o, ..)| o).collect();
        assert!(opens.len() >= N, "every block is foldable ({} of {N})", opens.len());
        doc.set_selections(scrive_core::SelectionSet::from_offsets(&opens));
        doc.fold_at_carets(false); // collapse every block
        doc.select_all(); // the doc-spanning Ctrl+A selection

        // Headless one-frame render (mirrors examples/shared/capture.rs): build →
        // update(RedrawRequested) → draw. `draw` resets the budget, every wash
        // bumps it, so afterwards it holds exactly this frame's visited rows.
        iced_tiny_skia::graphics::text::font_system()
            .write()
            .expect("font system lock")
            .load_font(std::borrow::Cow::Borrowed(crate::CODICON_FONT));
        let mut r = iced_renderer::fallback::Renderer::Secondary(
            iced_tiny_skia::Renderer::new(Font::default(), Pixels(14.0)),
        );
        let (w, h) = (500.0_f32, 320.0_f32);
        let cursor = mouse::Cursor::Available(IPoint::new(w / 2.0, h / 2.0));
        let element: iced::Element<'_, (), iced::Theme, iced::Renderer> =
            Editor::new(&doc, |_: Action| ()).into();
        let mut ui = UserInterface::build(element, Size::new(w, h), Cache::new(), &mut r);
        let mut msgs: Vec<()> = Vec::new();
        ui.update(
            &[iced::Event::Window(iced::window::Event::RedrawRequested(std::time::Instant::now()))],
            cursor,
            &mut r,
            &mut clipboard::Null,
            &mut msgs,
        );
        ui.draw(&mut r, &iced::Theme::Dark, &renderer::Style::default(), cursor);

        // ~320px / ~19px line ≈ 17 visible rows; the wash bumps ~2 per row. A few
        // hundred is the generous ceiling; O(document) would be ~2·N = 6000.
        // ~48 in practice (a ~17-row viewport, ~2 bumps/row). The lower bound
        // catches a vacuous pass (nothing drawn); the upper catches O(document)
        // (~2·N = 6000).
        let rows = draw_budget::rows();
        assert!((8..400).contains(&rows), "Ctrl+A wash over a folded {N}-block doc visited {rows} rows — expected O(viewport), not O(document)");
    }

    #[test]
    fn popup_anchors_are_display_space_below_a_fold() {
        use scrive_core::{HoverInfo, PopupList, Selection, SelectionId, SelectionSet};
        let mut doc = folded_doc();
        // Caret on "word here" (buffer row 5, col 0) — display row 2.
        let head = doc.buffer().point_to_offset(BufPoint::new(5, 0));
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), head, head));
        doc.set_selections(set);
        let ed = Editor::new(&doc, |_: Action| ());
        let geo = test_geo();
        let fm = ed.fold_map();

        // The one shared anchor must be display-space: row 2 × 10 px, NOT buffer
        // row 5 × 10 px — so a popup below a fold sits at the right height.
        let (x, top, bottom) = ed.popup_anchor(&fm, &geo, head);
        assert_eq!((top, bottom), (20.0, 30.0), "display row 2, not buffer row 5");
        assert_eq!(x, 56.0, "gutter 50 + TEXT_PAD 6 + col 0");

        // Both testable panels consume it: the completion flips below-first…
        let list = PopupList { items: vec![], filtered: vec![], selected: 0, anchor: head };
        let (origin, _) = ed.popup_layout(&list, &geo);
        assert_eq!(origin.y, 30.0, "completion sits below the DISPLAY row");
        // …the hover above-first. (The signature box calls the same
        // popup_anchor by construction.)
        let info = HoverInfo { markdown: "hi".into(), range: head..head + 4 };
        let l = ed.hover_layout(&info, &geo);
        assert_eq!(l.rect.y, top - l.rect.height, "hover sits above the DISPLAY row");
    }

    #[test]
    fn hit_test_inverts_offset_screen_x_across_fold_geometry() {
        // An inline fold AND a block fold on one header — the hardest case for
        // the projection, exercising every path at once: `f([ … ]) { … }`.
        let text = "f([a, b]) {\ninner\n}\nafter\n";
        let mut doc = Document::new(text).unwrap();
        let inline_open = text.find('[').unwrap() as u32;
        let close = text.find(']').unwrap() as u32;
        let block_open = text.find('{').unwrap() as u32;
        assert!(doc.toggle_fold_opener(inline_open));
        assert!(doc.toggle_fold_opener(block_open));
        let ed = Editor::new(&doc, |_: Action| ());
        let geo = test_geo();
        let fm = ed.fold_map();
        let buffer = doc.buffer();
        let tail = buffer.point_to_offset(BufPoint::new(2, 0)); // the real `}`
        let after = buffer.point_to_offset(BufPoint::new(3, 2)); // below the fold
        // Every landable offset round-trips: forward-project to screen, then
        // hit-test the same pixel back. Fails against either a fold-blind x
        // or a buffer-space chip compare.
        for off in [0, inline_open + 1, close, block_open, tail, after] {
            let p = fm.display_position(buffer, off, TAB).expect("landable offset");
            let pos = Point::new(ed.offset_screen_x(&fm, &geo, off), geo.row_y(p.row) + 5.0);
            assert_eq!(ed.hit_test(&geo, pos), off, "offset {off} round-trips");
        }
    }

    /// Deep-scroll row positions stay exact. At ~5.9M rows scrolled to the bottom,
    /// content y-space is ~112M px — past that scale a flat `f32` pixel scroll
    /// would quantize to 8-px representable steps, rendering consecutive rows 16
    /// or 24 px apart instead of 19 and losing half-row precision in the y→row
    /// inverse (wrong-line clicks). The [`ScrollAnchor`] model (row-space integers
    /// plus a bounded sub-row offset) is exact for any u32-addressable document —
    /// this pins both the uniform step and the hit round-trip at depth.
    #[test]
    fn deep_scroll_row_positions_stay_exact() {
        let rows: u32 = 5_900_000;
        let doc = Document::new(&"x\n".repeat(rows as usize)).unwrap();
        let ed = Editor::new(&doc, |_: Action| ());
        let fm = ed.fold_map();
        let line_h = 19.0_f32;
        let bounds = Rectangle { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
        // Scrolled to the very bottom, exactly as layout()'s clamp computes it
        // (a fractional row position, so offset_px is non-zero too).
        let max_rows = f64::from(rows) + 1.0 - f64::from(bounds.height) / f64::from(line_h);
        let anchor = ScrollAnchor::from_rows(max_rows, line_h);
        let geo = Geo::new(bounds, 50.0, 10.0, line_h, 0.0, anchor);
        let first = fm.display_row_at(geo.rows_from_top(bounds.y));
        for i in 0..30 {
            let row = fm.to_display_row(BufferRow(first.index() + i));
            let next = fm.to_display_row(BufferRow(first.index() + i + 1));
            let step = geo.row_y(next) - geo.row_y(row);
            assert!((step - line_h).abs() < 0.01, "row {i}: step {step}, want {line_h}");
            // The y→row inverse hits the row it was projected from.
            let hit = fm.display_row_at(geo.rows_from_top(geo.row_y(row) + 1.0));
            assert_eq!(hit, row, "row {i} round-trips");
        }
        // And the anchor round-trips its own row-unit projection exactly.
        assert!((anchor.rows(line_h) - max_rows).abs() < 1e-6);
    }

    #[test]
    fn max_line_px_shrinks_when_the_widest_lines_fold() {
        // Viewport-max semantics: with the whole small doc in the window, the
        // widest visible row is the global max — hidden rows contribute nothing,
        // the collapsed header spans its whole `f() { … }` placeholder, from the
        // one HeaderLayout owner.
        let vp = Rectangle { x: 0.0, y: 0.0, width: 800.0, height: 600.0 };
        let text = "f() {\n    a_very_long_interior_line_wwwwwwwwwwwwwww\n}\nshort\n";
        let mut doc = Document::new(text).unwrap();
        let unfolded = Editor::new(&doc, |_: Action| ()).max_line_px(10.0, 20.0, vp, 0.0);
        let opener = text.find('{').unwrap() as u32;
        assert!(doc.toggle_fold_opener(opener));
        let ed = Editor::new(&doc, |_: Action| ());
        let folded = ed.max_line_px(10.0, 20.0, vp, 0.0);
        let fm = ed.fold_map();
        let hl = fm.header_layout(doc.buffer(), BufferRow(0), TAB).unwrap();
        assert_eq!(folded, hl.width() as f32 * 10.0);
        assert!(folded < unfolded, "collapsing the widest line shrinks the h-range");
    }

    #[test]
    fn max_line_px_is_viewport_scoped() {
        // The widest line OFF-screen does not widen the h-range: a 2-row
        // viewport over the short head of a doc whose widest line is far
        // below reports the head's width — the adaptive h-scrollbar
        // (field-gated; a document-owned line-width index would restore a
        // global max if the adaptive feel fails).
        let mut text = String::from("ab\ncd\n");
        text.push_str(&"x".repeat(200));
        text.push('\n');
        let doc = Document::new(&text).unwrap();
        let ed = Editor::new(&doc, |_: Action| ());
        let two_rows = Rectangle { x: 0.0, y: 0.0, width: 800.0, height: 40.0 };
        assert_eq!(ed.max_line_px(10.0, 20.0, two_rows, 0.0), 2.0 * 10.0, "only visible rows count");
        // Scrolled down 2 rows to the wide line, the range adapts up.
        assert_eq!(ed.max_line_px(10.0, 20.0, two_rows, 2.0), 200.0 * 10.0);
    }
}
