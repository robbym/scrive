//! Monospace text metrics, **measured** once per (font, size) from the
//! renderer's own shaper — never a hardcoded advance.
//!
//! The caret, hit-testing, gutter, and selections all lay out in
//! `column × advance`. For that to line up with the glyphs, `advance` must be
//! the exact value the shaper (cosmic-text → harfrust) places glyphs by — so we
//! shape a run of identical cells through iced's paragraph API and divide the
//! measured width out. Because both iced renderers (wgpu, tiny-skia) share that
//! shaper, one measurement is correct in the window and in the headless
//! capture, for whatever monospace font the app configures.

use iced::advanced::text::{
    Alignment, LineHeight, Paragraph as _, Shaping, Text, Wrapping,
};
use iced::alignment::Vertical;
use iced::{Font, Pixels, Size};

/// The concrete paragraph type of the default iced renderer — its `with_text`
/// shapes through the same font system the renderer draws with.
type Paragraph = <iced::Renderer as iced::advanced::text::Renderer>::Paragraph;

/// Cells sampled when measuring; dividing by the count averages out any
/// start/end shaping so a single-pixel error can't skew the per-cell advance.
const SAMPLE: &str = "0000000000000000";

/// Cell advance and row height for a (font, size), measured from the shaper.
#[derive(Copy, Clone, Debug, PartialEq)]
pub struct Metrics {
    /// Advance width of one monospace cell, in logical pixels.
    pub advance: f32,
    /// Row height, in logical pixels.
    pub line_height: f32,
    /// The font size these metrics were measured at.
    pub size: f32,
}

impl Metrics {
    /// Measure `font` at `size` with row height `line_height` by shaping
    /// `SAMPLE` and dividing out the width — the advance the renderer's glyph
    /// placement actually uses, so the caret cannot drift from the text.
    #[must_use]
    pub fn measure(font: Font, size: f32, line_height: f32) -> Self {
        let text = Text {
            content: SAMPLE,
            bounds: Size::new(f32::INFINITY, f32::INFINITY),
            size: Pixels(size),
            line_height: LineHeight::Absolute(Pixels(line_height)),
            font,
            align_x: Alignment::Left,
            align_y: Vertical::Top,
            shaping: Shaping::Basic,
            wrapping: Wrapping::None,
        };
        let width = Paragraph::with_text(text).min_width();
        let advance = width / SAMPLE.chars().count() as f32;
        Self { advance, line_height, size }
    }
}
