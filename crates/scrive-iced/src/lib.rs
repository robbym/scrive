//! `scrive-iced` — the iced 0.14 integration for the scrive code editor.
//!
//! Turns [`scrive_core`] into an on-screen widget: a direct
//! `iced::advanced::Widget` (deliberately *not* a `canvas::Program`, because
//! only the low-level widget API exposes the `operate()` hook needed to join
//! iced's focus/operation protocol), with a gutter, N-caret selections, syntect
//! highlighting, diagnostic squiggles, a completion popup, and hover.
//!
//! [`Editor`] renders a [`scrive_core::Document`] and emits [`Action`]s the
//! application applies to its document. See `examples/scratch.rs` for the
//! wiring and a runnable window.

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod clipboard;
pub mod editor;
mod geo;
pub mod metrics;
pub mod popup;

pub use editor::{default_autoscroll_margin, Action, Editor, SCROLLBAR_WIDTH};
pub use metrics::Metrics;

/// The bundled [Codicon](https://github.com/microsoft/vscode-codicons) icon font
/// (v0.0.45) — VS Code's own UI glyph set. The host application **must** load
/// these bytes into iced's font system at startup (e.g.
/// `iced::application(..).font(scrive_iced::CODICON_FONT)`); after that the
/// widget's fold-gutter chevrons and any app chrome can draw glyphs in the
/// [`CODICON`] font. Icons © Microsoft, CC BY 4.0 (see `assets/CODICON-LICENSE.md`).
pub const CODICON_FONT: &[u8] = include_bytes!("../assets/codicon.ttf");

/// The [`iced::Font`] handle for the bundled [`CODICON_FONT`] (family `"codicon"`).
pub const CODICON: iced::Font = iced::Font::with_name("codicon");

/// Every font the widget needs registered in iced's font system at startup —
/// register them all and the fold-gutter chevrons and find-bar icons render;
/// omit one and its glyphs fall back to per-machine tofu. One owner so an
/// integrator can load the whole set instead of enumerating it by hand (today
/// just [`CODICON_FONT`]):
/// `scrive_iced::required_fonts().iter().fold(app, |app, f| app.font(*f))`.
#[must_use]
pub fn required_fonts() -> &'static [&'static [u8]] {
    &[CODICON_FONT]
}

/// Codicon glyph codepoints scrive draws. Names and values are from the codicon
/// `mapping.json` (verified against v0.0.45); the private-use-area codepoints are
/// only meaningful rendered in the [`CODICON`] font.
pub mod icon {
    /// `chevron-right` (U+EAB6) — the collapsed-fold gutter indicator, and the
    /// find bar's collapsed replace-row toggle.
    pub const CHEVRON_RIGHT: char = '\u{eab6}';
    /// `chevron-down` (U+EAB4) — the expanded-fold gutter indicator, and the
    /// find bar's expanded replace-row toggle.
    pub const CHEVRON_DOWN: char = '\u{eab4}';
    /// `arrow-up` (U+EAA1) — find "previous match".
    pub const ARROW_UP: char = '\u{eaa1}';
    /// `arrow-down` (U+EA9A) — find "next match".
    pub const ARROW_DOWN: char = '\u{ea9a}';
    /// `close` (U+EA76) — find "close".
    pub const CLOSE: char = '\u{ea76}';
    /// `replace` (U+EB3D) — find "replace this match".
    pub const REPLACE: char = '\u{eb3d}';
    /// `replace-all` (U+EB3C) — find "replace every match".
    pub const REPLACE_ALL: char = '\u{eb3c}';
    /// `case-sensitive` (U+EAB1) — the find bar's `Aa` option toggle.
    pub const CASE_SENSITIVE: char = '\u{eab1}';
    /// `whole-word` (U+EB7E) — the find bar's `ab|` option toggle.
    pub const WHOLE_WORD: char = '\u{eb7e}';
    /// `regex` (U+EB38) — the find bar's `.*` option toggle.
    pub const REGEX: char = '\u{eb38}';
    /// `list-selection` (U+EB85) — the find bar's "find in selection" toggle.
    /// The codicon set names this glyph `list-selection`; `selection` is an
    /// alias for it, and is what VS Code calls the same button.
    pub const SELECTION: char = '\u{eb85}';
}
