//! `scrive-iced` — the iced 0.14 integration for the scrive code editor.
//!
//! Turns [`scrive_core`] into an on-screen widget: a direct
//! `iced::advanced::Widget` (deliberately *not* a `canvas::Program`, because
//! only the low-level widget API exposes the `operate()` hook needed to join
//! iced's focus/operation protocol), with a gutter, N-caret selections, syntect
//! highlighting, diagnostic squiggles, a completion popup, and hover.
//!
//! # Two tiers
//!
//! - [`CodeEditor`] — **start here.** The batteries-included tier: it owns a
//!   [`scrive_core::Document`] and runs the highlighting, find, focus, and
//!   language-intelligence plumbing internally, so integrating is three wires
//!   ([`update`](CodeEditor::update), [`view`](CodeEditor::view),
//!   [`subscription`](CodeEditor::subscription)) plus registering
//!   [`required_fonts`] at startup. Highlighting is coloured at load with no
//!   scroll needed; the find bar, selection, undo, and folding are on by default.
//!   See `examples/minimal.rs`.
//! - [`Editor`] — the low-level *controlled* widget: it renders a `Document` and
//!   emits semantic [`Action`]s the application applies by hand. Full control,
//!   all the plumbing on you. See `examples/scratch.rs`.
//!
//! The minimal integration:
//!
//! ```no_run
//! use iced::{Element, Subscription, Task};
//! use scrive_iced::{CodeEditor, Event};
//!
//! struct App { editor: CodeEditor }
//!
//! #[derive(Debug, Clone)]
//! enum Message { Editor(Event) }
//!
//! impl App {
//!     fn new() -> Self { Self { editor: CodeEditor::new("fn main() {}\n") } }
//!     fn update(&mut self, m: Message) -> Task<Message> {
//!         match m { Message::Editor(e) => self.editor.update(e).map(Message::Editor) }
//!     }
//!     fn view(&self) -> Element<'_, Message> { self.editor.view().map(Message::Editor) }
//!     fn subscription(&self) -> Subscription<Message> {
//!         self.editor.subscription().map(Message::Editor)
//!     }
//! }
//! ```

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod clipboard;
pub mod code_editor;
pub mod editor;
mod geo;
mod highlight_pool;
pub mod metrics;
pub mod popup;

pub use code_editor::{CodeEditor, Event};
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

/// The bundled **Scrive Dark** syntax theme — an original, MIT-licensed dark
/// theme (the one that shipped with 0.1.0). It is the sensible default so a host
/// gets colored text from a grammar alone, without supplying a `.tmTheme`: the
/// batteries-included editor tier applies it unless the host overrides it with
/// another [`TokenTheme`](scrive_core::TokenTheme).
///
/// The theme is compiled in, so parsing it cannot fail at runtime — a malformed
/// asset is a packaging bug the crate's own tests catch, not a caller error.
/// That is why this returns the theme directly rather than a `Result`.
#[must_use]
pub fn scrive_dark_theme() -> scrive_core::TokenTheme {
    scrive_core::TokenTheme::from_tm_theme(include_str!("../assets/scrive-dark.tmTheme"))
        .expect("bundled Scrive Dark theme parses")
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
    /// `preserve-case` (U+EB2E) — the replace bar's `AB` option toggle.
    pub const PRESERVE_CASE: char = '\u{eb2e}';
    /// `whole-word` (U+EB7E) — the find bar's `ab|` option toggle.
    pub const WHOLE_WORD: char = '\u{eb7e}';
    /// `regex` (U+EB38) — the find bar's `.*` option toggle.
    pub const REGEX: char = '\u{eb38}';
    /// `list-selection` (U+EB85) — the find bar's "find in selection" toggle.
    /// The codicon set names this glyph `list-selection`; `selection` is an
    /// alias for it, and is what VS Code calls the same button.
    pub const SELECTION: char = '\u{eb85}';
}

#[cfg(test)]
mod tests {
    /// The compiled-in Scrive Dark theme must parse — the `expect` in
    /// [`super::scrive_dark_theme`] would otherwise panic in every host that
    /// takes the default. This is the packaging guard the doc comment promises.
    #[test]
    fn bundled_scrive_dark_theme_parses() {
        let _ = super::scrive_dark_theme();
    }
}
