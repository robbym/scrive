# scrive

[![CI](https://github.com/robbym/scrive/actions/workflows/ci.yml/badge.svg)](https://github.com/robbym/scrive/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/scrive-iced)](https://crates.io/crates/scrive-iced)
[![docs.rs](https://img.shields.io/docsrs/scrive-iced)](https://docs.rs/scrive-iced)

A from-scratch code-editor widget for [iced](https://iced.rs) — a headless
editing engine plus an `iced::advanced::Widget` that renders it.

![The scrive editor: syntax highlighting, multi-cursor rename, find and replace with match navigation and replace-all, code folding, and completion](https://raw.githubusercontent.com/robbym/scrive/master/docs/showcase.gif)

<sub>Recorded deterministically, headless — the demo renders itself, no external tools:
`cargo run -p scrive-iced --example record_showcase --release -- docs/showcase.gif`</sub>

Two crates, with the dependency pointing one way:

- **`scrive-core`** — the headless engine: a rope-backed buffer, atomic
  multi-range transactions with mechanically-derived undo/redo, selections and
  multi-cursor, tracked-range decorations (diagnostics, find matches, snippet
  stops), an incremental syntax-highlight cache, folding, and the
  language-intelligence controllers. Imports no GUI crate.
- **`scrive-iced`** — an `iced::advanced::Widget` wrapping the core: gutter and
  line numbers, N-caret selections, syntax highlighting, bracket-pair colors and
  matching, indent guides, diagnostic squiggles, code folding, and the
  completion / hover / signature-help popups. Depends on `scrive-core`; the
  dependency never points back.

## Features

- **Rope-backed buffer** with an O(1) snapshot for background work; the
  document-size envelope is bounded only by the u32 offset space, not a load cap.
- **Atomic transactions with undo/redo** — every edit is one transaction with a
  mechanically-derived inverse, and undo/redo flow through the same
  view-rebase path as forward edits.
- **Multi-cursor** editing, column (box) selection, and word-wise motion.
- **Syntax highlighting** via an incremental, viewport-windowed
  [syntect](https://github.com/trishume/syntect) cache; the app supplies the
  grammar and theme (the core ships neither).
- **Bracket-pair colorization, matching, and indent guides** — optionally
  comment/string-aware, so brackets inside line comments, strings, and char
  literals are skipped (opt-in per language; line-local).
- **Code folding** — foldable ranges, gutter chevrons, nested and inline
  (sub-line) collapse, and fold-aware movement.
- **Find and replace** — literal, whole-word, or regex search (with `$1` capture
  groups in the replacement), find-in-selection, case-preserving replace, and
  replace / replace-all as a single undo step. The match set is repaired per
  edit, so results never go stale.
- **Language intelligence** — completion, snippets, signature help, and hover,
  exposed as trait seams the integrating application implements.
- **Diagnostics** — squiggles, a diagnostic hover, and scrollbar overview marks.

Every derived position — a caret, a find match, a diagnostic, a snippet stop — is
moved through a windowed or O(log n) update per keystroke, so no editing
operation scales with the size of the document.

## Design

The load-bearing idea is **one fact, one owner**: every derived position moves
through a single mapping function on every edit, and every change is one atomic
transaction with a mechanically-derived inverse — so the class of bug where two
copies of one fact drift apart cannot occur, on the forward path or on undo.

The editor is a direct `iced::advanced::Widget` rather than a `canvas::Program`,
because only the low-level widget API participates in iced's focus/operation
protocol. The low-level `Editor` is a **controlled widget**: it borrows the
`Document` immutably for drawing and emits semantic actions (`Type`, `Move`,
`Undo`, …); the application owns the `Document` and applies the actions, so the
widget never mutates state behind the app's back. The `CodeEditor` tier wraps
that widget, owns the `Document` itself, and runs the plumbing for you.

## Quick start

`CodeEditor` owns the document and runs highlighting, find, focus, and language
intelligence internally — integrating is three wires plus registering the
bundled font:

```rust
use iced::{Element, Subscription, Task};
use scrive_iced::{CodeEditor, Event};

struct App { editor: CodeEditor }

#[derive(Debug, Clone)]
enum Message { Editor(Event) }

impl App {
    fn new() -> Self { Self { editor: CodeEditor::new("fn main() {}\n") } }
    fn update(&mut self, m: Message) -> Task<Message> {
        match m { Message::Editor(e) => self.editor.update(e).map(Message::Editor) }
    }
    fn view(&self) -> Element<'_, Message> { self.editor.view().map(Message::Editor) }
    fn subscription(&self) -> Subscription<Message> {
        self.editor.subscription().map(Message::Editor)
    }
}
```

Attach a grammar with `.language(grammar)` (highlighting is coloured at load, no
scroll needed) and override defaults with `.theme(..)`, `.find(..)`,
`.completions(..)`, `.hover(..)`, and `.signature(..)`. For full control, drop to
the low-level `Editor` widget. The host **must** register the bundled font —
`iced::application(..).font(scrive_iced::CODICON_FONT)` — so the fold chevrons
and find-bar icons render.

## Examples

```bash
cargo run -p scrive-iced --example minimal   # the CodeEditor quick start above
cargo run -p scrive-iced --example scratch   # the low-level Editor widget, full control
```

`scratch` opens a real editor over a sample Rust document — type, select, find
(Ctrl+F), fold, and undo/redo.

## License

MIT.
