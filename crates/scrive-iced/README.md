# scrive

[![CI](https://github.com/robbym/scrive/actions/workflows/ci.yml/badge.svg)](https://github.com/robbym/scrive/actions/workflows/ci.yml)
[![crates.io](https://img.shields.io/crates/v/scrive-iced)](https://crates.io/crates/scrive-iced)
[![docs.rs](https://img.shields.io/docsrs/scrive-iced)](https://docs.rs/scrive-iced)

A from-scratch code-editor widget for [iced](https://iced.rs) — a headless
editing engine plus an `iced::advanced::Widget` that renders it.

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
- **Bracket-pair colorization, matching, and indent guides.**
- **Code folding** — foldable ranges, gutter chevrons, nested and inline
  (sub-line) collapse, and fold-aware movement.
- **Find** — literal search with a match set repaired per edit, so results never
  go stale.
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
protocol. It is a **controlled widget**: it borrows the `Document` immutably for
drawing and emits semantic actions (`Type`, `Move`, `FindNext`, …); the
application owns the `Document` and applies the actions, so the widget never
mutates state behind the app's back.

## Example

```bash
cargo run -p scrive-iced --example scratch
```

opens a real editor over a sample Rust document — type, select, find (Ctrl+F),
fold, and undo/redo. The host application **must** load
`scrive_iced::CODICON_FONT` at startup
(`iced::application(..).font(scrive_iced::CODICON_FONT)`) so the fold-gutter
chevrons and find-bar icons render.

## License

MIT.
