# Changelog

Notable changes to `scrive-core` and `scrive-iced`. The two crates share one
version and are released together.

## 0.3.1 — 2026-10-17

### Fixed

- **scrive-iced:** on Windows and Linux, an Alt chord the editor has no binding
  for no longer types its character. Windows delivers Alt+letter's character as
  key text, so a host shortcut such as Shift+Alt+F (Format Document) also
  inserted an `F`. Alt+Space likewise inserted a space. The editor now emits no
  action for such a chord and leaves it to the host. This matches Chromium and
  VS Code, which treat any Alt-held key as a system key and insert nothing. AltGr
  typing is unaffected (winit reports AltGr with neither Ctrl nor Alt set on
  Windows, and as level 3 rather than Alt on Linux). macOS is unchanged: Option
  composes characters (`ƒ`, `é`) and still types.

### Internal

- **scrive-core:** satisfy clippy 1.98's `manual_slice_fill` lint. No behavior
  change.

## 0.3.0 — 2026-07-21

- **scrive-iced:** the batteries-included `CodeEditor` tier. It owns the
  `Document` and integrates through three wires (`update` / `view` /
  `subscription`). It brings the cold-load highlight fix, find/replace, sync and
  async language intelligence, IDE hooks (`load`, `set_theme`,
  `set_diagnostics`, `drain_changes`) and the large-document parallel highlight
  pool. Also bundles the Scrive Dark theme and adds the `minimal` example.
- **scrive-core:** opt-in comment/string-aware bracket matching (`BracketConfig`,
  `Document::set_bracket_lexing`). Also adds `Document::observe_changes` /
  `drain_changes`, `CompletionController::set_items`, and `TokenTheme: Clone`.

## 0.2.0 — 2026-07-18

- Find & replace: literal, whole-word and regex search, `$1` capture groups,
  find-in-selection, case-preserving replace, and replace / replace-all as one
  undo step.
- **Breaking:** `FindQuery` gained the `whole_word` and `regex` fields and is now
  `#[non_exhaustive]`.
- Rope performance: inline chunks and seam coalescing.

## 0.1.0 — 2026-07-16

- Initial release.
