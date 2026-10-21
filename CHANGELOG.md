# Changelog

Notable changes to `scrive-core` and `scrive-iced`. The two crates share one
version and are released together.

## 0.3.3 — 2026-10-21

### Fixed

- **scrive-iced:** the hover popup no longer closes at the slightest pointer
  move when the pointer rests on the right half of a word's last character, or
  just before a diagnostic's squiggle. The test that opens the popup and the
  test that keeps it open now use the same rule: the pointer's caret offset
  touches the range, `start..=end`.
- **scrive-iced:** a collapsed block's fold preview shows the folded pair as one
  piece: the opener, the hidden code, then the closer. It used to start at the
  first hidden row and end with the closing row's full text, so the `{` never
  showed while a trailing `} else {` or `];` did. The opener now sits at the
  closer's indent, so the two line up, and the body keeps its indentation under
  them. A long block's body used to lose that indent. A pair on one row still
  previews its whole row.
- **scrive-iced:** brackets in the fold preview take their pair-depth colours,
  as they do in the editor.

### Changed

Behaviour changes to check when upgrading. No public signatures changed.

- **scrive-iced:** the hover popup and the fold preview behave like a VS Code
  hover. Leaving the word or `…` chip starts a 300 ms hiding delay instead of
  closing at once, so the pointer can move into the panel. Inside it, a drag,
  double-click or triple-click selects text, Ctrl+C copies it as plain text
  (markdown markers removed, wrapped lines rejoined), and a press never reaches
  the code underneath. A panel that has been clicked into stays open until a
  click elsewhere, a key, Escape or an editor scroll.
- **scrive-iced:** the widget publishes `Action::HoverDismiss` in more places:
  when the hiding delay ends, and on a click outside the popup, a key press, or
  an editor scroll. A host driving `Editor` directly should treat it as the one
  signal to close its popup.
- **scrive-iced:** any key except a bare modifier closes an open hover panel.
  Escape closes a clicked-into panel and does nothing else.
- **scrive-iced:** the fold preview scrolls with the wheel and shows a
  scrollbar in place of the "… N more lines" footer.

### Internal

- **scrive-iced:** both panels share one private `panel` module for selection,
  copy, markdown wrapping and the fold preview's line choice. New end-to-end
  tests drive the real widget headlessly with an injected clock.

## 0.3.2 — 2026-10-19

### Fixed

- **scrive-core:** undo and redo restore the whole selection set: every caret
  of a multi-cursor edit, and wherever the command had put them. They used to
  collapse to a single caret.
- **scrive-core:** `drain_changes` now logs undo and redo, and orders ops that
  share a start offset correctly, so a language server mirroring the text stays
  in sync.
- **scrive-core:** host-supplied offsets, ranges and `Point`s are clamped to the
  buffer and snapped to a char boundary. A mid-char or past-the-end position can
  no longer panic or be stored. This covers selections, carets, drag and column
  selection, diagnostics, the find scope, decorations and fold openers.
- **scrive-core:** `Patch::map_offset` / `map_many` with `Bias::Right` map past
  every edit that shares the offset, not just the first.
- **scrive-core:** line-ending detection takes the majority, with ties going to
  LF. One stray CRLF no longer makes a mostly-LF file save as all CRLF.
- **scrive-core** and **scrive-iced:** the highlight cache refuses segments
  tokenized with a previous theme or grammar, and `CodeEditor` restarts its
  highlight pool when the theme changes, so old colours no longer linger after
  `set_theme`.
- **scrive-core:** whole-word find matches needles that start or end with
  punctuation, such as `(foo`, `==` and `-1`.
- **scrive-core:** move line and copy line act on every caret. A whole-line
  (triple-click) selection no longer pulls in the next line.
- **scrive-core:** Tab, Shift+Tab and backspace to a tab stop follow the line's
  indent kind. Tab-indented lines indent and outdent by tabs, and no Tab mixes
  tabs and spaces within an indent.
- **scrive-core:** backspace with several carets in one indentation run now
  deletes back to their shared tab stop instead of doing nothing.
- **scrive-core:** literal find folds case the way regex mode does (Unicode, not
  ASCII only), and preserve-case replace follows VS Code's rule.
- **scrive-core:** `replace_next` moves the selection to the next match within
  its own undo step, so redo shows it there.
- **scrive-core:** folding with several carets, and undo/redo, now move a caret
  out of a collapsed block's closing-line indent, as a single fold toggle
  already did.

### Changed

Behaviour changes to check when upgrading. No public signatures changed.

- **scrive-core:** a highlight engine is matched by identity. Off-thread
  tokenizing must use clones of `Document::highlight_engine`. Segments from a
  separately built `HighlightEngine::new` are refused even when it was built
  from the same grammar and theme.
- **scrive-core:** space indents that sit off the tab-stop grid snap to it:
  `"  foo"` indents to 4 columns, not 6.
- **scrive-core:** whole-word find counts letters, digits and `_` as word
  characters.
- **scrive-core:** a case-insensitive literal query now compiles a regex,
  costing 40–200 µs per query change (not per keystroke). A literal needle too
  large to compile reports a size error.
- **scrive-core:** Tab over a selection that ends before a mixed indent's last
  run deletes the selection and adds the level where block indent would, so
  tabs and spaces never interleave. VS Code types the level in place instead.

### Internal

- **scrive-core:** new `perf_gate` rows for undo/redo across many carets and
  folds, and for multi-caret move and copy line, plus `LEDGER.md` sections with
  the bench numbers.

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
