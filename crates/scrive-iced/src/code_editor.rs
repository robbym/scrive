//! `CodeEditor` — the batteries-included editor tier.
//!
//! [`Editor`] is a *controlled* widget: it borrows a
//! [`Document`] immutably to draw and emits semantic [`Action`]s the host
//! applies. That gives maximum control at the cost of a large amount of
//! plumbing — driving highlighting, find, focus, and the intel controllers is
//! all on the host. [`CodeEditor`] is the other end of that trade: it **owns** a
//! `Document`, runs the plumbing internally, and reduces integration to three
//! wires — [`update`](CodeEditor::update), [`view`](CodeEditor::view),
//! [`subscription`](CodeEditor::subscription) — plus registering
//! [`required_fonts`](crate::required_fonts) at startup.
//!
//! # Mechanism vs policy
//!
//! The split that keeps this honest: **mechanism** (the highlight pump, find
//! rescan, focus rings, autoscroll, the intel drive loops) is owned here, hidden,
//! and always-correct — a host cannot forget a step it never had to take.
//! **Policy** (grammar, theme, providers, sizing) is a builder override with a
//! sensible default. The one genuine fork is ownership: `CodeEditor` owns the
//! `Document` and exposes reads plus *blessed* mutations that run the post-edit
//! tail — there is deliberately no raw `&mut Document`, because that is the
//! trapdoor that lets a caller mutate behind the tail's back (the exact shape of
//! the cold-load highlight bug this tier exists to prevent). A host that must own
//! the buffer itself drops to the [`Editor`] power tier.
//!
//! This module is built up across milestones; M1 is the synchronous core loop
//! (owned document, the three wires, and the cold-load highlight fix). Find,
//! the intel drive loops, the large-document parallel sweep, and the async
//! intel round-trip land on top in later milestones.

use std::ops::Range;
use std::time::Instant;

use iced::advanced::widget;
use iced::alignment::{Horizontal, Vertical};
use iced::widget::operation::{focus, focus_next, focus_previous, is_focused};
use iced::widget::{button, column, container, row, stack, text, text_input};
use iced::{Alignment, Color, Element, Font, Length, Shadow, Subscription, Task, Theme, Vector};

use scrive_core::{
    default_indent_size, is_completion_word_char, CompletionController, CompletionCx, CompletionItem,
    CompletionState, CompletionTrigger, Completions, Diagnostic, DiagnosticsOutcome, Document, EditOp,
    FindQuery, Hover,
    HoverCx, HoverInfo, InsertText, Point, Revision, Selection, SelectionId, SelectionSet, Severity,
    SignatureCx, SignatureHelp, SignatureInfo, Snippet, SnippetSession, SyntaxDef, TabOutcome,
    TokenTheme, LOOKBACK_LINES,
};

use crate::editor::{Action, Editor};
use crate::highlight_pool::{HighlightPool, PARALLEL_MIN_BYTES};

/// The default widget id the editor is addressable by (focus / future
/// multi-pane). Overridable with [`CodeEditor::id`].
const DEFAULT_ID: &str = "scrive-editor";

/// Focusable ids for the find bar's inputs. iced's `focus` operation focuses one
/// and unfocuses every other focusable, so moving focus between an input and the
/// editor is a proper single-focus model.
const FIND_INPUT: &str = "scrive-find-input";
const REPLACE_INPUT: &str = "scrive-replace-input";

/// The opaque message a [`CodeEditor`] emits and consumes. The host never
/// matches on it — it only maps it through the three wires
/// (`self.editor.update(e).map(Message::Editor)` and the same for `view` /
/// `subscription`). It is deliberately opaque: the internal set churns with
/// every refactor, so host-relevant signals come through the curated read
/// accessors and builder callbacks instead. A host that genuinely needs the raw
/// semantic vocabulary drops to the [`Editor`] power tier, where
/// [`Action`] is the stable, match-on-me enum.
#[non_exhaustive]
#[derive(Debug, Clone)]
pub enum Event {
    /// A semantic action published by the underlying widget.
    Editor(Action),
    /// One frame tick of the highlight sweep — the internal pump that drives
    /// tokenization to convergence (and fixes cold load). Emitted by
    /// [`subscription`](CodeEditor::subscription) only while a dirty highlight
    /// frontier remains, so an idle document does zero per-frame work.
    HighlightSweep,
    // ── find bar chrome (internal; the host only maps these through) ─────────
    /// Open the find bar (Ctrl+F), seeding the query from a single-line selection.
    OpenFind,
    /// Open the find bar with the replace row expanded (Ctrl+H).
    OpenReplace,
    /// Close the find bar (Escape), returning focus to the editor.
    CloseFind,
    /// The query text changed.
    FindQuery(String),
    /// Toggle the case-sensitive (`Aa`) option.
    ToggleCase,
    /// Toggle the whole-word (`ab|`) option.
    ToggleWholeWord,
    /// Toggle the regex (`.*`) option.
    ToggleRegex,
    /// Toggle find-in-selection: capture the current selection as the scope.
    ToggleFindInSelection,
    /// Advance to the next match.
    FindNext,
    /// Step to the previous match.
    FindPrev,
    /// Turn every match into a caret (Alt+Enter).
    FindSelectAll,
    /// Toggle the replace row (the chevron).
    ToggleReplace,
    /// The replacement text changed.
    ReplaceText(String),
    /// Replace the active match and advance.
    ReplaceOne,
    /// Replace every match in one undo step.
    ReplaceAll,
    /// Toggle the preserve-case (`AB`) replace option.
    TogglePreserveCase,
    /// Tab / Shift+Tab moved focus between the bar's inputs and the editor.
    CycleFocus {
        /// Whether focus moved backwards (Shift+Tab).
        back: bool,
    },
    /// A left button press landed somewhere — re-assert single focus.
    PointerDown,
    /// A bar input gained or lost native focus; mirror it into the ring flags.
    Focused {
        /// Whether this is the replace input (else the find input).
        replace: bool,
        /// Whether it is now focused.
        on: bool,
    },
}

/// A batteries-included code editor: owns a [`Document`], renders the
/// [`Editor`] widget, and runs the editing/highlighting plumbing
/// internally. See the [module docs](self) for the mechanism-vs-policy split and
/// the ownership fork.
pub struct CodeEditor {
    /// The owned document — the single source of truth. Read through
    /// [`document`](CodeEditor::document); mutated only through the blessed
    /// tail-running methods, never a raw `&mut`.
    doc: Document,
    /// The one viewport fact (last range the widget reported). It aims the
    /// tokenize target and the highlight retention window from a single owner so
    /// they cannot drift.
    viewport: Range<u32>,
    /// The syntax theme (default [`scrive_dark_theme`](crate::scrive_dark_theme)).
    /// Retained (cheaply cloneable) so [`load`](CodeEditor::load) and
    /// [`set_theme`](CodeEditor::set_theme) can re-apply it across a reload or
    /// grammar swap.
    theme: TokenTheme,
    /// Whether a grammar has been attached — marks whether there is a highlight
    /// cache to pump.
    has_syntax: bool,
    /// The editor's widget id (focus addressing).
    id: widget::Id,
    /// Rendering policy.
    font: Font,
    text_size: f32,
    /// Monotonic clock start — the injected `now_ms` for find's debounce.
    start: Instant,
    /// Set by every committed edit; a host reads it for a dirty indicator via
    /// [`is_dirty`](CodeEditor::is_dirty) / [`take_dirty`](CodeEditor::take_dirty).
    dirty: bool,
    /// Whether the built-in find bar (Ctrl+F / Ctrl+H) is available. Default on;
    /// [`find`](CodeEditor::find) disables it.
    find_enabled: bool,
    /// Whether the find bar is currently open, and its live query text.
    find_open: bool,
    find_query: String,
    /// The `Aa` / `ab|` / `.*` options. Each is part of the QUERY, not chrome:
    /// flipping one re-scans exactly like a text change.
    find_case: bool,
    find_whole_word: bool,
    find_regex: bool,
    /// Whether the bar is expanded into find+replace (the chevron), and the live
    /// replacement text.
    replace_open: bool,
    replace_text: String,
    /// Which field wears the focus ring (a container border can't read its
    /// field's focus, so these mirror it).
    find_focused: bool,
    replace_focused: bool,
    /// The `AB` replace option — whether a replacement is re-cased to the match.
    replace_preserve_case: bool,
    // ── language intelligence (view-state + injected providers) ─────────────
    /// The completion controller (popup state) and the host-supplied provider.
    /// `None` provider ⇒ no completions. Driven on edits; its popup passes to the
    /// widget each frame.
    completion: CompletionController,
    comp_provider: Option<Box<dyn Completions>>,
    /// The active snippet tab-stop session, if any.
    snippet: Option<SnippetSession>,
    /// The signature-help provider and the current one-line box.
    sig_provider: Option<Box<dyn SignatureHelp>>,
    signature: Option<SignatureInfo>,
    /// The hover provider and the open hover popup.
    hover_provider: Option<Box<dyn Hover>>,
    hover: Option<HoverInfo>,
    /// A pending async completion request (recorded when the drive loop would
    /// query completions but no synchronous provider is set), for the host to
    /// pull via [`take_completion_request`](CodeEditor::take_completion_request).
    pending_completion_request: Option<CompletionRequest>,
    /// A pending async signature-help request (same pattern as completions).
    pending_signature_request: Option<SignatureRequest>,
    /// A pending async hover request (same pattern), recorded when the pointer
    /// rested over a word and no synchronous hover provider is set.
    pending_hover_request: Option<HoverRequest>,
    /// The off-thread parallel highlight sweep — `Some` for a large document
    /// (`PARALLEL_MIN_BYTES`), `None` for small ones (the synchronous path). Owned
    /// here so a batteries-included host gets large-document highlighting for free.
    hl_pool: Option<HighlightPool>,
}

/// What an applied edit means for the completion controller — computed from the
/// [`Action`] before it is consumed, then threaded through the post-edit tail.
#[derive(Clone, Copy)]
enum CompletionEvent {
    /// A character was typed (opens/filters the popup, or is a trigger char).
    Typed(char),
    /// A deletion (refilters an open popup; closes at the word start).
    Deleting,
    /// Anything else — a caret move, paste, undo… — closes the popup.
    CaretOrClose,
}

/// A request for completions the host fulfills asynchronously (off-thread or via
/// a language server). The editor emits one where it would call a synchronous
/// provider but none is set; the host pulls it with
/// [`CodeEditor::take_completion_request`], runs its query, and returns the
/// result through [`CodeEditor::set_completions`] stamped with `revision`.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct CompletionRequest {
    /// The revision the request was made at — pass it back to `set_completions`
    /// so a result computed against a stale snapshot is dropped.
    pub revision: Revision,
    /// The caret position (row, col) to query completions at.
    pub position: Point,
}

/// A request for signature help the host fulfills asynchronously. Emitted where
/// the editor would call a synchronous [`SignatureHelp`] provider but none is
/// set; the host pulls it with
/// [`CodeEditor::take_signature_request`], queries, and returns the result (or
/// `None` to close the box) through [`CodeEditor::set_signature`].
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct SignatureRequest {
    /// The revision the request was made at — pass it back to `set_signature`.
    pub revision: Revision,
    /// The caret position (row, col) to query signature help at.
    pub position: Point,
}

/// A request for hover documentation the host fulfills asynchronously. Emitted
/// where the editor would call a synchronous [`Hover`] provider but none is set;
/// the host pulls it with [`CodeEditor::take_hover_request`], queries, and
/// returns the card (or `None`) through [`CodeEditor::set_hover`]. Diagnostics at
/// the point are shown synchronously regardless; a host that wants them in the
/// async card can read them from [`CodeEditor::document`] and include them.
#[non_exhaustive]
#[derive(Clone, Debug)]
pub struct HoverRequest {
    /// The revision the request was made at — pass it back to `set_hover`.
    pub revision: Revision,
    /// The byte offset the pointer rested over.
    pub offset: u32,
}

impl CodeEditor {
    /// A new editor over `source`, with the default theme
    /// ([`scrive_dark_theme`](crate::scrive_dark_theme)) but **no grammar** —
    /// plain text until [`language`](CodeEditor::language) attaches one.
    ///
    /// # Panics
    /// Panics if `source` does not fit scrive's `u32` offset space (~4 GiB) —
    /// far past any editable document.
    #[must_use]
    pub fn new(source: impl Into<String>) -> Self {
        let source = source.into();
        let doc = Document::new(&source).expect("source fits the u32 offset space");
        Self {
            doc,
            viewport: 0..0,
            theme: crate::scrive_dark_theme(),
            has_syntax: false,
            id: widget::Id::new(DEFAULT_ID),
            font: Font::MONOSPACE,
            text_size: 14.0,
            start: Instant::now(),
            dirty: false,
            find_enabled: true,
            find_open: false,
            find_query: String::new(),
            find_case: false,
            find_whole_word: false,
            find_regex: false,
            replace_open: false,
            replace_text: String::new(),
            find_focused: false,
            replace_focused: false,
            replace_preserve_case: false,
            completion: CompletionController::new(),
            comp_provider: None,
            snippet: None,
            sig_provider: None,
            signature: None,
            hover_provider: None,
            hover: None,
            pending_completion_request: None,
            pending_signature_request: None,
            pending_hover_request: None,
            hl_pool: None,
        }
    }

    // ── builder (policy; every knob defaulted) ──────────────────────────────

    /// Attach a grammar, enabling syntax highlighting. Uses the theme set by
    /// [`theme`](CodeEditor::theme) if one was staged, else the bundled default.
    /// Order-independent with `theme`. Seeds the whole (small) document's colors
    /// immediately so the first paint is highlighted without waiting for a
    /// viewport report — the cold-load fix.
    #[must_use]
    pub fn language(mut self, grammar: SyntaxDef) -> Self {
        self.doc.set_syntax(grammar, self.theme.clone());
        self.has_syntax = true;
        self.seed_highlight();
        self
    }

    /// Override the syntax theme (default:
    /// [`scrive_dark_theme`](crate::scrive_dark_theme)). Order-independent with
    /// [`language`](CodeEditor::language): before a grammar is attached it is
    /// staged; after, it re-themes the live cache.
    #[must_use]
    pub fn theme(mut self, theme: TokenTheme) -> Self {
        if self.has_syntax {
            self.doc.set_theme(theme.clone());
        }
        self.theme = theme;
        self
    }

    /// Set the (monospace) font. Default [`Font::MONOSPACE`].
    #[must_use]
    pub fn font(mut self, font: Font) -> Self {
        self.font = font;
        self
    }

    /// Set the font size in logical pixels. Default `14.0`.
    #[must_use]
    pub fn text_size(mut self, px: f32) -> Self {
        self.text_size = px;
        self
    }

    /// Give the editor a widget id for focus addressing (default
    /// `"scrive-editor"`). Needed only if a multi-pane host targets focus at it.
    #[must_use]
    pub fn id(mut self, id: impl Into<widget::Id>) -> Self {
        self.id = id.into();
        self
    }

    /// Enable or disable the built-in find bar and its Ctrl+F / Ctrl+H bindings.
    /// Default on.
    #[must_use]
    pub fn find(mut self, enabled: bool) -> Self {
        self.find_enabled = enabled;
        self
    }

    /// Set the line-comment marker used by toggle-comment (e.g. `Some("//")`);
    /// `None` (the default) disables toggle-comment. It also drives comment-aware
    /// bracket matching, so brackets inside `// …` stop being matched / coloured /
    /// folded. Language config, like the grammar — the core ships none.
    #[must_use]
    pub fn line_comment(mut self, marker: Option<&str>) -> Self {
        self.doc.set_line_comment(marker);
        self
    }

    /// Configure comment/string-aware bracket matching's string and char-literal
    /// delimiters — brackets inside a string (e.g. `"{}"`) or char literal are not
    /// matched, coloured, folded, or indent-guided. (Line comments come from
    /// [`line_comment`](CodeEditor::line_comment).) `char_delim` is off by default:
    /// in some languages `'` is ambiguous (Rust lifetimes), so it is opt-in.
    #[must_use]
    pub fn bracket_lexing(mut self, string_delims: Vec<u8>, char_delim: Option<u8>) -> Self {
        self.doc.set_bracket_lexing(string_delims, char_delim);
        self
    }

    /// Supply a completion provider (default: none). The editor drives it on
    /// edits, filters and renders the popup, and applies an accepted item —
    /// including expanding a snippet insert into an interactive tab-stop session.
    #[must_use]
    pub fn completions(mut self, provider: impl Completions + 'static) -> Self {
        self.comp_provider = Some(Box::new(provider));
        self
    }

    /// Supply a hover provider (default: none). The editor queries it on the
    /// idle-hover action and renders the returned card above the word.
    #[must_use]
    pub fn hover(mut self, provider: impl Hover + 'static) -> Self {
        self.hover_provider = Some(Box::new(provider));
        self
    }

    /// Supply a signature-help provider (default: none). `(` opens the box;
    /// while open, edits/moves re-query and a `None` reply closes it.
    #[must_use]
    pub fn signature(mut self, provider: impl SignatureHelp + 'static) -> Self {
        self.sig_provider = Some(Box::new(provider));
        self
    }

    // ── reads (safe; no mutation) ───────────────────────────────────────────

    /// The owned document, for saving / diffing / inspection
    /// (`document().serialize(..)`, `.snapshot()`, `.revision()`). Read-only:
    /// mutations go through the blessed methods so the post-edit tail always runs.
    #[must_use]
    pub fn document(&self) -> &Document {
        &self.doc
    }

    /// Whether an edit has landed since the last [`take_dirty`](CodeEditor::take_dirty).
    #[must_use]
    pub fn is_dirty(&self) -> bool {
        self.dirty
    }

    /// Read and clear the dirty flag — for a host that marks a title bar or
    /// schedules a save on change.
    pub fn take_dirty(&mut self) -> bool {
        std::mem::take(&mut self.dirty)
    }

    /// The primary caret position as (row, col).
    #[must_use]
    pub fn cursor(&self) -> Point {
        let head = self.doc.selections().newest().head();
        self.doc.buffer().offset_to_point(head)
    }

    /// The primary selection's byte range (empty at a bare caret).
    #[must_use]
    pub fn selection(&self) -> Range<u32> {
        let s = self.doc.selections().newest();
        s.start()..s.end()
    }

    // ── blessed mutations (run the post-edit tail) ──────────────────────────

    /// Apply a programmatic batch of edits as one transaction, then run the
    /// post-edit tail. The tail is why this exists instead of a raw `&mut
    /// Document`: it keeps highlighting, find, and the intel controllers current.
    pub fn edit(&mut self, ops: Vec<EditOp>) {
        let before = self.doc.revision();
        if self.doc.edit(ops).is_ok() {
            self.after_edit(CompletionEvent::CaretOrClose);
            if self.doc.revision() != before {
                self.dirty = true;
            }
        }
    }

    /// Swap the whole buffer (load a new file), keeping the current grammar and
    /// theme unless `grammar` supplies a new language. The blessed whole-buffer
    /// swap: it runs as one transaction and re-tokenizes the visible document, so
    /// highlighting is correct immediately — never mutate the buffer behind the
    /// tail's back. (The swap is a normal transaction, so it is undoable.)
    pub fn load(&mut self, source: impl Into<String>, grammar: Option<SyntaxDef>) {
        let source = source.into();
        // Replace the whole buffer as one transaction; the grammar, theme, and
        // highlight cache stay attached and re-tokenize via `on_commit`.
        let len = self.doc.buffer().len();
        let _ = self.doc.edit(vec![EditOp::new(0..len, &source)]);
        if let Some(g) = grammar {
            self.doc.set_syntax(g, self.theme.clone());
            self.has_syntax = true;
        }
        // Re-seed: the document may have grown/shrunk, a grammar swap left the
        // cache all-dirty, and a new grammar needs a fresh pool engine (fixing the
        // grammar-swap-strands-the-pool bug).
        self.seed_highlight();
        self.dirty = true;
    }

    /// Collapse every foldable region (the "Fold All" command). Folds are view
    /// state, so this records nothing on the undo stack and runs no post-edit tail.
    pub fn fold_all(&mut self) {
        for (open, ..) in self.doc.collapsible_pairs() {
            self.doc.toggle_fold_opener(open);
        }
    }

    /// Swap the syntax theme at runtime. The retained theme updates too, so a
    /// later [`load`](CodeEditor::load) keeps it.
    pub fn set_theme(&mut self, theme: TokenTheme) {
        self.doc.set_theme(theme.clone());
        self.theme = theme;
        self.doc.tokenize_highlight(self.viewport.end);
    }

    /// Publish a diagnostic set from the host's (debounced, off-thread) compile
    /// or language-server pass. Stamped by `revision`: a set computed against a
    /// snapshot the buffer has moved past is dropped (the previous squiggles keep
    /// riding edits). This is the ingest half of recompile-on-edit — pair it with
    /// an edit signal ([`take_dirty`](CodeEditor::take_dirty) or a revision
    /// compare) and [`document`](CodeEditor::document)`().snapshot()`.
    pub fn set_diagnostics(
        &mut self,
        revision: Revision,
        diags: Vec<Diagnostic>,
    ) -> DiagnosticsOutcome {
        self.doc.set_diagnostics(revision, diags)
    }

    /// Take the pending async completion request, if any. A host with an
    /// off-thread / language-server provider (rather than a synchronous
    /// [`completions`](CodeEditor::completions) one) polls this after `update`,
    /// runs its query at the request's position and revision, and returns the
    /// result through [`set_completions`](CodeEditor::set_completions).
    pub fn take_completion_request(&mut self) -> Option<CompletionRequest> {
        self.pending_completion_request.take()
    }

    /// Ingest completion items from an async provider, stamped by `revision`.
    /// Strict staleness: if the buffer has moved since the request the items are
    /// dropped (the host re-requests on the next edit). On a current revision they
    /// open/refilter the popup against the *live* word — including snippet-format
    /// items ([`InsertText::Snippet`](scrive_core::InsertText)), which expand into
    /// an interactive tab-stop session on accept exactly like a synchronous
    /// provider's.
    pub fn set_completions(&mut self, revision: Revision, items: Vec<CompletionItem>) {
        if revision != self.doc.revision() {
            return; // stale — the host re-requests on the next edit
        }
        let word = self.completion_word_text();
        let anchor = self.completion_word().start;
        self.completion.set_items(items, &word, anchor);
    }

    /// Take the pending async signature-help request, if any — the signature
    /// twin of [`take_completion_request`](CodeEditor::take_completion_request).
    pub fn take_signature_request(&mut self) -> Option<SignatureRequest> {
        self.pending_signature_request.take()
    }

    /// Ingest a signature-help result from an async provider, stamped by
    /// `revision` (strict staleness). `None` closes the box. Current-revision
    /// results replace the shown signature.
    pub fn set_signature(&mut self, revision: Revision, info: Option<SignatureInfo>) {
        if revision == self.doc.revision() {
            self.signature = info;
        }
    }

    /// Take the pending async hover request, if any — the hover twin of
    /// [`take_completion_request`](CodeEditor::take_completion_request).
    pub fn take_hover_request(&mut self) -> Option<HoverRequest> {
        self.pending_hover_request.take()
    }

    /// Ingest a hover card from an async provider, stamped by `revision` (strict
    /// staleness). Replaces the shown card (the host builds it, and may fold in
    /// diagnostics read from [`document`](CodeEditor::document)); `None` clears it.
    pub fn set_hover(&mut self, revision: Revision, info: Option<HoverInfo>) {
        if revision == self.doc.revision() {
            self.hover = info;
        }
    }

    /// Enable or disable the incremental-change log — for a host mirroring edits
    /// to a language server (`textDocument/didChange`). Off by default (zero
    /// overhead); forwards to [`Document::observe_changes`]. Full-document sync
    /// hosts leave this off and re-read `document().snapshot()` instead.
    pub fn observe_changes(&mut self, on: bool) {
        self.doc.observe_changes(on);
    }

    /// Drain the incremental-change log: every applied edit since the last drain,
    /// as `EditOp` deltas ready to translate into LSP content changes. Empty
    /// unless [`observe_changes`](CodeEditor::observe_changes) is on. Forwards to
    /// [`Document::drain_changes`].
    pub fn drain_changes(&mut self) -> Vec<EditOp> {
        self.doc.drain_changes()
    }

    // ── the three wires ─────────────────────────────────────────────────────

    /// Fold one [`Event`] into the editor. Map it back to your message type:
    /// `Message::Editor(e) => self.editor.update(e).map(Message::Editor)`.
    pub fn update(&mut self, event: Event) -> Task<Event> {
        match event {
            // The widget reported a new visible range (scroll / resize /
            // autoscroll). Aim the retention window there and tokenize down to
            // it. Not an edit — no history, no find rescan — so it bypasses
            // `apply`.
            Event::Editor(Action::ViewportChanged(rows)) => {
                self.viewport = rows.clone();
                self.doc.set_highlight_window(rows.clone());
                if self.doc.buffer().len() >= PARALLEL_MIN_BYTES {
                    // Large document: the off-thread sweep owns dirt-clearing; the
                    // viewport is painted synchronously now and verified in place.
                    // Do NOT run the whole-doc synchronous walk (it would race the
                    // pool). Create the pool on first sight.
                    if let Some(mut pool) = self.hl_pool.take() {
                        pool.reaim(&mut self.doc, rows);
                        self.hl_pool = Some(pool);
                    } else if let Some(pool) = HighlightPool::new(&self.doc, rows.clone()) {
                        pool.speculate(&mut self.doc, rows);
                        self.hl_pool = Some(pool);
                    }
                } else {
                    self.doc.tokenize_highlight(rows.end);
                }
                self.hover = None; // scroll closes the hover
                Task::none()
            }
            // Escape with the bar open closes the BAR and keeps the selections
            // (matching mainstream editors) — this covers the editor-focused
            // press; the input-focused one arrives as `CloseFind` via the chord.
            Event::Editor(Action::Collapse) if self.find_open => {
                self.close_find();
                Task::none()
            }
            // Folds are view state (no text change, no undo step, no rehighlight);
            // handled here rather than through the edit tail.
            Event::Editor(Action::ToggleFold { opener }) => {
                self.doc.toggle_fold_opener(opener);
                Task::none()
            }
            Event::Editor(Action::FoldAtCarets { unfold }) => {
                self.doc.fold_at_carets(unfold);
                Task::none()
            }
            // Completion popup navigation (captured by the widget while open) —
            // drive the controller; no document edit except on accept.
            Event::Editor(Action::PopupUp) => {
                self.completion.move_selection(false);
                Task::none()
            }
            Event::Editor(Action::PopupDown) => {
                self.completion.move_selection(true);
                Task::none()
            }
            Event::Editor(Action::PopupDismiss) => {
                self.completion.escape();
                Task::none()
            }
            Event::Editor(Action::PopupAccept) => {
                self.accept_completion();
                Task::none()
            }
            Event::Editor(Action::PopupClickAccept(idx)) => {
                self.completion.set_selected(idx);
                self.accept_completion();
                Task::none()
            }
            // Snippet tab-stop navigation (captured while a session is active).
            Event::Editor(Action::SnippetTab) => {
                self.snippet_tab(true);
                Task::none()
            }
            Event::Editor(Action::SnippetTabPrev) => {
                self.snippet_tab(false);
                Task::none()
            }
            Event::Editor(Action::SnippetCancel) => {
                if let Some(mut s) = self.snippet.take() {
                    s.cancel(self.doc.decorations_mut());
                }
                Task::none()
            }
            Event::Editor(Action::SignatureClose) => {
                self.signature = None;
                Task::none()
            }
            // Hover: the pointer rested over `offset` — diagnostics first (their
            // messages ride the decoration store), then the provider's docs.
            Event::Editor(Action::HoverQuery(offset)) => {
                let diags: Vec<(Range<u32>, String)> = self
                    .doc
                    .diagnostics_in(offset..offset + 1)
                    .map(|(r, sev, msg)| (r, format!("**{}:** {msg}", severity_label(sev))))
                    .collect();
                let cx = self.build_hover_cx(offset);
                let word = (cx.word.start != cx.word.end)
                    .then(|| self.hover_provider.as_mut().and_then(|p| p.hover(&cx)))
                    .flatten();
                self.hover = if diags.is_empty() {
                    word
                } else {
                    let range = diags[0].0.clone();
                    let mut md: Vec<String> = diags.into_iter().map(|(_, m)| m).collect();
                    if let Some(w) = word {
                        md.push(String::new());
                        md.push(w.markdown);
                    }
                    Some(HoverInfo { markdown: md.join("\n"), range })
                };
                // With no synchronous provider, record an async request so the
                // host can supply docs; the diagnostics-only card (if any) shows
                // meanwhile and `set_hover` replaces it when the docs arrive.
                if self.hover_provider.is_none() && cx.word.start != cx.word.end {
                    self.pending_hover_request =
                        Some(HoverRequest { revision: self.doc.revision(), offset });
                }
                Task::none()
            }
            Event::Editor(Action::HoverDismiss) => {
                self.hover = None;
                Task::none()
            }
            Event::Editor(action) => {
                self.apply(action);
                Task::none()
            }
            // The idle sweep: one budgeted batch per frame toward convergence.
            // The subscription drops itself once the frontier is clean, so this
            // stops firing on an idle document.
            Event::HighlightSweep => {
                if self.doc.buffer().len() >= PARALLEL_MIN_BYTES {
                    if let Some(mut pool) = self.hl_pool.take() {
                        if pool.rev != self.doc.revision() {
                            // An edit landed: re-sweep from a fresh snapshot AND
                            // repaint the viewport now (the verified prefix in the
                            // cache survives).
                            pool.restart(&mut self.doc, self.viewport.clone());
                        } else {
                            // Drain finished jobs and advance the verified chain.
                            pool.poll(&mut self.doc);
                        }
                        let idle = !pool.active;
                        self.hl_pool = Some(pool);
                        // Once the sweep is idle, the synchronous phase-2 path
                        // refills any window rows the sweep evicted (dirt is
                        // cleared, so this is a cheap window refill, not O(doc)).
                        if idle {
                            let n = self.doc.buffer().line_count();
                            self.doc.tokenize_highlight(n);
                        }
                    } else if let Some(pool) = HighlightPool::new(&self.doc, self.viewport.clone()) {
                        // Large but no pool yet (grew past the threshold): create
                        // it rather than tokenize the whole document synchronously.
                        self.hl_pool = Some(pool);
                    }
                } else {
                    // Small document (or shrunk below the threshold): deactivate
                    // any lingering pool and drive the synchronous path.
                    if let Some(pool) = &mut self.hl_pool {
                        pool.active = false;
                    }
                    let n = self.doc.buffer().line_count();
                    self.doc.tokenize_highlight(n);
                }
                Task::none()
            }

            // ── find bar ────────────────────────────────────────────────────
            Event::OpenFind if self.find_enabled => {
                self.find_open = true;
                // Seed (or re-seed) the query from a non-empty, single-line
                // selection, as mainstream editors do; an empty selection leaves
                // the current query untouched.
                let sel = self.doc.selections().newest();
                let seed = (!sel.is_empty())
                    .then(|| self.doc.buffer().slice(sel.start()..sel.end()).into_owned())
                    .filter(|t| !t.contains('\n'));
                if let Some(text) = seed {
                    self.find_query = text;
                    self.push_find_query();
                }
                self.find_focused = true;
                self.replace_focused = false;
                focus(FIND_INPUT)
            }
            Event::OpenReplace if self.find_enabled => {
                // Ctrl+H is Ctrl+F with the replace row already out.
                self.replace_open = true;
                self.update(Event::OpenFind)
            }
            Event::CycleFocus { back } => {
                let moved = if back { focus_previous() } else { focus_next() };
                moved.chain(Self::sync_rings())
            }
            Event::PointerDown if self.find_open => {
                Task::batch([Self::resync_focus(), Self::sync_rings()])
            }
            Event::Focused { replace, on } => {
                if replace {
                    self.replace_focused = on;
                } else {
                    self.find_focused = on;
                }
                Task::none()
            }
            Event::CloseFind if self.find_open => {
                self.close_find();
                focus(self.id.clone())
            }
            Event::FindQuery(q) => {
                self.find_query = q;
                self.push_find_query();
                Task::none()
            }
            Event::ToggleCase => {
                self.find_case = !self.find_case;
                self.push_find_query();
                Task::none()
            }
            Event::ToggleWholeWord => {
                self.find_whole_word = !self.find_whole_word;
                self.push_find_query();
                Task::none()
            }
            Event::ToggleRegex => {
                self.find_regex = !self.find_regex;
                self.push_find_query();
                Task::none()
            }
            Event::ToggleFindInSelection if self.find_open => {
                // The scope lives in the document (it rides every edit), so read
                // the live one and set its opposite.
                let scope = if self.doc.find_scope().is_some() {
                    None
                } else {
                    let sel = self.doc.selections().newest();
                    (!sel.is_empty()).then(|| sel.start()..sel.end())
                };
                let now = self.now_ms();
                self.doc.set_find_scope(scope, now);
                Task::none()
            }
            Event::FindNext if self.find_open => {
                let now = self.now_ms();
                self.doc.find_next(now);
                Task::none()
            }
            Event::FindPrev if self.find_open => {
                let now = self.now_ms();
                self.doc.find_prev(now);
                Task::none()
            }
            Event::FindSelectAll if self.find_open => {
                // Every match becomes a caret; focus returns to the editor.
                if self.doc.select_find_matches() {
                    return focus(self.id.clone());
                }
                Task::none()
            }
            Event::ToggleReplace => {
                self.replace_open = !self.replace_open;
                self.replace_focused = self.replace_open;
                self.find_focused = !self.replace_open;
                focus(if self.replace_open { REPLACE_INPUT } else { FIND_INPUT })
            }
            Event::ReplaceText(t) => {
                self.replace_text = t;
                Task::none()
            }
            Event::ReplaceOne if self.find_open => {
                let now = self.now_ms();
                let before = self.doc.revision();
                self.doc.replace_next(&self.replace_text, self.replace_preserve_case, now);
                // The first press only NAVIGATES (shows the match before
                // overwriting it), which commits nothing — run the tail only if a
                // replacement actually landed.
                if self.doc.revision() != before {
                    self.after_edit(CompletionEvent::CaretOrClose);
                    self.dirty = true;
                }
                Task::none()
            }
            Event::ReplaceAll if self.find_open => {
                if self.doc.replace_all(&self.replace_text, self.replace_preserve_case) > 0 {
                    self.after_edit(CompletionEvent::CaretOrClose);
                    self.dirty = true;
                }
                Task::none()
            }
            Event::TogglePreserveCase => {
                self.replace_preserve_case = !self.replace_preserve_case;
                Task::none()
            }
            // Guarded find variants whose guard did not hold (bar closed, or find
            // disabled): ignore.
            Event::OpenFind
            | Event::OpenReplace
            | Event::CloseFind
            | Event::FindNext
            | Event::FindPrev
            | Event::FindSelectAll
            | Event::ToggleFindInSelection
            | Event::ReplaceOne
            | Event::ReplaceAll
            | Event::PointerDown => Task::none(),
        }
    }

    /// The editor element. Map it back to your message type:
    /// `self.editor.view().map(Message::Editor)`.
    #[must_use]
    pub fn view(&self) -> Element<'_, Event> {
        let popup = match self.completion.state() {
            CompletionState::Open(list) => Some(list),
            _ => None,
        };
        let editor = Editor::new(&self.doc, Event::Editor)
            .popup(popup)
            .snippet_active(self.snippet.is_some())
            .signature(self.signature.as_ref())
            .hover(self.hover.as_ref())
            .font(self.font)
            .text_size(self.text_size)
            .id(self.id.clone());
        if self.find_open {
            // Float the find bar over the editor, top-right (where mainstream
            // editors place it). The right padding clears the scrollbar lane so
            // the bar never sits over it; the overlay is transparent except the
            // bar, so clicks pass through.
            let overlay = container(self.find_bar())
                .width(Length::Fill)
                .align_x(Horizontal::Right)
                .padding(iced::Padding::new(8.0).right(8.0 + crate::SCROLLBAR_WIDTH));
            stack([editor.into(), overlay.into()]).into()
        } else {
            editor.into()
        }
    }

    /// The find bar: a floating panel with a query row (input + option toggles +
    /// match count + prev/next/scope/close) and, once the chevron expands it, a
    /// replace row (input + preserve-case + replace/replace-all).
    fn find_bar(&self) -> Element<'_, Event> {
        let count = self.doc.find_match_count();
        // A half-typed regex (`(`, `[a-`) is a NORMAL state, not a failure — but
        // it must say so rather than read as "No results".
        let invalid = self.doc.find_pattern_error().is_some();
        let label = if invalid {
            "Invalid regex".to_string()
        } else {
            match self.doc.active_find_match() {
                Some(i) => format!("{} of {}", i + 1, count),
                None if count > 0 => format!("{count} matches"),
                None if self.find_query.is_empty() => String::new(),
                None => "No results".to_string(),
            }
        };
        let label_color = if invalid || label == "No results" {
            Color::from_rgb8(0xF4, 0x87, 0x71)
        } else {
            Color::from_rgb8(0xCC, 0xCC, 0xCC)
        };
        // Both inputs share this width so the two boxes align under each other.
        const INPUT_W: f32 = 264.0;
        // Fixed so the nav buttons never shuffle as the match-count digits change.
        const COUNT_W: f32 = 78.0;
        const SPACING: f32 = 4.0;
        // One row's height, pinned so the chevron can span the rows exactly.
        const ROW_H: f32 = 26.0;
        // The box look lives on the CONTAINER (`box_of`), so the field is
        // transparent and sits inside it beside the in-box buttons as a row
        // sibling — not overlaid in a `Stack` (which early-returns on capture and
        // would leave a field stale-focused). One style for BOTH fields.
        let input_style = |_theme: &Theme, _status: text_input::Status| text_input::Style {
            background: Color::TRANSPARENT.into(),
            border: iced::border::rounded(0.0),
            icon: Color::from_rgb8(0xCC, 0xCC, 0xCC),
            placeholder: Color::from_rgb8(0xA6, 0xA6, 0xA6),
            value: Color::from_rgb8(0xCC, 0xCC, 0xCC),
            selection: Color::from_rgb8(0x26, 0x4F, 0x78),
        };
        const BTN_W: f32 = 22.0;
        const IN_BTN: f32 = 20.0;
        const IN_GAP: f32 = 3.0;
        const IN_MARGIN: f32 = 3.0;
        // Flat icon buttons: transparent at rest, translucent gray on hover; `on`
        // latches the background + a focus-blue border so an engaged option reads
        // as pressed at rest, not only under the pointer.
        let sized_btn = |glyph: char, on: bool, msg, w: f32, h: f32| {
            button(
                text(glyph.to_string())
                    .font(crate::CODICON)
                    .size(15)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Horizontal::Center)
                    .align_y(Vertical::Center),
            )
            .width(Length::Fixed(w))
            .height(Length::Fixed(h))
            .padding(0)
            .on_press(msg)
            .style(move |_theme, status| {
                let hover = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style {
                    background: (on || hover).then(|| Color::from_rgba8(90, 93, 94, 0.314).into()),
                    text_color: Color::from_rgb8(0xCC, 0xCC, 0xCC),
                    border: if on {
                        iced::border::rounded(5.0).width(1.0).color(Color::from_rgb8(0x00, 0x7F, 0xD4))
                    } else {
                        iced::border::rounded(5.0)
                    },
                    shadow: Shadow::default(),
                    snap: true,
                }
            })
        };
        let icon_btn = |glyph: char, on: bool, msg| sized_btn(glyph, on, msg, BTN_W, BTN_W);
        let btn = |glyph: char, msg| sized_btn(glyph, false, msg, BTN_W, BTN_W);
        let in_btn = |glyph: char, on: bool, msg| sized_btn(glyph, on, msg, IN_BTN, IN_BTN);
        let box_of =
            |field, buttons, focused| box_of(field, buttons, focused, INPUT_W, ROW_H, IN_GAP, IN_MARGIN);

        let find_box = box_of(
            text_input("Find", &self.find_query)
                .id(FIND_INPUT)
                .on_input(Event::FindQuery)
                .on_submit(Event::FindNext)
                .padding(iced::Padding::new(4.0).left(2.0))
                .size(13)
                .width(Length::Fill)
                .style(input_style)
                .into(),
            vec![
                in_btn(crate::icon::CASE_SENSITIVE, self.find_case, Event::ToggleCase).into(),
                in_btn(crate::icon::WHOLE_WORD, self.find_whole_word, Event::ToggleWholeWord).into(),
                in_btn(crate::icon::REGEX, self.find_regex, Event::ToggleRegex).into(),
            ],
            self.find_focused,
        );
        let find_row = row![
            find_box,
            text(label)
                .size(12)
                .color(label_color)
                .width(Length::Fixed(COUNT_W))
                .align_x(Horizontal::Left),
            btn(crate::icon::ARROW_UP, Event::FindPrev),
            btn(crate::icon::ARROW_DOWN, Event::FindNext),
            icon_btn(crate::icon::SELECTION, self.scoped(), Event::ToggleFindInSelection),
            btn(crate::icon::CLOSE, Event::CloseFind),
        ]
        .spacing(SPACING)
        .height(Length::Fixed(ROW_H))
        .align_y(Alignment::Center);
        let rows = if self.replace_open {
            let replace_box = box_of(
                text_input("Replace", &self.replace_text)
                    .id(REPLACE_INPUT)
                    .on_input(Event::ReplaceText)
                    .on_submit(Event::ReplaceOne)
                    .padding(iced::Padding::new(4.0).left(2.0))
                    .size(13)
                    .width(Length::Fill)
                    .style(input_style)
                    .into(),
                vec![in_btn(crate::icon::PRESERVE_CASE, self.replace_preserve_case, Event::TogglePreserveCase)
                    .into()],
                self.replace_focused,
            );
            let replace_row = row![
                replace_box,
                btn(crate::icon::REPLACE, Event::ReplaceOne),
                btn(crate::icon::REPLACE_ALL, Event::ReplaceAll),
            ]
            .spacing(SPACING)
            .height(Length::Fixed(ROW_H))
            .align_y(Alignment::Center);
            column![find_row, replace_row].spacing(SPACING)
        } else {
            column![find_row]
        };
        container(
            row![
                // The chevron spans both rows — its hit target grows with the bar
                // it toggles, VS Code's shape and the honest one (it acts on the
                // panel, not the query row it sits next to).
                sized_btn(
                    if self.replace_open { crate::icon::CHEVRON_DOWN } else { crate::icon::CHEVRON_RIGHT },
                    false,
                    Event::ToggleReplace,
                    BTN_W,
                    if self.replace_open { ROW_H * 2.0 + SPACING } else { ROW_H },
                ),
                rows,
            ]
            .spacing(SPACING)
            .align_y(Alignment::Center),
        )
        .padding([6, 8])
        .style(|_theme: &Theme| container::Style {
            background: Some(Color::from_rgb8(0x25, 0x25, 0x26).into()),
            border: iced::border::rounded(8.0).color(Color::from_rgb8(0x45, 0x45, 0x45)).width(1.0),
            shadow: Shadow {
                color: Color::from_rgba8(0, 0, 0, 0.36),
                offset: Vector::new(0.0, 2.0),
                blur_radius: 8.0,
            },
            ..container::Style::default()
        })
        .into()
    }

    /// The editor's subscription — the frames-gated highlight sweep. It runs
    /// only while a dirty highlight frontier remains, and because
    /// [`language`](CodeEditor::language) / an edit leaves the frontier dirty it
    /// fires on the very next frame with no input required (this is what makes
    /// highlighting appear at load instead of after the first scroll). At
    /// convergence it returns [`Subscription::none`], so an idle document does
    /// zero per-frame work. Map it: `self.editor.subscription().map(Message::Editor)`.
    pub fn subscription(&self) -> Subscription<Event> {
        let sweeping = self.doc.highlight_frontier().is_some()
            || self.hl_pool.as_ref().is_some_and(|p| p.active);
        let sweep = if sweeping {
            iced::window::frames().map(|_| Event::HighlightSweep)
        } else {
            Subscription::none()
        };
        // Find keys are chrome: caught here regardless of capture status, so
        // Escape closes the bar in one press even though the native input
        // captured it. `listen_with` filters, so non-find keys produce nothing
        // and the editor's own captured keystrokes still route through the widget.
        // The closure is a plain fn (non-capturing), so the `find_open` gating
        // happens in `update`.
        let keys = if self.find_enabled {
            iced::event::listen_with(|event, status, _window| match event {
                iced::Event::Keyboard(iced::keyboard::Event::KeyPressed { key, modifiers, .. }) => {
                    find_chord(&key, modifiers, status)
                }
                // A press can move focus natively, behind the app's back, leaving
                // two widgets focused — watched here because the press is captured
                // by the input and never surfaces as a widget callback.
                iced::Event::Mouse(iced::mouse::Event::ButtonPressed(iced::mouse::Button::Left)) => {
                    Some(Event::PointerDown)
                }
                _ => None,
            })
        } else {
            Subscription::none()
        };
        Subscription::batch([sweep, keys])
    }

    // ── internals ───────────────────────────────────────────────────────────

    /// Milliseconds since construction — the injected clock for find's debounce.
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// Colour the document at load (or after a grammar swap / buffer load): the
    /// large-document parallel sweep for a big buffer, else a synchronous seed of
    /// the whole (small) buffer. The cold-load fix — the first paint is coloured
    /// with no viewport report required, and a huge buffer never blocks the UI
    /// thread (its pool sweeps in the background, aimed at the current viewport,
    /// which the first `ViewportChanged` reaims). Recreating the pool here also
    /// picks up a new grammar's engine after a `load` language swap.
    fn seed_highlight(&mut self) {
        if self.doc.buffer().len() >= PARALLEL_MIN_BYTES {
            self.hl_pool = HighlightPool::new(&self.doc, self.viewport.clone());
        } else {
            self.hl_pool = None;
            let n = self.doc.buffer().line_count();
            self.doc.set_highlight_window(0..n);
            self.doc.tokenize_highlight(n);
        }
    }

    /// Push the live query text + its options into the document — the ONE place a
    /// [`FindQuery`] is built. Every input that changes what matches (the text
    /// and each option toggle) routes here, so the bar's controls cannot disagree
    /// about the live query. Empty text means "no query", never match-all.
    fn push_find_query(&mut self) {
        let query = (!self.find_query.is_empty()).then(|| {
            // `FindQuery` is `#[non_exhaustive]`: build via `new` + the fields.
            let mut q = FindQuery::new(self.find_query.clone());
            q.case_sensitive = self.find_case;
            q.whole_word = self.find_whole_word;
            q.regex = self.find_regex;
            q
        });
        let now = self.now_ms();
        self.doc.set_find_query(query, now);
    }

    /// Close the bar and drop its query. Also collapses the replace row so the
    /// next open starts from a clean shape (Ctrl+F find-only, Ctrl+H with replace).
    fn close_find(&mut self) {
        self.find_open = false;
        self.replace_open = false;
        self.find_focused = false;
        self.replace_focused = false;
        self.find_query.clear();
        let now = self.now_ms();
        self.doc.set_find_query(None, now);
    }

    /// Whether find is currently scoped to a selection (the find-in-selection
    /// toggle latches off the document's live scope, not an app-side copy).
    fn scoped(&self) -> bool {
        self.doc.find_scope().is_some()
    }

    /// Read the live widget focus back into the ring flags — for focus changes
    /// that can't be predicted here (a click, a Tab).
    fn sync_rings() -> Task<Event> {
        Task::batch([
            is_focused(FIND_INPUT).map(|on| Event::Focused { replace: false, on }),
            is_focused(REPLACE_INPUT).map(|on| Event::Focused { replace: true, on }),
        ])
    }

    /// Re-assert single focus after a press: a click inside an input focuses it
    /// natively and captures the event, so the editor's own press handler never
    /// runs to unfocus itself — focusing whichever input iced reports as focused
    /// is exactly the repair. When the press landed in the editor, neither input
    /// is focused and both arms are no-ops.
    fn resync_focus() -> Task<Event> {
        Task::batch([
            is_focused(FIND_INPUT).then(|f| if f { focus(FIND_INPUT) } else { Task::none() }),
            is_focused(REPLACE_INPUT).then(|f| if f { focus(REPLACE_INPUT) } else { Task::none() }),
        ])
    }

    /// Apply one editing/selection [`Action`] to the document, then run the
    /// post-edit tail. The intel / popup / snippet / hover / viewport / fold
    /// actions are handled before `apply` (or are no-ops here) and land as the
    /// catch-all arm; they gain behavior as later milestones wire the controllers.
    fn apply(&mut self, action: Action) {
        // Capture what this action means for completion before the match consumes
        // `action`.
        let comp_event = match &action {
            Action::Type(c) => CompletionEvent::Typed(*c),
            Action::Backspace | Action::DeleteWordBack | Action::Delete | Action::DeleteWordForward => {
                CompletionEvent::Deleting
            }
            _ => CompletionEvent::CaretOrClose,
        };
        let rev_before = self.doc.revision();
        match action {
            Action::Type(ch) => self.doc.type_char(ch),
            Action::Backspace => self.doc.backspace(),
            Action::Delete => self.doc.delete_forward(),
            Action::DeleteWordBack => self.doc.delete_word_back(),
            Action::DeleteWordForward => self.doc.delete_word_forward(),
            Action::Enter => self.doc.enter(),
            Action::Tab => self.doc.tab(),
            Action::Outdent => self.doc.outdent(),
            Action::ToggleComment => self.doc.toggle_line_comment(),
            Action::DeleteLine => self.doc.delete_line(),
            Action::InsertLine { down } => self.doc.insert_line(down),
            Action::Cut => self.doc.cut(),
            Action::Paste { text, entire_line } => self.doc.paste(&text, entire_line),
            Action::Move { motion, extend } => self.doc.move_carets(motion, extend),
            Action::PlaceCaret(offset) => {
                let mut set = SelectionSet::new(0);
                set.set_single(Selection::caret(SelectionId(0), offset));
                self.doc.set_selections(set);
            }
            Action::DragSelect { granularity, origin, head } => {
                self.doc.drag_select(granularity, origin, head);
            }
            Action::AddCaret(offset) => self.doc.add_caret(offset),
            Action::AddNextOccurrence => self.doc.add_next_occurrence(),
            Action::SelectAllOccurrences => self.doc.select_all_occurrences(),
            Action::AddCaretVertical { down } => self.doc.add_caret_vertical(down),
            Action::JumpToBracket => self.doc.jump_to_bracket(),
            Action::NextDiagnostic { forward } => {
                self.doc.next_diagnostic(forward);
            }
            Action::ExpandSelection => self.doc.expand_selection(),
            Action::ShrinkSelection => self.doc.shrink_selection(),
            Action::Collapse => self.doc.collapse_selections(),
            Action::SelectAll => self.doc.select_all(),
            Action::Undo => {
                self.doc.undo();
            }
            Action::Redo => {
                self.doc.redo();
            }
            Action::ColumnSelect(dir) => self.doc.column_select(dir),
            Action::ColumnDrag { anchor, active } => self.doc.column_drag(anchor, active),
            Action::MoveLine { down } => self.doc.move_line(down),
            Action::CopyLine { down } => self.doc.copy_line(down),
            // Handled in `update` (viewport, folds) or not yet wired (popup /
            // snippet / signature / hover — later milestones). Exhaustive no-op.
            Action::ViewportChanged(_)
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
            | Action::FoldAtCarets { .. } => {}
        }
        self.after_edit(comp_event);
        // Dirty only on an actual text change — a bare caret move / selection must
        // not schedule a host recompile or flip the save indicator.
        if self.doc.revision() != rev_before {
            self.dirty = true;
        }
    }

    /// Everything that must follow a document mutation — the ONE owner of the
    /// post-edit tail. Every edit entry point runs it, so no second entry point
    /// can silently do half of it (a path that skipped `tokenize_highlight` would
    /// paint stale colors; one that skipped `drive_completion` would strand the
    /// popup). The `on_edit` host signal joins it in a later milestone.
    fn after_edit(&mut self, comp_event: CompletionEvent) {
        // Bring the highlight cache current down to the reported viewport bottom
        // only — convergence stops at the edited lines for a normal edit, and the
        // viewport bound caps a state cascade to the screen.
        self.doc.tokenize_highlight(self.viewport.end);
        // Keep find fresh while editing: matches ride the edit via the decoration
        // mover; a debounced re-scan picks up appearing/disappearing matches.
        let now = self.now_ms();
        self.doc.maybe_rescan_find(now);
        // Drive completion (typing opens/filters, deleting refilters, else close),
        // signature help, and reconcile the snippet session; any edit closes hover.
        self.drive_completion(comp_event);
        self.drive_signature(comp_event);
        self.reconcile_snippet();
        self.hover = None;
        // `dirty` is set by the callers on an actual text change (a bare caret
        // move runs the tail but must not dirty the document — see `apply`).
    }

    /// Drive the completion controller after an edit. The controller action is
    /// decided from the event here; whether it runs a synchronous provider or
    /// records an async request is [`request_completions`](Self::request_completions)'s
    /// job. No-op unless the host wired completions (sync provider or async pull).
    fn drive_completion(&mut self, event: CompletionEvent) {
        match event {
            CompletionEvent::Typed(c) => {
                let trigger = if is_completion_word_char(c) {
                    CompletionTrigger::Typed(c)
                } else if matches!(c, '(' | ',' | '=' | ':' | '.' | ' ') {
                    CompletionTrigger::TriggerChar(c)
                } else {
                    self.completion.on_boundary();
                    self.pending_completion_request = None;
                    return;
                };
                self.request_completions(trigger);
            }
            CompletionEvent::Deleting => {
                if self.completion.is_open() {
                    let word = self.completion_word_text();
                    match word.chars().last() {
                        Some(c) => self.request_completions(CompletionTrigger::Typed(c)),
                        None => {
                            self.completion.close();
                            self.pending_completion_request = None;
                        }
                    }
                }
            }
            CompletionEvent::CaretOrClose => {
                self.completion.close();
                self.pending_completion_request = None;
            }
        }
    }

    /// Query completions for `trigger`: a synchronous provider fills the popup
    /// inline; with none set, record an async [`CompletionRequest`] the host
    /// fulfills via [`take_completion_request`](Self::take_completion_request) +
    /// [`set_completions`](Self::set_completions). No-op if neither is wired (the
    /// request is recorded regardless, but a host that never polls it just drops
    /// it — cheap).
    fn request_completions(&mut self, trigger: CompletionTrigger) {
        // Take the provider out so `self` is free for `build_cx`, then restore it
        // (avoids an is_some/unwrap dance and the whole-self borrow conflict).
        let Some(mut provider) = self.comp_provider.take() else {
            // No synchronous provider: record an async request for the host.
            let head = self.doc.selections().newest().head();
            self.pending_completion_request = Some(CompletionRequest {
                revision: self.doc.revision(),
                position: self.doc.buffer().offset_to_point(head),
            });
            return;
        };
        let cx = self.build_cx(trigger);
        let word = self.completion_word_text();
        self.completion.on_input(&cx, &word, &mut *provider);
        self.comp_provider = Some(provider);
    }

    /// Drive the signature-help box: `(` opens it; while open, every relevant
    /// edit/move re-queries and a `None` reply closes it. No-op without a provider.
    fn drive_signature(&mut self, event: CompletionEvent) {
        let query = matches!(event, CompletionEvent::Typed('(')) || self.signature.is_some();
        if !query {
            return;
        }
        if let Some(mut provider) = self.sig_provider.take() {
            let cx = self.build_sig_cx();
            self.signature = provider.signature(&cx);
            self.sig_provider = Some(provider);
        } else {
            // No synchronous provider: record an async request for the host.
            let head = self.doc.selections().newest().head();
            self.pending_signature_request = Some(SignatureRequest {
                revision: self.doc.revision(),
                position: self.doc.buffer().offset_to_point(head),
            });
        }
    }

    /// Accept the popup's selected item: replace the completion word with the
    /// item's insertion (a snippet expands and starts an interactive tab-stop
    /// session, selecting the first stop), sealed as one edit. Fires the retrigger
    /// if requested.
    fn accept_completion(&mut self) {
        let Some(item) = self.completion.accept() else { return };
        let replace = item.replace.clone().unwrap_or_else(|| self.completion_word());
        self.set_selection_range(replace.clone());

        let expanded = match &item.insert {
            InsertText::Plain(s) => {
                self.doc.insert_text(s);
                None
            }
            InsertText::Snippet(body) => match Snippet::parse(body) {
                Ok(snip) => {
                    let indent = self.line_indent(replace.start);
                    let e = snip.for_insertion(&indent, default_indent_size() as usize);
                    self.doc.insert_text(&e.text);
                    Some(e)
                }
                Err(_) => {
                    self.doc.insert_text(body);
                    None
                }
            },
        };

        if let Some(mut s) = self.snippet.take() {
            s.cancel(self.doc.decorations_mut());
        }
        if let Some(e) = expanded {
            match SnippetSession::start(&e, replace.start, self.doc.decorations_mut()) {
                Some((session, first)) => {
                    self.set_selection_range(first);
                    self.snippet = Some(session);
                }
                None => {
                    let fin = e.stops.last().map_or(e.text.len() as u32, |s| s.range.start);
                    self.set_caret(replace.start + fin);
                }
            }
        }
        self.doc.tokenize_highlight(self.viewport.end);
        let now = self.now_ms();
        self.doc.maybe_rescan_find(now);
        self.dirty = true;

        if item.retrigger && self.snippet.is_none() {
            let cx = self.build_cx(CompletionTrigger::Manual);
            let word = self.completion_word_text();
            if let Some(provider) = self.comp_provider.as_mut() {
                self.completion.on_input(&cx, &word, &mut **provider);
            }
        }
    }

    /// Tab / Shift+Tab through the active snippet session.
    fn snippet_tab(&mut self, forward: bool) {
        let Some(mut session) = self.snippet.take() else { return };
        match session.tab(forward, self.doc.decorations_mut()) {
            TabOutcome::Move(range) => {
                self.set_selection_range(range);
                self.snippet = Some(session);
            }
            TabOutcome::Finish(offset) => self.set_caret(offset),
            TabOutcome::Stay => self.snippet = Some(session),
        }
    }

    /// Cancel the snippet session if the primary caret has left every stop.
    fn reconcile_snippet(&mut self) {
        if self.snippet.is_none() {
            return;
        }
        let head = self.doc.selections().newest().head();
        let escaped = self.snippet.as_ref().unwrap().edit_escapes(&(head..head), self.doc.decorations());
        if escaped {
            let mut s = self.snippet.take().unwrap();
            s.cancel(self.doc.decorations_mut());
        }
    }

    /// Move the primary caret to `offset`.
    fn set_caret(&mut self, offset: u32) {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::caret(SelectionId(0), offset));
        self.doc.set_selections(set);
    }

    /// Select `range` as the primary selection.
    fn set_selection_range(&mut self, range: Range<u32>) {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), range.start, range.end));
        self.doc.set_selections(set);
    }

    /// The completion-word byte range ending at the primary caret (empty at a
    /// boundary), computed with `is_completion_word_char`.
    fn completion_word(&self) -> Range<u32> {
        let head = self.doc.selections().newest().head();
        let p = self.doc.buffer().offset_to_point(head);
        let line_start = head - p.col;
        let prefix = &self.doc.buffer().line(p.row)[..p.col as usize];
        let word_start = prefix
            .char_indices()
            .rev()
            .take_while(|(_, c)| is_completion_word_char(*c))
            .last()
            .map_or(prefix.len(), |(i, _)| i);
        (line_start + word_start as u32)..head
    }

    /// The text of the completion word under the caret.
    fn completion_word_text(&self) -> String {
        let w = self.completion_word();
        self.doc.buffer().slice(w.start..w.end).into_owned()
    }

    /// The leading whitespace of the line containing byte `offset`.
    fn line_indent(&self, offset: u32) -> String {
        let row = self.doc.buffer().offset_to_point(offset).row;
        self.doc.buffer().line(row).chars().take_while(|c| *c == ' ' || *c == '\t').collect()
    }

    /// Build a completion request from the current document state. The lookback
    /// slice (up to `LOOKBACK_LINES`) is skipped when the popup is already open
    /// and a word char was typed — that path only refilters locally and never
    /// reads it, so the hot per-keystroke path avoids the slice copy.
    fn build_cx(&self, trigger: CompletionTrigger) -> CompletionCx {
        let head = self.doc.selections().newest().head();
        let position = self.doc.buffer().offset_to_point(head);
        let refilter_only =
            self.completion.is_open() && matches!(trigger, CompletionTrigger::Typed(_));
        let lookback = if refilter_only {
            String::new()
        } else {
            let start_row = position.row.saturating_sub(LOOKBACK_LINES - 1);
            let lb_start = self.doc.buffer().point_to_offset(Point::new(start_row, 0));
            self.doc.buffer().slice(lb_start..head).into_owned()
        };
        CompletionCx {
            doc: self.doc.buffer().doc_id(),
            revision: self.doc.revision().0,
            position,
            word: self.completion_word(),
            lookback,
            trigger,
        }
    }

    /// Build a signature request from the current document state.
    fn build_sig_cx(&self) -> SignatureCx {
        let head = self.doc.selections().newest().head();
        let position = self.doc.buffer().offset_to_point(head);
        let start_row = position.row.saturating_sub(LOOKBACK_LINES - 1);
        let lb_start = self.doc.buffer().point_to_offset(Point::new(start_row, 0));
        SignatureCx {
            doc: self.doc.buffer().doc_id(),
            revision: self.doc.revision().0,
            position,
            lookback: self.doc.buffer().slice(lb_start..head).into_owned(),
        }
    }

    /// The word range around `offset` (scanning both directions with
    /// `is_completion_word_char`) — the word under the hover pointer.
    fn word_around(&self, offset: u32) -> Range<u32> {
        let p = self.doc.buffer().offset_to_point(offset);
        let line = self.doc.buffer().line(p.row);
        let line_start = offset - p.col;
        let col = p.col as usize;
        let start = line[..col]
            .char_indices()
            .rev()
            .take_while(|(_, c)| is_completion_word_char(*c))
            .last()
            .map_or(col, |(i, _)| i);
        let mut end = col;
        for c in line[col..].chars().take_while(|c| is_completion_word_char(*c)) {
            end += c.len_utf8();
        }
        (line_start + start as u32)..(line_start + end as u32)
    }

    /// Build a hover request for the word under `offset`. The lookback runs
    /// through the word's END, so the provider reads the full word from its tail.
    fn build_hover_cx(&self, offset: u32) -> HoverCx {
        let word = self.word_around(offset);
        let position = self.doc.buffer().offset_to_point(offset);
        let start_row = position.row.saturating_sub(LOOKBACK_LINES - 1);
        let lb_start = self.doc.buffer().point_to_offset(Point::new(start_row, 0));
        HoverCx {
            doc: self.doc.buffer().doc_id(),
            revision: self.doc.revision().0,
            position,
            word: word.clone(),
            lookback: self.doc.buffer().slice(lb_start..word.end).into_owned(),
        }
    }
}

/// Place `[Fill field | buttons]` in a styled, fixed-width container — so both
/// find and replace boxes come out identical regardless of how many buttons each
/// holds. The focus ring lives here (a container can't read its field's focus),
/// driven by the caller-tracked `focused` flag.
fn box_of<'a>(
    field: Element<'a, Event>,
    buttons: Vec<Element<'a, Event>>,
    focused: bool,
    w: f32,
    h: f32,
    gap: f32,
    margin: f32,
) -> Element<'a, Event> {
    container(
        row![field, row(buttons).spacing(gap).align_y(Alignment::Center)]
            .spacing(gap)
            .align_y(Alignment::Center),
    )
    .width(Length::Fixed(w))
    .height(Length::Fixed(h))
    .padding(iced::Padding::from([0.0, margin]))
    .style(move |_theme: &Theme| container::Style {
        background: Some(Color::from_rgb8(0x3C, 0x3C, 0x3C).into()),
        border: iced::border::rounded(4.0).width(1.0).color(if focused {
            Color::from_rgb8(0x00, 0x7F, 0xD4) // focus ring
        } else {
            Color::TRANSPARENT
        }),
        ..container::Style::default()
    })
    .into()
}

/// A severity's display name for the diagnostic hover.
fn severity_label(sev: Severity) -> &'static str {
    match sev {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
        Severity::Hint => "hint",
    }
}

/// The find bar's global chord table — a free fn so it is testable without a
/// window. `status` is what the widget tree did with the key BEFORE this saw it,
/// and is load-bearing for Tab: a focused editor captures Tab (it indents), so
/// gating on `Ignored` keeps indent working while the bar is open.
fn find_chord(
    key: &iced::keyboard::Key,
    modifiers: iced::keyboard::Modifiers,
    status: iced::event::Status,
) -> Option<Event> {
    use iced::keyboard::{key::Named, Key};
    let ctrl = modifiers.command() || modifiers.control();
    match key {
        Key::Character(c) if ctrl && c.as_str() == "f" => Some(Event::OpenFind),
        Key::Character(c) if ctrl && c.as_str() == "h" => Some(Event::OpenReplace),
        Key::Named(Named::Escape) => Some(Event::CloseFind),
        // Alt+Enter selects all matches — safe as a global chord because the
        // editor ignores Alt+Enter. Plain Enter is NOT here: in the input it
        // navigates via `on_submit`, in the editor it must only type a newline.
        Key::Named(Named::Enter) if modifiers.alt() => Some(Event::FindSelectAll),
        // Tab moves focus between the inputs — but only when nothing else took
        // the key (the editor captures Tab to indent).
        Key::Named(Named::Tab) if status == iced::event::Status::Ignored => {
            Some(Event::CycleFocus { back: modifiers.shift() })
        }
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A minimal grammar: one keyword-scoped rule over word runs. Enough to drive
    // the highlight cache without pulling the example's Rust grammar into the lib.
    const GRAMMAR: &str = "%YAML 1.2\n---\nname: T\nscope: source.t\ncontexts:\n  main:\n    - match: '\\w+'\n      scope: keyword.t\n";

    /// The cold-load fix, as a fails-first regression: attaching a grammar
    /// tokenizes the visible document *at construction*, so the first paint is
    /// coloured with NO `update`/`ViewportChanged` ever called. Without the seed
    /// in [`CodeEditor::language`] the frontier stays fully dirty after
    /// `set_syntax` and this assertion fails — which was exactly the
    /// "no highlighting until one scroll tick" bug.
    #[test]
    fn language_tokenizes_at_load_without_a_viewport_report() {
        let grammar = SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses");
        let editor = CodeEditor::new("fn main() {}\nlet x = 1;\n").language(grammar);
        assert!(
            editor.document().highlight_frontier().is_none(),
            "grammar attach left the highlight frontier dirty — cold load would show uncoloured text until a scroll",
        );
    }

    /// Plain-text (no grammar) is a valid state: no highlight cache, nothing to
    /// pump, the sweep subscription stays idle.
    #[test]
    fn no_grammar_leaves_no_frontier_to_pump() {
        let editor = CodeEditor::new("plain text\n");
        assert!(editor.document().highlight_frontier().is_none());
    }

    /// Replace-all driven through the relocated find bar's [`Event`]s must land as
    /// one transaction: a single undo restores the document byte-for-byte. Proves
    /// the find/replace message path commits once, not once per match.
    #[test]
    fn replace_all_through_events_is_one_undo_step() {
        let mut ed = CodeEditor::new("foo foo foo\n");
        let _ = ed.update(Event::OpenFind);
        let _ = ed.update(Event::FindQuery("foo".into()));
        let _ = ed.update(Event::ReplaceText("bar".into()));
        let _ = ed.update(Event::ReplaceAll);
        assert_eq!(ed.document().text().into_owned(), "bar bar bar\n");
        let _ = ed.update(Event::Editor(Action::Undo));
        assert_eq!(ed.document().text().into_owned(), "foo foo foo\n");
    }

    /// A disabled find bar swallows its open chords — the editor stays find-less.
    #[test]
    fn find_disabled_ignores_open() {
        let mut ed = CodeEditor::new("abc\n").find(false);
        let _ = ed.update(Event::OpenFind);
        assert!(!ed.find_open, "find(false) must not open the bar");
    }

    /// A half-typed regex is a NORMAL invalid state (not "no results") — the bar
    /// reports the pattern error rather than implying the document lacks a match.
    #[test]
    fn half_typed_regex_reports_invalid() {
        let mut ed = CodeEditor::new("abc\n");
        let _ = ed.update(Event::OpenFind);
        let _ = ed.update(Event::ToggleRegex);
        let _ = ed.update(Event::FindQuery("(".into())); // unbalanced while typing
        assert!(
            ed.document().find_pattern_error().is_some(),
            "a half-typed regex is a normal invalid state, surfaced as a pattern error",
        );
    }

    /// Find-in-selection scopes matches to the selection: the third `foo` outside
    /// the selected span is not counted.
    #[test]
    fn find_in_selection_scopes_the_matches() {
        let mut ed = CodeEditor::new("foo foo foo\n");
        let _ = ed.update(Event::OpenFind);
        let _ = ed.update(Event::Editor(Action::DragSelect {
            granularity: scrive_core::Granularity::Char,
            origin: 0,
            head: 7, // "foo foo"
        }));
        let _ = ed.update(Event::ToggleFindInSelection);
        assert!(ed.document().find_scope().is_some(), "the toggle sets the document scope");
        let _ = ed.update(Event::FindQuery("foo".into()));
        assert_eq!(ed.document().find_match_count(), 2, "matches are scoped to the selection");
    }

    /// The replace button NAVIGATES to a match before it overwrites: the first
    /// press selects, the second replaces (so you always see what you replace).
    #[test]
    fn replace_navigates_before_it_overwrites() {
        let mut ed = CodeEditor::new("foo foo\n");
        let _ = ed.update(Event::OpenFind);
        let _ = ed.update(Event::FindQuery("foo".into()));
        let _ = ed.update(Event::ReplaceText("bar".into()));
        let _ = ed.update(Event::ReplaceOne);
        assert_eq!(ed.document().text().into_owned(), "foo foo\n", "first press only navigates");
        let _ = ed.update(Event::ReplaceOne);
        assert_eq!(ed.document().text().into_owned(), "bar foo\n", "second press overwrites the match");
    }

    /// The find chord table maps the global keys (a pure fn, testable without a
    /// window).
    #[test]
    fn find_chords_map_the_global_keys() {
        use iced::keyboard::{key::Named, Key, Modifiers};
        let ignored = iced::event::Status::Ignored;
        assert!(matches!(
            find_chord(&Key::Character("f".into()), Modifiers::CTRL, ignored),
            Some(Event::OpenFind)
        ));
        assert!(matches!(
            find_chord(&Key::Character("h".into()), Modifiers::CTRL, ignored),
            Some(Event::OpenReplace)
        ));
        assert!(matches!(
            find_chord(&Key::Named(Named::Escape), Modifiers::empty(), ignored),
            Some(Event::CloseFind)
        ));
    }

    struct OneCompletion;
    impl Completions for OneCompletion {
        fn complete(&mut self, _cx: &CompletionCx) -> Vec<scrive_core::CompletionItem> {
            vec![scrive_core::CompletionItem::plain("hello", scrive_core::CompletionKind::Keyword)]
        }
    }

    /// End-to-end through the relocated intel loop: typing a word char opens the
    /// popup off the injected provider, and accepting it inserts the item as one
    /// edit. Proves `drive_completion` + `accept_completion` are wired to `update`.
    #[test]
    fn typing_opens_and_accepting_inserts_a_completion() {
        let mut ed = CodeEditor::new("").completions(OneCompletion);
        let _ = ed.update(Event::Editor(Action::Type('h')));
        assert!(
            matches!(ed.completion.state(), CompletionState::Open(_)),
            "typing a word char with a provider opens the popup",
        );
        let _ = ed.update(Event::Editor(Action::PopupAccept));
        assert_eq!(ed.document().text().into_owned(), "hello");
    }

    /// No provider ⇒ no popup, even on a word char (the drive loop no-ops).
    #[test]
    fn no_provider_never_opens_a_popup() {
        let mut ed = CodeEditor::new("");
        let _ = ed.update(Event::Editor(Action::Type('h')));
        assert!(matches!(ed.completion.state(), CompletionState::Closed));
    }

    /// The blessed whole-buffer swap replaces the text AND re-tokenizes the
    /// visible document at load — with no `ViewportChanged`. Fails-first against a
    /// `load` that swapped the buffer but left the (grammar-swapped) cache dirty.
    #[test]
    fn load_swaps_the_buffer_and_retokenizes() {
        let grammar = SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses");
        let mut ed = CodeEditor::new("old\n").language(grammar);
        let g2 = SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses");
        ed.load("brand new content\nsecond line\n", Some(g2));
        assert_eq!(ed.document().text().into_owned(), "brand new content\nsecond line\n");
        assert!(
            ed.document().highlight_frontier().is_none(),
            "load must re-tokenize the visible document",
        );
    }

    /// The `bracket_lexing` builder wires through to the document's bracket
    /// matching: a bracket inside a string is not counted, so it is not coloured,
    /// folded, or indent-guided.
    #[test]
    fn bracket_lexing_skips_in_string_brackets() {
        let grammar = SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses");
        // `let s = "(";\nf(x);\n` — the ( at 9 is inside the string; f(x)'s ( ) code.
        let ed = CodeEditor::new("let s = \"(\";\nf(x);\n")
            .language(grammar)
            .bracket_lexing(vec![b'"'], None);
        let offs: Vec<u32> = ed.document().brackets().all().iter().map(|b| b.offset).collect();
        assert_eq!(offs, vec![14, 16], "the ( inside the string is skipped; f(x) counts");
    }

    /// A programmatic edit runs the tail (applies, marks dirty).
    #[test]
    fn edit_applies_and_marks_dirty() {
        let mut ed = CodeEditor::new("abc\n");
        ed.edit(vec![scrive_core::EditOp::new(0..0, "X")]);
        assert_eq!(ed.document().text().into_owned(), "Xabc\n");
        assert!(ed.is_dirty());
    }

    /// Async completions, end to end: with no sync provider, typing a word char
    /// records a request; the host fulfills it and ingests the result, which opens
    /// the popup; accepting inserts the item.
    #[test]
    fn async_completion_request_and_ingest_round_trip() {
        let mut ed = CodeEditor::new("");
        let _ = ed.update(Event::Editor(Action::Type('h')));
        let req = ed.take_completion_request().expect("a word char records an async request");
        ed.set_completions(
            req.revision,
            vec![CompletionItem::plain("hello", scrive_core::CompletionKind::Keyword)],
        );
        assert!(
            matches!(ed.completion.state(), CompletionState::Open(_)),
            "ingesting items at the current revision opens the popup",
        );
        let _ = ed.update(Event::Editor(Action::PopupAccept));
        assert_eq!(ed.document().text().into_owned(), "hello");
    }

    /// Strict staleness: a result computed against a revision the buffer has moved
    /// past is dropped (the host re-requests on the next edit).
    #[test]
    fn stale_set_completions_is_dropped() {
        let mut ed = CodeEditor::new("");
        let _ = ed.update(Event::Editor(Action::Type('h')));
        let req = ed.take_completion_request().unwrap();
        // The buffer moves on before the async result arrives.
        let _ = ed.update(Event::Editor(Action::Type('i')));
        ed.set_completions(
            req.revision,
            vec![CompletionItem::plain("hello", scrive_core::CompletionKind::Keyword)],
        );
        assert!(
            matches!(ed.completion.state(), CompletionState::Closed),
            "a stale completion result must be dropped, not shown",
        );
    }

    /// Snippets ride completions with no async-specific work: an async-ingested
    /// snippet item, once accepted, starts an interactive tab-stop session.
    #[test]
    fn async_snippet_item_starts_a_session_on_accept() {
        let mut ed = CodeEditor::new("");
        let _ = ed.update(Event::Editor(Action::Type('i')));
        let req = ed.take_completion_request().unwrap();
        let snippet = CompletionItem::new(
            "iflet",
            scrive_core::CompletionKind::Keyword,
            InsertText::Snippet("if ${1:cond} {\n\t$0\n}".into()),
        );
        ed.set_completions(req.revision, vec![snippet]);
        assert!(matches!(ed.completion.state(), CompletionState::Open(_)));
        let _ = ed.update(Event::Editor(Action::PopupAccept));
        assert!(
            ed.snippet.is_some(),
            "accepting an async-ingested snippet item starts a tab-stop session",
        );
    }

    /// Typing `(` with no synchronous signature provider records an async
    /// request — the signature twin of the completion seam.
    #[test]
    fn typing_open_paren_records_async_signature_request() {
        let mut ed = CodeEditor::new("");
        let _ = ed.update(Event::Editor(Action::Type('(')));
        assert!(
            ed.take_signature_request().is_some(),
            "'(' with no provider records an async signature request",
        );
    }

    /// Hovering a word with no synchronous hover provider records an async
    /// request — the hover twin of the completion seam.
    #[test]
    fn hover_over_a_word_records_async_request() {
        let mut ed = CodeEditor::new("hello world\n");
        let _ = ed.update(Event::Editor(Action::HoverQuery(2))); // inside "hello"
        assert!(
            ed.take_hover_request().is_some(),
            "hovering a word with no provider records an async hover request",
        );
    }

    /// A large document (≥ the parallel threshold) spins up the off-thread pool
    /// at load; a small one keeps the synchronous path. This is what stops a huge
    /// buffer from blocking the UI thread tokenizing synchronously.
    #[test]
    fn large_document_uses_the_parallel_pool() {
        let grammar = SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses");
        let big = "fn f() {}\n".repeat(230_000); // ~2.3 MB, over PARALLEL_MIN_BYTES
        let ed = CodeEditor::new(big).language(grammar);
        assert!(ed.hl_pool.is_some(), "a large document spins up the parallel highlight pool");

        let g2 = SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses");
        let small = CodeEditor::new("fn f() {}\n").language(g2);
        assert!(small.hl_pool.is_none(), "a small document keeps the synchronous path");
    }

    /// `dirty` tracks actual text change: a bare caret move must not dirty the
    /// document (else a host would schedule a needless recompile / save), but
    /// typing does.
    #[test]
    fn caret_move_does_not_dirty_but_typing_does() {
        let mut ed = CodeEditor::new("abc\n");
        let _ = ed.update(Event::Editor(Action::PlaceCaret(1)));
        assert!(!ed.is_dirty(), "a caret move must not dirty the document");
        let _ = ed.update(Event::Editor(Action::Type('x')));
        assert!(ed.is_dirty(), "typing dirties the document");
    }

    /// The incremental-change log (LSP `didChange`) is off by default and logs
    /// applied edits once enabled, draining clean. Full-sync hosts never touch it.
    #[test]
    fn drain_changes_mirrors_edits_when_observing() {
        let mut ed = CodeEditor::new("hello\n");
        assert!(ed.drain_changes().is_empty(), "the change log is off by default");
        ed.observe_changes(true);
        let _ = ed.update(Event::Editor(Action::Type('X'))); // insert 'X' at the caret (offset 0)
        let changes = ed.drain_changes();
        assert_eq!(changes.len(), 1, "one keystroke logs one change");
        assert_eq!(changes[0].text, "X");
        assert!(ed.drain_changes().is_empty(), "draining clears the log");
    }
}
