//! `scrive-core` — the headless core of the scrive code editor.
//!
//! This crate owns everything about *what* the editor shows and *what changed*,
//! and knows nothing about pixels: no `iced`, no `winit`, no renderer. The iced
//! integration lives in the sibling `scrive-iced` crate, which depends on this
//! one — the dependency points one way, so the editing engine can be tested and
//! reused without dragging in a GUI toolkit.
//!
//! # The one load-bearing idea: one fact, one owner
//!
//! Every derived position — a caret, a diagnostic squiggle, a snippet stop —
//! moves through a single mapping function on every edit, and every change is a
//! single atomic transaction with a mechanically-derived inverse. Almost every
//! serious editor bug is two copies of one fact drifting apart; keeping each
//! position in exactly one place, moved through one patch, is the structural
//! antidote.
//!
//! # Where do I…?
//!
//! - store text / convert coordinates → [`buffer`], [`coords`]
//! - apply an edit / undo / redo → [`transaction`], [`history`]
//! - move or extend the caret → [`selection`], [`movement`]
//! - type / paste / indent → [`verbs`]
//! - expand tabs to columns → [`display_map`]
//! - syntax-highlight a line → [`highlight`]
//! - attach diagnostics / snippet stops → [`decorations`]
//! - complete / hover / signature help → [`intel::providers`]

#![deny(missing_docs)]
#![forbid(unsafe_code)]

mod autoclose; // auto-close pair rules (crate-internal)
pub mod bracket;
mod bracket_tree; // bracket matching on the SumTree (shape monoid)
pub mod buffer;
pub mod coords;
pub mod decorations;
pub mod display_map;
pub mod document;
pub mod find;
pub mod fold_map;
pub mod highlight;
pub mod history;
pub mod intel;
pub mod movement;
mod offset_set; // delta-gap SumTree of offsets (folds/decorations backing)
pub mod patch;
mod perf; // op-count work meter behind the complexity gate (crate-internal)
mod rope; // the text rope (SumTree<Chunk>) backing Buffer
#[cfg(test)]
mod perf_gate; // scale-matrix test that fails the build if a shared primitive scales worse than linearly (tests only)
pub mod row_layout;
pub mod selection;
mod sum_tree; // the one augmented balanced tree every position-tracked structure descends; crate-internal for now
pub mod transaction;
pub mod verbs;

pub use bracket::{Bracket, Brackets};
pub use buffer::{Buffer, DocId, EolFlavor, LoadError, Revision, Snapshot};
pub use coords::{Bias, Point};
pub use decorations::{
    DecorationId, DecorationKind, DecorationStore, Diagnostic, DiagnosticsOutcome, EmptyPolicy,
    Severity, Stickiness, TrackedRange,
};
pub use display_map::{
    default_tab_size, BufferRow, DisplayChunk, DisplayChunks, DisplayEdit, DisplayPoint,
    DisplayRow, TabMap,
};
pub use document::{Document, RevealMode};
pub use find::{default_find_debounce, FindQuery, FindState, FIND_MATCH_CAP};
pub use fold_map::{FoldMap, FoldSet, InlineFold, VisibleRow};
pub use highlight::{
    HIGHLIGHT_CHECKPOINT_STRIDE, HIGHLIGHT_MAX_LINES_PER_CALL, HIGHLIGHT_MAX_WINDOW_ROWS,
    HIGHLIGHT_WINDOW_SLACK,
};
pub use highlight::{
    padded_highlight_window, tokenize_segment, HighlightCache, HighlightEngine, HighlightSpan,
    Highlighter, Rgba, SegmentBoundary, SegmentStart, SegmentTokens, SpanStyle, SyntaxDef,
    TokenTheme,
};
pub use history::{GroupingHint, OpClass};
pub use intel::completion::{CompletionController, CompletionState, PopupList};
pub use intel::hover::{Hover, HoverCx, HoverInfo, HOVER_IDLE_DELAY_MS};
pub use intel::signature::{SignatureCx, SignatureHelp, SignatureInfo};
pub use intel::snippet::{CaretOutcome, Snippet, SnippetError, SnippetSession, TabOutcome, TabStop};
pub use intel::providers::{
    is_completion_word_char, CompletionCx, CompletionItem, CompletionKind, CompletionTrigger,
    Completions, InsertText, LOOKBACK_LINES,
};
pub use movement::{move_selections, ColumnDir, Granularity, Motion};
pub use patch::{Edit, Patch};
pub use row_layout::{
    tail_start_col, virtual_cell, CaretCell, Chip, DisplayPosition, HeaderHit, HeaderLayout,
    RowLayout, TailGlyph, FOLD_PLACEHOLDER_CELLS, INLINE_CHIP_CELLS,
};
pub use selection::{Selection, SelectionId, SelectionSet};
pub use transaction::{Committed, EditOp, TransactionError};
pub use verbs::default_indent_size;
