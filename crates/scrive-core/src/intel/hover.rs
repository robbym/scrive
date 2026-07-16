//! Hover — the [`Hover`] trait the app satisfies plus the plain data it
//! returns. Like the other language services it is **synchronous by contract**:
//! a hover query is an in-memory lookup keyed by the word under the pointer,
//! so the widget calls it on the mouse-idle tick and renders the reply the same
//! frame. Because the answer is produced synchronously there is no reply
//! envelope and nothing in flight, so a reply can never arrive stale.

use core::ops::Range;

use crate::{DocId, Point};

/// Mouse-idle delay before a hover query fires, in milliseconds. Tuned to the
/// ~300 ms mainstream editors use, so the popup feels neither jumpy nor
/// sluggish. The widget arms one query when the pointer rests over a word this
/// long without moving.
pub const HOVER_IDLE_DELAY_MS: u64 = 300;

/// The provider seam the integrating app implements to answer hover queries.
pub trait Hover {
    /// The doc for the word under the pointer, or `None` when there is none (or
    /// the pointer is not over a word) — the hover popup stays closed.
    fn hover(&mut self, cx: &HoverCx) -> Option<HoverInfo>;
}

/// A revision-stamped hover request — everything the provider may read.
#[derive(Clone, Debug)]
pub struct HoverCx {
    /// Which document the request is for.
    pub doc: DocId,
    /// The document revision the request was snapshotted at.
    pub revision: u64,
    /// The point under the pointer, clipped to a valid char boundary.
    pub position: Point,
    /// Absolute byte range of the word under the pointer, computed with
    /// `is_completion_word_char` — empty ⇒ the query is skipped.
    pub word: Range<u32>,
    /// The preceding source text, back the same number of lines as
    /// `CompletionCx`, giving the classifier the context the spec lookup needs
    /// (dotted receiver, in-call position).
    pub lookback: String,
}

/// A resolved hover: the markdown to render and the word it describes.
#[derive(Clone, Debug)]
pub struct HoverInfo {
    /// Markdown body (a minimal block/inline subset; richer degrades to plain
    /// text at render).
    pub markdown: String,
    /// The word range the doc describes — the popup anchors here and the widget
    /// re-tests pointer containment against it for dismissal.
    pub range: Range<u32>,
}
