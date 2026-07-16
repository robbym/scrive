//! Signature help — the `SignatureHelp` trait the app satisfies plus the
//! plain data it returns. No controller: signature help is stateless per query
//! (the app re-runs it on `(` / `,` / edits and shows or hides the one-line box
//! from the reply), so unlike completion there is no sticky state machine.
//!
//! Synchronous by contract, same rationale as `Completions`: the query is
//! an `enclosingCall` + active-parameter count over a few lines of lookback —
//! microseconds — so the widget calls it and renders the reply the same frame,
//! with no reply envelope to go stale.

use core::ops::Range;

use crate::{DocId, Point};

/// The signature-help seam.
pub trait SignatureHelp {
    /// The signature of the call the caret is inside, or `None` when it is not
    /// inside a known call (the box closes).
    fn signature(&mut self, cx: &SignatureCx) -> Option<SignatureInfo>;
}

/// A revision-stamped signature request — everything the provider may read.
#[derive(Clone, Debug)]
pub struct SignatureCx {
    /// Which document the request is for.
    pub doc: DocId,
    /// The document revision the request was snapshotted at.
    pub revision: u64,
    /// Caret position, always clipped to a `char` boundary so the provider can
    /// slice `lookback` without splitting a multi-byte character.
    pub position: Point,
    /// The same `LOOKBACK_LINES` lookback as `CompletionCx` — `enclosingCall` +
    /// the active-parameter count need nothing else.
    pub lookback: String,
}

/// A resolved signature: the rendered line plus which parameter is active.
#[derive(Clone, Debug)]
pub struct SignatureInfo {
    /// The signature line, e.g. `wait(timer: duration)`.
    pub label: String,
    /// Byte ranges of each parameter's label within [`label`](Self::label). The
    /// provider builds `label`, so these are exact (no substring matching).
    pub params: Vec<Range<u32>>,
    /// The active parameter — the top-level comma count, clamped to
    /// `params.len() - 1`.
    pub active: u32,
    /// Optional documentation for the call.
    pub doc: Option<String>,
}

impl SignatureInfo {
    /// The byte range of the active parameter within `label`, if any — the
    /// substring the box highlights.
    #[must_use]
    pub fn active_param(&self) -> Option<Range<u32>> {
        self.params.get(self.active as usize).cloned()
    }
}
