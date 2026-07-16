//! Compile-time proof that syntect's per-line carry state supports the
//! incremental-highlight convergence probe.
//!
//! The incremental highlight cache decides when re-highlighting can stop early
//! by comparing each line's end state `(ParseState, HighlightState)` with `==`:
//! once a re-parsed line reaches an end state equal to the one cached before the
//! edit, every following line is already correct and the sweep halts. That
//! comparison requires both halves to be `Clone + PartialEq`.
//!
//! This test encodes the requirement as a trait bound, so the answer is a build
//! outcome, not a runtime assertion:
//!
//! - Compiles and passes: `(ParseState, HighlightState)` is a valid convergence
//!   key — the cache can compare end states directly.
//! - If `HighlightState` lacks `PartialEq`/`Clone`, this FAILS TO COMPILE with a
//!   trait-bound error naming the culprit, and convergence must instead be keyed
//!   on the resolved end-of-line `SpanStyle` run.
//!
//! It is pinned against `syntect` "5" (default-fancy), the same dependency the
//! highlight cache uses, so the result holds for the real code.

use syntect::highlighting::HighlightState;
use syntect::parsing::ParseState;

/// Instantiates only if `T: Clone + PartialEq`. Calling it for a type is a
/// compile-time assertion that the type satisfies both bounds.
fn assert_clone_partial_eq<T: Clone + PartialEq>() {}

#[test]
fn syntect_line_states_support_the_convergence_probe() {
    // ParseState — the parser side of the per-line carry state.
    assert_clone_partial_eq::<ParseState>();
    // HighlightState — the style side of the per-line carry state.
    assert_clone_partial_eq::<HighlightState>();
}
