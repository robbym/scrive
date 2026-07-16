//! A global op-count **work meter** — the deterministic engine behind the
//! complexity gate (`document.rs`'s `perf_gate` matrix).
//!
//! Instrumented hot-path primitives (`Buffer::offset_to_point`/`point_to_offset`,
//! `Patch::map_offset`/`map_many`, the bracket enclosing walk,
//! `FoldMap::display_position`, `FoldMap::new`, every document-scale
//! `sort`/`retain`/`dedup`) charge the meter by the *units of work they touch*.
//! A matrix test runs each representative editor operation at 2× every scale
//! dimension (document size, caret count, fold count) and asserts the meter grows
//! within that operation's declared budget. So an accidental superlinear
//! hot-path cost fails the build instead of surfacing as field lag; a new slow
//! function is caught even though no one wrote a counter for it, because it
//! necessarily runs through these primitives. The few genuinely inherent
//! superlinear costs (an unbalanced-bracket cascade to end-of-file, a
//! fold-break `retain`) are the matrix's documented exceptions.
//!
//! Machine-independent and deterministic: it counts operations, not wall-clock,
//! so the thresholds are stable across machines and runs. Live only under
//! `debug_assertions`/`test`; the release [`charge`] is an empty
//! `#[inline(always)]` no-op the optimizer erases along with its argument, so the
//! instrumentation is zero-cost in shipped builds.

#[cfg(any(test, debug_assertions))]
thread_local! {
    static WORK: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Charge `units` of work to the thread's meter (debug/test only).
#[cfg(any(test, debug_assertions))]
#[inline]
pub(crate) fn charge(units: u64) {
    WORK.with(|c| c.set(c.get().wrapping_add(units)));
}

/// Release build: a no-op the optimizer erases (its argument included).
#[cfg(not(any(test, debug_assertions)))]
#[inline(always)]
pub(crate) fn charge(_units: u64) {}

/// The meter's current value. Only the `perf_gate` matrix reads it, so it is
/// gated to tests (the `charge` sites, by contrast, live in any debug build).
#[cfg(test)]
pub(crate) fn meter() -> u64 {
    WORK.with(std::cell::Cell::get)
}

/// Reset the meter to zero before a measured operation (tests only).
#[cfg(test)]
pub(crate) fn reset() {
    WORK.with(|c| c.set(0));
}
