//! Literal find — the search core.
//!
//! Search is **literal substring + an ASCII case-fold toggle** — no regex, no
//! whole-word — a literal scan covers the common editing workflows without
//! either. The search *bar* is app-side chrome; this module owns only
//! the match model and the scan.
//!
//! The one design choice: the match set **lives only in the decoration store**
//! — the sorted `FindMatch` decorations ARE the match set, in document order
//! (one fact, one owner). [`FindState`] holds the query and the scan
//! bookkeeping, never a shadow list of handles. The set is kept current
//! **eagerly**: every committed transaction (forward edit, undo, redo — they
//! share the same view-rebase path) runs [`FindState::on_commit`], a windowed
//! repair around each edit, so consumers never see a stale match. The one
//! remaining debounced path is the capped-set refill
//! ([`FindState::maybe_rescan`]).
//!
//! Documented, test-pinned relaxation: for needles that cannot overlap
//! themselves the repaired set is byte-identical to a fresh [`scan`]; a
//! self-overlapping needle (`"aa"`) repairs to a *maximal valid*
//! non-overlapping set whose phase near the edit may differ from the greedy
//! full scan until the next query change.

use core::ops::Range;

use crate::buffer::Buffer;
use crate::coords::Bias;
use crate::decorations::{
    DecorationId, DecorationKind, DecorationStore, Stickiness, TrackedRange,
};
use crate::patch::Patch;

/// An active find query: a literal substring plus an ASCII case-fold toggle.
/// `PartialEq` so the `Document` surface can no-op an unchanged query without
/// re-scanning.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct FindQuery {
    /// The literal text to find. Empty means "no matches" (never match-all).
    pub text: String,
    /// When `false`, `a`/`A` fold together. Case folding is ASCII-only.
    pub case_sensitive: bool,
}

/// Debounce window (ms) for the capped-set refill — a user-facing default,
/// exported as a `pub fn`. This is the ONLY debounced find path: ordinary edits
/// repair the match set eagerly at the commit.
#[must_use]
pub fn default_find_debounce() -> u64 {
    100
}

/// The scan stops after this many matches and records `capped` — cheap
/// insurance against a pathological query. A capped set is a *prefix of the
/// document*: everything up to [`FindState`]'s coverage frontier is exact,
/// nothing beyond it is represented.
pub const FIND_MATCH_CAP: usize = 10_000;

/// Every non-overlapping match of `query` in `text`, leftmost-first, as byte
/// spans; the `bool` is whether the scan hit [`FIND_MATCH_CAP`].
///
/// Literal byte-wise substring — the next probe starts at the previous match's
/// end (non-overlapping). Case-insensitive mode folds **ASCII only** (the
/// documented limitation). An empty query yields no matches, never match-all.
/// Byte-wise is exact for the ASCII DSL; a multi-byte needle is out of scope.
///
/// This is the pure oracle every other find path defers to: the query-change
/// full scan, the capped refill, and the per-edit window repairs all run this
/// same function, so the rules cannot fork. Case-sensitive search rides
/// `memchr::memmem`; the fold path probes the needle's first byte's two case
/// forms via `memchr2` and verifies each candidate window — both bit-identical
/// to the naive reference scan (pinned by `scan_equals_the_naive_oracle`).
#[must_use]
pub fn scan(text: &str, query: &FindQuery) -> (Vec<Range<u32>>, bool) {
    let needle = query.text.as_bytes();
    if needle.is_empty() {
        return (Vec::new(), false); // empty needle = zero matches
    }
    let hay = text.as_bytes();
    let mut spans = Vec::new();
    if query.case_sensitive {
        // memmem yields non-overlapping occurrences leftmost-first — the next
        // probe starts at the previous match's end.
        for i in memchr::memmem::find_iter(hay, needle) {
            if spans.len() == FIND_MATCH_CAP {
                return (spans, true); // cap reached
            }
            spans.push(i as u32..(i + needle.len()) as u32);
        }
    } else {
        let (lo, up) = (needle[0].to_ascii_lowercase(), needle[0].to_ascii_uppercase());
        let mut i = 0usize;
        while i + needle.len() <= hay.len() {
            // Jump to the next candidate first byte (either case form)…
            let Some(j) = memchr::memchr2(lo, up, &hay[i..]) else {
                break;
            };
            let c = i + j;
            if c + needle.len() > hay.len() {
                break;
            }
            // …and verify the window byte-wise (ASCII-only fold).
            if hay[c..c + needle.len()].eq_ignore_ascii_case(needle) {
                if spans.len() == FIND_MATCH_CAP {
                    return (spans, true); // cap reached
                }
                spans.push(c as u32..(c + needle.len()) as u32);
                i = c + needle.len(); // non-overlapping: resume past this match
            } else {
                i = c + 1;
            }
        }
    }
    (spans, false)
}

/// [`scan`] driven through the buffer's backing-agnostic ranged reads:
/// fixed-size windows with a needle-length−1 overlap, so a match straddling a
/// window boundary is found by the next window and the result is
/// **byte-identical to `scan(&buffer.text(), query)`** (pinned by
/// `windowed_scan_equals_the_whole_text_scan`). Never materializes the
/// document — peak transient is one window — and a capped dense query stops
/// after O(bytes-until-cap), which `buffer.text()` would defeat by copying
/// the whole rope up front.
fn scan_buffer(buffer: &Buffer, query: &FindQuery, window: u32) -> (Vec<Range<u32>>, bool) {
    let k = query.text.len() as u64;
    if k == 0 {
        return (Vec::new(), false); // empty needle = zero matches
    }
    let len = buffer.len();
    // ≥ 2k, so every iteration provably advances even for giant needles.
    let window = u64::from(window).max(k * 2);
    let mut spans: Vec<Range<u32>> = Vec::new();
    let mut pos: u32 = 0; // the non-overlapping scan cursor (next probe start)
    while u64::from(pos) + k <= u64::from(len) {
        // The window end snaps RIGHT to a char boundary: `slice` stays on the
        // str-slicing contract, the window is never empty, and growing a
        // window never loses a candidate.
        let win_end = buffer
            .clip_offset((u64::from(pos) + window).min(u64::from(len)) as u32, Bias::Right);
        let (local, local_capped) = scan(&buffer.slice(pos..win_end), query);
        let found = !local.is_empty();
        for r in local {
            if spans.len() == FIND_MATCH_CAP {
                return (spans, true); // cap reached
            }
            spans.push(pos + r.start..pos + r.end);
        }
        // A capped in-window scan left the window's tail unscanned — keep
        // going from the last match even at the document end (the overflow
        // match it found still has to trip the cap above).
        if win_end == len && !local_capped {
            break;
        }
        // Resume through the one owner of the seam logic ([`Buffer::scan_resume`]):
        // past the last match when the window found one (the non-overlap cursor —
        // the window's tail may be re-scanned but is never skipped), else at the
        // char boundary before `win_end − (k−1)`.
        let last_end = found.then(|| spans.last().expect("found ⇒ spans non-empty").end);
        pos = buffer.scan_resume(pos, win_end, k as u32, last_end);
    }
    (spans, false)
}

/// Whether a tracked range is a find match — the one predicate the store
/// queries and batch removals share.
fn is_find(r: &TrackedRange) -> bool {
    matches!(r.kind, DecorationKind::FindMatch)
}

/// The search view-state: the active query, the active match's decoration id,
/// and the coverage/cap bookkeeping.
///
/// The one design rule: **the store IS the match set** — the sorted `FindMatch`
/// decorations, in `(start, id)` = document order. This state deliberately
/// holds no handle list; count, order, and N-of-M all derive from store queries
/// (one fact, one owner). The set is repaired eagerly per commit via
/// [`FindState::on_commit`], so it is always current — undo/redo inherit the
/// repair through the shared view-rebase mover.
#[derive(Debug, Default)]
pub struct FindState {
    query: Option<FindQuery>,
    /// The decoration id of the highlighted/navigated match, if any.
    active: Option<DecorationId>,
    /// The active match's tracked START offset — the position half of the active
    /// handle, kept in lockstep with [`active`](Self::active) (both `Some` or
    /// both `None`). It rides each commit's patch in
    /// [`on_commit`](Self::on_commit) with **`Bias::Right`** — the start bias of
    /// a `FindMatch`'s `NeverGrows` stickiness — so it stays equal to the active
    /// decoration's current start (pinned by `active_start_tracks…`). It earns
    /// its place as the O(log) index that lets the per-commit presence check and
    /// the N-of-M ordinal avoid an O(M) whole-store walk, not as a display
    /// convenience.
    active_start: Option<u32>,
    /// Whether the match set is a capped *prefix* of the document.
    capped: bool,
    /// Byte offset the scan has covered: `buffer.len()` when `!capped`, else
    /// the last kept match's end. Rides each commit's patch (`Bias::Left`);
    /// while capped, repairs clamp their windows to it and edits beyond it are
    /// the refill's job.
    coverage_end: u32,
    /// While capped: the live count has dropped below the cap, so coverage
    /// could grow — the ONLY condition under which [`FindState::maybe_rescan`]
    /// re-scans. (At exactly the cap the set provably equals the ideal capped
    /// prefix: repairs keep the covered prefix exact, and its first
    /// [`FIND_MATCH_CAP`] matches are the document's first ones.)
    capped_stale: bool,
    /// `now_ms` of the last full scan — the capped-refill debounce anchor.
    last_scan_ms: u64,
}

impl FindState {
    /// An empty state — no query, no matches.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The active query, or `None` if find is idle.
    #[must_use]
    pub fn query(&self) -> Option<&FindQuery> {
        self.query.as_ref()
    }

    /// How many live matches the current query has — the store IS the match set,
    /// so this is the O(1) root-summary [`find_count`] read.
    ///
    /// [`find_count`]: DecorationStore::find_count
    #[must_use]
    pub fn match_count(&self, store: &DecorationStore) -> usize {
        store.find_count()
    }

    /// The decoration handle of the active match, if any — the render layer
    /// compares each `FindMatch` decoration to this to pick the distinct style.
    #[must_use]
    pub fn active_id(&self) -> Option<DecorationId> {
        self.active
    }

    /// The active match's tracked start offset — the position half of the active
    /// handle, in lockstep with [`active_id`](Self::active_id). `Document` reads
    /// it for the O(log) N-of-M ordinal (`active_find_match`).
    #[must_use]
    pub(crate) fn active_start(&self) -> Option<u32> {
        self.active_start
    }

    /// Whether the match set is a capped prefix of the document
    /// ([`FIND_MATCH_CAP`]).
    #[must_use]
    pub fn capped(&self) -> bool {
        self.capped
    }

    /// Set (or clear with `None`) the query and scan synchronously; never
    /// scrolls, and clears the active match. A query equal to the current one is
    /// a no-op (no needless re-scan).
    ///
    /// The full scan runs `scan_buffer` — whole-document in coverage but
    /// windowed in memory, and a capped dense query stops at
    /// O(bytes-until-cap) instead of copying the rope's tail.
    pub fn set_query(
        &mut self,
        query: Option<FindQuery>,
        buffer: &Buffer,
        store: &mut DecorationStore,
        now_ms: u64,
    ) {
        if query == self.query {
            return;
        }
        self.query = query;
        self.active = None; // a query change drops the active match
        self.active_start = None; // …and its tracked position (lockstep)
        self.rescan(buffer, store, now_ms);
    }

    /// Refill a **capped** match set: re-scan iff a query is active, the set is
    /// capped with room below the cap, and the debounce window has elapsed.
    /// `now_ms` is an injected monotonic clock — the widget passes real time,
    /// the headless suite a fake. Returns whether it scanned.
    ///
    /// Because the eager per-commit repair keeps the set current, the only job
    /// here is growing a capped set's coverage after matches inside it died.
    pub fn maybe_rescan(&mut self, buffer: &Buffer, store: &mut DecorationStore, now_ms: u64) -> bool {
        if self.query.is_none()
            || !(self.capped && self.capped_stale)
            || now_ms.saturating_sub(self.last_scan_ms) < default_find_debounce()
        {
            return false;
        }
        self.rescan(buffer, store, now_ms);
        true
    }

    /// Repair the match set around a committed patch — the mover hook called
    /// from the view-rebase path AFTER `DecorationStore::apply_patch` (it needs
    /// post-patch positions). No-op when find is idle.
    ///
    /// Per edit (post-edit coordinates), with `k` = needle byte length: every
    /// affected placement must overlap the influence window
    /// `new.start−(k−1) .. new.end+(k−1)` (a created/destroyed match contains a
    /// changed byte or the join point). Matches touching the window are removed
    /// wholesale ([`DecorationStore::take_matching_in`]); the re-scan zone is
    /// then the window widened (a) to the removed extents — a merely-touching
    /// match must be re-findable, not lost — and (b) a further `k−1` bytes each
    /// way, because a placement overlapping the removal zone (one whose old
    /// greedy blocker just died — the self-overlap phase repair) can start up
    /// to `k−1` bytes left of it and end as far right. Two clamps keep the
    /// zone from double-owning text: the scan starts at the last surviving
    /// match ending in the left margin (the anchor), and candidates
    /// overlapping the first surviving match right of the zone are dropped
    /// (the guard). The zone is greedily re-scanned with the same pure
    /// [`scan`] and batch-reinserted. Matches outside every zone keep their
    /// decoration ids (id stability is what lets the active match survive
    /// unrelated edits).
    ///
    /// Active survival: untouched active stays (same id); an active removed by a
    /// window transfers to a re-created match at its exact post-patch start;
    /// otherwise it clears.
    ///
    /// Cap: `coverage_end` rides the patch first; windows clamp to it, so
    /// repairs beyond a capped set's coverage are skipped (the refill's job).
    /// If a repair pushes the live count past [`FIND_MATCH_CAP`], the tail is
    /// trimmed so the set stays a prefix of the document.
    pub fn on_commit(&mut self, patch: &Patch, buffer: &Buffer, store: &mut DecorationStore) {
        let Some(query) = &self.query else { return };
        let k = query.text.len() as u32;
        if k == 0 || patch.is_empty() {
            return; // an empty needle has no matches to repair
        }
        // Coverage rides the patch ONCE, before the per-edit loop; an uncapped
        // set always covers the whole (post-edit) document.
        self.coverage_end = if self.capped {
            patch.map_offset(self.coverage_end, Bias::Left)
        } else {
            buffer.len()
        };
        // The active handle's position rides the SAME committed patch as the
        // decoration it tracks — `Bias::Right`, a FindMatch's `NeverGrows` start
        // bias — so it stays equal to that decoration's post-patch start.
        self.active_start = self.active_start.map(|s| patch.map_offset(s, Bias::Right));
        let mut removed_active_start: Option<u32> = None;
        for e in patch.edits() {
            // The influence window, snapped OUTWARD to char boundaries and
            // clamped to the covered prefix.
            let w_start = buffer.clip_offset(e.new.start.saturating_sub(k - 1), Bias::Left);
            let w_end_raw = e.new.end.saturating_add(k - 1).min(self.coverage_end);
            let w_end = buffer.clip_offset(w_end_raw, Bias::Right);
            if w_start > w_end {
                continue; // entirely beyond a capped coverage: refill's job
            }
            let removed = store.take_matching_in(w_start..w_end, is_find);
            if let Some(active) = self.active {
                if let Some(r) = removed.iter().find(|r| r.id == active) {
                    removed_active_start = Some(r.range.start);
                    self.active = None;
                    self.active_start = None; // lockstep; the transfer below re-sets both
                }
            }
            // Widen to the removed extents: a match that merely touched the
            // window (start left of it / end right of it) must be re-findable,
            // or removal would lose it.
            let ext_lo =
                removed.iter().map(|r| r.range.start).min().map_or(w_start, |s| s.min(w_start));
            let ext_hi =
                removed.iter().map(|r| r.range.end).max().map_or(w_end, |e| e.max(w_end));
            // Widen a further k−1 each way: a placement overlapping the
            // removal zone — one whose old greedy blocker died (self-overlap
            // phase) — can start up to k−1 bytes left of `ext_lo` and end as
            // far right of `ext_hi`. Removed-match boundaries are char
            // boundaries (byte-wise hits of a valid-UTF-8 needle); the ±(k−1)
            // arithmetic can unsnap, so clip again.
            let scan_lo = buffer.clip_offset(ext_lo.saturating_sub(k - 1), Bias::Left);
            let scan_hi = buffer
                .clip_offset(ext_hi.saturating_add(k - 1).min(self.coverage_end), Bias::Right);
            // Left anchor: the widened margin may hold a surviving match —
            // start scanning at its end so its text is never double-owned.
            // (Survivors intersecting `scan_lo..ext_lo` provably end at or
            // before `ext_lo`: anything reaching further would have touched
            // the window or a removed extent and been removed itself.)
            let anchor = store
                .decorations_in(scan_lo..ext_lo)
                .filter(is_find)
                .map(|r| r.range.end)
                .max();
            let scan_lo = anchor.map_or(scan_lo, |a| a.max(scan_lo));
            // Right guard: candidates may not overlap the first surviving
            // match right of the zone (survivors there provably start at or
            // after `ext_hi`, same argument as the anchor).
            let guard = store
                .decorations_in(ext_hi..scan_hi)
                .filter(|r| is_find(r) && r.range.start >= ext_hi)
                .map(|r| r.range.start)
                .min();
            // Greedy re-scan through the one pure `scan`, mapped back to
            // absolute offsets.
            let slice = buffer.slice(scan_lo..scan_hi);
            let (rel, _) = scan(&slice, query);
            let mut spans: Vec<Range<u32>> =
                rel.into_iter().map(|r| scan_lo + r.start..scan_lo + r.end).collect();
            if let Some(guard) = guard {
                spans.retain(|s| s.end <= guard);
            }
            // Windowed insert: the batch merges into its ≤band without the
            // O(M log M) whole-store `to_vec()`+re-sort of `add_sorted_batch`.
            // Byte-identical to that here (the window is already cleared of
            // finds), pinned by `splice_batch_equals_naive`.
            store.splice_sorted_batch(&spans, DecorationKind::FindMatch, Stickiness::NeverGrows);
        }
        // An active match a window removed transfers to a re-created match at
        // its exact (post-patch) start, if any — the position half moves with
        // it (lockstep).
        if let Some(start) = removed_active_start {
            let found = store
                .decorations_in(start..start)
                .find(|r| is_find(r) && r.range.start == start);
            self.active = found.as_ref().map(|r| r.id);
            self.active_start = found.map(|r| r.range.start);
        }
        // Cap bookkeeping — only when it can matter. The O(1) `find_count`
        // gates it (not `store.len()`, which counts diagnostics/snippets too, so
        // gating on it would scale with unrelated decorations). The BODY is
        // O(log) too: the live count is `find_count()` (O(1) root read) and the
        // two trim boundaries are ordinal `nth_find` lookups (O(log M)) — so a
        // capped commit costs the same regardless of decoration count (pinned by
        // `capped_commit_is_decoration_count_independent`).
        let fc = store.find_count();
        if fc > FIND_MATCH_CAP {
            // Trim from the TAIL: the set stays a prefix of the document. The two
            // boundaries are the (cap-1)-th and cap-th finds in `(start,id)` order.
            if let (Some((_, kept)), Some((_, first_trimmed))) =
                (store.nth_find(FIND_MATCH_CAP - 1), store.nth_find(FIND_MATCH_CAP))
            {
                let (kept_end, trim_from) = (kept.end, first_trimmed.start);
                store.take_matching_in(trim_from..u32::MAX, |r| {
                    is_find(r) && r.range.start >= trim_from
                });
                self.capped = true;
                self.coverage_end = kept_end;
                self.capped_stale = false; // full to the brim — nothing to refill
            }
        } else if self.capped {
            // Below the cap with frozen coverage: the refill can grow it.
            self.capped_stale = fc < FIND_MATCH_CAP;
        }
        // An active match destroyed without passing through a window (e.g.
        // collapsed and dropped by `apply_patch`, or tail-trimmed) clears —
        // probed at the tracked start (O(log)). `active_start` tracks the
        // decoration's start exactly (Bias::Right), so a present match overlaps
        // the zero-width `s..s` query.
        if let (Some(id), Some(s)) = (self.active, self.active_start) {
            let present = store.decorations_in(s..s).any(|r| is_find(&r) && r.id == id);
            if !present {
                self.active = None;
                self.active_start = None;
            }
        }
    }

    /// Select the next match from the caret and return its live range. The set
    /// is always current, so this is a pure store walk. `head`/`selection` are
    /// the newest selection's head and extent.
    pub fn find_next(
        &mut self,
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
    ) -> Option<Range<u32>> {
        self.navigate(head, selection, store, true)
    }

    /// Select the previous match from the caret — the
    /// [`find_next`](Self::find_next) mirror.
    pub fn find_prev(
        &mut self,
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
    ) -> Option<Range<u32>> {
        self.navigate(head, selection, store, false)
    }

    /// Wholesale-replace the `FindMatch` decorations from a fresh scan — the
    /// query-change path and the capped refill (the two legitimately
    /// whole-document find ops). The active match survives iff a new match
    /// starts exactly at the old active's tracked start.
    fn rescan(&mut self, buffer: &Buffer, store: &mut DecorationStore, now_ms: u64) {
        let old_active_start =
            self.active.and_then(|id| store.decoration_range(id)).map(|r| r.start);
        store.take_matching_in(0..u32::MAX, is_find);
        self.active = None;
        self.active_start = None;
        if let Some(q) = &self.query {
            let (spans, capped) = scan_buffer(buffer, q, crate::buffer::SCAN_WINDOW);
            self.capped = capped;
            self.coverage_end =
                if capped { spans.last().map_or(0, |s| s.end) } else { buffer.len() };
            let ids = store.add_sorted_batch(
                spans.iter().cloned(),
                DecorationKind::FindMatch,
                Stickiness::NeverGrows,
            );
            if let Some(start) = old_active_start {
                if let Some(i) = spans.iter().position(|s| s.start == start) {
                    self.active = Some(ids[i]); // the active match survives
                    self.active_start = Some(spans[i].start); // == start (lockstep)
                }
            }
        } else {
            self.capped = false;
            self.coverage_end = buffer.len();
        }
        self.capped_stale = false;
        self.last_scan_ms = now_ms;
    }

    /// Pick the next/previous match relative to the caret and set the active
    /// id. The live set is the store's `FindMatch` decorations in `(start, id)`
    /// = document order; collapsed (empty) ones are skipped (normally vacuous,
    /// since the mover drops them). Sitting exactly on a match steps by position
    /// (repeated-press cycling); otherwise it scans from `head` and wraps.
    fn navigate(
        &mut self,
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
        forward: bool,
    ) -> Option<Range<u32>> {
        // The live set is the store's `FindMatch` decorations in `(start, id)`
        // order — but we never materialize it. Count, rank, and the r-th match
        // are three O(log) store queries that reproduce the full-list branch
        // table exactly (pinned by `navigate_equals…`).
        let count = store.find_count();
        if count == 0 {
            self.active = None;
            self.active_start = None;
            return None;
        }
        // "On a match" ⇔ the find starting at `selection.start` IS `selection`.
        // `find_rank_before(selection.start)` is that find's rank in one descent;
        // only a find whose start equals `selection.start` can equal `selection`
        // (finds are disjoint), so this reproduces `position(|r| *r == selection)`.
        let on = {
            let r = store.find_rank_before(selection.start);
            store.nth_find(r).filter(|(_, rng)| *rng == selection).map(|_| r)
        };
        let pick = match (on, forward) {
            (Some(r), true) => (r + 1) % count,
            (Some(r), false) => (r + count - 1) % count,
            // First find with start ≥ head, wrapping past the last to 0.
            (None, true) => store.find_rank_before(head) % count,
            // Last find with start < head, wrapping before the first to count−1.
            (None, false) => (store.find_rank_before(head) + count - 1) % count,
        };
        let (id, range) = store.nth_find(pick)?;
        self.active = Some(id);
        self.active_start = Some(range.start);
        Some(range)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(text: &str, case_sensitive: bool) -> FindQuery {
        FindQuery { text: text.into(), case_sensitive }
    }

    #[test]
    fn matches_are_leftmost_and_non_overlapping() {
        // "aaaa" / "aa" pairs into two, not three overlapping.
        assert_eq!(scan("aaaa", &q("aa", true)), (vec![0..2, 2..4], false));
        // "ababa" / "aba": second probe starts at 3, "ba" is too short → one hit.
        let (spans, capped) = scan("ababa", &q("aba", true));
        assert_eq!((spans.len(), spans.first().cloned(), capped), (1, Some(0..3), false));
    }

    #[test]
    fn case_fold_is_ascii_and_toggled() {
        assert_eq!(scan("AbAb", &q("ab", false)), (vec![0..2, 2..4], false));
        assert_eq!(scan("AbAb", &q("ab", true)).0, Vec::<Range<u32>>::new());
        assert_eq!(scan("AbAb", &q("Ab", true)), (vec![0..2, 2..4], false));
    }

    #[test]
    fn empty_query_matches_nothing() {
        assert_eq!(scan("anything", &q("", false)), (Vec::new(), false));
        assert_eq!(scan("anything", &q("", true)), (Vec::new(), false));
    }

    #[test]
    fn a_query_longer_than_the_text_finds_nothing() {
        assert_eq!(scan("hi", &q("hello", true)), (Vec::new(), false));
    }

    #[test]
    fn scan_equals_the_naive_oracle() {
        // The memchr fast paths must be bit-identical to the straightforward
        // naive scan, kept here verbatim as the oracle.
        fn naive(text: &str, query: &FindQuery) -> (Vec<Range<u32>>, bool) {
            let needle = query.text.as_bytes();
            if needle.is_empty() {
                return (Vec::new(), false);
            }
            let hay = text.as_bytes();
            let hit = |window: &[u8]| {
                if query.case_sensitive {
                    window == needle
                } else {
                    window.iter().zip(needle).all(|(a, b)| a.eq_ignore_ascii_case(b))
                }
            };
            let mut spans = Vec::new();
            let mut i = 0;
            while i + needle.len() <= hay.len() {
                if hit(&hay[i..i + needle.len()]) {
                    if spans.len() == FIND_MATCH_CAP {
                        return (spans, true);
                    }
                    spans.push(i as u32..(i + needle.len()) as u32);
                    i += needle.len();
                } else {
                    i += 1;
                }
            }
            (spans, false)
        }

        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let alphabet = ['a', 'A', 'b', 'B', 'c', ' '];
        for round in 0..200 {
            let text: String = (0..(next() % 120)).map(|_| alphabet[(next() % 6) as usize]).collect();
            let needle: String = (0..1 + (next() % 4)).map(|_| alphabet[(next() % 6) as usize]).collect();
            for cs in [true, false] {
                let query = q(&needle, cs);
                assert_eq!(
                    scan(&text, &query),
                    naive(&text, &query),
                    "round {round}: {needle:?} (cs={cs}) in {text:?}"
                );
            }
        }
    }

    #[test]
    fn scan_caps_and_reports() {
        // One past the cap trips it (both case paths)…
        let text = "ab".repeat(FIND_MATCH_CAP + 5);
        let (spans, capped) = scan(&text, &q("ab", true));
        assert_eq!((spans.len(), capped), (FIND_MATCH_CAP, true));
        let text = "aB".repeat(FIND_MATCH_CAP + 1);
        let (spans, capped) = scan(&text, &q("ab", false));
        assert_eq!((spans.len(), capped), (FIND_MATCH_CAP, true));
        // …exactly the cap does not (the cap trips only on an overflowing hit).
        let text = "ab".repeat(FIND_MATCH_CAP);
        let (spans, capped) = scan(&text, &q("ab", true));
        assert_eq!((spans.len(), capped), (FIND_MATCH_CAP, false));
    }

    /// The windowed buffer scan must be byte-identical to `scan` over the
    /// materialized text — spans AND capped flag — for every window size,
    /// including windows tiny enough that matches straddle every seam, the
    /// needle exceeds the window, multibyte chars sit on window ends (the
    /// boundary-snap paths), and self-overlapping needles cross seams.
    #[test]
    fn windowed_scan_equals_the_whole_text_scan() {
        let corpora = [
            String::new(),
            "aaaa".into(),
            "abababab".into(),
            "aa aa aaa aaaa a".into(),
            "the fox\nreturns the fox to the FOX den\nfoxfoxfox\n".repeat(40),
            // Multibyte chars packed so window ends land mid-char for most
            // small window sizes; matches sit between and across them.
            "ä🦀ab🦀äab日本語ab".repeat(30),
            "🦀🦀🦀ab🦀🦀🦀".repeat(50),
        ];
        let needles = ["a", "ab", "aa", "aaa", "fox", "FOX", "🦀ä", "ab🦀", "abababababab", "語ab"];
        for text in &corpora {
            let b = Buffer::new(text).unwrap();
            let full_text = b.text();
            for needle in needles {
                for cs in [true, false] {
                    let query = q(needle, cs);
                    let expect = scan(&full_text, &query);
                    for window in [2, 3, 5, 7, 16, 64, 4096] {
                        assert_eq!(
                            scan_buffer(&b, &query, window),
                            expect,
                            "window {window}, needle {needle:?} (cs={cs}) in {:?}…",
                            &text[..text.len().min(24)]
                        );
                    }
                }
            }
        }
    }

    /// The windowed scan reproduces the cap semantics exactly: capped ⇒ the
    /// identical 10k-prefix, and exactly-at-cap ⇒ not capped.
    #[test]
    fn windowed_scan_caps_like_the_whole_text_scan() {
        for extra in [0usize, 3] {
            let text = "ab".repeat(FIND_MATCH_CAP + extra);
            let b = Buffer::new(&text).unwrap();
            let query = q("ab", true);
            assert_eq!(scan_buffer(&b, &query, 4096), scan(&b.text(), &query), "extra {extra}");
        }
    }

    /// The O(log) [`FindState::navigate`]
    /// (`find_count`/`find_rank_before`/`nth_find`) must pick the byte-identical
    /// `(id, range)` — and set `active`/`active_start` in lockstep — as a
    /// straightforward whole-list walk, over random find sets and random
    /// `(head, selection, forward)`: on-match cycling both ways, off-match scan +
    /// wrap both ways, and the empty set. Non-find decorations are interleaved as
    /// noise that neither path may count.
    #[test]
    fn navigate_equals_the_full_list_walk() {
        // The full-list navigate, retained verbatim as the oracle.
        fn navigate_ref(
            store: &DecorationStore,
            head: u32,
            selection: Range<u32>,
            forward: bool,
        ) -> Option<(DecorationId, Range<u32>)> {
            let live: Vec<(DecorationId, Range<u32>)> = store
                .decorations_in(0..u32::MAX)
                .filter(|r| is_find(r) && r.range.start < r.range.end)
                .map(|r| (r.id, r.range.clone()))
                .collect();
            if live.is_empty() {
                return None;
            }
            let on = live.iter().position(|(_, r)| *r == selection);
            let pick = match (on, forward) {
                (Some(p), true) => (p + 1) % live.len(),
                (Some(p), false) => (p + live.len() - 1) % live.len(),
                (None, true) => live.iter().position(|(_, r)| r.start >= head).unwrap_or(0),
                (None, false) => {
                    live.iter().rposition(|(_, r)| r.start < head).unwrap_or(live.len() - 1)
                }
            };
            Some(live[pick].clone())
        }

        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for trial in 0..3000u32 {
            let mut store = DecorationStore::new();
            // Disjoint ascending non-empty finds.
            let n = next() % 12;
            let mut pos = 0u32;
            let mut find_spans: Vec<Range<u32>> = Vec::new();
            for _ in 0..n {
                pos += (next() % 5) as u32;
                let len = 1 + (next() % 4) as u32;
                find_spans.push(pos..pos + len);
                pos += len;
            }
            store.add_sorted_batch(
                find_spans.iter().cloned(),
                DecorationKind::FindMatch,
                Stickiness::NeverGrows,
            );
            // Non-find noise (must be ignored by both paths).
            for _ in 0..(next() % 5) {
                let s = (next() % u64::from(pos + 1)) as u32;
                let l = (next() % 4) as u32;
                store.add_decoration(s..s + l, DecorationKind::AutoClosePair, Stickiness::AlwaysGrows);
            }
            // No persisted find may be empty (FindMatch is EmptyPolicy::Drop);
            // the O(1) `find_count` counts all finds, so this must hold for it to
            // equal a `start < end` filter.
            debug_assert!(
                store.decorations_in(0..u32::MAX).filter(is_find).all(|r| r.range.start < r.range.end),
                "trial {trial}: an empty FindMatch persisted"
            );

            let head = (next() % u64::from(pos + 6)) as u32;
            // Half the time, sit exactly on a match to exercise on-match cycling.
            let selection = if !find_spans.is_empty() && next() % 2 == 0 {
                find_spans[(next() as usize) % find_spans.len()].clone()
            } else {
                let a = (next() % u64::from(pos + 6)) as u32;
                let b = a + (next() % 5) as u32;
                a..b
            };
            for forward in [true, false] {
                let want = navigate_ref(&store, head, selection.clone(), forward);
                let mut state = FindState::new();
                let got = state.navigate(head, selection.clone(), &store, forward);
                match (&want, &got) {
                    (Some((id, range)), Some(got_range)) => {
                        assert_eq!(got_range, range, "trial {trial} fwd={forward}: range");
                        assert_eq!(state.active, Some(*id), "trial {trial} fwd={forward}: active id");
                        assert_eq!(
                            state.active_start,
                            Some(range.start),
                            "trial {trial} fwd={forward}: active_start tracks the pick"
                        );
                    }
                    (None, None) => {
                        assert_eq!(state.active, None, "trial {trial} fwd={forward}");
                        assert_eq!(state.active_start, None, "trial {trial} fwd={forward}");
                    }
                    _ => panic!("trial {trial} fwd={forward}: {want:?} vs {got:?}"),
                }
            }
        }
    }
}
