//! The change currency: [`Edit`] / [`Patch`] and the one position-mapping
//! function every derived position flows through.
//!
//! A [`Patch`] is the sole record of "what changed" produced by a committed
//! transaction. Selections, diagnostics, snippet stops, and find matches all
//! move through [`Patch::map_offset`] — one function, one contract — so a
//! position is never maintained two ways and cannot drift out of sync.
//!
//! Mapping is **prefix-preserving**: an edit deletes `[s, e)` and inserts `L`
//! bytes at `s`; a position inside the overwritten prefix keeps its absolute
//! offset, only the deleted tail *beyond* the insert clamps, and a pure
//! deletion collapses its interior to `s`. Bias matters at exactly one place —
//! a position sitting on the insertion point. Touching edits can stack several
//! insertions on one point (a formatter's `TextEdit` list, a snippet
//! expansion); bias then treats the stack as one insertion — `Right` lands
//! after all of it, `Left` before it.

use std::cell::RefCell;
use std::ops::Range;

use crate::coords::Bias;

/// One span's coordinate transform: the pre-edit byte range `old` became the
/// post-edit byte range `new`. For a single edit `old.start == new.start`; in a
/// multi-edit [`Patch`], `new` is already in final post-edit coordinates.
#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Edit {
    /// Pre-edit byte range that was replaced.
    pub old: Range<u32>,
    /// Post-edit byte range it became.
    pub new: Range<u32>,
}

/// An ordered set of [`Edit`]s — the one change currency.
///
/// Invariant (debug-asserted on every [`Patch::push`]): edits are sorted
/// ascending and disjoint in *both* coordinate spaces (touching is allowed).
#[derive(Clone, Default, Debug)]
pub struct Patch(Vec<Edit>);

impl Patch {
    /// An empty patch (maps every position to itself).
    #[must_use]
    pub fn new() -> Self {
        Self(Vec::new())
    }

    /// A patch of a single edit.
    #[must_use]
    pub fn single(edit: Edit) -> Self {
        Self(vec![edit])
    }

    /// The edits, in ascending order.
    #[must_use]
    pub fn edits(&self) -> &[Edit] {
        &self.0
    }

    /// Whether the patch changes nothing.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Append an edit. Debug-panics if it breaks the ascending-disjoint
    /// invariant in either coordinate space.
    pub fn push(&mut self, edit: Edit) {
        debug_assert!(edit.old.start <= edit.old.end && edit.new.start <= edit.new.end);
        if let Some(last) = self.0.last() {
            debug_assert!(
                edit.old.start >= last.old.end,
                "old ranges must be ascending and disjoint"
            );
            debug_assert!(
                edit.new.start >= last.new.end,
                "new ranges must be ascending and disjoint"
            );
        }
        self.0.push(edit);
    }

    /// Map a pre-edit byte offset to its post-edit offset (the transit table).
    ///
    /// `bias` only matters when `offset` sits exactly on an insertion point:
    /// `Left` keeps it before the inserted text, `Right` moves it after.
    ///
    /// Several edits may share that point (touching is allowed): one replace
    /// ending at it, any number of pure inserts at it, one replace starting at
    /// it. `Right` moves past **all** of their text; `Left` stays before
    /// everything inserted *at* the point — though after a replace that *ends*
    /// there, since the point is that replace's trailing boundary.
    #[must_use]
    pub fn map_offset(&self, offset: u32, bias: Bias) -> u32 {
        // Complexity gate: this single-offset map scans the edit list, so calling
        // it once per derived position (folds/decorations) is O(positions·edits) —
        // the meter charges the scan so that cost shows as superlinear growth,
        // pushing callers with many positions toward `map_many`, whose batched
        // index charges only O(edits + queries).
        crate::perf::charge(self.0.len() as u64);
        let p = offset as i64;
        let mut shift: i64 = 0;
        for (i, e) in self.0.iter().enumerate() {
            let s = e.old.start as i64;
            let end = e.old.end as i64;
            if p < s {
                // In the gap before this edit — only prior deltas apply.
                return (p + shift) as u32;
            }
            if p <= end {
                // `e` is the FIRST edit at `p` — the one `Left` maps through.
                // `Right` maps through the LAST (see `last_at_point`); the edits
                // between them sit at `p` too, so their deltas join the shift.
                let k = if bias == Bias::Right { last_at_point(&self.0, i, p) } else { i };
                shift += self.0[i..k].iter().map(net_delta).sum::<i64>();
                let e = &self.0[k];
                let s = i64::from(e.old.start);
                let sn = s + shift; // == e.new.start
                return map_within(p, s, i64::from(e.old.end), sn, inserted_len(e), bias) as u32;
            }
            // This edit is fully before `p`; accumulate its delta and continue.
            shift += net_delta(e);
        }
        (p + shift) as u32
    }

    /// Map a pre-edit byte range, biasing each endpoint independently, and
    /// never inverting: if the mapped start would exceed the mapped end, both
    /// collapse to the mapped end.
    #[must_use]
    pub fn map_range(
        &self,
        range: Range<u32>,
        start_bias: Bias,
        end_bias: Bias,
    ) -> Range<u32> {
        let start = self.map_offset(range.start, start_bias);
        let end = self.map_offset(range.end, end_bias);
        start.min(end)..end
    }

    /// Map a batch of offsets — each with its own [`Bias`] — through the patch,
    /// appending results to `out` in query order. Semantically each entry is
    /// exactly `map_offset(offset, bias)`, but the batch builds a prefix-shift
    /// index once (O(edits)) and binary-searches per query (O(log edits)), so
    /// the whole call is **O(edits + queries·log edits)** instead of the
    /// per-offset loop's O(queries·edits). That is the difference between
    /// O(C²) and O(C) when a document-scale multi-cursor edit produces a
    /// C-edit patch and every derived view (C selections, N decorations, F fold
    /// openers) must rebase through it.
    ///
    /// No ordering precondition on `queries` — the binary search makes it
    /// order-independent, so nested/overlapping ranges (decorations) are fine.
    pub fn map_many(&self, queries: &[(u32, Bias)], out: &mut Vec<u32>) {
        // Complexity gate: deliberately UNMETERED. The batched map is the
        // accepted O(edits + queries) bandwidth-class rebase — shifting derived
        // positions is plain offset arithmetic, not a semantic pass. The meter
        // measures semantic work plus the per-query `map_offset` cost; charging
        // the batched mover here would set a linear floor that masks an *added*
        // semantic O(n) cost elsewhere. Reverting a caller from `map_many` back
        // to a per-item `map_offset` loop reappears on the meter (that call IS
        // charged).
        out.clear();
        out.reserve(queries.len());
        let edits = &self.0;
        if edits.is_empty() {
            out.extend(queries.iter().map(|&(p, _)| p));
            return;
        }
        // `pref[i]` = accumulated net length delta of `edits[..i]`. Since edits
        // are sorted ascending and disjoint, their old ranges — and old.end —
        // are ascending, so a `partition_point` on old.end locates the first
        // edit with `old.end >= p` (map_offset's "first `p <= end`") and
        // `pref[k]` is the shift from all strictly-earlier edits (all `end < p`).
        // `Right` then moves `k` on to the last edit at `p` exactly as the scan
        // does (the same `last_at_point`), and `pref` at the moved index already
        // holds the skipped edits' deltas.
        // The prefix table is a thread-local reused across every `map_many` call —
        // and every mover (selections, decorations, folds, offset sets) flows
        // through here per edit — so the batch mapper allocates nothing.
        thread_local! {
            static PREF: RefCell<Vec<i64>> = const { RefCell::new(Vec::new()) };
        }
        PREF.with(|cell| {
            let pref = &mut *cell.borrow_mut();
            pref.clear();
            pref.push(0);
            for e in edits {
                pref.push(pref[pref.len() - 1] + net_delta(e));
            }
            for &(offset, bias) in queries {
                let p = i64::from(offset);
                let first = edits.partition_point(|e| i64::from(e.old.end) < p);
                let mapped = if first == edits.len() || p < i64::from(edits[first].old.start) {
                    // Past every edit, or in the gap before edit `first` — only
                    // prior deltas apply.
                    p + pref[first]
                } else {
                    let k = if bias == Bias::Right { last_at_point(edits, first, p) } else { first };
                    let e = &edits[k];
                    let s = i64::from(e.old.start);
                    map_within(p, s, i64::from(e.old.end), s + pref[k], inserted_len(e), bias)
                };
                out.push(mapped as u32);
            }
        });
    }
}

/// An edit's net length change (inserted minus deleted bytes).
fn net_delta(e: &Edit) -> i64 {
    inserted_len(e) - i64::from(e.old.end - e.old.start)
}

/// An edit's inserted length (its post-edit span).
fn inserted_len(e: &Edit) -> i64 {
    i64::from(e.new.end - e.new.start)
}

/// The LAST edit sitting at point `p`, given `first` — the first edit with
/// `old.end >= p`, which must also have `old.start <= p` (`p` is in or on it).
///
/// Ascending-disjoint-with-touching admits exactly one shape of edits sharing a
/// point, contiguous and in this order: at most one replace *ending* at `p`, any
/// number of pure inserts *at* `p`, at most one replace *starting* at `p`. They
/// are `edits[first..=last]`, where `last` is the final edit with
/// `old.start <= p` — old starts are non-decreasing, so a partition point finds
/// it in O(log edits). When `p` is strictly inside `first`, nothing later can
/// start at or before `p`, so `last == first` and interior mapping is unchanged.
///
/// `Bias::Right` maps through `last` (after ALL text at `p`); `Left` through
/// `first`. Both mappers share this one choice, so they cannot drift apart.
fn last_at_point(edits: &[Edit], first: usize, p: i64) -> usize {
    debug_assert!(i64::from(edits[first].old.start) <= p && p <= i64::from(edits[first].old.end));
    first + edits[first..].partition_point(|e| i64::from(e.old.start) <= p) - 1
}

/// The per-edit transit table for a position `p` known to lie in `[s, end]`.
/// `sn` is the edit's post-edit start (`s + accumulated shift`), `l` the
/// inserted length.
fn map_within(p: i64, s: i64, end: i64, sn: i64, l: i64, bias: Bias) -> i64 {
    if p == s {
        // Insertion point / replace start — the only bias-dependent case.
        return if bias == Bias::Right { sn + l } else { sn };
    }
    if p < end {
        // Strictly interior. Prefix (within the inserted length) keeps its
        // relative offset; the deleted tail beyond it clamps to sn + l.
        let rel = p - s;
        return if rel <= l { sn + rel } else { sn + l };
    }
    // p == end: the trailing boundary, continuous with the post-edit shift.
    sn + l
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::coords::Bias::{Left, Right};

    fn edit(os: u32, oe: u32, ns: u32, ne: u32) -> Edit {
        Edit { old: os..oe, new: ns..ne }
    }

    #[test]
    fn empty_patch_is_identity() {
        let p = Patch::new();
        for o in 0..10 {
            assert_eq!(p.map_offset(o, Left), o);
            assert_eq!(p.map_offset(o, Right), o);
        }
    }

    #[test]
    fn pure_insertion_biases_at_the_point() {
        // Insert 3 bytes at offset 5: old 5..5, new 5..8.
        let p = Patch::single(edit(5, 5, 5, 8));
        assert_eq!(p.map_offset(4, Left), 4); // before: unchanged
        assert_eq!(p.map_offset(5, Left), 5); // at point, Left: stays before
        assert_eq!(p.map_offset(5, Right), 8); // at point, Right: after insert
        assert_eq!(p.map_offset(6, Left), 9); // after: shifted +3
    }

    #[test]
    fn pure_deletion_collapses_interior_to_start() {
        // Delete 3..7 (L = 0): old 3..7, new 3..3.
        let p = Patch::single(edit(3, 7, 3, 3));
        assert_eq!(p.map_offset(2, Left), 2); // before
        assert_eq!(p.map_offset(3, Left), 3); // start
        assert_eq!(p.map_offset(5, Left), 3); // interior collapses to start
        assert_eq!(p.map_offset(7, Left), 3); // end collapses to start
        assert_eq!(p.map_offset(8, Left), 4); // after: shifted -4
    }

    #[test]
    fn net_growth_replace_preserves_interior_prefix() {
        // Replace 2..4 (len 2) with 6 bytes: old 2..4, new 2..8. L = 6.
        let p = Patch::single(edit(2, 4, 2, 8));
        // A marker at offset 3 (inside the old range, within the inserted
        // prefix since 3-2=1 ≤ 6) keeps its absolute offset — NOT collapsed.
        assert_eq!(p.map_offset(3, Left), 3);
        assert_eq!(p.map_offset(2, Left), 2);
        assert_eq!(p.map_offset(2, Right), 8);
        assert_eq!(p.map_offset(4, Left), 8); // end → sn + L
        assert_eq!(p.map_offset(5, Left), 9); // after: shifted +4
    }

    #[test]
    fn net_shrink_replace_clamps_tail_beyond_insert() {
        // Replace 2..8 (len 6) with 2 bytes: old 2..8, new 2..4. L = 2.
        let p = Patch::single(edit(2, 8, 2, 4));
        assert_eq!(p.map_offset(3, Left), 3); // prefix (3-2=1 ≤ 2): kept
        assert_eq!(p.map_offset(4, Left), 4); // prefix (4-2=2 ≤ 2): kept
        assert_eq!(p.map_offset(5, Left), 4); // tail (5-2=3 > 2): clamps to sn+L
        assert_eq!(p.map_offset(8, Left), 4); // end: sn+L
        assert_eq!(p.map_offset(9, Left), 5); // after: shifted -4
    }

    #[test]
    fn multi_edit_accumulates_shift() {
        // Two edits: insert 2 at offset 1 (old 1..1,new 1..3), delete 5..6
        // (old 5..6,new 7..7 — post-edit coords include the +2 from edit 1).
        let mut p = Patch::new();
        p.push(edit(1, 1, 1, 3)); // +2
        p.push(edit(5, 6, 7, 7)); // -1, in post-edit space
        assert_eq!(p.map_offset(0, Left), 0); // before both
        assert_eq!(p.map_offset(1, Right), 3); // at first insert, Right
        assert_eq!(p.map_offset(4, Left), 6); // between edits: +2
        assert_eq!(p.map_offset(5, Left), 7); // start of deletion
        assert_eq!(p.map_offset(6, Left), 7); // end of deletion → collapses
        assert_eq!(p.map_offset(7, Left), 8); // after both: +2 -1 = +1
    }

    /// Both mappers, asserted to agree, for one query — so every pinned
    /// boundary case below checks `map_many` as well as `map_offset`.
    fn mapped(p: &Patch, offset: u32, bias: Bias) -> u32 {
        let single = p.map_offset(offset, bias);
        let mut batch = Vec::new();
        p.map_many(&[(offset, bias)], &mut batch);
        assert_eq!(batch, vec![single], "map_many diverged from map_offset at {offset} {bias:?}");
        single
    }

    #[test]
    fn right_bias_passes_a_touching_replace_and_the_insert_after_it() {
        // Regression (fails-first): "abcd" with [replace(1..2, "X"), insert(2,
        // "Y")] → "aXYcd". A Right-biased caret at 2 (between `b` and `c`) must
        // land after ALL text at 2 — offset 3, between `Y` and `c` — not stop
        // after the touching replace's `X` (offset 2, splitting X|Y).
        let mut p = Patch::new();
        p.push(edit(1, 2, 1, 2)); // "b" → "X"
        p.push(edit(2, 2, 2, 3)); // insert "Y" at 2, touching the replace's end
        assert_eq!(mapped(&p, 2, Right), 3, "Right passes X and Y");
        // Left keeps before the text inserted AT 2; the replace's end is its
        // trailing boundary (continuous with its post-edit shift), so after X.
        assert_eq!(mapped(&p, 2, Left), 2, "Left stays before Y");
    }

    #[test]
    fn right_bias_passes_every_same_offset_insert() {
        // Regression (fails-first): two pure inserts at 2 ("Y" then "Z", the
        // stable-sorted transaction order). Right must pass both (4), not stop
        // between them (3); Left stays before both (2).
        let mut p = Patch::new();
        p.push(edit(2, 2, 2, 3));
        p.push(edit(2, 2, 3, 4));
        assert_eq!(mapped(&p, 2, Right), 4, "Right passes both inserts");
        assert_eq!(mapped(&p, 2, Left), 2, "Left stays before both inserts");
        // The neighbours are untouched by the fix: plain shift on either side.
        assert_eq!(mapped(&p, 1, Right), 1);
        assert_eq!(mapped(&p, 3, Left), 5);
    }

    #[test]
    fn map_range_never_inverts() {
        // Delete a range that fully contains a tracked range → it collapses,
        // start never exceeds end.
        let p = Patch::single(edit(0, 10, 0, 0)); // delete everything
        let r = p.map_range(3..7, Right, Left);
        assert!(r.start <= r.end, "range inverted: {r:?}");
        assert_eq!(r, 0..0);
    }

    /// Build a valid (ascending, disjoint, post-edit-coordinate) patch from a
    /// script of `(gap_before, old_len, new_len)` triples — exactly the shape a
    /// committed transaction produces.
    fn patch_from_script(script: &[(u32, u32, u32)]) -> Patch {
        let mut p = Patch::new();
        let mut old = 0u32;
        let mut shift: i64 = 0;
        for &(gap, old_len, new_len) in script {
            old += gap;
            let ns = (i64::from(old) + shift) as u32;
            p.push(Edit { old: old..old + old_len, new: ns..ns + new_len });
            shift += i64::from(new_len) - i64::from(old_len);
            old += old_len;
        }
        p
    }

    /// Deterministic LCG — no rand dep, reproducible. `next(n)` draws from `0..n`.
    fn lcg(seed: u64) -> impl FnMut(u32) -> u32 {
        let mut state = seed;
        move |n| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % n
        }
    }

    /// A script dense in the points several edits SHARE — the shapes a uniform
    /// draw rarely stacks: a zero gap half the time (the next edit touches the
    /// previous one's end) and a pure insert half the time (so a run of zero-gap
    /// inserts is several edits at ONE offset, optionally capped by a replace
    /// starting there). 0..=5 edits; the extent stays under 40 bytes.
    fn shared_point_script(next: &mut impl FnMut(u32) -> u32) -> Vec<(u32, u32, u32)> {
        (0..next(6))
            .map(|_| {
                let gap = if next(2) == 0 { 0 } else { 1 + next(4) };
                let old_len = if next(2) == 0 { 0 } else { 1 + next(3) };
                (gap, old_len, next(4))
            })
            .collect()
    }

    /// Whether old offset `p` is a point two edits share, as `(touching,
    /// same_offset)`: `touching` — a replace ENDS at `p` and another edit starts
    /// there; `same_offset` — two or more edits START at `p`.
    fn shared_point_kind(patch: &Patch, p: u32) -> (bool, bool) {
        let starts = patch.edits().iter().filter(|e| e.old.start == p).count();
        let ends_here = patch.edits().iter().any(|e| e.old.start < p && e.old.end == p);
        (ends_here && starts > 0, starts > 1)
    }

    /// Where a zero-width marker at old offset `p` REALLY lands — an oracle
    /// independent of the shift arithmetic. It rebuilds the post-edit text as a
    /// token stream (each surviving old byte is `Some(i)`, each inserted byte
    /// `None`), then places the marker by what it is attached to:
    ///
    /// - `Left` binds to the byte before it: just after old byte `p - 1` if that
    ///   survived (so before everything inserted at `p`); if an edit ENDING at
    ///   `p` replaced it, just after that replacement (`p` is its trailing
    ///   boundary). At `p == 0`, the very start.
    /// - `Right` binds to the byte after it: just before old byte `p` if that
    ///   survived (so after everything inserted at `p`); if an edit STARTING at
    ///   `p` replaced it, just after that replacement (a replace is an insert at
    ///   its start). At `p == old_len`, the very end.
    ///
    /// `None` for a strictly interior position (byte on the needed side replaced
    /// by an edit that reaches past `p`) — that is the prefix-preserving clamp,
    /// pinned by the single-edit tests, not a bias question.
    fn marker_oracle(old_len: u32, patch: &Patch, p: u32, bias: Bias) -> Option<u32> {
        let edits = patch.edits();
        let mut toks: Vec<Option<u32>> = Vec::new();
        let mut owner: Vec<Option<usize>> = vec![None; old_len as usize];
        let mut replacement_end: Vec<usize> = Vec::new(); // token index after edit k's text
        let mut cursor = 0;
        for (k, e) in edits.iter().enumerate() {
            toks.extend((cursor..e.old.start).map(Some));
            for i in e.old.clone() {
                owner[i as usize] = Some(k);
            }
            toks.extend((e.new.start..e.new.end).map(|_| None));
            replacement_end.push(toks.len());
            cursor = e.old.end;
        }
        toks.extend((cursor..old_len).map(Some));
        let survivor = |i: u32| toks.iter().position(|&t| t == Some(i)).expect("surviving byte");
        let at = match bias {
            Left if p == 0 => 0,
            Left => match owner[p as usize - 1] {
                None => survivor(p - 1) + 1,
                Some(k) if edits[k].old.end == p => replacement_end[k],
                Some(_) => return None,
            },
            Right if p == old_len => toks.len(),
            Right => match owner[p as usize] {
                None => survivor(p),
                Some(k) if edits[k].old.start == p => replacement_end[k],
                Some(_) => return None,
            },
        };
        Some(at as u32)
    }

    #[test]
    fn bias_tracks_a_real_marker_through_touching_and_same_offset_edits() {
        // Property: at every non-interior offset, both biases, `map_offset` (and
        // `map_many`, via `mapped`) puts a marker exactly where the token-stream
        // rebuild says it really is — over random patches dense in touching and
        // same-offset edits, the shapes formatter `TextEdit` lists and snippet
        // expansion produce.
        let mut next = lcg(0xD1B5_4A32_D192_ED03);
        let (mut touching, mut same_offset) = (0u32, 0u32);
        for _ in 0..2000 {
            let script = shared_point_script(&mut next);
            let patch = patch_from_script(&script);
            let old_len = patch.edits().last().map_or(0, |e| e.old.end) + next(3);
            for p in 0..=old_len {
                let (t, s) = shared_point_kind(&patch, p);
                for bias in [Left, Right] {
                    let Some(want) = marker_oracle(old_len, &patch, p, bias) else {
                        continue;
                    };
                    assert_eq!(
                        mapped(&patch, p, bias),
                        want,
                        "marker at {p} {bias:?} misplaced, script {script:?}"
                    );
                    touching += u32::from(t);
                    same_offset += u32::from(s);
                }
            }
        }
        // The generator must actually reach the shapes this test exists for.
        assert!(touching > 500, "only {touching} touching-point checks");
        assert!(same_offset > 500, "only {same_offset} same-offset checks");
    }

    #[test]
    fn map_many_matches_map_offset_oracle() {
        let mut next = lcg(0x9E37_79B9_7F4A_7C15);
        let (mut touching, mut same_offset) = (0u32, 0u32);
        for round in 0..800 {
            // Even rounds: uniform scripts (0..=5 edits; gaps + del/ins incl. 0,
            // so pure inserts/deletes). Odd rounds: scripts dense in the points
            // several edits share, where Right must pass every edit at the point
            // — the batch's partition-point search must land on the same edit
            // as the scan.
            let script: Vec<(u32, u32, u32)> = if round % 2 == 0 {
                (0..next(6)).map(|_| (next(5), next(4), next(4))).collect()
            } else {
                shared_point_script(&mut next)
            };
            let patch = patch_from_script(&script);
            // Query every offset in a window spanning before/through/after all
            // edits, both biases — unsorted order to exercise order-independence.
            let hi = 40u32;
            let mut queries: Vec<(u32, Bias)> = Vec::new();
            for o in 0..hi {
                queries.push((o, Left));
                queries.push((o, Right));
            }
            // Shuffle-ish: reverse half so the batch is not ascending.
            queries[..hi as usize].reverse();
            let mut got = Vec::new();
            patch.map_many(&queries, &mut got);
            assert_eq!(got.len(), queries.len());
            for (i, &(o, b)) in queries.iter().enumerate() {
                assert_eq!(
                    got[i],
                    patch.map_offset(o, b),
                    "map_many diverged at offset {o} bias {b:?}, script {script:?}"
                );
                let (t, s) = shared_point_kind(&patch, o);
                touching += u32::from(t);
                same_offset += u32::from(s);
            }
        }
        // Coverage, not luck: the shared-point shapes were actually queried.
        assert!(touching > 200, "only {touching} touching-point queries");
        assert!(same_offset > 200, "only {same_offset} same-offset queries");
    }

    #[test]
    fn map_many_empty_patch_is_identity() {
        let p = Patch::new();
        let q = vec![(7, Left), (3, Right), (0, Left)];
        let mut out = vec![999]; // pre-populated: map_many must clear it
        p.map_many(&q, &mut out);
        assert_eq!(out, vec![7, 3, 0]);
    }
}
