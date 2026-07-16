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
//! a position sitting on the insertion point.

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
        for e in &self.0 {
            let s = e.old.start as i64;
            let end = e.old.end as i64;
            let l = (e.new.end - e.new.start) as i64;
            if p < s {
                // In the gap before this edit — only prior deltas apply.
                return (p + shift) as u32;
            }
            if p <= end {
                let sn = s + shift; // == e.new.start
                return map_within(p, s, end, sn, l, bias) as u32;
            }
            // This edit is fully before `p`; accumulate its delta and continue.
            shift += l - (end - s);
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
                let delta =
                    i64::from(e.new.end - e.new.start) - i64::from(e.old.end - e.old.start);
                pref.push(pref[pref.len() - 1] + delta);
            }
            for &(offset, bias) in queries {
                let p = i64::from(offset);
                let k = edits.partition_point(|e| i64::from(e.old.end) < p);
                let mapped = if k == edits.len() {
                    p + pref[k]
                } else {
                    let e = &edits[k];
                    let s = i64::from(e.old.start);
                    let shift = pref[k];
                    if p < s {
                        p + shift // in the gap before edit k — only prior deltas apply
                    } else {
                        let end = i64::from(e.old.end);
                        let l = i64::from(e.new.end - e.new.start);
                        map_within(p, s, end, s + shift, l, bias)
                    }
                };
                out.push(mapped as u32);
            }
        });
    }
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

    #[test]
    fn map_many_matches_map_offset_oracle() {
        // Deterministic LCG — no rand dep, reproducible.
        let mut state: u64 = 0x9E37_79B9_7F4A_7C15;
        let mut next = |n: u32| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as u32) % n
        };
        for _ in 0..400 {
            let n_edits = next(6); // 0..=5 edits, incl. the empty-patch path
            let script: Vec<(u32, u32, u32)> = (0..n_edits)
                .map(|_| (next(5), next(4), next(4))) // gaps + del/ins incl. 0 (pure insert/delete)
                .collect();
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
            }
        }
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
