//! `OffsetSet` — a sorted set of byte offsets on a `SumTree`, stored **delta-gap**
//! (each item is the gap from the previous offset) so a text edit shifts the whole
//! set in O(edits · log n) rather than the O(n) of a per-offset rebase. Membership,
//! window, and first-at queries are O(log n) cursor descents. Backs `FoldSet`
//! (fold openers) and the decoration store.
//!
//! The offset of item `i` is the prefix sum of gaps through `i`; the tree's total
//! span is therefore the last offset. All offsets are unique and ascending.

use crate::coords::Bias;
use crate::patch::{Edit, Patch};
use crate::sum_tree::{Dimension, Item, SumTree, Summary};

#[derive(Clone, Copy, Debug)]
struct Gap(u32);

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct GapSummary {
    span: u32,
    count: u32,
}

impl Summary for GapSummary {
    fn add_summary(&mut self, o: &Self) {
        self.span += o.span;
        self.count += o.count;
    }
}

impl Item for Gap {
    type Summary = GapSummary;
    fn summary(&self) -> GapSummary {
        GapSummary { span: self.0, count: 1 }
    }
}

/// Seek by cumulative offset (the prefix sum of gaps).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Span(u32);
impl Dimension<GapSummary> for Span {
    fn add_summary(&mut self, s: &GapSummary) {
        self.0 += s.span;
    }
}

/// Seek by item index.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct Count(u32);
impl Dimension<GapSummary> for Count {
    fn add_summary(&mut self, s: &GapSummary) {
        self.0 += s.count;
    }
}

#[derive(Clone, Debug, Default)]
pub struct OffsetSet {
    tree: SumTree<Gap>,
}

impl OffsetSet {
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Build from strictly-ascending offsets.
    #[must_use]
    pub fn from_sorted(offsets: &[u32]) -> Self {
        debug_assert!(offsets.windows(2).all(|w| w[0] < w[1]), "offsets must be strictly ascending");
        let mut prev = 0;
        let gaps = offsets.iter().map(|&o| {
            let g = Gap(o - prev);
            prev = o;
            g
        });
        OffsetSet { tree: SumTree::from_items(gaps) }
    }

    #[must_use]
    pub fn len(&self) -> usize {
        self.tree.summary().count as usize
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.tree.is_empty()
    }

    /// The last offset (== the tree's total span), or 0 when empty.
    fn total(&self) -> u32 {
        self.tree.summary().span
    }

    /// Every offset, ascending — the O(n) reduce-to-Vec (fold-map rebuild, tests).
    #[must_use]
    pub fn offsets(&self) -> Vec<u32> {
        let mut acc = 0;
        self.tree
            .items()
            .into_iter()
            .map(|g| {
                acc += g.0;
                acc
            })
            .collect()
    }

    /// Whether `offset` is in the set. O(log n).
    ///
    /// A member is the END of an item's gap-span (`cumsum[i]`). `seek` returns the
    /// item CONTAINING `offset` with its span-start (`= cumsum[idx-1]`) and index
    /// `idx`; `offset` is a member iff it equals that start AND the start is a real
    /// member (`idx >= 1` — the span-start of item 0 is the origin, not a member),
    /// or `offset` is the tree total (always the last member).
    #[must_use]
    pub fn contains(&self, offset: u32) -> bool {
        if self.is_empty() {
            return false;
        }
        if offset == self.total() {
            return true;
        }
        match self.tree.seek::<Span, Count>(&Span(offset)) {
            Some((_, Span(start), Count(idx))) => start == offset && idx >= 1,
            None => false,
        }
    }

    /// The smallest offset `>= offset`, if any. O(log n).
    #[must_use]
    pub fn first_at_or_after(&self, offset: u32) -> Option<u32> {
        if self.is_empty() || offset > self.total() {
            return None;
        }
        let (g, Span(start), Count(idx)) = self.tree.seek::<Span, Count>(&Span(offset))?;
        // If `offset` sits on item `idx`'s span start (a real member), that's it;
        // otherwise the next member is this item's own end.
        if offset == start && idx >= 1 {
            Some(offset)
        } else {
            Some(start + g.0)
        }
    }

    /// Every offset in `[lo, hi)`, ascending. O(log n + hits).
    #[must_use]
    pub fn in_range(&self, lo: u32, hi: u32) -> Vec<u32> {
        let mut out = Vec::new();
        // A member at offset 0 has an empty (gap-0) span that `for_each_in_range`
        // prunes; it is the only member that can, so add it up front.
        if lo == 0 && hi > 0 && self.contains(0) {
            out.push(0);
        }
        // A member `m` is the END of its item's span `[prev, m)`, which overlaps
        // `[lo-1, hi)` exactly when `m >= lo` and `prev < hi` — a superset we
        // filter to `[lo, hi)`. (`for_each_in_range` never yields offset 0.)
        self.tree.for_each_in_range(&Span(lo.saturating_sub(1)), &Span(hi), &mut |g: &Gap, &Span(start)| {
            let offset = start + g.0;
            if offset >= lo && offset < hi {
                out.push(offset);
            }
        });
        out
    }

    /// Insert `offset`. Returns false if already present. O(log n).
    pub fn insert(&mut self, offset: u32) -> bool {
        if self.contains(offset) {
            return false;
        }
        if self.is_empty() || offset > self.total() {
            let gap = Gap(offset - self.total()); // total() == 0 when empty
            self.tree = self.tree.append(&SumTree::from_items([gap]));
            return true;
        }
        // `offset` falls strictly inside item `idx`'s span `[start, start+gap)`;
        // split it so `offset` becomes a member between them.
        let (start, gap, idx) = {
            let (g, Span(start), Count(idx)) =
                self.tree.seek::<Span, Count>(&Span(offset)).expect("non-empty");
            (start, g.0, idx)
        };
        let whole_end = start + gap;
        self.tree =
            self.tree.replace(Count(idx)..Count(idx + 1), [Gap(offset - start), Gap(whole_end - offset)]);
        true
    }

    /// Remove `offset`. Returns whether it was present. O(log n).
    pub fn remove(&mut self, offset: u32) -> bool {
        if !self.contains(offset) {
            return false;
        }
        let n = self.len() as u32;
        if offset == self.total() {
            // The last member: drop the last item (nothing follows to merge into).
            self.tree = self.tree.replace(Count(n - 1)..Count(n), std::iter::empty());
            return true;
        }
        // A non-last member `offset[j]` is item `idx-1`'s end and item `idx`'s
        // span start (so `seek` returns `start == offset`, `idx == j+1`). Removing
        // it merges item `j` into item `j+1`.
        let j = {
            let (_, Span(_), Count(idx)) =
                self.tree.seek::<Span, Count>(&Span(offset)).expect("member");
            idx - 1
        };
        let merged = Gap(self.item_gap(j) + self.item_gap(j + 1));
        self.tree = self.tree.replace(Count(j)..Count(j + 2), [merged]);
        true
    }

    /// Remove every offset in `drop` (unsorted ok). O((n + |drop|)·log n) worst.
    pub fn remove_all(&mut self, drop: &[u32]) {
        for &o in drop {
            self.remove(o);
        }
    }

    fn item_gap(&self, index: u32) -> u32 {
        self.tree.item_at(&Count(index)).map_or(0, |(g, _)| g.0)
    }

    /// Shift every offset through a committed patch with `Bias::Right` (a member in
    /// a deleted span maps into it; the caller's reconcile drops it).
    ///
    /// A **single edit** (the common keystroke) takes the windowed path: remap only
    /// the O(edit-window) offsets inside the edit's span and shift the whole suffix
    /// in O(log n) by reanchoring one delta-gap seam — **O(window + log n)**, not
    /// O(members). A multi-edit (multi-caret) patch falls back to the naive
    /// whole-set remap. Both paths produce identical results.
    pub fn apply_patch(&mut self, patch: &Patch) {
        if patch.is_empty() || self.is_empty() {
            return;
        }
        match patch.edits() {
            [edit] => self.apply_single_edit(edit, patch),
            _ => self.apply_patch_naive(patch),
        }
    }

    /// Whole-set remap: map every offset, re-sort (`map_offset(_, Right)` is not
    /// monotonic across a replacement), dedup a deletion's collisions, rebuild. The
    /// reference the windowed path equals, and the multi-edit fallback.
    fn apply_patch_naive(&mut self, patch: &Patch) {
        let queries: Vec<(u32, Bias)> = self.offsets().into_iter().map(|o| (o, Bias::Right)).collect();
        let mut mapped = Vec::new();
        patch.map_many(&queries, &mut mapped);
        mapped.sort_unstable();
        mapped.dedup();
        *self = Self::from_sorted(&mapped);
    }

    /// The windowed single-edit shift. Offsets are points (`Bias::Right`), so unlike
    /// ranged decorations there is no straddler reaching in from the left:
    /// `offset < os` is a fixed point (shared), `offset ∈ [os, oe]` is the remapped
    /// window (re-sorted + deduped — a deletion can collapse several onto one, and
    /// only here, never across a band edge), `offset > oe` shifts uniformly by
    /// `delta` via one reanchored seam.
    fn apply_single_edit(&mut self, edit: &Edit, patch: &Patch) {
        let (os, oe) = (edit.old.start, edit.old.end);
        let delta = (i64::from(edit.new.end) - i64::from(edit.new.start))
            - (i64::from(oe) - i64::from(os));

        let k_lo = self.tree.summary_before(&Span(os)).count; // offset < os
        let k_hi = self.tree.summary_before(&Span(oe.saturating_add(1))).count; // offset ≤ oe
        let (left, rest) = self.tree.split_at(&Count(k_lo));
        let (middle, suffix) = rest.split_at(&Count(k_hi - k_lo));
        let left_span = left.extent::<Span>().0; // absolute offset where `middle` begins

        let mut win: Vec<u32> = decode_offsets(&middle, left_span)
            .into_iter()
            .map(|o| patch.map_offset(o, Bias::Right))
            .collect();
        win.sort_unstable();
        win.dedup();
        let middle_new = SumTree::from_items(encode_gaps(&win, left_span));

        let head = left.append(&middle_new);
        let prev = head.extent::<Span>().0; // last offset before the suffix
        let suffix_new = reanchor(&suffix, k_hi, &self.tree, delta, prev);
        self.tree = head.append(&suffix_new);
    }
}

/// Accumulate a delta-gap subtree's gaps into absolute offsets, starting at `base`
/// (the offset where a split-off subtree begins).
fn decode_offsets(tree: &SumTree<Gap>, base: u32) -> Vec<u32> {
    let mut acc = base;
    tree.items()
        .into_iter()
        .map(|g| {
            acc += g.0;
            acc
        })
        .collect()
}

/// Delta-gap encode ascending `offsets`, the first gap relative to `base`.
fn encode_gaps(offsets: &[u32], base: u32) -> impl Iterator<Item = Gap> + '_ {
    let mut prev = base;
    offsets.iter().map(move |&o| {
        debug_assert!(o >= prev, "offsets must be ascending and ≥ base");
        let g = Gap(o - prev);
        prev = o;
        g
    })
}

/// Re-anchor a split-off `suffix` onto a rebuilt head: its first offset becomes
/// `old_offset + delta`, every later gap unchanged (relative). O(log n). `k` is that
/// first offset's overall index; `orig` is the pre-split tree (read for its old
/// absolute offset); `prev` is the last offset placed before it.
fn reanchor(suffix: &SumTree<Gap>, k: u32, orig: &SumTree<Gap>, delta: i64, prev: u32) -> SumTree<Gap> {
    if suffix.is_empty() {
        return suffix.clone();
    }
    let (g, _c, Span(s)) = orig
        .seek::<Count, Span>(&Count(k))
        .expect("suffix non-empty ⇒ a k-th offset exists");
    let first_old = s + g.0;
    let new_gap = (i64::from(first_old) + delta - i64::from(prev)) as u32;
    suffix.replace(Count(0)..Count(1), std::iter::once(Gap(new_gap)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::{Edit, Patch};

    // The oracle: a plain sorted-unique Vec with the same semantics.
    #[derive(Clone, Default)]
    struct Model(Vec<u32>);
    impl Model {
        fn insert(&mut self, o: u32) -> bool {
            match self.0.binary_search(&o) {
                Ok(_) => false,
                Err(i) => {
                    self.0.insert(i, o);
                    true
                }
            }
        }
        fn remove(&mut self, o: u32) -> bool {
            match self.0.binary_search(&o) {
                Ok(i) => {
                    self.0.remove(i);
                    true
                }
                Err(_) => false,
            }
        }
        fn apply(&mut self, patch: &Patch) {
            let mut m: Vec<u32> = self.0.iter().map(|&o| patch.map_offset(o, Bias::Right)).collect();
            m.sort_unstable();
            m.dedup();
            self.0 = m;
        }
    }

    fn agree(set: &OffsetSet, model: &Model) {
        assert_eq!(set.offsets(), model.0, "offsets");
        assert_eq!(set.len(), model.0.len(), "len");
        let max = model.0.last().copied().unwrap_or(0) + 5;
        for o in 0..=max {
            assert_eq!(set.contains(o), model.0.binary_search(&o).is_ok(), "contains {o}");
            let want = model.0.iter().copied().find(|&x| x >= o);
            assert_eq!(set.first_at_or_after(o), want, "first_at_or_after {o}");
        }
        for lo in 0..max {
            for hi in lo..max {
                let want: Vec<u32> = model.0.iter().copied().filter(|&x| x >= lo && x < hi).collect();
                assert_eq!(set.in_range(lo, hi), want, "in_range {lo}..{hi}");
            }
        }
    }

    #[test]
    fn queries_match_the_model() {
        let src = [3u32, 7, 10, 11, 25, 40];
        let set = OffsetSet::from_sorted(&src);
        let model = Model(src.to_vec());
        agree(&set, &model);
    }

    #[test]
    fn insert_remove_match_the_model_under_random_ops() {
        let mut set = OffsetSet::new();
        let mut model = Model::default();
        let mut state = 0xC0FFEEu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..2000 {
            let o = next() % 60;
            if next() & 1 == 0 {
                assert_eq!(set.insert(o), model.insert(o), "insert {o}");
            } else {
                assert_eq!(set.remove(o), model.remove(o), "remove {o}");
            }
            assert_eq!(set.offsets(), model.0, "after op on {o}");
        }
        agree(&set, &model);
    }

    #[test]
    fn apply_patch_matches_the_model() {
        let mut state = 0x5EEDu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for _ in 0..3000 {
            // A random set and a random 1-3 edit patch (disjoint, ascending). Both
            // the single-edit windowed path and the multi-edit naive fallback ride
            // this; edits at 0 exercise the front-insert whole-suffix shift.
            let mut offs: Vec<u32> = (0..10).map(|_| next() % 50).collect();
            offs.sort_unstable();
            offs.dedup();
            let mut set = OffsetSet::from_sorted(&offs);
            let mut model = Model(offs);

            let mut patch = Patch::new();
            let mut cursor = 0u32;
            let mut new_cursor = 0u32;
            for _ in 0..(1 + next() % 3) {
                let gap = next() % 6;
                let old_len = next() % 5;
                let new_len = next() % 5;
                let os = cursor + gap;
                let oe = os + old_len;
                let ns = new_cursor + gap;
                let ne = ns + new_len;
                patch.push(Edit { old: os..oe, new: ns..ne });
                cursor = oe;
                new_cursor = ne;
            }
            set.apply_patch(&patch);
            model.apply(&patch);
            agree(&set, &model);
        }
    }

    #[test]
    fn windowed_apply_patch_is_sublinear_in_set_size() {
        use crate::sum_tree::NODE_ALLOCS;
        // A single-edit shift on an N-offset set must allocate O(window + log N) tree
        // nodes, not O(N). A front insert shifts the whole suffix, so it is the
        // O(N)-with-absolute-offsets / O(log N)-with-delta-gap case — the tightest
        // check that the suffix reanchors at one seam rather than rebuilding.
        let build = |n: u32| OffsetSet::from_sorted(&(0..n).map(|i| i * 10).collect::<Vec<_>>());
        let allocs_for = |n: u32| -> f64 {
            let mut s = build(n);
            let patch = Patch::single(Edit { old: 5..5, new: 5..6 });
            NODE_ALLOCS.with(|c| c.set(0));
            s.apply_patch(&patch);
            NODE_ALLOCS.with(std::cell::Cell::get) as f64
        };
        let (small, big) = (allocs_for(1000), allocs_for(4000));
        eprintln!("[offset_set] apply_patch node allocs {small} -> {big}  ({:.2}x)", big / small);
        assert!(
            big <= small * 2.0,
            "OffsetSet::apply_patch went superlinear ({small} -> {big} nodes): it rebuilt \
             the whole set instead of reanchoring the suffix at one seam"
        );
    }
}
