//! `SumTree<T>` — the one augmented balanced tree the whole editor rides.
//!
//! A generic, `Arc`-backed B-tree whose nodes cache a monoidal **summary** of
//! their subtree; a **cursor** seeks by any **dimension** (a seekable projection
//! of that summary — byte offset, `Point`, item count, bracket depth) in
//! O(log n). Text, brackets, folds, and decorations all become summaries on this
//! one structure, so "what encloses / stabs / is in this window" is an O(log)
//! descent by construction — there is no flat `Vec` to scan.
//!
//! `Arc` children give O(1) structural-sharing snapshots (clone shares nodes) and
//! copy-on-write edits: an edit rebuilds only the touched leaves and the spine
//! above them, leaving every other subtree shared with the pre-edit tree.
//!
//! The full API — build, summary, dimensional seek, split, append, batched
//! multi-edit — is oracle-tested against a plain `Vec` (see the module tests).

// A complete generic B-tree API; not every method is exercised in every build
// configuration, so unused-code warnings are silenced module-wide.
#![allow(dead_code)]

use std::ops::ControlFlow;
use std::sync::Arc;

/// B-tree fan-out: every node except a lone root holds `B..=2*B` children/items.
const B: usize = 6;
const MAX: usize = 2 * B;

// Global node-allocation counter — a proxy for copy-on-write cost, which a
// semantic operation count alone cannot observe. Lets a scale test assert a hot
// path allocates O(items), not O(items · log) or O(items²). Debug/test only.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static NODE_ALLOCS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Count one `Node` allocation (a build, a split, an append, an edit's spine clone),
/// so a scale test can assert a hot path allocates O(items), not O(items · log) or
/// O(items²). A no-op in release.
#[inline]
fn count_node_alloc() {
    #[cfg(any(test, debug_assertions))]
    NODE_ALLOCS.with(|c| c.set(c.get() + 1));
}

/// A leaf/child payload that reports its [`Summary`].
pub trait Item: Clone + std::fmt::Debug {
    type Summary: Summary;
    fn summary(&self) -> Self::Summary;
}

/// An associative monoid aggregated up every node. Identity is [`Default`];
/// `add_summary` appends `other` on the right and must be associative.
pub trait Summary: Clone + Default + std::fmt::Debug {
    fn add_summary(&mut self, other: &Self);
}

/// A seekable projection of a [`Summary`] — a running total a cursor compares to a
/// target (byte offset, `Point`, item count…). `Ord` so seeking can compare; the
/// zero value is [`Default`].
pub trait Dimension<S: Summary>: Clone + Default + Ord {
    fn add_summary(&mut self, summary: &S);

    /// Fold a whole summary into a fresh dimension value.
    fn from_summary(summary: &S) -> Self {
        let mut d = Self::default();
        d.add_summary(summary);
        d
    }
}

/// Every summary trivially projects to itself — lets `extent::<S>()` read the root
/// total and callers seek by a raw summary when convenient.
impl<S: Summary + Ord> Dimension<S> for S {
    fn add_summary(&mut self, summary: &S) {
        Summary::add_summary(self, summary);
    }
}

#[derive(Debug)]
enum Node<T: Item> {
    Internal {
        height: u8,
        summary: T::Summary,
        child_summaries: Vec<T::Summary>,
        children: Vec<SumTree<T>>,
    },
    Leaf {
        summary: T::Summary,
        items: Vec<T>,
        item_summaries: Vec<T::Summary>,
    },
}

/// The tree. Cloning is O(1) (an `Arc` bump) and shares structure with the clone.
#[derive(Debug)]
pub struct SumTree<T: Item>(Arc<Node<T>>);

impl<T: Item> Clone for SumTree<T> {
    fn clone(&self) -> Self {
        SumTree(Arc::clone(&self.0))
    }
}

impl<T: Item> Default for SumTree<T> {
    fn default() -> Self {
        Self::new()
    }
}

fn sum_of<T: Item>(summaries: &[T::Summary]) -> T::Summary {
    let mut acc = T::Summary::default();
    for s in summaries {
        acc.add_summary(s);
    }
    acc
}

/// The bucket `b` (`0..bounds.len()-1`) whose half-open span `[bounds[b],
/// bounds[b+1])` contains `x`, or `None` when `x < bounds[0]` or `x >=
/// bounds[last]` (below the first / at-or-above the last boundary — no bucket).
/// `bounds` must be ascending. O(log buckets).
fn bucket_of<D: Ord>(bounds: &[D], x: &D) -> Option<usize> {
    let p = bounds.partition_point(|bnd| bnd <= x); // count of bounds ≤ x
    if p == 0 || p >= bounds.len() {
        None
    } else {
        Some(p - 1)
    }
}

/// The SIGNED bucket position of `x`: `-1` below `bounds[0]`, `b` inside bucket
/// `b` (`0..bounds.len()-1`), or `bounds.len()-1` at/above the last boundary.
/// Unlike [`bucket_of`], this distinguishes "below all" from "above all" — a
/// subtree spanning `pre` below the range and `post` above it must NOT be folded
/// as one bucket. `bounds` must be ascending. O(log buckets).
fn bucket_pos<D: Ord>(bounds: &[D], x: &D) -> i64 {
    bounds.partition_point(|bnd| bnd <= x) as i64 - 1
}

impl<T: Item> SumTree<T> {
    /// The empty tree (a leaf with no items).
    #[must_use]
    pub fn new() -> Self {
        Self::leaf(Vec::new())
    }

    /// Build a balanced tree from items — leaves of `MAX`, then internal levels
    /// bottom-up, so every node has the same-height children and height is
    /// O(log n). (A from-scratch build may leave the final node of a level short of
    /// `B`; edits rebalance to the strict `B..=MAX` invariant.)
    #[must_use]
    pub fn from_items(items: impl IntoIterator<Item = T>) -> Self {
        let items: Vec<T> = items.into_iter().collect();
        if items.is_empty() {
            return Self::new();
        }
        let mut level: Vec<SumTree<T>> =
            items.chunks(MAX).map(|chunk| Self::leaf(chunk.to_vec())).collect();

        let mut height = 1u8;
        while level.len() > 1 {
            level = level.chunks(MAX).map(|chunk| Self::internal(height, chunk.to_vec())).collect();
            height += 1;
        }
        level.pop().expect("non-empty items ⇒ ≥1 node")
    }

    /// This subtree's summary (O(1)).
    #[must_use]
    pub fn summary(&self) -> &T::Summary {
        match &*self.0 {
            Node::Internal { summary, .. } | Node::Leaf { summary, .. } => summary,
        }
    }

    /// The total extent in dimension `D` (the root summary folded into `D`).
    #[must_use]
    pub fn extent<D: Dimension<T::Summary>>(&self) -> D {
        D::from_summary(self.summary())
    }

    /// Whether the tree holds no items.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        match &*self.0 {
            Node::Leaf { items, .. } => items.is_empty(),
            Node::Internal { .. } => false,
        }
    }

    fn height(&self) -> u8 {
        match &*self.0 {
            Node::Internal { height, .. } => *height,
            Node::Leaf { .. } => 0,
        }
    }

    fn as_children(&self) -> &[SumTree<T>] {
        match &*self.0 {
            Node::Internal { children, .. } => children,
            Node::Leaf { .. } => &[],
        }
    }

    fn as_items(&self) -> &[T] {
        match &*self.0 {
            Node::Leaf { items, .. } => items,
            Node::Internal { .. } => &[],
        }
    }

    fn leaf(items: Vec<T>) -> SumTree<T> {
        count_node_alloc();
        let item_summaries: Vec<T::Summary> = items.iter().map(Item::summary).collect();
        SumTree(Arc::new(Node::Leaf { summary: sum_of::<T>(&item_summaries), items, item_summaries }))
    }

    fn internal(height: u8, children: Vec<SumTree<T>>) -> SumTree<T> {
        count_node_alloc();
        debug_assert!(!children.is_empty() && children.len() <= MAX);
        let child_summaries: Vec<T::Summary> = children.iter().map(|c| c.summary().clone()).collect();
        SumTree(Arc::new(Node::Internal {
            height,
            summary: sum_of::<T>(&child_summaries),
            child_summaries,
            children,
        }))
    }

    /// Pack `items` (up to `2*MAX`) into one leaf, or two even leaves if it would
    /// overflow. Each result holds `<= MAX` items (and `>= B` when the input did).
    fn from_items_1or2(mut items: Vec<T>) -> Vec<SumTree<T>> {
        if items.len() <= MAX {
            vec![Self::leaf(items)]
        } else {
            let right = items.split_off(items.len() / 2);
            vec![Self::leaf(items), Self::leaf(right)]
        }
    }

    /// The internal-node analogue of [`Self::from_items_1or2`].
    fn from_children_1or2(height: u8, mut children: Vec<SumTree<T>>) -> Vec<SumTree<T>> {
        if children.len() <= MAX {
            vec![Self::internal(height, children)]
        } else {
            let right = children.split_off(children.len() / 2);
            vec![Self::internal(height, children), Self::internal(height, right)]
        }
    }

    /// Wrap a slice of same-height children (all height `height - 1`) into a tree of
    /// `height`, or the empty tree — O(children), no per-child append. The split
    /// helper: it rebuilds a partition's untouched children in one shot, so a
    /// `split_at` stays O(log n); folding `append` across them one at a time
    /// would cost O(log² n).
    fn wrap_children(height: u8, children: &[SumTree<T>]) -> SumTree<T> {
        if children.is_empty() {
            SumTree::new()
        } else {
            Self::finalize(Self::from_children_1or2(height, children.to_vec()))
        }
    }

    /// Concatenate two trees, O(height). The shorter is grafted onto the taller's
    /// near edge and the touched spine is resplit to the `B..=MAX` invariant; a
    /// two-node top makes the result one level taller.
    #[must_use]
    pub fn append(&self, other: &SumTree<T>) -> SumTree<T> {
        if self.is_empty() {
            return other.clone();
        }
        if other.is_empty() {
            return self.clone();
        }
        let parts = if self.height() >= other.height() {
            self.concat_right(other)
        } else {
            other.concat_left(self)
        };
        Self::finalize(parts)
    }

    fn finalize(mut parts: Vec<SumTree<T>>) -> SumTree<T> {
        debug_assert!(!parts.is_empty());
        if parts.len() == 1 {
            parts.pop().expect("len checked")
        } else {
            let height = parts[0].height() + 1;
            Self::internal(height, parts)
        }
    }

    /// Append `other` to the right of `self`. Precondition: both non-empty and
    /// `self.height() >= other.height()`. Returns 1–2 trees of `self`'s height.
    fn concat_right(&self, other: &SumTree<T>) -> Vec<SumTree<T>> {
        let h = self.height();
        if h == other.height() {
            if h == 0 {
                let mut items = self.as_items().to_vec();
                items.extend_from_slice(other.as_items());
                Self::from_items_1or2(items)
            } else {
                let mut children = self.as_children().to_vec();
                children.extend_from_slice(other.as_children());
                Self::from_children_1or2(h, children)
            }
        } else {
            let mut children = self.as_children().to_vec();
            let last = children.pop().expect("internal node has children");
            let mut grafted = last.concat_right(other);
            children.append(&mut grafted);
            Self::from_children_1or2(h, children)
        }
    }

    /// Prepend `other` to the front of `self`. Precondition: both non-empty and
    /// `self.height() >= other.height()`. Returns 1–2 trees of `self`'s height.
    fn concat_left(&self, other: &SumTree<T>) -> Vec<SumTree<T>> {
        let h = self.height();
        if h == other.height() {
            if h == 0 {
                let mut items = other.as_items().to_vec();
                items.extend_from_slice(self.as_items());
                Self::from_items_1or2(items)
            } else {
                let mut children = other.as_children().to_vec();
                children.extend_from_slice(self.as_children());
                Self::from_children_1or2(h, children)
            }
        } else {
            let mut children = self.as_children().to_vec();
            let first = children.remove(0);
            let mut grafted = first.concat_left(other);
            grafted.extend(children);
            Self::from_children_1or2(h, grafted)
        }
    }

    /// Split into `(before, from)` at dimension `at`: the item containing `at`
    /// (and everything after) goes right, everything strictly before goes left. An
    /// `at` on an item boundary splits cleanly between items. O(log n).
    #[must_use]
    pub fn split_at<D: Dimension<T::Summary>>(&self, at: &D) -> (SumTree<T>, SumTree<T>) {
        self.split_abs(&D::default(), at)
    }

    /// `start` is the absolute dimension at this subtree's first item.
    fn split_abs<D: Dimension<T::Summary>>(&self, start: &D, at: &D) -> (SumTree<T>, SumTree<T>) {
        if at <= start {
            return (SumTree::new(), self.clone());
        }
        let mut whole_end = start.clone();
        whole_end.add_summary(self.summary());
        if at >= &whole_end {
            return (self.clone(), SumTree::new());
        }
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut acc = start.clone();
                let mut k = 0;
                while k < items.len() {
                    let mut end = acc.clone();
                    end.add_summary(&item_summaries[k]);
                    if &end > at {
                        break; // item k straddles `at` (or starts exactly at it) → goes right
                    }
                    acc = end;
                    k += 1;
                }
                (Self::leaf(items[..k].to_vec()), Self::leaf(items[k..].to_vec()))
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut acc = start.clone();
                for (i, child) in children.iter().enumerate() {
                    let mut end = acc.clone();
                    end.add_summary(&child_summaries[i]);
                    if &end > at {
                        let (l, r) = child.split_abs(&acc, at);
                        // The untouched children on each side are wrapped in one shot,
                        // then the split child's half grafted on — O(log), not the
                        // O(log²) per-child append fold.
                        let h = self.height();
                        let left = Self::wrap_children(h, &children[..i]).append(&l);
                        let right = r.append(&Self::wrap_children(h, &children[i + 1..]));
                        return (left, right);
                    }
                    acc = end;
                }
                unreachable!("`at < whole_end` guarantees a straddling child")
            }
        }
    }

    /// Replace the items spanning `range` (in dimension `D`) with `new`, O(log n +
    /// |new|). The seam is `left(before range.start) ++ new ++ right(from
    /// range.end)`; endpoints on item boundaries cut cleanly.
    #[must_use]
    pub fn replace<D: Dimension<T::Summary>>(
        &self,
        range: std::ops::Range<D>,
        new: impl IntoIterator<Item = T>,
    ) -> SumTree<T> {
        let left = self.split_at(&range.start).0;
        let right = self.split_at(&range.end).1;
        left.append(&SumTree::from_items(new)).append(&right)
    }

    /// Apply MANY disjoint edits in ONE recursive pass, rebuilding only the edited
    /// leaves and the spine above them ONCE while sharing every untouched subtree
    /// (an O(1) `Arc` clone) — the batched twin of N sequential [`Self::replace`]s,
    /// which each re-clone the whole root spine (the document-scale multi-caret
    /// storm). `edits` are sorted by `start` and disjoint in dimension `D`;
    /// `rebuild_leaf(items, leaf_start, edits_in_leaf)` produces a leaf's new items
    /// from its old ones and the edits falling inside it (the caller owns how a
    /// replacement `R` splices into items — e.g. the rope reconstructs chunk text).
    ///
    /// Returns `None` if any edit STRADDLES a child boundary (a large/multi-leaf
    /// replacement); the caller falls back to sequential [`Self::replace`]. Dense
    /// tiny edits (multi-caret typing) never straddle, so they always take this
    /// O(edited leaves + spine) path. For a fully covered spine that is O(edits);
    /// each rebuilt node is built once, not once per edit.
    #[must_use]
    pub fn edit_many<D, R, F>(&self, edits: &[(std::ops::Range<D>, R)], rebuild_leaf: &F) -> Option<SumTree<T>>
    where
        D: Dimension<T::Summary>,
        F: Fn(&[T], &D, &[(std::ops::Range<D>, R)]) -> Vec<T>,
    {
        self.edit_rec(&D::default(), edits, rebuild_leaf)
    }

    fn edit_rec<D, R, F>(
        &self,
        base: &D,
        edits: &[(std::ops::Range<D>, R)],
        rebuild_leaf: &F,
    ) -> Option<SumTree<T>>
    where
        D: Dimension<T::Summary>,
        F: Fn(&[T], &D, &[(std::ops::Range<D>, R)]) -> Vec<T>,
    {
        if edits.is_empty() {
            return Some(self.clone()); // untouched subtree — shared, O(1)
        }
        match &*self.0 {
            Node::Leaf { items, .. } => Some(SumTree::from_items(rebuild_leaf(items, base, edits))),
            Node::Internal { children, child_summaries, .. } => {
                let mut parts: Vec<SumTree<T>> = Vec::with_capacity(children.len());
                let mut cstart = base.clone();
                let mut ei = 0usize;
                let last = children.len() - 1;
                for (i, (child, csum)) in children.iter().zip(child_summaries).enumerate() {
                    let mut cend = cstart.clone();
                    cend.add_summary(csum);
                    // The edits whose start falls in this child — a contiguous run
                    // (edits are sorted). The last child also takes a start == cend
                    // (an append at the node's very end).
                    let lo = ei;
                    while ei < edits.len()
                        && (edits[ei].0.start < cend || (i == last && edits[ei].0.start <= cend))
                    {
                        // A replacement reaching past this child straddles the boundary
                        // — bail to the sequential path (rare: a multi-leaf edit).
                        if edits[ei].0.end > cend {
                            return None;
                        }
                        ei += 1;
                    }
                    parts.push(child.edit_rec(&cstart, &edits[lo..ei], rebuild_leaf)?);
                    cstart = cend;
                }
                Some(Self::concat_all(parts))
            }
        }
    }

    /// Balanced concatenation of a subtree list (divide-and-conquer `append`, so the
    /// result is O(log)-tall, not a right-leaning O(k) chain). The edit rebuild's
    /// child-combine; `append` tolerates the mixed heights a leaf's growth produces.
    fn concat_all(mut trees: Vec<SumTree<T>>) -> SumTree<T> {
        match trees.len() {
            0 => SumTree::new(),
            1 => trees.pop().expect("len 1"),
            n => {
                let right = trees.split_off(n / 2);
                Self::concat_all(trees).append(&Self::concat_all(right))
            }
        }
    }

    /// The combined summary of every item strictly before dimension `target` — the
    /// prefix fold (e.g. the bracket shape of everything left of a caret). Items
    /// are added left-to-right through `Summary::add_summary`, so an offset-bearing
    /// summary resolves to absolute coordinates. O(log n).
    #[must_use]
    pub fn summary_before<D: Dimension<T::Summary>>(&self, target: &D) -> T::Summary {
        let mut acc = T::Summary::default();
        self.accumulate_before(&D::default(), &mut acc, target);
        acc
    }

    fn accumulate_before<D: Dimension<T::Summary>>(
        &self,
        start: &D,
        acc: &mut T::Summary,
        target: &D,
    ) {
        let mut whole_end = start.clone();
        whole_end.add_summary(self.summary());
        if &whole_end < target {
            acc.add_summary(self.summary()); // this subtree is entirely before target
            return;
        }
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut d = start.clone();
                for (item, s) in items.iter().zip(item_summaries) {
                    let mut end = d.clone();
                    end.add_summary(s);
                    if &end < target {
                        acc.add_summary(&item.summary());
                        d = end;
                    } else {
                        break;
                    }
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut d = start.clone();
                for (child, s) in children.iter().zip(child_summaries) {
                    let mut end = d.clone();
                    end.add_summary(s);
                    if &end < target {
                        acc.add_summary(s);
                        d = end;
                    } else {
                        child.accumulate_before(&d, acc, target);
                        break;
                    }
                }
            }
        }
    }

    /// The suffix twin of [`Self::summary_before`]: visit, in document order, the
    /// **canonical decomposition** of every item at or after dimension `target`
    /// (its accumulated `D`-end `>= target`) into O(log n) maximal subtree
    /// summaries — a whole subtree entirely in the suffix is handed over as one
    /// cached summary (never descended), only the single straddling spine is split
    /// down to items. Each visit gets the subtree's summary and the `D` accumulated
    /// at its start (so an offset-bearing summary resolves to absolute:
    /// `start + entry.off`). The visitor returns [`ControlFlow`]; the walk stops at
    /// the first `Break(r)` and yields `Some(r)`, else `None`.
    ///
    /// This is the primitive a stateful right-scan rides — e.g. an opener's partner
    /// is the first suffix closer that reaches the opener's stack level, found by
    /// folding these summaries into a running stack and breaking on the match,
    /// O(log n + unmatched brackets crossed) with balanced subtrees folded whole.
    #[must_use]
    pub fn try_suffix_summaries<D, R, F>(&self, target: &D, f: &mut F) -> Option<R>
    where
        D: Dimension<T::Summary>,
        F: FnMut(&T::Summary, &D) -> ControlFlow<R>,
    {
        match self.try_suffix_from(&D::default(), target, f) {
            ControlFlow::Break(r) => Some(r),
            ControlFlow::Continue(()) => None,
        }
    }

    fn try_suffix_from<D, R, F>(&self, start: &D, target: &D, f: &mut F) -> ControlFlow<R>
    where
        D: Dimension<T::Summary>,
        F: FnMut(&T::Summary, &D) -> ControlFlow<R>,
    {
        // `ControlFlow`'s `Try` impl is unstable, so early-exit is propagated by
        // hand: a `Break` from any visit/recursion returns immediately.
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut d = start.clone();
                for (item, s) in items.iter().zip(item_summaries) {
                    let mut end = d.clone();
                    end.add_summary(s);
                    // The mirror of `accumulate_before`'s `end < target` prefix cut:
                    // an item whose accumulated end reaches `target` is in the suffix.
                    if &end >= target {
                        if let ControlFlow::Break(r) = f(&item.summary(), &d) {
                            return ControlFlow::Break(r);
                        }
                    }
                    d = end;
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut d = start.clone();
                for (child, s) in children.iter().zip(child_summaries) {
                    let mut end = d.clone();
                    end.add_summary(s);
                    if &end < target {
                        // Entirely before the suffix — skip whole.
                    } else if &d >= target {
                        // Entirely in the suffix — hand over the cached summary.
                        if let ControlFlow::Break(r) = f(s, &d) {
                            return ControlFlow::Break(r);
                        }
                    } else {
                        // The lone straddling child — descend to split it at `target`.
                        if let ControlFlow::Break(r) = child.try_suffix_from(&d, target, f) {
                            return ControlFlow::Break(r);
                        }
                    }
                    d = end;
                }
            }
        }
        ControlFlow::Continue(())
    }

    /// The dimension `D` accumulated over every item strictly before dimension
    /// `target` (measured in `A`) — the alloc-free twin of [`Self::summary_before`]
    /// when only a light `Copy` projection is needed (e.g. the bracket COUNT before
    /// a byte offset), never the whole Vec-bearing summary. O(log n), zero heap.
    #[must_use]
    pub fn measure_before<A: Dimension<T::Summary>, D: Dimension<T::Summary>>(&self, target: &A) -> D {
        let mut d = D::default();
        self.measure_before_rec(&A::default(), target, &mut d);
        d
    }

    fn measure_before_rec<A: Dimension<T::Summary>, D: Dimension<T::Summary>>(
        &self,
        start: &A,
        target: &A,
        acc: &mut D,
    ) {
        let mut whole_end = start.clone();
        whole_end.add_summary(self.summary());
        if &whole_end < target {
            acc.add_summary(self.summary()); // this subtree is entirely before target
            return;
        }
        match &*self.0 {
            Node::Leaf { item_summaries, .. } => {
                let mut a = start.clone();
                for s in item_summaries {
                    let mut end = a.clone();
                    end.add_summary(s);
                    if &end < target {
                        acc.add_summary(s);
                        a = end;
                    } else {
                        break;
                    }
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut a = start.clone();
                for (child, s) in children.iter().zip(child_summaries) {
                    let mut end = a.clone();
                    end.add_summary(s);
                    if &end < target {
                        acc.add_summary(s);
                        a = end;
                    } else {
                        child.measure_before_rec(&a, target, acc);
                        break;
                    }
                }
            }
        }
    }

    /// Seek by dimension `A` to the item containing `target`, reporting that item
    /// and BOTH dimensions accumulated at its start. The tool for cross-dimension
    /// conversions — e.g. seek by byte offset (`A`) and read the `Point` (`B`) at
    /// the item's start, then finish inside the item. O(log n).
    #[must_use]
    pub fn seek<A: Dimension<T::Summary>, B: Dimension<T::Summary>>(
        &self,
        target: &A,
    ) -> Option<(&T, A, B)> {
        if self.is_empty() {
            return None;
        }
        let mut node = &self.0;
        let (mut a, mut b) = (A::default(), B::default());
        loop {
            match &**node {
                Node::Internal { children, child_summaries, .. } => {
                    let mut i = 0;
                    loop {
                        let mut end = a.clone();
                        end.add_summary(&child_summaries[i]);
                        if &end > target || i + 1 == children.len() {
                            node = &children[i].0;
                            break;
                        }
                        a = end;
                        b.add_summary(&child_summaries[i]);
                        i += 1;
                    }
                }
                Node::Leaf { items, item_summaries, .. } => {
                    let mut i = 0;
                    loop {
                        let mut end = a.clone();
                        end.add_summary(&item_summaries[i]);
                        if &end > target || i + 1 == items.len() {
                            return Some((&items[i], a, b));
                        }
                        a = end;
                        b.add_summary(&item_summaries[i]);
                        i += 1;
                    }
                }
            }
        }
    }

    /// Split precisely at dimension `at`, splitting the straddling item with
    /// `split_item(item, item_start, at)` when `at` falls strictly inside it (an
    /// `at` on an item boundary needs no split). The left half of the split item
    /// ends the left tree; the right half begins the right tree. This is how a
    /// text rope cuts mid-chunk. O(log n).
    #[must_use]
    pub fn split_with<D, F>(&self, at: &D, split_item: &mut F) -> (SumTree<T>, SumTree<T>)
    where
        D: Dimension<T::Summary>,
        F: FnMut(&T, &D, &D) -> (T, T),
    {
        self.split_abs_with(&D::default(), at, split_item)
    }

    fn split_abs_with<D, F>(&self, start: &D, at: &D, split_item: &mut F) -> (SumTree<T>, SumTree<T>)
    where
        D: Dimension<T::Summary>,
        F: FnMut(&T, &D, &D) -> (T, T),
    {
        if at <= start {
            return (SumTree::new(), self.clone());
        }
        let mut whole_end = start.clone();
        whole_end.add_summary(self.summary());
        if at >= &whole_end {
            return (self.clone(), SumTree::new());
        }
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut acc = start.clone();
                let mut k = 0;
                while k < items.len() {
                    let mut end = acc.clone();
                    end.add_summary(&item_summaries[k]);
                    if &end > at {
                        break;
                    }
                    acc = end;
                    k += 1;
                }
                if &acc == at {
                    // `at` lands exactly on item k's start — a clean cut, no split.
                    (Self::leaf(items[..k].to_vec()), Self::leaf(items[k..].to_vec()))
                } else {
                    // `at` is strictly inside item k — split it.
                    let (l, r) = split_item(&items[k], &acc, at);
                    let mut left = items[..k].to_vec();
                    left.push(l);
                    let mut right = vec![r];
                    right.extend_from_slice(&items[k + 1..]);
                    (Self::finalize(Self::from_items_1or2(left)), Self::finalize(Self::from_items_1or2(right)))
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut acc = start.clone();
                for (i, child) in children.iter().enumerate() {
                    let mut end = acc.clone();
                    end.add_summary(&child_summaries[i]);
                    if &end > at {
                        let (l, r) = child.split_abs_with(&acc, at, split_item);
                        let h = self.height();
                        let left = Self::wrap_children(h, &children[..i]).append(&l);
                        let right = r.append(&Self::wrap_children(h, &children[i + 1..]));
                        return (left, right);
                    }
                    acc = end;
                }
                unreachable!("`at < whole_end` guarantees a straddling child")
            }
        }
    }

    /// The item whose `D`-span contains `target`, and the accumulated `D` at that
    /// item's start — the first item whose end-`D` exceeds `target` (so an exact
    /// boundary biases right, to the following item). Clamps to the last item when
    /// `target` is at or past the total. `None` only on an empty tree. O(log n).
    #[must_use]
    pub fn item_at<D: Dimension<T::Summary>>(&self, target: &D) -> Option<(&T, D)> {
        if self.is_empty() {
            return None;
        }
        let mut node = &self.0;
        let mut start = D::default();
        loop {
            match &**node {
                Node::Internal { children, child_summaries, .. } => {
                    let mut i = 0;
                    loop {
                        let mut end = start.clone();
                        end.add_summary(&child_summaries[i]);
                        if &end > target || i + 1 == children.len() {
                            node = &children[i].0;
                            break;
                        }
                        start = end;
                        i += 1;
                    }
                }
                Node::Leaf { items, item_summaries, .. } => {
                    let mut i = 0;
                    loop {
                        let mut end = start.clone();
                        end.add_summary(&item_summaries[i]);
                        if &end > target || i + 1 == items.len() {
                            return Some((&items[i], start));
                        }
                        start = end;
                        i += 1;
                    }
                }
            }
        }
    }

    /// Items in order — the oracle handle and the reduce-to-Vec escape hatch.
    #[must_use]
    pub fn items(&self) -> Vec<T> {
        let mut out = Vec::new();
        self.collect_items(&mut out);
        out
    }

    /// Visit every item overlapping the dimension range `[from, to)`, in order,
    /// with its start dimension — subtrees entirely outside the range are pruned,
    /// so this is O(log n + items in range). The windowed read behind a text
    /// slice.
    pub fn for_each_in_range<D: Dimension<T::Summary>, F: FnMut(&T, &D)>(
        &self,
        from: &D,
        to: &D,
        f: &mut F,
    ) {
        let start = D::default();
        self.visit_range(&start, from, to, f);
    }

    /// Visit items whose subtree `descend` accepts, in order, each with the
    /// dimension `D` accumulated before it. `descend(start, summary)` decides
    /// whether to enter a subtree given the `D` before it and its summary — the
    /// hook for pruning by a summary field the seek dimension can't express (an
    /// interval tree's max-end: enter iff `start < hi && start + max_end > lo`).
    /// O(log n + visited).
    pub fn filter_visit<D: Dimension<T::Summary>, F: Fn(&D, &T::Summary) -> bool, G: FnMut(&T, &D)>(
        &self,
        descend: &F,
        visit: &mut G,
    ) {
        self.filter_visit_from(&D::default(), descend, visit);
    }

    fn filter_visit_from<D: Dimension<T::Summary>, F: Fn(&D, &T::Summary) -> bool, G: FnMut(&T, &D)>(
        &self,
        start: &D,
        descend: &F,
        visit: &mut G,
    ) {
        if !descend(start, self.summary()) {
            return;
        }
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut acc = start.clone();
                for (item, s) in items.iter().zip(item_summaries) {
                    visit(item, &acc);
                    acc.add_summary(s);
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut acc = start.clone();
                for (child, s) in children.iter().zip(child_summaries) {
                    child.filter_visit_from(&acc, descend, visit);
                    acc.add_summary(s);
                }
            }
        }
    }

    /// Reduce items into `bounds.len()-1` buckets partitioned by their position in
    /// dimension `D`: bucket `b` collects every item whose `D` position lands in
    /// `[bounds[b], bounds[b+1])`. A subtree lying wholly within one bucket folds
    /// through its cached summary in O(1); only bucket-straddling spine is
    /// descended, so this is O(buckets + log n + straddlers), never O(items) — the
    /// scrollbar-overview primitive that must not scan every decoration per frame.
    ///
    /// `bounds` must be ascending. `fold(&mut acc, summary)` must agree whether it
    /// is handed one item's summary or a whole subtree's — i.e. it reads a summary
    /// field that composes the same way under [`Summary::add_summary`] (a max, a
    /// sum). Items whose `D` position is `< bounds[0]` or `>= bounds[last]` fall in
    /// no bucket and are skipped.
    #[must_use]
    pub fn bucketed_reduce<D, A, F>(&self, bounds: &[D], init: A, fold: F) -> Vec<A>
    where
        D: Dimension<T::Summary>,
        A: Clone,
        F: Fn(&mut A, &T::Summary),
    {
        let n = bounds.len().saturating_sub(1);
        let mut out = vec![init; n];
        if n > 0 && !self.is_empty() {
            self.bucketed_reduce_from(&D::default(), bounds, &mut out, &fold);
        }
        out
    }

    fn bucketed_reduce_from<D, A, F>(&self, before: &D, bounds: &[D], out: &mut [A], fold: &F)
    where
        D: Dimension<T::Summary>,
        F: Fn(&mut A, &T::Summary),
    {
        // `before` is the D accumulated before this subtree — a lower bound on the
        // first item's D (items sit at or after it); `after` is the last item's D.
        // If both land in the SAME signed position, so does every item between
        // them — fold whole when that shared position is a real bucket, skip when
        // it is the below-all or above-all zone. (A plain `bucket_of` would fold a
        // subtree spanning from below `bounds[0]` to above `bounds[last]`, since
        // both ends map to `None` — the below/above distinction is load-bearing.)
        let mut after = before.clone();
        after.add_summary(self.summary());
        let nb = out.len() as i64;
        let (pb, pa) = (bucket_pos(bounds, before), bucket_pos(bounds, &after));
        if pb == pa {
            if pb >= 0 && pb < nb {
                fold(&mut out[pb as usize], self.summary());
            }
            return; // wholly in one bucket, or wholly below/above the range
        }
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut acc = before.clone();
                for (item, s) in items.iter().zip(item_summaries) {
                    acc.add_summary(s); // acc is now THIS item's D position
                    if let Some(b) = bucket_of(bounds, &acc) {
                        fold(&mut out[b], &item.summary());
                    }
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut acc = before.clone();
                for (child, s) in children.iter().zip(child_summaries) {
                    child.bucketed_reduce_from(&acc, bounds, out, fold);
                    acc.add_summary(s);
                }
            }
        }
    }

    /// Borrowed references to every item in a subtree the `keep` predicate accepts
    /// (checked on each subtree's summary), in order — the interval-overlap window
    /// read that must borrow (e.g. an absolute-offset item store whose summary
    /// carries min-start/max-end). Subtrees whose summary is rejected are pruned;
    /// surviving leaves hand back all their items (the caller does the exact
    /// per-item test). O(log n + collected).
    #[must_use]
    pub fn filter_refs<F: Fn(&T::Summary) -> bool>(&self, keep: &F) -> Vec<&T> {
        let mut out = Vec::new();
        self.filter_refs_into(keep, &mut out);
        out
    }

    fn filter_refs_into<'a, F: Fn(&T::Summary) -> bool>(&'a self, keep: &F, out: &mut Vec<&'a T>) {
        if !keep(self.summary()) {
            return;
        }
        match &*self.0 {
            Node::Leaf { items, .. } => out.extend(items.iter()),
            Node::Internal { children, .. } => {
                for c in children {
                    c.filter_refs_into(keep, out);
                }
            }
        }
    }

    fn visit_range<D: Dimension<T::Summary>, F: FnMut(&T, &D)>(
        &self,
        start: &D,
        from: &D,
        to: &D,
        f: &mut F,
    ) {
        let mut whole_end = start.clone();
        whole_end.add_summary(self.summary());
        if &whole_end <= from || start >= to {
            return; // this subtree is entirely outside the window
        }
        match &*self.0 {
            Node::Leaf { items, item_summaries, .. } => {
                let mut acc = start.clone();
                for (item, s) in items.iter().zip(item_summaries) {
                    let mut end = acc.clone();
                    end.add_summary(s);
                    if &end > from && &acc < to {
                        f(item, &acc);
                    }
                    acc = end;
                }
            }
            Node::Internal { children, child_summaries, .. } => {
                let mut acc = start.clone();
                for (child, s) in children.iter().zip(child_summaries) {
                    let mut end = acc.clone();
                    end.add_summary(s);
                    if &end > from && &acc < to {
                        child.visit_range(&acc, from, to, f);
                    }
                    acc = end;
                }
            }
        }
    }

    fn collect_items(&self, out: &mut Vec<T>) {
        match &*self.0 {
            Node::Leaf { items, .. } => out.extend_from_slice(items),
            Node::Internal { children, .. } => {
                for c in children {
                    c.collect_items(out);
                }
            }
        }
    }

    /// Borrowed references to every item in order (no clone) — a leaf/chunk walk.
    #[must_use]
    pub fn item_refs(&self) -> Vec<&T> {
        let mut out = Vec::new();
        self.collect_refs(&mut out);
        out
    }

    fn collect_refs<'a>(&'a self, out: &mut Vec<&'a T>) {
        match &*self.0 {
            Node::Leaf { items, .. } => out.extend(items.iter()),
            Node::Internal { children, .. } => {
                for c in children {
                    c.collect_refs(out);
                }
            }
        }
    }

    /// Debug-only structural invariant check: every internal node's children share
    /// one height, and cached summaries equal the recomputed sum.
    #[cfg(test)]
    fn assert_invariants(&self) {
        match &*self.0 {
            Node::Leaf { summary, items, item_summaries } => {
                assert_eq!(items.len(), item_summaries.len());
                assert!(items.len() <= MAX, "leaf overfull: {}", items.len());
                let recomputed = sum_of::<T>(item_summaries);
                assert_eq!(format!("{summary:?}"), format!("{recomputed:?}"), "leaf summary drift");
            }
            Node::Internal { height, summary, child_summaries, children } => {
                assert_eq!(children.len(), child_summaries.len());
                assert!(!children.is_empty() && children.len() <= MAX);
                for c in children {
                    assert_eq!(c.height() + 1, *height, "children heights disagree");
                    c.assert_invariants();
                }
                let recomputed = sum_of::<T>(child_summaries);
                assert_eq!(format!("{summary:?}"), format!("{recomputed:?}"), "internal summary drift");
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // A stand-in leaf: a run of `len` bytes tagged with an id. Summary tracks the
    // total byte length and the item count — two independent dimensions.
    #[derive(Clone, Debug, PartialEq)]
    struct Run {
        len: u32,
        id: u32,
    }
    #[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    struct Sum {
        len: u32,
        count: u32,
    }
    impl Summary for Sum {
        fn add_summary(&mut self, other: &Self) {
            self.len += other.len;
            self.count += other.count;
        }
    }
    impl Item for Run {
        type Summary = Sum;
        fn summary(&self) -> Sum {
            Sum { len: self.len, count: 1 }
        }
    }
    // Seek by cumulative byte length.
    #[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    struct ByLen(u32);
    impl Dimension<Sum> for ByLen {
        fn add_summary(&mut self, s: &Sum) {
            self.0 += s.len;
        }
    }
    // Seek by item index.
    #[derive(Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    struct ByCount(u32);
    impl Dimension<Sum> for ByCount {
        fn add_summary(&mut self, s: &Sum) {
            self.0 += s.count;
        }
    }

    fn runs(n: u32) -> Vec<Run> {
        (0..n).map(|i| Run { len: (i % 7) + 1, id: i }).collect()
    }

    #[test]
    fn build_preserves_items_and_summary() {
        for n in [0u32, 1, 5, 12, 13, 100, 1000] {
            let src = runs(n);
            let tree = SumTree::from_items(src.clone());
            assert_eq!(tree.items(), src, "n={n}");
            assert_eq!(tree.summary().count, n, "n={n}");
            assert_eq!(tree.summary().len, src.iter().map(|r| r.len).sum::<u32>(), "n={n}");
            // `extent` folds the root summary into a chosen dimension.
            assert_eq!(tree.extent::<ByCount>(), ByCount(n), "n={n}");
            assert_eq!(tree.extent::<ByLen>(), ByLen(src.iter().map(|r| r.len).sum()), "n={n}");
            if n > 0 {
                tree.assert_invariants();
            }
            // A tree of 1000 items at fan-out 12 is short.
            assert!(tree.height() <= 4, "n={n} height={}", tree.height());
        }
    }

    #[test]
    fn seek_by_index_matches_the_vec() {
        let src = runs(300);
        let tree = SumTree::from_items(src.clone());
        for k in 0..src.len() as u32 {
            let (item, start) = tree.item_at(&ByCount(k)).unwrap();
            assert_eq!(*item, src[k as usize], "k={k}");
            assert_eq!(start, ByCount(k));
        }
    }

    #[test]
    fn seek_by_length_finds_the_containing_run() {
        let src = runs(300);
        let tree = SumTree::from_items(src.clone());
        // Brute-force map: byte offset -> (item index, item start offset).
        let mut starts = Vec::new();
        let mut acc = 0u32;
        for r in &src {
            starts.push(acc);
            acc += r.len;
        }
        let total = acc;
        for off in 0..total {
            let (item, start) = tree.item_at(&ByLen(off)).unwrap();
            // Expected: the run whose [start, start+len) contains off.
            let idx = starts.partition_point(|&s| s <= off) - 1;
            assert_eq!(*item, src[idx], "off={off}");
            assert_eq!(start, ByLen(starts[idx]), "off={off}");
        }
        // At/after the total, clamp to the last run.
        let (item, _) = tree.item_at(&ByLen(total)).unwrap();
        assert_eq!(*item, *src.last().unwrap());
    }

    #[test]
    fn empty_tree_seeks_to_nothing() {
        let tree = SumTree::<Run>::new();
        assert!(tree.is_empty());
        assert_eq!(tree.summary().count, 0);
        assert!(tree.item_at(&ByLen(0)).is_none());
    }

    fn runs_from(base: u32, n: u32) -> Vec<Run> {
        (0..n).map(|i| Run { len: (i % 7) + 1, id: base + i }).collect()
    }

    #[test]
    fn append_matches_vec_concat() {
        // Distinct id ranges so a mis-ordered or dropped run is caught.
        for &(a, b) in &[(0, 0), (1, 0), (0, 1), (1, 1), (5, 7), (13, 1), (1, 13), (100, 50), (50, 100), (200, 200)] {
            let va = runs_from(0, a);
            let vb = runs_from(10_000, b);
            let ta = SumTree::from_items(va.clone());
            let tb = SumTree::from_items(vb.clone());
            let joined = ta.append(&tb);
            let mut expected = va.clone();
            expected.extend(vb.clone());
            assert_eq!(joined.items(), expected, "a={a} b={b}");
            assert_eq!(joined.summary().count, a + b, "a={a} b={b}");
            if !expected.is_empty() {
                joined.assert_invariants();
            }
        }
    }

    #[test]
    fn split_at_partitions_the_items() {
        let src = runs_from(0, 250);
        let tree = SumTree::from_items(src.clone());
        for k in 0..=src.len() as u32 {
            let (left, right) = tree.split_at(&ByCount(k));
            assert_eq!(left.items(), src[..k as usize], "split before k={k}");
            assert_eq!(right.items(), src[k as usize..], "split from k={k}");
            if !left.is_empty() {
                left.assert_invariants();
            }
            if !right.is_empty() {
                right.assert_invariants();
            }
        }
    }

    #[test]
    fn replace_matches_vec_splice_under_random_edits() {
        // The oracle: the same replace applied to a SumTree and a Vec must agree,
        // item-for-item, across 500 edits — with the tree staying valid throughout.
        let mut v = runs_from(0, 200);
        let mut tree = SumTree::from_items(v.clone());
        let mut state = 0x9e37_79b9u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        let mut next_id = 1_000_000u32;
        for step in 0..500 {
            let len = v.len();
            let a = (next() as usize) % (len + 1);
            let b = a + (next() as usize) % (len - a + 1);
            let insert_n = (next() as usize) % 6;
            let new: Vec<Run> = (0..insert_n)
                .map(|_| {
                    next_id += 1;
                    Run { len: (next() % 7) + 1, id: next_id }
                })
                .collect();
            tree = tree.replace(ByCount(a as u32)..ByCount(b as u32), new.clone());
            v.splice(a..b, new);
            assert_eq!(tree.items(), v, "step {step}: replace {a}..{b}");
            assert_eq!(tree.summary().count, v.len() as u32, "step {step}: count");
            assert_eq!(tree.summary().len, v.iter().map(|r| r.len).sum::<u32>(), "step {step}: len");
            if !v.is_empty() {
                tree.assert_invariants();
            }
        }
    }

    #[test]
    fn seek_stays_correct_after_edits() {
        // After a delete + insert, item_at must still map offsets to the right run.
        let src = runs_from(0, 60);
        let mut tree = SumTree::from_items(src);
        tree = tree.replace(ByCount(10)..ByCount(40), runs_from(9_000, 3));
        let items = tree.items();
        let mut starts = Vec::new();
        let mut acc = 0u32;
        for r in &items {
            starts.push(acc);
            acc += r.len;
        }
        for off in 0..acc {
            let (item, start) = tree.item_at(&ByLen(off)).unwrap();
            let idx = starts.partition_point(|&s| s <= off) - 1;
            assert_eq!(*item, items[idx], "off={off}");
            assert_eq!(start, ByLen(starts[idx]));
        }
    }

    #[test]
    fn seek_reports_both_dimensions() {
        let src = runs_from(0, 200);
        let tree = SumTree::from_items(src.clone());
        let mut byte_start = 0u32;
        for (idx, r) in src.iter().enumerate() {
            let (item, ByCount(c), ByLen(b)) =
                tree.seek::<ByCount, ByLen>(&ByCount(idx as u32)).unwrap();
            assert_eq!(*item, *r, "idx={idx}");
            assert_eq!(c, idx as u32);
            assert_eq!(b, byte_start, "idx={idx}: byte start");
            byte_start += r.len;
        }
    }

    fn split_run(r: &Run, start: &ByLen, at: &ByLen) -> (Run, Run) {
        let pos = at.0 - start.0; // bytes into the run
        (Run { len: pos, id: r.id }, Run { len: r.len - pos, id: r.id })
    }

    // An interval item for the filter_visit (interval-tree) test: delta-gap start,
    // `len` gives end = start + len; the summary carries the subtree's max end.
    #[derive(Clone, Debug)]
    struct Iv {
        gap: u32,
        len: u32,
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
    struct IvSum {
        span: u32,
        count: u32,
        max_end: u32,
    }
    impl Summary for IvSum {
        fn add_summary(&mut self, o: &Self) {
            self.max_end = self.max_end.max(self.span + o.max_end);
            self.span += o.span;
            self.count += o.count;
        }
    }
    impl Item for Iv {
        type Summary = IvSum;
        fn summary(&self) -> IvSum {
            IvSum { span: self.gap, count: 1, max_end: self.gap + self.len }
        }
    }
    #[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
    struct StartOf(u32);
    impl Dimension<IvSum> for StartOf {
        fn add_summary(&mut self, s: &IvSum) {
            self.0 += s.span;
        }
    }

    #[test]
    fn filter_visit_prunes_to_interval_overlaps() {
        let spans = [(2u32, 3u32), (5, 1), (9, 10), (12, 0), (20, 4), (25, 7), (33, 0), (40, 2), (41, 9)];
        let mut prev = 0;
        let items: Vec<Iv> = spans
            .iter()
            .map(|&(s, l)| {
                let iv = Iv { gap: s - prev, len: l };
                prev = s;
                iv
            })
            .collect();
        let tree = SumTree::from_items(items);
        let top = spans.iter().map(|&(s, l)| s + l).max().unwrap() + 2;
        for lo in 0..=top {
            for hi in lo..=top {
                // Enter a subtree iff it could hold an interval overlapping [lo,hi):
                // some start < hi (start > `before`) AND some end > lo (max end =
                // before + subtree max_end).
                let mut got: Vec<(u32, u32)> = Vec::new();
                tree.filter_visit::<StartOf, _, _>(
                    &|before: &StartOf, sum: &IvSum| before.0 < hi && before.0 + sum.max_end > lo,
                    &mut |iv: &Iv, &StartOf(before)| {
                        let (start, end) = (before + iv.gap, before + iv.gap + iv.len);
                        if start < hi && end > lo {
                            got.push((start, end));
                        }
                    },
                );
                let want: Vec<(u32, u32)> =
                    spans.iter().map(|&(s, l)| (s, s + l)).filter(|&(s, e)| s < hi && e > lo).collect();
                assert_eq!(got, want, "overlap [{lo},{hi})");
            }
        }
    }

    #[test]
    fn bucketed_reduce_matches_linear_per_bucket_scan() {
        // The reduce must bucket items by START (delta-gap: start = before + gap)
        // exactly as a naive per-bucket scan, for every monotonic bound set —
        // folding whole subtrees through their cached `count` where they fit, only
        // descending straddlers. A reduce that mis-assigns a boundary item
        // (off-by-one in `bucket_of`) or double-counts a straddling subtree
        // diverges from the linear scan here.
        let spans = [(2u32, 3u32), (5, 1), (5, 4), (9, 10), (12, 0), (20, 4), (25, 7), (33, 0), (40, 2), (41, 9), (60, 1)];
        let mut prev = 0;
        let items: Vec<Iv> = spans
            .iter()
            .map(|&(s, l)| {
                let iv = Iv { gap: s - prev, len: l };
                prev = s;
                iv
            })
            .collect();
        let tree = SumTree::from_items(items);
        let starts: Vec<u32> = spans.iter().map(|&(s, _)| s).collect();
        // Every ascending 3-boundary (2-bucket) set over the offset range, plus a
        // few wider partitions, so both the whole-subtree fold and the straddler
        // descent are exercised at many alignments.
        let bound_sets: &[&[u32]] = &[
            &[0, 10, 70],
            &[0, 5, 6, 100],
            &[5, 5, 42],   // a degenerate empty bucket
            &[0, 1, 2, 3, 4, 5, 6, 40, 41, 42, 61, 62],
            &[3, 33, 41],  // bounds landing exactly on item starts
        ];
        for bounds in bound_sets {
            let bnd: Vec<StartOf> = bounds.iter().map(|&b| StartOf(b)).collect();
            let got: Vec<u32> = tree.bucketed_reduce(&bnd, 0u32, |a: &mut u32, s: &IvSum| *a += s.count);
            let want: Vec<u32> = bounds
                .windows(2)
                .map(|w| starts.iter().filter(|&&s| s >= w[0] && s < w[1]).count() as u32)
                .collect();
            assert_eq!(got, want, "bounds {bounds:?}");
        }
    }

    #[test]
    fn split_with_cuts_inside_an_item() {
        let src = runs_from(0, 100);
        let tree = SumTree::from_items(src.clone());
        let total: u32 = src.iter().map(|r| r.len).sum();
        for at in 0..=total {
            let (left, right) = tree.split_with(&ByLen(at), &mut split_run);
            // Byte-precise: left holds exactly the first `at` bytes, nothing lost.
            assert_eq!(left.summary().len, at, "at={at}: left byte count");
            assert_eq!(left.summary().len + right.summary().len, total, "at={at}: total preserved");
            if !left.is_empty() {
                left.assert_invariants();
            }
            if !right.is_empty() {
                right.assert_invariants();
            }
        }
    }
}
