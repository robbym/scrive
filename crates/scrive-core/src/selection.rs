//! Selections and the multi-cursor set.
//!
//! A [`Selection`] is a byte-offset range with a direction: `start <= end`
//! always, and `reversed` says which end the caret (the *head*) is on, so the
//! set is always stored in document order while the caret can face either way.
//! Empty selections (`start == end`) are bare carets.
//!
//! A [`SelectionSet`] holds one or more disjoint selections, kept sorted and
//! merged. Multi-cursor is a first-class model concept: the gestures that add
//! cursors live in the UI layer, and this is the data they act on. Selections
//! rebase through every edit's [`Patch`] eagerly, so a caret is always the
//! current position and never a stale copy of an offset from before the edit.

use std::cell::RefCell;

use crate::coords::Bias;
use crate::patch::Patch;

/// Reused per-thread scratch for [`SelectionSet::rebase`]'s batch map: the
/// `(offset, bias)` queries and their mapped results.
type RebaseScratch = (Vec<(u32, Bias)>, Vec<u32>);

/// Identity for a selection, monotonic within a [`SelectionSet`]. Larger ids are
/// newer; used to pick the newest (autoscroll target) and oldest (Escape).
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct SelectionId(pub usize);

/// A directional byte-offset range. `start <= end` is an invariant; `reversed`
/// records whether the caret is at `start` (`true`) or `end` (`false`).
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub struct Selection {
    /// Stable identity within its set.
    pub id: SelectionId,
    start: u32,
    end: u32,
    reversed: bool,
    /// Goal for vertical movement: a **display cell** — the caret's visual
    /// column, kept through tab stops and collapsed folds. Cleared by any
    /// horizontal move or edit.
    pub goal: Option<u32>,
}

impl Selection {
    /// A caret (empty selection) with the given id at `offset`.
    #[must_use]
    pub fn caret(id: SelectionId, offset: u32) -> Self {
        Self { id, start: offset, end: offset, reversed: false, goal: None }
    }

    /// A selection from `anchor` to `head` (either order); `reversed` is derived
    /// so the caret sits at `head`.
    #[must_use]
    pub fn from_anchor(id: SelectionId, anchor: u32, head: u32) -> Self {
        Self {
            id,
            start: anchor.min(head),
            end: anchor.max(head),
            reversed: head < anchor,
            goal: None,
        }
    }

    /// Lower bound (inclusive).
    #[must_use]
    pub fn start(&self) -> u32 {
        self.start
    }

    /// Upper bound (exclusive-ish — the position past the selection).
    #[must_use]
    pub fn end(&self) -> u32 {
        self.end
    }

    /// The moving end (where the caret is).
    #[must_use]
    pub fn head(&self) -> u32 {
        if self.reversed {
            self.start
        } else {
            self.end
        }
    }

    /// The fixed end (the anchor).
    #[must_use]
    pub fn tail(&self) -> u32 {
        if self.reversed {
            self.end
        } else {
            self.start
        }
    }

    /// Whether this is a bare caret.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.start == self.end
    }

    /// Move the head to `offset`, keeping the tail fixed (extends/shrinks the
    /// selection, flipping direction if the head crosses the tail). Clears the
    /// goal column.
    pub fn set_head(&mut self, offset: u32) {
        let tail = self.tail();
        self.start = tail.min(offset);
        self.end = tail.max(offset);
        self.reversed = offset < tail;
        self.goal = None;
    }

    /// Collapse to a caret at the head.
    pub fn collapse_to_head(&mut self) {
        let h = self.head();
        self.start = h;
        self.end = h;
        self.reversed = false;
    }

    /// Collapse to a caret at `offset`, keeping this selection's id and clearing
    /// the goal column. The movement layer's "plain move" primitive.
    pub fn move_to_caret(&mut self, offset: u32) {
        self.start = offset;
        self.end = offset;
        self.reversed = false;
        self.goal = None;
    }
}

/// A non-empty, sorted, disjoint set of selections. Merged after every
/// mutation so no two selections overlap.
#[derive(Clone, Debug)]
pub struct SelectionSet {
    selections: Vec<Selection>,
    next_id: usize,
}

impl SelectionSet {
    /// A set with a single caret at `offset`.
    #[must_use]
    pub fn new(offset: u32) -> Self {
        Self { selections: vec![Selection::caret(SelectionId(0), offset)], next_id: 1 }
    }

    /// A set of carets at the given offsets (merged). Used by the verbs layer to
    /// place carets after an edit. Panics if `offsets` is empty — a set is never
    /// empty.
    #[must_use]
    pub fn from_offsets(offsets: &[u32]) -> Self {
        assert!(!offsets.is_empty(), "a selection set is never empty");
        let mut set = Self { selections: Vec::new(), next_id: 0 };
        for &o in offsets {
            let id = set.mint();
            set.selections.push(Selection::caret(id, o));
        }
        set.normalize();
        set
    }

    /// Build a set from `(anchor, head)` ranges (any order); the range at index
    /// `newest` becomes the primary (largest id) — the autoscroll target. Used to
    /// install a column (box) selection. Panics if `ranges` is empty or
    /// `newest` is out of bounds.
    #[must_use]
    pub fn from_ranges(ranges: &[(u32, u32)], newest: usize) -> Self {
        assert!(!ranges.is_empty(), "a selection set is never empty");
        assert!(newest < ranges.len(), "newest index out of bounds");
        let mut set = Self { selections: Vec::new(), next_id: 0 };
        // Assign ids so `newest` gets the largest: push the others first, it last.
        for (i, &(anchor, head)) in ranges.iter().enumerate() {
            if i != newest {
                let id = set.mint();
                set.selections.push(Selection::from_anchor(id, anchor, head));
            }
        }
        let id = set.mint();
        let (anchor, head) = ranges[newest];
        set.selections.push(Selection::from_anchor(id, anchor, head));
        set.normalize();
        set
    }

    /// The selections, in document order.
    #[must_use]
    pub fn all(&self) -> &[Selection] {
        &self.selections
    }

    /// How many selections (always ≥ 1).
    #[must_use]
    pub fn len(&self) -> usize {
        self.selections.len()
    }

    /// Always `false` — a set is never empty. Present for lint-friendliness
    /// alongside [`SelectionSet::len`].
    #[must_use]
    pub fn is_empty(&self) -> bool {
        false
    }

    /// The newest selection (largest id) — the autoscroll / primary target.
    #[must_use]
    pub fn newest(&self) -> &Selection {
        self.selections.iter().max_by_key(|s| s.id).expect("set is non-empty")
    }

    /// Mint the next selection id.
    fn mint(&mut self) -> SelectionId {
        let id = SelectionId(self.next_id);
        self.next_id += 1;
        id
    }

    /// Add a caret at `offset`, merging into any selection it touches.
    pub fn add_caret(&mut self, offset: u32) {
        let id = self.mint();
        self.selections.push(Selection::caret(id, offset));
        self.normalize();
    }

    /// Add a selection from `anchor` to `head`, merging as needed.
    pub fn add_selection(&mut self, anchor: u32, head: u32) {
        let id = self.mint();
        self.selections.push(Selection::from_anchor(id, anchor, head));
        self.normalize();
    }

    /// Replace the whole set with a single selection (Escape collapses to the
    /// oldest; here the caller supplies it).
    pub fn set_single(&mut self, sel: Selection) {
        self.selections = vec![sel];
    }

    /// Collapse to just the newest selection, as a caret at its head (the
    /// standard Escape behavior for multi-cursor).
    pub fn collapse_to_newest(&mut self) {
        let mut n = *self.newest();
        n.collapse_to_head();
        self.selections = vec![n];
    }

    /// Collapse to just the oldest selection (smallest id), as a caret at its
    /// head — the multi-cursor Escape that keeps the *primary* cursor.
    /// With a single ranged selection this simply deselects it to a caret.
    pub fn collapse_to_primary(&mut self) {
        let mut primary = *self.selections.iter().min_by_key(|s| s.id).expect("set is non-empty");
        primary.collapse_to_head();
        self.selections = vec![primary];
    }

    /// Mutate every selection through `f`, then re-merge. The building block for
    /// movement.
    pub fn map_each(&mut self, mut f: impl FnMut(&mut Selection)) {
        for s in &mut self.selections {
            f(s);
        }
        self.normalize();
    }

    /// Rebase every selection through an edit's [`Patch`] — the eager mover that
    /// keeps every endpoint current as text is inserted and deleted. Endpoints
    /// use `Left` bias for the start and `Right` for the end so a selection
    /// grows over text typed at its edges; carets follow the insertion.
    /// Re-merges afterward.
    pub fn rebase(&mut self, patch: &Patch) {
        // One batch map through the patch instead of a per-selection loop of
        // `map_offset` (which restarts the edit scan for every endpoint —
        // O(carets²) when a document-scale multi-cursor edit produces a
        // carets-sized patch). Each selection contributes one query (caret) or
        // two (range); results are consumed back in the same order.
        // The `queries`/`mapped` scratch is a thread-local reused across every
        // rebase, so a committed edit costs no fresh allocation for the batch map
        // (the mover runs on every keystroke that changes text, plus undo/redo).
        // Thread-local rather than a `SelectionSet` field, so it adds no per-set
        // memory and `clone` is unaffected; cleared before use and left empty at
        // rest so the retained buffer doesn't pin memory between edits.
        thread_local! {
            static SCRATCH: RefCell<RebaseScratch> =
                const { RefCell::new((Vec::new(), Vec::new())) };
        }
        SCRATCH.with(|cell| {
            let (queries, mapped) = &mut *cell.borrow_mut();
            queries.clear();
            for s in &self.selections {
                if s.start == s.end {
                    // A caret follows the insertion (Right bias) so it stays a
                    // caret rather than splitting into a selection.
                    queries.push((s.start, Bias::Right));
                } else {
                    queries.push((s.start, Bias::Left));
                    queries.push((s.end, Bias::Right));
                }
            }
            patch.map_many(&queries[..], mapped);
            let mut mi = 0;
            for s in &mut self.selections {
                if s.start == s.end {
                    let o = mapped[mi];
                    mi += 1;
                    s.start = o;
                    s.end = o;
                } else {
                    let (ns, ne) = (mapped[mi], mapped[mi + 1]);
                    mi += 2;
                    s.start = ns.min(ne);
                    s.end = ne.max(ns);
                }
                s.goal = None;
            }
            queries.clear(); // don't pin the batch's memory between edits
        });
        self.normalize();
    }

    /// Sort by start and merge overlapping / touching-a-caret selections.
    ///
    /// Merge rule: merge on overlap, shared start, or a caret touching a
    /// boundary; two *non-empty* selections that merely touch do NOT merge.
    /// The surviving selection keeps the newest member's id, direction, and
    /// goal.
    fn normalize(&mut self) {
        self.selections.sort_by_key(|s| (s.start, s.end));
        // In-place compaction (dedup-style): the write cursor `w` keeps the merged
        // prefix; each later selection either merges into `selections[w]` or is
        // moved up to `selections[w + 1]`. Nothing merges in the common 1-few-caret
        // case, so this allocates nothing — versus building a fresh `merged` Vec
        // every time, and `normalize` rides both caret movement AND every edit.
        // (`Selection: Copy`, so reading `s`/`prev` out by value is free.)
        let mut w = 0;
        for r in 1..self.selections.len() {
            let s = self.selections[r];
            if should_merge(&self.selections[w], &s) {
                let prev = self.selections[w];
                // Keep the newer member's identity/direction/goal.
                let newer = if s.id > prev.id { s } else { prev };
                let dst = &mut self.selections[w];
                dst.start = prev.start.min(s.start);
                dst.end = prev.end.max(s.end);
                dst.id = newer.id;
                dst.reversed = newer.reversed;
                dst.goal = newer.goal;
            } else {
                w += 1;
                self.selections[w] = s;
            }
        }
        self.selections.truncate(w + 1);
    }
}

/// Whether `b` (which starts at or after `a`) should merge into `a`.
fn should_merge(a: &Selection, b: &Selection) -> bool {
    debug_assert!(a.start <= b.start);
    if b.start < a.end {
        return true; // strict overlap
    }
    if a.start == b.start {
        return true; // shared start
    }
    if b.start == a.end {
        // Touching: merge only if at least one side is a caret.
        return a.is_empty() || b.is_empty();
    }
    false
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::patch::Edit;

    #[test]
    fn head_tail_and_direction() {
        let s = Selection::from_anchor(SelectionId(0), 2, 8);
        assert_eq!((s.start(), s.end(), s.head(), s.tail()), (2, 8, 8, 2));
        let r = Selection::from_anchor(SelectionId(0), 8, 2); // caret before anchor
        assert_eq!((r.start(), r.end(), r.head(), r.tail()), (2, 8, 2, 8));
    }

    #[test]
    fn set_head_flips_when_crossing_tail() {
        let mut s = Selection::from_anchor(SelectionId(0), 5, 8); // tail 5, head 8
        s.set_head(2); // head crosses tail to the left
        assert_eq!((s.start(), s.end(), s.head(), s.tail()), (2, 5, 2, 5));
    }

    #[test]
    fn overlapping_selections_merge() {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 0, 5));
        set.add_selection(3, 9); // overlaps 0..5
        assert_eq!(set.len(), 1);
        assert_eq!((set.all()[0].start(), set.all()[0].end()), (0, 9));
    }

    #[test]
    fn nonempty_touching_selections_do_not_merge() {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 0, 3));
        set.add_selection(3, 6); // touches at 3, both non-empty
        assert_eq!(set.len(), 2, "non-empty touching selections stay separate");
    }

    #[test]
    fn caret_touching_a_boundary_merges() {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 0, 3));
        set.add_caret(3); // a caret at the boundary
        assert_eq!(set.len(), 1);
    }

    #[test]
    fn newest_wins_identity_on_merge() {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 0, 5));
        set.add_selection(3, 9); // id 1 (newer)
        assert_eq!(set.all()[0].id, SelectionId(1));
    }

    #[test]
    fn rebase_follows_an_insertion() {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 5, 10));
        // Insert 3 bytes at offset 0: everything shifts +3.
        let patch = Patch::single(Edit { old: 0..0, new: 0..3 });
        set.rebase(&patch);
        assert_eq!((set.all()[0].start(), set.all()[0].end()), (8, 13));
    }

    #[test]
    fn rebase_collapses_a_selection_inside_a_deletion() {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), 3, 7));
        // Delete 0..10: the selection collapses to the deletion point.
        let patch = Patch::single(Edit { old: 0..10, new: 0..0 });
        set.rebase(&patch);
        assert!(set.all()[0].is_empty());
        assert_eq!(set.all()[0].start(), 0);
    }

    #[test]
    fn from_ranges_installs_a_box_with_the_chosen_newest() {
        // Three rows' worth of ranges; index 2 is the active row (newest).
        let set = SelectionSet::from_ranges(&[(0, 3), (10, 13), (20, 23)], 2);
        assert_eq!(set.len(), 3);
        assert_eq!((set.newest().start(), set.newest().end()), (20, 23));
    }

    #[test]
    fn collapse_to_primary_keeps_the_oldest_as_a_caret() {
        let mut set = SelectionSet::new(0); // id 0 at offset 0
        set.add_selection(5, 9); // id 1
        set.add_caret(20); // id 2
        assert!(set.len() >= 2);
        set.collapse_to_primary();
        assert_eq!(set.len(), 1);
        assert_eq!(set.all()[0].id, SelectionId(0), "the oldest cursor is kept");
        assert!(set.all()[0].is_empty());
        assert_eq!(set.all()[0].head(), 0);
    }

    #[test]
    fn collapse_to_newest_keeps_one_caret() {
        let mut set = SelectionSet::new(0);
        set.add_selection(5, 9);
        set.add_caret(20);
        assert!(set.len() >= 2);
        set.collapse_to_newest();
        assert_eq!(set.len(), 1);
        assert!(set.all()[0].is_empty());
    }
}
