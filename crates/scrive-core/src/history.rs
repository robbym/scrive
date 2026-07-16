//! Undo/redo as a **branching tree**, the grouping mechanism, and dirty
//! tracking.
//!
//! History stores each committed transaction's ops in **both** directions
//! (`forward` = parent→node, `inverse` = node→parent). Undo replays a node's
//! inverse ops and redo replays a child's forward ops — both through the same
//! `apply` engine (undo/redo are just transactions), so there is
//! one mutation path in every direction and an inverse can never drift from
//! what was applied.
//!
//! The structure is an **arena tree** (`UndoNode`), not two stacks: node `0`
//! is the base state, every other node holds one `Element` (the edit from its
//! parent). A new edit made after an undo becomes a *new child* of the current
//! node — the branch you undid out of is **retained** as a sibling rather than
//! discarded, so redo can be steered to it (`History::select_redo_branch`).
//! Plain `undo`/`redo` follow
//! `preferred_child`, which always points at the most-recent child, so with no
//! branch navigation the tree reproduces classic linear undo exactly.
//!
//! Grouping here is the **mechanism**, not the policy. A [`GroupingHint`]
//! supplied per edit says whether to seal the open element and what class the
//! edit is; consecutive same-class edits merge into one undo node while it stays
//! open. The classifier that *produces* those hints from keystrokes (seal on
//! cursor move, glue on space, and the other coalescing rules that match how
//! mainstream editors group typing into undo steps) lives in the verbs layer.
//!
//! Dirty tracking uses an alternative-version id (`alt`): a fresh id per recorded
//! transaction, stored per element and *restored* by undo/redo. This makes
//! dirtiness a comparison against the id captured at the last save — undoing
//! back to a save point reads clean, and a divergent edit made after an undo
//! reads dirty even if the buffer text happens to coincide with the saved text.
//! Because ids are monotonic and never reused, this stays correct across
//! branches.

use crate::buffer::Buffer;
use crate::selection::SelectionSet;
use crate::transaction::{apply, Committed, EditOp};

/// Coarse edit class, for merge compatibility. The full classifier is the verbs
/// layer; this is the vocabulary the grouping mechanism keys on.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum OpClass {
    /// Character insertion — merges with adjacent typing.
    Type,
    /// Backspace/delete — merges with adjacent deletion.
    Delete,
    /// Anything discrete (paste, replace, indent, programmatic) — never merges.
    Other,
}

/// Grouping instruction for one edit. `seal_before` closes the open element
/// before this edit records; `seal_after` closes it afterward. `op` decides
/// merge compatibility with the open element.
#[derive(Copy, Clone, Debug)]
pub struct GroupingHint {
    /// This edit's class.
    pub op: OpClass,
    /// Close the open element before recording (start a fresh one).
    pub seal_before: bool,
    /// Close the element after recording.
    pub seal_after: bool,
}

impl GroupingHint {
    /// A typing/deleting edit that may merge with its neighbours (seals neither
    /// side).
    #[must_use]
    pub fn mergeable(op: OpClass) -> Self {
        Self { op, seal_before: false, seal_after: false }
    }

    /// A discrete edit that is its own undo step (seals both sides).
    #[must_use]
    pub fn discrete() -> Self {
        Self { op: OpClass::Other, seal_before: true, seal_after: true }
    }
}

/// One transaction inside an [`Element`], stored in both directions so a tree
/// edge is traversable either way with no re-derivation. `forward` reproduces
/// the edit (parent→node, replayed on redo); `inverse` reverts it (node→parent,
/// replayed on undo). Both are produced by the one [`apply`] engine at record
/// time (`forward` is the original ops; `inverse` is [`Committed::into_inverse`]).
#[derive(Clone, Debug)]
struct Step {
    forward: Vec<EditOp>,
    inverse: Vec<EditOp>,
}

/// One undo unit: the edit from a node's parent to the node, as a chronological
/// list of [`Step`]s (one for a discrete edit, several for a merged typing run).
/// `alt_before`/`alt_after` are the document versions on either side of the unit.
#[derive(Clone, Debug)]
struct Element {
    steps: Vec<Step>,
    op: OpClass,
    alt_before: u64,
    alt_after: u64,
}

/// Arena index into [`History::nodes`]. Node `0` is the root (base state).
type NodeId = usize;

/// One node of the undo tree. The root and pruned tombstones carry no
/// [`Element`]; every other node's element is the edit from `parent` to here.
#[derive(Clone, Debug)]
struct UndoNode {
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    /// The child [`redo`](History::redo) follows when several exist — always the
    /// most-recent child unless [`select_redo_branch`](History::select_redo_branch)
    /// re-points it, so plain redo is classic-linear.
    preferred_child: Option<NodeId>,
    /// The edit from `parent` to this node (`None` for the root/tombstones).
    elem: Option<Element>,
}

/// The branching undo/redo tree plus dirty tracking. Owned by the `Document`
/// aggregate, which passes its buffer in for the replay.
#[derive(Debug)]
pub(crate) struct History {
    /// Arena; node `0` is the root. Indices are stable for a node's lifetime.
    nodes: Vec<UndoNode>,
    /// Where we are in the tree.
    current: NodeId,
    /// Whether `current`'s element may still accept a merged edit.
    open: bool,
    /// Current alternative-version id.
    alt: u64,
    /// Monotonic id source (never reused → branch-safe dirty tracking).
    alt_seq: u64,
    /// The `alt` at the last save.
    saved_alt: u64,
    /// Opt-in cap on undo reach: keep at most this many undo units on the line to
    /// `current`. `None` (the default) = unbounded. Dropping the oldest unit also
    /// drops any branch that diverged before it (older than the kept window).
    max_undo: Option<usize>,
}

impl History {
    pub(crate) fn new() -> Self {
        Self {
            nodes: vec![UndoNode { parent: None, children: Vec::new(), preferred_child: None, elem: None }],
            current: 0,
            open: false,
            alt: 0,
            alt_seq: 0,
            saved_alt: 0,
            max_undo: None,
        }
    }

    /// Set the undo-reach cap (`None` = unbounded, the default) and prune to it
    /// immediately. Bounds history memory when a host wants it; the default
    /// keeps every unit.
    pub(crate) fn set_max_undo(&mut self, limit: Option<usize>) {
        self.max_undo = limit;
        self.prune_if_needed();
    }

    /// How many undo units are reachable from `current` (its ancestry depth).
    pub(crate) fn undo_depth(&self) -> usize {
        let mut depth = 0;
        let mut node = self.current;
        while let Some(parent) = self.nodes[node].parent {
            node = parent;
            depth += 1;
        }
        depth
    }

    /// Whether there is an edit at `current` to revert.
    pub(crate) fn can_undo(&self) -> bool {
        self.current != 0
    }

    /// Whether `current` has a redo branch to follow.
    pub(crate) fn can_redo(&self) -> bool {
        self.nodes[self.current].preferred_child.is_some()
    }

    /// Record a committed transaction under `hint`, given its ops in both
    /// directions (`forward` = the applied ops, `inverse` =
    /// [`Committed::into_inverse`]).
    ///
    /// Merges into `current`'s element when the classes match and nothing seals
    /// between them; otherwise opens a **new child** of `current` and moves there.
    /// Unlike a two-stack history there is no redo to clear — a sibling branch
    /// created earlier simply remains reachable via [`select_redo_branch`].
    pub(crate) fn record(&mut self, forward: Vec<EditOp>, inverse: Vec<EditOp>, hint: GroupingHint) {
        if hint.seal_before {
            self.open = false;
        }

        let alt_before = self.alt;
        self.alt_seq += 1;
        self.alt = self.alt_seq;
        let alt_after = self.alt;

        let step = Step { forward, inverse };
        let can_merge = self.open
            && hint.op != OpClass::Other
            && self.nodes[self.current].elem.as_ref().is_some_and(|e| e.op == hint.op);

        if can_merge {
            let elem = self.nodes[self.current].elem.as_mut().expect("open implies an element");
            elem.steps.push(step);
            elem.alt_after = alt_after;
        } else {
            let parent = self.current;
            let new_id = self.nodes.len();
            self.nodes.push(UndoNode {
                parent: Some(parent),
                children: Vec::new(),
                preferred_child: None,
                elem: Some(Element { steps: vec![step], op: hint.op, alt_before, alt_after }),
            });
            let parent_node = &mut self.nodes[parent];
            parent_node.children.push(new_id);
            parent_node.preferred_child = Some(new_id);
            self.current = new_id;
            self.open = true;
        }

        if hint.seal_after {
            self.open = false;
        }
        self.prune_if_needed();
    }

    /// Drop the oldest undo units (and any branch older than the kept window) so
    /// the reach from `current` stays within `max_undo`. A no-op when unbounded
    /// or already within budget (so the default path pays nothing). Rebuilds the
    /// arena from the new base — O(kept nodes), and only when over budget.
    fn prune_if_needed(&mut self) {
        let Some(max) = self.max_undo else { return };
        if self.undo_depth() <= max {
            return;
        }
        // The ancestor `max` steps above `current` becomes the new base state,
        // keeping exactly `max` units on the line to `current` plus every branch
        // that hangs off them; everything older is dropped.
        let mut new_base = self.current;
        for _ in 0..max {
            new_base = self.nodes[new_base].parent.expect("depth > max implies enough ancestors");
        }
        self.rebuild_from(new_base);
    }

    /// Rebuild the arena keeping only the subtree rooted at `new_base`, remapping
    /// ids into a fresh dense `Vec`. `new_base` becomes the root (no parent, no
    /// element — it is now the base state); `current` and every `preferred_child`
    /// are remapped. Nodes above/beside `new_base` are reclaimed.
    fn rebuild_from(&mut self, new_base: NodeId) {
        // BFS the kept subtree; `order[k]` is the old id that becomes new id `k`.
        let mut remap = vec![usize::MAX; self.nodes.len()];
        let mut order = vec![new_base];
        remap[new_base] = 0;
        let mut i = 0;
        while i < order.len() {
            let old = order[i];
            for &child in &self.nodes[old].children {
                if remap[child] == usize::MAX {
                    remap[child] = order.len();
                    order.push(child);
                }
            }
            i += 1;
        }
        let fresh: Vec<UndoNode> = order
            .iter()
            .enumerate()
            .map(|(new_id, &old)| {
                let node = &self.nodes[old];
                UndoNode {
                    // The new base has no parent and no element (it is the base
                    // state); its old parent lies outside the kept subtree.
                    parent: if new_id == 0 { None } else { node.parent.map(|p| remap[p]) },
                    children: node.children.iter().map(|&c| remap[c]).collect(),
                    preferred_child: node.preferred_child.map(|c| remap[c]),
                    elem: if new_id == 0 { None } else { node.elem.clone() },
                }
            })
            .collect();
        self.current = remap[self.current];
        self.nodes = fresh;
    }

    /// Undo the edit at `current` against `buffer`, moving to the parent and
    /// rebasing `selections` through each reverted step so carets stay valid.
    /// `on_step` fires after each reverted step with its [`Committed`] and the
    /// post-step buffer, so the caller rebases its derived views through the same
    /// patch — the one mover that keeps every derived view consistent across
    /// undo. Returns `false` if there is nothing to undo.
    pub(crate) fn undo(
        &mut self,
        buffer: &mut Buffer,
        selections: &mut SelectionSet,
        on_step: impl FnMut(&Committed, &Buffer),
    ) -> bool {
        if self.current == 0 {
            return false;
        }
        let elem = self.nodes[self.current].elem.clone().expect("non-root nodes carry an element");
        // Revert the element's steps newest-first; each step's inverse is in the
        // coordinates of the buffer state at that point (see module docs).
        replay(buffer, selections, elem.steps.iter().rev().map(|s| &s.inverse), on_step);
        self.current = self.nodes[self.current].parent.expect("non-root nodes have a parent");
        self.alt = elem.alt_before;
        self.open = false;
        true
    }

    /// Redo along `current`'s preferred branch — the [`undo`](Self::undo) mirror,
    /// replaying the child's forward steps through the same mover. Returns
    /// `false` if there is nothing to redo.
    pub(crate) fn redo(
        &mut self,
        buffer: &mut Buffer,
        selections: &mut SelectionSet,
        on_step: impl FnMut(&Committed, &Buffer),
    ) -> bool {
        let Some(child) = self.nodes[self.current].preferred_child else {
            return false;
        };
        let elem = self.nodes[child].elem.clone().expect("child nodes carry an element");
        // Re-apply the element's steps oldest-first (forward coordinates).
        replay(buffer, selections, elem.steps.iter().map(|s| &s.forward), on_step);
        self.current = child;
        self.alt = elem.alt_after;
        self.open = false;
        true
    }

    /// The number of redo branches available from `current` — 1 for the classic
    /// single-line case, ≥2 where an undo was followed by a divergent edit.
    pub(crate) fn redo_branch_count(&self) -> usize {
        self.nodes[self.current].children.len()
    }

    /// Steer the next [`redo`](Self::redo) to the `index`-th branch of `current`
    /// (0 = oldest child), returning `false` if out of range. The chosen branch
    /// becomes `preferred_child`, so it also becomes the default thereafter.
    pub(crate) fn select_redo_branch(&mut self, index: usize) -> bool {
        let Some(&child) = self.nodes[self.current].children.get(index) else {
            return false;
        };
        self.nodes[self.current].preferred_child = Some(child);
        true
    }

    /// Close the current undo group so the next edit starts a fresh element even
    /// if its op class would otherwise coalesce. A jump-class selection change —
    /// such as find navigation moving the caret elsewhere — seals through this so
    /// typing after the jump never merges with the typing run before it.
    pub(crate) fn seal(&mut self) {
        self.open = false;
    }

    /// Whether the document differs from the last save point.
    pub(crate) fn is_dirty(&self) -> bool {
        self.alt != self.saved_alt
    }

    /// Mark the current state as saved.
    pub(crate) fn mark_saved(&mut self) {
        self.saved_alt = self.alt;
    }
}

/// Apply each op-set to `buffer` in the given order, rebasing `selections`
/// through each patch so carets track the change, and firing `on_step` with each
/// step's [`Committed`] + the post-step buffer (so the caller rebases its derived
/// views through the same patch). The ops are already stored in the direction
/// being replayed, so — unlike a two-stack history — nothing is collected or
/// flipped here.
fn replay<'a>(
    buffer: &mut Buffer,
    selections: &mut SelectionSet,
    steps: impl Iterator<Item = &'a Vec<EditOp>>,
    mut on_step: impl FnMut(&Committed, &Buffer),
) {
    for step in steps {
        let committed = apply(buffer, step.clone()).expect("history ops are disjoint by construction");
        selections.rebase(committed.patch());
        on_step(&committed, buffer);
    }
}
