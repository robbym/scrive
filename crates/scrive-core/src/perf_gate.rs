//! The **complexity gate** — a proactive matrix that ends the recurring
//! "discover a superlinear hot-path cost by feeling lag" loop.
//!
//! It does NOT enumerate functions. It enumerates a handful of representative
//! editor *operations* and runs each at 2× a scale dimension (caret count, fold
//! count, decoration count, document size), asserting the [`crate::perf`] work
//! meter grows within that operation's declared budget. Coverage comes from the
//! meter sitting on the shared *semantic* primitives every hot path flows through
//! (`offset_to_point`, the per-query `map_offset`, the bracket enclosing-walk
//! step, `display_position`, the document-scale `sort`/`retain`), so ANY function
//! — named or not, present or future — that does superlinear work makes its
//! operation trip here. The contracts double as documentation of what a keystroke
//! is *allowed* to cost.
//!
//! ## What it deliberately does NOT charge
//!
//! Batched position rebases (`Patch::map_many`, routing selection/decoration/fold
//! shifts) are a pure *bandwidth* class — offset arithmetic, not a semantic pass —
//! so they are unmetered; charging them would set a linear floor that masks an
//! *added* semantic O(n) cost. A batched `map_many` replaced by a per-item
//! `map_offset` loop still reappears on the meter (that call IS charged), so the
//! gate keeps catching an added semantic pass even while it ignores the bandwidth
//! floor.
//!
//! ## Scope
//!
//! This matrix covers the fold / bracket / decoration / caret-movement
//! interactions — the ones most prone to superlinear surprises. Two hot paths are
//! gated elsewhere and
//! intentionally out of scope here: incremental **highlighting** (its own 256-line
//! budget + checkpoint canaries) and the **widget draw** pass (the draw-budget
//! canary in `scrive-iced`).
//!
//! Budgets are loose on purpose (a doubling may ~2.6× for a linear op, must stay
//! ~flat for a constant one): the gate is a tripwire for *algorithmic* blow-ups
//! (a quadratic reads ~4×), not a micro-benchmark. Wall-clock lives in the
//! criterion benches.

use crate::{Diagnostic, Document, FindQuery, Motion, SelectionSet, Severity};

/// One foldable block: 3 lines, a `mark` token for find, `fn` for a caret target.
const BLOCK: &str = "fn f() {\n    mark = 0\n}\n";

fn block_len() -> u32 {
    BLOCK.len() as u32
}
fn block_start(i: usize) -> u32 {
    i as u32 * block_len()
}
/// The header line's end (just after the block's `{`) — a *visible* caret when
/// the block is collapsed, and inside the fold's span, so an edit there exercises
/// `expand_folds_touched` the way typing at the end of a collapsed header does.
fn header_end(i: usize) -> u32 {
    block_start(i) + BLOCK.find('{').unwrap() as u32 + 1
}
fn brace_off(i: usize) -> u32 {
    block_start(i) + BLOCK.find('{').unwrap() as u32
}

fn build(blocks: usize) -> Document {
    let mut s = String::with_capacity(blocks * BLOCK.len());
    for _ in 0..blocks {
        s.push_str(BLOCK);
    }
    Document::new(&s).unwrap()
}

/// Collapse the first `k` blocks, then warm the fold-map cache so a measurement
/// sees steady-state cost, not the one-time O(folds) rebuild.
fn fold_first(d: &mut Document, k: usize) {
    for i in 0..k {
        d.toggle_fold_opener(brace_off(i));
    }
    let _ = d.fold_map();
}
fn find_on(d: &mut Document) {
    d.set_find_query(Some(FindQuery { text: "mark".into(), case_sensitive: true }), 0);
}

/// Reset the meter, run `op`, return the semantic work it charged.
fn meter_of(d: &mut Document, op: impl FnOnce(&mut Document)) -> u64 {
    crate::perf::reset();
    op(d);
    crate::perf::meter()
}

enum Budget {
    /// Must not grow with this dimension.
    Constant,
    /// May grow ~linearly; a quadratic reads ~4× and trips.
    Linear,
}

#[track_caller]
fn assert_budget(name: &str, budget: Budget, small: u64, big: u64) {
    let ratio = big as f64 / small.max(1) as f64;
    eprintln!("[perf_gate] {name:<38} {small:>7} -> {big:>7}  ({ratio:.2}x)");
    let ok = match budget {
        Budget::Constant => big <= small + small / 4 + 256,
        Budget::Linear => big <= small * 13 / 5 + 256, // 2.6× ceiling; O(n²) ≈ 4×
    };
    assert!(ok, "{name}: meter {small} -> {big} ({ratio:.2}x) violates its {} budget",
        match budget { Budget::Constant => "O(1)", Budget::Linear => "O(n)" });
}

// ── Type at N carets, each on a folded block's header. Expanding the touched
//    folds must stay linear in the caret count; an O(carets²) walk reads ~4× ────
#[test]
fn multicaret_over_folds_is_linear() {
    const BLOCKS: usize = 700;
    let (s, b) = (150usize, 300usize);
    let cell = |k: usize| {
        let mut d = build(BLOCKS);
        fold_first(&mut d, k);
        let ranges: Vec<(u32, u32)> = (0..k).map(|i| (header_end(i), header_end(i))).collect();
        d.set_selections(SelectionSet::from_ranges(&ranges, 0));
        meter_of(&mut d, |d| d.insert_text("a"))
    };
    assert_budget("repro: type, N carets on N folds", Budget::Linear, cell(s), cell(b));
}

// ── DECORATIONS × carets: the mover must stay linear. A per-item `map_offset`
//    rebase would be O(decorations·edits) and read quadratic here ──────────────
#[test]
fn multicaret_over_decorations_is_linear() {
    let (s, b) = (150usize, 300usize);
    let cell = |k: usize| {
        let mut d = build(k); // k blocks ⇒ k `mark` matches ⇒ k decorations
        find_on(&mut d);
        let ranges: Vec<(u32, u32)> = (0..k).map(|i| (block_start(i), block_start(i))).collect();
        d.set_selections(SelectionSet::from_ranges(&ranges, 0));
        meter_of(&mut d, |d| d.insert_text("a"))
    };
    assert_budget("decos: type, N carets, N matches", Budget::Linear, cell(s), cell(b));
}

// ── Caret movement scales with caret count (linear), not quadratically ───────
#[test]
fn multicaret_movement_is_linear() {
    const BLOCKS: usize = 700;
    let (s, b) = (150usize, 300usize);
    let cell = |k: usize| {
        let mut d = build(BLOCKS);
        let ranges: Vec<(u32, u32)> = (0..k).map(|i| (block_start(i), block_start(i))).collect();
        d.set_selections(SelectionSet::from_ranges(&ranges, 0));
        meter_of(&mut d, |d| d.move_carets(Motion::Right, false))
    };
    assert_budget("carets: arrow, N carets", Budget::Linear, cell(s), cell(b));
}

// ── A single caret's move must not scale with document size ──────────────────
#[test]
fn single_caret_movement_is_size_independent() {
    let (s, b) = (400usize, 800usize);
    let cell = |blocks: usize| {
        let mut d = build(blocks);
        d.set_selections(SelectionSet::new(block_start(blocks / 2)));
        meter_of(&mut d, |d| d.move_carets(Motion::Right, false))
    };
    assert_budget("size: arrow, 1 caret", Budget::Constant, cell(s), cell(b));
}

// ── ALLOCATION dimension: typing at N carets over N folds must not allocate
//    `SumTree` nodes superlinearly. The semantic work meter is blind to
//    copy-on-write tree allocation, so this cell watches `NODE_ALLOCS` directly.
//    An O(N²) shape (a per-caret whole-tree rebuild, or a per-fold at()/split
//    loop) reads ~4× per-caret here; O(N·log) editing stays flat. ──────────────
#[test]
fn multicaret_over_folds_allocates_linearly() {
    use crate::sum_tree::NODE_ALLOCS;
    let (s, b) = (400usize, 1600usize); // 4× apart for a clean per-caret signal
    // Fold the first `k` blocks, put a caret at each folded header's visible end,
    // type once, and return node allocations PER caret.
    let per_caret = |k: usize| {
        let mut d = build(k);
        let opens: Vec<(u32, u32)> = (0..k).map(|i| (brace_off(i), brace_off(i))).collect();
        d.set_selections(SelectionSet::from_ranges(&opens, 0));
        d.fold_at_carets(false);
        d.move_carets(Motion::LineEnd, false);
        let _ = d.fold_map();
        NODE_ALLOCS.with(|c| c.set(0));
        d.insert_text("a");
        let _ = d.fold_map();
        NODE_ALLOCS.with(std::cell::Cell::get) as f64 / k as f64
    };
    let (ps, pb) = (per_caret(s), per_caret(b));
    eprintln!("[perf_gate] fold+type node_allocs/caret   {ps:>7.1} -> {pb:>7.1}  ({:.2}x)", pb / ps);
    // Per-caret is O(log doc) — it creeps up with the deeper tree but must stay far
    // below the 4× a quadratic would show; 1.8× leaves headroom for that log growth.
    assert!(pb <= ps * 1.8, "fold+type allocates superlinearly: {ps:.1} -> {pb:.1}/caret");
    // And an absolute ceiling. With the batched edit path the whole commit touches
    // only a few nodes per caret (~5 here): the buffer applies all N edits in ONE
    // spine rebuild and the views shift in bulk. Per-caret tree work — the batch
    // falling to sequential splices (~48/caret), or a view's bulk shift done as N
    // splices — lands in the tens; a per-edit split storm (O(log²)) or per-fold
    // at()/rebuild higher still. A ratio test can't see such a constant jump, only
    // this bound. 30 is ~6× the healthy value: ample margin for log growth, tight
    // enough to trip the moment a view goes per-caret.
    assert!(pb <= 30.0, "fold+type allocates {pb:.1} nodes/caret — a view went per-caret, not batched");
}

// ── A single edit that touches ONE fold must not scan the whole document: it
//    windows to the local fold, so it is flat in document size ────────────────
#[test]
fn single_fold_edit_is_size_independent() {
    let (s, b) = (400usize, 800usize);
    let cell = |blocks: usize| {
        let mut d = build(blocks);
        fold_first(&mut d, 1); // fold block 0 only
        d.set_selections(SelectionSet::new(header_end(0)));
        meter_of(&mut d, |d| d.insert_text("a"))
    };
    assert_budget("size: type on 1 fold", Budget::Constant, cell(s), cell(b));
}

// ── AUTO-CLOSE × decorations. The ≤C auto-close provenance set lives in its own
//    store, so `validate_autoclose` (every commit) and `clear_autoclose` (every
//    arrow key) are O(C) — flat in the bulk diagnostic count — rather than an
//    O(decorations) whole-store scan. The diagnostics are packed at the FRONT, far
//    from the pair/edit at EOF, so the windowed commit mover never charges for
//    them; the only diagnostic-count signal these cells can pick up is the
//    autoclose validate/clear scan. ───────────────────────────────────────────

/// Build a fixed-size doc, arm ONE auto-close pair at EOF, and publish `k` plain
/// diagnostics packed at the front. The buffer size is INDEPENDENT of `k`, so the
/// document-scale constant work (offset_to_point, bracket shift, the windowed
/// mover) is identical across the two decoration-count cells — only the diagnostic
/// count varies, isolating the autoclose scan.
fn armed_with_diags(k: usize) -> Document {
    use crate::{Diagnostic, Severity};
    // 6000 lines of "a\n" = 12k bytes: room for k diagnostics at offset 2*i, well
    // clear of the pair region at EOF, for k up to a few thousand.
    let mut s = String::with_capacity(12_000);
    for _ in 0..6000 {
        s.push_str("a\n");
    }
    let mut d = Document::new(&s).unwrap();
    let end = d.text().len() as u32;
    d.set_selections(SelectionSet::new(end));
    d.type_char('('); // arm one pair at EOF (auto-close fires: the char after the caret is EOL)
    let diags: Vec<Diagnostic> =
        (0..k as u32).map(|i| Diagnostic::new(i * 2..i * 2 + 1, Severity::Warning, "d")).collect();
    d.set_diagnostics(d.revision(), diags);
    d
}

// Typing a plain char while a pair is armed runs `validate_autoclose` on commit.
// It hits only the ≤C own store, not the whole `decorations` store, so the
// keystroke is flat in the diagnostic count.
#[test]
fn plain_type_with_pair_armed_is_decoration_count_independent() {
    let (s, b) = (1000usize, 2000usize);
    let cell = |k: usize| {
        let mut d = armed_with_diags(k);
        meter_of(&mut d, |d| d.type_char('x')) // a plain char INSIDE the armed pair
    };
    assert_budget("autoclose: type, pair armed, k diags", Budget::Constant, cell(s), cell(b));
}

// Every caret move runs `reset_gesture`→`clear_autoclose`, which empties the ≤C
// own store rather than scanning the whole `decorations` store, so the move is
// flat in the diagnostic count.
#[test]
fn move_caret_with_pair_armed_is_decoration_count_independent() {
    let (s, b) = (1000usize, 2000usize);
    let cell = |k: usize| {
        let mut d = armed_with_diags(k);
        meter_of(&mut d, |d| d.move_carets(Motion::Right, false))
    };
    assert_budget("autoclose: arrow, pair armed, k diags", Budget::Constant, cell(s), cell(b));
}

// Visit canary: `autoclose_ranges()` materializes only the ≤C own store, so it
// VISITS O(C) items at any diagnostic count. The work meter is blind here (a
// `to_vec` is unmetered by design — it is the mover's band path), so this reads the
// `DECORATION_VISITS` canary directly: iterating the own store is 1 visit
// regardless of how many diagnostics the bulk `decorations` store holds.
#[test]
fn autoclose_ranges_is_pair_count_independent() {
    use crate::decorations::DECORATION_VISITS;
    let visits_for = |k: usize| -> u64 {
        let d = armed_with_diags(k);
        let base = DECORATION_VISITS.with(std::cell::Cell::get);
        let _ = d.autoclose_ranges();
        DECORATION_VISITS.with(std::cell::Cell::get) - base
    };
    let (s, b) = (visits_for(1000), visits_for(2000));
    eprintln!("[perf_gate] autoclose_ranges visits         {s:>7} -> {b:>7}");
    assert!(
        b <= s + 4,
        "autoclose_ranges visited {s} -> {b}: it scanned the bulk decoration store, not the ≤C own store",
    );
}

// ── FIND. The store IS the match set, so count / navigate / N-of-M and the
//    per-commit active-presence check are O(1)/O(log) in the match count, never a
//    `decorations_in(0..MAX)` / `decoration_range` whole-store walk.
//    Each `build(k)` block carries one `mark` match, so `k` blocks ⇒ `k` finds; the
//    document-scale per-op cost is proven flat by the `*_size_independent` cells
//    above, so the only match-count signal these cells pick up is the find walk. ──

/// Build a `k`-block document with the "mark" find query active (⇒ `k` matches).
fn find_k(k: usize) -> Document {
    let mut d = build(k);
    find_on(&mut d);
    d
}

// `find_match_count()` is the O(1) root-summary `find_count`, not an O(M) walk.
#[test]
fn find_count_is_match_count_independent() {
    let (s, b) = (1000usize, 2000usize);
    let cell = |k: usize| {
        let mut d = find_k(k);
        meter_of(&mut d, |d| {
            let _ = d.find_match_count();
        })
    };
    assert_budget("find: count, k matches", Budget::Constant, cell(s), cell(b));
}

// Navigating to the next match is three O(log) store queries (`find_count` /
// `find_rank_before` / `nth_find`), not a whole-store `live` collect.
#[test]
fn find_navigate_is_match_count_independent() {
    let (s, b) = (1000usize, 2000usize);
    let cell = |k: usize| {
        let mut d = find_k(k);
        d.set_selections(SelectionSet::new(0));
        meter_of(&mut d, |d| {
            d.find_next(0); // picks match 0 regardless of k
        })
    };
    assert_budget("find: navigate, k matches", Budget::Constant, cell(s), cell(b));
}

// Typing a plain char while a match is parked-active runs the per-commit presence
// survivor, which probes the tracked start (`decorations_in(s..s)`) = O(log), not
// a whole-store `decoration_range` scan. The caret is at EOF, far from the active
// match at block 0, so the windowed find repair never touches it — the only
// match-count signal is the presence check.
#[test]
fn plain_type_with_active_match_is_match_count_independent() {
    let (s, b) = (1000usize, 2000usize);
    let cell = |k: usize| {
        let mut d = find_k(k);
        d.set_selections(SelectionSet::new(0));
        d.find_next(0); // activate match 0 (block 0, near the document start)
        let end = d.text().len() as u32;
        d.set_selections(SelectionSet::new(end)); // park the caret far from it
        meter_of(&mut d, |d| d.type_char('x'))
    };
    assert_budget("find: type, active match, k matches", Budget::Constant, cell(s), cell(b));
}

// ── SCROLLBAR OVERVIEW. The overview reduce (`overview_marks`) folds the two
//    lanes per FIXED track-pixel bucket in O(P + log M), never walking every
//    diagnostic per frame. The bucket count is held constant (200) so the only
//    signal is the diagnostic count `k`; a `diagnostics_in(0..MAX)` whole-store
//    scan would be O(M) — charged per hit — and read ~2× across a 2× diagnostic
//    set. The summary reduce (`bucketed_reduce` + `first_start_with_severity` +
//    `find_rank_before`/`nth_find`) charges nothing, so the meter is flat. ──────
#[test]
fn diagnostic_overview_is_diag_count_independent() {
    // 201 ascending bounds ⇒ 200 buckets, fixed regardless of k.
    let bounds: Vec<u32> = (0..=200u32).map(|i| i * 32).collect();
    let (s, b) = (1000usize, 2000usize);
    let cell = |k: usize| {
        let mut d = build(k); // k blocks ⇒ room for k diagnostics at distinct offsets
        let rev = d.revision();
        let diags: Vec<Diagnostic> = (0..k)
            .map(|i| Diagnostic::new(block_start(i)..block_start(i) + 1, Severity::Warning, "w"))
            .collect();
        d.set_diagnostics(rev, diags);
        let mut sev = Vec::new();
        let mut find = Vec::new();
        meter_of(&mut d, |d| d.overview_marks(&bounds, &mut sev, &mut find))
    };
    assert_budget("overview: marks, k diagnostics", Budget::Constant, cell(s), cell(b));
}
