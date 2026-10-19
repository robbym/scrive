# Large-documents milestone — bench ledger

`cargo bench -p scrive-core --bench perf`. Medians, same machine, same
generated corpus (`support::gen_doc`, seed 7). Baselines are PRE-milestone
(commit `95fa8a6` + bench scaffolding only).

## Baselines (2026-07-11, pre-fix)

| bench                                   | 100 KB   | 1 MB     |
|-----------------------------------------|----------|----------|
| keystroke/insert_mid                    | 112.6 µs | 1.447 ms |
| keystroke/enter_mid                     | 106.3 µs | 1.390 ms |
| keystroke/undo_step                     | 107.4 µs | 1.252 ms |
| find/set_query                          | 1.774 ms | 43.59 ms |
| find/keystroke_while_active             | 2.397 ms | 64.96 ms |
| find/navigate_big_set (1 MB, 10k cap)   | —        | 16.38 ms |
| highlight/tokenize_viewport_after_edit  | 515 µs   | 11.68 ms |
| highlight/keystroke_deep_dirty_tail     | 176.6 µs | 2.284 ms |
| query/diagnostics_in_window_10k (1 MB)  | —        | 3.61 µs  |
| query/occurrences_viewport (1 MB)       | —        | 4.20 µs  |
| brackets/all_len (placeholder)          | —        | ~0 ns    |

Reading the shape: every keystroke-path bench scales ~linearly with document
size (the §17-forbidden O(file) passes: bracket re-match, decoration churn,
highlight invalid-set rebuild), find-while-active adds the full re-scan +
O(matches²) store churn on top, and a single `find_next` pays 16 ms of linear
handle re-resolution against a 10k match set. The two `query/*` benches are
the already-windowed floor the rest of the milestone aims for.

10 MB rows join after LD-1 (the load cap blocked them here); 100 MB behind
`SCRIVE_BENCH_HUGE=1`. Post-milestone numbers land in the LD-8 section below.

## LD-8 results (2026-07-11, post-milestone — medians)

| bench                                   | 100 KB base → now  | 1 MB base → now     | 10 MB (new) |
|-----------------------------------------|--------------------|---------------------|-------------|
| keystroke/insert_mid                    | 112.6 µs → 50.8 µs | 1.447 ms → 634 µs   | 10.3 ms     |
| keystroke/enter_mid                     | 106.3 µs → 41.1 µs | 1.390 ms → 617 µs   | 10.6 ms     |
| keystroke/undo_step                     | 107.4 µs → 40.0 µs | 1.252 ms → 472 µs   | 7.4 ms      |
| find/set_query                          | 1.774 ms → 75.7 µs | 43.59 ms → 782 µs   | 5.9 ms      |
| find/keystroke_while_active             | 2.397 ms → 55.2 µs | 64.96 ms → 715 µs   | 9.7 ms      |
| find/navigate_big_set (10k live)        | —                  | 16.38 ms → 17.3 µs  | —           |
| highlight/tokenize_viewport_after_edit  | 515 µs → 510 µs    | 11.68 ms → 1.11 ms  | 8.7 ms      |
| highlight/keystroke_deep_dirty_tail     | 176.6 µs → 65.3 µs | 2.284 ms → 885 µs   | 13.7 ms     |
| query/diagnostics_in_window_10k (1 MB)  | —                  | 3.61 µs → 45 ns     | —           |
| query/occurrences_viewport (1 MB)       | —                  | 4.20 µs → 4.31 µs   | —           |

Class moves confirmed: `find/keystroke_while_active` and `navigate_big_set`
dropped from O(document)+O(matches²) to O(edit window)+O(matches); the
highlight deep-dirty-tail commit lost its per-keystroke dirty-set rebuild;
`diagnostics_in` windowed queries went binary-search. `set_query` remains
O(document) BY DESIGN (you cannot count matches without scanning) — memmem
made it ~50× cheaper.

**Residual, honestly (as of LD-8 — superseded by the rope stage-2 section
below):** the keystroke path still scales with document size in the
*bandwidth* class — the String splice tail memmove (rope stage 2 lifts
it), the bracket suffix `+= delta` shift, and the bracket seed's O(prefix
brackets) forward filter (~320k entries at 10 MB on this bracket-dense
corpus; the design's deferred reverse pair-jump seed walk is the named next
optimization if 10 ms/keystroke at 10 MB matters). No *semantic* O(document)
pass remains on the keystroke or frame paths.

## Rope stage 2 (2026-07-11) — ropey backing swap, medians vs LD-8

Same machine/corpus. Reference column = the LD-8 table above (String backing).
`snapshot/mint` is new: it pins the stage-2 payoff (`Document::snapshot` was
an O(len) `Arc<str>` copy; it is now a rope clone).

| bench                                   | 100 KB LD-8 → rope | 1 MB LD-8 → rope  | 10 MB LD-8 → rope |
|-----------------------------------------|--------------------|-------------------|-------------------|
| keystroke/insert_mid                    | 50.8 µs → 55.2 µs  | 634 µs → 502 µs   | 10.3 ms → 6.0 ms  |
| keystroke/enter_mid                     | 41.1 µs → 41.7 µs  | 617 µs → 487 µs   | 10.6 ms → 6.2 ms  |
| keystroke/undo_step                     | 40.0 µs → 42.7 µs  | 472 µs → 501 µs   | 7.4 ms → 5.8 ms   |
| find/set_query                          | 75.7 µs → 91.0 µs  | 782 µs → 895 µs   | 5.9 ms → 6.2 ms   |
| find/keystroke_while_active             | 55.2 µs → 57.5 µs  | 715 µs → 592 µs   | 9.7 ms → 6.3 ms   |
| find/navigate_big_set (1 MB)            | —                  | 17.3 µs → 19.3 µs | —                 |
| highlight/tokenize_viewport_after_edit  | 510 µs → 621 µs    | 1.11 ms → 1.19 ms | 8.7 ms → 7.9 ms   |
| highlight/keystroke_deep_dirty_tail     | 65.3 µs → 62.5 µs  | 885 µs → 647 µs   | 13.7 ms → 8.0 ms  |
| query/diagnostics_in_window_10k (1 MB)  | —                  | 45 ns → 47 ns     | —                 |
| query/occurrences_viewport (1 MB)       | —                  | 4.31 µs → 18.8 µs | —                 |
| snapshot/mint (NEW — O(1) pin)          | 8.1 ns             | 8.2 ns            | 8.4 ns            |

What moved: `snapshot/mint` is **flat ~8 ns across two orders of magnitude**
of document size — the O(1) claim, proven. The keystroke path lost the splice
tail memmove (10 MB insert 10.3 → 6.0 ms; the survivor is the bracket suffix
shift + seed filter, unchanged by this swap and still the named next
optimization). `find/set_query` also stopped materializing the document: the
rescan now runs `scan_buffer` (64 KB slice windows, needle−1 overlap,
byte-identical to the whole-text scan by test) so a capped dense query is
O(bytes-until-cap) again and peak transient memory is one window.

**Residual, honestly:** per-lookup constants went from O(1) pointer math to
O(log chunks) tree walks, and a `Cow` crossing rope chunks (~1 KB leaves) is
now an `Owned` copy — visible as +6–20% on `set_query`'s scan constant and
4.3 → 18.8 µs on the per-frame `occurrences_viewport` query (a ~4 KB window
crosses chunks; still O(viewport), ~0.1% of a 60 fps frame budget). A
chunk-crossing `line()` read is an O(line) copy — for ordinary sub-KB lines
it stays borrowed, but every line ≥ ~1 KB pays per read on the draw path
(same class as the text-shaping cost that already dominates such lines;
a `line_prefix`/threaded-Cow scheme is the designed lift if a field gate
ever fails on long-line files). The bracket suffix shift + seed filter is
now the only bandwidth-class keystroke resident.

## Highlight virtualization (2026-07-11) — retention goes windowed

Field finding: RAM grew O(document) as the idle sweep tokenized (dense
per-line spans + boxed end states — hundreds of MB fully swept at 10 MB).
Retention became viewport-window (±512 rows, spans + states) + sparse
checkpoints (1 per 256 rows): fully swept ≈ **O(window + lines/256)** —
single-digit MB at any size, pinned by the retention canary
(`virtualized_sweep_retains_bounded_rows`: dense kept 10,000 rows; the bound
is 1,088). Correctness pinned by a randomized edits+window-moves oracle under
a stateful grammar. Wall-clock (medians, vs the rope stage-2 rows above):

| bench                                   | 100 KB      | 1 MB              | 10 MB           |
|-----------------------------------------|-------------|-------------------|-----------------|
| highlight/tokenize_viewport_after_edit  | 621→559 µs  | 1.19→1.01 ms      | 7.9→5.8 ms      |
| highlight/keystroke_deep_dirty_tail     | 62.5→47 µs  | 885 µs*→464 µs    | 8.0→5.5 ms      |
| highlight/cold_jump_1m (NEW)            | —           | 3.3 ms            | —               |

(*rope-stage-2 column; the commit path now splices a ~1k-row window +
~lines/256 checkpoints instead of two document-length Vecs — faster, not
slower.) `cold_jump_1m` = re-aim the window mid-document on a fully swept
1 MB doc + refill from the nearest checkpoint (~1,100 rows); the app pays it
budget-paced (≤256 lines/frame), rendering fallback-then-fill exactly like
the LD-7 load behavior. The trade, honestly: an out-of-window edit converges
at checkpoint grain (≤ a stride more re-tokenization than dense, test-pinned),
a cold jump costs a few frames of fill, and a viewport straddling a collapsed
fold hiding more than `HIGHLIGHT_MAX_WINDOW_ROWS` = 4096 rows renders its tail
fallback-styled until scrolled to (the cap that keeps a mega-fold from
re-growing retention to O(document)) — for a RAM envelope that stopped
scaling with the document. Adversarially reviewed: 4 findings (1 high — stale
off-grid checkpoints poisoning cold-jump warm-ups) all fixed with fails-first
regression tests.

## Speculative + parallel highlighting (2026-07-11) — first paint + throughput

Field problem: even with retention virtualized, a jump to the bottom of the
`--large 100` corpus (~4.5M lines) renders fallback for ~16-20 s of
single-core tokenization (the top-down state chain). The core gained a pure,
sync, `Send`-able **segment tokenizer** (`tokenize_segment` over an O(1)
`Snapshot`); the `scratch` app runs it on a worker pool, colours the viewport
immediately from a GUESSED fresh state (speculative), and verifies the whole
document left-to-right by stitching segment boundaries with the §9 convergence
rule (`SegmentBoundary` equality). Results feed back through
`Document::absorb_highlight` (verified ⇒ clears dirt + merges checkpoints;
speculative ⇒ shows spans on dirty rows, plants no checkpoints).

Criterion (the core primitives; the sweep's wall-clock is a field number):

| bench                          | median  | note                                       |
|--------------------------------|---------|--------------------------------------------|
| highlight/segment_tokenize_64k | 83.7 ms | one 64k-line segment, cold, single core (~1.3 µs/line) |
| highlight/stitch_repair_64k    | 1.76 ms | wrong-guess re-run (comment closes ~500 rows in) + early-stop splice — vs the 84 ms a full re-tokenize would cost |

Field numbers (`scratch`, release, headless self-test harness — driven then
reverted — running the real pool + threads with a MID-SWEEP jump to the bottom
(the review's moved-window case), then comparing the swept viewport to a
synchronous reference; machine = this dev box, ~8 cores):

| corpus            | bottom paints after jump | full verified sweep | viewport correctness |
|-------------------|--------------------------|---------------------|----------------------|
| 20 MB / 0.9M lines  | next frame             | ~0.6 s              | 50/50 rows           |
| 100 MB / 4.5M lines | next frame (~51 ms)    | 2.87 s              | 50/50 rows           |

So the field gate is met: the viewport colours on the next frame, the whole
document is verified in single-digit seconds, and the verified colours are
byte-identical to the synchronous path. Design notes worth keeping: (1) the
viewport is **painted synchronously on the UI thread** (one window of rows is
a few ms) rather than dispatched to a worker — so its paint never queues
behind the bulk sweep, even for a jump mid-sweep; workers do only the bulk
chain. (2) Wrong-guess re-runs also run **synchronously in `poll`**: syntect's
`first_line` flag makes a mid-document Fresh guess compare unequal to the true
state even in ground context, so *every* segment technically re-runs — but the
early-stop re-converges at the first checkpoint (~one stride), so a re-run is
~0.3 ms and the chain advances as fast as the parallel Fresh results arrive
instead of queueing tiny re-runs behind the bulk work.

Adversarially reviewed (3 lenses, Opus): all three confirmed findings fixed —
(HIGH) a converging re-run dispatched with a moved viewport's `spans_for` while
`converge_against` carried the old window violated the splice precondition
(debug panic → chain stall; release → misaligned clean spans); fixed by having
the core FORCE the window to the old segment's when converging (the passed
`spans_for` is ignored), pinned by `converging_rerun_ignores_a_moved_spans_for`
(fails-first: panics against the old code). (MEDIUM) restart didn't drain
stale queued jobs — fixed by clearing the queue + a shared atomic revision
workers check before tokenizing.

**Residual, honestly:** a segment cut inside a construct that never closes
(one giant string/comment) makes every boundary mismatch and the stitch
serialize to a sequential pass — the inherent worst case of the top-down
grammar model, same class as the bracket engine's unbalanced-opener EOF
replay. Verified absorb of a spans-less segment covering a viewport the sweep
passes *mid-run* (a middle scroll position, not the held bottom) evicts the
window there (→ fallback) until the sync phase-2 refill repaints it from
correct checkpoints. Editing at 100 MB re-sweeps the whole document off-thread
(~3 s of background CPU per edit burst); the UI never stalls (the synchronous
viewport paint keeps it coloured), but the churn is real — a
from-the-dirty-frontier re-dispatch is the designed trim, deferred until the
field gate asks. The checkpoint store is unchanged from the virtualization
section (O(lines/256)); the parallel sweep plants the same checkpoints the
sync walk would.

## F3 — sparse-checkpoint delta-gap SumTree (2026-07-15)

The last flat-Vec position mover. `Retention.checkpoints` was a
`Vec<(u32, Box<LineState>)>` whose `shift_checkpoints` did `std::mem::take` +
rebuild on EVERY commit and every undo step (`on_commit_patch`) = O(N / 256).
It now rides the same delta-gap `SumTree` as decorations/folds/brackets: reads
are an O(log) seek (`state_at`/`floor`), a single-edit shift re-anchors ONE
seam gap (O(edit + log) + one heavy `LineState` clone), and only a scattered
MULTI-edit commit falls back to the O(#checkpoints) walk (kept verbatim as the
correctness-first path).

Bench group **`checkpoints`** (`enter_top_swept_{100k,1m}`): an Enter at the
document top of a fully swept cache shifts every checkpoint down one row —
K ≈ lines/256, so the 1M case shifts ~3.9k checkpoints. Pre-fix that Vec
rebuild grew O(K); post-fix it is O(log K), so `enter_mid` at the top no longer
scales with the swept-checkpoint count. (Op-count proof: the `perf_gate`-style
`checkpoint_shift_is_checkpoint_count_independent` cell charges a flat 1 post-fix
vs K → 2K when forced down the walk fallback. Behaviour is pinned by
`checkpoint_tree_equals_the_vec_reference_under_random_edits`, an edit-for-edit
oracle against the retained flat-Vec algorithm; `--capture` PNGs are unaffected —
checkpoints are internal state cache.)

## Rope stage 3 (2026-07-17) — inline chunks + seam coalescing

Two changes to the leaf chunk, benched together because they are entangled:
inline storage is only safe once fill is bounded.

- `Chunk(String)` → `Chunk(ArrayString<CHUNK_MAX>)`. The bytes live in the leaf
  rather than behind a pointer, so a chunk costs no allocation of its own
  (~78k of them per 10 MB on load) and a copy-on-write leaf clone is a flat
  memcpy instead of a walk of scattered pointers. Same choice zed's rope makes,
  at the same 128-byte chunk size.
- `Rope::replace` coalesces. It re-chunks a *window* around the edit instead of
  splicing at the endpoints, snapping the window to chunk boundaries (splitting
  mid-chunk would create the very fragment it removes) and walking it past any
  sub-`CHUNK_MIN` neighbour (a seam is one byte longer than the window it
  replaced, so its remainder otherwise lands just outside the next edit's window
  and escapes forever — one per keystroke).

**These numbers are NOT comparable to the sections above**: different machine,
and the workspace `[profile.release]` / `[profile.bench]` overrides those rows
were taken under have since been dropped. Only the before→after delta *within*
this section is meaningful. Both columns were measured back-to-back on the same
box, and an identical-code control run first established a **±5–9% noise floor**
(an earlier attempt was thrown away entirely: the box was drifting so hard that
the same code measured 174 µs → 234 µs across two runs, and criterion reported
the drift as a confident p=0.00 regression).

| bench                | 100 KB       | 1 MB           | 10 MB          |
|----------------------|--------------|----------------|----------------|
| keystroke/insert_mid | 167 → 158 µs | 1.24 → 1.16 ms | 27.5 → 25.3 ms |
| keystroke/enter_mid  | 146 → 141 µs | 1.20 → 1.18 ms | 26.2 → 23.6 ms |
| keystroke/undo_step  | 118 → 112 µs | 1.13 → 1.09 ms | 25.7 → 23.6 ms |
| rope/load (NEW)      | 407 → 389 µs | 4.44 → 4.26 ms | 77.1 → 69.6 ms |

A net win everywhere, growing with document size. Read it honestly: only the
10 MB column (keystroke −9…−12%, load −11%) clears the noise floor; the 100 KB
and 1 MB deltas (−3…−7%) sit inside it and mean "not a regression", not a
measured gain. The direction matters more than the magnitude — coalescing adds
per-edit copying and still comes out ahead, because the shallower tree pays for
it.

`rope/load` is new here: the load path is one chunk construction per 128 bytes,
which isolates the chunk's storage from everything the keystroke benches mix in.

The anti-fragmentation result is an op count, so it lives in a unit test
(`chunk_count_stays_proportional_to_bytes_not_edits`) rather than in this file:
200 keystrokes at one caret took a 61-chunk rope to **262 chunks** before — the
sizes at the caret reading `[1, 1, 1, …]`, one chunk per character, growth in
*keystrokes* rather than bytes — and to **64** after, at 98% average fill.

**Residual, honestly:** `replace` now copies a ~384-byte seam per edit that it
did not before. That shows up nowhere above, but it is real work on the
keystroke path. The undersized-neighbour walk is unbounded by construction: a
step or two once the invariant holds, longer on an already-fragmented rope —
which is the direction the cost should run, since it heals as it goes. And the
benches cannot see the fragmented steady state at all (`iter_batched` hands
every iteration a pristine rope from `from_str`), so the gain from a shallower
tree during a *real* editing session is not measured here — only the canary
speaks to it, and it is the larger of the two effects.

## Brackets — comment/string awareness (`SkipContext`)

Optional comment/string-aware bracket matching (`BracketConfig` + the line-local
`SkipContext` adapter): brackets inside line comments / strings / char literals
are no longer matched, coloured, folded, or indent-guided. Opt-in and zero-cost
off; block comments / multi-line strings deliberately out of scope (non-local
state would break the O(edit) bound).

`brackets/match_text` vs `brackets/match_text_aware` (1 MB corpus, `//` + `"`):

| | median | vs plain |
|---|---|---|
| `match_text` (structural byte scan) | ~3.3 ms | 1.0× |
| `match_text_aware` (`SkipContext`)  | ~5.2 ms | ~1.6× |

Load stays O(n): a ~1.6× constant — the state machine plus a first-byte-guarded
line-comment `starts_with` (the guard cut it from ~2.0×). ~160 ms extrapolated to
a 1.4M-line file, a rounding error next to a full syntect tokenize (~9.8 s) —
which is the whole reason the scan is a self-contained lexer, not a ride on the
lazy highlight layer.

**Per-edit** stays O(edit): a keystroke widens to its line (skip-state is
line-local, so `SkipContext` restarts fresh at the boundary), one line-region
spliced in O(log n) — the same document-size scaling as the structural path.
Correctness is the `incremental_matches_scratch_with_comment_awareness` oracle
(600 random single- and multi-edit commits over comment/string content), deep-
equal to a scratch `match_text_with` after every commit.

**Residual, honestly:** the config-active path cannot take the >64-edit
structure-neutral *bulk-shift* fast path — with awareness a non-bracket byte
(`/`, `"`) can change the bracket set, so that path's core assumption fails.
Document-scale multi-caret typing WITH a config set is therefore O(carets·log²)
per commit rather than bulk O(brackets): bounded by caret count (the edit
dimension), not document size — no doc-scaling regression — but a real constant
on that one scenario. Not benched here; a `keystroke_aware` cell would pin it.

## Find — Unicode case folding in literal mode

A case-insensitive literal now folds exactly as regex mode does (Unicode simple
folding: `é`/`É`, `σ`/`Σ`/`ς`, `k`/`K` KELVIN SIGN), by matching through the
regex crate's `(?i)` over the needle's escape instead of the old ASCII-only
`memchr2` + `eq_ignore_ascii_case` scan. Folded matches can be a different byte
length than the needle (`K` is 3 bytes), so the literal path's fixed `k`
became a bound: the per-char `fold_width` (exact for ASCII, 4 beyond it).

`find` group, medians, pre → post (`FindQuery::new("return")`, case-insensitive):

| | 100 KB | 1 MB | 10 MB |
|---|---|---|---|
| `find/set_query`                | 209 µs → 410 µs  | 1.46 ms → 2.15 ms | 31.3 ms → 22.4 ms |
| `find/keystroke_while_active`   | 162 µs → 158 µs  | 1.12 ms → 1.07 ms | 21.9 ms → 21.1 ms |
| `find/navigate_big_set` (1 MB)  | —                | 1.07 µs → 0.97 µs | —                 |

No class change anywhere: the scan is still O(bytes-until-cap), the repair
still O(edit window). The keystroke row is flat **because** the bound is
tight for ASCII: a first cut at a flat 4-bytes-per-char bound measured +50% per
keystroke on this corpus (dense `return` hits — a 6× wider repair window tears
out and re-inserts matches it did not need to). The 10 MB `set_query` drop has
no mechanism in this change; read it as run-to-run noise (other builds were
running on the machine throughout both runs).

**Residual, honestly:** `set_query` pays two new constants. (1) Compiling
`(?i)<needle>` — about 40–200 µs per query change in isolation (more for a
needle with `s`/`k`, whose classes gain the 2–3-byte partners), so a find-bar
keystroke on a small document is now compile-bound rather than scan-bound. (2)
About 50–100 ns more per dense hit than the byte compare it replaces (1 MB,
10k hits: ~0.86 → ~1.3 ms isolated). Both are per query change, never per
document keystroke. An exact ASCII fast path (an ASCII needle without `k`/`s`
has only ASCII fold partners — the fact `fold_width` already rests on, pinned
exhaustively) would win them back at the price of a second matcher; not taken,
so folding has one owner.

## Core-bug fixes — host clipping, whole-word find, undo/redo (2026-10-19)

The hot paths this run changed that had no numbers yet: `Buffer::point_to_offset`
(host clipping added a char-boundary snap), the whole-word find engine (the
`\b…\b` wrap became a per-match edge check), and undo/redo (now restoring the
unit's whole selection set; the review fix then made the restore's caret
ejection read the memoized fold map instead of building a throwaway one). New
cells and one new group cover them: `find/{set_query,keystroke}_whole_word`,
`keystroke/{undo,redo}_multicaret_folded_1m`, and `position/point_to_offset`.

Master (`d56da01`, exported with this bench file) and this branch, back-to-back
on the same box, medians. Other worktrees were building during the runs; an
identical-code rerun of the folded undo/redo cells moved 3–7%, so read anything
under ~7% as noise (the stage-3 section's ±5–9% floor).

| bench | 100 KB | 1 MB | 10 MB |
|---|---|---|---|
| `position/point_to_offset` (NEW, 1,024 calls) | 154 → 113 µs | 170 → 130 µs | 205 → 163 µs |
| `find/set_query_whole_word` (NEW) | 1.70 → 1.43 ms | 9.70 → 8.83 ms | 30.4 → 29.5 ms |
| `find/keystroke_whole_word` (NEW) | 182 → 180 µs | 1.25 → 1.10 ms | 21.8 → 20.9 ms |
| `keystroke/undo_step` | 106 → 107 µs | 954 → 927 µs | 20.7 → 21.8 ms |
| `keystroke/insert_mid` | 154 → 148 µs | 994 → 982 µs | 22.5 → 23.1 ms |
| `find/set_query` (literal, folding) | 153 → 310 µs | 1.34 → 2.02 ms | 21.6 → 22.6 ms |

- **`point_to_offset`: faster, not slower.** The snap costs one chunk descent,
  and `Rope::point_to_offset` dropped a redundant line seek in the same change
  (rope.rs: "no net descent"), so the door with the clip in it now runs
  −20…27% (≈150 → 110 ns a call at 100 KB). No class change: O(log chunks).
- **Whole-word find: no class change.** The query-change scan is still
  O(document) and line-scoped, the keystroke repair still O(the touched lines);
  the 10 MB keystroke row sits on the same ~21 ms floor as every other 10 MB
  keystroke cell, plain typing included, so that floor is not the repair's.
  `set_query` came out faster at 100 KB / 1 MB (−16% / −9%) and level at 10 MB
  — plausibly the engine no longer evaluating `\b` assertions, not measured
  separately.
- **Single-caret undo: flat.** The selection restore's per-unit snapshot is a
  one-selection clone.
- The literal `set_query` rise is the case folding's `(?i)` compile and fold,
  recorded in its own section above — reproduced here, not new.

Undo/redo of a multi-caret unit over a document-scale fold set (1 MB: 4,352
collapsed `fn` bodies and a caret on each block's comment line, one char typed;
the measured step ends with the frame's `fold_map` read). Three trees, one
back-to-back run: master; the integration tip with the selection restore but
before the review fix (`0629b75`); this branch:

| bench | master | selection restore, pre-fix | branch |
|---|---|---|---|
| `keystroke/undo_multicaret_folded_1m` (NEW) | 12.0 ms | 14.0 ms | 12.6 ms |
| `keystroke/redo_multicaret_folded_1m` (NEW) | 11.5 ms | 14.0 ms | 12.1 ms |

The selection restore alone cost ~2 ms here: restoring 4,352 carets instead of
one, plus ejecting them through a throwaway `FoldMap::new` that the next frame's
`fold_map` then rebuilt — two O(folds) builds per undo/redo. Routing the
ejection through the memo (the review fix) makes that build the frame's,
recovering ~1.4–1.9 ms; what remains over master (~0.5 ms, inside the noise) is
the restore itself. The
complexity class is pinned by op count, not here: `perf_gate`'s
`undo_redo_over_folds_and_carets_is_linear` (N carets over N folds, 2.00×) and
`undo_redo_is_size_independent` (1.00×), and `undo_and_redo_build_the_fold_map_once`
counts the builds.

**Residual, honestly:** undo and redo stay O(folds) per press — the whole-set
`reconcile_folds` bumps the fold generation unconditionally, so one rebuild is
the floor; shifting the memo through the replayed patches (as the forward edit
path does) would need reconcile to bump only when it drops a fold. And history
now holds two selection sets per undo unit: memory proportional to carets ×
units, cloned only when an edit opens a unit (a merged typing run clones
nothing). The replace path is a discrete action and is not benched; the review
fix removed a pattern compile from every `replace_next` / `replace_all` press
(40–200 µs per compile, per the Unicode case-folding section).
