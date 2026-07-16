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
