//! Wall-clock benchmarks for large-document editing. Everything drives
//! `Document`'s PUBLIC API only, so the numbers stay comparable across internal
//! refactors — the point is to track the cost of each hot operation as
//! implementations change, recorded in `benches/LEDGER.md`.
//!
//! Run: `cargo bench -p scrive-core` (add `SCRIVE_BENCH_HUGE=1` for 100 MB).

mod support;

use std::time::Duration;

use criterion::{black_box, criterion_group, criterion_main, BatchSize, Criterion};
use scrive_core::{Document, FindQuery, SelectionSet};

/// A document with a single caret at `caret`.
fn doc_at(text: &str, caret: u32) -> Document {
    let mut d = Document::new(text).expect("bench doc loads");
    d.set_selections(SelectionSet::new(caret));
    d
}

/// Mid-document offset snapped to a line start (the generator is ASCII, so
/// any boundary works; a line start keeps Enter/indent benches meaningful).
fn mid_line_start(doc: &Document, text: &str) -> u32 {
    let mid = (text.len() / 2) as u32;
    let row = doc.buffer().offset_to_point(mid).row;
    doc.buffer().point_to_offset(scrive_core::Point::new(row, 0))
}

fn keystroke(c: &mut Criterion) {
    let mut g = c.benchmark_group("keystroke");
    g.sample_size(10).measurement_time(Duration::from_secs(3));
    for (label, text) in support::sized() {
        let probe = doc_at(&text, 0);
        let mid = mid_line_start(&probe, &text);
        drop(probe);
        g.bench_function(format!("insert_mid_{label}"), |b| {
            b.iter_batched(
                || {
                    let mut d = doc_at(&text, mid);
                    support::install_diagnostics(&mut d, 128);
                    d
                },
                |mut d| d.type_char('x'),
                BatchSize::PerIteration,
            );
        });
        g.bench_function(format!("enter_mid_{label}"), |b| {
            b.iter_batched(
                || doc_at(&text, mid),
                |mut d| d.enter(),
                BatchSize::PerIteration,
            );
        });
        g.bench_function(format!("undo_step_{label}"), |b| {
            b.iter_batched(
                || {
                    let mut d = doc_at(&text, mid);
                    d.type_char('x');
                    d
                },
                |mut d| {
                    d.undo();
                },
                BatchSize::PerIteration,
            );
        });
    }
    g.finish();
}

fn find(c: &mut Criterion) {
    let mut g = c.benchmark_group("find");
    g.sample_size(10).measurement_time(Duration::from_secs(3));
    let query = || Some(FindQuery { text: "return".into(), case_sensitive: false, ..Default::default() });
    for (label, text) in support::sized() {
        g.bench_function(format!("set_query_{label}"), |b| {
            b.iter_batched(
                || doc_at(&text, 0),
                |mut d| d.set_find_query(query(), 0),
                BatchSize::PerIteration,
            );
        });
        let probe = doc_at(&text, 0);
        let mid = mid_line_start(&probe, &text);
        drop(probe);
        g.bench_function(format!("keystroke_while_active_{label}"), |b| {
            b.iter_batched(
                || {
                    let mut d = doc_at(&text, mid);
                    d.set_find_query(query(), 0);
                    d
                },
                |mut d| {
                    d.type_char('x');
                    // Past the debounce window → refresh the match set. The
                    // per-commit windowed repair keeps this cheap: the set is
                    // already current, so this confirms rather than re-scans.
                    d.maybe_rescan_find(1_000);
                },
                BatchSize::PerIteration,
            );
        });
    }
    // Navigation over a big live match set (independent of corpus size).
    let text = support::gen_doc(1_000_000, 11);
    let mut d = doc_at(&text, 0);
    d.set_find_query(query(), 0);
    g.bench_function("navigate_big_set", |b| {
        b.iter(|| black_box(d.find_next(0)));
    });
    g.finish();
}

fn highlight(c: &mut Criterion) {
    let mut g = c.benchmark_group("highlight");
    g.sample_size(10).measurement_time(Duration::from_secs(3));
    for (label, text) in support::sized() {
        let probe = doc_at(&text, 0);
        let mid = mid_line_start(&probe, &text);
        let mid_row = probe.buffer().offset_to_point(mid).row;
        drop(probe);
        // Edit mid-doc on a fully tokenized cache, then bring the viewport
        // current — measures the incremental re-tokenize + drive overhead.
        g.bench_function(format!("tokenize_viewport_after_edit_{label}"), |b| {
            b.iter_batched(
                || {
                    let mut d = doc_at(&text, mid);
                    d.set_syntax(support::syntax(), support::theme());
                    let n = d.buffer().line_count();
                    d.tokenize_highlight(n);
                    d.type_char('x');
                    d
                },
                |mut d| d.tokenize_highlight(mid_row + 50),
                BatchSize::PerIteration,
            );
        });
        // Viewport-only frontier (huge dirty tail), then a keystroke at the
        // top — exercises commit-time cache maintenance when a large invalid
        // tail is outstanding, which must stay proportional to the edit, not
        // to the size of the dirty set.
        g.bench_function(format!("keystroke_deep_dirty_tail_{label}"), |b| {
            b.iter_batched(
                || {
                    let mut d = doc_at(&text, 10);
                    d.set_syntax(support::syntax(), support::theme());
                    d.tokenize_highlight(50);
                    d
                },
                |mut d| d.type_char('x'),
                BatchSize::PerIteration,
            );
        });
    }
    // Highlight virtualization: a cold jump into a fully swept 1 MB document
    // re-aims the retention window and refills it from the nearest sparse
    // checkpoint — a stride of warm-up plus window+slack lines, budget-paced.
    let text = support::gen_doc(1_000_000, 23);
    g.bench_function("cold_jump_1m", |b| {
        b.iter_batched(
            || {
                let mut d = doc_at(&text, 0);
                d.set_syntax(support::syntax(), support::theme());
                let n = d.buffer().line_count();
                while d.highlight_frontier().is_some() {
                    d.tokenize_highlight(n);
                }
                d
            },
            |mut d| {
                let mid = d.buffer().line_count() / 2;
                d.set_highlight_window(mid..mid + 40);
                while d.highlight_frontier().is_some() {
                    d.tokenize_highlight(mid + 40);
                }
            },
            BatchSize::PerIteration,
        );
    });
    // Off-thread segment tokenizer: the pure per-segment throughput the app
    // parallelizes across cores during a speculative full-document sweep.
    let text = support::gen_doc(4_000_000, 29); // ~64k-line segments
    let seg = {
        let mut d = doc_at(&text, 0);
        d.set_syntax(support::syntax(), support::theme());
        (d.highlight_engine().unwrap(), d.snapshot())
    };
    g.bench_function("segment_tokenize_64k", |b| {
        b.iter(|| {
            black_box(scrive_core::tokenize_segment(
                &seg.0,
                &seg.1,
                0..65_536,
                scrive_core::SegmentStart::Fresh,
                None,
                None,
            ))
        });
    });
    // Wrong-guess repair with the early-stop splice: a 64k segment guessed
    // Fresh (ground) but truly starting INSIDE a block comment, re-run from
    // the true start. A stateful grammar with `/* */` is needed for the guess
    // to be genuinely wrong; the comment closes ~500 rows in, so the re-run
    // converges (splices the tail) shortly after — the per-boundary repair cost.
    let (engine, snapshot, guessed, true_start) = {
        let engine = scrive_core::HighlightEngine::new(cmt_syntax(), support::theme());
        let mut lines: Vec<String> = (0..70_000).map(|i| format!("fn word {i}")).collect();
        lines[100] = "/*".into(); // a comment open above the segment…
        lines[1_500] = "*/ fn".into(); // …closing ~500 rows into the segment
        let text = lines.join("\n");
        let snap = Document::new(&text).expect("bench doc loads").snapshot();
        let seg_rows = 1_000..66_536;
        let guessed = scrive_core::tokenize_segment(
            &engine, &snap, seg_rows, scrive_core::SegmentStart::Fresh, None, None,
        );
        let prefix = scrive_core::tokenize_segment(
            &engine, &snap, 0..1_000, scrive_core::SegmentStart::Fresh, None, None,
        );
        (engine, snap, guessed, prefix.end_boundary().clone())
    };
    g.bench_function("stitch_repair_64k", |b| {
        b.iter(|| {
            black_box(scrive_core::tokenize_segment(
                &engine,
                &snapshot,
                1_000..66_536,
                scrive_core::SegmentStart::After(true_start.clone()),
                None,
                Some(&guessed),
            ))
        });
    });
    g.finish();
}

/// A stateful bench grammar: `/* … */` block comments (asymmetric, so a wrong
/// guess self-heals at the close) plus `fn`/`return`/`switch` keywords.
fn cmt_syntax() -> scrive_core::SyntaxDef {
    const G: &str = "%YAML 1.2\n\
        ---\n\
        name: BenchCmt\n\
        scope: source.benchcmt\n\
        contexts:\n\
        \x20 main:\n\
        \x20   - match: '/\\*'\n\
        \x20     push: comment\n\
        \x20   - match: '\\b(fn|return|switch)\\b'\n\
        \x20     scope: keyword.control.bench\n\
        \x20 comment:\n\
        \x20   - match: '\\*/'\n\
        \x20     pop: true\n\
        \x20   - match: '.'\n\
        \x20     scope: comment.block.bench\n";
    scrive_core::SyntaxDef::from_sublime_syntax(G).expect("bench cmt grammar parses")
}

fn checkpoints(c: &mut Criterion) {
    let mut g = c.benchmark_group("checkpoints");
    g.sample_size(10).measurement_time(Duration::from_secs(3));
    // The per-keystroke sparse-checkpoint shift. An Enter at the TOP of a fully
    // swept document conceptually moves EVERY checkpoint (K ≈ lines / 256) down
    // one row. The delta-gap SumTree stores checkpoints by row gaps, so it
    // re-anchors just ONE seam (O(log K) + one heavy `LineState` clone) instead
    // of rewriting all K entries. Small K = small doc (cheap regardless); the
    // payoff is at large K, so the group grows the document size.
    for (label, text) in [
        ("100k", support::gen_doc(100_000, 31)),
        ("1m", support::gen_doc(1_000_000, 31)),
    ] {
        g.bench_function(format!("enter_top_swept_{label}"), |b| {
            b.iter_batched(
                || {
                    let mut d = doc_at(&text, 0);
                    d.set_syntax(support::syntax(), support::theme());
                    let n = d.buffer().line_count();
                    while d.highlight_frontier().is_some() {
                        d.tokenize_highlight(n);
                    }
                    d
                },
                |mut d| d.enter(),
                BatchSize::PerIteration,
            );
        });
    }
    g.finish();
}

fn query(c: &mut Criterion) {
    let mut g = c.benchmark_group("query");
    g.sample_size(20).measurement_time(Duration::from_secs(3));
    // Windowed decoration query against a big store (10k diagnostics).
    let text = support::gen_doc(1_000_000, 13);
    let mut d = doc_at(&text, 0);
    support::install_diagnostics(&mut d, 10_000);
    let len = d.buffer().len();
    let win = len / 2..len / 2 + 4_000;
    g.bench_function("diagnostics_in_window_10k", |b| {
        b.iter(|| black_box(d.diagnostics_in(win.clone()).count()));
    });
    // The per-frame occurrence wash query (already windowed — the floor).
    let word_at = d.buffer().slice(0..len).find("return").unwrap() as u32 + 1;
    d.set_selections(SelectionSet::new(word_at));
    g.bench_function("occurrences_viewport_1m", |b| {
        b.iter(|| black_box(d.caret_word_occurrences(win.clone()).len()));
    });
    g.finish();
}

fn snapshot(c: &mut Criterion) {
    let mut g = c.benchmark_group("snapshot");
    g.sample_size(20).measurement_time(Duration::from_secs(3));
    // Minting the compile thread's snapshot is O(1): the rope backing shares
    // structure, so a snapshot clones a handle rather than copying the text.
    for (label, text) in support::sized() {
        let d = doc_at(&text, 0);
        g.bench_function(format!("mint_{label}"), |b| {
            b.iter(|| black_box(d.snapshot()));
        });
    }
    g.finish();
}

fn brackets(c: &mut Criterion) {
    let mut g = c.benchmark_group("brackets");
    g.sample_size(20).measurement_time(Duration::from_secs(3));
    // Baseline: query the full bracket set of a 1 MB document, pinning the
    // whole-set size so the fixture is exercised. Rematch and windowed-query
    // benches will join this group against the same corpus.
    let text = support::gen_doc(1_000_000, 17);
    let d = doc_at(&text, 0);
    g.bench_function("all_len_1m", |b| {
        b.iter(|| black_box(d.brackets().all().len()));
    });
    g.finish();
}

criterion_group!(benches, keystroke, find, highlight, checkpoints, query, snapshot, brackets);
criterion_main!(benches);
