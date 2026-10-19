//! Off-thread parallel + speculative highlight sweep — the large-document
//! highlight mechanism `CodeEditor` drives internally.
//!
//! At millions of lines the top-down highlight state chain makes a jump to the
//! document bottom fall back to a long single-core tokenize. This pool tokenizes
//! the document as SEGMENTS on worker threads over an O(1) [`Snapshot`] clone,
//! stitches boundaries with the core's convergence rule ([`SegmentBoundary`]
//! equality), and feeds verified results back through
//! [`Document::absorb_highlight`]. The viewport is coloured immediately from a
//! GUESSED fresh state (speculative), then verified in place. scrive-core stays
//! synchronous and thread-free; all threading lives here.
//!
//! This mirrors the pool in `examples/scratch.rs` (which drives the low-level
//! widget by hand); `CodeEditor` owns this copy so a batteries-included host gets
//! large-document highlighting without wiring a worker pool itself.

use std::ops::Range;

use scrive_core::{
    Document, HighlightEngine, Revision, SegmentBoundary, SegmentStart, SegmentTokens, Snapshot,
};

/// Documents at least this many bytes use the parallel sweep; smaller ones keep
/// the synchronous path (so the sync path and its captures stay unchanged).
pub(crate) const PARALLEL_MIN_BYTES: u32 = 2 * 1_048_576;

/// Max rows per sweep segment. Small enough that workers turn over quickly and
/// the document splits into many more segments than workers, for load balancing
/// and fine verified-progress grain (~40 ms of syntect each).
const SEGMENT_MAX_ROWS: u32 = 32_768;

/// Rows the speculative viewport paint backs off ABOVE the window, so a short
/// local construct (a string/comment opened a few lines up) is usually absorbed
/// by the guess.
const SPECULATION_BACKOFF: u32 = 128;

/// Verified-chain segments [`HighlightPool::poll`] absorbs per frame — bounds a
/// frame that got many contiguous-ready segments to a constant, pacing total
/// verification over ⌈#segments/budget⌉ frames (the `active` flag keeps the
/// frame subscription firing).
const POLL_VERIFY_BUDGET: usize = 4;

/// Synchronous mis-guess re-runs [`HighlightPool::poll`] performs per frame — the
/// heavy unit (a full `tokenize_segment` on the UI thread), weighted tighter than
/// the verify budget; a frame defers a re-run that would exceed it.
const POLL_RERUN_BUDGET: usize = 1;

// Test-only tally of synchronous mis-guess re-runs — the pacing test reads it to
// prove the per-frame re-run budget holds.
#[cfg(test)]
thread_local! {
    static POLL_RERUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// One bulk chain segment to tokenize off-thread. `against` is the wrong-guessed
/// old result on a corrected re-run (enables the early-stop splice); `sweep` lets
/// a worker drop a job a restart has superseded. The job carries its sweep's
/// `engine` (a cheap `Arc` clone) rather than the worker holding one from spawn
/// time, so a restart after a theme swap (`CodeEditor`'s
/// `repaint_after_theme_swap` → [`HighlightPool::restart`]) tokenizes with the
/// new engine on the SAME long-lived threads. A grammar swap (`.language`,
/// `load`) does not restart this pool: `CodeEditor::seed_highlight` builds a
/// fresh one, threads and all.
struct Job {
    engine: HighlightEngine,
    idx: usize,
    snapshot: std::sync::Arc<Snapshot>,
    rows: Range<u32>,
    start: SegmentStart,
    spans_for: Range<u32>,
    against: Option<SegmentTokens>,
    sweep: u64,
}

/// A finished segment, echoing its sweep so the coordinator can drop work a
/// restart has superseded.
struct Done {
    idx: usize,
    seg: SegmentTokens,
    sweep: u64,
}

/// The worker pool + verification coordinator. Threads are spawned once and live
/// for the pool's lifetime; [`HighlightPool::start`] (re)dispatches a sweep. The
/// workers tokenize only the BULK chain; the viewport is painted synchronously
/// (see [`HighlightPool::speculate`]).
pub(crate) struct HighlightPool {
    /// The current sweep's grammar/theme handle — the document's engine as of
    /// the last [`HighlightPool::start`]. Used for the UI-thread speculation and
    /// re-runs, and cloned into every job.
    engine: HighlightEngine,
    /// The document-top fresh state — the coordinator compares each segment's
    /// guessed start against a prior segment's verified end with it.
    fresh: SegmentBoundary,
    /// The job queue (a deque; a re-run — critical-path — jumps not-yet-started
    /// bulk segments). The `Condvar` parks idle workers so idle costs no CPU.
    queue: std::sync::Arc<(std::sync::Mutex<std::collections::VecDeque<Job>>, std::sync::Condvar)>,
    /// The live sweep, shared with the workers: a worker drops a job whose
    /// sweep no longer matches BEFORE tokenizing it.
    cur_sweep: std::sync::Arc<std::sync::atomic::AtomicU64>,
    done_rx: std::sync::mpsc::Receiver<Done>,
    _workers: Vec<std::thread::JoinHandle<()>>,
    /// The snapshot the current sweep tokenizes; each job carries a clone.
    snapshot: std::sync::Arc<Snapshot>,
    /// The revision the current sweep is for — `CodeEditor` compares it against
    /// the live document revision to decide whether to restart.
    pub(crate) rev: Revision,
    /// The current sweep's number, bumped by every [`HighlightPool::start`] —
    /// what jobs and results are tagged with. NOT the revision: a theme swap
    /// restarts the sweep at the SAME revision, and the outgoing sweep's
    /// still-arriving results must not be mistaken for the new one's.
    sweep: u64,
    /// The segment ranges of the current sweep (for re-dispatching a re-run).
    seg_rows: Vec<Range<u32>>,
    results: Vec<Option<SegmentTokens>>,
    next_verify: usize,
    /// The verified end boundary before `next_verify` (None => fresh / row 0).
    prev_end: Option<SegmentBoundary>,
    /// The padded viewport window — `spans_for` on chain segments + the paint.
    window: Range<u32>,
    /// Whether a sweep is still verifying — keeps the frame subscription alive.
    pub(crate) active: bool,
}

impl HighlightPool {
    /// Spawn the worker pool over a document's grammar/theme, then dispatch the
    /// first sweep aimed at `viewport`. `None` without a grammar.
    pub(crate) fn new(doc: &Document, viewport: Range<u32>) -> Option<Self> {
        let engine = doc.highlight_engine()?;
        let count = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Done>();
        let queue: std::sync::Arc<(std::sync::Mutex<std::collections::VecDeque<Job>>, std::sync::Condvar)> =
            std::sync::Arc::new((std::sync::Mutex::new(std::collections::VecDeque::new()), std::sync::Condvar::new()));
        let cur_sweep = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0));
        let workers = (0..count)
            .map(|_| {
                let queue = queue.clone();
                let cur_sweep = cur_sweep.clone();
                let done_tx = done_tx.clone();
                std::thread::spawn(move || loop {
                    let job = {
                        let (lock, cvar) = &*queue;
                        let mut q = lock.lock().unwrap();
                        while q.is_empty() {
                            q = cvar.wait(q).unwrap();
                        }
                        q.pop_front().unwrap()
                    };
                    // Drop a stale job (an edit or a theme swap restarted the
                    // sweep) BEFORE spending a full tokenization on it.
                    if job.sweep != cur_sweep.load(std::sync::atomic::Ordering::Relaxed) {
                        continue;
                    }
                    let seg = scrive_core::tokenize_segment(
                        &job.engine, &job.snapshot, job.rows, job.start, Some(job.spans_for),
                        job.against.as_ref(),
                    );
                    let _ = done_tx.send(Done { idx: job.idx, seg, sweep: job.sweep });
                })
            })
            .collect();
        let fresh = engine.fresh_boundary();
        let mut pool = Self {
            engine,
            fresh,
            queue,
            cur_sweep,
            done_rx,
            _workers: workers,
            snapshot: std::sync::Arc::new(doc.snapshot()),
            rev: doc.revision(),
            sweep: 0,
            seg_rows: Vec::new(),
            results: Vec::new(),
            next_verify: 0,
            prev_end: None,
            window: 0..0,
            active: false,
        };
        pool.start(doc, viewport);
        Some(pool)
    }

    /// (Re)dispatch a full sweep at the document's current revision AND engine —
    /// the initial run, the post-edit restart, and the post-theme-swap restart
    /// all come here. The verified prefix already absorbed survives an edit (it
    /// rides `on_commit`); this re-sweeps from row 0 with a fresh snapshot.
    fn start(&mut self, doc: &Document, viewport: Range<u32>) {
        self.rev = doc.revision();
        // Re-read the engine every sweep: the document refuses segments from any
        // engine but its current one, and a theme swap replaces it without
        // moving the revision. (`None` — no highlighter — cannot follow a
        // pool's creation; keeping the old handle is harmless, nothing absorbs.)
        if let Some(engine) = doc.highlight_engine() {
            self.fresh = engine.fresh_boundary();
            self.engine = engine;
        }
        // A new sweep number, published to the workers, and DRAIN the stale
        // queue, so a restart does not pile prior-sweep jobs ahead of the
        // current chain (and their in-flight results are dropped on arrival).
        self.sweep = self.sweep.wrapping_add(1);
        self.cur_sweep.store(self.sweep, std::sync::atomic::Ordering::Relaxed);
        self.queue.0.lock().unwrap().clear();
        self.snapshot = std::sync::Arc::new(doc.snapshot());
        let n = self.snapshot.line_count();
        self.set_window(viewport, n);
        let workers = self._workers.len() as u32;
        let segs = (n.div_ceil(SEGMENT_MAX_ROWS)).max(workers).max(1);
        let seg_len = n.div_ceil(segs).max(1);
        self.seg_rows = (0..n)
            .step_by(seg_len as usize)
            .map(|s| s..(s + seg_len).min(n))
            .collect();
        self.results = (0..self.seg_rows.len()).map(|_| None).collect();
        self.next_verify = 0;
        self.prev_end = None;
        self.active = !self.seg_rows.is_empty();
        let (lock, cvar) = &*self.queue;
        let mut q = lock.lock().unwrap();
        for (idx, rows) in self.seg_rows.iter().enumerate() {
            q.push_back(Job {
                engine: self.engine.clone(),
                idx,
                snapshot: self.snapshot.clone(),
                rows: rows.clone(),
                start: SegmentStart::Fresh,
                spans_for: self.window.clone(),
                against: None,
                sweep: self.sweep,
            });
        }
        cvar.notify_all();
    }

    fn set_window(&mut self, viewport: Range<u32>, n: u32) {
        // Through the core's ONE window owner so this pool paints/tokenizes
        // exactly the rows the cache retains — the length cap matters because a
        // collapsed mega-fold makes the visible range span the fold's hidden
        // interior, which an uncapped `speculate` would tokenize synchronously.
        self.window = scrive_core::padded_highlight_window(viewport, n);
    }

    /// Restart the sweep after an edit (fresh snapshot) or a theme swap (fresh
    /// engine; a grammar swap builds a fresh pool instead) AND repaint the
    /// viewport now — the pair must go together:
    /// a bare `start` leaves the visible rows showing stale colours until the
    /// chain verifies down to them.
    pub(crate) fn restart(&mut self, doc: &mut Document, viewport: Range<u32>) {
        self.start(doc, viewport.clone());
        self.speculate(doc, viewport);
    }

    /// Re-aim the window on a scroll (no edit) AND repaint the newly-visible rows
    /// now — the same pairing as [`Self::restart`], one owner.
    pub(crate) fn reaim(&mut self, doc: &mut Document, viewport: Range<u32>) {
        let n = doc.buffer().line_count();
        self.set_window(viewport.clone(), n);
        self.speculate(doc, viewport);
    }

    /// Paint the viewport window immediately from a GUESSED fresh state,
    /// SYNCHRONOUSLY — one window of rows is a few ms. Absorbed `verified=false`:
    /// spans show on the still-dirty rows, no checkpoints are planted, and the
    /// parallel chain re-verifies them per line. No-op once the sweep is done or
    /// if the pool's snapshot is stale.
    pub(crate) fn speculate(&self, doc: &mut Document, viewport: Range<u32>) {
        if !self.active || self.rev != doc.revision() {
            return;
        }
        let n = self.snapshot.line_count();
        let _ = viewport;
        let rows = self.window.start.saturating_sub(SPECULATION_BACKOFF)..self.window.end.min(n);
        if rows.start >= rows.end {
            return;
        }
        let seg = scrive_core::tokenize_segment(
            &self.engine,
            &self.snapshot,
            rows,
            SegmentStart::Fresh,
            Some(self.window.clone()),
            None,
        );
        doc.absorb_highlight(self.rev, seg, false);
    }

    /// Drain finished jobs and advance the verified chain, absorbing into `doc`.
    /// BUDGETED per frame; a partial frame stays `active` so the frame
    /// subscription resumes it next frame.
    pub(crate) fn poll(&mut self, doc: &mut Document) {
        while let Ok(done) = self.done_rx.try_recv() {
            self.intake(done);
        }
        let mut verified = 0usize;
        let mut reruns = 0usize;
        while self.next_verify < self.results.len() {
            if verified >= POLL_VERIFY_BUDGET {
                break; // paced out this frame; `active` stays true → resume next
            }
            let Some(seg_ref) = self.results[self.next_verify].as_ref() else { break };
            let true_start_fresh =
                self.next_verify == 0 || self.prev_end.as_ref() == Some(&self.fresh);
            let needs_rerun = seg_ref.started_fresh() && !true_start_fresh;
            if needs_rerun && reruns >= POLL_RERUN_BUDGET {
                break; // one more heavy re-run would exceed the frame budget
            }
            let seg = self.results[self.next_verify].take().expect("peeked Some");
            if needs_rerun {
                // Wrong guess: re-run from the true prior end synchronously; the
                // early-stop makes this near-instant in the common case.
                let start = SegmentStart::After(self.prev_end.clone().expect("prev segment is verified"));
                let fixed = scrive_core::tokenize_segment(
                    &self.engine,
                    &self.snapshot,
                    self.seg_rows[self.next_verify].clone(),
                    start,
                    None, // ignored: converge_against forces the old window
                    Some(&seg),
                );
                let end = fixed.end_boundary().clone();
                doc.absorb_highlight(self.rev, fixed, true);
                self.prev_end = Some(end);
                reruns += 1;
                #[cfg(test)]
                POLL_RERUNS.with(|c| c.set(c.get() + 1));
            } else {
                let end = seg.end_boundary().clone();
                doc.absorb_highlight(self.rev, seg, true);
                self.prev_end = Some(end);
            }
            self.next_verify += 1;
            verified += 1;
        }
        if self.next_verify >= self.results.len() && self.active {
            self.active = false;
        }
    }

    /// File one finished job into the current sweep's results, or drop it if
    /// it belongs to a superseded sweep — the ONE staleness rule for arrivals.
    fn intake(&mut self, done: Done) {
        if done.sweep == self.sweep {
            self.results[done.idx] = Some(done.seg);
        } // else stale — drop it
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scrive_core::SyntaxDef;

    // A minimal grammar with a string context, so an unterminated `"` leaves a
    // non-fresh (string-context) end — enough to exercise the mis-guess re-run
    // budget without the example's full Rust grammar (not reachable from `src/`).
    const GRAMMAR: &str = "%YAML 1.2\n---\nscope: source.t\ncontexts:\n  main:\n    - match: '\"'\n      push: string\n  string:\n    - match: '\"'\n      pop: true\n";

    fn grammar_doc(source: &str) -> Document {
        let mut doc = Document::new(source).expect("fits the u32 offset space");
        doc.set_syntax(
            SyntaxDef::from_sublime_syntax(GRAMMAR).expect("grammar parses"),
            crate::scrive_dark_theme(),
        );
        doc
    }

    /// `poll` must PACE the verified chain: one frame absorbs at most
    /// `POLL_VERIFY_BUDGET` segments and at most `POLL_RERUN_BUDGET` heavy
    /// synchronous re-runs, then leaves the pool `active` until the whole document
    /// verifies over several frames — never draining every contiguous-ready
    /// segment (and re-run) in one blocking call.
    #[test]
    fn poll_verifies_at_most_budget_per_frame() {
        // 12 one-row segments (> POLL_VERIFY_BUDGET). Rows 2 and 4 open a string
        // with an unterminated `"`, so the Fresh-guessed segment that follows a
        // string-context end is a genuine mis-guess (exercises the re-run budget).
        const N: usize = 12;
        let mut lines = [""; N];
        lines[2] = "\"";
        lines[4] = "\"";
        let source = lines.join("\n");
        let mut doc = grammar_doc(&source);
        assert_eq!(doc.buffer().line_count(), N as u32, "one row per segment");

        let mut pool = HighlightPool::new(&doc, 0..N as u32).expect("grammar → engine");

        // Neutralize the worker sweep `new` dispatched so `results` is entirely
        // under test control (retag to an unused revision and sweep, stop
        // workers, drain).
        pool.rev = scrive_core::Revision(u64::MAX);
        pool.sweep = u64::MAX;
        pool.cur_sweep.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
        pool.queue.0.lock().unwrap().clear();
        while pool.done_rx.try_recv().is_ok() {}

        // One ready `Some(SegmentTokens)` per row, each a Fresh guess.
        pool.snapshot = std::sync::Arc::new(doc.snapshot());
        pool.seg_rows = (0..N as u32).map(|r| r..r + 1).collect();
        pool.window = pool.snapshot.line_count()..pool.snapshot.line_count();
        pool.results = pool
            .seg_rows
            .iter()
            .map(|rows| {
                Some(scrive_core::tokenize_segment(
                    &pool.engine,
                    &pool.snapshot,
                    rows.clone(),
                    SegmentStart::Fresh,
                    None,
                    None,
                ))
            })
            .collect();
        pool.next_verify = 0;
        pool.prev_end = None;
        pool.active = true;

        assert!(
            pool.results.iter().all(|s| s.as_ref().unwrap().started_fresh()),
            "all seeded segments are Fresh guesses",
        );
        assert!(
            pool.results[2].as_ref().unwrap().end_boundary() != &pool.fresh,
            "an unterminated `\"` leaves the string context open (non-fresh end)",
        );

        let mut frames = 0usize;
        loop {
            let before = pool.next_verify;
            let reruns_before = POLL_RERUNS.with(std::cell::Cell::get);
            pool.poll(&mut doc);
            let delta = pool.next_verify - before;
            let reruns = POLL_RERUNS.with(std::cell::Cell::get) - reruns_before;
            frames += 1;

            assert!(delta <= POLL_VERIFY_BUDGET, "frame absorbed {delta} > verify budget");
            assert!(reruns <= POLL_RERUN_BUDGET as u64, "frame ran {reruns} > rerun budget");
            if pool.next_verify < pool.results.len() {
                assert!(pool.active, "a partial frame stays active so the subscription re-fires");
                assert!(delta >= 1, "a partial frame must make progress (no live-lock)");
            }
            if !pool.active {
                break;
            }
            assert!(frames <= N + 4, "must converge, not spin");
        }

        assert_eq!(pool.next_verify, N, "every segment verified");
        assert!(!pool.active, "active clears only once the document is fully verified");
        assert!(frames > 1, "verification was paced across frames, not drained in one");
    }

    /// A theme swap leaves the revision where it was, so the revision alone
    /// cannot tell the pool its sweep went stale. `restart` must therefore (a)
    /// drop the previous sweep's results still arriving — they match the
    /// revision but were tokenized by the outgoing engine — and (b) sweep with
    /// the document's NEW engine, or the document refuses every segment and a
    /// large buffer silently falls back to the synchronous walk. Fails-first
    /// against a pool that tagged work by revision and kept its spawn-time engine.
    #[test]
    fn restart_after_a_theme_swap_adopts_the_new_engine_and_drops_the_old_sweep() {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(60);
        // 200 rows: the pool's padded window spans the whole document, so its
        // segments carry spans for every row and accepted absorbs alone clean
        // the frontier — no synchronous walk is ever run here.
        let mut doc = grammar_doc(&"x\n".repeat(199));
        let mut pool = HighlightPool::new(&doc, 0..40).expect("grammar → engine");
        // Hold the first sweep's finished results: in-flight work at the swap.
        let mut in_flight = Vec::new();
        while in_flight.len() < pool.seg_rows.len() {
            let left = deadline.saturating_duration_since(std::time::Instant::now());
            in_flight.push(pool.done_rx.recv_timeout(left).expect("the first sweep finishes"));
        }

        // An identical theme still mints a new engine: identity, not content.
        doc.set_theme(crate::scrive_dark_theme());
        pool.restart(&mut doc, 0..40);
        for done in in_flight {
            pool.intake(done); // the old sweep's results arrive after the restart
        }
        assert!(
            pool.results.iter().all(Option::is_none),
            "results from the sweep before the restart must be dropped",
        );

        while pool.active {
            pool.poll(&mut doc);
            assert!(std::time::Instant::now() < deadline, "the restarted sweep never finished");
            std::thread::yield_now();
        }
        assert_eq!(
            doc.highlight_frontier(),
            None,
            "the document accepted every verified segment — the pool swept with its current engine",
        );
    }
}
