//! `scratch` — the development window used to review the editor by eye.
//!
//! ```text
//! cargo run -p scrive-iced --example scratch                              # open the window
//! cargo run -p scrive-iced --example scratch -- --capture out.png         # headless PNG
//! cargo run -p scrive-iced --example scratch -- --capture-folds out.png   # …all folds collapsed
//! cargo run -p scrive-iced --example scratch -- --capture-find out.png    # …find+replace bar open
//! ```
//!
//! `--capture-folds` is the fold-geometry verification harness: it collapses
//! every collapsible pair before rendering, so a refactor of the fold/display
//! projections can be byte-diffed against a baseline PNG (chips, collapsed
//! tails, gutter fold gaps all on screen at once).
//!
//! A real editor now: type, move (arrows / Home / End / Ctrl+arrows), backspace,
//! enter, tab, click to place the caret — all applied to a `scrive_core::Document`
//! through the [`scrive_iced::Editor`] widget's [`Action`]s.

// On Windows, a release build is a GUI app with no console window; debug builds
// keep the console so panics and the `--capture` messages stay visible. Ignored
// on non-Windows targets.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[path = "shared/capture.rs"]
mod capture;

use std::time::Instant;

use iced::alignment::{Horizontal, Vertical};
use iced::widget::operation::focus;
use iced::widget::{button, column, container, row, stack, text, text_input};
use iced::{Alignment, Color, Element, Length, Shadow, Subscription, Task, Theme, Vector};
use std::ops::Range;

use scrive_core::{
    default_indent_size, is_completion_word_char, CompletionController, CompletionCx, CompletionItem,
    CompletionKind, CompletionState, CompletionTrigger, Completions, Diagnostic, Document, FindQuery,
    Hover, HoverCx, HoverInfo, InsertText, Point, Selection, SelectionId, SelectionSet, Severity,
    SignatureCx, SignatureHelp, SignatureInfo, Snippet, SnippetSession, SyntaxDef, TabOutcome,
    TokenTheme, LOOKBACK_LINES,
};
use scrive_iced::{Action, Editor};

/// Focusable ids. iced's `focus` operation focuses one and unfocuses all other
/// focusables, so moving focus between the find input and the editor is a proper
/// single-focus model (no double-typing).
const FIND_INPUT: &str = "scrive-find-input";
const REPLACE_INPUT: &str = "scrive-replace-input";
const EDITOR: &str = "editor";

/// The `--large <MB>` corpus, generated in `main` before the app starts.
static LARGE_DOC: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Skip the demo lint's full-text scan above this size: the stub stands in
/// for a real compile loop (which consumes snapshots off-thread); scanning a
/// 100 MB document on the UI thread per keystroke is not what it demonstrates.
const RELINT_MAX_BYTES: u32 = 2 * 1_048_576;

/// Debounce window (ms) for the demo re-lint. The stub compile pass scans
/// the whole document; a real off-thread compiler is debounced the same way, so
/// a burst of keystrokes coalesces into at most one scan per window instead of
/// one O(N log N) scan per key. Mirrors find's `default_find_debounce` cadence
/// (slightly longer — a lint pass is heavier than a match refill). Field-gate
/// class: tune on the running app.
const RELINT_DEBOUNCE_MS: u64 = 250;

// Test-only tally of `App::relint` scan-body executions (post size-guard) — the
// `relint_debounces_off_the_keystroke_path` test reads it to prove the
// whole-document scan runs once per debounce window, not once per keystroke.
#[cfg(test)]
thread_local! {
    static RELINT_RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

// Test-only tally of synchronous mis-guess re-runs inside `HighlightPool::poll`
// — the `poll_verifies_at_most_budget_per_frame` test reads its per-frame delta
// to prove no more than `POLL_RERUN_BUDGET` heavy re-runs fire in one frame.
#[cfg(test)]
thread_local! {
    static POLL_RERUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// Synthetic Rust-shaped text for `--large <MB>` — the large-document stress
/// corpus. Duplicates the bench generator's SHAPE (benches can't be
/// imported across crates): nested `fn` / `match` blocks, comments, `return`
/// needles, ~9 lines and 8 bracket pairs per ~250 B block.
fn gen_large(mb: usize) -> String {
    let target = mb * 1_048_576;
    let mut rng = 7u64;
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = String::with_capacity(target + 512);
    let mut i = 0usize;
    while s.len() < target {
        s.push_str(&format!("// block {i}: canned readings for id 0x{:02X}\n", next() & 0xFF));
        s.push_str(&format!("fn read_{i}(id: u8) -> u8 {{\n    match id {{\n"));
        for _ in 0..3 {
            s.push_str(&format!("        0x{:02X} => {{ return {}; }}\n", next() & 0xFF, next() % 200));
        }
        s.push_str("        _ => { return 0; }\n    }\n}\n\n");
        i += 1;
    }
    s
}

/// A document taller than the window, so scrolling is visible. Line numbers are
/// embedded in the text so it's obvious which rows the viewport is showing.
fn long_sample() -> String {
    // A small, self-contained Rust program chosen to exercise the editor: nested
    // foldable blocks (impl / fn / match), inline literals, generics, doc comments,
    // and the hover / signature / completion vocabulary.
    r#"//! A fixed-capacity window over the most recent sensor readings, with a
//! rolling average — a small, self-contained example.

use std::collections::VecDeque;

/// The most recent readings, oldest first, capped at `capacity`.
pub struct Samples {
    values: VecDeque<f64>,
    capacity: usize,
}

impl Samples {
    /// Create an empty window that keeps at most `capacity` readings.
    pub fn new(capacity: usize) -> Self {
        Samples {
            values: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    /// Record one reading, evicting the oldest once the window is full.
    pub fn push(&mut self, reading: f64) {
        if self.values.len() == self.capacity {
            self.values.pop_front();
        }
        self.values.push_back(reading);
    }

    /// The mean of the retained readings, or `None` when the window is empty.
    pub fn average(&self) -> Option<f64> {
        match self.values.len() {
            0 => None,
            n => {
                let total: f64 = self.values.iter().sum();
                Some(total / n as f64)
            }
        }
    }
}

fn main() {
    let mut window = Samples::new(4);
    for reading in [21.5, 22.0, 23.25, 24.0, 25.5] {
        window.push(reading);
        println!("average = {:?}", window.average());
    }
}
"#
    .to_string()
}

// ======================================================================
// Off-thread parallel + speculative highlight sweep.
//
// At `--large 100` (~3.2M lines) the top-down state chain makes a jump to the
// bottom render fallback for ~16-20s of single-core work. This app-side pool
// tokenizes the document as SEGMENTS on worker threads over an O(1) `Snapshot`
// clone, stitches boundaries with the core's convergence rule
// (`SegmentBoundary` equality), and feeds verified results back through
// `Document::absorb_highlight`. The viewport is coloured immediately from a
// GUESSED fresh state (speculative), then verified in place. Core stays sync
// and thread-free; all threading lives here.
// ======================================================================

/// Documents at least this many bytes use the parallel sweep; smaller ones
/// keep the synchronous path unchanged (so captures stay byte-identical).
const PARALLEL_MIN_BYTES: u32 = 2 * 1_048_576;

/// Max rows per sweep segment. Small enough that workers turn over quickly and
/// the document splits into many more segments than workers, for load
/// balancing and fine verified-progress grain (~40 ms of syntect each).
const SEGMENT_MAX_ROWS: u32 = 32_768;

/// Rows the speculative viewport paint backs off ABOVE the window, so a short
/// local construct (a string/comment opened a few lines up) is usually
/// absorbed by the guess.
const SPECULATION_BACKOFF: u32 = 128;

/// Verified-chain segments [`HighlightPool::poll`] absorbs per frame.
/// Without it a frame whose channel handed back many contiguous-ready segments
/// verifies them all at once — O(#segments) work (plus any mis-guess re-runs)
/// in one paint. Bounding it paces total verification over ⌈#segments/budget⌉
/// frames (the `active` flag keeps `window::frames` firing), each frame a
/// constant. Field-gate class: start small, calibrate on `--large`.
const POLL_VERIFY_BUDGET: usize = 4;

/// Synchronous mis-guess re-runs [`HighlightPool::poll`] performs per frame —
/// the HEAVY unit (a full `tokenize_segment` on the UI thread), so weighted
/// tighter than the verify budget. A frame stops before a re-run that would
/// exceed this, deferring it (and its `active`-kept reschedule) to the next
/// frame. Field-gate class.
const POLL_RERUN_BUDGET: usize = 1;

/// One bulk chain segment to tokenize off-thread. `against` is the
/// wrong-guessed old result on a corrected re-run (enables the early-stop
/// splice); `rev` lets a worker drop a job an edit has superseded.
struct Job {
    idx: usize,
    snapshot: std::sync::Arc<scrive_core::Snapshot>,
    rows: Range<u32>,
    start: scrive_core::SegmentStart,
    spans_for: Range<u32>,
    against: Option<scrive_core::SegmentTokens>,
    rev: scrive_core::Revision,
}

/// A finished segment, echoing the revision so the coordinator can drop stale
/// work.
struct Done {
    idx: usize,
    seg: scrive_core::SegmentTokens,
    rev: scrive_core::Revision,
}

/// The worker pool + verification coordinator. Threads are spawned once and
/// live for the app; [`HighlightPool::start`] (re)dispatches a sweep. The
/// workers tokenize only the BULK chain; the viewport is painted synchronously
/// (see [`HighlightPool::speculate`]) — one window of rows is cheap on the UI
/// thread and colours instantly, and wrong-guess re-runs also run on the UI
/// thread (the early-stop makes them near-instant), so the verified chain
/// advances as fast as the parallel Fresh results arrive.
struct HighlightPool {
    /// Grammar/theme handle — for the UI-thread speculation + re-runs (the
    /// workers hold their own clones).
    engine: scrive_core::HighlightEngine,
    /// The document-top fresh state — the coordinator compares each segment's
    /// guessed start against a prior segment's verified end with it.
    fresh: scrive_core::SegmentBoundary,
    /// The job queue (a deque; a re-run — critical-path — jumps not-yet-started
    /// bulk segments). The `Condvar` parks idle workers so idle costs no CPU.
    queue: std::sync::Arc<(std::sync::Mutex<std::collections::VecDeque<Job>>, std::sync::Condvar)>,
    /// The live revision, shared with the workers: a worker drops a job whose
    /// revision no longer matches BEFORE tokenizing it, so an edit's restart
    /// does not burn full tokenizations on results `poll` would only discard.
    cur_rev: std::sync::Arc<std::sync::atomic::AtomicU64>,
    done_rx: std::sync::mpsc::Receiver<Done>,
    _workers: Vec<std::thread::JoinHandle<()>>,
    /// The snapshot the current sweep tokenizes; each job carries a clone.
    snapshot: std::sync::Arc<scrive_core::Snapshot>,
    rev: scrive_core::Revision,
    /// The segment ranges of the current sweep (for re-dispatching a re-run).
    seg_rows: Vec<Range<u32>>,
    results: Vec<Option<scrive_core::SegmentTokens>>,
    next_verify: usize,
    /// The verified end boundary before `next_verify` (None => fresh / row 0).
    prev_end: Option<scrive_core::SegmentBoundary>,
    /// The padded viewport window — `spans_for` on chain segments + the paint.
    window: Range<u32>,
    active: bool,
}

impl HighlightPool {
    /// Spawn the worker pool over a document's grammar/theme, then dispatch the
    /// first sweep aimed at `viewport`. `None` without a grammar.
    fn new(doc: &Document, viewport: Range<u32>) -> Option<Self> {
        let engine = doc.highlight_engine()?;
        let count = std::thread::available_parallelism()
            .map(|n| n.get().saturating_sub(1).max(1))
            .unwrap_or(1);
        let (done_tx, done_rx) = std::sync::mpsc::channel::<Done>();
        let queue: std::sync::Arc<(std::sync::Mutex<std::collections::VecDeque<Job>>, std::sync::Condvar)> =
            std::sync::Arc::new((std::sync::Mutex::new(std::collections::VecDeque::new()), std::sync::Condvar::new()));
        let cur_rev = std::sync::Arc::new(std::sync::atomic::AtomicU64::new(doc.revision().0));
        let workers = (0..count)
            .map(|_| {
                let engine = engine.clone();
                let queue = queue.clone();
                let cur_rev = cur_rev.clone();
                let done_tx = done_tx.clone();
                std::thread::spawn(move || loop {
                    // Park until a job is available; take it, then release the
                    // lock so processing is concurrent.
                    let job = {
                        let (lock, cvar) = &*queue;
                        let mut q = lock.lock().unwrap();
                        while q.is_empty() {
                            q = cvar.wait(q).unwrap();
                        }
                        q.pop_front().unwrap()
                    };
                    // Drop a stale job (an edit restarted the sweep) BEFORE
                    // spending a full tokenization on it.
                    if job.rev.0 != cur_rev.load(std::sync::atomic::Ordering::Relaxed) {
                        continue;
                    }
                    let seg = scrive_core::tokenize_segment(
                        &engine, &job.snapshot, job.rows, job.start, Some(job.spans_for),
                        job.against.as_ref(),
                    );
                    let _ = done_tx.send(Done { idx: job.idx, seg, rev: job.rev });
                })
            })
            .collect();
        let fresh = engine.fresh_boundary();
        let mut pool = Self {
            engine,
            fresh,
            queue,
            cur_rev,
            done_rx,
            _workers: workers,
            snapshot: std::sync::Arc::new(doc.snapshot()),
            rev: doc.revision(),
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

    /// (Re)dispatch a full sweep at the document's current revision — the
    /// initial run and the post-edit restart both come here. The verified
    /// prefix already absorbed into the cache survives (it rides `on_commit`);
    /// this simply re-sweeps from row 0 with a fresh snapshot.
    fn start(&mut self, doc: &Document, viewport: Range<u32>) {
        self.rev = doc.revision();
        // Publish the new revision to the workers and DRAIN the stale queue, so
        // a restart (per edit) does not pile prior-revision jobs ahead of the
        // current chain — the workers would fully tokenize each only for `poll`
        // to discard it, starving the newest sweep.
        self.cur_rev.store(self.rev.0, std::sync::atomic::Ordering::Relaxed);
        self.queue.0.lock().unwrap().clear();
        self.snapshot = std::sync::Arc::new(doc.snapshot());
        let n = self.snapshot.line_count();
        self.set_window(viewport, n);
        // ~1 segment per worker at least, capped so a huge document yields many
        // more segments than workers (queued) for load balance + progress grain.
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
                idx,
                snapshot: self.snapshot.clone(),
                rows: rows.clone(),
                start: scrive_core::SegmentStart::Fresh,
                spans_for: self.window.clone(),
                against: None,
                rev: self.rev,
            });
        }
        cvar.notify_all();
    }

    fn set_window(&mut self, viewport: Range<u32>, n: u32) {
        // Through the core's ONE window owner (`padded_highlight_window`) so this
        // pool paints/tokenizes exactly the rows the cache retains — the length
        // cap matters because a collapsed mega-fold makes the visible range span
        // the fold's hidden interior, which an uncapped `speculate` would
        // tokenize synchronously on the UI thread.
        self.window = scrive_core::padded_highlight_window(viewport, n);
    }

    /// Restart the sweep after an edit (fresh snapshot) AND repaint the viewport
    /// now — the pair must go together: a bare `start` leaves the visible rows
    /// showing pre-edit colours until the chain verifies down to them. The one
    /// owner of that ordering (no update arm sequences `start`/`speculate` by
    /// hand and risks getting it backwards, which `speculate`'s rev guard would
    /// silently turn into a no-op).
    fn restart(&mut self, doc: &mut Document, viewport: Range<u32>) {
        self.start(doc, viewport.clone());
        self.speculate(doc, viewport);
    }

    /// Re-aim the window on a scroll (no edit) AND repaint the newly-visible
    /// rows now — the same pairing as [`Self::restart`], one owner.
    fn reaim(&mut self, doc: &mut Document, viewport: Range<u32>) {
        let n = doc.buffer().line_count();
        self.set_window(viewport.clone(), n);
        self.speculate(doc, viewport);
    }

    /// Paint the viewport window immediately from a GUESSED fresh state,
    /// SYNCHRONOUSLY on the UI thread — one window of rows is a few ms, and the
    /// result colours on the next frame with no worker wait. Absorbed
    /// `verified=false`: spans show on the still-dirty rows, no checkpoints are
    /// planted, and the parallel chain re-verifies them per line (O(1) if the
    /// guess was right; a wrong guess is evicted and refilled). No-op once the
    /// sweep is done (the sync phase-2 refill from correct checkpoints serves
    /// moved windows then) or if the pool's snapshot is stale (an edit landed).
    fn speculate(&self, doc: &mut Document, viewport: Range<u32>) {
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
            scrive_core::SegmentStart::Fresh,
            Some(self.window.clone()),
            None,
        );
        doc.absorb_highlight(self.rev, seg, false);
    }

    /// Drain finished jobs and advance the verified chain, absorbing into
    /// `doc`. BUDGETED: at most `POLL_VERIFY_BUDGET` segments and
    /// `POLL_RERUN_BUDGET` synchronous mis-guess re-runs per frame, so one paint
    /// never absorbs an unbounded contiguous-ready run (nor bursts many syntect
    /// re-runs). A partial frame leaves `next_verify` where it stopped and stays
    /// `active`, so the `window::frames` subscription (keyed on `active`)
    /// re-fires and resumes next frame — zero new plumbing. Total verification
    /// is unchanged, paced over ⌈#segments/budget⌉ frames; `active=false` only
    /// once the whole document verifies.
    fn poll(&mut self, doc: &mut Document) {
        // Unbudgeted: draining the channel is a cheap pointer move per message
        // (Design 0's point), and a result left in the channel would stall the
        // chain that consumes it.
        while let Ok(done) = self.done_rx.try_recv() {
            if done.rev == self.rev {
                self.results[done.idx] = Some(done.seg);
            } // else stale — drop it
        }
        // `verified` counts every segment advanced this frame; `reruns` counts
        // only the heavy synchronous re-runs (weighted tighter — the frame's
        // real cost).
        let mut verified = 0usize;
        let mut reruns = 0usize;
        while self.next_verify < self.results.len() {
            if verified >= POLL_VERIFY_BUDGET {
                break; // paced out this frame; `active` stays true → resume next
            }
            // Peek WITHOUT taking: a not-yet-ready slot ends the contiguous run,
            // and a re-run we defer for budget must stay in `results` for the
            // next frame.
            let Some(seg_ref) = self.results[self.next_verify].as_ref() else { break };
            let true_start_fresh =
                self.next_verify == 0 || self.prev_end.as_ref() == Some(&self.fresh);
            // A Fresh-guessed segment is verified iff the true start is fresh;
            // otherwise the guess was wrong and it needs a synchronous re-run.
            let needs_rerun = seg_ref.started_fresh() && !true_start_fresh;
            if needs_rerun && reruns >= POLL_RERUN_BUDGET {
                break; // one more heavy re-run would exceed the frame budget
            }
            let seg = self.results[self.next_verify].take().expect("peeked Some");
            if needs_rerun {
                // Wrong guess: re-run from the true prior end SYNCHRONOUSLY. The
                // early-stop makes this near-instant in the common case (a Fresh
                // guess differs from the mid-document true state only in
                // syntect's first-line flag, so it re-converges at the first
                // checkpoint — ~one stride), so the chain does not wait for a
                // worker to free from the bulk sweep. A genuine deep-construct
                // mis-guess costs more here (bounded by the construct's close) —
                // the accepted degenerate case, now paced one-per-frame.
                let start = scrive_core::SegmentStart::After(
                    self.prev_end.clone().expect("prev segment is verified"),
                );
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
}

pub struct App {
    doc: Document,
    /// Whether the app-side find bar is open; its live query text.
    find_open: bool,
    find_query: String,
    /// The `Aa` / `ab|` / `.*` options. Each is part of the QUERY, not just
    /// chrome: flipping one changes what matches, so all of them re-scan through
    /// [`App::push_find_query`] exactly like a text change.
    find_case: bool,
    find_whole_word: bool,
    find_regex: bool,
    /// Whether the bar is expanded into find+replace (the chevron), and the live
    /// replacement text. Expansion deliberately SURVIVES a close/reopen (as
    /// mainstream editors do) — `close_find` drops the query, not the shape of
    /// the bar.
    replace_open: bool,
    replace_text: String,
    /// Monotonic clock start — `find`'s debounced re-scan needs a `now_ms`.
    start: Instant,
    /// The last visible buffer-row range the editor reported
    /// (`Action::ViewportChanged`) — the ONE viewport fact. It aims three things
    /// that must agree: the tokenize target (`viewport.end` — edits tokenize
    /// only down to here, so highlighting keeps pace with what's on screen), the
    /// core's highlight retention window, and the parallel sweep's `spans_for` /
    /// restart target. (One field, so a future re-aim path can't update one and
    /// desync the others.)
    viewport: Range<u32>,
    /// The completion controller (view-state) + the app's provider. The app
    /// owns both (the provider needs `&mut`, which `view()` can't give), drives
    /// them on edits/popup keys, and passes the popup list to the editor.
    completion: CompletionController,
    provider: StubCompletions,
    /// The active snippet tab-stop session, if any — owns one position-tracked
    /// range per stop; Tab/Shift+Tab move between them.
    snippet: Option<SnippetSession>,
    /// The signature-help provider (stub) + the current one-line box.
    sig_provider: StubSignatures,
    signature: Option<SignatureInfo>,
    /// The hover provider (stub) + the open hover popup.
    hover_provider: StubHover,
    hover: Option<HoverInfo>,
    /// The off-thread parallel/speculative highlight sweep — `Some` for a large
    /// document (`PARALLEL_MIN_BYTES`), created on the first viewport report;
    /// `None` for small docs (the synchronous path, unchanged). Aimed by the one
    /// [`App::viewport`] field, so its window can't drift from the core's.
    hl_pool: Option<HighlightPool>,
    /// The demo re-lint is debounced OFF the keystroke path: an edit only
    /// sets `relint_dirty` here (O(1)); a `now_ms`-gated tick (mirroring find's
    /// `maybe_rescan`) then runs the whole-document stub scan at most once per
    /// [`RELINT_DEBOUNCE_MS`], so a burst of keystrokes coalesces into one scan.
    /// `last_relint_ms` anchors the debounce window (the injected clock's stamp
    /// of the last completed scan). Between scans, diagnostics ride edits via the
    /// decoration mover's stickiness, so continuity holds.
    relint_dirty: bool,
    last_relint_ms: u64,
}

/// A stand-in `Hover` provider: a fixed Rust vocabulary → markdown doc, keyed by
/// the word (the trailing run of the lookback, which runs through the word end).
struct StubHover;

impl Hover for StubHover {
    fn hover(&mut self, cx: &HoverCx) -> Option<HoverInfo> {
        let rev: String = cx.lookback.chars().rev().take_while(|c| is_completion_word_char(*c)).collect();
        let word: String = rev.chars().rev().collect();
        let markdown = match word.as_str() {
            "fn" => "**fn** `name(args) -> ret` — defines a function.",
            "let" => "**let** — bind a value to a name; add `mut` to allow reassignment.",
            "mut" => "**mut** — make a binding or reference mutable.",
            "const" => "**const** — a compile-time constant.",
            "pub" => "**pub** — export an item from its module.",
            "use" => "**use** — bring a path into scope.",
            "mod" => "**mod** — declare a module.",
            "struct" => "**struct** `Name { fields }` — a named record type.",
            "enum" => "**enum** `Name { variants }` — a type that is one of several variants.",
            "impl" => "**impl** — associate methods or a trait with a type.",
            "trait" => "**trait** — a set of methods a type can implement.",
            "match" => "**match** `x { pat => expr, … }` — branch on a value's shape.",
            "for" => "**for** `x in iter { … }` — iterate over an iterator.",
            "while" => "**while** `cond { … }` — loop while the condition holds.",
            "loop" => "**loop** — repeat the body forever, until `break`.",
            "if" => "**if** `cond { … } else { … }` — a conditional.",
            "else" => "**else** — the branch taken when the `if` condition is false.",
            "return" => "**return** — return a value from the enclosing function.",
            "self" => "**self** — the receiver of a method.",
            "Self" => "**Self** — the type the enclosing `impl` block is for.",
            "as" => "**as** — a primitive cast, e.g. `n as f64`.",
            "Option" => "**Option<T>** — either `Some(T)` or `None`.",
            "Some" => "**Some(T)** — an `Option` that holds a value.",
            "None" => "**None** — an `Option` that holds nothing.",
            "Result" => "**Result<T, E>** — either `Ok(T)` or `Err(E)`.",
            "Vec" => "**Vec<T>** — a growable, heap-allocated array.",
            "VecDeque" => "**VecDeque<T>** — a double-ended queue.",
            "String" => "**String** — an owned, growable UTF-8 string.",
            "u8" => "**u8** — an unsigned 8-bit integer.",
            "u32" => "**u32** — an unsigned 32-bit integer.",
            "usize" => "**usize** — a pointer-sized unsigned integer.",
            "f64" => "**f64** — a 64-bit floating-point number.",
            "bool" => "**bool** — either `true` or `false`.",
            "str" => "**str** — a borrowed string slice.",
            _ => return None,
        };
        Some(HoverInfo { markdown: markdown.to_string(), range: cx.word.clone() })
    }
}

/// A stand-in `SignatureHelp` provider: a fixed table of Rust call signatures,
/// resolved from the enclosing call in the lookback.
struct StubSignatures;

impl SignatureHelp for StubSignatures {
    #[allow(clippy::single_range_in_vec_init)] // params is Vec<Range>; some calls have one
    fn signature(&mut self, cx: &SignatureCx) -> Option<SignatureInfo> {
        let (name, comma) = enclosing_call(&cx.lookback)?;
        let (label, params): (&str, Vec<Range<u32>>) = match name.as_str() {
            "new" => ("new(capacity: usize) -> Samples", vec![4..19]),
            "with_capacity" => ("with_capacity(capacity: usize) -> VecDeque<T>", vec![14..29]),
            "push" => ("push(&mut self, reading: f64)", vec![5..14, 16..28]),
            "push_back" => ("push_back(&mut self, value: T)", vec![10..19, 21..29]),
            "average" => ("average(&self) -> Option<f64>", vec![8..13]),
            _ => return None,
        };
        let active = comma.min(params.len().saturating_sub(1) as u32);
        let doc = match name.as_str() {
            "new" => Some("Create an empty window that keeps at most `capacity` readings.".to_string()),
            "push" => Some("Record one reading, evicting the oldest once the window is full.".to_string()),
            "average" => Some("The mean of the retained readings, or `None` when empty.".to_string()),
            _ => None,
        };
        Some(SignatureInfo { label: label.to_string(), params, active, doc })
    }
}

/// The innermost unclosed call in `lookback`: the callee name and the top-level
/// comma count before the caret (the active parameter). Depth-tracked through
/// `()` / `[]`.
fn enclosing_call(lookback: &str) -> Option<(String, u32)> {
    let chars: Vec<char> = lookback.chars().collect();
    let mut depth = 0i32;
    let mut commas = 0u32;
    let mut i = chars.len();
    while i > 0 {
        i -= 1;
        match chars[i] {
            ')' | ']' => depth += 1,
            '(' | '[' if depth > 0 => depth -= 1,
            '(' => {
                let mut j = i;
                while j > 0 && is_completion_word_char(chars[j - 1]) {
                    j -= 1;
                }
                let name: String = chars[j..i].iter().collect();
                return (!name.is_empty()).then_some((name, commas));
            }
            ',' if depth == 0 => commas += 1,
            _ => {}
        }
    }
    None
}

/// A stand-in `Completions` provider: a fixed Rust vocabulary the controller
/// prefix-filters.
struct StubCompletions {
    items: Vec<CompletionItem>,
}

impl StubCompletions {
    fn new() -> Self {
        let kw = |l: &str| CompletionItem::plain(l, CompletionKind::Keyword);
        let ty = |l: &str| CompletionItem::plain(l, CompletionKind::Type);
        Self {
            items: vec![
                // Bindings & items.
                kw("let"),
                kw("mut"),
                kw("pub"),
                kw("const"),
                kw("use"),
                kw("mod"),
                CompletionItem::new("fn", CompletionKind::Construct, InsertText::Snippet("fn ${1:name}(${2:args}) -> ${3:()} {\n\t$0\n}".into()))
                    .with_detail("name(args) -> ret")
                    .with_doc("Define a function."),
                CompletionItem::new("struct", CompletionKind::Construct, InsertText::Snippet("struct ${1:Name} {\n\t$0\n}".into()))
                    .with_detail("Name { fields }")
                    .with_doc("Define a record type."),
                CompletionItem::new("enum", CompletionKind::Construct, InsertText::Snippet("enum ${1:Name} {\n\t$0\n}".into()))
                    .with_detail("Name { variants }")
                    .with_doc("Define a variant type."),
                CompletionItem::new("impl", CompletionKind::Construct, InsertText::Snippet("impl ${1:Type} {\n\t$0\n}".into()))
                    .with_detail("Type { … }")
                    .with_doc("Associate methods with a type."),
                CompletionItem::new("trait", CompletionKind::Construct, InsertText::Snippet("trait ${1:Name} {\n\t$0\n}".into()))
                    .with_detail("Name { … }")
                    .with_doc("Define a trait."),
                // Control flow.
                CompletionItem::new("match", CompletionKind::Construct, InsertText::Snippet("match ${1:expr} {\n\t${2:pattern} => $0,\n}".into()))
                    .with_detail("expr { arms }")
                    .with_doc("Branch on a value's shape."),
                kw("if"),
                kw("else"),
                kw("for"),
                kw("while"),
                kw("loop"),
                kw("break"),
                kw("continue"),
                kw("return"),
                kw("as"),
                kw("where"),
                // Types.
                ty("u8"),
                ty("u16"),
                ty("u32"),
                ty("u64"),
                ty("usize"),
                ty("i32"),
                ty("f64"),
                ty("bool"),
                ty("char"),
                ty("str"),
                ty("String"),
                ty("Vec"),
                ty("Option"),
                ty("Result"),
            ],
        }
    }
}

impl Completions for StubCompletions {
    fn complete(&mut self, _cx: &CompletionCx) -> Vec<CompletionItem> {
        self.items.clone()
    }
}

/// What an applied action means for completion / signature help (captured before
/// the action is consumed by `apply`'s match).
#[derive(Clone, Copy)]
enum CompletionEvent {
    Typed(char),
    Deleting,
    CaretOrClose,
}

impl Default for App {
    fn default() -> Self {
        // `--large <MB>` swaps in the synthetic corpus (the field-gate doc).
        let sample;
        let source: &str = match LARGE_DOC.get() {
            Some(s) => s,
            None => {
                sample = long_sample();
                &sample
            }
        };
        let mut doc = Document::new(source).expect("sample fits the u32 offset space");
        // Grammar + theme are app-supplied assets: scrive-core is
        // language-agnostic and ships neither. The doc owns the incremental
        // highlight cache from here on.
        // The comment prefix is app-supplied language config, like the grammar.
        doc.set_line_comment(Some("//"));
        doc.set_syntax(
            SyntaxDef::from_sublime_syntax(include_str!("assets/rust.sublime-syntax"))
                .expect("bundled Rust grammar parses"),
            TokenTheme::from_tm_theme(include_str!("assets/scrive-dark.tmTheme"))
                .expect("bundled theme parses"),
        );
        // No load-time tokenization: the first
        // `ViewportChanged` tokenizes the visible rows, and the idle sweep
        // subscription (window::frames, active only while a dirty frontier
        // remains) progresses the rest in budgeted batches. Untokenized rows
        // render in the fallback style — never a stall.
        let mut app = Self {
            doc,
            find_open: false,
            find_query: String::new(),
            find_case: false,
            find_whole_word: false,
            find_regex: false,
            replace_open: false,
            replace_text: String::new(),
            start: Instant::now(),
            viewport: 0..0,
            completion: CompletionController::new(),
            provider: StubCompletions::new(),
            snippet: None,
            sig_provider: StubSignatures,
            signature: None,
            hover_provider: StubHover,
            hover: None,
            hl_pool: None,
            relint_dirty: false,
            last_relint_ms: 0,
        };
        app.relint(); // seed the stub diagnostics (the sample carries a TODO)
        app
    }
}

#[derive(Debug, Clone)]
pub enum Msg {
    Editor(Action),
    /// Open the find bar (Ctrl+F) and focus its input.
    OpenFind,
    /// Open the find bar with the replace row already expanded (Ctrl+H).
    OpenReplace,
    /// Close the find bar (Escape / ✕), clear the query, refocus the editor.
    CloseFind,
    /// The find query text changed — re-scan synchronously.
    FindQuery(String),
    /// The `Aa` / `ab|` / `.*` option toggles: flip one and re-scan.
    ToggleCase,
    ToggleWholeWord,
    ToggleRegex,
    /// The find-in-selection toggle: scope find to the current selection, or
    /// clear the scope if one is already set.
    ToggleFindInSelection,
    /// Go to the next / previous match (Enter / Shift+Enter / buttons).
    FindNext,
    FindPrev,
    /// Alt+Enter in the find bar: every live match becomes a selection.
    FindSelectAll,
    /// The chevron: expand/collapse the bar's replace row.
    ToggleReplace,
    /// The replacement text changed — no scan; the query is untouched.
    ReplaceText(String),
    /// Replace the active match and advance (Enter in the replace input / the
    /// replace button).
    ReplaceOne,
    /// Replace every match in one undo step (the replace-all button).
    ReplaceAll,
    /// One idle-sweep tick (window::frames while a dirty highlight frontier
    /// remains): tokenize the next budgeted batch toward convergence.
    HighlightSweep,
    /// One debounced re-lint tick (window::frames while `relint_dirty`): run the
    /// whole-document stub lint if the debounce window has elapsed, else a no-op.
    /// The tick self-cancels once the scan clears `relint_dirty`.
    MaybeRelint,
}

impl App {
    /// Close the find bar: drop the query text and its matches/decorations. The
    /// ONE owner of "close find" — both Escape paths (editor-focused `Collapse`
    /// and input-focused `CloseFind`, which a single Escape fires together) call
    /// it, so they cannot diverge into two meanings of "close".
    fn close_find(&mut self) {
        self.find_open = false;
        self.find_query.clear();
        self.doc.set_find_query(None, self.now_ms());
    }

    /// Push the live query text + its options into the document — the ONE place
    /// a [`FindQuery`] is built.
    ///
    /// Every input that can change what matches routes through here (the text,
    /// and each option toggle), so the bar's controls cannot disagree about the
    /// live query: an option is not chrome, it is part of the query, and
    /// flipping one re-scans exactly like typing does. An empty text means "no
    /// query" — never match-all.
    fn push_find_query(&mut self) {
        let query = (!self.find_query.is_empty()).then(|| FindQuery {
            text: self.find_query.clone(),
            case_sensitive: self.find_case,
            whole_word: self.find_whole_word,
            regex: self.find_regex,
        });
        let now = self.now_ms();
        self.doc.set_find_query(query, now); // synchronous scan
    }

    pub fn update(&mut self, msg: Msg) -> Task<Msg> {
        match msg {
            // The editor reported a new visible range (scroll / resize /
            // autoscroll): aim the highlight retention window there
            // (virtualization — spans/states are kept only around the
            // viewport) and tokenize highlights down to its end. Not an
            // edit — no find re-scan, no history — so it bypasses `apply`.
            Msg::Editor(Action::ViewportChanged(rows)) => {
                // The one viewport fact feeds all three aims (field, core window,
                // pool) from here so they can't diverge.
                self.viewport = rows.clone();
                self.doc.set_highlight_window(rows.clone());
                if self.doc.buffer().len() >= PARALLEL_MIN_BYTES {
                    // Large document: the off-thread sweep owns dirt-clearing;
                    // the viewport is PAINTED synchronously now (a few ms for
                    // one window) and verified in place by the chain. Do NOT run
                    // the whole-doc synchronous walk here (it would race the
                    // pool). Create the pool on first sight.
                    let App { hl_pool, doc, .. } = self;
                    match hl_pool {
                        Some(pool) => pool.reaim(doc, rows),
                        None => {
                            if let Some(pool) = HighlightPool::new(doc, rows.clone()) {
                                pool.speculate(doc, rows);
                                *hl_pool = Some(pool);
                            }
                        }
                    }
                } else {
                    // Small document: the synchronous path, unchanged.
                    self.doc.tokenize_highlight(rows.end);
                }
                self.hover = None; // scroll closes the hover
                Task::none()
            }
            // Completion popup navigation (captured by the editor while open) —
            // drive the controller; no document edit except on accept.
            Msg::Editor(Action::PopupUp) => {
                self.completion.move_selection(false);
                Task::none()
            }
            Msg::Editor(Action::PopupDown) => {
                self.completion.move_selection(true);
                Task::none()
            }
            Msg::Editor(Action::PopupDismiss) => {
                self.completion.escape();
                Task::none()
            }
            Msg::Editor(Action::PopupAccept) => {
                self.accept_completion();
                Task::none()
            }
            Msg::Editor(Action::PopupClickAccept(idx)) => {
                self.completion.set_selected(idx);
                self.accept_completion();
                Task::none()
            }
            // Snippet tab-stop navigation (captured by the editor while a session
            // is active).
            Msg::Editor(Action::SnippetTab) => {
                self.snippet_tab(true);
                Task::none()
            }
            Msg::Editor(Action::SnippetTabPrev) => {
                self.snippet_tab(false);
                Task::none()
            }
            Msg::Editor(Action::SnippetCancel) => {
                if let Some(mut s) = self.snippet.take() {
                    s.cancel(self.doc.decorations_mut());
                }
                Task::none()
            }
            Msg::Editor(Action::SignatureClose) => {
                self.signature = None;
                Task::none()
            }
            // Hover: the pointer rested over `offset` — diagnostics first
            // (the message rides the decoration store, so hovering a
            // squiggle shows why it's flagged), then the word provider's docs.
            Msg::Editor(Action::HoverQuery(offset)) => {
                let diags: Vec<(Range<u32>, String)> = self
                    .doc
                    .diagnostics_in(offset..offset + 1)
                    .map(|(r, sev, msg)| (r, format!("**{}:** {msg}", severity_label(sev))))
                    .collect();
                let cx = self.build_hover_cx(offset);
                let word = (cx.word.start != cx.word.end).then(|| self.hover_provider.hover(&cx)).flatten();
                self.hover = if diags.is_empty() {
                    word
                } else {
                    // The popup anchors on the squiggled span; provider docs
                    // (if any) append below the diagnostic messages.
                    let range = diags[0].0.clone();
                    let mut md: Vec<String> = diags.into_iter().map(|(_, m)| m).collect();
                    if let Some(w) = word {
                        md.push(String::new());
                        md.push(w.markdown);
                    }
                    Some(HoverInfo { markdown: md.join("\n"), range })
                };
                Task::none()
            }
            Msg::Editor(Action::HoverDismiss) => {
                self.hover = None;
                Task::none()
            }
            Msg::Editor(Action::ToggleFold { opener }) => {
                self.doc.toggle_fold_opener(opener);
                Task::none()
            }
            Msg::Editor(Action::FoldAtCarets { unfold }) => {
                self.doc.fold_at_carets(unfold);
                Task::none()
            }
            // Escape with the find bar open closes the BAR and keeps the
            // selections (matching mainstream editors) — vital after
            // Alt+Enter built a multi-caret set the plain Collapse would
            // destroy. (The subscription's CloseFind covers the input-focused
            // press; this covers the editor-focused one.)
            Msg::Editor(Action::Collapse) if self.find_open => {
                self.close_find();
                Task::none()
            }
            Msg::Editor(action) => {
                self.apply(action);
                Task::none()
            }
            Msg::OpenFind => {
                self.find_open = true;
                // Seed (or RE-seed) the query from a non-empty, single-line editor
                // selection (as mainstream editors do): Ctrl+F searches the selection, and pressing
                // it again after selecting something else replaces the query. An
                // empty selection leaves the current query untouched.
                let sel = self.doc.selections().newest();
                let seed = (!sel.is_empty())
                    .then(|| self.doc.buffer().slice(sel.start()..sel.end()).into_owned())
                    .filter(|t| !t.contains('\n'));
                if let Some(text) = seed {
                    self.find_query = text;
                    self.push_find_query();
                }
                focus(FIND_INPUT) // focus the input, unfocusing the editor
            }
            Msg::OpenReplace => {
                // Ctrl+H is Ctrl+F with the replace row already out. The query
                // seeding and focus rules are `OpenFind`'s job — reuse them
                // rather than growing a second copy that can drift.
                self.replace_open = true;
                self.update(Msg::OpenFind)
            }
            Msg::CloseFind if self.find_open => {
                self.close_find(); // drop matches + decorations
                focus(EDITOR) // hand focus back to the editor
            }
            Msg::FindQuery(q) => {
                self.find_query = q;
                self.push_find_query();
                Task::none()
            }
            Msg::ToggleCase => {
                self.find_case = !self.find_case;
                self.push_find_query(); // the flag changes what matches: re-scan
                Task::none()
            }
            Msg::ToggleWholeWord => {
                self.find_whole_word = !self.find_whole_word;
                self.push_find_query();
                Task::none()
            }
            Msg::ToggleRegex => {
                self.find_regex = !self.find_regex;
                self.push_find_query();
                Task::none()
            }
            Msg::ToggleFindInSelection if self.find_open => {
                // The scope lives in the document (it has to ride every edit),
                // so there is no app-side copy to flip — read the live one and
                // set its opposite. A scope the user cannot produce (no
                // selection) simply leaves find unscoped.
                let scope = if self.doc.find_scope().is_some() {
                    None
                } else {
                    let sel = self.doc.selections().newest();
                    (!sel.is_empty()).then(|| sel.start()..sel.end())
                };
                let now = self.now_ms();
                self.doc.set_find_scope(scope, now);
                Task::none()
            }
            Msg::FindNext if self.find_open => {
                let now = self.now_ms();
                self.doc.find_next(now);
                Task::none()
            }
            Msg::FindPrev if self.find_open => {
                let now = self.now_ms();
                self.doc.find_prev(now);
                Task::none()
            }
            Msg::FindSelectAll if self.find_open => {
                // Every match becomes a caret; typing replaces them all. Focus
                // returns to the editor (matching mainstream editors); the bar stays open.
                if self.doc.select_find_matches() {
                    return focus(EDITOR);
                }
                Task::none()
            }
            Msg::ToggleReplace => {
                self.replace_open = !self.replace_open;
                // Expanding puts the caret where the user is about to type;
                // collapsing hands it back to the query rather than stranding
                // focus on a row that no longer exists.
                focus(if self.replace_open { REPLACE_INPUT } else { FIND_INPUT })
            }
            Msg::ReplaceText(t) => {
                self.replace_text = t;
                Task::none() // the query is untouched: nothing to re-scan
            }
            Msg::ReplaceOne if self.find_open => {
                let now = self.now_ms();
                let before = self.doc.revision();
                self.doc.replace_next(&self.replace_text, now);
                // The first press only NAVIGATES (it shows the match before
                // overwriting it), which commits nothing — running the post-edit
                // tail then would arm a re-lint for an edit that never happened.
                if self.doc.revision() != before {
                    self.after_edit(CompletionEvent::CaretOrClose);
                }
                Task::none()
            }
            Msg::ReplaceAll if self.find_open => {
                if self.doc.replace_all(&self.replace_text) > 0 {
                    self.after_edit(CompletionEvent::CaretOrClose);
                }
                Task::none()
            }
            // Find-nav keys arriving while the bar is closed: ignore (the guards
            // above only match when `find_open`). The editor still handles its own
            // Enter/Escape via the widget when it holds focus.
            // The idle sweep: one budgeted batch per frame toward the whole
            // document; the subscription drops itself once converged.
            Msg::HighlightSweep => {
                let App { doc, hl_pool, viewport, .. } = self;
                // The pool drives only while the document is still large; if an
                // edit shrank it below the threshold, it falls to the sync path
                // and the pool goes inert (its last sweep finishes, no restart)
                // — so it never redundantly re-sweeps a now-small document.
                // It resumes if the document grows back.
                let large = doc.buffer().len() >= PARALLEL_MIN_BYTES;
                match hl_pool {
                    Some(pool) if large => {
                        if pool.rev != doc.revision() {
                            // An edit landed: re-sweep from a fresh snapshot AND
                            // repaint the viewport now (the verified prefix in the
                            // cache survives). The restart+repaint pairing lives in
                            // one owner (`HighlightPool::restart`), so an edit's
                            // colours land next frame instead of waiting for the
                            // chain to verify down to the visible segment.
                            pool.restart(doc, viewport.clone());
                        } else {
                            // Always poll: drains finished jobs (so nothing
                            // piles up) and advances the verified chain; a no-op
                            // once the sweep is done.
                            pool.poll(doc);
                        }
                        // Once the sweep is idle, the synchronous phase-2 path
                        // refills any window rows the sweep evicted (from the
                        // now-correct checkpoints). It can't race the pool: all
                        // dirt is cleared, so the dirty walk is a no-op and only
                        // the window refill runs.
                        if !pool.active {
                            let n = doc.buffer().line_count();
                            doc.tokenize_highlight(n);
                        }
                    }
                    other => {
                        // Small document (shrunk below the threshold, or never
                        // large): deactivate any lingering pool so the frame
                        // subscription can settle to idle-zero-work, and drive
                        // the sync path.
                        if let Some(pool) = other {
                            pool.active = false;
                        }
                        let n = doc.buffer().line_count();
                        doc.tokenize_highlight(n);
                    }
                }
                Task::none()
            }
            // The debounced re-lint tick (window::frames while `relint_dirty`):
            // run the whole-document stub scan iff the window elapsed, then let
            // the subscription drop itself once the flag clears.
            Msg::MaybeRelint => {
                self.maybe_relint(self.now_ms());
                Task::none()
            }
            Msg::CloseFind
            | Msg::FindNext
            | Msg::FindPrev
            | Msg::FindSelectAll
            | Msg::ReplaceOne
            | Msg::ReplaceAll
            | Msg::ToggleFindInSelection => Task::none(),
        }
    }

    /// Milliseconds since app start — the injected clock for find's debounce.
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// The debounced trailing edge of the demo re-lint: if an edit marked
    /// the document dirty and the debounce window has elapsed on the injected
    /// clock, run the whole-document stub scan once and re-arm the window.
    /// Mirrors [`Document::maybe_rescan_find`]. Returns whether it scanned (the
    /// debounce test observes this; the app ignores it).
    fn maybe_relint(&mut self, now: u64) -> bool {
        if !self.relint_dirty || now.saturating_sub(self.last_relint_ms) < RELINT_DEBOUNCE_MS {
            return false;
        }
        self.relint();
        self.relint_dirty = false;
        self.last_relint_ms = now;
        true
    }

    fn apply(&mut self, action: Action) {
        // Capture what this action means for completion before the match
        // consumes `action`.
        let comp_event = match &action {
            Action::Type(c) => CompletionEvent::Typed(*c),
            Action::Backspace | Action::DeleteWordBack | Action::Delete | Action::DeleteWordForward => {
                CompletionEvent::Deleting
            }
            _ => CompletionEvent::CaretOrClose,
        };
        match action {
            Action::Type(ch) => self.doc.type_char(ch),
            Action::Backspace => self.doc.backspace(),
            Action::Delete => self.doc.delete_forward(),
            Action::DeleteWordBack => self.doc.delete_word_back(),
            Action::DeleteWordForward => self.doc.delete_word_forward(),
            Action::Enter => self.doc.enter(),
            Action::Tab => self.doc.tab(),
            Action::Outdent => self.doc.outdent(),
            Action::ToggleComment => self.doc.toggle_line_comment(),
            Action::DeleteLine => self.doc.delete_line(),
            Action::InsertLine { down } => self.doc.insert_line(down),
            Action::Cut => self.doc.cut(),
            Action::Paste { text, entire_line } => self.doc.paste(&text, entire_line),
            Action::Move { motion, extend } => self.doc.move_carets(motion, extend),
            Action::PlaceCaret(offset) => {
                // A single caret at the clicked offset.
                use scrive_core::{Selection, SelectionId, SelectionSet};
                let mut set = SelectionSet::new(0);
                set.set_single(Selection::caret(SelectionId(0), offset));
                // Replace via a no-op-safe path: move a fresh set in.
                self.doc.set_selections(set);
            }
            Action::DragSelect { granularity, origin, head } => {
                self.doc.drag_select(granularity, origin, head);
            }
            Action::AddCaret(offset) => self.doc.add_caret(offset),
            Action::AddNextOccurrence => self.doc.add_next_occurrence(),
            Action::SelectAllOccurrences => self.doc.select_all_occurrences(),
            Action::AddCaretVertical { down } => self.doc.add_caret_vertical(down),
            Action::JumpToBracket => self.doc.jump_to_bracket(),
            Action::NextDiagnostic { forward } => {
                self.doc.next_diagnostic(forward);
            }
            Action::ExpandSelection => self.doc.expand_selection(),
            Action::ShrinkSelection => self.doc.shrink_selection(),
            Action::Collapse => self.doc.collapse_selections(),
            Action::SelectAll => self.doc.select_all(),
            Action::Undo => {
                self.doc.undo();
            }
            Action::Redo => {
                self.doc.redo();
            }
            Action::ColumnSelect(dir) => self.doc.column_select(dir),
            Action::ColumnDrag { anchor, active } => self.doc.column_drag(anchor, active),
            Action::MoveLine { down } => self.doc.move_line(down),
            Action::CopyLine { down } => self.doc.copy_line(down),
            // Handled in `update` before reaching `apply` (viewport report +
            // popup / snippet navigation); unreachable here but keeps the match
            // exhaustive.
            Action::ViewportChanged(_)
            | Action::PopupUp
            | Action::PopupDown
            | Action::PopupAccept
            | Action::PopupClickAccept(_)
            | Action::PopupDismiss
            | Action::SnippetTab
            | Action::SnippetTabPrev
            | Action::SnippetCancel
            | Action::SignatureClose
            | Action::HoverQuery(_)
            | Action::HoverDismiss
            | Action::ToggleFold { .. }
            | Action::FoldAtCarets { .. } => {} // handled in update() before apply()
        }
        self.after_edit(comp_event);
    }

    /// The find bar's option toggles + the nav/close cluster share this — the
    /// find-in-selection button latches off the DOCUMENT's live scope, not an
    /// app-side copy, so an edit that collapses the scope un-latches the button
    /// with no code here at all.
    fn scoped(&self) -> bool {
        self.doc.find_scope().is_some()
    }

    /// Everything that must follow a document mutation — the ONE owner of the
    /// post-edit tail. Both edit entry points run it: the keystroke path
    /// ([`App::apply`]) and the replace verbs. Keeping it in one place is what
    /// stops a second entry point from silently doing half of it (a replace that
    /// forgot `tokenize_highlight` would paint stale colors; one that forgot
    /// `relint_dirty` would strand the diagnostics).
    fn after_edit(&mut self, comp_event: CompletionEvent) {
        // Bring the incremental highlight cache current down to the reported
        // viewport bottom only — convergence stops at the edited lines for a
        // normal edit, and the viewport bound caps a state-cascade to the screen.
        self.doc.tokenize_highlight(self.viewport.end);
        // Keep find fresh while editing: matches ride the edit via
        // the decoration mover; a debounced re-scan then picks up appearing /
        // disappearing matches (incl. anything an undo restored).
        let now = self.now_ms();
        self.doc.maybe_rescan_find(now);
        // Drive completion off the edit: typing opens/filters, deleting
        // refilters, anything else closes the popup.
        self.drive_completion(comp_event);
        // Drive signature help: `(` opens the box; while open, edits/moves
        // re-query (a `None` reply closes it).
        self.drive_signature(comp_event);
        // Cancel a snippet session if the caret/edit left every stop.
        self.reconcile_snippet();
        // Any edit closes an open hover.
        self.hover = None;
        // Re-lint after the edit, but OFF the keystroke path: only mark the
        // document dirty (O(1)). A `now_ms`-gated `window::frames` tick then runs
        // the whole-document stub scan at most once per `RELINT_DEBOUNCE_MS`
        // (`maybe_relint`), so a keystroke never triggers the O(N log N) scan
        // directly. Diagnostics ride the edit via the decoration
        // mover's stickiness until the next scan, so continuity holds.
        self.relint_dirty = true;
    }

    /// A stub diagnostics pass — a stand-in for the app-side compile loop:
    /// flags `FIXME` (error) and `TODO` (warning) markers so squiggles, the
    /// diagnostic hover, F8 navigation, and the scrollbar marks are all
    /// exercisable in the demo without a real language server. Scans line by
    /// line (the markers never span a newline): typical rows borrow straight
    /// from the rope, so an edit never pays a whole-document copy.
    fn relint(&mut self) {
        if self.doc.buffer().len() > RELINT_MAX_BYTES {
            return; // demo-lint threshold — see RELINT_MAX_BYTES
        }
        // Count each whole-document scan the debounce coalesces (post
        // size-guard, so it tallies only real scans).
        #[cfg(test)]
        RELINT_RUNS.with(|c| c.set(c.get() + 1));
        let mut diags = Vec::new();
        {
            let buffer = self.doc.buffer();
            for row in 0..buffer.line_count() {
                let line = buffer.line(row);
                let line_start = buffer.point_to_offset(Point::new(row, 0));
                for (needle, sev, msg) in [
                    ("FIXME", Severity::Error, "unresolved FIXME (demo lint)"),
                    ("TODO", Severity::Warning, "unresolved TODO (demo lint)"),
                ] {
                    let mut from = 0usize;
                    while let Some(i) = line[from..].find(needle) {
                        let start = line_start + (from + i) as u32;
                        diags.push(Diagnostic::new(start..start + needle.len() as u32, sev, msg));
                        from = from + i + needle.len();
                    }
                }
            }
        }
        let rev = self.doc.revision();
        let _ = self.doc.set_diagnostics(rev, diags);
    }

    /// The word range around `offset` (scanning both directions with
    /// `is_completion_word_char`) — the word under the hover pointer.
    fn word_around(&self, offset: u32) -> Range<u32> {
        let p = self.doc.buffer().offset_to_point(offset);
        let line = self.doc.buffer().line(p.row);
        let line_start = offset - p.col;
        let col = p.col as usize;
        let start = line[..col]
            .char_indices()
            .rev()
            .take_while(|(_, c)| is_completion_word_char(*c))
            .last()
            .map_or(col, |(i, _)| i);
        let mut end = col;
        for c in line[col..].chars().take_while(|c| is_completion_word_char(*c)) {
            end += c.len_utf8();
        }
        (line_start + start as u32)..(line_start + end as u32)
    }

    /// Build a hover request for the word under `offset`. The lookback runs
    /// through the word's END, so the provider reads the full word from its tail.
    fn build_hover_cx(&self, offset: u32) -> HoverCx {
        let word = self.word_around(offset);
        let position = self.doc.buffer().offset_to_point(offset);
        let start_row = position.row.saturating_sub(LOOKBACK_LINES - 1);
        let lb_start = self.doc.buffer().point_to_offset(Point::new(start_row, 0));
        HoverCx {
            doc: self.doc.buffer().doc_id(),
            revision: self.doc.revision().0,
            position,
            word: word.clone(),
            lookback: self.doc.buffer().slice(lb_start..word.end).into_owned(),
        }
    }

    /// Drive the signature-help box: `(` opens it; while open, every
    /// relevant edit/move re-queries and a `None` reply closes it.
    fn drive_signature(&mut self, event: CompletionEvent) {
        let query = matches!(event, CompletionEvent::Typed('(')) || self.signature.is_some();
        if query {
            let cx = self.build_sig_cx();
            self.signature = self.sig_provider.signature(&cx);
        }
    }

    /// Build a signature request from the current document state.
    fn build_sig_cx(&self) -> SignatureCx {
        let head = self.doc.selections().newest().head();
        let position = self.doc.buffer().offset_to_point(head);
        let start_row = position.row.saturating_sub(LOOKBACK_LINES - 1);
        let lb_start = self.doc.buffer().point_to_offset(Point::new(start_row, 0));
        SignatureCx {
            doc: self.doc.buffer().doc_id(),
            revision: self.doc.revision().0,
            position,
            lookback: self.doc.buffer().slice(lb_start..head).into_owned(),
        }
    }

    /// Drive the completion controller after an edit.
    fn drive_completion(&mut self, event: CompletionEvent) {
        match event {
            CompletionEvent::Typed(c) => {
                let trigger = if is_completion_word_char(c) {
                    CompletionTrigger::Typed(c)
                } else if matches!(c, '(' | ',' | '=' | ':' | '.' | ' ') {
                    CompletionTrigger::TriggerChar(c)
                } else {
                    self.completion.on_boundary();
                    return;
                };
                let cx = self.build_cx(trigger);
                let word = self.completion_word_text();
                self.completion.on_input(&cx, &word, &mut self.provider);
            }
            CompletionEvent::Deleting => {
                if self.completion.is_open() {
                    let word = self.completion_word_text();
                    match word.chars().last() {
                        Some(c) => {
                            let cx = self.build_cx(CompletionTrigger::Typed(c));
                            self.completion.on_input(&cx, &word, &mut self.provider);
                        }
                        None => self.completion.close(),
                    }
                }
            }
            CompletionEvent::CaretOrClose => self.completion.close(),
        }
    }

    /// Accept the popup's selected item: replace the completion word with
    /// the item's insertion (a snippet expands, caret at its final stop), sealed
    /// as one edit; a snippet expands and starts an interactive tab-stop session,
    /// selecting the first stop. Fires the retrigger if requested.
    fn accept_completion(&mut self) {
        let Some(item) = self.completion.accept() else { return };
        let replace = item.replace.clone().unwrap_or_else(|| self.completion_word());
        self.set_selection_range(replace.clone());

        // Insert the item, capturing the expanded snippet (if any) to start a
        // session from.
        let expanded = match &item.insert {
            InsertText::Plain(s) => {
                self.doc.insert_text(s);
                None
            }
            InsertText::Snippet(body) => match Snippet::parse(body) {
                Ok(snip) => {
                    let indent = self.line_indent(replace.start);
                    let e = snip.for_insertion(&indent, default_indent_size() as usize);
                    self.doc.insert_text(&e.text);
                    Some(e)
                }
                Err(_) => {
                    self.doc.insert_text(body);
                    None
                }
            },
        };

        // Cancel any prior session; start a new one for a multi-stop snippet.
        if let Some(mut s) = self.snippet.take() {
            s.cancel(self.doc.decorations_mut());
        }
        if let Some(e) = expanded {
            match SnippetSession::start(&e, replace.start, self.doc.decorations_mut()) {
                Some((session, first)) => {
                    self.set_selection_range(first);
                    self.snippet = Some(session);
                }
                None => {
                    // Only a final stop — collapse the caret there.
                    let fin = e.stops.last().map_or(e.text.len() as u32, |s| s.range.start);
                    self.set_caret(replace.start + fin);
                }
            }
        }
        self.doc.tokenize_highlight(self.viewport.end);
        self.doc.maybe_rescan_find(self.now_ms());

        if item.retrigger && self.snippet.is_none() {
            let cx = self.build_cx(CompletionTrigger::Manual);
            let word = self.completion_word_text();
            self.completion.on_input(&cx, &word, &mut self.provider);
        }
    }

    /// Tab / Shift+Tab through the active snippet session: select the next
    /// stop, or end the session at the final stop.
    fn snippet_tab(&mut self, forward: bool) {
        let Some(mut session) = self.snippet.take() else { return };
        match session.tab(forward, self.doc.decorations_mut()) {
            TabOutcome::Move(range) => {
                self.set_selection_range(range);
                self.snippet = Some(session);
            }
            TabOutcome::Finish(offset) => self.set_caret(offset),
            TabOutcome::Stay => self.snippet = Some(session),
        }
    }

    /// Cancel the snippet session if the primary caret has left every stop (a
    /// caret move or an edit outside all stops). Never a transaction.
    fn reconcile_snippet(&mut self) {
        if self.snippet.is_none() {
            return;
        }
        let head = self.doc.selections().newest().head();
        let escaped = self.snippet.as_ref().unwrap().edit_escapes(&(head..head), self.doc.decorations());
        if escaped {
            let mut s = self.snippet.take().unwrap();
            s.cancel(self.doc.decorations_mut());
        }
    }

    /// Move the primary caret to `offset`.
    fn set_caret(&mut self, offset: u32) {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::caret(SelectionId(0), offset));
        self.doc.set_selections(set);
    }

    /// Select `range` as the primary selection.
    fn set_selection_range(&mut self, range: Range<u32>) {
        let mut set = SelectionSet::new(0);
        set.set_single(Selection::from_anchor(SelectionId(0), range.start, range.end));
        self.doc.set_selections(set);
    }

    /// The completion-word byte range ending at the primary caret (empty at a
    /// boundary), computed with `is_completion_word_char`.
    fn completion_word(&self) -> Range<u32> {
        let head = self.doc.selections().newest().head();
        let p = self.doc.buffer().offset_to_point(head);
        let line_start = head - p.col;
        let prefix = &self.doc.buffer().line(p.row)[..p.col as usize];
        let word_start = prefix
            .char_indices()
            .rev()
            .take_while(|(_, c)| is_completion_word_char(*c))
            .last()
            .map_or(prefix.len(), |(i, _)| i);
        (line_start + word_start as u32)..head
    }

    /// The text of the completion word under the caret.
    fn completion_word_text(&self) -> String {
        let w = self.completion_word();
        self.doc.buffer().slice(w.start..w.end).into_owned()
    }

    /// The leading whitespace of the line containing byte `offset`.
    fn line_indent(&self, offset: u32) -> String {
        let row = self.doc.buffer().offset_to_point(offset).row;
        self.doc.buffer().line(row).chars().take_while(|c| *c == ' ' || *c == '\t').collect()
    }

    /// Build a completion request from the current document state.
    fn build_cx(&self, trigger: CompletionTrigger) -> CompletionCx {
        let head = self.doc.selections().newest().head();
        let position = self.doc.buffer().offset_to_point(head);
        // The lookback slice (up to `LOOKBACK_LINES`) is only read when the
        // provider will actually run. A word char typed while the popup is
        // already open just refilters locally (`CompletionController::on_input`)
        // and never reads it — so skip the up-to-40-line slice copy on that hot
        // per-keystroke path; build it only when a provider call is coming.
        let refilter_only =
            self.completion.is_open() && matches!(trigger, CompletionTrigger::Typed(_));
        let lookback = if refilter_only {
            String::new()
        } else {
            let start_row = position.row.saturating_sub(LOOKBACK_LINES - 1);
            let lb_start = self.doc.buffer().point_to_offset(Point::new(start_row, 0));
            self.doc.buffer().slice(lb_start..head).into_owned()
        };
        CompletionCx {
            doc: self.doc.buffer().doc_id(),
            revision: self.doc.revision().0,
            position,
            word: self.completion_word(),
            lookback,
            trigger,
        }
    }

    pub fn view(&self) -> Element<'_, Msg> {
        // Addressable by id (focus + future multi-pane); the editor reads
        // highlight spans straight from the document's cache and the completion
        // popup from the controller.
        let popup = match self.completion.state() {
            CompletionState::Open(list) => Some(list),
            _ => None,
        };
        let editor = Editor::new(&self.doc, Msg::Editor)
            .popup(popup)
            .snippet_active(self.snippet.is_some())
            .signature(self.signature.as_ref())
            .hover(self.hover.as_ref())
            .id(EDITOR);
        if self.find_open {
            // Float the find bar over the editor, top-right (where mainstream
            // editors place it): a Stack layer, right-aligned with a margin. The right margin clears
            // the vertical scrollbar lane so the bar never sits over it. The
            // overlay is transparent except the bar, so clicks pass through.
            let overlay = container(self.find_bar())
                .width(Length::Fill)
                .align_x(Horizontal::Right)
                .padding(iced::Padding::new(8.0).right(8.0 + scrive_iced::SCROLLBAR_WIDTH));
            stack([editor.into(), overlay.into()]).into()
        } else {
            editor.into()
        }
    }

    /// The app-side find bar: a floating panel holding a query row (input +
    /// match count + prev/next/close) and, once the chevron expands it, a
    /// replace row (input + replace/replace-all). The widget owns no find keys —
    /// the app drives the Document methods.
    fn find_bar(&self) -> Element<'_, Msg> {
        let count = self.doc.find_match_count();
        // A half-typed regex (`(`, `[a-`) is a NORMAL state, not a failure — but
        // it must say so. Reporting the honest zero matches as "No results"
        // would imply the pattern is fine and the document simply lacks it.
        let invalid = self.doc.find_pattern_error().is_some();
        let label = if invalid {
            "Invalid regex".to_string()
        } else {
            match self.doc.active_find_match() {
                Some(i) => format!("{} of {}", i + 1, count),
                None if count > 0 => format!("{count} matches"),
                None if self.find_query.is_empty() => String::new(),
                None => "No results".to_string(),
            }
        };
        // Error red for anything the user has to notice or fix; otherwise the
        // neutral `#CCCCCC`.
        let label_color = if invalid || label == "No results" {
            Color::from_rgb8(0xF4, 0x87, 0x71)
        } else {
            Color::from_rgb8(0xCC, 0xCC, 0xCC)
        };
        // Both inputs share this width, and both rows a count-slot width, so the
        // two rows' columns line up under each other.
        const INPUT_W: f32 = 220.0;
        // Fixed and centered so the panel never reflows as the match count's
        // digit count changes (e.g. "1 of 5" ↔ "50 of 500"). Sized to fit the
        // widest label — "10000 of 10000" at the FIND_MATCH_CAP. The replace row
        // mirrors it with an EMPTY slot, which is what puts its buttons under
        // the nav buttons.
        const COUNT_W: f32 = 90.0;
        // The one gap between every control in the panel — shared so the
        // derived toggle-cluster width below matches the row it mirrors.
        const SPACING: f32 = 4.0;
        // Colors are neutral dark-theme grays (theme-agnostic) so the find bar
        // reads as standard editor chrome over the dark editor theme.
        //
        // ONE style for BOTH inputs — it captures nothing, so it is `Copy` and
        // can be handed to each. The query and replacement boxes cannot drift
        // apart visually.
        let input_style = |_theme: &Theme, status: text_input::Status| {
            let focused = matches!(status, text_input::Status::Focused { .. });
            text_input::Style {
                background: Color::from_rgb8(0x3C, 0x3C, 0x3C).into(),
                border: iced::border::rounded(4.0).width(1.0).color(if focused {
                    Color::from_rgb8(0x00, 0x7F, 0xD4) // focus ring
                } else {
                    Color::TRANSPARENT
                }),
                icon: Color::from_rgb8(0xCC, 0xCC, 0xCC),
                placeholder: Color::from_rgb8(0xA6, 0xA6, 0xA6),
                value: Color::from_rgb8(0xCC, 0xCC, 0xCC),
                selection: Color::from_rgb8(0x26, 0x4F, 0x78),
            }
        };
        let input = text_input("Find", &self.find_query)
            .id(FIND_INPUT)
            .on_input(Msg::FindQuery)
            // Enter-in-the-input navigates. `on_submit` fires only while the
            // INPUT holds focus, so an Enter typed in the editor can never
            // double-dispatch into find navigation; Shift+
            // Enter previous-match is the ↑ button (on_submit is modifier-
            // blind), and Alt+Enter select-all stays a subscription chord.
            .on_submit(Msg::FindNext)
            .padding(4)
            .size(13)
            .width(Length::Fixed(INPUT_W))
            .style(input_style);
        // Every button in the bar is this 22×22 square.
        const BTN_W: f32 = 22.0;
        // Flat icon buttons: transparent at rest, translucent gray on hover.
        // `on` LATCHES that background and adds the focus-blue border, so an
        // engaged option (`Aa`) reads as pressed at rest rather than only under
        // the pointer — a toggle whose state you cannot see is a trap.
        // Glyphs are Codicons, rendered in the bundled font.
        let icon_btn = |glyph: char, on: bool, msg| {
            button(
                text(glyph.to_string())
                    .font(scrive_iced::CODICON)
                    .size(16)
                    .width(Length::Fill)
                    .height(Length::Fill)
                    .align_x(Horizontal::Center)
                    .align_y(Vertical::Center),
            )
            .width(Length::Fixed(BTN_W))
            .height(Length::Fixed(BTN_W))
            .padding(0)
            .on_press(msg)
            .style(move |_theme, status| {
                let hover = matches!(status, button::Status::Hovered | button::Status::Pressed);
                button::Style {
                    background: (on || hover).then(|| Color::from_rgba8(90, 93, 94, 0.314).into()),
                    text_color: Color::from_rgb8(0xCC, 0xCC, 0xCC),
                    border: if on {
                        iced::border::rounded(5.0).width(1.0).color(Color::from_rgb8(0x00, 0x7F, 0xD4))
                    } else {
                        iced::border::rounded(5.0)
                    },
                    shadow: Shadow::default(),
                    snap: true,
                }
            })
        };
        // The plain (never-latched) action buttons.
        let btn = |glyph: char, msg| icon_btn(glyph, false, msg);
        // The find row's option toggles — each one part of the QUERY, not chrome.
        // The replace row mirrors this cluster with an empty slot of the same
        // width so the two rows' buttons line up; deriving that width from this
        // very list is what stops the mirror from drifting when a toggle is
        // added or removed.
        let toggles: Vec<Element<'_, Msg>> = vec![
            icon_btn(scrive_iced::icon::CASE_SENSITIVE, self.find_case, Msg::ToggleCase).into(),
            icon_btn(scrive_iced::icon::WHOLE_WORD, self.find_whole_word, Msg::ToggleWholeWord)
                .into(),
            icon_btn(scrive_iced::icon::REGEX, self.find_regex, Msg::ToggleRegex).into(),
        ];
        let toggles_w = toggles.len() as f32 * BTN_W
            + toggles.len().saturating_sub(1) as f32 * SPACING;
        let find_row = row![
            input,
            row(toggles).spacing(SPACING).align_y(Alignment::Center),
            text(label)
                .size(12)
                .color(label_color)
                .width(Length::Fixed(COUNT_W))
                .align_x(Horizontal::Center),
            btn(scrive_iced::icon::ARROW_UP, Msg::FindPrev), // previous match
            btn(scrive_iced::icon::ARROW_DOWN, Msg::FindNext), // next match
            // Find in selection — a latched toggle, sitting with the nav cluster
            // where VS Code puts it rather than with the option toggles.
            icon_btn(scrive_iced::icon::SELECTION, self.scoped(), Msg::ToggleFindInSelection),
            btn(scrive_iced::icon::CLOSE, Msg::CloseFind), // close
        ]
        .spacing(SPACING)
        .align_y(Alignment::Center);
        let rows = if self.replace_open {
            let replace_row = row![
                text_input("Replace", &self.replace_text)
                    .id(REPLACE_INPUT)
                    .on_input(Msg::ReplaceText)
                    // Enter here replaces the active match and advances — the
                    // same INPUT-focused rule as the query's `on_submit`, so an
                    // Enter typed in the editor still only makes a newline.
                    .on_submit(Msg::ReplaceOne)
                    .padding(4)
                    .size(13)
                    .width(Length::Fixed(INPUT_W))
                    .style(input_style),
                // The empty twins of the toggle cluster and the count slot —
                // together these are what land the replace buttons directly
                // under the nav buttons.
                text("").width(Length::Fixed(toggles_w)),
                text("").width(Length::Fixed(COUNT_W)),
                btn(scrive_iced::icon::REPLACE, Msg::ReplaceOne), // replace the active match
                btn(scrive_iced::icon::REPLACE_ALL, Msg::ReplaceAll), // …and every other one
            ]
            .spacing(SPACING)
            .align_y(Alignment::Center);
            column![find_row, replace_row].spacing(SPACING)
        } else {
            column![find_row]
        };
        container(
            // The chevron sits left of BOTH rows, so it reads as a handle on the
            // whole panel rather than as another find-row button.
            row![
                btn(
                    if self.replace_open {
                        scrive_iced::icon::CHEVRON_DOWN
                    } else {
                        scrive_iced::icon::CHEVRON_RIGHT
                    },
                    Msg::ToggleReplace
                ),
                rows,
            ]
            .spacing(SPACING)
            .align_y(Alignment::Center),
        )
        .padding([6, 8])
        .style(|_theme: &Theme| container::Style {
            background: Some(Color::from_rgb8(0x25, 0x25, 0x26).into()),
            border: iced::border::rounded(8.0).color(Color::from_rgb8(0x45, 0x45, 0x45)).width(1.0),
            shadow: Shadow {
                color: Color::from_rgba8(0, 0, 0, 0.36),
                offset: Vector::new(0.0, 2.0),
                blur_radius: 8.0,
            },
            ..container::Style::default()
        })
        .into()
    }

    fn subscription(&self) -> Subscription<Msg> {
        // The highlight sweep tick: one budgeted tick per frame while a dirty
        // frontier or a window gap remains, OR while the off-thread pool is
        // still verifying; the subscription drops at convergence, so an idle
        // document does zero work per frame.
        let sweeping = self.doc.highlight_frontier().is_some()
            || self.hl_pool.as_ref().is_some_and(|p| p.active);
        let sweep = if sweeping {
            iced::window::frames().map(|_| Msg::HighlightSweep)
        } else {
            Subscription::none()
        };
        // The debounced re-lint tick: fire frames only while an edit left a
        // pending re-lint; the `now_ms` gate in `maybe_relint` runs the actual
        // scan at most once per window, and clearing the flag drops this
        // subscription — idle-zero-work otherwise. A distinct map closure from
        // `sweep` (its own `TypeId`) so iced tracks the two frame recipes apart.
        let relint = if self.relint_dirty {
            iced::window::frames().map(|_| Msg::MaybeRelint)
        } else {
            Subscription::none()
        };
        // Find keys are app chrome: caught here, not widget keys.
        // `listen_with` (unlike `listen`) delivers events *regardless of capture
        // status* — the native text_input captures Escape and blurs itself, so
        // `listen` never saw it; here we still get it and can close the bar in
        // one press. It filter-maps, so non-find keys produce no message, and the
        // editor's own captured keystrokes still route through the widget. The
        // closure is a fn (non-capturing), so `find_open` is gated in `update`.
        let keys = iced::event::listen_with(|event, _status, _window| {
            use iced::keyboard::{key::Named, Event, Key};
            let iced::Event::Keyboard(Event::KeyPressed { key, modifiers, .. }) = event else {
                return None;
            };
            match key {
                Key::Character(c) if (modifiers.command() || modifiers.control()) && c.as_str() == "f" => {
                    Some(Msg::OpenFind)
                }
                // Ctrl+H — find, with the replace row already expanded.
                Key::Character(c) if (modifiers.command() || modifiers.control()) && c.as_str() == "h" => {
                    Some(Msg::OpenReplace)
                }
                Key::Named(Named::Escape) => Some(Msg::CloseFind),
                // Alt+Enter: select all matches — safe as a global chord
                // because the editor deliberately ignores Alt+Enter (no
                // newline dispatched alongside it). Plain Enter is NOT handled
                // here: in the input it navigates via `on_submit`; in the
                // editor it must only type a newline.
                Key::Named(Named::Enter) if modifiers.alt() => Some(Msg::FindSelectAll),
                _ => None,
            }
        });
        Subscription::batch([keys, sweep, relint])
    }
}

/// A severity's display name for the diagnostic hover.
fn severity_label(sev: Severity) -> &'static str {
    match sev {
        Severity::Error => "error",
        Severity::Warning => "warning",
        Severity::Info => "info",
        Severity::Hint => "hint",
    }
}

/// An original dark theme for the demo's iced chrome (find bar, popups,
/// scrollbar), matching the `Scrive Dark` syntax theme.
pub fn scrive_dark() -> Theme {
    use iced::theme::Palette;
    use iced::Color;
    Theme::custom(
        "Scrive Dark".to_string(),
        Palette {
            background: Color::from_rgb8(0x1c, 0x1e, 0x24), // #1C1E24
            text: Color::from_rgb8(0xdf, 0xe1, 0xe6),       // #DFE1E6
            primary: Color::from_rgb8(0xec, 0x6a, 0x88),    // rose accent (caret/selection)
            success: Color::from_rgb8(0xa3, 0xc7, 0x6d),    // green
            warning: Color::from_rgb8(0xe0, 0xb6, 0x58),    // amber
            danger: Color::from_rgb8(0xe0, 0x57, 0x5b),     // red
        },
    )
}

fn theme(_state: &App) -> Theme {
    scrive_dark()
}

fn main() -> iced::Result {
    let args: Vec<String> = std::env::args().collect();
    // `--large <MB>` (default 10): open a synthetic Rust-shaped document of
    // that size instead of the sample — the large-document stress case
    // (type, scroll, jump to end; colors fill in behind the idle sweep).
    if let Some(i) = args.iter().position(|a| a == "--large") {
        let mb: usize = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
        let _ = LARGE_DOC.set(gen_large(mb));
    }
    if let Some(i) = args.iter().position(|a| a == "--capture") {
        let path = args.get(i + 1).map_or("scratch.png", String::as_str);
        // Render the syntax-highlighted Rust sample in the Scrive Dark theme.
        // The static harness discards published messages, so it stands in for
        // the app loop's first ViewportChanged: tokenize the (budget-capped)
        // first batch so the visible rows carry their colors.
        let mut app = App::default();
        app.doc.tokenize_highlight(app.doc.buffer().line_count());
        let (w, h) = capture::render_to_png(app.view(), 900, 560, &scrive_dark(), &[], path);
        eprintln!("captured {w}x{h} -> {path}");
        return Ok(());
    }
    // Fold-geometry verification harness (see the module doc): collapse every
    // collapsible pair, then render one frame — maximal fold geometry on screen.
    if let Some(i) = args.iter().position(|a| a == "--capture-folds") {
        let path = args.get(i + 1).map_or("scratch-folds.png", String::as_str);
        let mut app = App::default();
        app.doc.tokenize_highlight(app.doc.buffer().line_count());
        for (open, ..) in app.doc.collapsible_pairs() {
            app.doc.toggle_fold_opener(open);
        }
        let (w, h) = capture::render_to_png(app.view(), 900, 560, &scrive_dark(), &[], path);
        eprintln!("captured {w}x{h} -> {path}");
        return Ok(());
    }

    // Find+replace layout harness (see the module doc): open the bar with the
    // replace row expanded over a live query and an active match, then render
    // one frame — chevron, both inputs, the match count, and every button on
    // screen at once. Headless tests cannot see pixels; this is how the bar's
    // layout gets judged.
    if let Some(i) = args.iter().position(|a| a == "--capture-find") {
        let path = args.get(i + 1).map_or("scratch-find.png", String::as_str);
        let mut app = App::default();
        app.doc.tokenize_highlight(app.doc.buffer().line_count());
        let _ = app.update(Msg::OpenReplace); // find + the replace row out
        let _ = app.update(Msg::FindQuery("self".into()));
        let _ = app.update(Msg::ReplaceText("this".into()));
        // Latch two of the three options, so both the engaged and the resting
        // toggle style are on screen to compare.
        let _ = app.update(Msg::ToggleCase);
        let _ = app.update(Msg::ToggleWholeWord);
        // Scope find to a block of the sample, so the scope wash + its latched
        // toggle are on screen too.
        app.apply(Action::DragSelect {
            granularity: scrive_core::Granularity::Char,
            origin: 0,
            head: 600,
        });
        let _ = app.update(Msg::ToggleFindInSelection);
        let _ = app.update(Msg::FindNext); // activate a match ⇒ a real "N of M"
        let (w, h) = capture::render_to_png(app.view(), 900, 560, &scrive_dark(), &[], path);
        eprintln!("captured {w}x{h} -> {path}");
        return Ok(());
    }

    let app = iced::application(App::default, App::update, App::view)
        .title("scrive — scratch")
        .theme(theme)
        .subscription(App::subscription);
    // Register every font the widget requires (fold chevrons + find-bar icons)
    // through the one owner, so the set can't be loaded piecemeal and leave tofu.
    scrive_iced::required_fonts()
        .iter()
        .fold(app, |app, font| app.font(*font))
        .run()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Replace-all driven through the BAR (not the core verb directly) must land
    /// as one transaction: a single undo restores the document byte-for-byte.
    /// This is the app-side half of the one-undo-step claim — it proves the
    /// message path commits once, not once per match.
    #[test]
    fn replace_all_through_the_bar_is_one_undo_step() {
        let mut app = App::default();
        // Owned: `text()` borrows the document, and every `update` below needs it
        // mutably.
        let original = app.doc.text().into_owned();
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::FindQuery("self".into()));
        assert!(app.doc.find_match_count() > 0, "the sample must contain the needle");
        let _ = app.update(Msg::ReplaceText("this".into()));
        let _ = app.update(Msg::ReplaceAll);
        assert!(!app.doc.text().contains("self"), "every match replaced");
        assert!(app.doc.text().contains("this"));
        app.apply(Action::Undo);
        assert_eq!(app.doc.text(), original.as_str(), "replace-all must undo as ONE step");
    }

    /// The replace button shows the match before it overwrites it: the first
    /// press only navigates, the second replaces.
    #[test]
    fn the_replace_button_selects_before_it_replaces() {
        let mut app = App::default();
        let original = app.doc.text().into_owned();
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::FindQuery("self".into()));
        let _ = app.update(Msg::ReplaceText("this".into()));
        let _ = app.update(Msg::ReplaceOne);
        assert_eq!(app.doc.text(), original.as_str(), "the first press only navigates");
        let _ = app.update(Msg::ReplaceOne);
        assert_ne!(app.doc.text(), original.as_str(), "the second press replaces");
    }

    /// The chevron's state is the shape of the bar, not part of the query —
    /// closing find drops the query text and leaves the shape alone, so
    /// reopening comes back the way you left it.
    #[test]
    fn the_replace_row_survives_a_close() {
        let mut app = App::default();
        assert!(!app.replace_open);
        let _ = app.update(Msg::ToggleReplace);
        assert!(app.replace_open);
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::FindQuery("self".into()));
        let _ = app.update(Msg::CloseFind);
        assert!(app.find_query.is_empty(), "close drops the query");
        assert!(app.doc.find_query().is_none(), "…and its matches");
        assert!(app.replace_open, "…but not the chevron's state");
    }

    /// The `Aa` toggle is part of the QUERY, not chrome: flipping it must
    /// re-scan and change the live match set, not merely restyle the button.
    #[test]
    fn the_case_toggle_rescans() {
        let mut app = App::default();
        let _ = app.update(Msg::OpenFind);
        // The sample has `Self` (the type) and `self` (the receiver), so the
        // case flag is observable in the count.
        let _ = app.update(Msg::FindQuery("self".into()));
        let insensitive = app.doc.find_match_count();
        let _ = app.update(Msg::ToggleCase);
        assert!(app.find_case);
        let sensitive = app.doc.find_match_count();
        assert!(
            sensitive < insensitive,
            "case-sensitive must drop the `Self` matches ({sensitive} vs {insensitive})"
        );
        assert!(app.doc.find_query().is_some_and(|q| q.case_sensitive));
        // …and back.
        let _ = app.update(Msg::ToggleCase);
        assert_eq!(app.doc.find_match_count(), insensitive);
    }

    /// The `ab|` and `.*` toggles are part of the query too: each re-scans and
    /// changes the live match set, and they compose (a whole-word regex).
    #[test]
    fn the_whole_word_and_regex_toggles_rescan() {
        let mut app = App::default();
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::FindQuery("self".into()));
        let plain = app.doc.find_match_count();
        // Whole word drops the `self` inside longer identifiers, if any, and can
        // never ADD matches.
        let _ = app.update(Msg::ToggleWholeWord);
        assert!(app.doc.find_query().is_some_and(|q| q.whole_word));
        assert!(app.doc.find_match_count() <= plain);
        let _ = app.update(Msg::ToggleWholeWord);
        assert_eq!(app.doc.find_match_count(), plain, "…and toggling back restores it");

        // Regex: `.` is a metacharacter only with the option on.
        let _ = app.update(Msg::FindQuery("s.lf".into()));
        assert_eq!(app.doc.find_match_count(), 0, "literal `s.lf` is not in the sample");
        let _ = app.update(Msg::ToggleRegex);
        assert!(app.doc.find_match_count() > 0, "as a regex it matches `self`");
    }

    /// A half-typed regex is a normal state: no matches, no panic, and the bar
    /// says *why* rather than implying the document simply lacks the text.
    #[test]
    fn a_half_typed_regex_reports_invalid_rather_than_no_results() {
        let mut app = App::default();
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::ToggleRegex);
        let _ = app.update(Msg::FindQuery("(".into()));
        assert_eq!(app.doc.find_match_count(), 0);
        assert!(app.doc.find_pattern_error().is_some(), "the bar has a reason to show");
        // Editing while the pattern is broken must not panic.
        app.apply(Action::Type('x'));
        // Finishing it resumes matching.
        let _ = app.update(Msg::FindQuery("(self)".into()));
        assert!(app.doc.find_pattern_error().is_none());
        assert!(app.doc.find_match_count() > 0);
    }

    /// Find-in-selection scopes to the live selection, restricts the match set,
    /// and toggles back off. Its latched state is the DOCUMENT's scope, so there
    /// is no app-side copy that could disagree with what is actually searched.
    #[test]
    fn find_in_selection_scopes_to_the_selection() {
        let mut app = App::default();
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::FindQuery("self".into()));
        let all = app.doc.find_match_count();
        assert!(all > 1);
        // Select a prefix of the document, then scope to it.
        app.apply(Action::PlaceCaret(0));
        let half = app.doc.buffer().len() / 2;
        app.apply(Action::DragSelect {
            granularity: scrive_core::Granularity::Char,
            origin: 0,
            head: half,
        });
        let _ = app.update(Msg::ToggleFindInSelection);
        assert!(app.scoped(), "the button latches off the document's scope");
        let scoped = app.doc.find_match_count();
        assert!(scoped < all, "the scope must drop the out-of-range matches");
        assert!(
            app.doc.find_matches_in(0..u32::MAX).all(|(m, _)| m.end <= half),
            "no match may sit outside the scope"
        );
        // …and toggling off restores the full set.
        let _ = app.update(Msg::ToggleFindInSelection);
        assert!(!app.scoped());
        assert_eq!(app.doc.find_match_count(), all);
    }

    /// Ctrl+H is Ctrl+F with the replace row already out, seeding the query the
    /// same way (it delegates to `OpenFind`).
    #[test]
    fn ctrl_h_opens_find_with_the_replace_row_out() {
        let mut app = App::default();
        let _ = app.update(Msg::OpenReplace);
        assert!(app.find_open);
        assert!(app.replace_open);
    }

    /// A replace press that only navigates commits nothing, so it must not arm
    /// the re-lint — the post-edit tail runs for edits, not for gestures.
    #[test]
    fn a_navigate_only_replace_press_does_not_arm_the_relint() {
        let mut app = App::default();
        let _ = app.update(Msg::OpenFind);
        let _ = app.update(Msg::FindQuery("self".into()));
        let _ = app.update(Msg::ReplaceText("this".into()));
        app.relint_dirty = false;
        let _ = app.update(Msg::ReplaceOne); // navigates only — no commit
        assert!(!app.relint_dirty, "a navigate-only press must not arm a re-lint");
        let _ = app.update(Msg::ReplaceOne); // this one edits
        assert!(app.relint_dirty, "…but a real replace must");
    }

    /// The demo re-lint must be debounced OFF the keystroke path: a burst of
    /// keystrokes inside one debounce window triggers ZERO whole-document scans
    /// (each key is O(1) — it only flips `relint_dirty`), and a single trailing
    /// tick past the window runs EXACTLY ONE scan, then self-cancels. Guards
    /// against the whole-document O(N log N) scan running once per keystroke.
    #[test]
    fn relint_debounces_off_the_keystroke_path() {
        let mut app = App::default();
        // The seed scan (`App::default`) already ran once — measure the delta.
        let base = RELINT_RUNS.with(std::cell::Cell::get);

        // Anchor the debounce window at a fixed fake "now" on the injected clock;
        // start clean so only edits below can arm a re-lint.
        let t0 = 100_000u64;
        app.last_relint_ms = t0;
        app.relint_dirty = false;

        // A burst of keystrokes, all inside one debounce window. Each edit only
        // flips the flag; ticks arriving mid-burst are gated → no scan runs.
        const N: usize = 25;
        for _ in 0..N {
            app.apply(Action::Type('x'));
            assert!(!app.maybe_relint(t0 + 10), "a tick inside the window must not scan");
        }
        assert_eq!(RELINT_RUNS.with(std::cell::Cell::get) - base, 0, "the burst ran ZERO whole-doc scans");
        assert!(app.relint_dirty, "a trailing scan is still pending");

        // The trailing-edge tick after the window elapses runs exactly one scan.
        assert!(app.maybe_relint(t0 + RELINT_DEBOUNCE_MS + 1), "the trailing tick scans");
        assert_eq!(RELINT_RUNS.with(std::cell::Cell::get) - base, 1, "exactly one scan for the whole burst");
        assert!(!app.relint_dirty, "flag cleared after the scan");

        // Further idle ticks are no-ops (no re-scan while clean).
        assert!(!app.maybe_relint(t0 + 10 * RELINT_DEBOUNCE_MS), "idle: no re-scan");
        assert_eq!(RELINT_RUNS.with(std::cell::Cell::get) - base, 1, "still one");
    }

    /// The debounced trailing scan makes diagnostics CURRENT: a `FIXME` typed
    /// during the window is not flagged until the scan fires — proving the scan
    /// genuinely re-runs the whole-document pass, not merely toggling a flag.
    #[test]
    fn relint_trailing_scan_makes_diagnostics_current() {
        let mut app = App::default();
        let t0 = 100_000u64;
        app.last_relint_ms = t0;
        app.relint_dirty = false;

        // Type a fresh `FIXME` at the end of the buffer (no existing diagnostic
        // there, so the check is independent of the sample's contents).
        let start = app.doc.buffer().len();
        app.apply(Action::PlaceCaret(start));
        for ch in "FIXME".chars() {
            app.apply(Action::Type(ch));
        }
        let typed = start..start + 5;

        // Inside the window: no scan yet, so the new marker is NOT flagged
        // (diagnostics only ride existing positions via stickiness).
        assert!(!app.maybe_relint(t0 + 10), "still inside the debounce window");
        let flagged_before = app
            .doc
            .diagnostics_in(typed.clone())
            .any(|(r, sev, _)| r == typed && matches!(sev, Severity::Error));
        assert!(!flagged_before, "the freshly-typed FIXME is not flagged mid-burst");

        // The trailing scan brings diagnostics current.
        assert!(app.maybe_relint(t0 + RELINT_DEBOUNCE_MS + 1), "the trailing tick scans");
        let flagged_after = app
            .doc
            .diagnostics_in(typed.clone())
            .any(|(r, sev, _)| r == typed && matches!(sev, Severity::Error));
        assert!(flagged_after, "after the debounced scan the FIXME is flagged");
    }

    /// A `Document` carrying the bundled Rust grammar/theme — the parallel
    /// highlight sweep (`HighlightPool`) needs an engine to exist.
    fn grammar_doc(source: &str) -> Document {
        let mut doc = Document::new(source).expect("fits the u32 offset space");
        doc.set_syntax(
            SyntaxDef::from_sublime_syntax(include_str!("assets/rust.sublime-syntax"))
                .expect("bundled Rust grammar parses"),
            TokenTheme::from_tm_theme(include_str!("assets/scrive-dark.tmTheme"))
                .expect("bundled theme parses"),
        );
        doc
    }

    /// `HighlightPool::poll` must PACE the verified chain: one frame
    /// absorbs at most `POLL_VERIFY_BUDGET` segments and performs at most
    /// `POLL_RERUN_BUDGET` heavy synchronous mis-guess re-runs, then leaves the
    /// pool `active` (so the `window::frames` subscription re-fires) until the
    /// whole document verifies over several frames.
    ///
    /// Guards against a single `poll` draining every contiguous-ready segment
    /// (and every ready re-run) in one call: a frame must advance `next_verify`
    /// by at most the budget and keep `active` true until the whole chain is
    /// verified, so verification spreads across frames instead of blocking one.
    #[test]
    fn poll_verifies_at_most_budget_per_frame() {
        // 12 one-row segments (> POLL_VERIFY_BUDGET, so it cannot finish in one
        // frame). Rows 2 and 4 open a Rust string with an unterminated
        // `"`, so the Fresh-GUESSED segment that follows a string-context end is
        // a genuine mis-guess — this exercises the re-run budget too.
        const N: usize = 12;
        let mut lines = [""; N];
        lines[2] = "\"";
        lines[4] = "\"";
        let source = lines.join("\n");
        let mut doc = grammar_doc(&source);
        assert_eq!(doc.buffer().line_count(), N as u32, "one row per segment");

        let mut pool = HighlightPool::new(&doc, 0..N as u32).expect("grammar → engine");

        // Neutralize the worker sweep `new` dispatched so `results` is entirely
        // under test control: retag the pool to a revision no worker used (their
        // in-flight `Done`s then mismatch and are dropped on drain, and
        // `absorb_highlight` no-ops — irrelevant to the pacing this test
        // asserts), stop the workers, and drain the channel.
        pool.rev = scrive_core::Revision(u64::MAX);
        pool.cur_rev.store(u64::MAX, std::sync::atomic::Ordering::Relaxed);
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
                    scrive_core::SegmentStart::Fresh,
                    None,
                    None,
                ))
            })
            .collect();
        pool.next_verify = 0;
        pool.prev_end = None;
        pool.active = true;

        // Setup invariants: every seeded segment is a Fresh guess, and an
        // unterminated `"` really does leave a non-fresh (string-context) end —
        // so the segment after it is a genuine mis-guess. (If these fail the
        // grammar changed and the pacing assertions below would test nothing.
        // `SegmentBoundary` is not `Debug`, so compare with `!=`, not
        // `assert_ne!`.)
        assert!(
            pool.results.iter().all(|s| s.as_ref().unwrap().started_fresh()),
            "all seeded segments are Fresh guesses",
        );
        assert!(
            pool.results[2].as_ref().unwrap().end_boundary() != &pool.fresh,
            "an unterminated `\"` leaves the string context open (non-fresh end)",
        );

        // Drive `poll` to convergence one frame at a time, asserting the
        // per-frame budgets and the reschedule gate hold on every frame.
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

        // Converged: the whole chain verified, and it took MORE than one frame
        // — the proof that verification is paced, not drained in a single call.
        assert_eq!(pool.next_verify, N, "every segment verified");
        assert!(!pool.active, "active clears only once the document is fully verified");
        assert!(frames > 1, "verification was paced across frames, not drained in one");
    }
}
