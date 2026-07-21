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
//! This example drives the batteries-included [`scrive_iced::CodeEditor`]: the
//! app owns the editor, injects stub language-intelligence providers, and wires a
//! stub compile loop (the debounced `FIXME` / `TODO` re-lint) through the editor's
//! dirty signal + `set_diagnostics`. `examples/minimal.rs` is the bare-minimum
//! integration; this one exercises the overrides and the recompile-on-edit hook.

// On Windows, a release build is a GUI app with no console window; debug builds
// keep the console so panics and the `--capture` messages stay visible.
#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

#[path = "shared/capture.rs"]
mod capture;

use std::ops::Range;
use std::time::Instant;

use iced::{Element, Subscription, Task, Theme};

use scrive_core::{
    is_completion_word_char, CompletionCx, CompletionItem, CompletionKind, Completions, Diagnostic,
    Granularity, Hover, HoverCx, HoverInfo, InsertText, Point, Severity, SignatureCx, SignatureHelp,
    SignatureInfo, SyntaxDef,
};
use scrive_iced::{Action, CodeEditor, Event};

/// The `--large <MB>` corpus, generated in `main` before the app starts.
static LARGE_DOC: std::sync::OnceLock<String> = std::sync::OnceLock::new();

/// Skip the demo lint's full-text scan above this size: the stub stands in for a
/// real compile loop (which consumes snapshots off-thread); scanning a 100 MB
/// document on the UI thread per keystroke is not what it demonstrates.
const RELINT_MAX_BYTES: u32 = 2 * 1_048_576;

/// Debounce window (ms) for the demo re-lint — a burst of keystrokes coalesces
/// into at most one whole-document scan per window, mirroring how a real
/// off-thread compiler would be debounced.
const RELINT_DEBOUNCE_MS: u64 = 250;

// Test-only tally of `App::relint` scan-body executions — the debounce tests
// read it to prove the whole-document scan runs once per window, not per key.
#[cfg(test)]
thread_local! {
    static RELINT_RUNS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
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

/// The demo application: it owns a [`CodeEditor`] and runs a stub compile loop
/// (the debounced `FIXME` / `TODO` re-lint) off the editor's dirty signal — the
/// recompile-on-edit pattern a real host would use for a language server.
pub struct App {
    editor: CodeEditor,
    /// Monotonic clock start — the injected `now_ms` for the re-lint debounce.
    start: Instant,
    /// An edit armed the debounced re-lint; a `now_ms`-gated tick runs the scan.
    relint_dirty: bool,
    last_relint_ms: u64,
}

/// The app message: it only ever *maps* the editor's opaque [`Event`] (never
/// matches on it), plus the app's own debounced re-lint tick.
#[derive(Debug, Clone)]
pub enum Message {
    /// A message from the editor.
    Editor(Event),
    /// The debounced stub-compiler tick.
    Relint,
}

impl Default for App {
    fn default() -> Self {
        Self::new()
    }
}

impl App {
    pub fn new() -> Self {
        // `--large <MB>` swaps in the synthetic corpus (the field-gate doc).
        let sample;
        let source: &str = match LARGE_DOC.get() {
            Some(s) => s,
            None => {
                sample = long_sample();
                &sample
            }
        };
        // Grammar + line comment are app-supplied language config (the core ships
        // neither); the theme defaults to the bundled Scrive Dark. `.language`
        // colours the document at load — no first scroll needed.
        let grammar = SyntaxDef::from_sublime_syntax(include_str!("assets/rust.sublime-syntax"))
            .expect("bundled Rust grammar parses");
        let editor = CodeEditor::new(source)
            .language(grammar)
            .line_comment(Some("//"))
            // Rust: `"` strings; char literals off (`'` is also a lifetime marker),
            // so brackets inside a string like `"{:?}"` are not matched/coloured.
            .bracket_lexing(vec![b'"'], None)
            .completions(StubCompletions::new())
            .hover(StubHover)
            .signature(StubSignatures);
        let mut app = Self { editor, start: Instant::now(), relint_dirty: false, last_relint_ms: 0 };
        app.relint(); // seed the stub diagnostics (the sample carries a TODO)
        app
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Editor(event) => {
                let task = self.editor.update(event).map(Message::Editor);
                // An edit arms the debounced re-lint (this app's stand-in for an
                // off-thread compile), read off the editor's dirty signal.
                if self.editor.take_dirty() {
                    self.relint_dirty = true;
                }
                task
            }
            Message::Relint => {
                self.maybe_relint(self.now_ms());
                Task::none()
            }
        }
    }

    pub fn view(&self) -> Element<'_, Message> {
        self.editor.view().map(Message::Editor)
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let editor = self.editor.subscription().map(Message::Editor);
        // The debounced re-lint tick: frames only while an edit left a pending
        // re-lint; `maybe_relint`'s clock gate runs the scan at most once/window,
        // and clearing the flag drops this subscription (idle-zero-work).
        let relint = if self.relint_dirty {
            iced::window::frames().map(|_| Message::Relint)
        } else {
            Subscription::none()
        };
        Subscription::batch([editor, relint])
    }

    /// Milliseconds since app start — the injected clock for the re-lint debounce.
    fn now_ms(&self) -> u64 {
        self.start.elapsed().as_millis() as u64
    }

    /// The debounced trailing edge of the demo re-lint: if an edit armed a re-lint
    /// and the window elapsed on the injected clock, run the whole-document stub
    /// scan once and re-arm. Returns whether it scanned (the debounce tests observe
    /// this; the app ignores it).
    fn maybe_relint(&mut self, now: u64) -> bool {
        if !self.relint_dirty || now.saturating_sub(self.last_relint_ms) < RELINT_DEBOUNCE_MS {
            return false;
        }
        self.relint();
        self.relint_dirty = false;
        self.last_relint_ms = now;
        true
    }

    /// A stub diagnostics pass — a stand-in for the app-side compile loop: flags
    /// `FIXME` (error) and `TODO` (warning) markers so squiggles, the diagnostic
    /// hover, F8 navigation, and the scrollbar marks are all exercisable without a
    /// real language server. Reads the editor's document and publishes through
    /// `CodeEditor::set_diagnostics` (revision-stamped, dropped if stale).
    fn relint(&mut self) {
        if self.editor.document().buffer().len() > RELINT_MAX_BYTES {
            return; // demo-lint threshold — see RELINT_MAX_BYTES
        }
        #[cfg(test)]
        RELINT_RUNS.with(|c| c.set(c.get() + 1));
        let mut diags = Vec::new();
        {
            let buffer = self.editor.document().buffer();
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
        let rev = self.editor.document().revision();
        let _ = self.editor.set_diagnostics(rev, diags);
    }
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
    // `--large <MB>` (default 10): open a synthetic Rust-shaped document of that
    // size instead of the sample — the large-document stress case (the parallel
    // highlight sweep colours it in behind the idle sweep).
    if let Some(i) = args.iter().position(|a| a == "--large") {
        let mb: usize = args.get(i + 1).and_then(|s| s.parse().ok()).unwrap_or(10);
        let _ = LARGE_DOC.set(gen_large(mb));
    }
    if let Some(i) = args.iter().position(|a| a == "--capture") {
        let path = args.get(i + 1).map_or("scratch.png", String::as_str);
        // `CodeEditor::new(..).language(..)` colours the document at construction
        // (the cold-load seed), so the visible rows carry their colours with no
        // viewport report needed.
        let app = App::new();
        let (w, h) = capture::render_to_png(app.view(), 900, 560, &scrive_dark(), &[], path);
        eprintln!("captured {w}x{h} -> {path}");
        return Ok(());
    }
    // Fold-geometry verification harness (see the module doc): collapse every
    // collapsible pair, then render one frame — maximal fold geometry on screen.
    if let Some(i) = args.iter().position(|a| a == "--capture-folds") {
        let path = args.get(i + 1).map_or("scratch-folds.png", String::as_str);
        let mut app = App::new();
        app.editor.fold_all();
        let (w, h) = capture::render_to_png(app.view(), 900, 560, &scrive_dark(), &[], path);
        eprintln!("captured {w}x{h} -> {path}");
        return Ok(());
    }
    // Find+replace layout harness (see the module doc): open the bar with the
    // replace row expanded over a live query and an active match, then render one
    // frame — chevron, both inputs, the match count, and every button on screen.
    if let Some(i) = args.iter().position(|a| a == "--capture-find") {
        let path = args.get(i + 1).map_or("scratch-find.png", String::as_str);
        let mut app = App::new();
        let ev = |e| Message::Editor(e);
        let _ = app.update(ev(Event::OpenReplace)); // find + the replace row out
        let _ = app.update(ev(Event::FindQuery("self".into())));
        let _ = app.update(ev(Event::ReplaceText("this".into())));
        // Latch two of the three options, so both the engaged and the resting
        // toggle style are on screen to compare.
        let _ = app.update(ev(Event::ToggleCase));
        let _ = app.update(ev(Event::ToggleWholeWord));
        let _ = app.update(ev(Event::TogglePreserveCase)); // latch the replace box's AB
        // Scope find to a block of the sample, so the scope wash + its latched
        // toggle are on screen too.
        let _ = app.update(ev(Event::Editor(Action::DragSelect {
            granularity: Granularity::Char,
            origin: 0,
            head: 600,
        })));
        let _ = app.update(ev(Event::ToggleFindInSelection));
        let _ = app.update(ev(Event::FindNext)); // activate a match ⇒ a real "N of M"
        let (w, h) = capture::render_to_png(app.view(), 900, 560, &scrive_dark(), &[], path);
        eprintln!("captured {w}x{h} -> {path}");
        return Ok(());
    }

    let app = iced::application(App::new, App::update, App::view)
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

    /// The stub compile loop is debounced OFF the keystroke path: a burst of edits
    /// arms one re-lint, and only the trailing tick past the window runs the single
    /// whole-document scan. Proves recompile-on-edit driven by the editor's dirty
    /// signal coalesces correctly.
    #[test]
    fn relint_debounces_off_the_keystroke_path() {
        let mut app = App::new();
        // The seed scan (`App::new`) already ran once — measure the delta.
        let base = RELINT_RUNS.with(std::cell::Cell::get);

        // Anchor the debounce window at a fixed fake "now"; start clean so only the
        // edits below can arm a re-lint.
        let t0 = 100_000u64;
        app.last_relint_ms = t0;
        app.relint_dirty = false;

        // A burst of keystrokes, all inside one debounce window. Each edit only
        // arms the flag (via the editor's dirty signal); ticks arriving mid-burst
        // are gated → no scan runs.
        const N: usize = 25;
        for _ in 0..N {
            let _ = app.update(Message::Editor(Event::Editor(Action::Type('x'))));
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
        let mut app = App::new();
        let t0 = 100_000u64;
        app.last_relint_ms = t0;
        app.relint_dirty = false;

        // Type a fresh `FIXME` at the end of the buffer (no existing diagnostic
        // there, so the check is independent of the sample's contents).
        let start = app.editor.document().buffer().len();
        let _ = app.update(Message::Editor(Event::Editor(Action::PlaceCaret(start))));
        for ch in "FIXME".chars() {
            let _ = app.update(Message::Editor(Event::Editor(Action::Type(ch))));
        }
        let typed = start..start + 5;

        // Inside the window: no scan yet, so the new marker is NOT flagged
        // (diagnostics only ride existing positions via stickiness).
        assert!(!app.maybe_relint(t0 + 10), "still inside the debounce window");
        let flagged_before = app
            .editor
            .document()
            .diagnostics_in(typed.clone())
            .any(|(r, sev, _)| r == typed && matches!(sev, Severity::Error));
        assert!(!flagged_before, "the freshly-typed FIXME is not flagged mid-burst");

        // The trailing scan brings diagnostics current.
        assert!(app.maybe_relint(t0 + RELINT_DEBOUNCE_MS + 1), "the trailing tick scans");
        let flagged_after = app
            .editor
            .document()
            .diagnostics_in(typed.clone())
            .any(|(r, sev, _)| r == typed && matches!(sev, Severity::Error));
        assert!(flagged_after, "after the debounced scan the FIXME is flagged");
    }
}
