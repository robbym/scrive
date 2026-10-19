//! Find — the search core.
//!
//! Search is a **literal substring** with a case-fold toggle, or — behind the
//! whole-word / regex options — a **line-scoped** pattern. (The private
//! `Matcher` type carries the full rationale for why a line, and not a byte
//! window, is the unit there.) Case folding is Unicode **simple** folding in
//! every mode (`é`/`É`, `σ`/`Σ`/`ς`; never the multi-char `ß`/`ss`), so
//! toggling an option never changes what folds. The search *bar* is app-side
//! chrome; this module owns only the match model, the scan, and what a
//! replacement means.
//!
//! The one design choice: the match set **lives only in the decoration store**
//! — the sorted `FindMatch` decorations ARE the match set, in document order
//! (one fact, one owner). [`FindState`] holds the query and the scan
//! bookkeeping, never a shadow list of handles. The set is kept current
//! **eagerly**: every committed transaction (forward edit, undo, redo — they
//! share the same view-rebase path) runs [`FindState::on_commit`], a windowed
//! repair around each edit, so consumers never see a stale match. The one
//! remaining debounced path is the capped-set refill
//! ([`FindState::maybe_rescan`]).
//!
//! Documented, test-pinned relaxation: for needles that cannot overlap
//! themselves the repaired set is byte-identical to a fresh [`scan`]; a
//! self-overlapping needle (`"aa"`) repairs to a *maximal valid*
//! non-overlapping set whose phase near the edit may differ from the greedy
//! full scan until the next query change.

use core::ops::Range;

use crate::buffer::Buffer;
use crate::coords::{Bias, Point};
use crate::decorations::{
    DecorationId, DecorationKind, DecorationStore, Stickiness, TrackedRange,
};
use crate::patch::Patch;

/// An active find query: the pattern plus its option flags.
///
/// `PartialEq` so the `Document` surface can no-op an unchanged query without
/// re-scanning — the flags are part of that identity, because each one changes
/// *what matches*, not how it is displayed.
///
/// The flags pick the matcher: plain text takes the literal byte scan;
/// `whole_word` or `regex` take the line-scoped engine.
///
/// `#[non_exhaustive]` so a future option can be added without a breaking
/// change: outside this crate, build one with [`FindQuery::new`] (a plain
/// literal query) and set the options you want on its public fields, rather
/// than a struct literal.
#[derive(Clone, Default, PartialEq, Eq, Debug)]
#[non_exhaustive]
pub struct FindQuery {
    /// The text to find — a literal, or a regular expression when
    /// [`regex`](Self::regex) is set. Empty means "no matches" (never
    /// match-all).
    pub text: String,
    /// When `false`, case variants fold together — Unicode **simple** case
    /// folding (one char to one char: `a`/`A`, `é`/`É`, `σ`/`Σ`/`ς`, `k`/`K`
    /// KELVIN SIGN; never full folding's `ß`/`ss`), the SAME in the literal,
    /// whole-word and regex modes, so toggling them never changes what folds.
    pub case_sensitive: bool,
    /// Match only whole words — the `ab|` option. Each side of a match is judged
    /// by the match's own edge character there (VS Code's rule): a side whose
    /// edge is a **word** character must meet a non-word neighbour or the edge
    /// of its line; a side whose edge is a **separator** holds whatever its
    /// neighbour is. So `foo` never matches inside `foobar`, while `(foo`
    /// matches in ` (foo) ` and `==` in `a == b`. "Word character" is the
    /// crate's one word rule — alphanumeric or `_`, the same one word motion
    /// uses. In regex mode the edges are the characters each match actually
    /// covered, not the pattern's spelling (`\(foo` behaves like `(foo`).
    pub whole_word: bool,
    /// Treat [`text`](Self::text) as a regular expression — the `.*` option.
    ///
    /// Matching is **line-scoped**: the engine is fed one line at a time, so a
    /// pattern can never match across a newline. An invalid
    /// pattern is not an error state — it is what every prefix of a pattern
    /// being typed looks like — so it simply yields no matches, and
    /// [`FindState::pattern_error`] carries the reason for the bar to show.
    pub regex: bool,
}

impl FindQuery {
    /// A plain literal query for `text` — every option off (case-insensitive,
    /// not whole-word, not regex). The common case; set the public option fields
    /// on the returned value to change it.
    #[must_use]
    pub fn new(text: impl Into<String>) -> Self {
        Self { text: text.into(), ..Self::default() }
    }
}

// Op-count canary: `Matcher::compile` calls on this thread. A query compiles
// ONCE, at `FindState::set_query`; a test asserts the replace paths read that
// cached matcher rather than compiling the pattern again per press.
// Thread-local so parallel tests don't race the count. Debug/test only;
// zero-cost in release.
#[cfg(any(test, debug_assertions))]
thread_local! {
    pub(crate) static MATCHER_COMPILES: std::cell::Cell<u64> = const { std::cell::Cell::new(0) };
}

/// The compiled form of a [`FindQuery`] — the one place that decides *how* a
/// query matches, so every scan path shares the decision.
///
/// # Why two matchers, and why the second is line-scoped
///
/// [`FindState::on_commit`]'s windowed repair — the thing that keeps find at
/// O(edit window) instead of O(document) per keystroke — is built on `k`, a
/// bound on the needle's match length: a match can start at most `k−1` bytes
/// left of a changed byte, so that is how far the influence window has to
/// reach. An **unbounded** pattern destroys that argument outright (`.*` lets
/// an edit anywhere create a match starting at column 0), and there is no `k`
/// to widen by.
///
/// The fix is to bound matches by something the repair *can* window on: a line.
/// [`Lines`](Self::Lines) feeds the engine **one line at a time**, so `\n` is
/// never in the haystack and a pattern therefore *cannot* match across one —
/// structurally, not by convention (statically deciding whether `\s` or `[^x]`
/// can match `\n` is fragile). An edit then only creates or destroys matches on
/// the lines it touches, so the influence window is those lines. Windowed again,
/// and simpler than the `k−1` dance.
///
/// Whole-word rides the same path for a different reason: judging a word
/// boundary needs the character *outside* the match, which a byte window's
/// slice does not contain — but a line always holds its own matches' context,
/// because a word can never span a newline. So `whole_word` is line-scoped too,
/// and its edge rule lives in exactly one place, [`LineMatcher`], which every
/// line-scoped scan path iterates through.
///
/// Plain literal text keeps the byte-window path: it is the common case, it is
/// what the windowed scan and the `k−1` repair were built for, and nothing
/// about it needs a line. (Case folding can make a literal match's byte length
/// differ from the needle's, but only within a bound the needle fixes — see
/// [`LiteralMatcher`] — and the `k−1` argument only ever needed a bound.)
#[derive(Debug)]
enum Matcher {
    /// Plain literal text — the byte-window scan, bounded `k`, `k−1` repair.
    Literal(LiteralMatcher),
    /// Whole-word or regex — the line-scoped engine (see the type docs).
    Lines(LineMatcher),
}

impl Matcher {
    /// Compile `query`, or return the pattern error to show the user.
    ///
    /// A syntax error is a NORMAL state, not a failure: every prefix of a
    /// pattern being typed (`(`, `[a-`) is invalid, so callers surface the
    /// message and match nothing rather than treating it as broken.
    fn compile(query: &FindQuery) -> Result<Self, String> {
        #[cfg(any(test, debug_assertions))]
        MATCHER_COMPILES.with(|c| c.set(c.get() + 1));
        if !query.whole_word && !query.regex {
            return LiteralMatcher::new(query).map(Self::Literal);
        }
        query_regex(query).map(|re| {
            Self::Lines(LineMatcher { re, whole_word: query.whole_word, expands_groups: query.regex })
        })
    }

    /// Whether this matcher is bounded to single lines — the flag every scan and
    /// repair path branches on.
    fn is_line_scoped(&self) -> bool {
        matches!(self, Self::Lines(_))
    }
}

/// THE regex compile of a query's text — the one owner of how a pattern
/// becomes an engine, shared by the line-scoped matcher and the folding
/// literal (which matches through its own escape, see [`LiteralMatcher`]).
///
/// The pattern is NEVER wrapped for whole-word (see [`LineMatcher`] for why
/// `\b…\b` is wrong): a literal is just its escaped text, a regex the user's
/// pattern verbatim. A whole-word literal is therefore the whole-word regex of
/// its own escape — `(foo` and `\(foo` compile to the same engine and the same
/// edge check, so they cannot disagree. Case folding is the query's flag, in
/// every mode, so toggling an option never changes what folds.
fn query_regex(query: &FindQuery) -> Result<regex::Regex, String> {
    let pattern = if query.regex {
        std::borrow::Cow::Borrowed(query.text.as_str())
    } else {
        std::borrow::Cow::Owned(regex::escape(&query.text))
    };
    regex::RegexBuilder::new(&pattern)
        .case_insensitive(!query.case_sensitive)
        // The line engine's haystack is one line, so `.` never sees a `\n`
        // anyway; this just makes `$`/`^` mean the line's edges, which is what a
        // line-scoped find should mean. (An escaped literal has no anchors, so
        // it changes nothing there.)
        .multi_line(true)
        .build()
        .map_err(|e| e.to_string())
}

/// The plain-literal engine: the needle, how it folds case, and the byte-length
/// bounds of its matches — the ONE owner of "where does this literal match", so
/// the display scan, the windowed buffer scan, the per-edit repair and
/// replace-all cannot fork on it.
///
/// # Why case folding goes through the regex crate
///
/// Case-insensitive literal search must fold EXACTLY as regex mode does, or
/// toggling `.*` on an escaped-equivalent query changes what matches (`é`
/// finding `É` only with the regex option on). The regex crate's `(?i)` —
/// Unicode **simple** case folding — is the owner of that rule, so a folding
/// literal is matched by the regex of its own escape, compiled `(?i)`: the two
/// modes then agree by construction, not by a second hand-kept fold table.
/// Simple folding maps one char to one char — never `ß` → `ss` (that is full
/// folding, out of scope in both modes).
///
/// # Why a match length is a *range*
///
/// Simple folding is one char ↔ one char, but not one byte length ↔ the same
/// length: `k` (1 byte) folds with `K` KELVIN SIGN (3), `s` (1) with `ſ` LONG
/// S (2), `ß` (2) with `ẞ` (3). So a folding literal's matches are exactly as
/// many CHARS as the needle, but not as many bytes. The windowed paths never
/// needed the exact length, only bounds:
/// [`max_len`](Self::max_len) for how far left of a changed byte a match can
/// start (the `k−1` overlap/repair margin), [`min_len`](Self::min_len) for
/// whether a match can still fit in what is left of a scan.
#[derive(Debug)]
struct LiteralMatcher {
    /// The needle as typed — the case-sensitive path's `memmem` needle.
    needle: String,
    /// Case-insensitive only: the needle's escape compiled `(?i)` (see the
    /// type docs). `None` ⇒ exact byte matching.
    fold: Option<regex::Regex>,
    /// No match is shorter than this many bytes. Case-sensitive: the needle's
    /// length. Folding: its CHAR count (a match is that many chars, each at
    /// least one byte).
    min_len: u32,
    /// No match is longer than this many bytes — the `k` of the windowed
    /// scan's seam overlap and the repair's `k−1` margins. Case-sensitive: the
    /// needle's length. Folding: the sum of each needle char's
    /// [`fold_width`](Self::fold_width).
    max_len: u32,
}

impl LiteralMatcher {
    /// Compile `query`'s literal. Exact matching cannot fail; a folding needle
    /// so large that its `(?i)` escape exceeds the regex crate's size limit
    /// reports that limit, exactly as the whole-word path (which compiles the
    /// same escape) always has.
    fn new(query: &FindQuery) -> Result<Self, String> {
        let needle = query.text.clone();
        let len = needle.len() as u32;
        if query.case_sensitive {
            return Ok(Self { needle, fold: None, min_len: len, max_len: len });
        }
        // The literal's escape, compiled `(?i)` by the one owner.
        let fold = query_regex(query)?;
        let min_len = needle.chars().count() as u32;
        let max_len = needle.chars().map(Self::fold_width).fold(0u32, u32::saturating_add);
        Ok(Self { needle, fold: Some(fold), min_len, max_len })
    }

    /// The widest UTF-8 encoding (bytes) of any char that simple-folds with
    /// `c` — one needle char's share of [`max_len`](Self::max_len).
    ///
    /// Exact for ASCII, the common needle and the one where a loose bound costs
    /// most: the repair window is `±(max_len−1)`, so a 4-bytes-per-char bound
    /// made a dense ASCII needle's keystroke repair tear out and re-insert
    /// several times the matches it needed to (measured: +50% per keystroke on
    /// the find bench's corpus). Only two non-ASCII chars simple-fold into
    /// ASCII — `K` KELVIN SIGN (3 bytes, with `k`) and `ſ` LONG S (2, with
    /// `s`) — a fact taken exhaustively from the regex crate itself, which owns
    /// folding, by `fold_width_covers_every_fold_partner_of_ascii`. Beyond
    /// ASCII the bound is simply 4, the widest any char can be: that needs no
    /// Unicode table to be right, and it widens a window by a constant factor
    /// of the NEEDLE, never by anything the document scales.
    fn fold_width(c: char) -> u32 {
        match c {
            'k' | 'K' => 3,
            's' | 'S' => 2,
            c if c.is_ascii() => 1,
            _ => 4,
        }
    }

    /// Every non-overlapping match in `text`, leftmost-first, as byte spans
    /// relative to it; the `bool` is whether `cap` was hit. An empty needle
    /// matches nothing (never match-all).
    ///
    /// A literal has no context: whether it matches at a start depends only on
    /// the bytes it covers, so scanning a char-boundary SLICE finds exactly the
    /// whole text's matches that fit in it — what lets the windowed paths feed
    /// it windows.
    fn scan(&self, text: &str, cap: usize) -> (Vec<Range<u32>>, bool) {
        if self.needle.is_empty() {
            return (Vec::new(), false); // empty needle = zero matches
        }
        let (mut exact, mut folded);
        let hits: &mut dyn Iterator<Item = Range<usize>> = match &self.fold {
            // memmem yields non-overlapping occurrences leftmost-first — the
            // next probe starts at the previous match's end.
            None => {
                let k = self.needle.len();
                exact = memchr::memmem::find_iter(text.as_bytes(), self.needle.as_bytes())
                    .map(move |i| i..i + k);
                &mut exact
            }
            // The regex crate's own leftmost-first, non-overlapping iteration.
            // The pattern is a run of single-char classes, so at most one match
            // can begin at any start — this IS try-every-start greedy.
            Some(re) => {
                folded = re.find_iter(text).map(|m| m.range());
                &mut folded
            }
        };
        let mut spans = Vec::new();
        for r in hits {
            if spans.len() == cap {
                return (spans, true); // cap reached
            }
            // The bounds every windowed path's correctness rests on.
            debug_assert!(
                (self.min_len..=self.max_len).contains(&(r.len() as u32)),
                "a literal match of {} bytes escaped its bounds {}..={}",
                r.len(),
                self.min_len,
                self.max_len
            );
            spans.push(r.start as u32..r.end as u32);
        }
        (spans, false)
    }
}

/// The line-scoped engine: a compiled pattern plus the whole-word edge rule —
/// the ONE owner of "which spans of a line match", so the display scan, the
/// per-edit repair, and replace-all cannot fork on it.
///
/// # Why whole-word is a per-match edge check, not `\b…\b`
///
/// `\b` asserts that the characters on its two sides differ in wordness, so
/// wrapping a needle whose edge is punctuation inverts the rule on that side:
/// `\b(foo` demands a WORD character before the `(` — it can never match in
/// ` (foo) ` — and `foo.\b` matches only when a word character follows the `.`.
/// The rule users mean (VS Code's) is about the match's *own* edge: a side
/// whose edge character is a word character needs a non-word neighbour (or the
/// line edge); a side whose edge is a separator is free. That depends on what
/// a match actually covered, which for a regex (`\(?foo`, `foo|==`) differs
/// match by match and is unknowable at compile time — so the check runs on
/// each match against its line, after the engine finds it.
///
/// The word rule is [`crate::movement::is_word_char`], the crate's single
/// owner, so whole-word find agrees with word motion and Ctrl+D about where a
/// word ends.
#[derive(Debug)]
struct LineMatcher {
    /// The pattern: the user's regex verbatim, or a literal's escape. Never
    /// wrapped — the whole-word rule is [`whole_word`](Self::whole_word)'s job.
    re: regex::Regex,
    /// Apply the edge check to each match (see the type docs).
    whole_word: bool,
    /// Whether a replacement expands `$1`-style group references: `re` is the
    /// user's regex, so its capture groups are the user's, rather than a
    /// whole-word literal's escape, whose replacement is plain text (see
    /// [`replacements`]).
    expands_groups: bool,
}

impl LineMatcher {
    /// Every match in `line` — ONE line, without its `\n` — leftmost-first and
    /// non-overlapping, as byte spans relative to `line`. Zero-width hits are
    /// dropped (nothing to navigate to, highlight, or replace).
    fn find_iter<'a>(&'a self, line: &'a str) -> LineMatches<'a> {
        LineMatches { m: self, line, at: 0 }
    }

    /// Whether the span `s..e` (non-empty, on char boundaries) of `line` passes
    /// the whole-word edge rule on both sides.
    fn is_whole_word(line: &str, s: usize, e: usize) -> bool {
        use crate::movement::is_word_char;
        // A side fails only when its edge char AND its neighbour are both word
        // characters; a line edge (`None`) is never a word character.
        let holds = |edge: Option<char>, neighbour: Option<char>| {
            !(edge.is_some_and(is_word_char) && neighbour.is_some_and(is_word_char))
        };
        holds(line[s..].chars().next(), line[..s].chars().next_back())
            && holds(line[..e].chars().next_back(), line[e..].chars().next())
    }
}

/// [`LineMatcher::find_iter`]'s iterator.
///
/// The cursor rule is what makes the set "the leftmost non-overlapping VALID
/// placements": an accepted match resumes at its end (non-overlapping), while a
/// REJECTED one — zero-width, or failing the edge check — resumes one char past
/// its *start*, never its end, so it cannot hide a valid match that starts
/// inside it (`a-a` in `xa-a-a`: `1..4` fails against `x`, `3..6` is bounded by
/// `-`). For a literal needle that is exactly try-every-start, so on needles
/// with word-character edges it keeps the old `\b…\b` results (up to the word
/// rule itself: `\b`'s Unicode `\w` also admitted marks and connector
/// punctuation; this is now the crate's own rule, as above). Without the
/// edge check (plain regex) it is byte-identical to `Regex::find_iter` with
/// empties dropped (pinned by `line_matches_without_whole_word_equal_find_iter`).
///
/// One limit, inherent to judging a match after the engine picks it: the engine
/// offers ONE match per start (leftmost-first), so a rejected match's start is
/// not retried with a different length (`foo|foobar` rejects `foo` in `foobar`
/// rather than taking `foobar`). Cost residual: each rejection re-searches from
/// one char further on, so a pathological whole-word regex whose long matches
/// are all rejected (`.*a` on a line of `xa`s) is quadratic in that LINE's
/// length — never in the document's, since the scan is line-scoped.
struct LineMatches<'a> {
    m: &'a LineMatcher,
    line: &'a str,
    /// The next search start (a char boundary of `line`).
    at: usize,
}

impl Iterator for LineMatches<'_> {
    type Item = Range<usize>;

    fn next(&mut self) -> Option<Range<usize>> {
        while self.at <= self.line.len() {
            // `find_at`, not a slice: the engine sees the text before `at`, so
            // the user's own `^` / `\b` keep their line context on a retry.
            let hit = self.m.re.find_at(self.line, self.at)?;
            let (s, e) = (hit.start(), hit.end());
            if s < e && (!self.m.whole_word || LineMatcher::is_whole_word(self.line, s, e)) {
                self.at = e;
                return Some(s..e);
            }
            // Rejected: retry one char past the START (see the type docs). At the
            // line's end there is no next char, which ends the loop.
            self.at = s + self.line[s..].chars().next().map_or(1, char::len_utf8);
        }
        None
    }
}

/// Debounce window (ms) for the capped-set refill — a user-facing default,
/// exported as a `pub fn`. This is the ONLY debounced find path: ordinary edits
/// repair the match set eagerly at the commit.
#[must_use]
pub fn default_find_debounce() -> u64 {
    100
}

/// The scan stops after this many matches and records `capped` — cheap
/// insurance against a pathological query. A capped set is a *prefix of the
/// document*: everything up to [`FindState`]'s coverage frontier is exact,
/// nothing beyond it is represented.
pub const FIND_MATCH_CAP: usize = 10_000;

/// Every non-overlapping match of `query` in `text`, leftmost-first, as byte
/// spans; the `bool` is whether the scan hit [`FIND_MATCH_CAP`].
///
/// The query's options pick the matcher. Plain text takes a **literal
/// substring** scan — the next probe starts at the previous match's end
/// (non-overlapping), and a folded match may differ in byte length from the
/// needle (`k` finds the 3-byte KELVIN SIGN). `whole_word` / `regex` take the
/// **line-scoped** engine, fed one line at a time so a match never crosses a
/// newline. Every mode folds case alike (Unicode simple folding). An empty
/// literal query yields no matches (never match-all); an unfinished regex
/// yields none until it parses.
///
/// The pure whole-text scan the tests use as the oracle. The live paths (the
/// query-change full scan, the capped refill, the per-edit window repairs) share
/// its matcher rather than calling it directly, so the match rules cannot fork.
/// Case-sensitive literal search rides `memchr::memmem`; the fold path is the
/// regex crate's `(?i)` over the needle's escape (see `LiteralMatcher`) — both
/// pinned against the naive reference scan where its byte-wise ASCII fold is
/// exact (`scan_equals_the_naive_oracle`), and the fold path against regex
/// mode everywhere (`literal_and_regex_fold_case_identically`).
#[must_use]
pub fn scan(text: &str, query: &FindQuery) -> (Vec<Range<u32>>, bool) {
    scan_capped(text, query, FIND_MATCH_CAP)
}

/// [`scan`] with the cap injected. Compiles the query's matcher, then defers to
/// [`scan_with`] — the same per-slice matcher the windowed paths use — so the
/// text you replace cannot diverge from the text you were shown.
fn scan_capped(text: &str, query: &FindQuery, cap: usize) -> (Vec<Range<u32>>, bool) {
    match Matcher::compile(query) {
        Ok(m) => scan_with(text, &m, cap),
        // An unfinished pattern (`(`, `[a-`) matches nothing. Not an error: it
        // is what every prefix of a pattern being typed looks like.
        Err(_) => (Vec::new(), false),
    }
}

/// [`scan_capped`] against an ALREADY-compiled matcher, so a windowed scan
/// compiles the pattern once rather than once per window.
fn scan_with(text: &str, m: &Matcher, cap: usize) -> (Vec<Range<u32>>, bool) {
    match m {
        Matcher::Literal(lit) => lit.scan(text, cap),
        Matcher::Lines(lm) => scan_lines(text, lm, cap),
    }
}

/// The line-scoped scan: the engine is fed one line at a time — **without its
/// newline** — so a pattern can never match across one. Offsets are relative to
/// `text`.
fn scan_lines(text: &str, lm: &LineMatcher, cap: usize) -> (Vec<Range<u32>>, bool) {
    let mut spans: Vec<Range<u32>> = Vec::new();
    let mut base = 0usize;
    for line in text.split_inclusive('\n') {
        // Strip the `\n` before matching: keeping it in the haystack is exactly
        // what would let `\s` or `[^x]` slip across a line and break the
        // line-windowed repair this design rests on.
        let body = line.strip_suffix('\n').unwrap_or(line);
        // Zero-width hits (`a*` against "bbb") never come out of `find_iter`:
        // there is nothing to navigate to, highlight, or replace, and the
        // store's `EmptyPolicy::Drop` would discard them anyway — so the count
        // stays honest about what the user can act on.
        for m in lm.find_iter(body) {
            if spans.len() == cap {
                return (spans, true); // cap reached
            }
            spans.push((base + m.start) as u32..(base + m.end) as u32);
        }
        base += line.len();
    }
    (spans, false)
}

/// [`scan`] driven through the buffer's backing-agnostic ranged reads:
/// fixed-size windows with a longest-match−1 overlap, so a match straddling a
/// window boundary is found by the next window and the result is
/// **byte-identical to `scan(&buffer.text(), query)`** (pinned by
/// `windowed_scan_equals_the_whole_text_scan`). Never materializes the
/// document — peak transient is one window — and a capped dense query stops
/// after O(bytes-until-cap), which `buffer.text()` would defeat by copying
/// the whole rope up front.
///
/// Takes the ALREADY-compiled matcher: [`FindState`] scans through the one it
/// cached at [`set_query`](FindState::set_query), so a query change compiles
/// its pattern once, not once per scan.
fn scan_buffer(
    buffer: &Buffer,
    m: &Matcher,
    window: u32,
    within: Range<u32>,
) -> (Vec<Range<u32>>, bool) {
    scan_buffer_capped(buffer, m, window, FIND_MATCH_CAP, within)
}

/// Every match of the compiled matcher `m` inside `within` — the whole document
/// when find is unscoped, or the find-in-selection scope — **uncapped**. The
/// replace-all scan.
///
/// Replace-all cannot read the live match set: that set is capped at
/// [`FIND_MATCH_CAP`] and is only a *prefix of its range*, so "all" built from
/// it would silently stop at the cap and leave the tail untouched. It runs the
/// same windowed [`scan_buffer_capped`] the display scan does — one match rule,
/// no fork — with the cap lifted. Whole-`within` in the worst case (you cannot
/// replace what you have not found), which is why it is reachable only from a
/// discrete user action, never a keystroke.
fn scan_all(buffer: &Buffer, m: &Matcher, within: Range<u32>) -> Vec<Range<u32>> {
    scan_buffer_capped(buffer, m, crate::buffer::SCAN_WINDOW, usize::MAX, within).0
}

/// [`scan_buffer`] with the cap injected — see [`scan_capped`].
///
/// `within` bounds the scan (the whole document, or the find-in-selection
/// scope). It is clamped to the document and snapped INWARD to char boundaries,
/// so a caller-supplied scope can be any offsets without breaking `slice`'s
/// str contract.
fn scan_buffer_capped(
    buffer: &Buffer,
    m: &Matcher,
    window: u32,
    cap: usize,
    within: Range<u32>,
) -> (Vec<Range<u32>>, bool) {
    // Snap INWARD: `hi` left, `lo` right, so the scanned span never grows past
    // the requested bound and never splits a char.
    let hi = buffer.clip_offset(within.end.min(buffer.len()), Bias::Left);
    let lo = buffer.clip_offset(within.start.min(hi), Bias::Right);
    let lit = match m {
        Matcher::Lines(lm) => return scan_buffer_lines(buffer, lm, cap, lo..hi),
        Matcher::Literal(lit) => lit,
    };
    // A folding match's byte length is a range, not the needle's length (see
    // `LiteralMatcher`): the seam overlap must cover the LONGEST possible
    // match, while "can a match still fit" asks about the SHORTEST.
    let (k, k_min) = (u64::from(lit.max_len), u64::from(lit.min_len));
    if k == 0 {
        return (Vec::new(), false); // empty needle = zero matches
    }
    // ≥ 2k, so every iteration provably advances even for giant needles.
    let window = u64::from(window).max(k * 2);
    let mut spans: Vec<Range<u32>> = Vec::new();
    let mut pos: u32 = lo; // the non-overlapping scan cursor (next probe start)
    while u64::from(pos) + k_min <= u64::from(hi) {
        // The window end snaps RIGHT to a char boundary: `slice` stays on the
        // str-slicing contract, the window is never empty, and growing a
        // window never loses a candidate.
        let win_end = buffer
            .clip_offset((u64::from(pos) + window).min(u64::from(hi)) as u32, Bias::Right);
        let (local, local_capped) = lit.scan(&buffer.slice(pos..win_end), cap);
        let found = !local.is_empty();
        for r in local {
            if spans.len() == cap {
                return (spans, true); // cap reached
            }
            spans.push(pos + r.start..pos + r.end);
        }
        // A capped in-window scan left the window's tail unscanned — keep
        // going from the last match even at the scan's end (the overflow
        // match it found still has to trip the cap above).
        if win_end == hi && !local_capped {
            break;
        }
        // Resume through the one owner of the seam logic ([`Buffer::scan_resume`]):
        // past the last match when the window found one (the non-overlap cursor —
        // the window's tail may be re-scanned but is never skipped), else at the
        // char boundary before `win_end − (k−1)`.
        let last_end = found.then(|| spans.last().expect("found ⇒ spans non-empty").end);
        pos = buffer.scan_resume(pos, win_end, k as u32, last_end);
    }
    (spans, false)
}

/// The line-scoped windowed scan over a buffer, in absolute offsets.
///
/// Whole lines, **no overlap**: a match cannot cross `\n`, so a line boundary
/// can never fall inside one and the literal path's `k−1` seam logic has nothing
/// to do here.
///
/// Boundaries are judged against the WHOLE line even when `within` opens or
/// closes mid-line — a scope edge must never fake a word boundary — so matches
/// are filtered to `within` afterwards rather than by truncating the haystack.
fn scan_buffer_lines(
    buffer: &Buffer,
    lm: &LineMatcher,
    cap: usize,
    within: Range<u32>,
) -> (Vec<Range<u32>>, bool) {
    let mut spans: Vec<Range<u32>> = Vec::new();
    if within.start >= within.end {
        return (spans, false);
    }
    let first = buffer.offset_to_point(within.start).row;
    let last = buffer.offset_to_point(within.end).row;
    for row in first..=last {
        let base = buffer.point_to_offset(Point::new(row, 0));
        let line = buffer.line(row); // excludes the trailing `\n`
        for m in lm.find_iter(&line) {
            let (s, e) = (base + m.start as u32, base + m.end as u32);
            if s < within.start || e > within.end {
                continue; // outside the scope, which may open/close mid-line
            }
            if spans.len() == cap {
                return (spans, true); // cap reached
            }
            spans.push(s..e);
        }
    }
    (spans, false)
}

/// The whole lines a (post-edit) byte range touches — the line-scoped repair's
/// influence window.
///
/// This is the line-scoped answer to the literal path's `±(k−1)`: a match cannot
/// cross a `\n`, so an edit can only have created or destroyed matches on the
/// lines it touched, and re-scanning exactly those lines is both necessary and
/// sufficient.
fn whole_lines(buffer: &Buffer, r: &Range<u32>) -> Range<u32> {
    let first = buffer.offset_to_point(r.start).row;
    let last = buffer.offset_to_point(r.end).row;
    let start = buffer.point_to_offset(Point::new(first, 0));
    // The last touched line's CONTENT end — deliberately NOT the next line's
    // start. `take_matching_in` removes everything *touching* the window, so a
    // window reaching the next line's first byte would carry off a match that
    // starts there, while the re-scan (bounded by this same range) would refuse
    // to re-add it as out of range — silently losing it. Ending before the `\n`
    // makes the window unable to touch anything off its own lines.
    let base = buffer.point_to_offset(Point::new(last, 0));
    start..base + buffer.line(last).len() as u32
}

/// Every match in `within`, paired with the text that should replace it — the
/// replace-all plan, and the one place that decides what a replacement *means*.
///
/// In **regex** mode `replacement` is a TEMPLATE: `$1` / `${name}` expand
/// against each match's own capture groups, as VS Code does. In every other mode
/// (literal, whole-word) it is literal text — a `$1` typed into the replace box
/// replaces with a literal `$1`, because nothing captured anything.
///
/// When `preserve_case` is set (the replace bar's `AB` toggle), each
/// replacement is re-cased to the pattern of the text it replaces, by VS Code's
/// rule — an ALL-CAPS match takes an upper-cased replacement, a Capitalized one
/// a capitalized one, a camelCase one a lower first letter, and `-`/`_`
/// segments are re-cased one by one — so a mechanical rename does not flatten
/// the casing of the identifiers it touches. See [`preserve_case_of`].
///
/// Takes the ALREADY-compiled matcher — the one [`FindState`] cached at
/// [`set_query`](FindState::set_query) (reached through
/// [`FindState::replacements`]) — so a replace press compiles nothing.
fn replacements(
    buffer: &Buffer,
    m: &Matcher,
    within: Range<u32>,
    replacement: &str,
    preserve_case: bool,
) -> Vec<(Range<u32>, String)> {
    let lm = match m {
        Matcher::Lines(lm) if lm.expands_groups => lm,
        // Literal and whole-word: one fixed string, every match — re-cased to
        // the match when `preserve_case` is on.
        _ => {
            return scan_all(buffer, m, within)
                .into_iter()
                .map(|r| {
                    let text = if preserve_case {
                        preserve_case_of(&buffer.slice(r.clone()), replacement)
                    } else {
                        replacement.to_string()
                    };
                    (r, text)
                })
                .collect();
        }
    };
    // Mirrors `scan_buffer_lines` exactly — same lines, same `LineMatcher`
    // iteration (so whole-word's edge check and cursor rule apply), same scope
    // filter — then fetches each kept match's capture groups, which expansion
    // needs and the plain scan throws away.
    let mut out = Vec::new();
    if within.start >= within.end {
        return out;
    }
    let first = buffer.offset_to_point(within.start).row;
    let last = buffer.offset_to_point(within.end).row;
    for row in first..=last {
        let base = buffer.point_to_offset(Point::new(row, 0));
        let line = buffer.line(row); // excludes the trailing `\n`
        for m in lm.find_iter(&line) {
            let (s, e) = (base + m.start as u32, base + m.end as u32);
            if s < within.start || e > within.end {
                continue; // outside the scope, which may open/close mid-line
            }
            // The SAME match, with its groups: no match starts in
            // `m.start..` before `m.start`, and leftmost-first picks one match
            // per start, so `captures_at` from the match's own start re-finds
            // exactly it (with the same line context for `^` / `\b`).
            let c = lm.re.captures_at(&line, m.start).expect("the match re-finds at its start");
            debug_assert_eq!(c.get(0).map(|g| g.range()), Some(m.clone()));
            let mut dst = String::new();
            c.expand(replacement, &mut dst);
            // Preserve-case keys off the WHOLE match (group 0), applied AFTER
            // template expansion — so `$1`-driven text is re-cased too. That is
            // scrive's extension: VS Code's rule is defined over a plain
            // replacement string, and this feeds it the expanded one.
            let text = if preserve_case { preserve_case_of(&line[m], &dst) } else { dst };
            out.push((s..e, text));
        }
    }
    out
}

/// Re-case `replacement` to the case *pattern* of `matched` — the engine behind
/// the replace bar's preserve-case (`AB`) toggle.
///
/// This is VS Code's rule (`buildReplaceStringWithCasePreserved`), ported
/// branch for branch and pinned by its own test table
/// (`preserve_case_follows_vs_codes_table`), first rule that applies wins:
///
/// 1. **Segments.** When both strings contain `-` and split into the same
///    number of `-`-segments, each segment is re-cased to its counterpart by
///    this same rule (`foo-BAR` + `new-name` ⇒ `new-NAME`); likewise `_`. If
///    BOTH separators qualify, neither does — the rest of the rules see the
///    whole strings.
/// 2. **All upper** — `matched` equals its own upper-casing ⇒ the replacement
///    upper-cased. (So a match with no cased letters at all, like `42`, counts
///    as upper: VS Code's quirk, kept for parity.)
/// 3. **All lower** ⇒ the replacement lower-cased.
/// 4. **First char upper** (lower-casing changes it) — `FooBar`, `Abc` ⇒ the
///    replacement's first char upper-cased, the rest as typed.
/// 5. **First char lower** (upper-casing changes it) — camelCase `fooBar` ⇒
///    the replacement's first char lower-cased, the rest as typed.
/// 6. Otherwise (a caseless first char in a mixed match) ⇒ as typed.
///
/// "First char" is the first Unicode scalar value where VS Code reads the
/// first UTF-16 unit — the same thing except for a match opening with an
/// astral cased letter (e.g. Deseret), which VS Code cannot case and this can.
/// Case mappings are Rust's full Unicode ones (`ß` upper-cases to `SS`), as
/// JavaScript's are.
fn preserve_case_of(matched: &str, replacement: &str) -> String {
    let Some(first) = matched.chars().next() else {
        return replacement.to_string(); // nothing to take a pattern from
    };
    // Rule 1: both sides split into the same number of `sep`-segments.
    let segmented = |sep: char| {
        matched.contains(sep)
            && replacement.contains(sep)
            && matched.split(sep).count() == replacement.split(sep).count()
    };
    let sep = match (segmented('-'), segmented('_')) {
        (true, false) => Some('-'),
        (false, true) => Some('_'),
        _ => None, // neither, or both: judge the whole strings
    };
    if let Some(sep) = sep {
        let mut out = String::with_capacity(replacement.len());
        for (i, (m, r)) in matched.split(sep).zip(replacement.split(sep)).enumerate() {
            if i > 0 {
                out.push(sep);
            }
            out.push_str(&preserve_case_of(m, r));
        }
        return out;
    }
    if matched.to_uppercase() == matched {
        return replacement.to_uppercase(); // rule 2
    }
    if matched.to_lowercase() == matched {
        return replacement.to_lowercase(); // rule 3
    }
    // Rules 4–6 re-case only the replacement's first char.
    let changes = |mapped: &mut dyn Iterator<Item = char>| !mapped.eq([first]);
    let mut rest = replacement.chars();
    let Some(r0) = rest.next() else {
        return String::new(); // an empty replacement has nothing to re-case
    };
    let head: String = if changes(&mut first.to_lowercase()) {
        r0.to_uppercase().collect() // rule 4: first char is upper
    } else if changes(&mut first.to_uppercase()) {
        r0.to_lowercase().collect() // rule 5: first char is lower
    } else {
        return replacement.to_string(); // rule 6
    };
    head + rest.as_str()
}

/// Whether a tracked range is a find match — the one predicate the store
/// queries and batch removals share.
fn is_find(r: &TrackedRange) -> bool {
    matches!(r.kind, DecorationKind::FindMatch)
}

/// The search view-state: the active query, the active match's decoration id,
/// and the coverage/cap bookkeeping.
///
/// The one design rule: **the store IS the match set** — the sorted `FindMatch`
/// decorations, in `(start, id)` = document order. This state deliberately
/// holds no handle list; count, order, and N-of-M all derive from store queries
/// (one fact, one owner). The set is repaired eagerly per commit via
/// [`FindState::on_commit`], so it is always current — undo/redo inherit the
/// repair through the shared view-rebase mover.
#[derive(Debug, Default)]
pub struct FindState {
    query: Option<FindQuery>,
    /// The decoration id of the highlighted/navigated match, if any.
    active: Option<DecorationId>,
    /// The active match's tracked START offset — the position half of the active
    /// handle, kept in lockstep with [`active`](Self::active) (both `Some` or
    /// both `None`). It rides each commit's patch in
    /// [`on_commit`](Self::on_commit) with **`Bias::Right`** — the start bias of
    /// a `FindMatch`'s `NeverGrows` stickiness — so it stays equal to the active
    /// decoration's current start (pinned by `active_start_tracks…`). It earns
    /// its place as the O(log) index that lets the per-commit presence check and
    /// the N-of-M ordinal avoid an O(M) whole-store walk, not as a display
    /// convenience.
    active_start: Option<u32>,
    /// Whether the match set is a capped *prefix* of the document.
    capped: bool,
    /// Byte offset the scan has covered: `buffer.len()` when `!capped`, else
    /// the last kept match's end. Rides each commit's patch (`Bias::Left`);
    /// while capped, repairs clamp their windows to it and edits beyond it are
    /// the refill's job.
    coverage_end: u32,
    /// While capped: the live count has dropped below the cap, so coverage
    /// could grow — the ONLY condition under which [`FindState::maybe_rescan`]
    /// re-scans. (At exactly the cap the set provably equals the ideal capped
    /// prefix: repairs keep the covered prefix exact, and its first
    /// [`FIND_MATCH_CAP`] matches are the document's first ones.)
    capped_stale: bool,
    /// `now_ms` of the last full scan — the capped-refill debounce anchor.
    last_scan_ms: u64,
    /// **Find in selection**: when `Some`, matches exist ONLY inside this byte
    /// range — the scan is restricted to it and every repair window clamps to
    /// it, exactly as they already clamp to [`coverage_end`](Self::coverage_end).
    ///
    /// A position like any other, so it rides each commit's patch in
    /// [`on_commit`](Self::on_commit) rather than being re-derived: biases
    /// `(Left, Right)` — the `AlwaysGrows` pair — so text typed at either edge
    /// lands *inside* the scope. Collapsing to empty drops it (the
    /// [`EmptyPolicy::Drop`] rule a `FindMatch` follows), because a zero-width
    /// scope can only mean "no matches ever" and silently searching nothing is
    /// worse than searching everything.
    ///
    /// [`EmptyPolicy::Drop`]: crate::decorations::EmptyPolicy::Drop
    scope: Option<Range<u32>>,
    /// `query` compiled — cached so a keystroke's repair never recompiles the
    /// pattern. Kept in lockstep with [`query`](Self::query) (both are set only
    /// in [`set_query`](Self::set_query)); `None` iff there is no query or the
    /// pattern does not parse.
    matcher: Option<Matcher>,
    /// Why the pattern does not parse, when `regex` is on and it does not.
    ///
    /// Not a failure mode: every prefix of a pattern being typed (`(`, `[a-`) is
    /// invalid, so this is state to SHOW — matching yields nothing meanwhile and
    /// resumes the moment the pattern parses again.
    pattern_error: Option<String>,
}

impl FindState {
    /// An empty state — no query, no matches.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// The active query, or `None` if find is idle.
    #[must_use]
    pub fn query(&self) -> Option<&FindQuery> {
        self.query.as_ref()
    }

    /// How many live matches the current query has — the store IS the match set,
    /// so this is the O(1) root-summary [`find_count`] read.
    ///
    /// [`find_count`]: DecorationStore::find_count
    #[must_use]
    pub fn match_count(&self, store: &DecorationStore) -> usize {
        store.find_count()
    }

    /// The decoration handle of the active match, if any — the render layer
    /// compares each `FindMatch` decoration to this to pick the distinct style.
    #[must_use]
    pub fn active_id(&self) -> Option<DecorationId> {
        self.active
    }

    /// The active match's tracked start offset — the position half of the active
    /// handle, in lockstep with [`active_id`](Self::active_id). `Document` reads
    /// it for the O(log) N-of-M ordinal (`active_find_match`).
    #[must_use]
    pub(crate) fn active_start(&self) -> Option<u32> {
        self.active_start
    }

    /// The active match's live range — `None` when no match is active, or when
    /// the tracked start no longer carries it.
    ///
    /// O(log M): the tracked start's rank, then that rank's span. Deliberately
    /// NOT [`DecorationStore::decoration_range`], which is documented as an
    /// O(store) cold path — keeping [`active_start`](Self::active_start) in
    /// lockstep exists precisely so the active handle resolves without a
    /// whole-store walk.
    #[must_use]
    pub(crate) fn active_range(&self, store: &DecorationStore) -> Option<Range<u32>> {
        let (id, start) = (self.active?, self.active_start?);
        // Finds are disjoint, so only the find starting exactly at `start` can be
        // the active one — the same identity `navigate` uses to test "on a match".
        let (found, range) = store.nth_find(store.find_rank_before(start))?;
        (found == id).then_some(range)
    }

    /// Whether the match set is a capped prefix of the document
    /// ([`FIND_MATCH_CAP`]).
    #[must_use]
    pub fn capped(&self) -> bool {
        self.capped
    }

    /// The find-in-selection scope — matches exist only inside it. `None` means
    /// the whole document.
    #[must_use]
    pub fn scope(&self) -> Option<Range<u32>> {
        self.scope.clone()
    }

    /// Set (or clear with `None`) the find-in-selection scope, re-scanning
    /// synchronously — the scope is part of what matches, so changing it is a
    /// query change, not a display filter.
    ///
    /// An empty range clears the scope rather than pinning zero matches
    /// forever: searching everything is a recoverable surprise, silently
    /// searching nothing is not. A scope equal to the current one is a no-op.
    pub fn set_scope(
        &mut self,
        scope: Option<Range<u32>>,
        buffer: &Buffer,
        store: &mut DecorationStore,
        now_ms: u64,
    ) {
        let scope = scope.filter(|s| s.start < s.end);
        if scope == self.scope {
            return;
        }
        self.scope = scope;
        self.rescan(buffer, store, now_ms);
    }

    /// Set (or clear with `None`) the query and scan synchronously; never
    /// scrolls, and clears the active match. A query equal to the current one is
    /// a no-op (no needless re-scan).
    ///
    /// The full scan runs `scan_buffer` — whole-document in coverage but
    /// windowed in memory, and a capped dense query stops at
    /// O(bytes-until-cap) instead of copying the rope's tail.
    pub fn set_query(
        &mut self,
        query: Option<FindQuery>,
        buffer: &Buffer,
        store: &mut DecorationStore,
        now_ms: u64,
    ) {
        if query == self.query {
            return;
        }
        self.query = query;
        // Compile ONCE here, in lockstep with the query — the per-keystroke
        // repair reads this cache rather than re-parsing the pattern.
        (self.matcher, self.pattern_error) = match &self.query {
            None => (None, None),
            Some(q) => match Matcher::compile(q) {
                Ok(m) => (Some(m), None),
                // An unfinished pattern is a normal state: keep the reason to
                // show, match nothing until it parses.
                Err(e) => (None, Some(e)),
            },
        };
        self.active = None; // a query change drops the active match
        self.active_start = None; // …and its tracked position (lockstep)
        self.rescan(buffer, store, now_ms);
    }

    /// Why the current pattern does not parse, if it does not — for the find bar
    /// to show. `None` whenever the query is matching normally (including when
    /// it simply has no hits).
    #[must_use]
    pub fn pattern_error(&self) -> Option<&str> {
        self.pattern_error.as_deref()
    }

    /// The replace plan for the current query inside `within`: every match with
    /// the text that should replace it (see [`replacements`] for what a
    /// replacement means per mode). Through the CACHED matcher, as every scan
    /// and repair is; empty when find is idle or the pattern does not parse.
    pub(crate) fn replacements(
        &self,
        buffer: &Buffer,
        within: Range<u32>,
        replacement: &str,
        preserve_case: bool,
    ) -> Vec<(Range<u32>, String)> {
        self.matcher.as_ref().map_or_else(Vec::new, |m| {
            replacements(buffer, m, within, replacement, preserve_case)
        })
    }

    /// Refill a **capped** match set: re-scan iff a query is active, the set is
    /// capped with room below the cap, and the debounce window has elapsed.
    /// `now_ms` is an injected monotonic clock — the widget passes real time,
    /// the headless suite a fake. Returns whether it scanned.
    ///
    /// Because the eager per-commit repair keeps the set current, the only job
    /// here is growing a capped set's coverage after matches inside it died.
    pub fn maybe_rescan(&mut self, buffer: &Buffer, store: &mut DecorationStore, now_ms: u64) -> bool {
        if self.query.is_none()
            || !(self.capped && self.capped_stale)
            || now_ms.saturating_sub(self.last_scan_ms) < default_find_debounce()
        {
            return false;
        }
        self.rescan(buffer, store, now_ms);
        true
    }

    /// Repair the match set around a committed patch — the mover hook called
    /// from the view-rebase path AFTER `DecorationStore::apply_patch` (it needs
    /// post-patch positions). No-op when find is idle.
    ///
    /// The influence window depends on the matcher. A **literal** query (post-edit
    /// coordinates, `k` = the longest a match can be — the needle's byte length
    /// when exact, a per-char bound when folding): every affected placement must
    /// overlap `new.start−(k−1) .. new.end+(k−1)` (a created/destroyed match
    /// contains a changed byte or the join point). A **line-scoped** query
    /// (whole-word / regex) instead repairs the whole lines the edit touched —
    /// a match cannot cross a newline, so nothing off those lines can have
    /// changed — with no `k−1` widening and no anchor/guard (the greedy phase the
    /// widening repairs does not exist there). The rest of this describes the
    /// literal path. Matches touching the window are removed
    /// wholesale ([`DecorationStore::take_matching_in`]); the re-scan zone is
    /// then the window widened (a) to the removed extents — a merely-touching
    /// match must be re-findable, not lost — and (b) a further `k−1` bytes each
    /// way, because a placement overlapping the removal zone (one whose old
    /// greedy blocker just died — the self-overlap phase repair) can start up
    /// to `k−1` bytes left of it and end as far right. Two clamps keep the
    /// zone from double-owning text: the scan starts at the last surviving
    /// match ending in the left margin (the anchor), and candidates
    /// overlapping the first surviving match right of the zone are dropped
    /// (the guard). The zone is greedily re-scanned with the same pure
    /// [`scan`] and batch-reinserted. Matches outside every zone keep their
    /// decoration ids (id stability is what lets the active match survive
    /// unrelated edits).
    ///
    /// Active survival: untouched active stays (same id); an active removed by a
    /// window transfers to a re-created match at its exact post-patch start;
    /// otherwise it clears.
    ///
    /// Cap: `coverage_end` rides the patch first; windows clamp to it, so
    /// repairs beyond a capped set's coverage are skipped (the refill's job).
    /// If a repair pushes the live count past [`FIND_MATCH_CAP`], the tail is
    /// trimmed so the set stays a prefix of the document.
    pub fn on_commit(&mut self, patch: &Patch, buffer: &Buffer, store: &mut DecorationStore) {
        if self.query.is_none() {
            return;
        }
        // No matcher ⇒ the pattern does not parse ⇒ there are no matches to
        // repair (the set is already empty, and stays so until it parses).
        let Some(matcher) = &self.matcher else { return };
        let line_scoped = matcher.is_line_scoped();
        // A literal's `k` is the LONGEST a match can be — the needle's length
        // when exact, a bound when folding (a folded match need not be as many
        // bytes as the needle; see `LiteralMatcher`). Every margin below only
        // needs "no match is longer than k", so the bound is all it takes.
        let k = match matcher {
            Matcher::Literal(lit) => lit.max_len,
            Matcher::Lines(_) => 0, // line-scoped repair never reads `k`
        };
        if patch.is_empty() || (!line_scoped && k == 0) {
            return; // an empty literal needle has no matches to repair
        }
        // Coverage rides the patch ONCE, before the per-edit loop; an uncapped
        // set always covers the whole (post-edit) document.
        self.coverage_end = if self.capped {
            patch.map_offset(self.coverage_end, Bias::Left)
        } else {
            buffer.len()
        };
        // The active handle's position rides the SAME committed patch as the
        // decoration it tracks — `Bias::Right`, a FindMatch's `NeverGrows` start
        // bias — so it stays equal to that decoration's post-patch start.
        self.active_start = self.active_start.map(|s| patch.map_offset(s, Bias::Right));
        // The scope is a derived position like any other, so it rides the SAME
        // committed patch rather than being re-derived — `Bias::Left`/`Right`
        // (the `AlwaysGrows` pair) so text typed at either edge lands inside it.
        // Collapsed ⇒ dropped: a zero-width scope can only match nothing, and
        // silently searching nothing is worse than searching everything.
        self.scope = self.scope.take().and_then(|s| {
            let lo = patch.map_offset(s.start, Bias::Left);
            let hi = patch.map_offset(s.end, Bias::Right);
            (lo < hi).then_some(lo..hi)
        });
        // Every repair window lives inside the covered prefix AND, when find is
        // scoped, inside the scope — matches cannot exist outside either, so an
        // edit out there repairs nothing and one straddling an edge repairs only
        // the inside. This is the same mechanism as the capped-coverage clamp,
        // with one more bound; both are applied AFTER the char-boundary snap, so
        // the snap can never carry a window back outside its bound.
        let (lo_bound, hi_bound) = match &self.scope {
            Some(s) => (s.start, s.end.min(self.coverage_end)),
            None => (0, self.coverage_end),
        };
        let mut removed_active_start: Option<u32> = None;
        for e in patch.edits() {
            // The influence window, clamped to the covered prefix (and the
            // scope). Literal: ±(k−1) bytes, snapped OUTWARD to char boundaries
            // — a match no longer than k can start at most k−1 bytes left of a
            // changed byte. Line-scoped: the WHOLE lines the edit touches, which
            // is the same argument in the only currency a variable-length
            // pattern has — a match cannot cross `\n`, so nothing off those
            // lines can have been created or destroyed.
            let (w_start, w_end) = if line_scoped {
                let lines = whole_lines(buffer, &e.new);
                (lines.start, lines.end)
            } else {
                let s = buffer.clip_offset(e.new.start.saturating_sub(k - 1), Bias::Left);
                let t_raw = e.new.end.saturating_add(k - 1).min(hi_bound);
                (s, buffer.clip_offset(t_raw, Bias::Right))
            };
            let (w_start, w_end) = (w_start.max(lo_bound), w_end.min(hi_bound));
            if w_start > w_end {
                continue; // entirely outside the coverage/scope: not our repair
            }
            let removed = store.take_matching_in(w_start..w_end, is_find);
            if let Some(active) = self.active {
                if let Some(r) = removed.iter().find(|r| r.id == active) {
                    removed_active_start = Some(r.range.start);
                    self.active = None;
                    self.active_start = None; // lockstep; the transfer below re-sets both
                }
            }
            // Line-scoped: lines are independent, so there is no greedy phase to
            // repair and no match can straddle the window's edge (it would have
            // to cross a `\n`). The re-scan zone IS the window, and the anchor
            // and guard below have nothing to do — the whole `k−1` widening
            // dance exists only to serve a bounded-length needle's greedy cursor.
            let (scan_lo, scan_hi, guard) = if line_scoped {
                (w_start, w_end, None)
            } else {
            // Widen to the removed extents: a match that merely touched the
            // window (start left of it / end right of it) must be re-findable,
            // or removal would lose it.
            let ext_lo =
                removed.iter().map(|r| r.range.start).min().map_or(w_start, |s| s.min(w_start));
            let ext_hi =
                removed.iter().map(|r| r.range.end).max().map_or(w_end, |e| e.max(w_end));
            // Widen a further k−1 each way: a placement overlapping the
            // removal zone — one whose old greedy blocker died (self-overlap
            // phase) — can start up to k−1 bytes left of `ext_lo` and end as
            // far right of `ext_hi`. Removed-match boundaries are char
            // boundaries (whole-char hits of a valid-UTF-8 needle); the ±(k−1)
            // arithmetic can unsnap, so clip again.
            let scan_lo =
                buffer.clip_offset(ext_lo.saturating_sub(k - 1), Bias::Left).max(lo_bound);
            let scan_hi = buffer
                .clip_offset(ext_hi.saturating_add(k - 1).min(hi_bound), Bias::Right)
                .min(hi_bound);
            // Left anchor: the widened margin may hold a surviving match —
            // start scanning at its end so its text is never double-owned.
            // (Survivors intersecting `scan_lo..ext_lo` provably end at or
            // before `ext_lo`: anything reaching further would have touched
            // the window or a removed extent and been removed itself.)
            let anchor = store
                .decorations_in(scan_lo..ext_lo)
                .filter(is_find)
                .map(|r| r.range.end)
                .max();
            let scan_lo = anchor.map_or(scan_lo, |a| a.max(scan_lo));
            // Right guard: candidates may not overlap the first surviving
            // match right of the zone (survivors there provably start at or
            // after `ext_hi`, same argument as the anchor).
            let guard = store
                .decorations_in(ext_hi..scan_hi)
                .filter(|r| is_find(r) && r.range.start >= ext_hi)
                .map(|r| r.range.start)
                .min();
                (scan_lo, scan_hi, guard)
            };
            // Re-scan the zone through the SAME cached matcher the display scan
            // runs, so a repaired match cannot differ from a freshly found one.
            let mut spans: Vec<Range<u32>> = match matcher {
                Matcher::Lines(lm) => {
                    scan_buffer_lines(buffer, lm, FIND_MATCH_CAP, scan_lo..scan_hi).0
                }
                Matcher::Literal(lit) => lit
                    .scan(&buffer.slice(scan_lo..scan_hi), FIND_MATCH_CAP)
                    .0
                    .into_iter()
                    .map(|r| scan_lo + r.start..scan_lo + r.end)
                    .collect(),
            };
            if let Some(guard) = guard {
                spans.retain(|s| s.end <= guard);
            }
            // Windowed insert: the batch merges into its ≤band without the
            // O(M log M) whole-store `to_vec()`+re-sort of `add_sorted_batch`.
            // Byte-identical to that here (the window is already cleared of
            // finds), pinned by `splice_batch_equals_naive`.
            store.splice_sorted_batch(&spans, DecorationKind::FindMatch, Stickiness::NeverGrows);
        }
        // An active match a window removed transfers to a re-created match at
        // its exact (post-patch) start, if any — the position half moves with
        // it (lockstep).
        if let Some(start) = removed_active_start {
            let found = store
                .decorations_in(start..start)
                .find(|r| is_find(r) && r.range.start == start);
            self.active = found.as_ref().map(|r| r.id);
            self.active_start = found.map(|r| r.range.start);
        }
        // Cap bookkeeping — only when it can matter. The O(1) `find_count`
        // gates it (not `store.len()`, which counts diagnostics/snippets too, so
        // gating on it would scale with unrelated decorations). The BODY is
        // O(log) too: the live count is `find_count()` (O(1) root read) and the
        // two trim boundaries are ordinal `nth_find` lookups (O(log M)) — so a
        // capped commit costs the same regardless of decoration count (pinned by
        // `capped_commit_is_decoration_count_independent`).
        let fc = store.find_count();
        if fc > FIND_MATCH_CAP {
            // Trim from the TAIL: the set stays a prefix of the document. The two
            // boundaries are the (cap-1)-th and cap-th finds in `(start,id)` order.
            if let (Some((_, kept)), Some((_, first_trimmed))) =
                (store.nth_find(FIND_MATCH_CAP - 1), store.nth_find(FIND_MATCH_CAP))
            {
                let (kept_end, trim_from) = (kept.end, first_trimmed.start);
                store.take_matching_in(trim_from..u32::MAX, |r| {
                    is_find(r) && r.range.start >= trim_from
                });
                self.capped = true;
                self.coverage_end = kept_end;
                self.capped_stale = false; // full to the brim — nothing to refill
            }
        } else if self.capped {
            // Below the cap with frozen coverage: the refill can grow it.
            self.capped_stale = fc < FIND_MATCH_CAP;
        }
        // An active match destroyed without passing through a window (e.g.
        // collapsed and dropped by `apply_patch`, or tail-trimmed) clears —
        // probed at the tracked start (O(log)). `active_start` tracks the
        // decoration's start exactly (Bias::Right), so a present match overlaps
        // the zero-width `s..s` query.
        if let (Some(id), Some(s)) = (self.active, self.active_start) {
            let present = store.decorations_in(s..s).any(|r| is_find(&r) && r.id == id);
            if !present {
                self.active = None;
                self.active_start = None;
            }
        }
    }

    /// Select the next match from the caret and return its live range. The set
    /// is always current, so this is a pure store walk. `head`/`selection` are
    /// the newest selection's head and extent.
    pub fn find_next(
        &mut self,
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
    ) -> Option<Range<u32>> {
        self.navigate(head, selection, store, true)
    }

    /// Select the previous match from the caret — the
    /// [`find_next`](Self::find_next) mirror.
    pub fn find_prev(
        &mut self,
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
    ) -> Option<Range<u32>> {
        self.navigate(head, selection, store, false)
    }

    /// Wholesale-replace the `FindMatch` decorations from a fresh scan — the
    /// query-change path and the capped refill (the two legitimately
    /// whole-document find ops). The active match survives iff a new match
    /// starts exactly at the old active's tracked start.
    fn rescan(&mut self, buffer: &Buffer, store: &mut DecorationStore, now_ms: u64) {
        let old_active_start =
            self.active.and_then(|id| store.decoration_range(id)).map(|r| r.start);
        store.take_matching_in(0..u32::MAX, is_find);
        self.active = None;
        self.active_start = None;
        if self.query.is_some() {
            // Scoped find scans only its range; unscoped, the whole document.
            let within = self.scope.clone().unwrap_or(0..buffer.len());
            // Through the CACHED matcher — the same one every repair uses. None
            // ⇒ the pattern does not parse ⇒ it matches nothing (until it does).
            let (spans, capped) = match &self.matcher {
                Some(m) => scan_buffer(buffer, m, crate::buffer::SCAN_WINDOW, within),
                None => (Vec::new(), false),
            };
            self.capped = capped;
            self.coverage_end =
                if capped { spans.last().map_or(0, |s| s.end) } else { buffer.len() };
            let ids = store.add_sorted_batch(
                spans.iter().cloned(),
                DecorationKind::FindMatch,
                Stickiness::NeverGrows,
            );
            if let Some(start) = old_active_start {
                if let Some(i) = spans.iter().position(|s| s.start == start) {
                    self.active = Some(ids[i]); // the active match survives
                    self.active_start = Some(spans[i].start); // == start (lockstep)
                }
            }
        } else {
            self.capped = false;
            self.coverage_end = buffer.len();
        }
        self.capped_stale = false;
        self.last_scan_ms = now_ms;
    }

    /// Pick the next/previous match relative to the caret and set the active
    /// id: [`pick`](Self::pick) then [`activate`](Self::activate).
    fn navigate(
        &mut self,
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
        forward: bool,
    ) -> Option<Range<u32>> {
        self.activate(Self::pick(head, selection, store, forward))
    }

    /// Make `target` (a [`pick`](Self::pick)) the active match and return its
    /// range — or, with no match to go to, clear the active one. Navigation's
    /// state half, split from the pure pick so a verb that has to choose the
    /// match inside its commit (`Document::replace_next`, whose redo must
    /// restore the selection it advanced to) picks there and activates after.
    pub(crate) fn activate(&mut self, target: Option<(DecorationId, Range<u32>)>) -> Option<Range<u32>> {
        let Some((id, range)) = target else {
            self.active = None;
            self.active_start = None;
            return None;
        };
        self.active = Some(id);
        self.active_start = Some(range.start);
        Some(range)
    }

    /// The match navigation goes to from the caret, as `(id, range)` — a pure
    /// read of the store. The live set is the store's `FindMatch` decorations in
    /// `(start, id)` = document order; collapsed (empty) ones are skipped
    /// (normally vacuous, since the mover drops them). Sitting exactly on a match
    /// steps by position (repeated-press cycling); otherwise it scans from `head`
    /// and wraps. `None` only when there are no matches.
    pub(crate) fn pick(
        head: u32,
        selection: Range<u32>,
        store: &DecorationStore,
        forward: bool,
    ) -> Option<(DecorationId, Range<u32>)> {
        // The live set is the store's `FindMatch` decorations in `(start, id)`
        // order — but we never materialize it. Count, rank, and the r-th match
        // are three O(log) store queries that reproduce the full-list branch
        // table exactly (pinned by `navigate_equals…`).
        let count = store.find_count();
        if count == 0 {
            return None;
        }
        // "On a match" ⇔ the find starting at `selection.start` IS `selection`.
        // `find_rank_before(selection.start)` is that find's rank in one descent;
        // only a find whose start equals `selection.start` can equal `selection`
        // (finds are disjoint), so this reproduces `position(|r| *r == selection)`.
        let on = {
            let r = store.find_rank_before(selection.start);
            store.nth_find(r).filter(|(_, rng)| *rng == selection).map(|_| r)
        };
        let pick = match (on, forward) {
            (Some(r), true) => (r + 1) % count,
            (Some(r), false) => (r + count - 1) % count,
            // First find with start ≥ head, wrapping past the last to 0.
            (None, true) => store.find_rank_before(head) % count,
            // Last find with start < head, wrapping before the first to count−1.
            (None, false) => (store.find_rank_before(head) + count - 1) % count,
        };
        store.nth_find(pick)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q(text: &str, case_sensitive: bool) -> FindQuery {
        FindQuery { text: text.into(), case_sensitive, ..Default::default() }
    }

    /// A whole-word query — the `ab|` option.
    fn qw(text: &str, case_sensitive: bool) -> FindQuery {
        FindQuery { text: text.into(), case_sensitive, whole_word: true, ..Default::default() }
    }

    /// A regex query — the `.*` option.
    fn qr(text: &str, case_sensitive: bool) -> FindQuery {
        FindQuery { text: text.into(), case_sensitive, regex: true, ..Default::default() }
    }

    #[test]
    fn matches_are_leftmost_and_non_overlapping() {
        // "aaaa" / "aa" pairs into two, not three overlapping.
        assert_eq!(scan("aaaa", &q("aa", true)), (vec![0..2, 2..4], false));
        // "ababa" / "aba": second probe starts at 3, "ba" is too short → one hit.
        let (spans, capped) = scan("ababa", &q("aba", true));
        assert_eq!((spans.len(), spans.first().cloned(), capped), (1, Some(0..3), false));
    }

    #[test]
    fn case_fold_is_toggled() {
        assert_eq!(scan("AbAb", &q("ab", false)), (vec![0..2, 2..4], false));
        assert_eq!(scan("AbAb", &q("ab", true)).0, Vec::<Range<u32>>::new());
        assert_eq!(scan("AbAb", &q("Ab", true)), (vec![0..2, 2..4], false));
    }

    /// Literal and regex modes fold case IDENTICALLY — Unicode simple case
    /// folding in both — so toggling `.*` never changes whether `é` finds `É`.
    /// Expected spans come from Unicode's `CaseFolding.txt` (status C/S), not
    /// from either engine. The fold partners include ones of a DIFFERENT UTF-8
    /// length (`K` KELVIN SIGN is 3 bytes, `k` 1; `ſ` LONG S 2, `s` 1; `ẞ` 3,
    /// `ß` 2), so a literal match's byte length is not the needle's.
    #[test]
    fn literal_and_regex_fold_case_identically() {
        let cases: [(&str, &str, Vec<Range<u32>>); 9] = [
            ("é", "É é", vec![0..2, 3..5]),
            ("É", "é É", vec![0..2, 3..5]),
            // Final sigma folds to σ too (`03C2; C; 03C3`).
            ("σ", "Σ σ ς", vec![0..2, 3..5, 6..8]),
            ("Σ", "ς σ", vec![0..2, 3..5]),
            ("k", "\u{212A} k K", vec![0..3, 4..5, 6..7]),
            ("s", "\u{17F} S s", vec![0..2, 3..4, 5..6]),
            ("ß", "\u{1E9E} ß", vec![0..3, 4..6]),
            ("ſé", "SÉ sé", vec![0..3, 4..7]),
            // Simple folding only: no multi-char expansions (`ß` ↛ `ss`).
            ("ss", "ß", vec![]),
        ];
        for (needle, text, want) in cases {
            for regex in [false, true] {
                let query =
                    FindQuery { text: needle.into(), case_sensitive: false, regex, ..Default::default() };
                assert_eq!(scan(text, &query).0, want, "{needle:?} in {text:?} (regex={regex})");
            }
        }
        // Case-sensitive, both modes still tell the forms apart.
        for regex in [false, true] {
            let query = FindQuery { text: "é".into(), case_sensitive: true, regex, ..Default::default() };
            assert_eq!(scan("É é", &query).0, vec![3..5], "case-sensitive (regex={regex})");
        }
    }

    /// The claim `LiteralMatcher::fold_width` rests on, taken EXHAUSTIVELY
    /// from the regex crate (the owner of folding), not from memory: the only
    /// non-ASCII chars that `(?i)` folds into ASCII are `ſ` LONG S and `K`
    /// KELVIN SIGN, and the width bound of every ASCII char covers each of its
    /// fold partners. A Unicode bump that adds a partner fails here, not in a
    /// silently-too-narrow repair window.
    #[test]
    fn fold_width_covers_every_fold_partner_of_ascii() {
        let all_non_ascii: String = (0x80..=char::MAX as u32).filter_map(char::from_u32).collect();
        let into_ascii = regex::RegexBuilder::new(r"[\x00-\x7F]")
            .case_insensitive(true)
            .build()
            .expect("compiles");
        let partners: Vec<char> =
            into_ascii.find_iter(&all_non_ascii).flat_map(|m| m.as_str().chars()).collect();
        assert_eq!(partners, ['\u{17F}', '\u{212A}'], "the non-ASCII chars that fold into ASCII");
        for x in partners {
            for a in (0u8..0x80).map(char::from) {
                let pattern = format!("^{}$", regex::escape(a.encode_utf8(&mut [0; 4])));
                let re = regex::RegexBuilder::new(&pattern).case_insensitive(true).build().unwrap();
                if re.is_match(x.encode_utf8(&mut [0; 4])) {
                    assert!(
                        LiteralMatcher::fold_width(a) >= x.len_utf8() as u32,
                        "{a:?} folds with {x:?} ({} bytes)",
                        x.len_utf8()
                    );
                }
            }
        }
    }

    #[test]
    fn empty_query_matches_nothing() {
        assert_eq!(scan("anything", &q("", false)), (Vec::new(), false));
        assert_eq!(scan("anything", &q("", true)), (Vec::new(), false));
    }

    #[test]
    fn a_query_longer_than_the_text_finds_nothing() {
        assert_eq!(scan("hi", &q("hello", true)), (Vec::new(), false));
    }

    #[test]
    fn scan_equals_the_naive_oracle() {
        // Both literal paths must be bit-identical to the straightforward naive
        // scan, kept here verbatim as the oracle. Its byte-wise ASCII fold IS
        // Unicode simple folding on this alphabet (no `k`/`s`, whose fold
        // partners are non-ASCII, and nothing beyond ASCII).
        fn naive(text: &str, query: &FindQuery) -> (Vec<Range<u32>>, bool) {
            let needle = query.text.as_bytes();
            if needle.is_empty() {
                return (Vec::new(), false);
            }
            let hay = text.as_bytes();
            let hit = |window: &[u8]| {
                if query.case_sensitive {
                    window == needle
                } else {
                    window.iter().zip(needle).all(|(a, b)| a.eq_ignore_ascii_case(b))
                }
            };
            let mut spans = Vec::new();
            let mut i = 0;
            while i + needle.len() <= hay.len() {
                if hit(&hay[i..i + needle.len()]) {
                    if spans.len() == FIND_MATCH_CAP {
                        return (spans, true);
                    }
                    spans.push(i as u32..(i + needle.len()) as u32);
                    i += needle.len();
                } else {
                    i += 1;
                }
            }
            (spans, false)
        }

        let mut rng = 0x2545F4914F6CDD1Du64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let alphabet = ['a', 'A', 'b', 'B', 'c', ' '];
        for round in 0..200 {
            let text: String = (0..(next() % 120)).map(|_| alphabet[(next() % 6) as usize]).collect();
            let needle: String = (0..1 + (next() % 4)).map(|_| alphabet[(next() % 6) as usize]).collect();
            for cs in [true, false] {
                let query = q(&needle, cs);
                assert_eq!(
                    scan(&text, &query),
                    naive(&text, &query),
                    "round {round}: {needle:?} (cs={cs}) in {text:?}"
                );
            }
        }
    }

    #[test]
    fn scan_caps_and_reports() {
        // One past the cap trips it (both case paths)…
        let text = "ab".repeat(FIND_MATCH_CAP + 5);
        let (spans, capped) = scan(&text, &q("ab", true));
        assert_eq!((spans.len(), capped), (FIND_MATCH_CAP, true));
        let text = "aB".repeat(FIND_MATCH_CAP + 1);
        let (spans, capped) = scan(&text, &q("ab", false));
        assert_eq!((spans.len(), capped), (FIND_MATCH_CAP, true));
        // …exactly the cap does not (the cap trips only on an overflowing hit).
        let text = "ab".repeat(FIND_MATCH_CAP);
        let (spans, capped) = scan(&text, &q("ab", true));
        assert_eq!((spans.len(), capped), (FIND_MATCH_CAP, false));
    }

    /// The windowed buffer scan must be byte-identical to `scan` over the
    /// materialized text — spans AND capped flag — for every window size,
    /// including windows tiny enough that matches straddle every seam, the
    /// needle exceeds the window, multibyte chars sit on window ends (the
    /// boundary-snap paths), and self-overlapping needles cross seams. The
    /// KELVIN SIGN / LONG S corpus makes folded matches far longer than their
    /// needle (`kkkkks` ↔ the 17-byte `KKKKKſ`), so a seam overlap sized from
    /// the needle rather than the longest match drops the straddlers.
    #[test]
    fn windowed_scan_equals_the_whole_text_scan() {
        let corpora = [
            String::new(),
            "aaaa".into(),
            "abababab".into(),
            "aa aa aaa aaaa a".into(),
            "the fox\nreturns the fox to the FOX den\nfoxfoxfox\n".repeat(40),
            // Multibyte chars packed so window ends land mid-char for most
            // small window sizes; matches sit between and across them.
            "ä🦀ab🦀äab日本語ab".repeat(30),
            "🦀🦀🦀ab🦀🦀🦀".repeat(50),
            "\u{212A}\u{212A}\u{212A}\u{212A}\u{212A}ſ kkkkks KK\u{212A}kKS \u{212A}ſ ks ".repeat(20),
        ];
        let needles = [
            "a", "ab", "aa", "aaa", "fox", "FOX", "🦀ä", "ab🦀", "abababababab", "語ab", "ks", "kkkkks",
        ];
        for text in &corpora {
            let b = Buffer::new(text).unwrap();
            let full_text = b.text();
            for needle in needles {
                for cs in [true, false] {
                    let query = q(needle, cs);
                    let expect = scan(&full_text, &query);
                    let m = Matcher::compile(&query).expect("a literal compiles");
                    for window in [2, 3, 5, 7, 16, 64, 4096] {
                        assert_eq!(
                            scan_buffer(&b, &m, window, 0..b.len()),
                            expect,
                            "window {window}, needle {needle:?} (cs={cs}) in {:?}…",
                            &text[..text.len().min(24)]
                        );
                    }
                }
            }
        }
    }

    /// Whole-word boundaries at a LINE's edges. The character outside a match at
    /// the start or end of a line is `\n` (or the document edge) — both
    /// non-word — so such a word IS whole. This is precisely the case the
    /// line-scoped model gets right for free and a byte-window scan cannot: a
    /// window's slice does not contain the character before its own first byte.
    #[test]
    fn whole_word_is_bounded_by_line_edges() {
        //          0123456 7 89.. 11 12....18 19
        let text = "foo bar\nfoo\nbarfoo foo";
        assert_eq!(
            scan(text, &qw("foo", false)).0,
            vec![0..3, 8..11, 19..22],
            "line-start, whole-line, and line-end words are whole; `barfoo` is not"
        );
        // Without the option, the substring inside `barfoo` matches too.
        assert_eq!(scan(text, &q("foo", false)).0, vec![0..3, 8..11, 15..18, 19..22]);
    }

    /// A regex can never match across a line: the engine is fed one line at a
    /// time WITHOUT its newline, so `\n` is not in the haystack at all — this is
    /// structural, not a convention, and it is what keeps the per-keystroke
    /// repair windowed for a variable-length pattern.
    #[test]
    fn a_regex_can_never_match_across_a_line() {
        let text = "foo\nbar";
        for pattern in [r"foo\nbar", r"foo.bar", r"foo\s+bar", r"foo[^x]bar"] {
            assert!(
                scan(text, &qr(pattern, false)).0.is_empty(),
                "{pattern:?} must not span the newline"
            );
        }
        // …while each line matches on its own, with `^`/`$` meaning its edges.
        assert_eq!(scan(text, &qr("^foo$", false)).0, vec![0..3]);
        assert_eq!(scan(text, &qr("^bar$", false)).0, vec![4..7]);
        assert_eq!(scan(text, &qr("[a-z]+", false)).0, vec![0..3, 4..7]);
    }

    /// An unfinished pattern is a NORMAL state — it is what every prefix of a
    /// regex being typed looks like — so it matches nothing rather than
    /// erroring, and resumes the moment it parses.
    #[test]
    fn an_unfinished_pattern_matches_nothing() {
        for pattern in ["(", "[a-", "a{2", "*"] {
            assert!(scan("aaa (bbb)", &qr(pattern, false)).0.is_empty(), "{pattern:?}");
        }
        assert_eq!(scan("aaa (bbb)", &qr(r"\(b+\)", false)).0, vec![4..9]);
    }

    /// Zero-width hits are dropped: there is nothing to navigate to, highlight,
    /// or replace, and the store would discard them anyway — so the count stays
    /// honest about what the user can act on.
    #[test]
    fn zero_width_regex_hits_are_dropped() {
        assert!(scan("bbb\n\nbbb", &qr("a*", false)).0.is_empty());
        assert_eq!(scan("bab", &qr("a*", false)).0, vec![1..2], "only the non-empty hit");
    }

    /// A needle whose edge character is punctuation is still findable as a
    /// whole word — the four repros of the `\b…\b` bug, where `\b` before `(`
    /// demanded a word character *outside* the match and so could never hold.
    /// The rule (VS Code's): a side whose edge character is a word character
    /// needs a non-word neighbour (or the line edge); a side whose edge
    /// character is a separator matches whatever its neighbour is.
    #[test]
    fn whole_word_matches_a_needle_with_punctuation_edges() {
        assert_eq!(scan(" (foo) ", &qw("(foo", false)).0, vec![1..5], "`(foo` inside parens");
        assert_eq!(scan("a == b", &qw("==", false)).0, vec![2..4], "an all-punctuation needle");
        //               0123456789.123456789
        assert_eq!(
            scan("foo.bar afoo. foo.", &qw("foo.", false)).0,
            vec![0..4, 14..18],
            "the `.` side is free (even against `b`); the `f` side still needs a boundary"
        );
        //               0123456789.1234
        assert_eq!(
            scan("x = -1; y-1 -12", &qw("-1", false)).0,
            vec![4..6, 9..11],
            "the `-` side is free (even against `y`); the `1` side rejects `-12`"
        );
    }

    /// A whole-word word needle is still rejected inside a longer word on either
    /// side — the half of the rule the punctuation fix must not loosen. `_` is a
    /// word character (the `movement::is_word_char` rule), so `foo_x` is one word.
    #[test]
    fn whole_word_still_rejects_a_word_inside_a_longer_word() {
        //          0123456789.123456789.123456
        let text = "foobar barfoo foo_x xfoo foo";
        assert_eq!(scan(text, &qw("foo", false)).0, vec![25..28]);
        assert_eq!(scan(text, &qw("foo", true)).0, vec![25..28]);
    }

    /// Whole-word REGEX judges each match by its own real edge characters —
    /// what the pattern actually matched, not the pattern's spelling — so
    /// `\(foo` behaves exactly like the literal `(foo`, and an optional
    /// `\(?` prefix makes the `(` side free only on the matches that took it.
    #[test]
    fn whole_word_regex_judges_each_matchs_real_edges() {
        let qrw = |p: &str| FindQuery {
            text: p.into(),
            whole_word: true,
            regex: true,
            ..Default::default()
        };
        assert_eq!(scan(" (foo) ", &qrw(r"\(foo")).0, vec![1..5], r"`\(foo` ≡ the literal `(foo`");
        //          0123456789.12
        let text = "(foo afoo foo";
        assert_eq!(
            scan(text, &qrw(r"\(?foo")).0,
            vec![0..4, 10..13],
            "the match that took the `(` is free on its left; the bare one inside `afoo` is not"
        );
        // The match's own edges, per match: `.foo` hits `(foo` (separator edge,
        // free) and ` afoo`'s `afoo` (word edge, bounded by the space) but not
        // `xafoo`'s `afoo` (word edge against `x`).
        //               0123456789.1234
        assert_eq!(scan("(foo afoo xafoo", &qrw(".foo")).0, vec![0..4, 5..9]);
    }

    /// Whole-word compiles the ESCAPED literal, so regex metacharacters in a
    /// plain whole-word query stay literal.
    #[test]
    fn whole_word_escapes_its_literal() {
        assert_eq!(scan("a.c abc", &qw("a.c", false)).0, vec![0..3], "`.` is literal here");
        // …and a whole-word REGEX edge-checks every alternative's match, so
        // alternation is bounded on both sides of each branch (the old wrapping
        // needed a `(?:…)` group for this; an edge check never splits a pattern).
        let q = FindQuery {
            text: "foo|bar".into(),
            whole_word: true,
            regex: true,
            ..Default::default()
        };
        assert_eq!(scan("foo bar foobar", &q).0, vec![0..3, 4..7], "neither inside `foobar`");
    }

    /// Replace-all walks the SAME whole-word iteration as the display scan and
    /// expands each kept match's own groups: the `(fo` inside `(foo` is
    /// rejected (its `o` edge meets `o`), the other two — `(` edges, free of
    /// their `x` / space neighbours — are replaced with `$1` expanded.
    #[test]
    fn whole_word_regex_replace_expands_only_the_kept_matches() {
        let q = FindQuery {
            text: r"\((fo)".into(),
            whole_word: true,
            regex: true,
            ..Default::default()
        };
        //          0123456789.1234
        let text = "(foo) (fo) x(fo";
        let b = Buffer::new(text).unwrap();
        let m = Matcher::compile(&q).expect("the pattern compiles");
        assert_eq!(
            replacements(&b, &m, 0..b.len(), "<$1>", false),
            vec![(6..9, "<fo>".to_string()), (12..15, "<fo>".to_string())]
        );
        assert_eq!(scan(text, &q).0, vec![6..9, 12..15], "the display scan agrees");
    }

    /// A rejected whole-word candidate resumes one char past its START, so it
    /// can never hide a valid match that begins inside it: in `xa-a-a`, `1..4`
    /// fails against the `x`, and `3..6` — bounded by the `-` — is still found.
    #[test]
    fn a_rejected_whole_word_candidate_does_not_hide_a_later_one() {
        assert_eq!(scan("xa-a-a", &qw("a-a", true)).0, vec![3..6]);
    }

    /// Without whole-word, the line iterator must be byte-identical to the
    /// regex crate's own `find_iter` with zero-width hits dropped — the plain
    /// regex mode routes through the new cursor too, and must not change. The
    /// empty-match-heavy patterns (`a*|b`, `\b`, `$|a`) are the ones whose
    /// resume rule differs most visibly if the two ever drift.
    #[test]
    fn line_matches_without_whole_word_equal_find_iter() {
        let mut rng = 0x9E37_79B9_7F4A_7C15u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let alphabet = ['a', 'b', 'A', ' ', '_', '(', 'é', '🦀', '1'];
        let patterns = [
            "a", "a*", "a*|b", r"\b", "x?", "[ab]+", "(?:)", "é*", "^", "$|a", ".", r"\w+",
            "b|", "a??", r"\(a", "🦀|a",
        ];
        for pattern in patterns {
            for cs in [true, false] {
                let Ok(Matcher::Lines(lm)) = Matcher::compile(&qr(pattern, cs)) else {
                    panic!("{pattern:?} compiles to the line engine");
                };
                for round in 0..60 {
                    let line: String =
                        (0..(next() % 24)).map(|_| alphabet[(next() % 9) as usize]).collect();
                    let want: Vec<Range<usize>> = lm
                        .re
                        .find_iter(&line)
                        .filter(|m| m.start() < m.end())
                        .map(|m| m.range())
                        .collect();
                    let got: Vec<Range<usize>> = lm.find_iter(&line).collect();
                    assert_eq!(got, want, "round {round}: {pattern:?} (cs={cs}) in {line:?}");
                }
            }
        }
    }

    /// Whole-word against an INDEPENDENT oracle: the greedy, leftmost,
    /// non-overlapping set of valid placements, judged byte by byte straight
    /// from the rule (a side fails only when its edge byte and its neighbour
    /// are both word bytes; `\n` and the text's ends are never word bytes).
    /// Random ASCII corpora rich in punctuation, `_`, and newlines, every
    /// needle both as a whole-word literal AND as the whole-word regex of its
    /// escape — which must agree (`\(foo` ≡ `(foo`).
    #[test]
    fn whole_word_equals_the_naive_placement_oracle() {
        fn naive(text: &str, needle: &str, cs: bool) -> Vec<Range<u32>> {
            let (hay, nd) = (text.as_bytes(), needle.as_bytes());
            let word = |b: u8| b.is_ascii_alphanumeric() || b == b'_';
            let mut out = Vec::new();
            let mut i = 0;
            while !nd.is_empty() && i + nd.len() <= hay.len() {
                let (w, j) = (&hay[i..i + nd.len()], i + nd.len());
                let hit = if cs { w == nd } else { w.eq_ignore_ascii_case(nd) };
                let left = !(word(w[0]) && i > 0 && word(hay[i - 1]));
                let right = !(word(w[nd.len() - 1]) && j < hay.len() && word(hay[j]));
                if hit && left && right {
                    out.push(i as u32..j as u32);
                    i = j;
                } else {
                    i += 1;
                }
            }
            out
        }

        let mut rng = 0x0DDB_A11C_AFE5_EED5u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        let alphabet = ['a', 'A', 'b', '_', '1', '(', '.', '-', '=', ' ', '\n'];
        let mut nonempty = 0;
        for round in 0..400 {
            let text: String =
                (0..(next() % 80)).map(|_| alphabet[(next() % 11) as usize]).collect();
            // Needles never hold `\n`: the line-scoped engine cannot match one.
            let needle: String =
                (0..1 + (next() % 4)).map(|_| alphabet[(next() % 10) as usize]).collect();
            for cs in [true, false] {
                let want = naive(&text, &needle, cs);
                nonempty += usize::from(!want.is_empty());
                assert_eq!(
                    scan(&text, &qw(&needle, cs)).0,
                    want,
                    "round {round}: literal {needle:?} (cs={cs}) in {text:?}"
                );
                let as_regex = FindQuery {
                    text: regex::escape(&needle),
                    case_sensitive: cs,
                    whole_word: true,
                    regex: true,
                };
                assert_eq!(
                    scan(&text, &as_regex).0,
                    want,
                    "round {round}: regex {:?} (cs={cs}) in {text:?}",
                    as_regex.text
                );
            }
        }
        assert!(nonempty > 100, "the oracle must not pass vacuously ({nonempty} non-empty)");
    }

    /// A **scoped** scan must equal the whole-document scan restricted to the
    /// scope — the oracle for find-in-selection. Scanning a sub-range is not the
    /// same as filtering the full scan (the greedy non-overlap cursor restarts
    /// at the scope's start), so this pins the one case where they must agree:
    /// a scope whose start sits on a match boundary. Scope edges are swept
    /// across every offset, including mid-char ones, which must clip inward
    /// rather than panic.
    #[test]
    fn scoped_scan_equals_the_whole_text_scan_within_the_scope() {
        let corpora = [
            "fox fox fox fox fox".to_string(),
            "the fox\nreturns the fox to the FOX den\nfoxfoxfox\n".repeat(8),
            "ä🦀ab🦀äab日本語ab".repeat(6),
        ];
        for text in &corpora {
            let b = Buffer::new(text).unwrap();
            let full = b.text();
            for needle in ["a", "ab", "fox", "🦀ä"] {
                for cs in [true, false] {
                    let query = q(needle, cs);
                    let all = scan(&full, &query).0;
                    let m = Matcher::compile(&query).expect("a literal compiles");
                    for lo in 0..=b.len() {
                        for hi in lo..=b.len() {
                            // Snap the bound the way the scan does, then take
                            // every whole-document match that fits inside it.
                            let (slo, shi) =
                                (b.clip_offset(lo, Bias::Right), b.clip_offset(hi, Bias::Left));
                            if slo > shi {
                                continue;
                            }
                            let expect: Vec<_> = all
                                .iter()
                                .filter(|m| m.start >= slo && m.end <= shi)
                                .cloned()
                                .collect();
                            let got = scan_buffer(&b, &m, 4096, lo..hi).0;
                            // Only compare where the scope opens ON a match
                            // boundary; elsewhere a scope can legitimately expose
                            // a match the greedy full scan skipped.
                            if expect.first().is_none_or(|m| m.start == slo)
                                && all.iter().all(|m| m.end <= slo || m.start >= slo)
                            {
                                assert_eq!(got, expect, "scope {lo}..{hi}, needle {needle:?}");
                            }
                            // Unconditionally: never escape the scope.
                            assert!(
                                got.iter().all(|m| m.start >= slo && m.end <= shi),
                                "scope {lo}..{hi} leaked a match {got:?}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// The windowed scan reproduces the cap semantics exactly: capped ⇒ the
    /// identical 10k-prefix, and exactly-at-cap ⇒ not capped.
    #[test]
    fn windowed_scan_caps_like_the_whole_text_scan() {
        for extra in [0usize, 3] {
            let text = "ab".repeat(FIND_MATCH_CAP + extra);
            let b = Buffer::new(&text).unwrap();
            let query = q("ab", true);
            let m = Matcher::compile(&query).expect("a literal compiles");
            assert_eq!(
                scan_buffer(&b, &m, 4096, 0..b.len()),
                scan(&b.text(), &query),
                "extra {extra}"
            );
        }
    }

    /// The O(log) [`FindState::navigate`]
    /// (`find_count`/`find_rank_before`/`nth_find`) must pick the byte-identical
    /// `(id, range)` — and set `active`/`active_start` in lockstep — as a
    /// straightforward whole-list walk, over random find sets and random
    /// `(head, selection, forward)`: on-match cycling both ways, off-match scan +
    /// wrap both ways, and the empty set. Non-find decorations are interleaved as
    /// noise that neither path may count.
    #[test]
    fn navigate_equals_the_full_list_walk() {
        // The full-list navigate, retained verbatim as the oracle.
        fn navigate_ref(
            store: &DecorationStore,
            head: u32,
            selection: Range<u32>,
            forward: bool,
        ) -> Option<(DecorationId, Range<u32>)> {
            let live: Vec<(DecorationId, Range<u32>)> = store
                .decorations_in(0..u32::MAX)
                .filter(|r| is_find(r) && r.range.start < r.range.end)
                .map(|r| (r.id, r.range.clone()))
                .collect();
            if live.is_empty() {
                return None;
            }
            let on = live.iter().position(|(_, r)| *r == selection);
            let pick = match (on, forward) {
                (Some(p), true) => (p + 1) % live.len(),
                (Some(p), false) => (p + live.len() - 1) % live.len(),
                (None, true) => live.iter().position(|(_, r)| r.start >= head).unwrap_or(0),
                (None, false) => {
                    live.iter().rposition(|(_, r)| r.start < head).unwrap_or(live.len() - 1)
                }
            };
            Some(live[pick].clone())
        }

        let mut rng = 0x1234_5678_9ABC_DEF0u64;
        let mut next = move || {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng
        };
        for trial in 0..3000u32 {
            let mut store = DecorationStore::new();
            // Disjoint ascending non-empty finds.
            let n = next() % 12;
            let mut pos = 0u32;
            let mut find_spans: Vec<Range<u32>> = Vec::new();
            for _ in 0..n {
                pos += (next() % 5) as u32;
                let len = 1 + (next() % 4) as u32;
                find_spans.push(pos..pos + len);
                pos += len;
            }
            store.add_sorted_batch(
                find_spans.iter().cloned(),
                DecorationKind::FindMatch,
                Stickiness::NeverGrows,
            );
            // Non-find noise (must be ignored by both paths).
            for _ in 0..(next() % 5) {
                let s = (next() % u64::from(pos + 1)) as u32;
                let l = (next() % 4) as u32;
                store.add_decoration(s..s + l, DecorationKind::AutoClosePair, Stickiness::AlwaysGrows);
            }
            // No persisted find may be empty (FindMatch is EmptyPolicy::Drop);
            // the O(1) `find_count` counts all finds, so this must hold for it to
            // equal a `start < end` filter.
            debug_assert!(
                store.decorations_in(0..u32::MAX).filter(is_find).all(|r| r.range.start < r.range.end),
                "trial {trial}: an empty FindMatch persisted"
            );

            let head = (next() % u64::from(pos + 6)) as u32;
            // Half the time, sit exactly on a match to exercise on-match cycling.
            let selection = if !find_spans.is_empty() && next() % 2 == 0 {
                find_spans[(next() as usize) % find_spans.len()].clone()
            } else {
                let a = (next() % u64::from(pos + 6)) as u32;
                let b = a + (next() % 5) as u32;
                a..b
            };
            for forward in [true, false] {
                let want = navigate_ref(&store, head, selection.clone(), forward);
                let mut state = FindState::new();
                let got = state.navigate(head, selection.clone(), &store, forward);
                match (&want, &got) {
                    (Some((id, range)), Some(got_range)) => {
                        assert_eq!(got_range, range, "trial {trial} fwd={forward}: range");
                        assert_eq!(state.active, Some(*id), "trial {trial} fwd={forward}: active id");
                        assert_eq!(
                            state.active_start,
                            Some(range.start),
                            "trial {trial} fwd={forward}: active_start tracks the pick"
                        );
                    }
                    (None, None) => {
                        assert_eq!(state.active, None, "trial {trial} fwd={forward}");
                        assert_eq!(state.active_start, None, "trial {trial} fwd={forward}");
                    }
                    _ => panic!("trial {trial} fwd={forward}: {want:?} vs {got:?}"),
                }
            }
        }
    }
}
