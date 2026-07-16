//! The snippet parser: the LSP placeholder subset `spec.rs` emits, parsed
//! into a body with defaults expanded plus an ordered list of tab stops. Pure
//! text — no document, no transaction; the session state machine that drives the
//! decoration store consumes this. Items stay inert data until accept time, so
//! parsing happens then, never earlier.
//!
//! Grammar: `$0` / `${0}` = the final stop (≤1; implicit at end-of-body when
//! absent); `$N` = an empty placeholder; `${N:default}` = a placeholder with
//! literal default text (nesting a placeholder inside a default is an error);
//! `${N|a,b,c|}` = a choice (the first is the inserted default); `\$ \} \\`
//! escape their characters. A duplicate index (mirrored placeholders) is an
//! error — no `spec.rs` snippet generates one.

use core::ops::Range;

use crate::decorations::{DecorationId, DecorationKind, DecorationStore, Stickiness};

/// A parsed snippet: the body with every placeholder's default expanded, plus
/// the tab stops to visit.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Snippet {
    /// The body with placeholders replaced by their default text. LF-normal
    /// already (the caller's expander normalizes EOL and adapts `\t` indent at
    /// insert time); this parser leaves `\t` literal.
    pub text: String,
    /// Stops in visit order: ascending placeholder index, the final stop last.
    pub stops: Vec<TabStop>,
}

/// One tab stop within a [`Snippet`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TabStop {
    /// 1-based placeholder index; [`u16::MAX`] marks the final stop (`$0`).
    pub index: u16,
    /// Byte range within [`Snippet::text`] covering the inserted default (empty
    /// for `$N` / `$0`).
    pub range: Range<u32>,
    /// Non-empty for `${N|a,b|}`; `choices[0]` is the inserted default.
    pub choices: Vec<String>,
}

impl TabStop {
    /// Whether this is the final stop (`$0`), where the caret lands and the
    /// session ends.
    #[must_use]
    pub fn is_final(&self) -> bool {
        self.index == u16::MAX
    }
}

/// Why a snippet body failed to parse. Real `spec.rs` snippets never trigger
/// these (its own suite pins validity); the insertion path treats a failure as
/// "insert the raw body verbatim" behind a `debug_assert`, so this stays a
/// clean `Result` here.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
pub enum SnippetError {
    /// The same placeholder index appeared twice (mirrored placeholders are
    /// unsupported). `0` reports the final stop (`$0`).
    #[error("duplicate placeholder index {0}")]
    DuplicateIndex(u16),
    /// A placeholder appeared inside a default (defaults are literal text).
    #[error("nested placeholder inside a default")]
    Nesting,
    /// A malformed placeholder token.
    #[error("malformed placeholder: {0}")]
    Malformed(&'static str),
}

impl Snippet {
    /// Adapt a parsed snippet for insertion at a line whose leading whitespace is
    /// `indent` (captured pre-edit), expanding tabs to `indent_size` spaces — the
    /// insertion transform (pure text; no transaction machinery):
    /// 1. EOL-normalize (`\r\n` / `\r` → `\n`);
    /// 2. every `\n` → `\n` + `indent`, so continuation lines align with the
    ///    insertion column (spec.rs bodies are written flush-left);
    /// 3. every `\t` → `indent_size` spaces (one indent unit);
    /// 4. stop ranges remap through the same rewrite.
    ///
    /// The caret lands at the final stop; the returned stop ranges are relative
    /// to the returned `text`, which the caller inserts as one transaction.
    #[must_use]
    pub fn for_insertion(&self, indent: &str, indent_size: usize) -> Snippet {
        use std::collections::BTreeMap;
        let tab = " ".repeat(indent_size);
        let mut out = String::with_capacity(self.text.len());
        // input byte offset (char boundary) → output byte offset.
        let mut map: BTreeMap<u32, u32> = BTreeMap::new();
        let mut it = self.text.char_indices().peekable();
        while let Some((byte_i, c)) = it.next() {
            map.insert(byte_i as u32, out.len() as u32);
            match c {
                '\r' => {
                    if it.peek().map(|&(_, c)| c) == Some('\n') {
                        it.next(); // swallow the LF of a CRLF
                    }
                    out.push('\n');
                    out.push_str(indent);
                }
                '\n' => {
                    out.push('\n');
                    out.push_str(indent);
                }
                '\t' => out.push_str(&tab),
                _ => out.push(c),
            }
        }
        map.insert(self.text.len() as u32, out.len() as u32);

        let remap = |off: u32| map.get(&off).copied().unwrap_or(off);
        let stops = self
            .stops
            .iter()
            .map(|s| TabStop {
                index: s.index,
                range: remap(s.range.start)..remap(s.range.end),
                choices: s.choices.clone(),
            })
            .collect();
        Snippet { text: out, stops }
    }

    /// Parse an LSP placeholder body (see the module grammar).
    ///
    /// # Errors
    /// [`SnippetError`] on a duplicate index, a nested placeholder in a default,
    /// or a malformed `${…}` token.
    pub fn parse(body: &str) -> Result<Self, SnippetError> {
        let chars: Vec<char> = body.chars().collect();
        let mut text = String::new();
        let mut stops: Vec<TabStop> = Vec::new();
        let mut seen: Vec<u16> = Vec::new();
        let mut i = 0;

        while i < chars.len() {
            match chars[i] {
                // Escape: \$ \} \\ → the bare character.
                '\\' if matches!(chars.get(i + 1), Some('$' | '}' | '\\')) => {
                    text.push(chars[i + 1]);
                    i += 2;
                }
                '$' if matches!(chars.get(i + 1), Some('{')) => {
                    i += 2; // past "${"
                    let index = read_index(&chars, &mut i)?;
                    match chars.get(i) {
                        Some('}') => {
                            i += 1;
                            let pos = text.len() as u32;
                            push_stop(&mut stops, &mut seen, index, pos..pos, Vec::new())?;
                        }
                        Some(':') => {
                            i += 1;
                            let start = text.len() as u32;
                            read_default(&chars, &mut i, &mut text)?;
                            let end = text.len() as u32;
                            push_stop(&mut stops, &mut seen, index, start..end, Vec::new())?;
                        }
                        Some('|') => {
                            i += 1;
                            let choices = read_choices(&chars, &mut i)?;
                            let start = text.len() as u32;
                            text.push_str(choices.first().map_or("", String::as_str));
                            let end = text.len() as u32;
                            push_stop(&mut stops, &mut seen, index, start..end, choices)?;
                        }
                        _ => return Err(SnippetError::Malformed("expected } : or | after ${N")),
                    }
                }
                '$' if matches!(chars.get(i + 1), Some(c) if c.is_ascii_digit()) => {
                    i += 1; // past '$'
                    let index = read_index(&chars, &mut i)?;
                    let pos = text.len() as u32;
                    push_stop(&mut stops, &mut seen, index, pos..pos, Vec::new())?;
                }
                // A bare `$` (not a placeholder) is literal.
                c => {
                    text.push(c);
                    i += 1;
                }
            }
        }

        // An absent final stop is implicit at end-of-body.
        if !seen.contains(&u16::MAX) {
            let pos = text.len() as u32;
            stops.push(TabStop { index: u16::MAX, range: pos..pos, choices: Vec::new() });
        }
        // Visit order: ascending index; u16::MAX (final) sorts last.
        stops.sort_by_key(|s| s.index);
        Ok(Snippet { text, stops })
    }
}

/// Read a run of ASCII digits into a placeholder index (`0` → the final stop's
/// `u16::MAX`). Advances `i` past the digits.
fn read_index(chars: &[char], i: &mut usize) -> Result<u16, SnippetError> {
    let start = *i;
    while matches!(chars.get(*i), Some(c) if c.is_ascii_digit()) {
        *i += 1;
    }
    if *i == start {
        return Err(SnippetError::Malformed("placeholder without an index"));
    }
    let n: u32 = chars[start..*i]
        .iter()
        .collect::<String>()
        .parse()
        .map_err(|_| SnippetError::Malformed("placeholder index out of range"))?;
    if n == 0 {
        Ok(u16::MAX)
    } else {
        u16::try_from(n).map_err(|_| SnippetError::Malformed("placeholder index out of range"))
    }
}

/// Read `${N:` default text into `text` until the unescaped closing `}`. A
/// placeholder (`$`) inside is nesting (an error).
fn read_default(chars: &[char], i: &mut usize, text: &mut String) -> Result<(), SnippetError> {
    loop {
        match chars.get(*i) {
            None => return Err(SnippetError::Malformed("unterminated ${N:default}")),
            Some('\\') if matches!(chars.get(*i + 1), Some('$' | '}' | '\\')) => {
                text.push(chars[*i + 1]);
                *i += 2;
            }
            Some('}') => {
                *i += 1;
                return Ok(());
            }
            Some('$') => return Err(SnippetError::Nesting),
            Some(&c) => {
                text.push(c);
                *i += 1;
            }
        }
    }
}

/// Read `${N|a,b,c|}` choices (comma-separated, `|}`-terminated). Advances `i`
/// past the closing `|}`.
fn read_choices(chars: &[char], i: &mut usize) -> Result<Vec<String>, SnippetError> {
    let mut choices = Vec::new();
    let mut cur = String::new();
    loop {
        match chars.get(*i) {
            None => return Err(SnippetError::Malformed("unterminated ${N|choices|}")),
            Some('\\') if matches!(chars.get(*i + 1), Some('$' | '}' | '\\')) => {
                cur.push(chars[*i + 1]);
                *i += 2;
            }
            Some(',') => {
                choices.push(core::mem::take(&mut cur));
                *i += 1;
            }
            Some('|') => {
                *i += 1;
                if chars.get(*i) != Some(&'}') {
                    return Err(SnippetError::Malformed("choice not closed with |}"));
                }
                *i += 1;
                choices.push(cur);
                return Ok(choices);
            }
            Some(&c) => {
                cur.push(c);
                *i += 1;
            }
        }
    }
}

/// What a Tab / Shift+Tab did to a live session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum TabOutcome {
    /// Moved to a stop; the caller selects this range.
    Move(Range<u32>),
    /// Reached the final stop — the session has ended (ranges unregistered);
    /// the caller collapses the caret at this offset.
    Finish(u32),
    /// No-op (Shift+Tab at the first stop).
    Stay,
}

/// What a caret move (click) did to a live session.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CaretOutcome {
    /// The caret entered a different stop; the caller selects this range.
    Move(Range<u32>),
    /// The caret stayed within the active stop.
    Stay,
    /// The caret left every stop; the caller cancels the session.
    Escaped,
}

/// A live snippet session: one tracked range per stop, the active stop
/// `AlwaysGrows` and the rest `NeverGrows`. It never touches the document or
/// history — the caller inserts the snippet text (one sealed transaction),
/// selects the range this reports, and the session only registers / swaps /
/// unregisters ranges. Cancellation is never a transaction: text stays, ranges
/// unregister, nothing enters history.
pub struct SnippetSession {
    /// Decoration handles for every stop in visit order; the final stop is last.
    stops: Vec<DecorationId>,
    /// Index into `stops` of the active stop — always a non-final stop
    /// (`0..stops.len() - 1`).
    active: usize,
}

impl SnippetSession {
    /// Start a session from a just-inserted snippet whose stop ranges are offset
    /// by `base` (the insert position). Registers one `SnippetStop` per stop
    /// (the first `AlwaysGrows`, the rest `NeverGrows`) and returns the session
    /// plus the first stop's content for the caller to select. `None` — no
    /// session — when the snippet has no stop besides the final.
    pub fn start(snippet: &Snippet, base: u32, store: &mut DecorationStore) -> Option<(Self, Range<u32>)> {
        if snippet.stops.len() < 2 {
            return None; // only the (implicit or explicit) final stop
        }
        let stops: Vec<DecorationId> = snippet
            .stops
            .iter()
            .enumerate()
            .map(|(i, s)| {
                let range = (base + s.range.start)..(base + s.range.end);
                let stickiness = if i == 0 { Stickiness::AlwaysGrows } else { Stickiness::NeverGrows };
                store.add_decoration(range, DecorationKind::SnippetStop { index: i as u8 }, stickiness)
            })
            .collect();
        let session = Self { stops, active: 0 };
        let first = store.decoration_range(session.stops[0]).expect("just registered");
        Some((session, first))
    }

    /// The ordinal (visit-order) index of the active stop.
    #[must_use]
    pub fn active_index(&self) -> usize {
        self.active
    }

    /// The active stop's current content range (post-edit), or `None` if it was
    /// dropped.
    #[must_use]
    pub fn active_range(&self, store: &DecorationStore) -> Option<Range<u32>> {
        store.decoration_range(self.stops[self.active])
    }

    /// Tab (`forward`) / Shift+Tab (`!forward`) between stops. Swaps stickiness
    /// on a move; Tab past the last non-final stop ends the session at the final
    /// stop ([`TabOutcome::Finish`], ranges already unregistered).
    pub fn tab(&mut self, forward: bool, store: &mut DecorationStore) -> TabOutcome {
        let last = self.stops.len() - 1; // the final stop's index in `stops`
        if forward {
            let next = self.active + 1;
            if next == last {
                let pos = store.decoration_range(self.stops[last]).map_or(0, |r| r.start);
                self.cancel(store);
                return TabOutcome::Finish(pos);
            }
            self.activate(next, store);
            TabOutcome::Move(self.active_range(store).expect("active stop live"))
        } else if self.active == 0 {
            TabOutcome::Stay
        } else {
            self.activate(self.active - 1, store);
            TabOutcome::Move(self.active_range(store).expect("active stop live"))
        }
    }

    /// A caret landed at `offset` (a click): re-activate a different stop it lands
    /// in, stay if it's the active stop, or report that it left every stop (the
    /// caller then cancels).
    pub fn on_caret(&mut self, offset: u32, store: &mut DecorationStore) -> CaretOutcome {
        let last = self.stops.len() - 1;
        for i in 0..last {
            if let Some(r) = store.decoration_range(self.stops[i]) {
                if offset >= r.start && offset <= r.end {
                    if i == self.active {
                        return CaretOutcome::Stay;
                    }
                    self.activate(i, store);
                    return CaretOutcome::Move(self.active_range(store).expect("active stop live"));
                }
            }
        }
        CaretOutcome::Escaped
    }

    /// Whether an edit spanning `range` lands wholly outside every stop (→ the
    /// caller cancels). An edit touching any stop keeps the session.
    #[must_use]
    pub fn edit_escapes(&self, range: &Range<u32>, store: &DecorationStore) -> bool {
        !self.stops.iter().any(|&id| {
            store.decoration_range(id).is_some_and(|r| range.start <= r.end && r.start <= range.end)
        })
    }

    /// Unregister every stop range. Cancellation is never a transaction.
    pub fn cancel(&mut self, store: &mut DecorationStore) {
        for id in self.stops.drain(..) {
            store.take_decoration(id);
        }
    }

    /// Swap the active stop to ordinal `i`, moving `AlwaysGrows` with it.
    fn activate(&mut self, i: usize, store: &mut DecorationStore) {
        store.set_decoration_stickiness(self.stops[self.active], Stickiness::NeverGrows);
        store.set_decoration_stickiness(self.stops[i], Stickiness::AlwaysGrows);
        self.active = i;
    }
}

/// Record a stop, rejecting a duplicate index (mirrored placeholders).
fn push_stop(
    stops: &mut Vec<TabStop>,
    seen: &mut Vec<u16>,
    index: u16,
    range: Range<u32>,
    choices: Vec<String>,
) -> Result<(), SnippetError> {
    if seen.contains(&index) {
        return Err(SnippetError::DuplicateIndex(if index == u16::MAX { 0 } else { index }));
    }
    seen.push(index);
    stops.push(TabStop { index, range, choices });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(body: &str) -> Snippet {
        Snippet::parse(body).expect("valid snippet")
    }

    fn stop(index: u16, range: Range<u32>, choices: &[&str]) -> TabStop {
        TabStop { index, range, choices: choices.iter().map(|s| s.to_string()).collect() }
    }

    #[test]
    fn plain_body_has_only_an_implicit_final_stop_at_the_end() {
        let s = parse("hello world");
        assert_eq!(s.text, "hello world");
        assert_eq!(s.stops, vec![stop(u16::MAX, 11..11, &[])]);
    }

    #[test]
    fn default_placeholder_expands_and_ranges_cover_it() {
        let s = parse("${1:name}");
        assert_eq!(s.text, "name");
        assert_eq!(s.stops, vec![stop(1, 0..4, &[]), stop(u16::MAX, 4..4, &[])]);
    }

    #[test]
    fn empty_and_multiple_placeholders_track_positions() {
        let s = parse("$1 and $2");
        assert_eq!(s.text, " and ");
        assert_eq!(s.stops, vec![stop(1, 0..0, &[]), stop(2, 5..5, &[]), stop(u16::MAX, 5..5, &[])]);
    }

    #[test]
    fn explicit_final_stop_is_placed_and_not_duplicated_at_end() {
        let s = parse("foo($0)bar");
        assert_eq!(s.text, "foo()bar");
        assert_eq!(s.stops, vec![stop(u16::MAX, 4..4, &[])]);
    }

    #[test]
    fn choice_inserts_the_first_and_keeps_all() {
        let s = parse("${1|hs_can,ms_can,kline|}");
        assert_eq!(s.text, "hs_can");
        assert_eq!(s.stops, vec![stop(1, 0..6, &["hs_can", "ms_can", "kline"]), stop(u16::MAX, 6..6, &[])]);
    }

    #[test]
    fn escapes_are_literal_and_do_not_start_placeholders() {
        let s = parse(r"\${1\} and \\");
        assert_eq!(s.text, r"${1} and \");
        assert_eq!(s.stops, vec![stop(u16::MAX, 10..10, &[])]);
    }

    #[test]
    fn stops_sort_by_index_regardless_of_textual_order() {
        let s = parse("${2:b}${1:a}");
        assert_eq!(s.text, "ba");
        // index 1 (range 1..2, the second textually) visits before index 2.
        assert_eq!(
            s.stops,
            vec![stop(1, 1..2, &[]), stop(2, 0..1, &[]), stop(u16::MAX, 2..2, &[])]
        );
    }

    #[test]
    fn duplicate_index_and_nesting_and_malformed_are_errors() {
        assert_eq!(Snippet::parse("$1 $1"), Err(SnippetError::DuplicateIndex(1)));
        assert_eq!(Snippet::parse("$0$0"), Err(SnippetError::DuplicateIndex(0)));
        assert_eq!(Snippet::parse("${1:${2:x}}"), Err(SnippetError::Nesting));
        assert!(matches!(Snippet::parse("${x}"), Err(SnippetError::Malformed(_))));
        assert!(matches!(Snippet::parse("${1:oops"), Err(SnippetError::Malformed(_))));
    }

    #[test]
    fn for_insertion_is_identity_on_a_flat_single_line() {
        let s = parse("${1:name}").for_insertion("    ", 4);
        assert_eq!(s.text, "name");
        assert_eq!(s.stops, vec![stop(1, 0..4, &[]), stop(u16::MAX, 4..4, &[])]);
    }

    #[test]
    fn for_insertion_reindents_continuation_lines_and_expands_tabs() {
        // A two-level body: line 1 flush, line 2 one tab in, line 3 flush.
        let s = parse("fn ${1:id} {\n\t$0\n}").for_insertion("    ", 4);
        // \n → \n + "    " (insertion indent); the body \t → 4 spaces.
        assert_eq!(s.text, "fn id {\n        \n    }");
        // stop 1 = "id" at 3..5; final = the empty $0 on line 2.
        assert_eq!(s.stops[0], stop(1, 3..5, &[]));
        assert!(s.stops[1].is_final());
        // $0 sits after "fn id {\n" (8) + "    " indent (4) + "\t"→4 = 16.
        assert_eq!(s.stops[1].range, 16..16);
    }

    #[test]
    fn for_insertion_remaps_stops_across_the_rewrite() {
        let s = parse("${1:a}\n${2:b}").for_insertion("  ", 4);
        assert_eq!(s.text, "a\n  b");
        assert_eq!(
            s.stops,
            vec![stop(1, 0..1, &[]), stop(2, 4..5, &[]), stop(u16::MAX, 5..5, &[])]
        );
    }

    #[test]
    fn for_insertion_normalizes_crlf() {
        let s = parse("a\r\nb").for_insertion("", 4);
        assert_eq!(s.text, "a\nb");
    }

    #[test]
    fn a_bare_dollar_not_starting_a_placeholder_is_literal() {
        // `$` followed by a non-digit / non-`{` is literal (a digit would make it
        // a placeholder — `$5` is tab stop 5).
        let s = parse("a $ b and c$d");
        assert_eq!(s.text, "a $ b and c$d");
        assert_eq!(s.stops, vec![stop(u16::MAX, 13..13, &[])]);
    }

    // --- session state machine ---

    /// The SnippetStop ranges + stickiness in the store, sorted by start.
    fn live_stops(store: &DecorationStore) -> Vec<(Range<u32>, Stickiness)> {
        let mut v: Vec<_> = store
            .iter()
            .filter(|r| matches!(r.kind, DecorationKind::SnippetStop { .. }))
            .map(|r| (r.range.clone(), r.stickiness))
            .collect();
        v.sort_by_key(|(r, _)| r.start);
        v
    }

    #[test]
    fn session_needs_a_stop_besides_the_final() {
        let mut store = DecorationStore::new();
        // Only a final stop → no session.
        assert!(SnippetSession::start(&parse("$0"), 0, &mut store).is_none());
        assert!(SnippetSession::start(&parse("plain"), 0, &mut store).is_none());
        // One real stop → a session, first content selected.
        let (_s, range) = SnippetSession::start(&parse("${1:x}"), 0, &mut store).expect("session");
        assert_eq!(range, 0..1);
    }

    #[test]
    fn start_registers_stops_with_only_the_first_active() {
        let mut store = DecorationStore::new();
        let (_s, first) = SnippetSession::start(&parse("${1:a}${2:b}"), 0, &mut store).expect("session");
        assert_eq!(first, 0..1);
        assert_eq!(
            live_stops(&store),
            vec![
                (0..1, Stickiness::AlwaysGrows), // stop 1, active
                (1..2, Stickiness::NeverGrows),  // stop 2
                (2..2, Stickiness::NeverGrows),  // final
            ]
        );
    }

    #[test]
    fn tab_moves_active_swaps_stickiness_then_finishes_at_the_final() {
        let mut store = DecorationStore::new();
        let (mut s, _) = SnippetSession::start(&parse("${1:a}${2:b}"), 0, &mut store).expect("session");
        // Tab → stop 2 active; AlwaysGrows moves with it.
        assert_eq!(s.tab(true, &mut store), TabOutcome::Move(1..2));
        assert_eq!(
            live_stops(&store),
            vec![(0..1, Stickiness::NeverGrows), (1..2, Stickiness::AlwaysGrows), (2..2, Stickiness::NeverGrows)]
        );
        // Tab again → the next stop is the final → finish; ranges unregistered.
        assert_eq!(s.tab(true, &mut store), TabOutcome::Finish(2));
        assert!(live_stops(&store).is_empty());
    }

    #[test]
    fn shift_tab_at_the_first_stop_is_a_no_op() {
        let mut store = DecorationStore::new();
        let (mut s, _) = SnippetSession::start(&parse("${1:a}${2:b}"), 0, &mut store).expect("session");
        assert_eq!(s.tab(false, &mut store), TabOutcome::Stay);
        assert_eq!(s.active_index(), 0);
        // Forward then back returns to the first.
        assert_eq!(s.tab(true, &mut store), TabOutcome::Move(1..2));
        assert_eq!(s.tab(false, &mut store), TabOutcome::Move(0..1));
        assert_eq!(s.active_index(), 0);
    }

    #[test]
    fn on_caret_reactivates_inside_a_stop_and_escapes_outside() {
        let mut store = DecorationStore::new();
        let (mut s, _) = SnippetSession::start(&parse("${1:aa} ${2:bb}"), 0, &mut store).expect("session");
        // stops: 1 @ 0..2, 2 @ 3..5, final @ 5..5. Click into stop 2.
        assert_eq!(s.on_caret(4, &mut store), CaretOutcome::Move(3..5));
        assert_eq!(s.active_index(), 1);
        // Click back into stop 1.
        assert_eq!(s.on_caret(0, &mut store), CaretOutcome::Move(0..2));
        // Click within the active stop is a stay; far outside escapes.
        assert_eq!(s.on_caret(1, &mut store), CaretOutcome::Stay);
        assert_eq!(s.on_caret(99, &mut store), CaretOutcome::Escaped);
    }

    #[test]
    fn edit_escapes_only_when_wholly_outside_every_stop() {
        let mut store = DecorationStore::new();
        let (s, _) = SnippetSession::start(&parse("${1:aa} ${2:bb}"), 0, &mut store).expect("session");
        assert!(!s.edit_escapes(&(1..2), &store), "inside stop 1");
        assert!(!s.edit_escapes(&(4..4), &store), "inside stop 2");
        assert!(s.edit_escapes(&(10..12), &store), "past every stop");
    }

    #[test]
    fn cancel_unregisters_all_ranges() {
        let mut store = DecorationStore::new();
        let (mut s, _) = SnippetSession::start(&parse("${1:a}${2:b}"), 0, &mut store).expect("session");
        assert_eq!(live_stops(&store).len(), 3);
        s.cancel(&mut store);
        assert!(live_stops(&store).is_empty());
    }
}
