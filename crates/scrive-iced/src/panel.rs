//! The floating hover panels' shared text model.
//!
//! Two panels float over the code: the **hover popup** (the app's markdown,
//! [`HoverInfo`]) and the **fold preview** (a collapsed block's hidden rows,
//! read from the document). They behave as one kind of panel, the way a VS Code
//! hover does (`editor.hover.sticky`): the pointer can move into it, crossing a
//! gap within the hiding delay; its text selects by drag, double-click (word),
//! and triple-click (line); Ctrl+C copies the selection; a click inside never
//! reaches the code beneath and pins the panel open; and content taller than
//! the panel scrolls.
//!
//! The widget owns placement and painting (it needs the document, the fold
//! map, and the renderer); this module is the pure part both panels share —
//! panel identity, positions in a panel's text, the selection, and the hover
//! markdown's wrap model, which a copy reads back through so a soft wrap never
//! leaks into the copied text.

use std::hash::{DefaultHasher, Hash, Hasher};
use std::ops::Range;
use std::time::Instant;

use iced::advanced::mouse;
use scrive_core::HoverInfo;

use crate::popup;

/// Which panel is open. Its view state (scroll, selection, pin, hiding delay)
/// belongs to that one panel and resets when this changes — a different word's
/// hover, the same word's hover with new content (async docs replacing a
/// diagnostics-only card), or a fold preview.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum PanelId {
    /// The hover popup over `start..end`, with a hash of its markdown.
    Hover { start: u32, end: u32, content: u64 },
    /// The fold preview of the collapsed pair opened at this offset.
    Fold(u32),
}

impl PanelId {
    /// The identity of the hover popup showing `info`.
    pub(crate) fn hover(info: &HoverInfo) -> Self {
        let mut h = DefaultHasher::new();
        info.markdown.hash(&mut h);
        Self::Hover { start: info.range.start, end: info.range.end, content: h.finish() }
    }
}

/// A caret position in a panel's text: content line and char column
/// (monospace ⇒ one char per cell). Ordered line-major.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
pub(crate) struct TextPos {
    pub(crate) line: usize,
    pub(crate) col: usize,
}

impl TextPos {
    pub(crate) fn new(line: usize, col: usize) -> Self {
        Self { line, col }
    }
}

/// A selection in a panel's text, anchored where the gesture began.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct PanelSelection {
    pub(crate) anchor: TextPos,
    pub(crate) head: TextPos,
}

impl PanelSelection {
    /// The selection between two positions, in either order.
    pub(crate) fn new(anchor: TextPos, head: TextPos) -> Self {
        Self { anchor, head }
    }

    /// Its ends, start first.
    pub(crate) fn ordered(&self) -> (TextPos, TextPos) {
        (self.anchor.min(self.head), self.anchor.max(self.head))
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.anchor == self.head
    }

    /// The columns of content line `line` (`len` chars long) it covers, and
    /// whether it runs on past the line's end (a selected line break, washed a
    /// half cell further, as the editor washes one). `None` if it misses the
    /// line or covers nothing on it.
    pub(crate) fn span_on(&self, line: usize, len: usize) -> Option<(usize, usize, bool)> {
        let (a, b) = self.ordered();
        if line < a.line || line > b.line {
            return None;
        }
        let from = if line == a.line { a.col.min(len) } else { 0 };
        let (to, eol) = if line == b.line { (b.col.min(len), false) } else { (len, true) };
        (from < to || eol).then_some((from, to, eol))
    }
}

/// The open panel's view state in the widget tree. Everything here belongs to
/// the panel `id` names and is reset (a fresh value) when that changes, so a
/// new panel never inherits the last one's scroll or selection.
#[derive(Debug, Default)]
pub(crate) struct PanelState {
    /// The panel this state belongs to; `None` while none is open.
    pub(crate) id: Option<PanelId>,
    /// First visible content line, for content taller than the panel.
    pub(crate) scroll: usize,
    /// The text selection, if the panel has been clicked into.
    pub(crate) sel: Option<PanelSelection>,
    /// A press inside the panel is being dragged — its moves extend `sel`.
    pub(crate) drag: bool,
    /// The previous press inside the panel, for double/triple-click (kept
    /// apart from the code area's, so a click there never counts here).
    pub(crate) click: Option<mouse::Click>,
    /// Clicked into: the panel then ignores where the pointer goes and stays
    /// until a click elsewhere, a key, or a scroll closes it — as a VS Code hover
    /// does once it has focus — so a selection survives the trip to Ctrl+C.
    pub(crate) pinned: bool,
    /// The hiding delay: armed when the pointer leaves both the panel and its
    /// subject, stamped with an accurate `now` on the next frame (like the
    /// hover idle timer); the panel closes when it elapses unless the pointer
    /// came back.
    pub(crate) hide_arm: bool,
    pub(crate) hide_at: Option<Instant>,
}

impl PanelState {
    /// Fresh state for the panel `id` (or none).
    pub(crate) fn new(id: Option<PanelId>) -> Self {
        Self { id, ..Self::default() }
    }

    /// Cancel a pending hide — the pointer is back over the panel or its subject.
    pub(crate) fn keep(&mut self) {
        self.hide_arm = false;
        self.hide_at = None;
    }
}

/// One line of a fold preview: which bytes of which buffer row it shows, and
/// where they start.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct PreviewLine {
    /// The buffer row.
    pub(crate) row: u32,
    /// The byte columns of the row it shows.
    pub(crate) cols: Range<u32>,
    /// Display cells of indent the line is shown at ahead of those bytes —
    /// the opener line's, which sets its `{` above the closing `}`. Zero for
    /// a line that starts at its row's column 0 (it carries its own indent).
    pub(crate) lead: u32,
}

/// What a fold preview shows: the folded pair as one fragment — the opener's
/// row from the opener on, the rows between whole, and the closer's row up to
/// and including the closer (never what follows it, such as ` else {` or
/// `;`) — with the opener line set at the closer row's indent so the two
/// delimiters line up. A pair on one row shows that whole row. Line `i` is
/// always buffer row `header + i`, so any line is O(1) to find: a
/// document-scale fold reads only the lines it shows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct FoldFragment {
    header: u32,
    open_col: u32,
    last: u32,
    close_col: u32,
    close_indent: u32,
}

impl FoldFragment {
    /// The fragment of the pair opened at `(header, open_col)` and closed at
    /// `(last, close_col)` (rows and byte columns), whose closing row is
    /// indented `close_indent` display cells.
    pub(crate) fn new((header, open_col): (u32, u32), (last, close_col): (u32, u32), close_indent: u32) -> Self {
        Self { header, open_col, last, close_col, close_indent }
    }

    /// How many lines it shows.
    pub(crate) fn len(&self) -> usize {
        (self.last - self.header) as usize + 1
    }

    /// Line `i`'s buffer row.
    pub(crate) fn row(&self, i: usize) -> u32 {
        self.header + i as u32
    }

    /// Line `i`, given the length in bytes of its row.
    pub(crate) fn line(&self, i: usize, row_len: u32) -> PreviewLine {
        let row = self.row(i);
        let (cols, lead) = if self.header == self.last {
            (0..row_len, 0) // a one-row pair: the whole row, as it reads
        } else if row == self.header {
            (self.open_col..row_len, self.close_indent)
        } else if row == self.last {
            // Delimiters are single ASCII bytes, so the closer ends one past it.
            (0..self.close_col + 1, 0)
        } else {
            (0..row_len, 0)
        };
        PreviewLine { row, cols, lead }
    }
}

/// The run of like chars around char `col` of `text`, for double-click
/// selection: a word (alphanumeric or `_`), a whitespace run, or a single
/// other char. `col` past the end clamps to the last char; an empty line
/// yields `(0, 0)`.
pub(crate) fn word_span(text: &str, col: usize) -> (usize, usize) {
    let chars: Vec<char> = text.chars().collect();
    let Some(last) = chars.len().checked_sub(1) else { return (0, 0) };
    let at = col.min(last);
    let class = |c: char| {
        if c.is_alphanumeric() || c == '_' {
            0
        } else if c.is_whitespace() {
            1
        } else {
            2
        }
    };
    let k = class(chars[at]);
    if k == 2 {
        return (at, at + 1);
    }
    let start = chars[..at].iter().rposition(|&c| class(c) != k).map_or(0, |i| i + 1);
    let end = chars[at..].iter().position(|&c| class(c) != k).map_or(chars.len(), |i| at + i);
    (start, end)
}

/// The chars `from..to` of `s` (char indices, clamped to its length).
pub(crate) fn char_slice(s: &str, from: usize, to: usize) -> String {
    s.chars().skip(from).take(to.saturating_sub(from)).collect()
}

/// Inline markdown styling for a hover run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum MdStyle {
    Plain,
    Bold,
    Code,
}

/// The hover popup's markdown as the panel shows it: each source line parsed
/// into styled runs and word-wrapped at the box width. Each visual line also
/// records where it starts in its source line's plain text, so a copy reads
/// the source back — rejoining soft wraps and restoring the spaces a wrap
/// dropped — and never the wrap.
pub(crate) struct HoverText {
    /// The visual lines (styled runs), wrapped at `cells`. Never empty.
    pub(crate) lines: Vec<Vec<(String, MdStyle)>>,
    /// Per visual line: its source line, and the char where it starts in that
    /// source line's plain text.
    origin: Vec<(usize, usize)>,
    /// Each source line's plain text (markers consumed).
    source: Vec<String>,
    /// The wrap width in cells: the widest source line, clamped to the popup's
    /// width bounds.
    pub(crate) cells: usize,
}

impl HoverText {
    /// Parse and wrap `markdown`.
    pub(crate) fn new(markdown: &str) -> Self {
        let mut md_lines: Vec<Vec<(String, MdStyle)>> = markdown.lines().map(parse_md_runs).collect();
        if md_lines.is_empty() {
            md_lines.push(Vec::new()); // an empty hover still has one (blank) line
        }
        let plain = |runs: &[(String, MdStyle)]| runs.iter().map(|(t, _)| t.as_str()).collect::<String>();
        let source: Vec<String> = md_lines.iter().map(|r| plain(r)).collect();
        let cells = source
            .iter()
            .map(|s| s.chars().count())
            .max()
            .unwrap_or(0)
            .clamp(popup::POPUP_MIN_CH, popup::POPUP_MAX_CH);
        let (mut lines, mut origin) = (Vec::new(), Vec::new());
        for (src, runs) in md_lines.iter().enumerate() {
            for (start, line) in wrap_runs(runs, cells) {
                lines.push(line);
                origin.push((src, start));
            }
        }
        Self { lines, origin, source, cells }
    }

    /// Visual line `i`'s length in chars.
    pub(crate) fn line_len(&self, i: usize) -> usize {
        self.lines[i].iter().map(|(t, _)| t.chars().count()).sum()
    }

    /// Visual line `i`'s plain text.
    pub(crate) fn line_text(&self, i: usize) -> String {
        self.lines[i].iter().map(|(t, _)| t.as_str()).collect()
    }

    /// The visual lines `first..=last` that source line of visual line `i`
    /// wraps into — what a triple-click selects (a paragraph, as in a page).
    pub(crate) fn paragraph(&self, i: usize) -> (usize, usize) {
        let src = self.origin[i].0;
        let first = self.origin.iter().position(|o| o.0 == src).unwrap_or(i);
        let last = self.origin.iter().rposition(|o| o.0 == src).unwrap_or(i);
        (first, last)
    }

    /// The plain text between two positions: the source text they map back
    /// to, source lines joined by `\n`. Positions are clamped to the text.
    pub(crate) fn copy(&self, a: TextPos, b: TextPos) -> String {
        let at = |p: TextPos| {
            let line = p.line.min(self.origin.len() - 1);
            let (src, start) = self.origin[line];
            (src, start + p.col.min(self.line_len(line)))
        };
        let ((sa, oa), (sb, ob)) = (at(a.min(b)), at(a.max(b)));
        (sa..=sb)
            .map(|s| {
                let from = if s == sa { oa } else { 0 };
                let to = if s == sb { ob } else { usize::MAX };
                char_slice(&self.source[s], from, to)
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

/// Split one markdown line into styled runs, consuming the `**bold**` and
/// `` `code` `` markers. A minimal inline subset (no nesting) — enough for the
/// hover's spec-derived docs; unmatched markers just toggle back at line end.
fn parse_md_runs(line: &str) -> Vec<(String, MdStyle)> {
    let mut runs = Vec::new();
    let mut cur = String::new();
    let mut style = MdStyle::Plain;
    let mut push = |cur: &mut String, style: MdStyle| {
        if !cur.is_empty() {
            runs.push((std::mem::take(cur), style));
        }
    };
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        match c {
            '*' if chars.peek() == Some(&'*') => {
                chars.next(); // second '*'
                push(&mut cur, style);
                style = if style == MdStyle::Bold { MdStyle::Plain } else { MdStyle::Bold };
            }
            '`' => {
                push(&mut cur, style);
                style = if style == MdStyle::Code { MdStyle::Plain } else { MdStyle::Code };
            }
            _ => cur.push(c),
        }
    }
    if !cur.is_empty() {
        runs.push((cur, style));
    }
    runs
}

/// Coalesce a per-char styled sequence into `(String, style)` segments.
fn coalesce_chars(chars: Vec<(char, MdStyle)>) -> Vec<(String, MdStyle)> {
    let mut segs: Vec<(String, MdStyle)> = Vec::new();
    for (c, s) in chars {
        match segs.last_mut() {
            Some((t, st)) if *st == s => t.push(c),
            _ => segs.push((c.to_string(), s)),
        }
    }
    segs
}

/// Word-wrap one source line's styled runs into visual lines that each fit in
/// `max_cells` (monospace ⇒ 1 cell = 1 char). Breaks at spaces (dropped at the
/// break); a word longer than the width hard-breaks. Per-char style rides along.
/// Each visual line comes with the char index where it starts in the source
/// line — a visual line is always a contiguous slice of it, since only the
/// spaces at a break are dropped.
fn wrap_runs(runs: &[(String, MdStyle)], max_cells: usize) -> Vec<(usize, Vec<(String, MdStyle)>)> {
    let max = max_cells.max(1);
    let flat: Vec<(char, MdStyle)> =
        runs.iter().flat_map(|(t, s)| t.chars().map(move |c| (c, *s))).collect();
    let mut lines: Vec<(usize, Vec<(char, MdStyle)>)> = Vec::new();
    let mut cur: Vec<(char, MdStyle)> = Vec::new();
    let mut cur_start = 0;
    let trim = |line: &mut Vec<(char, MdStyle)>| {
        while line.last().map(|&(c, _)| c) == Some(' ') {
            line.pop();
        }
    };
    let mut i = 0;
    while i < flat.len() {
        if flat[i].0 == ' ' {
            if cur.len() < max {
                if cur.is_empty() {
                    cur_start = i;
                }
                cur.push(flat[i]); // spaces only count while the line has room
            }
            i += 1;
            continue;
        }
        let start = i;
        while i < flat.len() && flat[i].0 != ' ' {
            i += 1;
        }
        let word = &flat[start..i];
        if word.len() > max {
            // Hard-break an over-long word across lines.
            for (k, &ch) in word.iter().enumerate() {
                if cur.len() >= max {
                    lines.push((cur_start, std::mem::take(&mut cur)));
                }
                if cur.is_empty() {
                    cur_start = start + k;
                }
                cur.push(ch);
            }
        } else {
            if cur.len() + word.len() > max {
                trim(&mut cur);
                lines.push((cur_start, std::mem::take(&mut cur)));
            }
            if cur.is_empty() {
                cur_start = start;
            }
            cur.extend_from_slice(word);
        }
    }
    trim(&mut cur);
    lines.push((cur_start, cur));
    lines.into_iter().map(|(s, l)| (s, coalesce_chars(l))).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn text(line: &[(String, MdStyle)]) -> String {
        line.iter().map(|(t, _)| t.as_str()).collect()
    }

    #[test]
    fn wrap_runs_wraps_at_word_boundaries() {
        use MdStyle::*;
        let texts = |w: Vec<(usize, Vec<(String, MdStyle)>)>| w.iter().map(|(_, l)| text(l)).collect::<Vec<_>>();
        // Greedy word wrap at width 8.
        assert_eq!(texts(wrap_runs(&[("hello world foo".to_string(), Plain)], 8)), vec!["hello", "world", "foo"]);
        // A word longer than the width hard-breaks.
        assert_eq!(texts(wrap_runs(&[("verylongword".to_string(), Plain)], 4)), vec!["very", "long", "word"]);
        // Short content stays one line and keeps its styles.
        let s = wrap_runs(&[("a".to_string(), Bold), (" b".to_string(), Plain)], 40);
        assert_eq!(s.len(), 1);
        assert_eq!(s[0].1, vec![("a".to_string(), Bold), (" b".to_string(), Plain)]);
    }

    /// Every visual line is the slice of its source line its recorded start
    /// names — the invariant `HoverText::copy` reads the source back through.
    #[test]
    fn wrap_runs_starts_index_the_source() {
        use MdStyle::*;
        for (src, width) in [
            ("hello world foo", 8),
            ("verylongword", 4),
            ("  lead   gaps  between words  ", 7),
            ("ab cdefghijklmnop q", 5),
            ("", 3),
        ] {
            for (start, line) in wrap_runs(&[(src.to_string(), Plain)], width) {
                let t = text(&line);
                assert_eq!(char_slice(src, start, start + t.chars().count()), t, "{src:?} at width {width}");
            }
        }
    }

    #[test]
    fn parse_md_runs_splits_bold_and_code() {
        use MdStyle::*;
        let runs = parse_md_runs("**name** `id { … }` — handle it.");
        assert_eq!(
            runs,
            vec![
                ("name".to_string(), Bold),
                (" ".to_string(), Plain),
                ("id { … }".to_string(), Code),
                (" — handle it.".to_string(), Plain),
            ]
        );
        // Plain text with no markers is a single Plain run.
        assert_eq!(parse_md_runs("just words"), vec![("just words".to_string(), Plain)]);
        // A single '*' is literal (only '**' toggles bold).
        assert_eq!(parse_md_runs("a * b"), vec![("a * b".to_string(), Plain)]);
    }

    /// A copy reads the source back: markers stripped, a soft wrap rejoined
    /// with the space it dropped, source lines joined by `\n`.
    #[test]
    fn hover_copy_reads_the_source_not_the_wrap() {
        // The second source line is 63 chars, so it wraps at the popup's 60-cell
        // cap — after "tail", dropping the space before "end".
        let long = "0123456789 0123456789 0123456789 0123456789 0123456789 tail end";
        let t = HoverText::new(&format!("**the** quick `brown` fox\n{long}"));
        assert_eq!(t.cells, 60);
        let lines: Vec<String> = (0..t.lines.len()).map(|i| t.line_text(i)).collect();
        assert_eq!(lines, vec!["the quick brown fox", &long[..59], "end"]);
        // Across the soft wrap: the dropped space comes back, no line break.
        assert_eq!(t.copy(TextPos::new(1, 55), TextPos::new(2, 3)), "tail end");
        // Across a source line: one `\n`, markers gone. Order-insensitive.
        assert_eq!(t.copy(TextPos::new(1, 5), TextPos::new(0, 10)), "brown fox\n01234");
        // The whole text.
        assert_eq!(t.copy(TextPos::new(0, 0), TextPos::new(2, 3)), format!("the quick brown fox\n{long}"));
        // A triple-click's paragraph spans the wrapped visual lines.
        assert_eq!(t.paragraph(2), (1, 2));
        assert_eq!(t.paragraph(0), (0, 0));
    }

    #[test]
    fn hover_text_of_empty_markdown_has_one_blank_line() {
        let t = HoverText::new("");
        assert_eq!(t.lines.len(), 1);
        assert_eq!(t.line_len(0), 0);
        assert_eq!(t.copy(TextPos::new(0, 0), TextPos::new(5, 9)), "");
    }

    /// The rows and byte ranges a fold preview shows, for a pair folded at
    /// `open` and closed at `close` (byte offsets into `text`).
    fn fragment(text: &str, open: usize, close: usize) -> Vec<(u32, String, u32)> {
        let rows: Vec<&str> = text.split('\n').collect();
        let at = |off: usize| {
            let row = text[..off].matches('\n').count();
            (row as u32, (off - text[..off].rfind('\n').map_or(0, |i| i + 1)) as u32)
        };
        let close_row = rows[at(close).0 as usize];
        let indent = (close_row.len() - close_row.trim_start().len()) as u32;
        let f = FoldFragment::new(at(open), at(close), indent);
        (0..f.len())
            .map(|i| {
                let row = rows[f.row(i) as usize];
                let l = f.line(i, row.len() as u32);
                (l.row, row[l.cols.start as usize..l.cols.end as usize].to_string(), l.lead)
            })
            .collect()
    }

    /// The three reported shapes: the opener line leads (from the opener on, at the
    /// closer's indent), the closer line ends at the closer, every bracket
    /// kind alike — and a one-row pair keeps its whole row.
    #[test]
    fn fold_fragment_runs_opener_to_closer() {
        let s = |v: &[(u32, &str, u32)]| v.iter().map(|&(r, t, l)| (r, t.to_string(), l)).collect::<Vec<_>>();
        let t = "fn foo() {\n    my_code_here\n}\n";
        assert_eq!(fragment(t, 9, t.rfind('}').unwrap()), s(&[(0, "{", 0), (1, "    my_code_here", 0), (2, "}", 0)]));
        let t = "if a {\n    x\n} else {\n    y\n}\n";
        assert_eq!(fragment(t, 5, t.find('}').unwrap()), s(&[(0, "{", 0), (1, "    x", 0), (2, "}", 0)]));
        let t = "let xs = [\n    1,\n    2,\n];\n";
        assert_eq!(fragment(t, 9, t.find(']').unwrap()), s(&[(0, "[", 0), (1, "    1,", 0), (2, "    2,", 0), (3, "]", 0)]));
        // Nested: the opener line takes the closer row's indent.
        let t = "mod m {\n    fn f() { // body\n        x\n    }\n}\n";
        assert_eq!(
            fragment(t, t.find("{ //").unwrap(), t.find("    }").unwrap() + 4),
            s(&[(1, "{ // body", 4), (2, "        x", 0), (3, "    }", 0)])
        );
        // One row: the whole row, unchanged.
        let t = "let v = vec![1, 2, 3];\n";
        assert_eq!(fragment(t, 12, 20), s(&[(0, "let v = vec![1, 2, 3];", 0)]));
    }

    #[test]
    fn word_span_picks_the_run_under_the_char() {
        let s = "let x_1 = foo(bar);";
        assert_eq!(word_span(s, 0), (0, 3)); // "let"
        assert_eq!(word_span(s, 5), (4, 7)); // "x_1"
        assert_eq!(word_span(s, 3), (3, 4)); // a space run
        assert_eq!(word_span(s, 13), (13, 14)); // "(" alone
        assert_eq!(word_span(s, 99), (18, 19)); // past the end → the last char
        assert_eq!(word_span("", 4), (0, 0));
    }

    #[test]
    fn selection_span_on_each_line() {
        let s = PanelSelection::new(TextPos::new(3, 2), TextPos::new(1, 4));
        assert_eq!(s.span_on(0, 10), None);
        assert_eq!(s.span_on(1, 10), Some((4, 10, true))); // first line: to its end, + break
        assert_eq!(s.span_on(2, 0), Some((0, 0, true))); // a blank middle line: just the break
        assert_eq!(s.span_on(3, 10), Some((0, 2, false))); // last line: up to the head
        assert_eq!(s.span_on(4, 10), None);
        // An empty selection covers nothing.
        assert_eq!(PanelSelection::new(TextPos::new(1, 4), TextPos::new(1, 4)).span_on(1, 10), None);
    }
}
