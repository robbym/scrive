//! The text rope — a `SumTree<Chunk>`. Backs
//! [`crate::buffer::Buffer`]; the LF-only line model lives in the chunk
//! [`TextSummary`] (a `\n` count + trailing-line byte count), so line/point math
//! is a summary fold, not a second index. All reads are O(log n) and the clone is
//! O(1) (an `Arc` bump) — the structural-sharing snapshot the highlight sweep
//! rides.
//!
//! Byte offsets are `str`-style: every public offset is (or is clamped to) a char
//! boundary, and chunk boundaries are char boundaries by construction.

use std::borrow::Cow;
use std::ops::Range;

use arrayvec::ArrayString;

use crate::coords::Point;
use crate::sum_tree::{Dimension, Item, SumTree, Summary};

/// Target bytes per leaf chunk; `from_str` packs to this, and every edit
/// re-chunks its seam to it ([`Rope::replace`]).
const CHUNK_MAX: usize = 128;

/// The fill floor the chunking maintains **on average**: a rope of `n` bytes
/// holds at most `n / CHUNK_MIN` chunks, however it was edited.
///
/// This is not a per-chunk minimum — re-chunking a seam always leaves one
/// remainder, which can be a single byte — it is the anti-fragmentation
/// invariant, and the reason [`Rope::replace`] coalesces instead of splicing
/// naively. Without it, chunk count grows with *keystrokes* rather than bytes:
/// a naive splice puts the typed text in its own chunk and the caret then sits
/// on that chunk's boundary, so every subsequent character adds another 1-byte
/// chunk. That deepens the tree and inflates every O(log n) read and every
/// summary fold — pinned by `chunk_count_stays_proportional_to_bytes_not_edits`.
const CHUNK_MIN: usize = CHUNK_MAX / 2;

/// One leaf chunk: **inline** bytes, never a heap allocation of its own.
///
/// A chunk is build-once and read-only — nothing ever grows one in place; the
/// edit paths rebuild a span's text and re-chunk it wholesale ([`Rope::replace`]
/// re-chunks its seam, [`rebuild_leaf`] a whole leaf).
/// So its capacity is fixed at [`CHUNK_MAX`] by construction, and paying a
/// `String`'s pointer indirection + allocation per chunk buys nothing: a 10 MB
/// document is ~78k chunks, i.e. ~78k allocations on load and up to one per
/// chunk again on every leaf rebuild.
///
/// Inline storage puts the bytes in the leaf's own array instead, so a leaf
/// clone — which copy-on-write does on *every* edit that touches it — is a flat
/// memcpy rather than a walk of scattered pointers.
///
/// [`ArrayString`] panics rather than growing past its capacity; that is a
/// feature here, since it makes the ≤ [`CHUNK_MAX`] invariant load-bearing
/// instead of implicit. Every construction site goes through a checked
/// `from`/`expect` that names the invariant it relies on.
#[derive(Clone, Debug)]
struct Chunk(ArrayString<CHUNK_MAX>);

/// Build a chunk from `s`, which must already be `<= CHUNK_MAX` bytes — the ONE
/// place the capacity invariant is enforced, so a violation names itself instead
/// of surfacing as a truncation.
fn chunk_of(s: &str) -> Chunk {
    Chunk(ArrayString::from(s).expect("chunks are split to <= CHUNK_MAX by construction"))
}

/// The monoid behind every line/point query: total bytes, `\n` count, and bytes
/// since the last `\n` (all bytes when the run has none) — enough to fold a
/// `(row, col)` [`Point`] across a concatenation, associatively.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct TextSummary {
    bytes: u32,
    newlines: u32,
    last_line_bytes: u32,
}

impl Summary for TextSummary {
    fn add_summary(&mut self, o: &Self) {
        self.last_line_bytes =
            if o.newlines > 0 { o.last_line_bytes } else { self.last_line_bytes + o.last_line_bytes };
        self.bytes += o.bytes;
        self.newlines += o.newlines;
    }
}

impl Item for Chunk {
    type Summary = TextSummary;
    fn summary(&self) -> TextSummary {
        summarize(&self.0)
    }
}

fn summarize(s: &str) -> TextSummary {
    let newlines = s.bytes().filter(|&b| b == b'\n').count() as u32;
    let last_line_bytes = match s.rfind('\n') {
        Some(i) => (s.len() - i - 1) as u32,
        None => s.len() as u32,
    };
    TextSummary { bytes: s.len() as u32, newlines, last_line_bytes }
}

/// Seek by byte offset.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, PartialOrd, Ord)]
struct ByteDim(u32);
impl Dimension<TextSummary> for ByteDim {
    fn add_summary(&mut self, s: &TextSummary) {
        self.0 += s.bytes;
    }
}

/// Seek by `(row, col)` — the LF-only Point fold.
impl Dimension<TextSummary> for Point {
    fn add_summary(&mut self, s: &TextSummary) {
        if s.newlines > 0 {
            self.row += s.newlines;
            self.col = s.last_line_bytes;
        } else {
            self.col += s.last_line_bytes;
        }
    }
}

/// A rope of UTF-8 text. LF-only line model; `Clone` is O(1).
#[derive(Clone, Debug, Default)]
pub struct Rope(SumTree<Chunk>);

impl Rope {
    /// Build from `s` (already LF-only), packed into `CHUNK_MAX` chunks.
    #[must_use]
    pub fn from_str(s: &str) -> Rope {
        Rope(SumTree::from_items(chunk_str(s)))
    }

    /// Total byte length.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.0.summary().bytes
    }

    /// Number of lines = `\n` count + 1 (always ≥ 1).
    #[must_use]
    pub fn line_count(&self) -> u32 {
        self.0.summary().newlines + 1
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The text in byte range `range` (clamped to the end). `Borrowed` when it sits
    /// in one chunk, `Owned` across chunks.
    #[must_use]
    pub fn slice(&self, range: Range<u32>) -> Cow<'_, str> {
        let end = range.end.min(self.len());
        let start = range.start.min(end);
        if start == end {
            return Cow::Borrowed("");
        }
        if let Some((chunk, ByteDim(cs))) = self.0.item_at(&ByteDim(start)) {
            if end <= cs + chunk.0.len() as u32 {
                return Cow::Borrowed(&chunk.0[(start - cs) as usize..(end - cs) as usize]);
            }
        }
        let mut out = String::with_capacity((end - start) as usize);
        self.0.for_each_in_range(&ByteDim(start), &ByteDim(end), &mut |chunk: &Chunk, &ByteDim(cs)| {
            let a = (start.max(cs) - cs) as usize;
            let b = (end.min(cs + chunk.0.len() as u32) - cs) as usize;
            out.push_str(&chunk.0[a..b]);
        });
        Cow::Owned(out)
    }

    /// One row's text, excluding its trailing `\n`. Out-of-range → `""`.
    #[must_use]
    pub fn line(&self, row: u32) -> Cow<'_, str> {
        if row >= self.line_count() {
            return Cow::Borrowed("");
        }
        self.slice(self.line_to_byte(row)..self.line_end_byte(row))
    }

    /// Byte length of one row's text (excludes `\n`). Out-of-range → 0.
    #[must_use]
    pub fn line_len(&self, row: u32) -> u32 {
        if row >= self.line_count() {
            return 0;
        }
        self.line_end_byte(row) - self.line_to_byte(row)
    }

    /// Byte offset of `row`'s start (clamps a past-the-end row to the doc end).
    #[must_use]
    pub fn line_to_byte(&self, row: u32) -> u32 {
        if row == 0 {
            return 0;
        }
        if row >= self.line_count() {
            return self.len();
        }
        match self.0.seek::<Point, ByteDim>(&Point::new(row, 0)) {
            Some((chunk, point_start, ByteDim(byte_start))) => {
                byte_start + point_offset_in_str(&chunk.0, point_start, Point::new(row, 0))
            }
            None => 0,
        }
    }

    /// Byte offset just before `row`'s trailing `\n` (or the doc end on the last
    /// line).
    fn line_end_byte(&self, row: u32) -> u32 {
        if row + 1 < self.line_count() {
            self.line_to_byte(row + 1) - 1
        } else {
            self.len()
        }
    }

    /// Byte offset → `Point`, clamping past-the-end to the doc end. Byte-based (a
    /// col is a byte count), so a mid-char offset still maps sanely.
    #[must_use]
    pub fn byte_to_point(&self, offset: u32) -> Point {
        let offset = offset.min(self.len());
        match self.0.seek::<ByteDim, Point>(&ByteDim(offset)) {
            Some((chunk, ByteDim(cs), point_start)) => {
                let within = (offset - cs) as usize;
                let bytes = &chunk.0.as_bytes()[..within];
                let newlines = bytes.iter().filter(|&&b| b == b'\n').count() as u32;
                if newlines > 0 {
                    let last_nl = bytes.iter().rposition(|&b| b == b'\n').expect("newlines > 0");
                    Point::new(point_start.row + newlines, (within - last_nl - 1) as u32)
                } else {
                    Point::new(point_start.row, point_start.col + within as u32)
                }
            }
            None => Point::new(0, 0),
        }
    }

    /// `Point` → byte offset, clamping the row to the last line and the col to that
    /// row's byte length.
    #[must_use]
    pub fn point_to_offset(&self, point: Point) -> u32 {
        let row = point.row.min(self.line_count() - 1);
        self.line_to_byte(row) + point.col.min(self.line_len(row))
    }

    /// The chunk containing byte `offset` (the right-hand chunk at an exact
    /// boundary), and that chunk's start offset. Empty rope → `("", 0)`.
    #[must_use]
    pub fn chunk_at(&self, offset: u32) -> (&str, u32) {
        match self.0.item_at(&ByteDim(offset.min(self.len()))) {
            Some((chunk, ByteDim(cs))) => (&chunk.0, cs),
            None => ("", 0),
        }
    }

    /// Whether `offset` sits on a char boundary (`0` and `len` do).
    #[must_use]
    pub fn is_char_boundary(&self, offset: u32) -> bool {
        if offset == 0 || offset >= self.len() {
            return offset <= self.len();
        }
        let (chunk, cs) = self.chunk_at(offset);
        chunk.is_char_boundary((offset - cs) as usize)
    }

    /// Visit each chunk's text in order — the cold-path whole-document walk
    /// (serialize).
    pub fn for_each_chunk(&self, mut f: impl FnMut(&str)) {
        self.0.for_each_in_range(&ByteDim(0), &ByteDim(self.len()), &mut |chunk: &Chunk, _| f(&chunk.0));
    }

    /// The leaf chunks in order — the seam-enumeration handle for tests and the
    /// serialize walk. Borrows the rope; no copy.
    pub fn chunks(&self) -> impl Iterator<Item = &str> {
        self.0.item_refs().into_iter().map(|c| c.0.as_str())
    }

    /// Replace byte `range` with `text` (already LF-only), O(log n + |text|).
    ///
    /// **Coalescing.** Splicing naively — split at both ends, drop `text` in as
    /// its own chunk — is what fragments the rope: the two splits each leave an
    /// undersized remainder, `text` becomes a chunk however short it is, and the
    /// caret then sits on that chunk's boundary, so the next keystroke adds
    /// another 1-byte chunk beside it. Typing a word leaves one chunk per letter
    /// and the count grows with keystrokes, not bytes.
    ///
    /// So instead of splicing at the endpoints, this peels a **window** of
    /// [`CHUNK_MAX`] bytes either side of the edit, rebuilds that whole span as
    /// one string, and re-chunks it packed. The window is what makes it
    /// self-healing rather than merely tidy: a previous edit here can only have
    /// left a remainder of `< CHUNK_MAX` bytes, so the next edit's window is
    /// guaranteed to swallow and repack it. Fragments can never accumulate,
    /// which is the [`CHUNK_MIN`] invariant.
    ///
    /// Cost is unchanged in class: the peeled span is `<= 2 * CHUNK_MAX +
    /// text.len()`, i.e. O(|text|) extra copying on top of the same two splits
    /// and two appends.
    pub fn replace(&mut self, range: Range<u32>, text: &str) {
        let start = range.start.min(self.len());
        let end = range.end.min(self.len()).max(start);
        // The peel window, snapped OUTWARD to CHUNK boundaries — not merely to
        // char boundaries. This is load-bearing: splitting at an arbitrary offset
        // cuts a chunk in two and leaves an undersized half on the far side of
        // the window, which is exactly the fragment this method exists to
        // prevent. Snapping to a boundary means the splits fall BETWEEN chunks
        // and create nothing. (Chunk boundaries are char boundaries by
        // construction, so the str contract comes along for free.)
        let mut lo = self.chunk_at(start.saturating_sub(CHUNK_MAX as u32)).1;
        let probe = end.saturating_add(CHUNK_MAX as u32);
        let mut hi = if probe >= self.len() {
            self.len()
        } else {
            let (c, cs) = self.chunk_at(probe);
            cs + c.len() as u32
        };
        // Then keep going past any UNDERSIZED chunk on either side. This is the
        // part that makes coalescing converge rather than merely tidy up: the
        // seam is one byte longer than the window it replaced, so ITS remainder
        // lands just *outside* where the next keystroke's window would stop. A
        // window that halts at a fixed `± CHUNK_MAX` can never reach that
        // remainder, and one escapes per keystroke — the original bug, only
        // slower. Swallowing sub-`CHUNK_MIN` neighbours instead means every
        // fragment is repacked by the next edit that comes near it.
        //
        // Bounded in practice by the invariant itself: re-chunking leaves at most
        // one undersized chunk per seam, so these walks take a step or two. On a
        // rope that is already fragmented they take more and heal it — which is
        // the direction you want the cost to run.
        while lo > 0 {
            let (c, cs) = self.chunk_at(lo - 1);
            if c.is_empty() || c.len() >= CHUNK_MIN {
                break;
            }
            lo = cs;
        }
        while hi < self.len() {
            let (c, cs) = self.chunk_at(hi);
            if c.is_empty() || c.len() >= CHUNK_MIN {
                break;
            }
            hi = cs + c.len() as u32;
        }
        // Read the whole seam before rebuilding: everything outside `lo..hi` is
        // untouched and stays structurally shared.
        let head = self.split_byte(lo).0;
        let tail = self.split_byte(hi).1;
        let mut seam =
            String::with_capacity((hi - lo) as usize - (end - start) as usize + text.len());
        seam.push_str(&self.slice(lo..start));
        seam.push_str(text);
        seam.push_str(&self.slice(end..hi));
        let mid = SumTree::from_items(chunk_str(&seam));
        self.0 = head.append(&mid).append(&tail);
    }

    fn split_byte(&self, offset: u32) -> (SumTree<Chunk>, SumTree<Chunk>) {
        self.0.split_with(&ByteDim(offset), &mut |chunk: &Chunk, start: &ByteDim, at: &ByteDim| {
            let pos = (at.0 - start.0) as usize;
            // Both halves of a <= CHUNK_MAX chunk are <= CHUNK_MAX.
            (chunk_of(&chunk.0[..pos]), chunk_of(&chunk.0[pos..]))
        })
    }

    /// Apply MANY disjoint edits (`(byte range, replacement)`, sorted ascending by
    /// start, non-overlapping, in-bounds, all LF-only) in ONE pass — the batched
    /// twin of N [`Self::replace`]s. Only the edited leaves and the spine above them
    /// are rebuilt; every untouched subtree is shared. A rare edit straddling a leaf
    /// boundary falls back to sequential replace (applied descending). This is what
    /// turns document-scale multi-caret typing from O(carets · log) rope splices
    /// into O(carets + spine).
    pub fn edit_many(&mut self, edits: &[(Range<u32>, &str)]) {
        if edits.is_empty() {
            return;
        }
        let dim: Vec<(Range<ByteDim>, &str)> =
            edits.iter().map(|(r, t)| (ByteDim(r.start)..ByteDim(r.end), *t)).collect();
        match self.0.edit_many(&dim, &rebuild_leaf) {
            Some(tree) => self.0 = tree,
            // Straddling edit: fall back to sequential replace, descending so earlier
            // offsets stay valid (the same order the transaction engine used).
            None => {
                for (range, text) in edits.iter().rev() {
                    self.replace(range.clone(), text);
                }
            }
        }
    }
}

/// Rebuild one leaf's chunks with the edits falling inside it: reconstruct the
/// leaf's text, splice each edit (leaf-relative, descending so offsets stay valid),
/// re-chunk. Only edited leaves pay this; untouched leaves are shared whole.
fn rebuild_leaf(chunks: &[Chunk], base: &ByteDim, edits: &[(Range<ByteDim>, &str)]) -> Vec<Chunk> {
    let mut text = String::with_capacity(chunks.iter().map(|c| c.0.len()).sum());
    for c in chunks {
        text.push_str(&c.0);
    }
    for (range, ins) in edits.iter().rev() {
        let s = (range.start.0 - base.0) as usize;
        let e = (range.end.0 - base.0) as usize;
        text.replace_range(s..e, ins);
    }
    chunk_str(&text)
}

/// Split `s` into `<= CHUNK_MAX`-byte chunks at char boundaries.
fn chunk_str(s: &str) -> Vec<Chunk> {
    if s.is_empty() {
        return Vec::new();
    }
    let mut chunks = Vec::new();
    let mut start = 0;
    while start < s.len() {
        let mut end = (start + CHUNK_MAX).min(s.len());
        while end > start && !s.is_char_boundary(end) {
            end -= 1;
        }
        debug_assert!(end > start, "a single char cannot exceed CHUNK_MAX");
        chunks.push(chunk_of(&s[start..end])); // `end - start <= CHUNK_MAX` by the walk above
        start = end;
    }
    chunks
}

/// Byte offset within `s` at which `target` is reached, given `s` begins at
/// `start` in point space. `start <= target <= start + summary(s)`.
fn point_offset_in_str(s: &str, start: Point, target: Point) -> u32 {
    let mut p = start;
    for (i, ch) in s.char_indices() {
        if p >= target {
            return i as u32;
        }
        if ch == '\n' {
            p = Point::new(p.row + 1, 0);
        } else {
            p.col += ch.len_utf8() as u32;
        }
    }
    s.len() as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matches_ropey_across_reads() {
        let text = "fn main() {\n    let x = 42;\n    println!(\"hi\");\n}\n\nfin";
        let mine = Rope::from_str(text);
        let oracle = ropey::Rope::from_str(text);
        assert_eq!(mine.len() as usize, oracle.len_bytes());
        assert_eq!(mine.line_count() as usize, oracle.len_lines());
        assert_eq!(mine.slice(0..mine.len()), text);
        for row in 0..mine.line_count() + 2 {
            let want = if (row as usize) < oracle.len_lines() {
                let r = oracle.line(row as usize);
                let s = r.as_str().unwrap_or_default();
                s.strip_suffix('\n').unwrap_or(s).to_string()
            } else {
                String::new()
            };
            assert_eq!(mine.line(row), want, "line {row}");
        }
        for off in 0..=mine.len() {
            let p = mine.byte_to_point(off);
            let want_row = oracle.byte_to_line(off as usize) as u32;
            let want_col = off - oracle.line_to_byte(want_row as usize) as u32;
            assert_eq!(p, Point::new(want_row, want_col), "byte_to_point {off}");
            assert_eq!(mine.point_to_offset(p), off, "round-trip {off}");
        }
    }

    // Byte offset of a random char boundary in `s`, biased across the doc.
    fn rand_boundary(s: &str, r: u32) -> usize {
        let mut off = (r as usize) % (s.len() + 1);
        while !s.is_char_boundary(off) {
            off -= 1;
        }
        off
    }

    #[test]
    fn matches_ropey_under_random_edits() {
        let mut text = String::from("the quick brown fox\njumps over\nthe lazy dog\n");
        let mut mine = Rope::from_str(&text);
        let mut oracle = ropey::Rope::from_str(&text);
        let inserts = ["", "x", "hello", "\n", "a\nb\n", "  ", "λ", "→ok"];
        let mut state = 0x1234_5678u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for step in 0..600 {
            let a = rand_boundary(&text, next());
            let b = rand_boundary(&text[a..], next()) + a;
            let ins = inserts[(next() as usize) % inserts.len()];
            mine.replace(a as u32..b as u32, ins);
            // Oracle via String (ropey edits mirror it).
            let ca = oracle.byte_to_char(a);
            let cb = oracle.byte_to_char(b);
            oracle.remove(ca..cb);
            oracle.insert(ca, ins);
            text.replace_range(a..b, ins);

            assert_eq!(mine.len() as usize, oracle.len_bytes(), "step {step}: len");
            assert_eq!(mine.line_count() as usize, oracle.len_lines(), "step {step}: lines");
            assert_eq!(mine.slice(0..mine.len()), text, "step {step}: text");
            // Spot-check conversions and a few line reads against the oracle.
            // Snap probes to char boundaries (multibyte inserts shift them).
            for &raw in &[0u32, mine.len() / 3, mine.len() / 2, mine.len().saturating_sub(1), mine.len()] {
                let mut off = raw;
                while !mine.is_char_boundary(off) {
                    off -= 1;
                }
                let p = mine.byte_to_point(off);
                let want_row = oracle.byte_to_line(off as usize) as u32;
                assert_eq!(p.row, want_row, "step {step}: byte_to_point row @{off}");
                assert_eq!(mine.point_to_offset(p), off, "step {step}: round-trip @{off}");
            }
            for row in 0..mine.line_count().min(6) {
                let r = oracle.line(row as usize);
                let s = r.as_str().map(|s| s.strip_suffix('\n').unwrap_or(s).to_string());
                if let Some(want) = s {
                    assert_eq!(mine.line(row), want, "step {step}: line {row}");
                }
            }
        }
    }

    /// The anti-fragmentation canary: chunk count tracks BYTES, not keystrokes.
    ///
    /// Fails hard against a naive splice — typing N characters used to leave N
    /// one-byte chunks (measured: the 200 keystrokes this test types took a
    /// 61-chunk rope to 262 chunks, ~201 of them under `CHUNK_MIN`, the sizes at
    /// the caret reading `[1, 1, 1, ...]`). That is unbounded growth in edits,
    /// and it deepens the tree and inflates every O(log n) read.
    ///
    /// Asserted as AVERAGE fill, not a per-chunk floor, because re-chunking a
    /// seam always leaves one remainder that may legitimately be tiny. Op counts
    /// only — no wall-clock — so it holds on any machine.
    #[test]
    fn chunk_count_stays_proportional_to_bytes_not_edits() {
        let text = "lorem ipsum dolor sit amet consectetur ".repeat(200);
        let mut r = Rope::from_str(&text);
        let packed = r.chunks().count();
        assert!(
            packed <= r.len() as usize / CHUNK_MIN,
            "a freshly packed rope must already satisfy the invariant"
        );

        // Type a run at one spot — the case that fragments, because after the
        // first keystroke the caret sits on a chunk boundary.
        let base = (text.len() / 2) as u32;
        for at in base..base + 200 {
            r.replace(at..at, "x");
        }
        let n = r.chunks().count();
        eprintln!(
            "[rope] packed {packed} chunks -> {n} after 200 keystrokes ({} bytes, avg fill {}, \
             ceiling {})",
            r.len(),
            r.len() as usize / n,
            r.len() as usize / CHUNK_MIN
        );
        assert!(
            n <= r.len() as usize / CHUNK_MIN,
            "typing fragmented the rope: {n} chunks for {} bytes (max {}), sizes {:?}",
            r.len(),
            r.len() as usize / CHUNK_MIN,
            r.chunks().map(str::len).collect::<Vec<_>>()
        );

        // Scattered edits, deletions, and multi-byte inserts must not either.
        for i in 0..200u32 {
            let at = (i * 37) % (r.len().saturating_sub(8));
            let at = (0..=at).rev().find(|&o| r.is_char_boundary(o)).unwrap_or(0);
            r.replace(at..at, "ß");
        }
        for _ in 0..200 {
            let at = r.len() / 3;
            let at = (0..=at).rev().find(|&o| r.is_char_boundary(at.min(o))).unwrap_or(0);
            let end = (at + 1..=r.len()).find(|&o| r.is_char_boundary(o)).unwrap_or(r.len());
            r.replace(at..end, "");
        }
        let n = r.chunks().count();
        assert!(
            n <= r.len() as usize / CHUNK_MIN,
            "scattered edits fragmented the rope: {n} chunks for {} bytes (max {})",
            r.len(),
            r.len() as usize / CHUNK_MIN
        );
        // Nothing above may exceed the capacity the chunk type is built on.
        assert!(r.chunks().all(|c| c.len() <= CHUNK_MAX));
    }

    #[test]
    fn edit_many_matches_sequential_replace_and_string_model() {
        // The batched multi-edit path must produce a rope byte-identical (in text,
        // length, line count, and point round-trips) to applying the same disjoint
        // edits one-by-one with `replace`. Batches include multiple edits landing in
        // ONE leaf, appends at the end, multibyte + newline inserts, and occasional
        // LARGE deletes that straddle leaves (exercising the sequential fallback).
        let base = "the quick brown fox\njumps over the lazy dog\n".repeat(24); // multi-leaf
        let inserts: &[&str] = &["", "x", "hello", "\n", "a\nb\n", "  ", "λ", "→ok"];
        let mut state = 0x00C0_FFEEu32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            state
        };
        for trial in 0..400 {
            let len = base.len();
            let k = 1 + next() as usize % 8; // 1..=8 disjoint edits
            let mut edits: Vec<(Range<u32>, &str)> = Vec::new();
            let mut cursor = 0usize;
            for _ in 0..k {
                if cursor >= len {
                    break;
                }
                let room = len - cursor;
                let mut s = cursor + next() as usize % (room / k + 1).max(1);
                while s < len && !base.is_char_boundary(s) {
                    s += 1;
                }
                s = s.min(len);
                // 1-in-4 edits are a big delete (spans leaves → fallback); else small.
                let cap = if next() % 4 == 0 { 2000 } else { 24 };
                let elen = next() as usize % (cap.min(len - s) + 1);
                let mut e = s + elen;
                while e < len && !base.is_char_boundary(e) {
                    e += 1;
                }
                let ins = inserts[next() as usize % inserts.len()];
                edits.push((s as u32..e as u32, ins));
                cursor = e.max(s) + 1; // strictly disjoint
            }
            if edits.is_empty() {
                continue;
            }
            let mut batched = Rope::from_str(&base);
            batched.edit_many(&edits);
            // Sequential (descending) + a plain String oracle.
            let mut sequential = Rope::from_str(&base);
            let mut model = base.clone();
            for (r, t) in edits.iter().rev() {
                sequential.replace(r.clone(), t);
                model.replace_range(r.start as usize..r.end as usize, t);
            }
            assert_eq!(batched.slice(0..batched.len()), model.as_str(), "trial {trial}: {edits:?}");
            assert_eq!(batched.len() as usize, model.len(), "trial {trial}: len");
            assert_eq!(batched.line_count(), sequential.line_count(), "trial {trial}: lines");
            for &raw in &[0u32, batched.len() / 3, batched.len() / 2, batched.len()] {
                let mut off = raw;
                while off > 0 && !batched.is_char_boundary(off) {
                    off -= 1;
                }
                assert_eq!(
                    batched.point_to_offset(batched.byte_to_point(off)),
                    off,
                    "trial {trial}: point round-trip @{off}"
                );
            }
        }
    }

    #[test]
    fn char_boundaries_and_chunks() {
        let mine = Rope::from_str("aλb\nc→d"); // multi-byte chars
        assert!(mine.is_char_boundary(0));
        assert!(mine.is_char_boundary(1)); // after 'a'
        assert!(!mine.is_char_boundary(2)); // inside 'λ' (2 bytes)
        assert!(mine.is_char_boundary(3)); // after 'λ'
        assert!(mine.is_char_boundary(mine.len()));
        // chunk_at yields the containing chunk and its start.
        let (chunk, cs) = mine.chunk_at(3);
        assert!(cs <= 3 && (cs + chunk.len() as u32) > 3);
    }

    #[test]
    fn serialize_walk_is_verbatim() {
        let text = "a\nbb\nccc\n";
        let mine = Rope::from_str(text);
        let mut out = String::new();
        mine.for_each_chunk(|c| out.push_str(c));
        assert_eq!(out, text);
    }
}
