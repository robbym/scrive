//! The document text store: a rope.
//!
//! Storage is a `Rope` — an augmented `SumTree<Chunk>`, the one
//! tree every position query rides. The **LF-only line model** lives in the
//! chunk summary (a `'\n'` count), so line/point math is a summary fold with no
//! second index to drift; a test pins that only `'\n'` breaks a line. An edit is
//! `O(edit + log chunks)` and coordinate lookups are `O(log chunks)` cursor
//! descents — no operation is bandwidth-bound in the document length. There is no
//! load-size policy: the only bound is the `u32` offset space (~4 GiB), checked at
//! load and on the edit path. (The rope is oracle-tested byte-for-byte against an
//! independent reference model.)
//!
//! **The read API is backing-agnostic:** every read hands out [`Cow`] slices or
//! single chars —
//! [`Buffer::slice`], [`Buffer::line`], [`Buffer::char_at`],
//! [`Buffer::char_before`] — so no consumer outside this module assumes the
//! text is one contiguous allocation. A `Cow` is `Borrowed` when the requested
//! range lies inside one rope chunk (always, for documents small enough to be
//! a single leaf) and `Owned` when it crosses chunks. [`Buffer::text`] is the
//! one whole-text read: **cold-path only** (find seeding, whole-document
//! scans, serialization, tests) — it is a genuine O(len) materialization on
//! any document large enough to span chunks, and per-frame/per-keystroke code
//! must not call it.
//!
//! Internal text is **LF-only** and the trailing terminator, if any, is an
//! empty final line — never a flag. The file's majority EOL flavor (see
//! [`EolFlavor`]) is remembered only to re-expand at [`Buffer::serialize`].

use std::borrow::Cow;
use std::ops::Range;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::coords::{snap_char_boundary, Bias, Point};
use crate::rope::Rope;

/// Fixed window size for the buffer's backing-agnostic literal scanners
/// ([`crate::find`]'s `scan_buffer` and the document's `scan_from`, both via
/// [`Buffer::scan_resume`]) — big enough that the per-window match scan
/// dominates the per-window bookkeeping, small enough that a near or capped
/// match stops after touching only a prefix of the rope, never materializing
/// the whole document. One knob so the two scanners cannot drift.
pub(crate) const SCAN_WINDOW: u32 = 64 * 1024;

/// Why a [`Buffer::new`] load was refused.
///
/// There is **no policy size limit**: the only refusal is representational —
/// byte offsets are `u32`, so a document must fit the u32 offset space. Hosts
/// wanting a policy cap (e.g. "don't open logs in the editor") enforce it before
/// loading.
#[non_exhaustive]
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum LoadError {
    /// The input doesn't fit the `u32` offset space (~4 GiB).
    #[error("document of {len} bytes exceeds the u32 offset space (4 GiB)")]
    TooLarge {
        /// The rejected input's byte length.
        len: usize,
    },
}

/// The load-size guard: `Err` iff `len` can't be addressed by `u32` offsets.
/// Pure and length-only, so it is unit-testable without allocating 4 GiB.
fn check_load_len(len: usize) -> Result<(), LoadError> {
    if len >= u32::MAX as usize {
        return Err(LoadError::TooLarge { len });
    }
    Ok(())
}

/// Monotonic per-document transaction counter: `0` at load, `+1` per committed
/// transaction *including undo and redo*. Never reused within a document's
/// lifetime, never runs backwards. The universal cache-key component and async
/// correlation stamp.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug)]
pub struct Revision(pub u64);

/// Process-unique document identity, minted fresh at each load. Async replies
/// stamped `(DocId, Revision)` can never collide across reloads.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub struct DocId(u64);

static NEXT_DOC_ID: AtomicU64 = AtomicU64::new(1);

/// The line-ending flavor detected at load; consulted only by
/// [`Buffer::serialize`]. Internal text is always LF.
///
/// Detection is a **majority vote** over the file's terminators, matching VS
/// Code: `CrLf` iff CRLF terminators strictly outnumber LF ones, so a **tie
/// goes to `Lf`** — including the no-terminator case (an empty or one-line
/// file). A lone `\r` is normalized to `\n` but casts no vote. Saving in the
/// detected flavor therefore round-trips a uniformly terminated file
/// byte-for-byte and rewrites only the minority lines of a mixed one.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum EolFlavor {
    /// Unix `\n`.
    Lf,
    /// Windows `\r\n`.
    CrLf,
}

impl EolFlavor {
    /// The one owner of the vote rule (see the type docs): strict CRLF
    /// majority, else `Lf`.
    fn majority(lf: usize, crlf: usize) -> EolFlavor {
        if crlf > lf {
            EolFlavor::CrLf
        } else {
            EolFlavor::Lf
        }
    }
}

/// Normalize `\r\n | \r → \n` in one pass over `input`'s CRs, starting at
/// `first_cr` (the first one; the caller already found it), and report what it
/// rewrote: `(normalized, crlf, lone_cr)`. The counts feed the [`EolFlavor`]
/// vote (the caller derives the LF side from them), gathered here so
/// detection costs no pass of its own.
/// Slicing at CR/LF positions is sound: both are ASCII, so every cut is a char
/// boundary.
fn normalize_eol(input: &str, first_cr: usize) -> (String, usize, usize) {
    let bytes = input.as_bytes();
    let mut out = String::with_capacity(input.len());
    let (mut crlf, mut lone_cr) = (0, 0);
    let mut copied = 0; // input[..copied] is already in `out` (normalized)
    for cr in memchr::memchr_iter(b'\r', &bytes[first_cr..]).map(|i| first_cr + i) {
        out.push_str(&input[copied..cr]);
        out.push('\n');
        // `\r\r\n` is a lone CR then a CRLF: the pair test looks only at the
        // next byte, and a skipped '\n' is never a CR the iterator could yield.
        if bytes.get(cr + 1) == Some(&b'\n') {
            crlf += 1;
            copied = cr + 2;
        } else {
            lone_cr += 1;
            copied = cr + 1;
        }
    }
    out.push_str(&input[copied..]);
    (out, crlf, lone_cr)
}

/// An immutable, `Send + Sync` copy of the document for background consumers
/// (e.g. a debounced compile thread).
///
/// Creation is **O(1)** — a rope clone shares structure. The consumer pays for
/// what it reads: [`Snapshot::text`]
/// is an O(len) materialization that now runs on the *consumer's* thread, not
/// the UI thread; the ranged reads mirror [`Buffer`]'s backing-agnostic API.
#[derive(Clone, Debug)]
pub struct Snapshot {
    text: Rope,
    doc_id: DocId,
    revision: Revision,
}

impl Snapshot {
    /// Total byte length of the snapshot.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.text.len()
    }

    /// Whether the snapshot is empty (a single empty line — never truly zero).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The whole snapshot text, LF-only. `Borrowed` for single-chunk (small)
    /// documents; an O(len) materialization otherwise — run it on the
    /// consumer's thread.
    #[must_use]
    pub fn text(&self) -> Cow<'_, str> {
        self.text.slice(0..self.text.len())
    }

    /// The text in `range` (byte offsets on char boundaries). Clamps to the
    /// snapshot end.
    ///
    /// # Panics
    ///
    /// If either end of the (clamped) range splits a multibyte char — the same
    /// kept precondition as [`Buffer::slice`]. A snapshot is the buffer's text
    /// at one revision, so offsets clipped by that buffer's
    /// [`Buffer::clip_offset`] at that revision are safe here.
    #[must_use]
    pub fn slice(&self, range: Range<u32>) -> Cow<'_, str> {
        self.text.slice(range)
    }

    /// The text of one row, excluding its trailing `\n`. Out-of-range rows
    /// return `""`.
    #[must_use]
    pub fn line(&self, row: u32) -> Cow<'_, str> {
        self.text.line(row)
    }

    /// Number of lines. Always ≥ 1; a trailing `\n` yields a final empty line.
    #[must_use]
    pub fn line_count(&self) -> u32 {
        self.text.line_count()
    }

    /// The document this snapshot came from.
    #[must_use]
    pub fn doc_id(&self) -> DocId {
        self.doc_id
    }

    /// The revision this snapshot froze.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.revision
    }
}

/// The single owner of document text.
///
/// Invariants, upheld inside every mutation so no observer sees them disagree:
/// 1. `text` is LF-only (no `\r`). The trailing terminator, if any, is an empty
///    final line in `text`, not a flag. Line boundaries are the rope's own
///    bookkeeping (LF-only by feature selection) — there is no second index to
///    drift.
/// 2. `revision` bumps once per committed transaction (via `Buffer::bump_revision`,
///    called by the transaction engine).
#[derive(Clone, Debug)]
pub struct Buffer {
    text: Rope,
    revision: Revision,
    doc_id: DocId,
    eol_flavor: EolFlavor,
}

impl Buffer {
    /// Load `input` as a new document.
    ///
    /// Refuses only input past the `u32` offset space (see [`LoadError`] — no
    /// policy limit); normalizes `\r\n | \r → \n`, remembering the majority
    /// [`EolFlavor`] (tie → `Lf`); mints a fresh [`DocId`] at revision `0`.
    pub fn new(input: &str) -> Result<Self, LoadError> {
        check_load_len(input.len())?;
        // Detection rides passes load already makes — never a counting pass of
        // its own. A CR-free input (the common case, settled by one memchr) is
        // already LF-only: it has no CRLF terminators, so the majority is LF
        // whatever its newline count, and the input is chunked without a copy.
        // Otherwise the one normalizing pass counts the CRLFs and lone CRs it
        // rewrites, and the LF count falls out of the rope's O(1) newline
        // total: every terminator became exactly one '\n'. (Normalization only
        // shrinks, so the length check above covers both paths.)
        let (text, eol_flavor) = match memchr::memchr(b'\r', input.as_bytes()) {
            None => (Rope::from_str(input), EolFlavor::Lf),
            Some(first_cr) => {
                let (normalized, crlf, lone_cr) = normalize_eol(input, first_cr);
                let text = Rope::from_str(&normalized);
                let lf = (text.line_count() - 1) as usize - crlf - lone_cr;
                (text, EolFlavor::majority(lf, crlf))
            }
        };
        Ok(Self {
            text,
            revision: Revision(0),
            doc_id: DocId(NEXT_DOC_ID.fetch_add(1, Ordering::Relaxed)),
            eol_flavor,
        })
    }

    /// Total byte length of the document.
    #[must_use]
    pub fn len(&self) -> u32 {
        self.text.len()
    }

    /// Whether the document is empty (a single empty line — never truly zero).
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// The whole document text, LF-only, as one slice.
    ///
    /// **Cold-path read:** `Borrowed` only while the
    /// document is a single rope chunk; an O(len) materialization otherwise.
    /// For whole-document scans (find seeding, select-all-occurrences),
    /// serialization, and tests — per-frame and per-keystroke code reads via
    /// [`Buffer::slice`] / [`Buffer::line`] / [`Buffer::char_at`] /
    /// [`Buffer::char_before`].
    #[must_use]
    pub fn text(&self) -> Cow<'_, str> {
        self.text.slice(0..self.text.len())
    }

    /// The text in `range` (byte offsets; must lie on char boundaries, like
    /// `str` slicing). Clamps to the document end. The workhorse ranged read —
    /// `Borrowed` while the range sits in one rope chunk, `Owned` across
    /// chunks.
    ///
    /// # Panics
    ///
    /// If either end of the (clamped) range splits a multibyte char. The
    /// precondition is kept rather than clipped: this low-level read sits
    /// under every scan, where a silently snapped end would turn a caller's
    /// off-by-a-byte bug into a wrong read. Snap host offsets first with
    /// [`Buffer::clip_offset`] (the editing APIs on
    /// [`Document`](crate::Document) already do).
    #[must_use]
    pub fn slice(&self, range: Range<u32>) -> Cow<'_, str> {
        self.text.slice(range)
    }

    /// The text of one row, excluding its trailing `\n`. Out-of-range rows
    /// return `""`.
    #[must_use]
    pub fn line(&self, row: u32) -> Cow<'_, str> {
        self.text.line(row)
    }

    /// The char whose first byte is at `offset` — `None` at/past the end.
    /// (`offset` must be a char boundary.) The forward one-char read every
    /// boundary scan uses instead of slicing the whole text. `O(log chunks)`.
    ///
    /// # Panics
    ///
    /// If `offset` is inside the document but splits a multibyte char — kept,
    /// like [`Buffer::slice`]'s, so a boundary-scan bug cannot read a char
    /// that is not there; snap host offsets with [`Buffer::clip_offset`].
    #[must_use]
    pub fn char_at(&self, offset: u32) -> Option<char> {
        let off = offset.min(self.len());
        // chunk_at returns the chunk *containing* `off` (the right-hand chunk at
        // an exact boundary), so the tail slice is non-empty except at the
        // document end.
        let (chunk, chunk_start) = self.text.chunk_at(off);
        chunk[(off - chunk_start) as usize..].chars().next()
    }

    /// The char ending at `offset` — `None` at 0. (`offset` must be a char
    /// boundary.) The backward twin of [`Buffer::char_at`].
    ///
    /// # Panics
    ///
    /// If `offset` is inside the document but splits a multibyte char — the
    /// same kept precondition as [`Buffer::char_at`].
    #[must_use]
    pub fn char_before(&self, offset: u32) -> Option<char> {
        let off = offset.min(self.len());
        if off == 0 {
            return None;
        }
        // The char ending at `off` starts at most 4 bytes back; chunk
        // boundaries are char boundaries, so the chunk holding byte `off - 1`
        // holds the whole char.
        let (chunk, chunk_start) = self.text.chunk_at(off - 1);
        chunk[..(off - chunk_start) as usize].chars().next_back()
    }

    /// Number of lines. Always ≥ 1 (an empty document is one empty line); a
    /// trailing `\n` yields a final empty line, so `line_count("a\nb\n") == 3`.
    #[must_use]
    pub fn line_count(&self) -> u32 {
        self.text.line_count()
    }

    /// Byte length of one row's text (excludes `\n`). Out-of-range → 0.
    /// Allocation-free (range arithmetic, no line materialization).
    #[must_use]
    pub fn line_len(&self, row: u32) -> u32 {
        self.text.line_len(row)
    }

    /// The current revision.
    #[must_use]
    pub fn revision(&self) -> Revision {
        self.revision
    }

    /// This document's identity.
    #[must_use]
    pub fn doc_id(&self) -> DocId {
        self.doc_id
    }

    /// The EOL flavor detected at load: the file's majority terminator, tie →
    /// `Lf` (see [`EolFlavor`]). Pass it to [`Buffer::serialize`] to save in
    /// the file's own convention.
    #[must_use]
    pub fn eol_flavor(&self) -> EolFlavor {
        self.eol_flavor
    }

    /// Convert a byte offset to a [`Point`]. Offsets past the end clamp to the
    /// end of the document. `O(log chunks)`.
    #[must_use]
    pub fn offset_to_point(&self, offset: u32) -> Point {
        crate::perf::charge(1); // complexity gate: one position query
        self.text.byte_to_point(offset)
    }

    /// Convert a [`Point`] to a byte offset, clamping the row to the last line
    /// and the column to that line's length, then snapping a column that falls
    /// inside a multibyte char [`Bias::Left`] to that char's start — so the
    /// result is always a char boundary, whatever arithmetic (UTF-16 / LSP
    /// columns, a goal column carried across lines) produced `point`. For the
    /// other snap direction use [`Buffer::clip_point`]. `O(log chunks)`.
    #[must_use]
    pub fn point_to_offset(&self, point: Point) -> u32 {
        self.point_to_offset_snapped(point, Bias::Left)
    }

    /// [`Buffer::point_to_offset`] with the snap direction chosen — the one
    /// owner of the clamp-then-snap, shared with [`Buffer::clip_point`].
    fn point_to_offset_snapped(&self, point: Point, bias: Bias) -> u32 {
        crate::perf::charge(1); // complexity gate: one position query
        // The rope clamps row and col but counts the col in raw bytes; the
        // snap is one more chunk descent, and a boundary (col 0, the line
        // end, any ASCII column) comes back unchanged.
        self.clip_offset(self.text.point_to_offset(point), bias)
    }

    /// Clamp a byte offset into `[0, len]` and snap it to a char boundary
    /// (direction from `bias`). Idempotent.
    #[must_use]
    pub fn clip_offset(&self, offset: u32, bias: Bias) -> u32 {
        let off = offset.min(self.len());
        // A non-boundary offset sits strictly inside one char, and chunk
        // boundaries are char boundaries — so the snap never leaves the chunk
        // containing `off`.
        let (chunk, chunk_start) = self.text.chunk_at(off);
        chunk_start + snap_char_boundary(chunk, off - chunk_start, bias)
    }

    /// Clip a host-supplied range to valid offsets — THE rule every API that
    /// stores a caller's range (selections, diagnostics, the find scope) shares,
    /// so they cannot disagree about what a mid-char or past-end range means.
    ///
    /// Both ends clamp into `[0, len]`; an inverted range collapses to an empty
    /// one at its (clamped) end, as [`Buffer::slice`] reads it. A non-empty
    /// range then snaps **outward** (start [`Bias::Left`], end [`Bias::Right`]),
    /// so it never drops part of a char it covers — the transaction engine's
    /// rule for edit ranges. An empty one snaps both ends `Left`, so a caret or
    /// zero-width span stays empty, at the start of the char it split, instead
    /// of growing over that char.
    #[must_use]
    pub(crate) fn clip_range(&self, range: Range<u32>) -> Range<u32> {
        let end = range.end.min(self.len());
        let start = range.start.min(end);
        if start == end {
            let at = self.clip_offset(start, Bias::Left);
            return at..at;
        }
        self.clip_offset(start, Bias::Left)..self.clip_offset(end, Bias::Right)
    }

    /// Where a windowed literal scan resumes after examining the window
    /// `[pos, win_end)` for a `k`-byte needle (`k ≥ 1`). The **one** owner of
    /// the seam-scan resume rule, shared by [`crate::find`]'s `scan_buffer` and
    /// the document's `scan_from` so the subtle overlap arithmetic lives in a
    /// single place: resume past the last in-window match end (`last_match_end`)
    /// when the window found one; otherwise at the last char boundary at or
    /// before `win_end − (k−1)` — no match can START later and still end inside
    /// this window — clamped forward past `pos` so a multi-byte char can never
    /// stall the scan.
    #[must_use]
    pub(crate) fn scan_resume(&self, pos: u32, win_end: u32, k: u32, last_match_end: Option<u32>) -> u32 {
        match last_match_end {
            Some(end) => end,
            None => {
                let safe = self.clip_offset((u64::from(win_end) - u64::from(k - 1)) as u32, Bias::Left);
                safe.max(self.clip_offset(pos + 1, Bias::Right))
            }
        }
    }

    /// Clip a [`Point`] to a valid position: clamp the row/col, then snap the
    /// resulting offset to a char boundary (direction from `bias`).
    #[must_use]
    pub fn clip_point(&self, point: Point, bias: Bias) -> Point {
        // Not via `point_to_offset`: it has already snapped `Left`, which
        // would swallow a `Right` bias.
        self.offset_to_point(self.point_to_offset_snapped(point, bias))
    }

    /// An immutable copy for background consumers. **O(1)** — the
    /// rope clone shares structure; nothing is materialized until the
    /// consumer reads.
    #[must_use]
    pub fn snapshot(&self) -> Snapshot {
        Snapshot { text: self.text.clone(), doc_id: self.doc_id, revision: self.revision }
    }

    /// Serialize for saving: verbatim EOL-expansion of the stored LF text into
    /// `flavor`. The empty final line reproduces a trailing terminator by
    /// construction; a no-trailing-newline file round-trips without one.
    /// Cold-path O(len), like any save.
    #[must_use]
    pub fn serialize(&self, flavor: EolFlavor) -> String {
        match flavor {
            EolFlavor::Lf => {
                let mut out = String::with_capacity(self.text.len() as usize);
                self.text.for_each_chunk(|chunk| out.push_str(chunk));
                out
            }
            EolFlavor::CrLf => {
                // One byte per '\n' on top of len; chunk boundaries are char
                // boundaries, so a '\n' never splits across chunks.
                let extra = self.text.line_count() as usize - 1;
                let mut out = String::with_capacity(self.text.len() as usize + extra);
                for chunk in self.text.chunks() {
                    let mut rest = chunk;
                    while let Some(i) = rest.find('\n') {
                        out.push_str(&rest[..i]);
                        out.push_str("\r\n");
                        rest = &rest[i + 1..];
                    }
                    out.push_str(rest);
                }
                out
            }
        }
    }

    /// Replace `range` (byte offsets in the current text) with `new_text`.
    /// `O(edit + log chunks)` — the rope's bookkeeping *is* the line index,
    /// so there is no second structure to splice.
    ///
    /// `pub(crate)`: reachable only through the transaction engine,
    /// which owns validation, revision bumping, and inverse derivation.
    /// `new_text` must already be LF-only (the transaction boundary normalizes).
    pub(crate) fn splice(&mut self, range: Range<u32>, new_text: &str) {
        debug_assert!(!new_text.as_bytes().contains(&b'\r'), "splice text must be LF-only");
        let (start, end) = (range.start, range.end);
        debug_assert!(start <= end && end <= self.len(), "splice range out of bounds");

        debug_assert!(self.text.is_char_boundary(start), "splice start is not a char boundary");
        debug_assert!(self.text.is_char_boundary(end), "splice end is not a char boundary");
        self.text.replace(start..end, new_text);
    }

    /// Apply MANY disjoint edits (`(range, replacement)`, sorted ascending, disjoint,
    /// in-bounds, LF-only) in ONE rope pass — the batched twin of N [`Self::splice`]s
    /// (the document-scale multi-caret path). The transaction engine owns the
    /// validation and revision bump, exactly as for `splice`.
    pub(crate) fn edit_many(&mut self, edits: &[(Range<u32>, &str)]) {
        debug_assert!(
            edits.windows(2).all(|w| w[0].0.end <= w[1].0.start),
            "edit_many wants sorted, disjoint ranges"
        );
        debug_assert!(
            edits.iter().all(|(r, t)| {
                r.end <= self.len()
                    && self.text.is_char_boundary(r.start)
                    && self.text.is_char_boundary(r.end)
                    && !t.as_bytes().contains(&b'\r')
            }),
            "edit_many ranges must be in-bounds char boundaries and text LF-only"
        );
        // The common single-caret case takes the direct rope splice — no batch
        // machinery for one edit; the batch pays off only with many carets.
        match edits {
            [] => {}
            [(range, text)] => self.splice(range.clone(), text),
            _ => self.text.edit_many(edits),
        }
    }

    /// Bump the revision. Called once per committed transaction by the engine;
    /// never call it from a raw splice.
    pub(crate) fn bump_revision(&mut self) {
        self.revision.0 += 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn buf(s: &str) -> Buffer {
        Buffer::new(s).unwrap()
    }

    #[test]
    fn empty_document_is_one_empty_line() {
        let b = buf("");
        assert_eq!(b.line_count(), 1);
        assert_eq!(b.line(0), "");
        assert_eq!(b.len(), 0);
        assert!(b.is_empty());
    }

    #[test]
    fn trailing_newline_is_an_empty_final_line() {
        // "foo\nbar\n" → ["foo","bar",""], count 3.
        let b = buf("foo\nbar\n");
        assert_eq!(b.line_count(), 3);
        assert_eq!(b.line(0), "foo");
        assert_eq!(b.line(1), "bar");
        assert_eq!(b.line(2), "");
        // No trailing newline → 2 lines, caret rests at end of "bar".
        let b2 = buf("foo\nbar");
        assert_eq!(b2.line_count(), 2);
        assert_eq!(b2.line(1), "bar");
    }

    #[test]
    fn crlf_normalized_on_load_flavor_remembered() {
        let b = buf("a\r\nb\r\n");
        assert_eq!(b.text(), "a\nb\n"); // LF-only internally
        assert_eq!(b.eol_flavor(), EolFlavor::CrLf);
        assert!(!b.text().contains('\r'));
        assert_eq!(buf("a\nb").eol_flavor(), EolFlavor::Lf);
    }

    #[test]
    fn serialize_round_trips_each_flavor() {
        // CRLF file → load (LF internal) → serialize(CrLf) == original.
        let original = "a\r\nb\r\n";
        let b = buf(original);
        assert_eq!(b.serialize(EolFlavor::CrLf), original);
        assert_eq!(b.serialize(EolFlavor::Lf), "a\nb\n");
        // No-trailing-newline LF file round-trips without one.
        let b2 = buf("x\ny");
        assert_eq!(b2.serialize(EolFlavor::Lf), "x\ny");
    }

    /// Regression: one stray CRLF in a mostly-LF file used to flip the whole
    /// file to CRLF, so an untouched save rewrote every line. The majority (3
    /// LF vs 1 CRLF) wins, and saving in it diffs only the stray line.
    #[test]
    fn one_stray_crlf_does_not_flip_a_mostly_lf_file() {
        let b = buf("a\nb\nc\nd\r\n");
        assert_eq!(b.eol_flavor(), EolFlavor::Lf);
        assert_eq!(b.serialize(b.eol_flavor()), "a\nb\nc\nd\n");
    }

    /// The mirror case: a stray LF in a CRLF file is the minority, so the
    /// file stays CRLF and only the stray line changes on save.
    #[test]
    fn one_stray_lf_does_not_flip_a_mostly_crlf_file() {
        let b = buf("a\r\nb\r\nc\nd\r\n");
        assert_eq!(b.eol_flavor(), EolFlavor::CrLf);
        assert_eq!(b.serialize(b.eol_flavor()), "a\r\nb\r\nc\r\nd\r\n");
    }

    /// The documented tie rule: equal counts go to LF, and so does a file
    /// with no terminators at all (0 vs 0).
    #[test]
    fn an_eol_tie_goes_to_lf() {
        assert_eq!(buf("a\r\nb\n").eol_flavor(), EolFlavor::Lf);
        assert_eq!(buf("a\nb\r\nc\r\nd\n").eol_flavor(), EolFlavor::Lf);
        assert_eq!(buf("").eol_flavor(), EolFlavor::Lf);
        assert_eq!(buf("one line").eol_flavor(), EolFlavor::Lf);
    }

    /// A lone CR is normalized to '\n' but casts no vote — in particular not
    /// an LF one, though it becomes one '\n' in the rope's newline total that
    /// the LF count is derived from. 2 CRLF vs 0 LF (3 lone CR) is CRLF.
    #[test]
    fn lone_cr_casts_no_eol_vote() {
        let b = buf("a\r\nb\r\nc\rd\re\rf");
        assert_eq!(b.text(), "a\nb\nc\nd\ne\nf");
        assert_eq!(b.eol_flavor(), EolFlavor::CrLf);
        // Only lone CRs: no vote either way, so the 0-0 tie rule applies.
        assert_eq!(buf("a\rb\r").eol_flavor(), EolFlavor::Lf);
    }

    /// Seeded oracle over random terminator mixes — uniform LF, uniform CRLF,
    /// random shares (near-ties included), and lone CRs incl. the `\r\r\n`
    /// straddle — with multibyte bodies and multi-chunk sizes. Load must
    /// normalize exactly like the retained two-`replace` chain, pick the
    /// flavor an independent per-terminator count gives (the LF count is
    /// derived from the rope total, so this pins that derivation), and
    /// round-trip a uniformly terminated input byte-for-byte.
    #[test]
    fn eol_detection_matches_a_counting_oracle() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64; // deterministic LCG
        let mut rand = move |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound.max(1)
        };
        let bodies = ["", "x", "fn main() {}", "äöü ß", "🦀 rope", "\t  indented", "日本語"];
        // Outcomes among MIXED inputs (both kinds present) — where the bug was.
        let (mut mixed_lf, mut mixed_crlf, mut round_trips, mut lone_cases, mut multi_chunk) =
            (0, 0, 0, 0, 0);
        for case in 0..400 {
            let mix = case % 4; // 0 all-LF, 1 all-CRLF, 2 random share, 3 + lone CRs
            let crlf_pct = match mix {
                0 => 0,
                1 => 100,
                _ => rand(101),
            };
            let lines = if case % 16 < 2 { 2000 + rand(500) } else { rand(40) };
            let mut input = String::new();
            for _ in 0..lines {
                input.push_str(bodies[rand(bodies.len())]);
                if mix == 3 && rand(6) == 0 {
                    input.push('\r'); // a lone CR, or the straddle `\r\r\n`
                }
                input.push_str(if rand(100) < crlf_pct { "\r\n" } else { "\n" });
            }
            if rand(2) == 0 {
                input.push_str("no trailing terminator");
            }

            // Independent count: a piece of `split_inclusive('\n')` ends in
            // exactly one terminator; a lone CR never ends a piece.
            let (mut lf, mut crlf) = (0, 0);
            for piece in input.split_inclusive('\n') {
                if piece.ends_with("\r\n") {
                    crlf += 1;
                } else if piece.ends_with('\n') {
                    lf += 1;
                }
            }
            let has_lone_cr = input.matches('\r').count() > crlf;
            let want = if crlf > lf { EolFlavor::CrLf } else { EolFlavor::Lf };

            let b = buf(&input);
            assert_eq!(
                b.text(),
                input.replace("\r\n", "\n").replace('\r', "\n"),
                "normalization diverged in case {case}"
            );
            assert_eq!(b.eol_flavor(), want, "case {case}: {lf} LF vs {crlf} CRLF");
            if !has_lone_cr && (lf == 0 || crlf == 0) {
                assert_eq!(b.serialize(b.eol_flavor()), input, "uniform case {case} must round-trip");
                round_trips += 1;
            }
            if lf > 0 && crlf > 0 {
                match want {
                    EolFlavor::Lf => mixed_lf += 1,
                    EolFlavor::CrLf => mixed_crlf += 1,
                }
            }
            lone_cases += usize::from(has_lone_cr);
            multi_chunk += usize::from(b.text.chunks().nth(1).is_some());
        }
        // Keep the oracle honest: every regime it claims must actually occur.
        assert!(mixed_lf >= 50 && mixed_crlf >= 50, "mixed: {mixed_lf} LF / {mixed_crlf} CRLF wins");
        assert!(round_trips >= 150, "only {round_trips} uniform round-trips");
        assert!(lone_cases >= 50, "only {lone_cases} lone-CR cases");
        assert!(multi_chunk >= 40, "only {multi_chunk} multi-chunk cases");
    }

    #[test]
    fn only_the_u32_offset_space_is_refused() {
        // No policy limit: a 2 MB document loads fine…
        assert!(Buffer::new(&"a".repeat(2 * 1_048_576)).is_ok());
        // …and the representational guard is tested pure (no 4 GiB alloc).
        assert!(check_load_len(u32::MAX as usize - 1).is_ok());
        assert!(matches!(
            check_load_len(u32::MAX as usize),
            Err(LoadError::TooLarge { len }) if len == u32::MAX as usize
        ));
        assert!(check_load_len(u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn bump_revision_increments() {
        // Exercised for real by the transaction engine; pinned here so the
        // counter's contract stands on its own.
        let mut b = buf("x");
        assert_eq!(b.revision(), Revision(0));
        b.bump_revision();
        assert_eq!(b.revision(), Revision(1));
    }

    #[test]
    fn offset_point_round_trip() {
        let b = buf("foo\nbar\nbaz");
        for off in 0..=b.len() {
            let p = b.offset_to_point(off);
            assert_eq!(b.point_to_offset(p), off, "offset {off}");
        }
        assert_eq!(b.offset_to_point(0), Point::new(0, 0));
        assert_eq!(b.offset_to_point(4), Point::new(1, 0)); // start of "bar"
        assert_eq!(b.offset_to_point(6), Point::new(1, 2)); // 'r' in "bar"
    }

    #[test]
    fn point_to_offset_clamps_row_and_col() {
        let b = buf("ab\ncd");
        assert_eq!(b.point_to_offset(Point::new(0, 99)), 2); // col clamps to line end
        assert_eq!(b.point_to_offset(Point::new(9, 0)), 3); // row clamps to last line start
    }

    /// Regression (fails-first): a column inside a multibyte char (UTF-16 /
    /// LSP column math) came back as a mid-char offset, which the next
    /// `char_at`/`slice` panicked on. Every column now lands on a char
    /// boundary — the one at or before it — checked exhaustively against a
    /// `String` model over every (row, col) in and past each line, multi-chunk
    /// so seams are crossed.
    #[test]
    fn point_to_offset_snaps_every_column_to_a_char_boundary() {
        let lines = ["aé🦀b", "", "日本語", "🦀🦀"];
        let mut model = String::new();
        for i in 0..40 {
            if i > 0 {
                model.push('\n');
            }
            model.push_str(lines[i % lines.len()]);
        }
        let b = buf(&model);
        assert!(b.text.chunks().nth(1).is_some(), "corpus must span multiple chunks");
        let mut start = 0usize;
        for (row, line) in model.split('\n').enumerate() {
            for col in 0..=line.len() + 2 {
                let want = (0..=col.min(line.len())).rev().find(|&c| line.is_char_boundary(c)).unwrap();
                let got = b.point_to_offset(Point::new(row as u32, col as u32));
                assert_eq!(got as usize, start + want, "({row}, {col})");
                assert_eq!(b.char_at(got), model[got as usize..].chars().next(), "readable at ({row}, {col})");
            }
            start += line.len() + 1;
        }
        // `clip_point` keeps its caller's bias: Right still snaps forward.
        assert_eq!(b.clip_point(Point::new(0, 2), Bias::Right), Point::new(0, 3));
        assert_eq!(b.clip_point(Point::new(0, 2), Bias::Left), Point::new(0, 1));
    }

    #[test]
    fn splice_insert_updates_index_and_text() {
        let mut b = buf("ac");
        b.splice(1..1, "b"); // insert 'b' between a and c
        assert_eq!(b.text(), "abc");
        assert_eq!(b.line_count(), 1);

        let mut b = buf("line1\nline2");
        b.splice(5..5, "X\nY"); // insert a newline mid-document
        assert_eq!(b.text(), "line1X\nY\nline2");
        assert_eq!(b.line_count(), 3);
        assert_eq!(b.line(0), "line1X");
        assert_eq!(b.line(1), "Y");
        assert_eq!(b.line(2), "line2");
    }

    #[test]
    fn splice_delete_across_lines_merges() {
        let mut b = buf("foo\nbar\nbaz");
        b.splice(2..9, ""); // delete "o\nbar\nb" → "fo" + "az"... check
        assert_eq!(b.text(), "foaz");
        assert_eq!(b.line_count(), 1);
    }

    #[test]
    fn splice_replace_shifts_following_line_starts() {
        let mut b = buf("a\nb\nc");
        // Replace "a" (0..1) with "AAAA": every following line start shifts +3.
        b.splice(0..1, "AAAA");
        assert_eq!(b.text(), "AAAA\nb\nc");
        assert_eq!(b.line(0), "AAAA");
        assert_eq!(b.line(1), "b");
        assert_eq!(b.line(2), "c");
    }

    // ------------------------------------------------------------------
    // Cross-chunk correctness: these pin the reads and edits that must behave
    // identically no matter how the rope splits text into chunks. The line
    // model in particular counts only '\n' — `only_lf_is_a_line_break` guards
    // that other Unicode line separators never break a line.
    // ------------------------------------------------------------------

    /// The editor's line model counts only '\n'. Other Unicode line separators
    /// — U+000B/U+000C/U+0085/U+2028/U+2029 and CR — must not break a line, so
    /// display rows map one-to-one to LF-delimited text lines.
    #[test]
    fn only_lf_is_a_line_break() {
        let b = buf("a\u{000B}b\u{000C}c\u{0085}d\u{2028}e\u{2029}f\ng");
        assert_eq!(b.line_count(), 2, "only \\n may break lines");
        assert_eq!(b.line(1), "g");
        assert_eq!(b.offset_to_point(b.len()).row, 1);
    }

    /// A document big enough to span many rope chunks must answer every read
    /// identically to a reference `String` model (multibyte chars included,
    /// positioned to land on chunk seams somewhere in the run).
    #[test]
    fn chunk_crossing_reads_match_a_string_model() {
        // ~64 KB of varied lines — far past one leaf chunk.
        let mut model = String::new();
        for i in 0..2000 {
            match i % 4 {
                0 => model.push_str(&format!("line {i}: the quick brown fox\n")),
                1 => model.push_str(&format!("Zeile {i}: äöü ßẞ €42 →←\n")),
                2 => model.push_str(&format!("行 {i}: 日本語のテキスト 🦀🚀\n")),
                _ => model.push_str(&format!("l{i}\n")),
            }
        }
        model.push_str("no trailing newline");
        let b = buf(&model);

        assert_eq!(b.len() as usize, model.len());
        assert_eq!(b.text(), model.as_str());

        // Reference line starts: [0] plus one entry after every '\n'.
        let mut starts = vec![0usize];
        starts.extend(memchr::memchr_iter(b'\n', model.as_bytes()).map(|i| i + 1));
        assert_eq!(b.line_count() as usize, starts.len());
        for (row, &start) in starts.iter().enumerate() {
            let end = starts.get(row + 1).map_or(model.len(), |&next| next - 1);
            assert_eq!(b.line(row as u32), &model[start..end], "line {row}");
            assert_eq!(b.line_len(row as u32) as usize, end - start, "line_len {row}");
        }

        // Sampled offsets (snapped to boundaries) — chars, points, slices, clips.
        let mut off = 0usize;
        while off <= model.len() {
            let o = off as u32;
            assert_eq!(b.char_at(o), model[off..].chars().next(), "char_at {off}");
            assert_eq!(b.char_before(o), model[..off].chars().next_back(), "char_before {off}");
            let p = b.offset_to_point(o);
            assert_eq!(b.point_to_offset(p), o, "point round-trip {off}");
            assert_eq!(starts[p.row as usize] + p.col as usize, off, "offset_to_point {off}");
            assert_eq!(b.clip_offset(o, Bias::Left), o, "boundary clips are identity");
            let slice_end = (off + 97).min(model.len());
            let slice_end = (slice_end..=model.len())
                .find(|&e| model.is_char_boundary(e))
                .unwrap_or(model.len());
            assert_eq!(b.slice(o..slice_end as u32), &model[off..slice_end], "slice at {off}");
            // Next char boundary ≥ off + 61 (prime stride to wander chunk seams).
            off = (off + 61..=model.len())
                .find(|&e| model.is_char_boundary(e))
                .unwrap_or(model.len() + 1);
        }

        // Non-boundary clips snap like str-based snapping did.
        for (i, _) in model.char_indices().take(500) {
            for probe in i + 1..(i + 4).min(model.len()) {
                if !model.is_char_boundary(probe) {
                    let left = (0..=probe).rev().find(|&x| model.is_char_boundary(x)).unwrap();
                    let right = (probe..=model.len()).find(|&x| model.is_char_boundary(x)).unwrap();
                    assert_eq!(b.clip_offset(probe as u32, Bias::Left) as usize, left);
                    assert_eq!(b.clip_offset(probe as u32, Bias::Right) as usize, right);
                }
            }
        }

        // Every chunk seam, structurally: the sampling strides above only
        // graze seams by numeric coincidence, and the seam is exactly where
        // char_at/char_before/clip pick their chunk. (The test module may
        // read the private rope to enumerate seams.)
        let seams: Vec<usize> = {
            let mut acc = 0usize;
            b.text
                .chunks()
                .map(|c| {
                    acc += c.len();
                    acc
                })
                .collect()
        };
        assert!(seams.len() > 1, "corpus must span multiple chunks");
        for &seam in &seams[..seams.len() - 1] {
            for probe in [seam - 1, seam, seam + 1] {
                let o = probe as u32;
                if model.is_char_boundary(probe) {
                    assert_eq!(b.char_at(o), model[probe..].chars().next(), "char_at seam {probe}");
                    assert_eq!(
                        b.char_before(o),
                        model[..probe].chars().next_back(),
                        "char_before seam {probe}"
                    );
                    assert_eq!(b.clip_offset(o, Bias::Left), o, "seam clip identity {probe}");
                    let end = ((probe + 5).min(model.len())..=model.len())
                        .find(|&e| model.is_char_boundary(e))
                        .unwrap();
                    assert_eq!(b.slice(o..end as u32), &model[probe..end], "slice seam {probe}");
                } else {
                    let left = (0..=probe).rev().find(|&x| model.is_char_boundary(x)).unwrap();
                    let right = (probe..=model.len()).find(|&x| model.is_char_boundary(x)).unwrap();
                    assert_eq!(b.clip_offset(o, Bias::Left) as usize, left, "seam {probe}");
                    assert_eq!(b.clip_offset(o, Bias::Right) as usize, right, "seam {probe}");
                }
            }
        }
    }

    /// Seeded random splice walk: the rope and a `String` model must agree on
    /// the full text and line count after every edit (multibyte inserts
    /// included). Seeded with a multi-chunk corpus — and pinned to STAY
    /// multi-chunk — so the walk exercises the rope's cross-leaf edit path, not
    /// just single-chunk splices (which the other splice tests already cover).
    #[test]
    fn splice_random_walk_matches_string_model() {
        let mut model = String::new();
        for i in 0..400 {
            model.push_str(&format!("seed line {i}: ää🦀 with some text\n"));
        }
        let mut b = buf(&model);
        let mut state = 0x2545F49_u64; // deterministic LCG
        let mut rand = move |bound: usize| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((state >> 33) as usize) % bound.max(1)
        };
        let inserts =
            ["x", "hello", "\n", "ab\ncd", "äöü", "🦀", "fn main() {}\n", "", "日本語", "\n\n\n"];
        let mut multi_chunk_steps = 0;
        for step in 0..600 {
            // Random boundary-snapped range in the current text.
            let mut s = rand(model.len() + 1);
            while !model.is_char_boundary(s) {
                s -= 1;
            }
            let mut e = (s + rand(24)).min(model.len());
            while !model.is_char_boundary(e) {
                e -= 1;
            }
            let e = e.max(s);
            let ins = inserts[rand(inserts.len())];
            model.replace_range(s..e, ins);
            b.splice(s as u32..e as u32, ins);
            assert_eq!(b.text(), model.as_str(), "text diverged at step {step}");
            assert_eq!(
                b.line_count() as usize,
                memchr::memchr_iter(b'\n', model.as_bytes()).count() + 1,
                "line count diverged at step {step}"
            );
            if b.text.chunks().nth(1).is_some() {
                multi_chunk_steps += 1;
            }
        }
        // The guard that keeps this test honest: if a future edit shrinks the
        // corpus (or deletes outpace inserts), the walk silently stops testing
        // the cross-leaf path — fail instead.
        assert!(
            multi_chunk_steps >= 550,
            "walk must stay multi-chunk (got {multi_chunk_steps}/600 steps)"
        );
    }

    /// `slice` clamps rather than panicking: inverted or past-end ranges yield
    /// empty or the available tail, never a panic, so callers can pass loosely
    /// computed ranges safely. Mirrored on `Snapshot`, which shares the helper.
    #[test]
    #[allow(clippy::reversed_empty_ranges)] // inverted ranges ARE the subject
    fn slice_clamps_inverted_and_past_end_ranges() {
        let b = buf("hello world");
        assert_eq!(b.slice(5..2), "");
        assert_eq!(b.slice(100..3), "");
        assert_eq!(b.slice(3..100), "lo world");
        assert_eq!(b.slice(100..200), "");
        assert_eq!(b.line(99), "");
        let snap = b.snapshot();
        assert_eq!(snap.slice(5..2), "");
        assert_eq!(snap.slice(3..100), "lo world");
        assert_eq!(snap.line(99), "");
    }

    /// A snapshot is isolated from later edits and carries the stamp it froze.
    /// (The O(1)-creation claim is wall-clock — pinned in `benches/perf.rs`.)
    #[test]
    fn snapshot_is_isolated_from_later_edits() {
        let mut b = buf("alpha\nbeta\n");
        b.bump_revision();
        let snap = b.snapshot();
        b.splice(0..5, "OMEGA");
        assert_eq!(snap.text(), "alpha\nbeta\n");
        assert_eq!(snap.line(0), "alpha");
        assert_eq!(snap.slice(6..10), "beta");
        assert_eq!(snap.line_count(), 3);
        assert_eq!(snap.len(), 11);
        assert!(!snap.is_empty());
        assert_eq!(snap.revision(), Revision(1));
        assert_eq!(snap.doc_id(), b.doc_id());
        assert_eq!(b.text(), "OMEGA\nbeta\n", "the buffer moved on");
    }

    /// Small (single-chunk) documents keep zero-copy reads — the common case
    /// returns a `Borrowed` `Cow` with no allocation.
    #[test]
    fn small_document_reads_are_borrowed() {
        let b = buf("let x = 1;\nlet y = 2;\n");
        assert!(matches!(b.text(), Cow::Borrowed(_)));
        assert!(matches!(b.slice(4..9), Cow::Borrowed(_)));
        assert!(matches!(b.line(1), Cow::Borrowed(_)));
    }
}
