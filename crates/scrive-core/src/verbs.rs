//! The editing verbs: type, backspace, delete, enter, paste, (out)dent, and
//! move / copy line.
//! Each turns the current selections into one atomic multi-range transaction,
//! then places a caret after each edit — so every verb is multi-cursor-correct
//! for free (one op per selection). The placement is handed to the commit
//! itself (`Document::edit_placing`), never assigned once the edit returns, so
//! history records it as the unit's after-selections and redo restores exactly
//! the carets the verb left.
//!
//! The one catch is deletion: a selection's delete can reach past its
//! neighbour (a tab-stop backspace, a word, a line block), so delete verbs
//! merge their ranges first (`merge_delete_ranges`) — the engine rejects
//! overlapping ops — and every verb commits through `verb_commit`, which turns
//! an overlap that slipped through into a debug panic instead of a keystroke
//! that silently does nothing.
//!
//! Two load-bearing semantics live here: **typing over a selection replaces
//! it** (one transaction, caret after), and **Enter's autoindent is the leading
//! whitespace truncated to the caret column** (never duplicated) — plus the two
//! brace rules (one unit deeper after a line-opening `{`; `{|}` splits onto
//! three lines), always computed for *inserted* text only, never re-indenting
//! existing lines, so autoindent cannot fight the user. Clipboard text is
//! normalized to LF on the way in; the LF→OS-flavor re-expansion for copy-out
//! lives in the widget layer.

use core::ops::Range;
use std::borrow::Cow;

use crate::autoclose;
use crate::buffer::Buffer;
use crate::coords::Point;
use crate::document::Document;
use crate::history::{GroupingHint, OpClass};
use crate::patch::Patch;
use crate::row_layout::tail_start_col;
use crate::selection::{Selection, SelectionSet};
use crate::transaction::{Committed, EditOp, TransactionError};

/// Spaces per indent level: the stop spacing Tab, Shift+Tab, the block
/// (out)dent and backspace's tab-stop delete use on a space-indented line (a
/// tab-indented line moves by one hard tab instead).
///
/// Deliberately a SEPARATE knob from [`crate::display_map::default_tab_size`]
/// (an *editing* width vs a *display* width — mainstream editors keep both).
/// They happen to share the value 4; if either ever changes, the widget's
/// indent guides key on the display knob and Tab/indent verbs on this one.
/// Space stops after a hard tab count from the tab, so they line up with the
/// drawn ones while the display width is a multiple of this one.
#[must_use]
pub fn default_indent_size() -> u32 {
    4
}

impl Document {
    /// Insert `ch` at every caret, replacing any non-empty selection. Merges
    /// into the current typing run for undo.
    pub fn type_char(&mut self, ch: char) {
        // Auto-close, applied uniformly across ALL carets (a single caret is
        // just the one-element case). Each helper handles the whole keystroke and
        // returns true; overtype takes precedence (a quote is opener AND closer).
        if self.try_overtype(ch) {
            return; // a closer typed over its own auto-inserted close, at every caret
        }
        if self.try_autoclose(ch) {
            return; // an opener at empty carets → insert the pair (per-caret guard)
        }
        if self.try_surround(ch) {
            return; // an opener over non-empty selections → surround each
        }
        // Plain insert (also the mixed / guard-failing / non-pair case). Replaces
        // every selection with `ch`; `edit_grouped` rebases and validates live
        // provenance, so typing inside a pair keeps overtype available.
        let mut s = [0u8; 4];
        let text: &str = ch.encode_utf8(&mut s);
        let ops = self.map_selections(|sel_start, sel_end, _| EditOp::new(sel_start..sel_end, text));
        self.run_edit(ops, GroupingHint::mergeable(OpClass::Type));
    }

    /// Whether pair-insert of `open` is allowed at an empty `caret` — the guard
    /// that keeps auto-close from firing where a closer would be unwelcome.
    fn can_autoclose(&self, open: char, caret: u32) -> bool {
        // The char after the caret must be whitespace, EOL, or a closer —
        // otherwise the pair would split a word or an existing token.
        match self.buffer.char_at(caret) {
            None => {}
            Some(c) if c.is_whitespace() || autoclose::is_closer(c) => {}
            Some(_) => return false,
        }
        if autoclose::is_quote(open) {
            // For a quote, the char before must not be a word char (else it is
            // an apostrophe or a suffix quote, not an opening string delimiter).
            if let Some(p) = self.buffer.char_before(caret) {
                if autoclose::is_word_char(p) {
                    return false;
                }
            }
            // An odd count of quotes on the line ⇒ this one closes an already
            // open string; insert it literally rather than pairing.
            let row = self.buffer.offset_to_point(caret).row;
            let quotes = self.buffer.line(row).chars().filter(|&c| c == '"').count();
            if quotes % 2 == 1 {
                return false;
            }
        }
        true
    }

    /// Overtype: if `ch` is a closer and EVERY selection is an empty caret
    /// sitting exactly on a live pair's matching close, skip over each (no
    /// insert) and consume all provenance. All-or-nothing, so a mixed set falls
    /// through. Returns whether it handled the keystroke.
    fn try_overtype(&mut self, ch: char) -> bool {
        if !autoclose::is_closer(ch) {
            return false;
        }
        let ranges = self.autoclose_ranges(); // ascending, disjoint ⇒ ends ascending
        let on_close = |caret: u32| {
            let i = ranges.partition_point(|r| r.end < caret);
            i < ranges.len() && ranges[i].end == caret
        };
        let sels = self.selections.all();
        let all = !ranges.is_empty()
            && sels.iter().all(|s| {
                s.is_empty() && self.buffer.char_at(s.head()) == Some(ch) && on_close(s.head())
            });
        if !all {
            return false;
        }
        let step = ch.len_utf8() as u32;
        let carets: Vec<u32> = sels.iter().map(|s| s.head() + step).collect();
        self.selections = SelectionSet::from_offsets(&carets);
        self.clear_autoclose(); // the pairs are consumed
        true
    }

    /// Auto-close: if `ch` is an opener and EVERY selection is an empty caret,
    /// insert `open+close` at each caret that passes the guards and a bare
    /// `open` at the rest (a per-caret decision, as mainstream editors make it),
    /// place each caret just after the opener, and record provenance for the
    /// paired ones. Returns whether it handled the keystroke (false if any
    /// selection is non-empty).
    fn try_autoclose(&mut self, ch: char) -> bool {
        let Some(close) = autoclose::opener_close(ch) else {
            return false;
        };
        if self.selections.all().iter().any(|s| !s.is_empty()) {
            return false; // a non-empty selection ⇒ surround/plain, not pair-insert
        }
        // Per-caret pair-or-plain decision, computed before the edit.
        let plan: Vec<bool> =
            self.selections.all().iter().map(|s| self.can_autoclose(ch, s.head())).collect();
        let mut pair_text = String::with_capacity(ch.len_utf8() + close.len_utf8());
        pair_text.push(ch);
        pair_text.push(close);
        let mut open_buf = [0u8; 4];
        let open_text: &str = ch.encode_utf8(&mut open_buf);
        let ops: Vec<EditOp> = self
            .selections
            .all()
            .iter()
            .zip(&plan)
            .map(|(s, &pair)| {
                let at = s.head();
                EditOp::new(at..at, if pair { pair_text.as_str() } else { open_text })
            })
            .collect();
        let hint = GroupingHint { op: OpClass::Type, seal_before: true, seal_after: false };
        // A caret just after each opener; provenance for the paired ones.
        let close_len = close.len_utf8() as u32;
        let mut pairs = Vec::new();
        let placed = self.edit_placing(ops, hint, |patch, _| {
            let edits = patch.edits();
            let mut carets = Vec::with_capacity(edits.len());
            for (e, &pair) in edits.iter().zip(&plan) {
                if pair {
                    let close_off = e.new.end - close_len; // start of the close char
                    carets.push(close_off); // caret between open and close
                    pairs.push((e.new.start, close_off)); // [open, close)
                } else {
                    carets.push(e.new.end); // after the bare opener
                }
            }
            Some(SelectionSet::from_offsets(&carets))
        });
        if !verb_commit(placed) {
            return true; // refused: no pair landed, so no provenance to record
        }
        if !pairs.is_empty() {
            // Supersede any prior provenance — e.g. an outer pair whose tracked
            // range grew to enclose this insertion (nested `(` inside `()`) — so
            // the live set stays disjoint. Nested pairs must not accumulate, or
            // overtype's ends-ascending binary search picks the wrong pair and a
            // closer inserts literally (`(()))`). One generation of provenance,
            // always.
            self.clear_autoclose();
            self.add_autoclose_pairs(pairs);
        }
        true
    }

    /// Surround: if `ch` is an opener and EVERY selection is non-empty, wrap
    /// each in the pair with the selection staying on the interior. Returns
    /// whether it handled the keystroke.
    fn try_surround(&mut self, ch: char) -> bool {
        let Some(close) = autoclose::opener_close(ch) else {
            return false;
        };
        let sels = self.selections.all();
        if sels.is_empty() || sels.iter().any(|s| s.is_empty()) {
            return false;
        }
        let (open_len, close_len) = (ch.len_utf8() as u32, close.len_utf8() as u32);
        // ONE replace op per selection (open + content + close). A single op per
        // selection keeps TOUCHING selections independent: two zero-width inserts
        // at the same offset (opener then closer) would be ambiguous to bias, so
        // a successor selection starting where its predecessor ends could swallow
        // the inserted opener. One replace has an unambiguous span instead.
        let ops: Vec<EditOp> = sels
            .iter()
            .map(|s| {
                let content = self.buffer.slice(s.start()..s.end());
                let mut wrapped = String::with_capacity(content.len() + open_len as usize + close_len as usize);
                wrapped.push(ch);
                wrapped.push_str(&content);
                wrapped.push(close);
                EditOp::new(s.start()..s.end(), &wrapped)
            })
            .collect();
        // Each selection stays on its interior, read straight from the edits.
        verb_commit(self.edit_placing(ops, GroupingHint::discrete(), |patch, _| {
            let ranges: Vec<(u32, u32)> =
                patch.edits().iter().map(|e| (e.new.start + open_len, e.new.end - close_len)).collect();
            Some(SelectionSet::from_ranges(&ranges, ranges.len() - 1))
        }));
        true
    }

    /// Insert `text` (e.g. a paste) at every caret, replacing selections. A
    /// discrete undo step; `\r\n|\r` is normalized to LF.
    pub fn insert_text(&mut self, text: &str) {
        let ops = self.map_selections(|sel_start, sel_end, _| {
            EditOp::new(sel_start..sel_end, text)
        });
        self.run_edit(ops, GroupingHint::discrete());
    }

    /// The text Copy/Cut should put on the clipboard, with the `is_entire_line`
    /// flag the paste side honors. Each non-empty selection contributes its
    /// text, joined by newlines. When **every** selection is an empty caret,
    /// each caret contributes its ENTIRE line — the one `Document::line_unit`
    /// rule — including the newline (a final line without a terminator still
    /// exports one so the paste splices cleanly), duplicate lines deduplicated,
    /// and the flag is `true`. LF-only; the OS-flavor re-expansion and the side
    /// table live in the widget.
    #[must_use]
    pub fn clipboard_payload(&self) -> (String, bool) {
        let sels = self.selections.all();
        if sels.iter().all(Selection::is_empty) {
            let mut units: Vec<(u32, u32)> = sels.iter().map(|s| self.line_unit(s.head())).collect();
            units.dedup();
            let lines: String = units
                .iter()
                .map(|&(start, end)| {
                    let slice = self.buffer.slice(start..end);
                    if slice.ends_with('\n') {
                        slice.into_owned()
                    } else {
                        format!("{slice}\n")
                    }
                })
                .collect();
            return (lines, true);
        }
        let joined = sels
            .iter()
            .filter(|s| !s.is_empty())
            .map(|s| self.buffer.slice(s.start()..s.end()))
            .collect::<Vec<_>>()
            .join("\n");
        (joined, false)
    }

    /// Cut's edit half: cut is copy-then-delete, so this deletes exactly what
    /// [`Document::clipboard_payload`] exported — non-empty selections; or,
    /// when EVERY selection is an empty caret (the payload's whole-line mode,
    /// same gate), each caret's `line_block`. In a
    /// mixed set the empty carets reached the clipboard as nothing, so they
    /// delete nothing — cut never destroys text that was not copied. The
    /// final-line block takes its PRECEDING newline while the payload
    /// synthesizes a trailing one: the paste splices a whole line either way,
    /// and no empty tail line is left behind. Overlapping ranges (two carets on
    /// one line) merge; one discrete transaction, caret after each deletion.
    pub fn cut(&mut self) {
        let all_empty = self.selections.all().iter().all(Selection::is_empty);
        let ranges: Vec<Range<u32>> = self
            .selections
            .all()
            .iter()
            .filter_map(|sel| {
                if sel.is_empty() {
                    all_empty.then(|| self.line_block(sel))
                } else {
                    Some(sel.start()..sel.end())
                }
            })
            .collect();
        let ops = merge_delete_ranges(ranges.into_iter());
        self.run_edit(ops, GroupingHint::discrete());
    }

    /// Paste. Plain: replace each selection, caret after (a discrete step).
    /// `entire_line` at all-empty carets: insert before each caret's line start
    /// with **no caret placement** — the carets rebase past the insertion and
    /// stay put on their own lines, matching mainstream editors.
    pub fn paste(&mut self, text: &str, entire_line: bool) {
        if entire_line && self.selections.all().iter().all(Selection::is_empty) {
            let mut starts: Vec<u32> =
                self.selections.all().iter().map(|s| self.line_unit(s.head()).0).collect();
            starts.dedup();
            let ops = starts.into_iter().map(|at| EditOp::new(at..at, text)).collect();
            // `edit_grouped` rebases the selections through the patch — never
            // `run_edit`, whose caret-after would yank the caret off its line.
            verb_commit(self.edit_grouped(ops, GroupingHint::discrete()));
            return;
        }
        self.insert_text(text);
    }

    /// Backspace: delete the selection, or the character before an empty caret
    /// — one indent level when the caret sits in leading whitespace: a hard
    /// tab, or spaces back to the previous stop (the unit Shift+Tab removes).
    pub fn backspace(&mut self) {
        // Pair-backspace: every empty caret strictly between a live *empty*
        // pair deletes both chars (`open + 1 == close` ⇒ nothing typed between
        // them). All-or-nothing (symmetric with overtype); a mixed set falls
        // through to the ordinary per-caret backspace below.
        let ranges = self.autoclose_ranges();
        if !ranges.is_empty() {
            let empty_pair_at = |caret: u32| -> Option<Range<u32>> {
                let i = ranges.partition_point(|r| r.end < caret);
                ranges.get(i).filter(|r| r.end == caret && r.start + 1 == r.end).cloned()
            };
            let pairs: Option<Vec<Range<u32>>> = self
                .selections
                .all()
                .iter()
                .map(|s| s.is_empty().then(|| empty_pair_at(s.head())).flatten())
                .collect();
            if let Some(pairs) = pairs {
                // Distinct carets sit in distinct (disjoint) pairs, so these
                // only ever touch — merged anyway, like every delete verb.
                let ops = merge_delete_ranges(pairs.iter().map(|r| {
                    let close_len = self.buffer.char_at(r.end).map_or(1, |c| c.len_utf8() as u32);
                    r.start..r.end + close_len
                }));
                self.clear_autoclose();
                let hint =
                    GroupingHint { op: OpClass::Delete, seal_before: true, seal_after: true };
                self.run_edit(ops, hint);
                return;
            }
        }
        let buffer = &self.buffer;
        let ranges = self.selections.all().iter().filter_map(|sel| {
            if !sel.is_empty() {
                return Some(sel.start()..sel.end());
            }
            let caret = sel.head();
            if caret == 0 {
                return None;
            }
            Some(backspace_target(buffer, caret)..caret)
        });
        // A tab-stop delete reaches back past its own caret's neighbourhood —
        // carets at cols 3 and 4 both delete to stop 0 (`0..3`, `0..4`), and a
        // caret deletes straight through a selection inside its indent run —
        // so the ranges can OVERLAP and must merge before the transaction.
        let ops = merge_delete_ranges(ranges);
        self.run_edit(ops, GroupingHint::mergeable(OpClass::Delete));
    }

    /// Forward delete: delete the selection, or the character after an empty
    /// caret (merging the next line when at end of line).
    pub fn delete_forward(&mut self) {
        let len = self.buffer.len();
        let buffer = &self.buffer;
        let ranges = self.selections.all().iter().filter_map(|sel| {
            if !sel.is_empty() {
                return Some(sel.start()..sel.end());
            }
            let caret = sel.head();
            if caret >= len {
                return None;
            }
            let next = buffer.char_at(caret).expect("caret < len");
            Some(caret..caret + next.len_utf8() as u32)
        });
        // One char forward from a caret can only TOUCH the next selection (a
        // caret touching a selection merged into it), never overlap — merged
        // anyway, so every delete verb builds its ops the same disjoint way.
        let ops = merge_delete_ranges(ranges);
        self.run_edit(ops, GroupingHint::mergeable(OpClass::Delete));
    }

    /// Delete the word before each caret — Ctrl+Backspace. A non-empty
    /// selection deletes as-is; an empty caret deletes back to the previous word
    /// start, or *only* the newline at column 0 (never newline + previous word).
    /// A discrete undo step (word-delete seals the run on both sides).
    ///
    /// Two carets inside one word yield *overlapping* delete ranges, which the
    /// transaction engine rejects — so the ranges are merged first (the shared
    /// span is deleted once and the carets collapse via the selection merge
    /// rule), as for every delete verb.
    pub fn delete_word_back(&mut self) {
        let buffer = &self.buffer;
        let ranges = self.selections.all().iter().filter_map(|sel| {
            if !sel.is_empty() {
                return Some(sel.start()..sel.end());
            }
            let caret = sel.head();
            let start = crate::movement::word_delete_left(buffer, caret);
            (start < caret).then_some(start..caret)
        });
        let ops = merge_delete_ranges(ranges);
        self.run_edit(ops, GroupingHint { op: OpClass::Delete, seal_before: true, seal_after: true });
    }

    /// Delete the word after each caret — Ctrl+Delete, the
    /// [`delete_word_back`](Self::delete_word_back) mirror (overlapping ranges
    /// merged the same way).
    pub fn delete_word_forward(&mut self) {
        let len = self.buffer.len();
        let buffer = &self.buffer;
        let ranges = self.selections.all().iter().filter_map(|sel| {
            if !sel.is_empty() {
                return Some(sel.start()..sel.end());
            }
            let caret = sel.head();
            if caret >= len {
                return None;
            }
            let end = crate::movement::word_delete_right(buffer, caret);
            (end > caret).then_some(caret..end)
        });
        let ops = merge_delete_ranges(ranges);
        self.run_edit(ops, GroupingHint { op: OpClass::Delete, seal_before: true, seal_after: true });
    }

    /// Enter: split the line at every caret, carrying the current line's
    /// leading indentation (truncated to the caret column — never duplicated),
    /// plus the two brace rules (both computed from raw text at insert time —
    /// existing lines are never re-indented, so autoindent cannot fight the
    /// user):
    ///
    /// - **(a)** the current line's text left of the caret, right-trimmed,
    ///   ends with `{` → the new line gets one extra `indent_unit`;
    /// - **(b)** the caret sits exactly between a pair (`{|}`) → insert two
    ///   lines (`\n indent+unit \n indent`), closer dedented onto its own
    ///   line, caret at the end of the middle line (one line up from the
    ///   inserted text's end, indented one unit deeper than the closer).
    ///
    /// Its own undo step (seals before; typing after merges into the run).
    pub fn enter(&mut self) {
        let buffer = &self.buffer;
        // Per-op distance from the inserted text's end BACK to the caret —
        // 0 except for rule (b), where the caret lands one line up.
        let mut backs: Vec<u32> = Vec::new();
        let ops: Vec<_> = self
            .selections
            .all()
            .iter()
            .map(|sel| {
                let indent = enter_indent(buffer, sel.start());
                let opens = line_opens_block(buffer, sel.start());
                let between = opens
                    && buffer.char_before(sel.start()) == Some('{')
                    && buffer.char_at(sel.end()) == Some('}');
                let mut t = String::with_capacity(2 * (1 + indent.len()) + 4);
                t.push('\n');
                t.push_str(&indent);
                if opens {
                    t.push_str(&indent_unit(&indent));
                }
                if between {
                    t.push('\n');
                    t.push_str(&indent);
                    backs.push(1 + indent.len() as u32); // back over "\n" + indent
                } else {
                    backs.push(0);
                }
                EditOp::new(sel.start()..sel.end(), t)
            })
            .collect();
        let hint = GroupingHint { op: OpClass::Type, seal_before: true, seal_after: false };
        // `run_edit`'s caret-after, minus each op's rule-(b) pull-back.
        verb_commit(self.edit_placing(ops, hint, |patch, _| Some(carets_pulled_back(patch, &backs))));
    }

    /// Tab: at an empty caret insert one indent unit of the caret's line — a
    /// hard tab on a tab-indented line, else spaces to the next indent stop;
    /// over a single-line selection type that unit over it; over a multi-line
    /// selection indent every spanned line one level, each by its own unit
    /// (the indent-unit rule Enter and Shift+Tab share). The unit goes at the
    /// caret, or in the selection's place, except from a caret or selection
    /// ending ahead of a mixed indent's trailing run (the `··` of `··\t`, the
    /// `\t` of `\t··`): there the selection is deleted and the unit goes where
    /// block indent puts it, so no Tab interleaves kinds within a line's
    /// indent. VS Code types the unit in the selection's place even there — a
    /// `·` selected in `··\t` becomes `\t·\t` — which is exactly the
    /// interleave; deleting the selection still replaces it with one level,
    /// just not between chars of the other kind.
    pub fn tab(&mut self) {
        if self.selection_spans_multiple_lines() {
            // Multi-line selection → indent the whole block, selection preserved.
            self.indent_lines(true);
        } else {
            // Carets and single-line selections → insert / replace-with one
            // unit. A single-line selection *types over* (as mainstream editors
            // do): it does NOT indent the whole line.
            let buffer = &self.buffer;
            let sels = self.selections.all();
            let mut ops = Vec::with_capacity(sels.len());
            for sel in sels {
                let (start, end) = (sel.start(), sel.end());
                let p = buffer.offset_to_point(start);
                let line = buffer.line(p.row);
                let ws = leading_ws(&line);
                // The LINE's kind (its whole leading whitespace), not the
                // prefix's: a caret at column 0 of `\tfoo` must add a tab,
                // where the empty prefix alone would say spaces.
                let unit = IndentUnit::of(ws);
                // A caret or selection ENDING ahead of the trailing run of the
                // unit's char would drop the unit between chars of the OTHER
                // kind (`\t··\t`). It takes block indent's insertion instead —
                // the run's start, one level padded as block indent pads it —
                // and a selection is still typed over: deleted where it lies.
                // Ending in or at an edge of the run (or past the indent), the
                // unit's own char joins its own run: in the selection's place,
                // padded from there (VS Code's caret Tab).
                let end_col = p.col + (end - start); // single-line: same row
                if end_col < level_start(ws, unit) {
                    if start < end {
                        ops.push(EditOp::delete(start..end));
                    }
                    let (at, text) = indent_insert(ws);
                    let mut at = start - p.col + at;
                    // A later selection typing over the run's start (it starts
                    // ahead of the run and ends in it) holds `at` strictly
                    // inside — an overlap. The run then starts where that
                    // selection does, so the unit goes there, touching its op.
                    // Sorted and disjoint ⇒ at most one selection holds `at`.
                    let over = sels.get(sels.partition_point(|o| o.end() <= at));
                    if let Some(over) = over.filter(|o| o.start() < at) {
                        at = over.start();
                    }
                    ops.push(EditOp::insert(at, text));
                } else {
                    ops.push(EditOp::new(start..end, unit.text(&line[..p.col as usize])));
                }
            }
            // A caret just past each typed unit: every unit is non-empty text,
            // so the edits that insert are exactly the units — the type-over
            // deletes (empty post-edit ranges) carry no caret.
            verb_commit(self.edit_placing(ops, GroupingHint::discrete(), |patch, _| {
                let carets: Vec<u32> =
                    patch.edits().iter().filter(|e| !e.new.is_empty()).map(|e| e.new.end).collect();
                Some(SelectionSet::from_offsets(&carets))
            }));
        }
    }

    /// Shift+Tab: outdent every spanned line by one indent level — its last
    /// hard tab, or its trailing spaces back to the previous stop. Unlike Tab
    /// (which types over a single-line selection), Shift+Tab always outdents —
    /// carets, single-line selections, and multi-line selections alike; the
    /// selection is preserved.
    pub fn outdent(&mut self) {
        self.indent_lines(false);
    }

    /// Delete each selection's whole-line block (Ctrl+Shift+K) — the one
    /// `line_block` rule per selection. Overlapping
    /// blocks (two carets on one line) merge; one discrete step, caret after.
    pub fn delete_line(&mut self) {
        let ranges: Vec<Range<u32>> =
            self.selections.all().iter().map(|sel| self.line_block(sel)).collect();
        let ops = merge_delete_ranges(ranges.into_iter());
        self.run_edit(ops, GroupingHint::discrete());
    }

    /// THE whole-line **deletion block** for a selection: every spanned line
    /// (a selection ending exactly at a line start does not span it — the
    /// [`last_spanned_offset`] rule shared with `line_runs`) plus ONE
    /// bounding newline: the trailing one, or, when the block reaches the
    /// buffer's final line, the PRECEDING one — so no empty tail line is left
    /// behind. Shared by [`Document::delete_line`] and the
    /// whole-line [`Document::cut`]. (Copy's per-line text is the different
    /// `Document::line_unit` fact — content never takes a preceding `\n`.)
    fn line_block(&self, sel: &Selection) -> Range<u32> {
        let buffer = &self.buffer;
        let first = buffer.offset_to_point(sel.start()).row;
        let last = buffer.offset_to_point(last_spanned_offset(sel)).row;
        let start = buffer.point_to_offset(Point::new(first, 0));
        if last + 1 < buffer.line_count() {
            start..buffer.point_to_offset(Point::new(last + 1, 0))
        } else {
            start.saturating_sub(1)..buffer.len()
        }
    }

    /// Open a fresh, indent-carrying line below (`down`, Ctrl+Enter) or above
    /// (Ctrl+Shift+Enter) each caret's line — without splitting the line the
    /// caret sat on — and land the caret at the new line's end. A line below
    /// a block-opening `{` line gains one indent unit, like Enter at the line's
    /// end. One discrete transaction.
    pub fn insert_line(&mut self, down: bool) {
        let buffer = &self.buffer;
        let mut backs: Vec<u32> = Vec::new();
        let mut ops: Vec<EditOp> = Vec::new();
        for sel in self.selections.all() {
            let row = buffer.offset_to_point(sel.head()).row;
            let line = buffer.line(row);
            let indent = leading_ws(&line);
            if down {
                let at = buffer.point_to_offset(Point::new(row, buffer.line_len(row)));
                if ops.last().is_some_and(|op| op.range.start == at) {
                    continue; // second caret on the same line — one new line
                }
                let mut t = String::with_capacity(1 + indent.len() + 4);
                t.push('\n');
                t.push_str(indent);
                if line_opens_block(buffer, at) {
                    t.push_str(&indent_unit(indent));
                }
                ops.push(EditOp::new(at..at, t));
                backs.push(0);
            } else {
                let at = buffer.point_to_offset(Point::new(row, 0));
                if ops.last().is_some_and(|op| op.range.start == at) {
                    continue;
                }
                // The new line's indent, then the newline that pushes the
                // current line down — caret pulls back over the `\n`.
                ops.push(EditOp::new(at..at, format!("{indent}\n")));
                backs.push(1);
            }
        }
        verb_commit(self.edit_placing(ops, GroupingHint::discrete(), |patch, _| {
            Some(carets_pulled_back(patch, &backs))
        }));
    }

    /// Move every selected line-block up or down one line (Alt+↑/↓). A block is
    /// a run of whole lines the selections span (`line_runs`: a
    /// whole-line selection ending at column 0 does not span that row), runs on
    /// consecutive rows merged — each block swaps with its neighbour line, which
    /// must therefore belong to no other block. Every selection rides its moved
    /// text, id and direction kept. A block at the document edge — or, moving
    /// down, against the trailing empty line — stays put while the others move.
    /// One undo step; never re-indents.
    pub fn move_line(&mut self, down: bool) {
        let line_count = self.buffer.line_count();
        let ends_nl = self.buffer.char_before(self.buffer.len()) == Some('\n');
        // The last row a real line occupies (the trailing empty line, if any, is
        // not movable — moving into it would drop the terminator).
        let last_movable = if ends_nl { line_count.saturating_sub(2) } else { line_count - 1 };
        // Each swap rewrites its region (block + neighbour) with text of the same
        // length, so offsets outside the moved regions never shift: every
        // destination is the pre-edit one, with nothing to accumulate.
        self.relocate_line_blocks(true, |buffer, run, _, block| {
            let (r0, r1) = (run.first, run.last);
            if (down && r1 >= last_movable) || (!down && r0 == 0) {
                return None; // at the edge: stays put while the others move
            }
            let (a, b) = if down { (r0, r1 + 1) } else { (r0 - 1, r1) };
            let has_nl = b + 1 < line_count; // row `b` carries a trailing newline
            let region_start = buffer.point_to_offset(Point::new(a, 0));
            let region_end =
                if has_nl { buffer.point_to_offset(Point::new(b + 1, 0)) } else { buffer.len() };
            let neighbour = buffer.line(if down { b } else { a });
            let (mut text, dest) = if down {
                // The line below moves to the front; the block shifts down past
                // it and its `\n`.
                (format!("{neighbour}\n{block}"), region_start + neighbour.len() as u32 + 1)
            } else {
                // The line above moves to the back; the block takes its place.
                (format!("{block}\n{neighbour}"), region_start)
            };
            if has_nl {
                text.push('\n');
            }
            // A whole-line selection's end sits past the block's `\n`; moved onto
            // the buffer's final line the block has no `\n`, so that end clamps to
            // the region (buffer) end.
            Some(Relocation { op: EditOp::new(region_start..region_end, text), dest, max: Some(region_end) })
        });
    }

    /// Duplicate every selected line-block above (`down = false`) or below
    /// (`down = true`) — Shift+Alt+↑/↓. A block is a run of whole lines the
    /// selections span (`line_runs`), runs that share a row merged
    /// so no line is copied twice; runs on consecutive rows stay apart, so
    /// carets on consecutive lines each copy their own line (VS Code). Every
    /// selection lands on its block's copy — the lower one copying down, the
    /// upper one copying up — id and direction kept. One undo step.
    pub fn copy_line(&mut self, down: bool) {
        let line_count = self.buffer.line_count();
        // Bytes the earlier copies insert. The runs — and so the insertion
        // points — ascend, so everything after an insertion, every later copy
        // included, lands that much further on post-edit.
        let mut grown = 0;
        self.relocate_line_blocks(false, |buffer, run, block_start, block| {
            // Where the copy is inserted, its text, and where its block starts.
            let (at, text, copy_start) = if !down {
                // Above: the copy takes the block's place and pushes it down.
                (block_start, format!("{block}\n"), block_start)
            } else if run.last + 1 < line_count {
                let at = buffer.point_to_offset(Point::new(run.last + 1, 0));
                (at, format!("{block}\n"), at)
            } else {
                // Duplicating the final line (no trailing newline): prepend one.
                let at = buffer.len();
                (at, format!("\n{block}"), at + 1)
            };
            let dest = copy_start + grown;
            grown += text.len() as u32;
            // The copy is the block verbatim, so a whole-line selection's end,
            // past the block's `\n`, lands past the copy's: nothing clamps.
            Some(Relocation { op: EditOp::insert(at, text), dest, max: None })
        });
    }

    /// Toggle the line comment on every line the selections span (Ctrl+/).
    /// If ANY spanned non-blank line is uncommented, comment them all —
    /// `prefix + " "` inserted at the block's minimum indent column,
    /// so the markers align — otherwise strip each line's prefix (and one
    /// following space). Blank lines are skipped, unless EVERY spanned line is
    /// blank (the start-a-comment-here case, which appends the prefix). One
    /// transaction; the selections rebase through it and survive. A no-op until
    /// the app injects a prefix ([`Document::set_line_comment`]) — the core
    /// knows no language.
    pub fn toggle_line_comment(&mut self) {
        let Some(prefix) = self.line_comment().map(str::to_owned) else { return };
        let rows = self.spanned_rows();
        let ops: Vec<EditOp> = {
            let buffer = &self.buffer;
            let lines: Vec<(u32, Cow<str>)> = rows.iter().map(|&r| (r, buffer.line(r))).collect();
            let non_blank: Vec<(u32, &str)> = lines
                .iter()
                .map(|(r, l)| (*r, l.as_ref()))
                .filter(|(_, l)| !l.trim().is_empty())
                .collect();
            if non_blank.is_empty() {
                // Every spanned line is blank: start a comment on each.
                lines
                    .iter()
                    .map(|&(row, _)| {
                        let at = buffer.point_to_offset(Point::new(row, buffer.line_len(row)));
                        EditOp::new(at..at, format!("{prefix} "))
                    })
                    .collect()
            } else if non_blank.iter().any(|(_, l)| !l.trim_start().starts_with(&prefix)) {
                // Comment: markers aligned at the block's minimum indent.
                let indent = |l: &str| tail_start_col(l);
                let min_indent = non_blank.iter().map(|(_, l)| indent(l)).min().unwrap_or(0);
                non_blank
                    .iter()
                    .map(|&(row, _)| {
                        let at = buffer.point_to_offset(Point::new(row, min_indent));
                        EditOp::new(at..at, format!("{prefix} "))
                    })
                    .collect()
            } else {
                // Uncomment: strip the prefix plus one following space.
                non_blank
                    .iter()
                    .map(|&(row, line)| {
                        let ws = tail_start_col(line) as usize;
                        let start = buffer.point_to_offset(Point::new(row, ws as u32));
                        let mut end = start + prefix.len() as u32;
                        if line[ws + prefix.len()..].starts_with(' ') {
                            end += 1;
                        }
                        EditOp::delete(start..end)
                    })
                    .collect()
            }
        };
        // `edit_grouped` rebases the selections through the patch, so the
        // selection (or caret) survives the toggle in place.
        verb_commit(self.edit_grouped(ops, GroupingHint::discrete()));
    }

    /// Whether any selection spans more than one line — the trigger for block
    /// (out)dent (vs. type-over / caret indent).
    fn selection_spans_multiple_lines(&self) -> bool {
        self.selections.all().iter().any(|s| {
            self.buffer.offset_to_point(s.start()).row != self.buffer.offset_to_point(s.end()).row
        })
    }

    /// Indent (`deeper`) or outdent every line the selections span by one
    /// level of ITS OWN unit ([`indent_insert`] / [`outdent_range`] of its
    /// leading whitespace — a block mixing tab- and space-indented lines moves
    /// each in kind), as one transaction. The **selection is preserved** — it
    /// rebases through the edit (its endpoints shift with the indent), as
    /// mainstream editors do, rather than collapsing to one caret per line.
    fn indent_lines(&mut self, deeper: bool) {
        let rows = self.spanned_rows();
        let buffer = &self.buffer;
        let mut ops = Vec::new();
        for row in rows {
            let line_start = buffer.point_to_offset(Point::new(row, 0));
            let line = buffer.line(row);
            let ws = leading_ws(&line);
            if deeper {
                let (at, text) = indent_insert(ws);
                ops.push(EditOp::insert(line_start + at, text));
            } else {
                let r = outdent_range(ws);
                if !r.is_empty() {
                    ops.push(EditOp::delete(line_start + r.start..line_start + r.end));
                }
            }
        }
        // `edit_grouped` rebases the selections through the patch (a range's
        // start biases Left and its end biases Right), so the selection survives
        // the (out)dent as a selection — not `run_edit`'s one-caret-per-op, which
        // would collapse it to a caret.
        verb_commit(self.edit_grouped(ops, GroupingHint::discrete()));
    }

    /// The distinct rows any selection touches, ascending (for block (out)dent
    /// and toggle-comment): the [`line_runs`](Self::line_runs) flattened — runs
    /// sharing a row are already merged, so no row repeats.
    fn spanned_rows(&self) -> Vec<u32> {
        self.line_runs(false).into_iter().flat_map(|run| run.first..=run.last).collect()
    }

    /// The whole-line **runs** the selections span — the unit move-line and
    /// copy-line act on. Each selection spans `first..=last` rows by the
    /// [`last_spanned_offset`] rule (a whole-line selection ending at column 0
    /// does not span that row); runs that share a row merge, and with
    /// `merge_adjacent` so do runs on consecutive rows. Every selection lies in
    /// exactly one run, and the set's sorted, disjoint order makes each run's
    /// selections a contiguous index range of [`SelectionSet::all`].
    fn line_runs(&self, merge_adjacent: bool) -> Vec<LineRun> {
        let buffer = &self.buffer;
        let reach = u32::from(merge_adjacent);
        let mut runs: Vec<LineRun> = Vec::new();
        for (i, sel) in self.selections.all().iter().enumerate() {
            let first = buffer.offset_to_point(sel.start()).row;
            let last = buffer.offset_to_point(last_spanned_offset(sel)).row;
            match runs.last_mut() {
                // Sorted and disjoint ⇒ `first` and `last` never decrease, so
                // only the newest run can absorb this selection.
                Some(run) if first <= run.last + reach => {
                    run.last = run.last.max(last);
                    run.sels.end = i + 1;
                }
                _ => runs.push(LineRun { first, last, sels: i..i + 1 }),
            }
        }
        runs
    }

    /// The shared shape of [`move_line`](Self::move_line) and
    /// [`copy_line`](Self::copy_line). Per line run
    /// ([`line_runs`](Self::line_runs)`(merge_adjacent)`), `plan` sees the run,
    /// its block's start offset and its text (rows `first..=last`, without the
    /// last row's `\n`) and returns the run's [`Relocation`] — or `None` to leave
    /// the run where it is. Destinations are pre-edit arithmetic, so `plan`
    /// accounts for any shift earlier runs' edits cause.
    ///
    /// Every selection of a relocated run rides its block: the block moves or is
    /// copied verbatim, so each offset keeps its distance from the block start
    /// (clamped to the relocation's `max`, if any); a run left in place keeps
    /// its spans. Ids and directions survive (`moved_to`). One discrete undo
    /// step, the placement inside the commit so redo restores it; no run
    /// relocated ⇒ no edit at all.
    fn relocate_line_blocks(
        &mut self,
        merge_adjacent: bool,
        mut plan: impl FnMut(&Buffer, &LineRun, u32, &str) -> Option<Relocation>,
    ) {
        let mut spans: Vec<(u32, u32)> =
            self.selections.all().iter().map(|s| (s.tail(), s.head())).collect();
        let mut ops = Vec::new();
        for run in self.line_runs(merge_adjacent) {
            let block_start = self.buffer.point_to_offset(Point::new(run.first, 0));
            let block_end =
                self.buffer.point_to_offset(Point::new(run.last, self.buffer.line_len(run.last)));
            let block = self.buffer.slice(block_start..block_end);
            let Some(Relocation { op, dest, max }) = plan(&self.buffer, &run, block_start, &block)
            else {
                continue;
            };
            let to = |o: u32| {
                let landed = dest + (o - block_start);
                max.map_or(landed, |max| landed.min(max))
            };
            for (tail, head) in &mut spans[run.sels] {
                (*tail, *head) = (to(*tail), to(*head));
            }
            ops.push(op);
        }
        if ops.is_empty() {
            return;
        }
        let placed = self.selections.moved_to(&spans);
        verb_commit(self.edit_placing(ops, GroupingHint::discrete(), |_, _| Some(placed)));
        // Drop provenance through the one owner so the `AutoClosePair`
        // decoration is removed with it — the store is the single owner of the
        // pair, with nothing else to keep in sync.
        self.clear_autoclose();
    }

    /// Build one op per selection via `f(start, end, buffer)`.
    fn map_selections(
        &self,
        mut f: impl FnMut(u32, u32, &Buffer) -> EditOp,
    ) -> Vec<EditOp> {
        self.selections
            .all()
            .iter()
            .map(|sel| f(sel.start(), sel.end(), &self.buffer))
            .collect()
    }

    /// Apply verb ops as one transaction and drop a caret after each edit.
    /// The ops must be disjoint ([`verb_commit`] holds the verb to it).
    fn run_edit(&mut self, ops: Vec<EditOp>, hint: GroupingHint) {
        // A caret after each edit: `new.end` is after inserted text, or the
        // deletion point for a pure delete.
        verb_commit(self.edit_placing(ops, hint, |patch, _| {
            let carets: Vec<u32> = patch.edits().iter().map(|e| e.new.end).collect();
            Some(SelectionSet::from_offsets(&carets))
        }));
    }
}

/// The verbs' one verdict on a commit — no verb drops a [`TransactionError`]
/// with a bare `let _ =`. Every verb builds DISJOINT ops by construction (one
/// per selection, or overlapping deletes merged by [`merge_delete_ranges`]),
/// so an [`Overlap`](TransactionError::Overlap) reaching here is a verb bug —
/// a keystroke that would silently do nothing (two carets in one indent run
/// backspacing to the same tab stop). Debug and test builds panic on it,
/// which turns every verb test, and the seeded multi-caret walks, into an
/// overlap oracle. [`WouldOverflow`](TransactionError::WouldOverflow) is the
/// one legitimate refusal — a paste past the u32 offset space, rejected
/// atomically with the document untouched — and the verbs' `()` surface has
/// no channel to report it, so it alone passes as a no-op. Returns whether the
/// transaction was applied.
pub(crate) fn verb_commit(result: Result<Committed, TransactionError>) -> bool {
    match result {
        Ok(_) => true,
        Err(e) => {
            debug_assert!(
                !matches!(e, TransactionError::Overlap { .. }),
                "a verb built overlapping ops (merge them first): {e}"
            );
            false
        }
    }
}

/// A maximal block of whole lines the selections span, with the selections
/// that span it ([`Document::line_runs`]).
struct LineRun {
    /// The run's first row.
    first: u32,
    /// The run's last row (inclusive).
    last: u32,
    /// The run's selections, as indices into [`SelectionSet::all`].
    sels: Range<usize>,
}

/// Where one line run's block goes — [`Document::relocate_line_blocks`]'s
/// per-run plan.
struct Relocation {
    /// The edit that moves or copies the block.
    op: EditOp,
    /// Where the block's first byte lands post-edit.
    dest: u32,
    /// The furthest a carried offset may land, or `None` when nothing clamps.
    max: Option<u32>,
}

/// One caret per edit, `backs[i]` bytes before edit `i`'s inserted end — the
/// Enter / insert-line placement (0 = just after the text, as `run_edit`; >0
/// pulls the caret back onto a line the insertion opened). `backs` is
/// parallel to the ops, which the engine keeps one edit each, in order.
fn carets_pulled_back(patch: &Patch, backs: &[u32]) -> SelectionSet {
    let carets: Vec<u32> = patch.edits().iter().zip(backs).map(|(e, back)| e.new.end - back).collect();
    SelectionSet::from_offsets(&carets)
}

/// The last offset a selection actually spans: one back from the exclusive
/// `end`, so a selection ending exactly at a line start does not span that
/// line. THE end-exclusivity rule, shared by the row enumeration
/// (`line_runs`, and through it `spanned_rows`) and the whole-line deletion
/// block (`line_block`).
fn last_spanned_offset(sel: &Selection) -> u32 {
    if sel.end() > sel.start() {
        sel.end() - 1
    } else {
        sel.end()
    }
}

/// Merge overlapping delete ranges into a minimal non-overlapping set of delete
/// ops, because the transaction engine rejects overlapping ops. EVERY
/// multi-caret delete verb builds its ops here, since a per-selection delete
/// can reach past its neighbour: two carets in one word (word-delete), two
/// carets on one line (line-delete, cut), two carets in one indent run backing
/// to the same tab stop, or a caret's stop reaching through a selection
/// (backspace). Merging deletes the shared region once — the union of what each
/// selection would delete alone — and `run_edit`'s one caret per merged edit
/// collapses the carets onto its start. Touching ranges (`a.end == b.start`)
/// stay separate; the engine accepts those, and their carets coincide and merge.
fn merge_delete_ranges(ranges: impl Iterator<Item = Range<u32>>) -> Vec<EditOp> {
    let mut sorted: Vec<Range<u32>> = ranges.collect();
    sorted.sort_by_key(|r| r.start);
    let mut merged: Vec<Range<u32>> = Vec::new();
    for r in sorted {
        match merged.last_mut() {
            Some(last) if r.start < last.end => last.end = last.end.max(r.end),
            _ => merged.push(r),
        }
    }
    merged.into_iter().map(EditOp::delete).collect()
}

/// The byte offset backspace should delete back to from an empty caret: one
/// indent level ([`outdent_range`] of the whitespace before the caret) if the
/// caret is within the line's leading whitespace, else one character.
fn backspace_target(buffer: &Buffer, caret: u32) -> u32 {
    let p = buffer.offset_to_point(caret);
    let line = buffer.line(p.row);
    if p.col > 0 && p.col <= tail_start_col(&line) {
        // The whitespace before the caret, read as an indent — the leading
        // whitespace truncated to the caret, exactly as Enter copies it.
        // Its level is removed from the END of its trailing run (the run is
        // one repeated char, so where within it is moot for the text), i.e.
        // just before the caret.
        let level = outdent_range(&line[..p.col as usize]).len() as u32;
        if level > 0 {
            return caret - level;
        }
        // Leading whitespace that is neither a tab nor a space (U+3000…)
        // has no level: an ordinary one-char delete.
    }
    let prev = buffer.char_before(caret).expect("caret > 0");
    caret - prev.len_utf8() as u32
}

/// The kind of one indent level — THE indent-unit rule. Enter's
/// block-opening level, Tab, block indent, Shift+Tab, and backspace's
/// tab-stop delete all classify a line here, so they cannot disagree about
/// it: a tab-indented line grows and shrinks by hard tabs, anything else by
/// spaces between stops [`default_indent_size`] apart. (A global
/// tabs-vs-spaces preference is a host setting this does not know; an
/// unindented line therefore takes spaces.)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum IndentUnit {
    /// One hard tab.
    Tab,
    /// Spaces to the neighbouring indent stop.
    Spaces,
}

impl IndentUnit {
    /// The unit leading whitespace `ws` calls for: the kind of its LAST char —
    /// a hard tab ⇒ [`Tab`](Self::Tab), spaces or nothing ⇒
    /// [`Spaces`](Self::Spaces). The last char because it is the one a level
    /// abuts: `\t··` is tab-indented then space-ALIGNED, and a level added
    /// after alignment spaces must be spaces, or the line would interleave
    /// kinds (`\t··\t`).
    fn of(ws: &str) -> Self {
        if ws.ends_with('\t') {
            Self::Tab
        } else {
            Self::Spaces
        }
    }

    /// The char a level of this unit is made of.
    fn char(self) -> char {
        match self {
            Self::Tab => '\t',
            Self::Spaces => ' ',
        }
    }

    /// The text of one level inserted right after `prefix` (the line's text
    /// before the insertion point): a hard tab, or the spaces to the next
    /// stop.
    fn text(self, prefix: &str) -> String {
        match self {
            Self::Tab => "\t".to_owned(),
            Self::Spaces => {
                let n = default_indent_size();
                " ".repeat((n - stop_col(prefix) % n) as usize)
            }
        }
    }
}

/// The column `prefix`'s end sits at for indent-stop arithmetic: its bytes
/// after the last hard tab (or from the line start). A hard tab ends on a tab
/// stop, so counting from it keeps the stops where they are drawn while the
/// display tab width ([`crate::display_map::default_tab_size`]) is a multiple
/// of the indent width — and keeps a level of spaces after a tab inverse to
/// its outdent. Byte-based like every column here.
fn stop_col(prefix: &str) -> u32 {
    (prefix.len() - prefix.rfind('\t').map_or(0, |i| i + 1)) as u32
}

/// Where one indent level goes in leading whitespace `ws`: the START of its
/// trailing run of the [`IndentUnit::of`] char. The run is one repeated char,
/// so the text comes out the same anywhere in it; its start is, for a pure
/// indent, column 0 — where these verbs always edited, so selections rebase
/// as they did (one beginning at an indented line's text stays on the text
/// through the (out)dent) — while a mixed line's earlier whitespace (the `\t`
/// before alignment spaces) is left in front, untouched.
fn level_start(ws: &str, unit: IndentUnit) -> u32 {
    ws.trim_end_matches(unit.char()).len() as u32
}

/// One indent level deeper for leading whitespace `ws`: insert `.1` at byte
/// `.0` of the line — a hard tab on a tab-indented line, else spaces padding
/// the trailing space run to its next stop.
fn indent_insert(ws: &str) -> (u32, String) {
    let unit = IndentUnit::of(ws);
    (level_start(ws, unit), unit.text(ws))
}

/// One indent level shallower for leading whitespace `ws` (or the part of it
/// before a caret): the byte range to delete — its last tab, or its trailing
/// spaces back to the previous stop (at most [`default_indent_size`], never
/// past a tab) — empty when `ws` has no level to give.
fn outdent_range(ws: &str) -> Range<u32> {
    let unit = IndentUnit::of(ws);
    let start = level_start(ws, unit);
    let run = ws.len() as u32 - start;
    let width = match unit {
        IndentUnit::Tab => 1, // `of` ⇒ the run holds at least this one tab
        IndentUnit::Spaces if run == 0 => 0,
        // `run` counts from the last tab or the line start: `stop_col(ws)`.
        IndentUnit::Spaces => (run - 1) % default_indent_size() + 1,
    };
    start..start + width
}

/// The leading-whitespace indent an Enter at `caret` should carry: the current
/// line's leading whitespace, truncated to the caret's column (so pressing
/// Enter inside the indentation does not duplicate it).
fn enter_indent(buffer: &Buffer, caret: u32) -> String {
    let p = buffer.offset_to_point(caret);
    let line = buffer.line(p.row);
    let ws = leading_ws(&line);
    ws[..ws.len().min(p.col as usize)].to_owned()
}

/// A line's leading whitespace — the indent every indent verb reads (Tab,
/// block (out)dent, Enter, insert-line). The one slice of the one
/// [`tail_start_col`] rule, so they cannot disagree on where an indent ends.
fn leading_ws(line: &str) -> &str {
    &line[..tail_start_col(line) as usize]
}

/// One indent unit matching the KIND of the `copied` leading whitespace
/// ([`IndentUnit::of`], the rule Tab and Shift+Tab share): a tab-indented
/// prefix grows by one hard tab, anything else (spaces or an empty prefix) by
/// [`default_indent_size`] spaces — so an indent-carrying Enter never mixes
/// tabs and spaces within a line. A WHOLE level — the unit's
/// [`text`](IndentUnit::text) from an indent stop (an empty prefix) — not
/// Tab's pad to the next stop from the caret: the two agree on every on-grid
/// indent.
fn indent_unit(copied: &str) -> String {
    IndentUnit::of(copied).text("")
}

/// Whether the current line opens a block at `caret`: the text left of `caret`,
/// right-trimmed, ends with `{` — the caret is entering a freshly opened block,
/// so the new line indents one unit deeper.
fn line_opens_block(buffer: &Buffer, caret: u32) -> bool {
    let p = buffer.offset_to_point(caret);
    buffer.line(p.row)[..p.col as usize].trim_end().ends_with('{')
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(s: &str) -> Document {
        Document::new(s).unwrap()
    }

    /// Count of live auto-close provenance pairs, read from the dedicated
    /// auto-close store that holds them.
    fn ac_count(d: &Document) -> usize {
        d.autoclose_ranges().len()
    }

    #[test]
    fn typing_inserts_and_advances_caret() {
        let mut d = doc("");
        d.type_char('h');
        d.type_char('i');
        assert_eq!(d.text(), "hi");
        assert_eq!(d.selections().all()[0].head(), 2);
        // One typing run → one undo.
        assert!(d.undo());
        assert_eq!(d.text(), "");
    }

    #[test]
    fn typing_over_a_selection_replaces_it() {
        // Select then type must replace the selection, leaving one caret after
        // the inserted char — no residue of the old selection.
        let mut d = doc("hello world");
        d.selections = SelectionSet::new(0);
        d.selections.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 5));
        d.type_char('X');
        assert_eq!(d.text(), "X world");
        assert_eq!(d.selections().all()[0].head(), 1);
        assert!(d.selections().all()[0].is_empty());
    }

    #[test]
    fn autoclose_inserts_pair_then_overtypes() {
        let mut d = doc("");
        d.type_char('(');
        assert_eq!(d.text(), "()");
        assert_eq!(d.selections().all()[0].head(), 1); // caret between
        d.type_char(')'); // overtype the auto-inserted close
        assert_eq!(d.text(), "()", "overtype skips — no second )");
        assert_eq!(d.selections().all()[0].head(), 2);
    }

    #[test]
    fn autoclose_survives_typing_inside_the_pair() {
        let mut d = doc("");
        d.type_char('(');
        d.type_char('x'); // "(x)", caret 2
        assert_eq!(d.text(), "(x)");
        d.type_char(')'); // still overtypes the tracked close
        assert_eq!(d.text(), "(x)");
        assert_eq!(d.selections().all()[0].head(), 3);
    }

    #[test]
    fn autoclose_guard_skips_pair_before_a_word() {
        let mut d = doc("abc"); // caret 0; the next char is a word char (guard fails)
        d.type_char('(');
        assert_eq!(d.text(), "(abc"); // just the opener
        assert_eq!(d.selections().all()[0].head(), 1);
    }

    #[test]
    fn delete_word_back_and_forward_respect_word_and_newline_bounds() {
        let mut d = doc("foo bar\nbaz");
        d.set_selections(crate::SelectionSet::new(11)); // end of "baz"
        d.delete_word_back();
        assert_eq!(d.text(), "foo bar\n"); // the word "baz" is gone
        d.delete_word_back(); // caret at col 0 now → deletes only the newline
        assert_eq!(d.text(), "foo bar");
        d.set_selections(crate::SelectionSet::new(0));
        d.delete_word_forward();
        assert_eq!(d.text(), " bar"); // "foo" gone, the space stays
    }

    #[test]
    fn delete_word_back_eats_only_a_multi_space_run() {
        let mut d = doc("foo   bar"); // 3 spaces
        d.set_selections(crate::SelectionSet::new(6)); // caret before "bar"
        d.delete_word_back();
        assert_eq!(d.text(), "foobar", "2+ whitespace → deletes only the run");
    }

    #[test]
    fn multi_caret_word_delete_merges_overlapping_ranges() {
        // Two carets in one word yield overlapping delete ranges; merged into a
        // single delete, they remove the shared span once instead of being
        // rejected as an overlapping transaction (which would delete nothing).
        let mut d = doc("hello");
        d.set_selections(crate::SelectionSet::from_offsets(&[3, 5]));
        d.delete_word_back();
        assert_eq!(d.text(), "", "merged into one 0..5 delete, not dropped");
        assert_eq!(d.selections().len(), 1);

        let mut d = doc("hello");
        d.set_selections(crate::SelectionSet::from_offsets(&[0, 2]));
        d.delete_word_forward();
        assert_eq!(d.text(), "", "forward merge too");
    }

    #[test]
    fn autoclose_pair_backspace_deletes_both() {
        let mut d = doc("");
        d.type_char('(');
        d.backspace();
        assert_eq!(d.text(), "");
        assert_eq!(d.selections().all()[0].head(), 0);
    }

    #[test]
    fn autoclose_surrounds_a_selection() {
        let mut d = doc("foo");
        d.selections.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 3));
        d.type_char('(');
        assert_eq!(d.text(), "(foo)");
        let s = d.selections().all()[0];
        assert_eq!((s.start(), s.end()), (1, 4)); // still on "foo"
    }

    #[test]
    fn autoclose_provenance_invalidates_on_caret_move() {
        let mut d = doc("");
        d.type_char('('); // "()", provenance live, caret 1
        d.move_carets(crate::Motion::Left, false); // caret 0 — leaves the pair
        d.type_char(')'); // no live provenance → literal insert, not overtype
        assert_eq!(d.text(), ")()");
    }

    #[test]
    fn autoclose_quote_parity_closes_an_open_string() {
        // Prev char is a space (the before-guard is fine), but the line already
        // has one quote — an odd count means this quote closes the string:
        // insert literally, no pair.
        let mut d = doc("\" ");
        d.selections = SelectionSet::new(2); // caret after the space
        d.type_char('"');
        assert_eq!(d.text(), "\" \""); // one quote added, not a pair
    }

    #[test]
    fn autoclose_pairs_at_every_caret() {
        // Multi-cursor '{' pairs at EVERY caret, not a lone '{'.
        let mut d = doc("a\nb\nc"); // the three line ends are offsets 1, 3, 5
        d.set_selections(crate::SelectionSet::from_offsets(&[1, 3, 5]));
        d.type_char('{');
        assert_eq!(d.text(), "a{}\nb{}\nc{}");
        assert_eq!(ac_count(&d), 3, "one live provenance pair per caret");
        // Each caret sits between its pair — typing lands inside every one.
        d.type_char('x');
        assert_eq!(d.text(), "a{x}\nb{x}\nc{x}");
    }

    #[test]
    fn autoclose_overtypes_at_every_caret() {
        // Typing the closer at every pair's close skips over each (no second).
        let mut d = doc("a\nb\nc");
        d.set_selections(crate::SelectionSet::from_offsets(&[1, 3, 5]));
        d.type_char('{'); // "a{}\nb{}\nc{}", each caret between
        d.type_char('}'); // overtype every one
        assert_eq!(d.text(), "a{}\nb{}\nc{}", "overtype skips at every caret — no doubled closer");
        assert_eq!(ac_count(&d), 0, "provenance consumed");
    }

    #[test]
    fn autoclose_surrounds_every_selection() {
        // '(' over a multi-selection surrounds each — "surround all occurrences".
        let mut d = doc("foo bar baz");
        d.set_selections(crate::SelectionSet::from_ranges(&[(0, 3), (4, 7), (8, 11)], 0));
        d.type_char('(');
        assert_eq!(d.text(), "(foo) (bar) (baz)");
    }

    #[test]
    fn autoclose_pair_backspace_at_every_caret() {
        // Backspace strictly between empty pairs deletes both at every caret.
        let mut d = doc("a\nb\nc");
        d.set_selections(crate::SelectionSet::from_offsets(&[1, 3, 5]));
        d.type_char('{'); // "a{}\nb{}\nc{}"
        d.backspace();
        assert_eq!(d.text(), "a\nb\nc", "each empty pair deleted whole");
        assert_eq!(ac_count(&d), 0);
    }

    #[test]
    fn autoclose_nested_opener_supersedes_provenance() {
        // Typing an opener INSIDE a live pair must supersede the prior provenance
        // (the outer pair grew to enclose the insertion), not accumulate it —
        // else overtype's ends-ascending search picks the wrong pair and the
        // closer inserts literally, yielding "(()))" instead of "(())".
        let mut d = doc("");
        d.type_char('('); // "()"
        d.type_char('('); // "(())", inner caret; prior provenance superseded
        assert_eq!(ac_count(&d), 1, "only the innermost pair is live");
        d.type_char(')'); // overtypes the inner close, not a literal insert
        assert_eq!(d.text(), "(())");
        // A different opener nested (brace then paren then close).
        let mut e = doc("");
        e.type_char('{');
        e.type_char('(');
        e.type_char(')');
        assert_eq!(e.text(), "{()}");
    }

    #[test]
    fn autoclose_nested_pair_backspace_deletes_the_inner() {
        // Backspace between the innermost empty pair deletes both (not one).
        let mut d = doc("");
        d.type_char('(');
        d.type_char('('); // "(())", caret between the inner pair
        d.backspace();
        assert_eq!(d.text(), "()", "the inner empty pair is deleted whole");
    }

    #[test]
    fn autoclose_surrounds_touching_selections() {
        // Two TOUCHING non-empty selections (e.g. find-all "ab" in "abab") must
        // each keep their own interior — the successor must land on "cd", not
        // swallow the opener and land on "(cd".
        let mut d = doc("abcd");
        d.set_selections(crate::SelectionSet::from_ranges(&[(0, 2), (2, 4)], 0));
        d.type_char('(');
        assert_eq!(d.text(), "(ab)(cd)");
        let sels = d.selections().all();
        assert_eq!((sels[0].start(), sels[0].end()), (1, 3), "interior 'ab'");
        assert_eq!((sels[1].start(), sels[1].end()), (5, 7), "interior 'cd', not '(cd'");
    }

    #[test]
    fn autoclose_per_caret_pairs_only_where_the_guard_passes() {
        // A per-caret decision: a caret before a word char inserts a BARE opener
        // (the guard fails), while a caret at EOF pairs.
        let mut d = doc("ab");
        d.set_selections(crate::SelectionSet::from_offsets(&[0, 2]));
        d.type_char('{');
        assert_eq!(d.text(), "{ab{}"); // bare '{' before 'a', a pair at EOF
        assert_eq!(ac_count(&d), 1, "only the guarded caret paired");
    }

    #[test]
    fn backspace_deletes_char_then_selection() {
        let mut d = doc("abc");
        d.selections = SelectionSet::new(3);
        d.backspace();
        assert_eq!(d.text(), "ab");
        d.selections = SelectionSet::new(0);
        d.selections.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 2));
        d.backspace();
        assert_eq!(d.text(), "");
    }

    #[test]
    fn backspace_in_indentation_deletes_to_tab_stop() {
        let mut d = doc("        x"); // 8 spaces
        d.selections = SelectionSet::new(8); // caret after the spaces
        d.backspace();
        assert_eq!(d.text(), "    x"); // removed 4 (one stop), not 1
    }

    #[test]
    fn two_carets_in_one_indent_run_backspace_to_their_shared_tab_stop() {
        // The repro: carets at cols 3 and 4 both target stop 0, so their
        // tab-stop deletes are 0..3 and 0..4 — overlapping. Unmerged, the
        // transaction rejected them and the keystroke silently did nothing.
        let mut d = doc("        x"); // 8 spaces
        d.set_selections(SelectionSet::from_offsets(&[3, 4]));
        d.backspace();
        assert_eq!(d.text(), "    x", "one 0..4 delete — the union of both carets' stops");
        let heads: Vec<u32> = d.selections().all().iter().map(Selection::head).collect();
        assert_eq!(heads, [0], "both carets collapse onto the shared stop");
        assert!(d.undo());
        assert_eq!(d.text(), "        x", "one undo step restores the text");
        assert!(!d.can_undo(), "…and it was the only step");
    }

    #[test]
    fn tab_stop_backspace_overlapping_a_selection_merges_into_it() {
        // The mixed form of the same overlap: a caret at col 3 deletes back to
        // stop 0 straight through a selection on cols 1..2 (never touching the
        // caret, so the selection set kept them apart).
        let mut d = doc("        x");
        let mut set = SelectionSet::new(3);
        set.add_selection(1, 2);
        d.set_selections(set);
        d.backspace();
        assert_eq!(d.text(), "     x", "0..3 ∪ 1..2 deleted once");
        let heads: Vec<u32> = d.selections().all().iter().map(Selection::head).collect();
        assert_eq!(heads, [0]);
        assert!(d.undo());
        assert_eq!(d.text(), "        x");
        assert!(!d.can_undo());
    }

    /// The heads of every selection, ascending.
    fn heads(d: &Document) -> Vec<u32> {
        d.selections().all().iter().map(Selection::head).collect()
    }

    /// Undo once and require it restored `text` and emptied the history.
    fn assert_one_undo_restores(d: &mut Document, text: &str) {
        assert!(d.undo());
        assert_eq!(d.text(), text, "one undo restores the text");
        assert!(!d.can_undo(), "…and the delete was the only step");
    }

    #[test]
    fn adjacent_carets_char_delete_each_char_once() {
        // The char-delete audit: one char per caret can only TOUCH the next
        // range (distinct carets; a caret touching a selection merges into it),
        // so both carets' chars go and the carets meet.
        let mut d = doc("abcd");
        d.set_selections(SelectionSet::from_offsets(&[1, 2]));
        d.backspace();
        assert_eq!((d.text().as_ref(), heads(&d)), ("cd", vec![0]));
        assert_one_undo_restores(&mut d, "abcd");
        let mut d = doc("abcd");
        d.set_selections(SelectionSet::from_offsets(&[1, 2]));
        d.delete_forward();
        assert_eq!((d.text().as_ref(), heads(&d)), ("ad", vec![1]));
        assert_one_undo_restores(&mut d, "abcd");
    }

    #[test]
    fn word_delete_reaching_through_a_selection_merges_into_it() {
        // The word-delete audit, mixed form (two carets in one word is
        // `multi_caret_word_delete_merges_overlapping_ranges`): the caret's word
        // span swallows a selection inside the same word.
        let mut d = doc("foo bar");
        let mut set = SelectionSet::new(3); // after "foo"
        set.add_selection(1, 2);
        d.set_selections(set);
        d.delete_word_back();
        assert_eq!((d.text().as_ref(), heads(&d)), (" bar", vec![0]), "0..3 ∪ 1..2");
        assert_one_undo_restores(&mut d, "foo bar");
        let mut d = doc("foo bar");
        let mut set = SelectionSet::new(0); // before "foo"
        set.add_selection(1, 2);
        d.set_selections(set);
        d.delete_word_forward();
        assert_eq!((d.text().as_ref(), heads(&d)), (" bar", vec![0]), "0..3 ∪ 1..2");
        assert_one_undo_restores(&mut d, "foo bar");
    }

    #[test]
    fn overlapping_line_blocks_delete_once() {
        // The line-delete audit: a two-row selection's block contains the block
        // of a caret on its second row.
        let mut d = doc("a\nbb\nc");
        let mut set = SelectionSet::new(4); // end of "bb", row 1
        set.add_selection(0, 3); // rows 0..=1 → block 0..5
        d.set_selections(set);
        d.delete_line();
        assert_eq!((d.text().as_ref(), heads(&d)), ("c", vec![0]), "0..5 ∪ 2..5");
        assert_one_undo_restores(&mut d, "a\nbb\nc");
        // Cut's whole-line mode: two carets on one line yield the SAME block.
        let mut d = doc("aaaa\nbb");
        d.set_selections(SelectionSet::from_offsets(&[1, 3]));
        d.cut();
        assert_eq!((d.text().as_ref(), heads(&d)), ("bb", vec![0]));
        assert_one_undo_restores(&mut d, "aaaa\nbb");
    }

    #[test]
    fn multi_caret_deletes_remove_the_union_of_each_selections_lone_delete() {
        // The overlap oracle for EVERY multi-caret delete verb. Spec: a
        // multi-selection delete removes exactly the union of what each
        // selection would delete alone — a shared span once, never a rejected
        // transaction — leaves a caret where each of those deletes began, and
        // one undo restores the text. Each lone delete is observed through the
        // public API only (a pure delete leaves its caret at the range start,
        // and the length drop is the width); the union is a per-byte mask, not
        // `merge_delete_ranges`' sort-and-sweep. Indent-heavy lines with up to
        // four selections make overlapping tab-stop, word and line ranges common.
        let mut rng = 0x0010_7AB5_70B5_u64;
        let mut next = move |m: u64| {
            rng ^= rng << 13;
            rng ^= rng >> 7;
            rng ^= rng << 17;
            rng % m
        };
        type Verb = fn(&mut Document);
        let verbs: [(&str, Verb); 6] = [
            ("backspace", Document::backspace),
            ("delete_forward", Document::delete_forward),
            ("delete_word_back", Document::delete_word_back),
            ("delete_word_forward", Document::delete_word_forward),
            ("delete_line", Document::delete_line),
            ("cut", Document::cut),
        ];
        const TAILS: &[&str] = &["", "x", "foo bar", "a.b", "(y)", "  z"];
        // Per verb: cases run, and cases whose lone deletes overlapped.
        let mut runs = [0usize; 6];
        let mut overlaps = [0usize; 6];
        for case in 0..3_000 {
            let text = (0..1 + next(3))
                .map(|_| {
                    // Mostly spaces, some tabs: pure, tab-then-aligned and
                    // interleaved indents all reach the tab-stop backspace.
                    let indent: String = (0..next(10))
                        .map(|_| if next(8) == 0 { '\t' } else { ' ' })
                        .collect();
                    format!("{indent}{}", TAILS[next(TAILS.len() as u64) as usize])
                })
                .collect::<Vec<_>>()
                .join("\n");
            let len = text.len() as u64; // ASCII: every offset is a char boundary
            let ranges: Vec<(u32, u32)> = (0..1 + next(4))
                .map(|_| {
                    let a = next(len + 1) as u32;
                    (a, if next(4) == 0 { next(len + 1) as u32 } else { a })
                })
                .collect();
            let set = SelectionSet::from_ranges(&ranges, 0);
            let v = next(verbs.len() as u64) as usize;
            let (name, verb) = verbs[v];
            // Cut's own rule: in a mixed set an empty caret copied nothing, so
            // it deletes nothing (alone it would cut its whole line).
            let mixed = set.all().iter().any(|s| !s.is_empty());
            let mut mask = vec![false; text.len()];
            let (mut starts, mut lone_total) = (Vec::new(), 0);
            for sel in set.all() {
                if name == "cut" && mixed && sel.is_empty() {
                    continue;
                }
                let mut lone = doc(&text);
                lone.set_selections(SelectionSet::from_ranges(&[(sel.tail(), sel.head())], 0));
                verb(&mut lone);
                let width = text.len() - lone.text().len();
                if width > 0 {
                    let start = lone.selections().newest().head() as usize;
                    mask[start..start + width].iter_mut().for_each(|m| *m = true);
                    starts.push(start);
                    lone_total += width;
                }
            }
            let union = mask.iter().filter(|&&m| m).count();
            runs[v] += 1;
            overlaps[v] += usize::from(union < lone_total);
            let expected: String =
                text.bytes().zip(&mask).filter(|&(_, &m)| !m).map(|(b, _)| char::from(b)).collect();

            let mut d = doc(&text);
            d.set_selections(set);
            verb(&mut d);
            let what = format!("case {case}: {name} on {text:?} at {ranges:?}");
            assert_eq!(d.text(), expected, "{what}: the union of the lone deletes");
            for s in starts {
                let at = (s - mask[..s].iter().filter(|&&m| m).count()) as u32;
                assert!(
                    d.selections().all().iter().any(|c| c.is_empty() && c.head() == at),
                    "{what}: a caret where the delete from {s} began (post-edit {at})",
                );
            }
            if union > 0 {
                assert!(d.undo(), "{what}: undoable");
                assert_eq!(d.text(), text, "{what}: one undo restores the text");
            }
            assert!(!d.can_undo(), "{what}: the whole delete was ONE undo step");
        }
        // A one-char forward delete can only TOUCH its neighbour (a caret
        // touching a selection merges into it); every other verb must have
        // really been driven through overlapping ranges, or the oracle is vacuous.
        // (This seed, in `verbs` order: ~500 runs each; overlaps 33 / 0 / 111 /
        // 104 / 290 / 155.)
        for (v, (name, _)) in verbs.iter().enumerate() {
            if *name == "delete_forward" {
                assert_eq!(overlaps[v], 0, "delete_forward cannot overlap");
            } else {
                assert!(
                    overlaps[v] > 20,
                    "walk too thin for {name}: runs {runs:?}, overlaps {overlaps:?}",
                );
            }
        }
    }

    #[test]
    fn enter_carries_indent_truncated_to_caret() {
        let mut d = doc("    hello");
        d.selections = SelectionSet::new(9); // end of line
        d.enter();
        assert_eq!(d.text(), "    hello\n    "); // 4-space indent carried
        // Enter at column 2 (inside the indent) carries only 2 spaces of
        // autoindent (truncated to the caret); the 2 spaces after the caret
        // stay with "hello" on the new line → line 1 has 4 spaces total, not 8
        // — the indent is truncated to the caret, never duplicated.
        let mut d2 = doc("    hello");
        d2.selections = SelectionSet::new(2);
        d2.enter();
        assert_eq!(d2.text(), "  \n    hello");
    }

    #[test]
    fn delete_line_removes_blocks_and_the_final_line_cleanly() {
        let mut d = doc("one\ntwo\nthree");
        d.set_selections(crate::SelectionSet::new(5)); // caret in "two"
        d.delete_line();
        assert_eq!(d.text(), "one\nthree");
        // The final line takes its PRECEDING newline — no empty tail remains.
        d.set_selections(crate::SelectionSet::new(d.buffer().len()));
        d.delete_line();
        assert_eq!(d.text(), "one");
        // A multi-line selection deletes every spanned line (one ending at a
        // line start does not span that line)…
        let mut m = doc("a\nb\nc\nd");
        let mut set = crate::SelectionSet::new(0);
        set.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 4)); // rows 0..=1
        m.set_selections(set);
        m.delete_line();
        assert_eq!(m.text(), "c\nd");
        // …and two carets on one line delete it once.
        let mut t = doc("xx\nyy");
        t.set_selections(crate::SelectionSet::from_offsets(&[0, 1]));
        t.delete_line();
        assert_eq!(t.text(), "yy");
    }

    #[test]
    fn insert_line_below_and_above_carry_indent() {
        // Below: a fresh line under the caret's line, same indent, caret on it
        // — the caret's line is NOT split, wherever the caret sat.
        let mut d = doc("    a\n    b");
        d.set_selections(crate::SelectionSet::new(4)); // before "a"
        d.insert_line(true);
        assert_eq!(d.text(), "    a\n    \n    b");
        assert_eq!(d.selections().newest().head(), 10, "caret at the new line's end");
        // Above: the current line is pushed down; caret on the new line.
        let mut u = doc("    a");
        u.set_selections(crate::SelectionSet::new(5));
        u.insert_line(false);
        assert_eq!(u.text(), "    \n    a");
        assert_eq!(u.selections().newest().head(), 4);
        // A line below a block-opening `{` line gains one indent unit.
        let mut b = doc("m {");
        b.set_selections(crate::SelectionSet::new(0));
        b.insert_line(true);
        assert_eq!(b.text(), "m {\n    ");
        assert_eq!(b.selections().newest().head(), 8);
    }

    #[test]
    fn toggle_line_comment_comments_aligned_then_uncomments() {
        // Ctrl+/: markers align at the block's MIN indent; blank lines are
        // skipped; a mixed block comments everything; toggling back strips
        // prefix + one space; the selection survives (rebase, not re-set).
        let mut d = doc("    a\n\n        b");
        d.set_line_comment(Some("//"));
        d.select_all();
        d.toggle_line_comment();
        assert_eq!(d.text(), "    // a\n\n    //     b");
        assert!(!d.selections().newest().is_empty(), "selection survives the toggle");
        d.toggle_line_comment();
        assert_eq!(d.text(), "    a\n\n        b");
        // A bare caret toggles only its own line…
        let mut c = doc("x\ny");
        c.set_line_comment(Some("//"));
        c.set_selections(crate::SelectionSet::new(0));
        c.toggle_line_comment();
        assert_eq!(c.text(), "// x\ny");
        // …stripping works without the space too.
        let mut s = doc("//x");
        s.set_line_comment(Some("//"));
        s.toggle_line_comment();
        assert_eq!(s.text(), "x");
        // Mixed commented/uncommented → comment ALL, then one toggle back
        // returns to all-commented.
        let mut m = doc("// a\nb");
        m.set_line_comment(Some("//"));
        m.select_all();
        m.toggle_line_comment();
        assert_eq!(m.text(), "// // a\n// b");
    }

    #[test]
    fn toggle_line_comment_without_a_prefix_is_a_no_op() {
        // The core knows no language: nothing injected → nothing happens.
        let mut d = doc("abc");
        d.toggle_line_comment();
        assert_eq!(d.text(), "abc");
        // An all-blank span starts a comment ready to type into.
        let mut b = doc("");
        b.set_line_comment(Some("//"));
        b.toggle_line_comment();
        assert_eq!(b.text(), "// ");
        assert_eq!(b.selections().newest().head(), 3, "caret rebased after the prefix");
    }

    #[test]
    fn copy_and_cut_expand_an_empty_caret_to_its_line() {
        // An empty-selection Copy takes the ENTIRE line including \n…
        let mut d = doc("one\ntwo\nthree");
        d.set_selections(crate::SelectionSet::new(5)); // caret inside "two"
        assert_eq!(d.clipboard_payload(), ("two\n".to_string(), true));
        // …the final line (no terminator in the buffer) still exports one…
        d.set_selections(crate::SelectionSet::new(9)); // inside "three"
        assert_eq!(d.clipboard_payload(), ("three\n".to_string(), true));
        // …a non-empty selection copies just itself, not the line…
        let mut set = crate::SelectionSet::new(0);
        set.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 3));
        d.set_selections(set);
        assert_eq!(d.clipboard_payload(), ("one".to_string(), false));
        // …and Cut on a bare caret deletes the whole line (one undo step).
        d.set_selections(crate::SelectionSet::new(5));
        d.cut();
        assert_eq!(d.text(), "one\nthree");
        assert!(d.undo());
        assert_eq!(d.text(), "one\ntwo\nthree");
    }

    #[test]
    fn mixed_set_cut_never_deletes_uncopied_lines() {
        // Cut IS copy+delete. In a mixed set the empty caret exports nothing, so
        // it must delete nothing — only the selection goes.
        let mut d = doc("foo bar\nbaz\n");
        let mut set = crate::SelectionSet::new(0);
        set.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 3)); // "foo"
        set.add_caret(9); // bare caret inside "baz"
        d.set_selections(set);
        assert_eq!(d.clipboard_payload(), ("foo".to_string(), false));
        d.cut();
        assert_eq!(d.text(), " bar\nbaz\n", "the uncopied `baz` line survives");
    }

    #[test]
    fn cut_on_the_final_line_takes_the_preceding_newline() {
        // Cut on the final line takes its PRECEDING newline (matching its
        // sibling delete_line), so no empty tail line is left behind.
        let mut d = doc("one\ntwo");
        d.set_selections(crate::SelectionSet::new(5)); // inside "two"
        assert_eq!(d.clipboard_payload(), ("two\n".to_string(), true));
        d.cut();
        assert_eq!(d.text(), "one", "no empty tail line remains");
        // Degenerate: a caret on the trailing empty line cuts that line (the
        // preceding newline), not nothing.
        let mut e = doc("abc\n");
        e.set_selections(crate::SelectionSet::new(4));
        e.cut();
        assert_eq!(e.text(), "abc");
    }

    #[test]
    fn two_carets_on_one_line_cut_it_once() {
        let mut d = doc("aaaa\nbb");
        d.set_selections(crate::SelectionSet::from_offsets(&[1, 3])); // both on row 0
        assert_eq!(d.clipboard_payload(), ("aaaa\n".to_string(), true), "the line exports once");
        d.cut();
        assert_eq!(d.text(), "bb", "identical ranges merged into one delete");
    }

    #[test]
    fn entire_line_paste_inserts_above_and_keeps_the_caret() {
        // An entire-line paste at an empty caret splices ABOVE the caret's
        // line; the caret rebases past the insert and stays on its own line.
        let mut d = doc("aaa\nbbb");
        d.set_selections(crate::SelectionSet::new(5)); // "bbb", col 1
        d.paste("two\n", true);
        assert_eq!(d.text(), "aaa\ntwo\nbbb");
        assert_eq!(d.selections().newest().head(), 9, "still col 1 of `bbb`");
        // Over a non-empty selection the flag is moot — plain replace.
        let mut set = crate::SelectionSet::new(0);
        set.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 3));
        d.set_selections(set);
        d.paste("X\n", true);
        assert_eq!(d.text(), "X\n\ntwo\nbbb");
    }

    #[test]
    fn enter_after_open_brace_indents_one_unit_deeper() {
        // Enter after a line-opening `{` gains one indent unit.
        let mut d = doc("    if {");
        d.selections = SelectionSet::new(8);
        d.enter();
        assert_eq!(d.text(), "    if {\n        ");
        // Trailing whitespace between `{` and the caret still triggers (the
        // rule reads the TRIMMED text left of the caret).
        let mut t = doc("m {  ");
        t.selections = SelectionSet::new(5);
        t.enter();
        assert_eq!(t.text(), "m {  \n    ");
        // …but a caret not after an opener stays keep-style.
        let mut p = doc("m { x");
        p.selections = SelectionSet::new(5);
        p.enter();
        assert_eq!(p.text(), "m { x\n");
    }

    #[test]
    fn enter_between_braces_splits_onto_three_lines() {
        // Type `{` (auto-close pairs it), Enter → opener line, indented middle
        // line holding the caret, closer dedented on its own.
        let mut d = doc("");
        d.type_char('{');
        assert_eq!(d.text(), "{}");
        d.enter();
        assert_eq!(d.text(), "{\n    \n}");
        assert_eq!(d.selections().newest().head(), 6, "caret at the middle line's end");
        // In an indented context the closer keeps the ORIGINAL indent.
        let mut n = doc("    {}");
        n.selections = SelectionSet::new(5);
        n.enter();
        assert_eq!(n.text(), "    {\n        \n    }");
        assert_eq!(n.selections().newest().head(), 14);
        // One undo reverts the whole split (a single transaction).
        assert!(n.undo());
        assert_eq!(n.text(), "    {}");
    }

    #[test]
    fn enter_indent_unit_matches_tab_kind() {
        // A hard-tab-indented line grows by a TAB, not spaces — the indent unit
        // matches the kind of the copied leading whitespace.
        let mut d = doc("\tif {");
        d.selections = SelectionSet::new(5);
        d.enter();
        assert_eq!(d.text(), "\tif {\n\t\t");
    }

    #[test]
    fn tab_inserts_to_next_stop() {
        let mut d = doc("ab");
        d.selections = SelectionSet::new(2); // col 2 → next stop at 4, insert 2
        d.tab();
        assert_eq!(d.text(), "ab  ");
        d.tab(); // col 4 → next stop 8, insert 4
        assert_eq!(d.text(), "ab      ");
    }

    #[test]
    fn tab_over_a_single_line_selection_types_over() {
        let mut d = doc("abcdef");
        d.selections.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 1, 4)); // "bcd"
        d.tab();
        // Single-line selection → replace with spaces to the next stop (from
        // col 1 that's 3), NOT a whole-line indent.
        assert_eq!(d.text(), "a   ef");
        assert!(d.selections().newest().is_empty(), "types over → collapses to a caret");
    }

    #[test]
    fn shift_tab_outdents_a_single_line_and_keeps_the_selection() {
        let mut d = doc("    indented");
        d.selections.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 5, 9)); // "nden"
        d.outdent();
        assert_eq!(d.text(), "indented"); // the line outdents (unlike Tab)
        // the selection rides the removed indent, still on "nden"
        assert_eq!((d.selections().newest().start(), d.selections().newest().end()), (1, 5));
    }

    #[test]
    fn block_indent_outdent_preserves_the_selection() {
        let mut d = doc("a\nb\nc");
        // Select from line 0 into line 2.
        d.selections.set_single(crate::Selection::from_anchor(crate::SelectionId(0), 0, 5));
        d.tab();
        assert_eq!(d.text(), "    a\n    b\n    c");
        // The selection is preserved — one selection over the indented block,
        // NOT one caret per line.
        assert_eq!(d.selections().len(), 1, "no extra cursors");
        let s = d.selections().newest();
        assert_eq!((s.start(), s.end()), (0, 17));
        assert!(!s.is_empty(), "still a selection, not a caret");
        // Outdent restores the text and keeps the selection.
        d.outdent();
        assert_eq!(d.text(), "a\nb\nc");
        assert_eq!(d.selections().len(), 1);
        assert_eq!(
            (d.selections().newest().start(), d.selections().newest().end()),
            (0, 5)
        );
    }

    /// Select `anchor..head` as the one selection.
    fn select(d: &mut Document, anchor: u32, head: u32) {
        d.set_selections(SelectionSet::from_ranges(&[(anchor, head)], 0));
    }

    /// Run `verb`, require `after`, and require ONE undo step restoring the
    /// text `verb` started from.
    #[track_caller]
    fn assert_one_step(d: &mut Document, verb: fn(&mut Document), after: &str) {
        let before = d.text().into_owned();
        verb(d);
        assert_eq!(d.text(), after, "{before:?}");
        assert_one_undo_restores(d, &before);
    }

    #[test]
    fn outdent_removes_a_hard_tab() {
        // The repro: Shift+Tab worked in spaces only, so a tab-indented
        // line could not be outdented at all.
        let mut d = doc("\tfoo");
        assert_one_step(&mut d, Document::outdent, "foo");
        let mut d = doc("\t\tfoo"); // one level per press, not the whole indent
        d.set_selections(SelectionSet::new(4)); // caret inside the text
        assert_one_step(&mut d, Document::outdent, "\tfoo");
    }

    #[test]
    fn block_indent_of_a_tab_indented_block_adds_tabs() {
        // The repro: Tab over a tab-indented block inserted SPACES ahead of
        // each tab (`"    \tfoo"`), mixing kinds within the line.
        let mut d = doc("\tfoo\n\tbar");
        d.select_all();
        assert_one_step(&mut d, Document::tab, "\t\tfoo\n\t\tbar");
        // A block selection begun at the text (after Home) stays on the text:
        // the new level lands ahead of the old one, never inside the selection.
        select(&mut d, 1, 9);
        d.tab();
        let s = d.selections().newest();
        assert_eq!((s.start(), s.end()), (2, 11), "the block selection survives");
        let mut d = doc("\t\tfoo\n\t\tbar");
        d.select_all();
        assert_one_step(&mut d, Document::outdent, "\tfoo\n\tbar");
    }

    #[test]
    fn caret_tab_on_a_tab_indented_line_inserts_a_tab() {
        // The caret form of the same mix: Tab at the line start (or anywhere on
        // a tab-indented line) inserts the line's own unit.
        let mut d = doc("\tfoo");
        d.set_selections(SelectionSet::new(0));
        assert_one_step(&mut d, Document::tab, "\t\tfoo");
        d.set_selections(SelectionSet::new(4)); // end of the line
        assert_one_step(&mut d, Document::tab, "\tfoo\t");
        // A space-indented (or unindented) line still pads to the next stop.
        let mut d = doc("  foo");
        d.set_selections(SelectionSet::new(1));
        assert_one_step(&mut d, Document::tab, "     foo");
    }

    #[test]
    fn space_indents_snap_to_the_neighbouring_stops() {
        // Indent pads to the NEXT stop and outdent cuts back to the PREVIOUS
        // one — an off-grid indent is realigned, not shifted by a fixed 4.
        let mut d = doc("  foo\n      bar");
        select(&mut d, 0, 15);
        assert_one_step(&mut d, Document::tab, "    foo\n        bar");
        select(&mut d, 0, 15);
        assert_one_step(&mut d, Document::outdent, "foo\n    bar");
        let mut d = doc("      foo"); // 6 spaces
        assert_one_step(&mut d, Document::outdent, "    foo");
    }

    #[test]
    fn mixed_leading_whitespace_moves_one_unit_of_its_last_kind() {
        // The indent unit is the kind of the leading whitespace's LAST char
        // (Enter's rule): tab-then-alignment-spaces is a SPACES line whose
        // stops count from the tab; spaces-then-tab is a TAB line.
        let mut d = doc("\t  foo");
        assert_one_step(&mut d, Document::outdent, "\tfoo");
        let mut d = doc("\t      foo"); // tab + 6: back to the stop at tab + 4
        assert_one_step(&mut d, Document::outdent, "\t    foo");
        let mut d = doc("\t  foo\nx");
        select(&mut d, 0, 8);
        assert_one_step(&mut d, Document::tab, "\t    foo\n    x");
        let mut d = doc("  \tfoo");
        assert_one_step(&mut d, Document::outdent, "  foo");
        let mut d = doc("  \tfoo\nx");
        select(&mut d, 0, 8);
        assert_one_step(&mut d, Document::tab, "  \t\tfoo\n    x");
        // A whitespace-only line is all leading whitespace.
        let mut d = doc("\t\t");
        assert_one_step(&mut d, Document::outdent, "\t");
    }

    #[test]
    fn caret_tab_before_a_mixed_indents_last_run_indents_like_block_indent() {
        // Fails-first: a caret Tab dropped the line's unit AT the caret, so at
        // column 0 of `··\tfoo` (a tab line: its indent ends in a tab) it gave
        // `\t··\tfoo` — interleaving kinds, the very thing the unit rule exists
        // to prevent — where block indent gives `··\t\tfoo`. A caret ahead of
        // the indent's trailing run now inserts where block indent does, and
        // lands just past the new level.
        for (line, col, want, caret) in [
            ("  \tfoo", 0, "  \t\tfoo", 3),
            ("  \tfoo", 1, "  \t\tfoo", 3),
            ("\t  foo", 0, "\t    foo", 3), // a spaces line (tab-then-alignment)
        ] {
            let mut d = doc(line);
            d.set_selections(SelectionSet::new(col));
            d.tab();
            assert_eq!(d.text(), want, "{line:?} @ {col}");
            assert_eq!(heads(&d), [caret], "{line:?} @ {col}: the caret lands past the new level");
            assert_one_undo_restores(&mut d, line);
            // Block indent of the same line agrees.
            let mut b = doc(&format!("{line}\nx"));
            select(&mut b, 0, line.len() as u32 + 2);
            b.tab();
            assert_eq!(b.text().split('\n').next(), Some(want), "{line:?}: block indent");
        }
        // In (or at an edge of) the trailing run, the level still goes at the
        // caret — inserting a run's own char into it mixes nothing — and spaces
        // still pad from the caret to its next stop (VS Code's caret Tab).
        for (line, col, want) in [
            ("  \tfoo", 2, "  \t\tfoo"),
            ("\t  foo", 3, "\t    foo"),
            ("    foo", 2, "      foo"),
            ("\t\tfoo", 1, "\t\t\tfoo"),
        ] {
            let mut d = doc(line);
            d.set_selections(SelectionSet::new(col));
            assert_one_step(&mut d, Document::tab, want);
        }
    }

    #[test]
    fn tab_over_a_selection_ahead_of_a_mixed_indents_last_run_never_interleaves() {
        // Fails-first: the caret fix above covered only empty carets, so a
        // selection ahead of the trailing run still had the unit typed in its
        // place — the `·` selected in `··\tfoo` became a tab, giving `\t·\tfoo`
        // where block indent gives `··\t\tfoo`. The selection is still typed
        // over (deleted), and the unit goes where block indent puts it.
        for (line, (from, to), want, caret) in [
            ("  \tfoo", (0, 1), " \t\tfoo", 2),
            ("  \tfoo", (1, 0), " \t\tfoo", 2), // either direction
            ("\t\t  foo", (0, 1), "\t    foo", 3), // a spaces line: tabs, then the run
        ] {
            let mut d = doc(line);
            select(&mut d, from, to);
            d.tab();
            assert_eq!(d.text(), want, "{line:?} {from}..{to}");
            assert_eq!(heads(&d), [caret], "{line:?} {from}..{to}: the caret lands past the new level");
            assert_one_undo_restores(&mut d, line);
        }
    }

    #[test]
    fn tab_from_several_selections_in_one_mixed_indent_stays_disjoint() {
        // Fails-first: each caret ahead of the run inserted at the run's start
        // even when another selection typed over that very point — two
        // overlapping ops, a verb bug the debug overlap assert caught (and a
        // silent no-op in release). Against a selection reaching into the run,
        // the unit goes at that selection's start (where the run starts once
        // it is typed over); several selections ahead of the run each delete
        // and each add a level there.
        for (line, sels, want, carets) in [
            // A caret ahead of the run + a selection straddling the run start.
            ("  \t\tfoo", &[(0, 0), (1, 3)][..], " \t\t\tfoo", &[2, 3][..]),
            // The same with a selection, not a caret, ahead of the run.
            ("   \t\tfoo", &[(0, 1), (2, 4)][..], " \t\t\tfoo", &[2, 3][..]),
            // Two selections ahead of the run: both deleted, two levels added.
            ("    \tfoo", &[(0, 1), (2, 3)][..], "  \t\t\tfoo", &[3, 4][..]),
        ] {
            let mut d = doc(line);
            d.set_selections(SelectionSet::from_ranges(sels, 0));
            d.tab();
            assert_eq!(d.text(), want, "{line:?} {sels:?}");
            assert_eq!(heads(&d), carets, "{line:?} {sels:?}");
            assert_one_undo_restores(&mut d, line);
        }
    }

    #[test]
    fn backspace_in_mixed_indentation_deletes_one_unit() {
        // Backspace-to-stop reads the same unit from the whitespace before the
        // caret: spaces after a tab go back to the stop counted from the tab…
        let mut d = doc("\t      foo"); // tab + 6
        d.set_selections(SelectionSet::new(7));
        assert_one_step(&mut d, Document::backspace, "\t    foo");
        d.set_selections(SelectionSet::new(7));
        d.backspace();
        assert_eq!(heads(&d), [5], "the caret lands on the stop");
        // …and a tab before the caret goes one tab at a time.
        let mut d = doc("  \t\tfoo");
        d.set_selections(SelectionSet::new(4));
        assert_one_step(&mut d, Document::backspace, "  \tfoo");
    }

    #[test]
    fn block_indent_over_mixed_kind_lines_keeps_each_lines_kind() {
        // One multi-line selection over tab-indented, space-indented, mixed and
        // unindented lines: every line moves by ITS OWN unit, in one step.
        let text = "\ta\n    b\n\t  c\nd\n  \te";
        let mut d = doc(text);
        d.select_all();
        assert_one_step(&mut d, Document::tab, "\t\ta\n        b\n\t    c\n    d\n  \t\te");
        d.select_all();
        assert_one_step(&mut d, Document::outdent, "a\nb\n\tc\nd\n  e");
        // Multi-caret Shift+Tab — one caret per line, mixed kinds — likewise.
        let mut d = doc(text);
        d.set_selections(SelectionSet::from_offsets(&[2, 8, 13, 15, 20]));
        assert_one_step(&mut d, Document::outdent, "a\nb\n\tc\nd\n  e");
    }

    #[test]
    fn multi_cursor_typing_edits_every_caret() {
        let mut d = doc("a b c");
        d.selections = SelectionSet::new(0);
        d.selections.add_caret(2);
        d.selections.add_caret(4);
        d.type_char('X');
        assert_eq!(d.text(), "Xa Xb Xc");
        assert_eq!(d.selections().len(), 3);
    }

    #[test]
    fn newline_paste_is_normalized_and_splits_lines() {
        let mut d = doc("");
        d.insert_text("one\r\ntwo");
        assert_eq!(d.text(), "one\ntwo");
        assert_eq!(d.buffer().line_count(), 2);
    }

    #[test]
    fn enter_undoes_and_redoes() {
        let mut d = doc("ab");
        d.selections = SelectionSet::new(1);
        d.enter();
        assert_eq!(d.text(), "a\nb");
        assert!(d.undo());
        assert_eq!(d.text(), "ab");
        assert!(d.redo());
        assert_eq!(d.text(), "a\nb");
    }

    #[test]
    fn caret_tracks_text_after_undo() {
        let mut d = doc("");
        for ch in "hello".chars() {
            d.type_char(ch);
        }
        d.undo(); // removes the whole run
        assert!(d.selections().all()[0].head() <= d.buffer().len(), "caret stays valid");
    }

    #[test]
    fn undo_returns_the_caret_to_the_edit() {
        // Undo restores the caret from before the run (the region's start here);
        // redo the caret the run left (its end) — as if the edit had just
        // (un)happened.
        let mut d = doc("");
        for ch in "hello".chars() {
            d.type_char(ch);
        }
        assert_eq!(d.selections().all()[0].head(), 5, "caret after typing");
        d.undo();
        assert_eq!(d.buffer().text(), "", "text undone");
        assert_eq!(d.selections().all()[0].head(), 0, "caret returns to the edit site");
        d.redo();
        assert_eq!(d.buffer().text(), "hello", "text redone");
        assert_eq!(d.selections().all()[0].head(), 5, "caret follows the redo");
    }

    #[test]
    fn undo_jumps_the_caret_to_the_edit_from_afar() {
        // Edit at the top, then click a caret elsewhere, then undo. The caret
        // must JUMP to the reverted edit (so you see what changed), landing at
        // the edit site rather than merely rebasing where it was parked.
        let mut d = doc("aaa\nbbb\nccc\nddd\n");
        d.set_selections(crate::SelectionSet::new(0)); // top of the document
        d.type_char('\n'); // an edit at offset 0 (like hitting Enter at the top)
        d.set_selections(crate::SelectionSet::new(10)); // click a caret down in the body
        assert_eq!(d.selections().all()[0].head(), 10, "caret parked in the body");
        assert!(d.undo());
        assert_eq!(d.buffer().text(), "aaa\nbbb\nccc\nddd\n", "the newline is gone");
        assert_eq!(
            d.selections().all()[0].head(),
            0,
            "undo jumps the caret to the reverted edit, not leaves it at 10"
        );
    }
}
