//! The completion controller state machine — core view-state, driven by
//! the widget and headlessly tested against a stub provider. It owns the popup
//! list, local prefix filtering, Escape-stickiness, and the provider-call budget
//! (at most one `complete()` per input event); it never touches the document.
//! Accepting returns the chosen item for the caller to apply as one sealed
//! transaction.

use super::providers::{CompletionCx, CompletionItem, CompletionTrigger, Completions};

/// The popup's open/closed/dismissed state.
pub enum CompletionState {
    /// No popup.
    Closed,
    /// A popup is showing `PopupList`.
    Open(PopupList),
    /// Dismissed by Escape and sticky until a word boundary: word chars
    /// extending the same word do NOT reopen the popup; a word boundary (or any
    /// non-word input) restores normal rules.
    DismissedUntilBoundary,
}

/// The live popup: the provider's items plus the local filter/selection over
/// them. The widget renders `items[filtered[i]]` for the visible window and
/// highlights `filtered[selected]`.
pub struct PopupList {
    /// The provider's items, unfiltered (the reference set the filter indexes).
    pub items: Vec<CompletionItem>,
    /// Indices into `items` passing the local prefix filter, in `sort_key`
    /// order. Rebuilt on every refilter.
    pub filtered: Vec<u32>,
    /// Index into `filtered` (not `items`); resets to 0 on every refilter.
    pub selected: u32,
    /// Absolute byte offset of the completion-word start — the popup's x anchor.
    pub anchor: u32,
}

impl PopupList {
    /// Rebuild `filtered` as the items whose label starts with `word` (folding
    /// ASCII case — completion labels are code identifiers), in `sort_key` order
    /// (so the provider's tiers survive), and reset the selection to the top.
    ///
    /// Runs on **every word char while the popup is open**, so it allocates
    /// nothing: it reuses `filtered`'s capacity (`clear` + `extend`) and compares
    /// prefixes in place — no per-item lowercased `String` and no fresh index
    /// `Vec` per keystroke, keeping the filter O(items) in comparisons only.
    fn refilter(&mut self, word: &str) {
        self.filtered.clear();
        self.filtered.extend(
            (0..self.items.len() as u32)
                .filter(|&i| prefix_matches(&self.items[i as usize].label, word)),
        );
        self.filtered
            .sort_by(|&a, &b| self.items[a as usize].sort_key.cmp(&self.items[b as usize].sort_key));
        self.selected = 0;
    }
}

/// Whether `label` begins with `prefix`, folding ASCII case only. Compares the
/// raw bytes (`as_bytes` never panics on a non-char-boundary split, and
/// `eq_ignore_ascii_case` leaves multi-byte UTF-8 exact), so no lowercased copy
/// is allocated — the hot completion-filter primitive.
fn prefix_matches(label: &str, prefix: &str) -> bool {
    let (l, p) = (label.as_bytes(), prefix.as_bytes());
    l.len() >= p.len() && l[..p.len()].eq_ignore_ascii_case(p)
}

/// The completion controller. Holds only the popup state; the document, provider,
/// and caret are supplied by the caller per event.
pub struct CompletionController {
    state: CompletionState,
}

impl Default for CompletionController {
    fn default() -> Self {
        Self::new()
    }
}

impl CompletionController {
    /// A fresh, closed controller.
    #[must_use]
    pub fn new() -> Self {
        Self { state: CompletionState::Closed }
    }

    /// The current state (for the widget to render).
    #[must_use]
    pub fn state(&self) -> &CompletionState {
        &self.state
    }

    /// Whether a popup is showing.
    #[must_use]
    pub fn is_open(&self) -> bool {
        matches!(self.state, CompletionState::Open(_))
    }

    /// Drive the machine on a completion-relevant keystroke, per `cx.trigger`:
    /// a **trigger char / Ctrl+Space** requests fresh items in any state (even
    /// while dismissed); a **word char** opens from `Closed`, refilters locally
    /// while `Open` (no provider call), and is ignored while dismissed. `word` is
    /// the live completion-word text under the caret. Calls `provider.complete`
    /// at most once per event.
    pub fn on_input(&mut self, cx: &CompletionCx, word: &str, provider: &mut dyn Completions) {
        match cx.trigger {
            CompletionTrigger::Typed(_) => match &mut self.state {
                CompletionState::Open(list) => {
                    list.refilter(word);
                    if list.filtered.is_empty() {
                        self.state = CompletionState::Closed;
                    }
                }
                CompletionState::DismissedUntilBoundary => {} // stay dismissed
                CompletionState::Closed => {
                    let items = provider.complete(cx);
                    self.set_from_items(items, word, cx.word.start);
                }
            },
            CompletionTrigger::TriggerChar(_) | CompletionTrigger::Manual => {
                // A fresh request in any state — a trigger char reopens even when
                // Escape-dismissed.
                let items = provider.complete(cx);
                self.set_from_items(items, word, cx.word.start);
            }
        }
    }

    /// Open from a fresh item list, or close if empty (either the provider
    /// returned nothing or nothing prefix-matches the live word).
    fn set_from_items(&mut self, items: Vec<CompletionItem>, word: &str, anchor: u32) {
        if items.is_empty() {
            self.state = CompletionState::Closed;
            return;
        }
        let mut list = PopupList { items, filtered: Vec::new(), selected: 0, anchor };
        list.refilter(word);
        self.state = if list.filtered.is_empty() {
            CompletionState::Closed
        } else {
            CompletionState::Open(list)
        };
    }

    /// A word boundary was crossed (a non-word / non-trigger char, or backspacing
    /// out of the word) — clears Escape-stickiness so the next word can reopen.
    pub fn on_boundary(&mut self) {
        if matches!(self.state, CompletionState::DismissedUntilBoundary) {
            self.state = CompletionState::Closed;
        }
    }

    /// Set the selected row to `index` (a filtered-list index), clamped; no-op
    /// when closed. Used by a popup row click.
    pub fn set_selected(&mut self, index: u32) {
        if let CompletionState::Open(list) = &mut self.state {
            let n = list.filtered.len() as u32;
            if n > 0 {
                list.selected = index.min(n - 1);
            }
        }
    }

    /// Move the selection (Up/Down) with wrap; no-op when closed.
    pub fn move_selection(&mut self, down: bool) {
        if let CompletionState::Open(list) = &mut self.state {
            let n = list.filtered.len() as u32;
            if n == 0 {
                return;
            }
            list.selected = if down {
                (list.selected + 1) % n
            } else {
                (list.selected + n - 1) % n
            };
        }
    }

    /// Escape while open → sticky-dismiss; returns whether the event was captured
    /// (so the widget knows not to let Escape fall through to other handlers).
    pub fn escape(&mut self) -> bool {
        if matches!(self.state, CompletionState::Open(_)) {
            self.state = CompletionState::DismissedUntilBoundary;
            true
        } else {
            false
        }
    }

    /// Accept the selected item: returns it (cloned) and closes the popup. The
    /// caller applies it as one sealed transaction — replace range = the item's
    /// `replace`, else the live completion word — and, if `retrigger`, fires one
    /// `Manual` `on_input`. Returns `None` when not open / nothing selected.
    pub fn accept(&mut self) -> Option<CompletionItem> {
        let item = match &self.state {
            CompletionState::Open(list) => {
                list.filtered.get(list.selected as usize).map(|&i| list.items[i as usize].clone())
            }
            _ => None,
        };
        if item.is_some() {
            self.state = CompletionState::Closed;
        }
        item
    }

    /// Hard close — a caret move not caused by typing (arrow/click/find-jump),
    /// focus loss, `set_text`, or undo/redo. Idempotent.
    pub fn close(&mut self) {
        self.state = CompletionState::Closed;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intel::providers::CompletionKind;
    use crate::Point;

    fn kw(label: &str, sort: &str) -> CompletionItem {
        CompletionItem::plain(label, CompletionKind::Keyword).with_sort_key(sort)
    }

    struct Stub {
        items: Vec<CompletionItem>,
        calls: u32,
    }
    impl Completions for Stub {
        fn complete(&mut self, _cx: &CompletionCx) -> Vec<CompletionItem> {
            self.calls += 1;
            self.items.clone()
        }
    }

    fn cx(word: &str, trigger: CompletionTrigger) -> CompletionCx {
        let doc = crate::Buffer::new("").unwrap().doc_id();
        CompletionCx {
            doc,
            revision: 0,
            position: Point::new(0, word.len() as u32),
            word: 0..word.len() as u32,
            lookback: word.to_string(),
            trigger,
        }
    }

    fn labels(c: &CompletionController) -> Vec<String> {
        match c.state() {
            CompletionState::Open(list) => {
                list.filtered.iter().map(|&i| list.items[i as usize].label.clone()).collect()
            }
            _ => vec![],
        }
    }

    #[test]
    fn word_char_opens_from_closed_and_prefix_filters() {
        let mut stub = Stub { items: vec![kw("send", "3_send"), kw("set", "3_set"), kw("let", "3_let")], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("s", CompletionTrigger::Typed('s')), "s", &mut stub);
        assert!(c.is_open());
        assert_eq!(labels(&c), ["send", "set"], "prefix 's' in sort_key order");
        assert_eq!(stub.calls, 1);
    }

    #[test]
    fn open_word_char_refilters_without_a_provider_call() {
        let mut stub = Stub { items: vec![kw("send", "3_send"), kw("set", "3_set")], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("s", CompletionTrigger::Typed('s')), "s", &mut stub);
        assert_eq!(stub.calls, 1);
        // Extending the word refilters locally — no second provider call.
        c.on_input(&cx("se", CompletionTrigger::Typed('e')), "se", &mut stub);
        assert_eq!(stub.calls, 1, "Open × word char must not call the provider");
        assert_eq!(labels(&c), ["send", "set"]);
        c.on_input(&cx("set", CompletionTrigger::Typed('t')), "set", &mut stub);
        assert_eq!(labels(&c), ["set"]);
        // Filtering to nothing closes it.
        c.on_input(&cx("setx", CompletionTrigger::Typed('x')), "setx", &mut stub);
        assert!(!c.is_open());
    }

    #[test]
    fn empty_provider_result_closes() {
        let mut stub = Stub { items: vec![], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("s", CompletionTrigger::Typed('s')), "s", &mut stub);
        assert!(!c.is_open());
    }

    #[test]
    fn escape_is_sticky_until_a_boundary_but_a_trigger_char_reopens() {
        let mut stub = Stub { items: vec![kw("send", "3_send")], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("s", CompletionTrigger::Typed('s')), "s", &mut stub);
        assert!(c.escape(), "Escape while open is captured");
        assert!(matches!(c.state(), CompletionState::DismissedUntilBoundary));
        // Extending the word stays dismissed — no reopen, no provider call.
        c.on_input(&cx("se", CompletionTrigger::Typed('e')), "se", &mut stub);
        assert!(!c.is_open());
        assert_eq!(stub.calls, 1);
        // A trigger char reopens even while dismissed.
        c.on_input(&cx("", CompletionTrigger::TriggerChar('(')), "", &mut stub);
        assert!(c.is_open());
        assert_eq!(stub.calls, 2);
        // Re-dismiss, then a word boundary restores normal rules.
        c.escape();
        c.on_boundary();
        assert!(matches!(c.state(), CompletionState::Closed));
        c.on_input(&cx("s", CompletionTrigger::Typed('s')), "s", &mut stub);
        assert!(c.is_open());
    }

    #[test]
    fn selection_wraps_and_accept_returns_it_then_closes() {
        let mut stub = Stub { items: vec![kw("a", "1"), kw("b", "2"), kw("c", "3")], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("", CompletionTrigger::Manual), "", &mut stub);
        assert_eq!(labels(&c), ["a", "b", "c"]);
        c.move_selection(true); // -> b
        c.move_selection(true); // -> c
        c.move_selection(true); // wrap -> a
        c.move_selection(false); // wrap -> c
        let accepted = c.accept().expect("open → accepts");
        assert_eq!(accepted.label, "c");
        assert!(matches!(c.state(), CompletionState::Closed));
        assert!(c.accept().is_none(), "closed → nothing to accept");
    }

    #[test]
    fn set_selected_selects_a_row_and_clamps() {
        let mut stub = Stub { items: vec![kw("a", "1"), kw("b", "2"), kw("c", "3")], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("", CompletionTrigger::Manual), "", &mut stub);
        c.set_selected(1);
        assert_eq!(c.accept().unwrap().label, "b", "clicked row 1");
        // Reopen; a past-the-end index clamps to the last row.
        c.on_input(&cx("", CompletionTrigger::Manual), "", &mut stub);
        c.set_selected(99);
        assert_eq!(c.accept().unwrap().label, "c", "clamped to the last row");
    }

    #[test]
    fn caret_move_closes_and_escape_when_closed_is_not_captured() {
        let mut stub = Stub { items: vec![kw("a", "1")], calls: 0 };
        let mut c = CompletionController::new();
        c.on_input(&cx("a", CompletionTrigger::Typed('a')), "a", &mut stub);
        assert!(c.is_open());
        c.close();
        assert!(matches!(c.state(), CompletionState::Closed));
        assert!(!c.escape(), "Escape while closed is not captured");
    }
}
