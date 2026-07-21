//! `minimal` — the smallest real integration of the batteries-included
//! [`scrive_iced::CodeEditor`] tier: a grammar and the three wires.
//!
//! ```text
//! cargo run -p scrive-iced --example minimal
//! ```
//!
//! Everything else — syntax highlighting (coloured at load, no scroll needed),
//! the find bar (Ctrl+F / Ctrl+H), selection, undo, folding — is on by default.
//! Compare with `scratch.rs`, which drives the low-level [`scrive_iced::Editor`]
//! widget by hand for full control.

use iced::{Element, Subscription, Task};

use scrive_core::SyntaxDef;
use scrive_iced::{CodeEditor, Event};

/// The whole application state: the editor owns its document.
struct App {
    editor: CodeEditor,
}

/// The host message type — it only ever *maps* the editor's opaque [`Event`]; it
/// never matches on it.
#[derive(Debug, Clone)]
enum Message {
    Editor(Event),
}

impl App {
    fn new() -> Self {
        // The grammar is host-supplied (scrive-core ships none); the theme
        // defaults to the bundled Scrive Dark, so `.language(..)` alone colours.
        let grammar = SyntaxDef::from_sublime_syntax(include_str!("assets/rust.sublime-syntax"))
            .expect("bundled Rust grammar parses");
        let source = "fn main() {\n    // edit me — highlighting is on at load\n    println!(\"hello, scrive\");\n}\n";
        Self { editor: CodeEditor::new(source).language(grammar) }
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::Editor(event) => self.editor.update(event).map(Message::Editor),
        }
    }

    fn view(&self) -> Element<'_, Message> {
        self.editor.view().map(Message::Editor)
    }

    fn subscription(&self) -> Subscription<Message> {
        self.editor.subscription().map(Message::Editor)
    }
}

fn main() -> iced::Result {
    let app = iced::application(App::new, App::update, App::view)
        .title("scrive — minimal")
        .subscription(App::subscription);
    // Register the fonts the widget needs (fold chevrons + find-bar icons).
    scrive_iced::required_fonts()
        .iter()
        .fold(app, |app, font| app.font(*font))
        .run()
}
