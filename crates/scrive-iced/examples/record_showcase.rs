//! Deterministic, headless demo recorder for the README GIF.
//!
//! Renders the REAL `scratch` editor frame-by-frame with the tiny-skia renderer
//! — no window, no GPU — while a scripted timeline drives it with genuine
//! keyboard events and app messages (find & replace, fold). Each beat renders
//! one frame,
//! held for its own delay; identical consecutive frames are coalesced. The
//! frames are encoded straight to an animated GIF with the pure-Rust `gif`
//! crate, so the recorder needs no external tools. Same script, same GIF, every
//! run.
//!
//! Run:
//! ```text
//! cargo run -p scrive-iced --example record_showcase --release -- docs/showcase.gif
//! ```

#[path = "scratch.rs"]
#[allow(dead_code)]
mod scratch;

use std::fs::File;
use std::task::{Context, Poll, Waker};

use iced::advanced::widget::Operation;
use iced::advanced::{clipboard, mouse, renderer};
use iced::keyboard::key::{Key, Named, NativeCode, Physical};
use iced::keyboard::{Event as KeyboardEvent, Location, Modifiers};
use iced::{Event, Font, Pixels, Point, Size, Theme};
use iced_runtime::user_interface::{Cache, UserInterface};
use iced_runtime::{task, Action};
use iced_tiny_skia::graphics::Viewport;

use scratch::{scrive_dark, App, Message};
use scrive_iced::Event as CeEvent;

const W: u32 = 900;
const H: u32 = 560;

/// One thing the script does before a frame is rendered.
enum Step {
    /// A real widget event (a key press), dispatched through the UI.
    Ev(Event),
    /// An app message, fed straight into `App::update` (the chrome the widget
    /// does not own — find bar open/close, query, navigation).
    Msg(Message),
}

/// A KeyPressed carrying `text` (what the editor types) and modifiers.
fn key_event(key: Key, text: Option<&str>, modifiers: Modifiers) -> Event {
    Event::Keyboard(KeyboardEvent::KeyPressed {
        key: key.clone(),
        modified_key: key,
        physical_key: Physical::Unidentified(NativeCode::Unidentified),
        location: Location::Standard,
        modifiers,
        text: text.map(Into::into),
        repeat: false,
    })
}

/// Type one character at the caret.
fn typed(c: &str) -> Step {
    Step::Ev(key_event(Key::Character(c.into()), Some(c), Modifiers::empty()))
}

/// Press a named key (optionally with modifiers).
fn press(named: Named, modifiers: Modifiers) -> Step {
    Step::Ev(key_event(Key::Named(named), None, modifiers))
}

/// A `Ctrl+<char>` chord (fold, unfold, …).
fn chord(c: &str, modifiers: Modifiers) -> Step {
    Step::Ev(key_event(Key::Character(c.into()), None, modifiers | Modifiers::CTRL))
}

/// The demo timeline: `(steps, hold)` — apply the steps, render one frame, hold
/// it for `hold` centiseconds. The single source of truth for what the GIF shows.
fn script() -> Vec<(Vec<Step>, u16)> {
    let mut t: Vec<(Vec<Step>, u16)> = Vec::new();
    let mut beat = |steps: Vec<Step>, hold: u16| t.push((steps, hold));

    // Open on the sample.
    beat(vec![], 110);

    // Drop the caret to the `values` field and into the word itself.
    for _ in 0..7 {
        beat(vec![press(Named::ArrowDown, Modifiers::empty())], 9);
    }
    for _ in 0..6 {
        beat(vec![press(Named::ArrowRight, Modifiers::empty())], 8);
    }
    beat(vec![], 45);

    // Multi-cursor: select `values`, then Ctrl+D each next occurrence — a caret
    // and selection appear on every one.
    for _ in 0..7 {
        beat(vec![chord("d", Modifiers::empty())], 27);
    }
    beat(vec![], 85);
    // Type once, edit everywhere: rename every selected occurrence at the same time.
    for c in ["b", "u", "f", "f", "e", "r"] {
        beat(vec![typed(c)], 14);
    }
    beat(vec![], 100);
    // Collapse back to a single caret.
    beat(vec![press(Named::Escape, Modifiers::empty())], 45);

    // Find & replace: open the replace bar (Ctrl+H), search a word, step a
    // match, then replace every occurrence at once.
    beat(vec![press(Named::Home, Modifiers::CTRL)], 20);
    beat(vec![Step::Msg(Message::Editor(CeEvent::OpenReplace))], 34);
    for q in ["r", "re", "rea", "read", "readi", "readin", "reading"] {
        beat(vec![Step::Msg(Message::Editor(CeEvent::FindQuery(q.into())))], 11);
    }
    beat(vec![], 55);
    // Walk the matches — the active one brightens, the count updates.
    beat(vec![Step::Msg(Message::Editor(CeEvent::FindNext))], 46);
    beat(vec![Step::Msg(Message::Editor(CeEvent::FindNext))], 46);
    beat(vec![], 30);
    // Type the replacement into the second row.
    for r in ["s", "sa", "sam", "samp", "sampl", "sample"] {
        beat(vec![Step::Msg(Message::Editor(CeEvent::ReplaceText(r.into())))], 12);
    }
    beat(vec![], 55);
    // Replace all — every `reading` becomes `sample` in one step.
    beat(vec![Step::Msg(Message::Editor(CeEvent::ReplaceAll))], 85);
    // Close to a clean view of the replaced text — the payoff — then hold.
    beat(vec![Step::Msg(Message::Editor(CeEvent::CloseFind))], 18);
    beat(vec![], 120);
    // Undo the replace so the rest of the tour runs on the original text (Undo
    // goes to the editor now the bar has handed focus back).
    beat(vec![chord("z", Modifiers::empty())], 55);
    beat(vec![], 35);

    // Fold a block from inside it, then unfold it.
    beat(vec![press(Named::Home, Modifiers::CTRL)], 25);
    for _ in 0..12 {
        beat(vec![press(Named::ArrowDown, Modifiers::empty())], 7);
    }
    beat(vec![], 35);
    beat(vec![chord("[", Modifiers::empty())], 95);
    beat(vec![chord("]", Modifiers::empty())], 80);
    beat(vec![], 55);

    // Completion: on a fresh line, type a prefix — the popup filters live.
    beat(vec![press(Named::End, Modifiers::empty())], 22);
    beat(vec![press(Named::Enter, Modifiers::empty())], 22);
    for c in ["s", "t", "r"] {
        beat(vec![typed(c)], 16);
    }
    beat(vec![], 90);
    beat(vec![press(Named::ArrowDown, Modifiers::empty())], 32);
    beat(vec![press(Named::Escape, Modifiers::empty())], 25);
    // Undo the completion demo edit (the `str` line) so the view settles back.
    beat(vec![chord("z", Modifiers::empty())], 30);
    beat(vec![press(Named::Home, Modifiers::CTRL)], 20);
    beat(vec![], 150);

    t
}

/// Drain a `Task` synchronously: ready message outputs go back into the inbox,
/// widget operations (focus, scroll) queue for the next UI build.
fn drain(task: iced::Task<Message>, inbox: &mut Vec<Message>, ops: &mut Vec<Box<dyn Operation>>) {
    let Some(mut stream) = task::into_stream(task) else {
        return;
    };
    let waker = Waker::noop();
    let mut cx = Context::from_waker(waker);
    loop {
        match stream.as_mut().poll_next(&mut cx) {
            Poll::Ready(Some(Action::Output(msg))) => inbox.push(msg),
            Poll::Ready(Some(Action::Widget(op))) => ops.push(op),
            Poll::Ready(Some(_)) => {}
            Poll::Ready(None) | Poll::Pending => break,
        }
    }
}

fn main() {
    let out = std::env::args().nth(1).unwrap_or_else(|| "showcase.gif".into());

    // Load every font the widget needs (fold chevrons + find-bar icons) into the
    // headless font system, or they render as tofu.
    let mut fs = iced_tiny_skia::graphics::text::font_system()
        .write()
        .expect("font system lock");
    for font in scrive_iced::required_fonts() {
        fs.load_font(std::borrow::Cow::Borrowed(font));
    }
    drop(fs);

    let mut renderer = iced_renderer::fallback::Renderer::Secondary(
        iced_tiny_skia::Renderer::new(Font::default(), Pixels(14.0)),
    );
    let viewport = Viewport::with_physical_size(Size::new(W, H), 1.0);
    let logical = Size::new(W as f32, H as f32);
    let cursor = mouse::Cursor::Available(Point::new(W as f32 / 2.0, H as f32 / 2.0));
    let theme: Theme = scrive_dark();

    let mut app = App::default();
    let mut cache = Cache::new();
    let mut inbox: Vec<Message> = Vec::new();
    let mut ops: Vec<Box<dyn Operation>> = Vec::new();

    // Encode frames as we go: (rgba, delay). Coalesce identical frames.
    let file = File::create(&out).expect("create GIF file");
    let mut encoder =
        gif::Encoder::new(std::io::BufWriter::new(file), W as u16, H as u16, &[]).expect("gif encoder");
    encoder.set_repeat(gif::Repeat::Infinite).expect("gif repeat");
    let mut pending: Option<(Vec<u8>, u16)> = None;
    let mut frames = 0u32;

    // Prime highlighting before the first recorded frame. The editor reports its
    // viewport (`ViewportChanged`) during `update`, and the app tokenizes the
    // visible rows only when it processes that message — which, in the frame
    // loop below, happens AFTER the draw. So the very first frame would paint in
    // the fallback (unhighlighted) style. Run one throwaway update+settle here,
    // discarding the draw, so the doc's highlight cache is warm when frame 1 is
    // drawn.
    {
        let events = [Event::Window(iced::window::Event::RedrawRequested(
            std::time::Instant::now(),
        ))];
        let mut ui: UserInterface<'_, Message, Theme, iced::Renderer> =
            UserInterface::build(app.view(), logical, cache, &mut renderer);
        let mut published: Vec<Message> = Vec::new();
        let _ = ui.update(&events, cursor, &mut renderer, &mut clipboard::Null, &mut published);
        cache = ui.into_cache();
        inbox.extend(published);
        while !inbox.is_empty() {
            for m in std::mem::take(&mut inbox) {
                drain(app.update(m), &mut inbox, &mut ops);
            }
        }
    }

    for (steps, hold) in script() {
        let mut events: Vec<Event> = vec![Event::Window(iced::window::Event::RedrawRequested(
            std::time::Instant::now(),
        ))];
        for step in steps {
            match step {
                Step::Ev(e) => events.push(e),
                Step::Msg(m) => inbox.push(m),
            }
        }

        // Settle any queued messages first (find open focuses via a Task, etc.).
        while !inbox.is_empty() {
            for m in std::mem::take(&mut inbox) {
                drain(app.update(m), &mut inbox, &mut ops);
            }
        }

        // Build → apply queued widget ops → dispatch events → draw.
        let mut ui: UserInterface<'_, Message, Theme, iced::Renderer> =
            UserInterface::build(app.view(), logical, cache, &mut renderer);
        for mut op in ops.drain(..) {
            ui.operate(&renderer, op.as_mut());
        }
        let mut published: Vec<Message> = Vec::new();
        let _ = ui.update(&events, cursor, &mut renderer, &mut clipboard::Null, &mut published);
        ui.draw(&mut renderer, &theme, &renderer::Style::default(), cursor);
        cache = ui.into_cache();

        // Feed back what the widget published (editor Actions), and settle.
        inbox.extend(published);
        while !inbox.is_empty() {
            for m in std::mem::take(&mut inbox) {
                drain(app.update(m), &mut inbox, &mut ops);
            }
        }

        // Rasterize offscreen.
        let iced_renderer::fallback::Renderer::Secondary(ts) = &mut renderer else {
            unreachable!("constructed the tiny-skia variant above");
        };
        let rgba = iced_tiny_skia::window::compositor::screenshot(ts, &viewport, theme.palette().background);

        // Coalesce a frame identical to the previous one into extra delay.
        match &mut pending {
            Some((prev, d)) if *prev == rgba => *d = d.saturating_add(hold),
            _ => {
                if let Some((prev, d)) = pending.take() {
                    write_frame(&mut encoder, prev, d);
                    frames += 1;
                }
                pending = Some((rgba, hold));
            }
        }
    }
    if let Some((prev, d)) = pending.take() {
        write_frame(&mut encoder, prev, d);
        frames += 1;
    }
    drop(encoder);
    eprintln!("wrote {out} ({frames} frames, {W}x{H})");
}

fn write_frame<W: std::io::Write>(encoder: &mut gif::Encoder<W>, mut rgba: Vec<u8>, delay: u16) {
    let mut frame = gif::Frame::from_rgba_speed(W as u16, H as u16, &mut rgba, 10);
    frame.delay = delay;
    encoder.write_frame(&frame).expect("write gif frame");
}
