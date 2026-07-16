//! Headless render-to-PNG — the self-validation capture path.
//!
//! Renders one frame of an iced view with the tiny-skia renderer (no window, no
//! GPU) and writes it to a PNG. Lives in `examples/shared/` (a subdirectory, so
//! cargo does not treat it as a standalone example) and is pulled into examples
//! via `#[path = "shared/capture.rs"] mod capture;`, so any example can
//! screenshot its widget for review. The rasterization mirrors
//! `iced_tiny_skia::window::compositor::screenshot`, the same path the runtime
//! uses for real window screenshots.

use std::fs::File;
use std::io::BufWriter;
use std::path::Path;

use iced::advanced::{clipboard, mouse, renderer};
use iced::{Element, Font, Pixels, Point, Size, Theme};
use iced_runtime::user_interface::{Cache, UserInterface};
use iced_tiny_skia::graphics::Viewport;

/// Render `view` at `width`×`height` logical pixels under `theme` and write the
/// result to `path` as an RGBA PNG. Returns the PNG's pixel dimensions.
///
/// `events` are dispatched to the widget tree before drawing (after an implicit
/// `RedrawRequested`), with the cursor at the viewport center — enough to
/// exercise wheel scrolling and other input-driven widget state in a single
/// headless frame. (Cross-frame flows like autoscroll need a real app loop; the
/// arithmetic for those is unit-tested instead.)
#[allow(dead_code)] // static-scene helper; used by the example capture scenes
pub fn render_to_png<Message>(
    view: Element<'_, Message, Theme, iced::Renderer>,
    width: u32,
    height: u32,
    theme: &Theme,
    events: &[iced::Event],
    path: impl AsRef<Path>,
) -> (u32, u32) {
    // Match the interactive app: load the bundled Codicon font into the global
    // font system so captured frames render the fold chevrons / find-bar glyphs
    // (otherwise cosmic-text would draw missing-glyph boxes). Idempotent.
    iced_tiny_skia::graphics::text::font_system()
        .write()
        .expect("font system lock")
        .load_font(std::borrow::Cow::Borrowed(scrive_iced::CODICON_FONT));
    let mut renderer = iced_renderer::fallback::Renderer::Secondary(
        iced_tiny_skia::Renderer::new(Font::default(), Pixels(14.0)),
    );
    let viewport = Viewport::with_physical_size(Size::new(width, height), 1.0);
    let logical = Size::new(width as f32, height as f32);
    let cursor = mouse::Cursor::Available(Point::new(width as f32 / 2.0, height as f32 / 2.0));

    // Build → update (RedrawRequested + caller events) → draw.
    let mut ui: UserInterface<'_, Message, Theme, iced::Renderer> =
        UserInterface::build(view, logical, Cache::new(), &mut renderer);
    let mut all = vec![iced::Event::Window(iced::window::Event::RedrawRequested(
        std::time::Instant::now(),
    ))];
    all.extend_from_slice(events);
    let mut published: Vec<Message> = Vec::new();
    ui.update(&all, cursor, &mut renderer, &mut clipboard::Null, &mut published);
    ui.draw(&mut renderer, theme, &renderer::Style::default(), cursor);

    // Rasterize the tiny-skia variant to raw RGBA.
    let iced_renderer::fallback::Renderer::Secondary(ts) = &mut renderer else {
        unreachable!("constructed the tiny-skia variant above");
    };
    let rgba =
        iced_tiny_skia::window::compositor::screenshot(ts, &viewport, theme.palette().background);

    encode_png(&rgba, width, height, path);
    (width, height)
}

/// Encode raw RGBA bytes to a PNG file.
fn encode_png(rgba: &[u8], width: u32, height: u32, path: impl AsRef<Path>) {
    let file = File::create(&path).expect("create PNG file");
    let mut encoder = png::Encoder::new(BufWriter::new(file), width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .expect("write PNG header")
        .write_image_data(rgba)
        .expect("write PNG data");
}

/// Render an interactive app to PNG, looping published messages back into
/// `state` — so mouse gestures (press → drag → release) are exercised end to
/// end, not merely rendered. `events` are dispatched one at a time; a
/// `CursorMoved` updates the tracked pointer first, so a following
/// `ButtonPressed`/drag sees the cursor where the script last moved it. The
/// final state is drawn.
#[allow(dead_code)] // used by the examples that validate mouse gestures
#[allow(clippy::too_many_arguments)] // a headless render needs all of them
pub fn render_interactive_to_png<State, Msg>(
    mut state: State,
    mut update: impl FnMut(&mut State, Msg),
    view: impl Fn(&State) -> Element<'_, Msg, Theme, iced::Renderer>,
    width: u32,
    height: u32,
    theme: &Theme,
    events: &[iced::Event],
    path: impl AsRef<Path>,
) -> (u32, u32) {
    // Match the interactive app: load the bundled Codicon font into the global
    // font system so captured frames render the fold chevrons / find-bar glyphs
    // (otherwise cosmic-text would draw missing-glyph boxes). Idempotent.
    iced_tiny_skia::graphics::text::font_system()
        .write()
        .expect("font system lock")
        .load_font(std::borrow::Cow::Borrowed(scrive_iced::CODICON_FONT));
    let mut renderer = iced_renderer::fallback::Renderer::Secondary(
        iced_tiny_skia::Renderer::new(Font::default(), Pixels(14.0)),
    );
    let viewport = Viewport::with_physical_size(Size::new(width, height), 1.0);
    let logical = Size::new(width as f32, height as f32);
    let mut cache = Cache::new();
    let mut cursor_pos = Point::new(width as f32 / 2.0, height as f32 / 2.0);

    for event in events {
        if let iced::Event::Mouse(mouse::Event::CursorMoved { position }) = event {
            cursor_pos = *position;
        }
        let cursor = mouse::Cursor::Available(cursor_pos);
        let mut ui = UserInterface::build(view(&state), logical, cache, &mut renderer);
        let mut messages: Vec<Msg> = Vec::new();
        ui.update(
            std::slice::from_ref(event),
            cursor,
            &mut renderer,
            &mut clipboard::Null,
            &mut messages,
        );
        cache = ui.into_cache();
        for message in messages {
            update(&mut state, message);
        }
    }

    let cursor = mouse::Cursor::Available(cursor_pos);
    let mut ui = UserInterface::build(view(&state), logical, cache, &mut renderer);
    ui.draw(&mut renderer, theme, &renderer::Style::default(), cursor);

    let iced_renderer::fallback::Renderer::Secondary(ts) = &mut renderer else {
        unreachable!("constructed the tiny-skia variant above");
    };
    let rgba =
        iced_tiny_skia::window::compositor::screenshot(ts, &viewport, theme.palette().background);
    encode_png(&rgba, width, height, path);
    (width, height)
}
