//! Shared bench fixtures: a deterministic code-shaped document generator plus
//! a tiny test grammar/theme, so every bench group measures against the same
//! corpus and numbers stay comparable across runs.

use scrive_core::{Diagnostic, Document, Severity, SyntaxDef, TokenTheme};

/// Deterministic code-shaped ASCII: nested `fn`/`switch` blocks (~250 B, 9
/// lines, 8 bracket pairs each), comments, and plenty of `return` needles for
/// the find/occurrence benches. `seed` varies the literals only — the shape
/// (and so the bracket/line structure) is stable across runs.
pub fn gen_doc(target_bytes: usize, seed: u64) -> String {
    let mut rng = seed.max(1);
    let mut next = move || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let mut s = String::with_capacity(target_bytes + 512);
    let mut i = 0usize;
    while s.len() < target_bytes {
        s.push_str(&format!(
            "// block {i}: canned sensor readings for pid 0x{:02X}\n",
            next() & 0xFF
        ));
        s.push_str(&format!("fn pid_{i}(pid: u8) -> u8 {{\n"));
        s.push_str("    switch pid {\n");
        for _ in 0..3 {
            s.push_str(&format!(
                "        0x{:02X} => {{ return {}; }}\n",
                next() & 0xFF,
                next() % 200
            ));
        }
        s.push_str("        _    => { return 0; }\n    }\n}\n\n");
        i += 1;
    }
    s
}

/// The bench corpus sizes. The 100 MB entry is included only when
/// `SCRIVE_BENCH_HUGE=1`, keeping default runs fast enough for routine use.
pub fn sized() -> Vec<(&'static str, String)> {
    let mut v = vec![
        ("100k", gen_doc(100_000, 7)),
        ("1m", gen_doc(1_000_000, 7)),
        ("10m", gen_doc(10_000_000, 7)),
    ];
    if std::env::var("SCRIVE_BENCH_HUGE").as_deref() == Ok("1") {
        v.push(("100m", gen_doc(100_000_000, 7)));
    }
    v
}

/// Spread `n` warning diagnostics evenly through the document, so the
/// keystroke path is benched with a realistic count of live decorations
/// present.
pub fn install_diagnostics(doc: &mut Document, n: u32) {
    let len = doc.buffer().len();
    let step = (len / (n + 1)).max(1);
    let diags: Vec<Diagnostic> = (1..=n)
        .map(|i| Diagnostic::new(i * step..i * step + 2, Severity::Warning, "bench"))
        .collect();
    let rev = doc.revision();
    let _ = doc.set_diagnostics(rev, diags);
}

/// A minimal grammar: keywords (`fn`/`return`/`switch`) coloured, everything
/// else plain — enough for the highlighter to do real per-line tokenizing work.
pub fn syntax() -> SyntaxDef {
    const GRAMMAR: &str = "%YAML 1.2\n\
        ---\n\
        name: Bench\n\
        scope: source.bench\n\
        contexts:\n\
        \x20 main:\n\
        \x20   - match: '\\b(fn|return|switch)\\b'\n\
        \x20     scope: keyword.control.bench\n";
    SyntaxDef::from_sublime_syntax(GRAMMAR).expect("bench grammar parses")
}

pub fn theme() -> TokenTheme {
    const THEME: &str = r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
<key>name</key><string>Bench</string>
<key>settings</key><array>
<dict><key>settings</key><dict><key>background</key><string>#000000</string><key>foreground</key><string>#ffffff</string></dict></dict>
<dict><key>scope</key><string>keyword</string><key>settings</key><dict><key>foreground</key><string>#ff0000</string></dict></dict>
</array></dict></plist>"#;
    TokenTheme::from_tm_theme(THEME).expect("bench theme parses")
}
