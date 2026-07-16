//! Buffer-space coordinates and the primitives every position shares.
//!
//! Two ideas live here: the [`Point`] coordinate (row + byte column) and the
//! [`Bias`] attachment side. Cell columns — the *display* space — are a
//! separate type in the display map; the two spaces are distinct types, so
//! mixing a byte column with a display cell is a compile error rather than a
//! silent off-by-column.

/// Attachment side for a position that sits *between* two characters.
///
/// `Left` means "stick to the character before me"; `Right` means "…after me".
/// It is the single shared vocabulary for clipping to char boundaries (here),
/// tracked-range stickiness (a named pair of biases), and hit-testing.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Debug)]
pub enum Bias {
    /// Bind to the character before the position.
    Left,
    /// Bind to the character after the position.
    Right,
}

/// A buffer-space position: zero-based `row`, and a **byte** `col` within that
/// row's text (the `\n` is not part of the row).
///
/// Columns are byte offsets, not cells and not grapheme clusters — cell columns
/// are [`DisplayPoint`](crate) territory, and grapheme-cluster navigation is a
/// deliberate non-goal. Ordering is row-major, so `Point`s compare in document
/// order.
#[derive(Copy, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Debug, Default)]
pub struct Point {
    /// Zero-based line index.
    pub row: u32,
    /// Byte offset within the row (excludes the trailing `\n`).
    pub col: u32,
}

impl Point {
    /// A `Point` at `(row, col)`.
    #[must_use]
    pub const fn new(row: u32, col: u32) -> Self {
        Self { row, col }
    }

    /// The origin, `(0, 0)`.
    pub const ZERO: Self = Self { row: 0, col: 0 };
}

/// Snap a byte offset to the nearest UTF-8 character boundary of `text`,
/// choosing the direction from `bias`.
///
/// If `offset` already lands on a boundary (or on either end of `text`) it is
/// returned unchanged. Otherwise `Bias::Left` moves to the boundary at or
/// before `offset`, `Bias::Right` to the boundary at or after. Offsets past the
/// end clamp to `text.len()`. This is the char-boundary half of the buffer's
/// clip rules, which keep every stored offset on a valid boundary.
#[must_use]
pub(crate) fn snap_char_boundary(text: &str, offset: u32, bias: Bias) -> u32 {
    let len = text.len() as u32;
    if offset >= len {
        return len;
    }
    let mut o = offset as usize;
    if text.is_char_boundary(o) {
        return offset;
    }
    match bias {
        Bias::Left => {
            while !text.is_char_boundary(o) {
                o -= 1;
            }
        }
        Bias::Right => {
            while !text.is_char_boundary(o) {
                o += 1;
            }
        }
    }
    o as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_order_row_major() {
        assert!(Point::new(0, 5) < Point::new(1, 0));
        assert!(Point::new(1, 2) < Point::new(1, 3));
        assert_eq!(Point::ZERO, Point::new(0, 0));
    }

    #[test]
    fn snap_is_identity_on_ascii_and_ends() {
        let s = "hello";
        for o in 0..=s.len() as u32 {
            assert_eq!(snap_char_boundary(s, o, Bias::Left), o);
            assert_eq!(snap_char_boundary(s, o, Bias::Right), o);
        }
    }

    #[test]
    fn snap_moves_off_multibyte_interior() {
        // "é" is 2 bytes (0xC3 0xA9): "aé" = bytes [a=0][é=1,2], len 3.
        let s = "aé";
        assert_eq!(s.len(), 3);
        // Offset 2 is inside 'é'.
        assert_eq!(snap_char_boundary(s, 2, Bias::Left), 1);
        assert_eq!(snap_char_boundary(s, 2, Bias::Right), 3);
        // Boundaries are untouched.
        assert_eq!(snap_char_boundary(s, 1, Bias::Left), 1);
        assert_eq!(snap_char_boundary(s, 3, Bias::Right), 3);
    }
}
