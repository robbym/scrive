//! Language services — completion, signature help, hover.
//!
//! In-process, no LSP. Each service is **one small trait defined here in the
//! core and satisfied by the app** — not a god-trait with no-op defaults; a
//! service ruled out of scope is simply not built. The plain data the traits
//! exchange lives here too, so a provider never reaches an editor internal.
//! Completion is **synchronous by contract** (the classifier is a regex
//! ladder over a few dozen lines — microseconds), so the widget calls the
//! provider in `update()` and opens the popup from the returned `Vec` the same
//! frame; no async reply, no revision guard.
//!
//! The controller state machines (completion / snippet session) that consume
//! these providers are core view-state and land in the submodules below.

pub mod completion;
pub mod hover;
pub mod providers;
pub mod signature;
pub mod snippet;
