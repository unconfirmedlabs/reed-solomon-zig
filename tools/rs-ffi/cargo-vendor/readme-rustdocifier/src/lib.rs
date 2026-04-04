#![doc = include_str!(concat!(env!("OUT_DIR"), "/README-rustdocified.md"))]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

pub use inner::{rustdocify, Error};

mod inner;
