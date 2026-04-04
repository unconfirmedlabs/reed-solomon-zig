//! Reed-Solomon erasure coding for Zig.
//!
//! Implements Leopard-RS over GF(2^16) with platform-specific SIMD acceleration.
//! Designed to match the Rust `reed-solomon-simd` crate's output format.

pub const gf = @import("gf.zig");
pub const fwht = @import("fwht.zig");
pub const tables = @import("tables.zig");
pub const engine = @import("engine.zig");
pub const codec = @import("codec.zig");

pub const Encoder = codec.Encoder;
pub const Decoder = codec.Decoder;
pub const Error = codec.Error;

test {
    _ = gf;
    _ = fwht;
    _ = tables;
    _ = engine;
    _ = codec;
}
