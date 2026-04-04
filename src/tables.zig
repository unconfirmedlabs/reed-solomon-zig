//! Lookup tables for Leopard-RS: Skew factors for FFT/IFFT, LogWalsh for decoding.
//!
//! Tables are initialized at runtime (too large for comptime with Zig's branch quota limits
//! for the skew table's nested loops). The engine builds them once via `std.once`.

const std = @import("std");
const gf = @import("gf.zig");
const fwht = @import("fwht.zig");
const GfElement = gf.GfElement;
const GF_BITS = gf.GF_BITS;
const GF_ORDER = gf.GF_ORDER;
const GF_MODULUS = gf.GF_MODULUS;
const CANTOR_BASIS = gf.CANTOR_BASIS;

// ── Cantor-basis Exp/Log tables ────────────────────────────────────────
//
// The Rust crate generates its exp/log tables differently from our gf.zig —
// it converts to Cantor basis representation after LFSR generation.
// We must match this exactly for FFT compatibility.

pub const ExpLog = struct {
    exp: [GF_ORDER]GfElement,
    log: [GF_ORDER]GfElement,
};

/// Initialize exp/log tables matching the Rust reference implementation exactly.
/// The key difference from gf.zig's tables: these are in Cantor basis representation.
pub fn initExpLog() ExpLog {
    var exp: [GF_ORDER]GfElement = undefined;
    var log: [GF_ORDER]GfElement = undefined;

    // Step 1: Generate LFSR table (exp[state] = i)
    // Note: Rust stores it as exp[state] = i (inverted from typical convention)
    var state: usize = 1;
    for (0..GF_MODULUS) |i| {
        exp[state] = @truncate(i);
        state <<= 1;
        if (state >= GF_ORDER) {
            state ^= gf.GF_POLYNOMIAL;
        }
    }
    exp[0] = GF_MODULUS;

    // Step 2: Convert to Cantor basis
    log[0] = 0;
    for (0..GF_BITS) |i| {
        const width: usize = @as(usize, 1) << @intCast(i);
        for (0..width) |j| {
            log[j + width] = log[j] ^ CANTOR_BASIS[i];
        }
    }

    // Step 3: Compose tables
    for (0..GF_ORDER) |i| {
        log[i] = exp[log[i]];
    }
    for (0..GF_ORDER) |i| {
        exp[log[i]] = @truncate(i);
    }
    exp[GF_MODULUS] = exp[0];

    return .{ .exp = exp, .log = log };
}

/// Multiply using Cantor-basis tables.
pub fn mul(x: GfElement, log_m: GfElement, el: *const ExpLog) GfElement {
    if (x == 0) return 0;
    return el.exp[gf.addMod(el.log[x], log_m)];
}

// ── Skew table ─────────────────────────────────────────────────────────

pub const Skew = [GF_MODULUS]GfElement;

pub fn initSkew(el: *const ExpLog) Skew {
    var skew: Skew = undefined;
    var temp: [GF_BITS - 1]GfElement = undefined;

    for (1..GF_BITS) |i| {
        temp[i - 1] = @as(GfElement, 1) << @intCast(i);
    }

    for (0..GF_BITS - 1) |m| {
        const step: usize = @as(usize, 1) << @intCast(m + 1);

        skew[(@as(usize, 1) << @intCast(m)) - 1] = 0;

        for (m..GF_BITS - 1) |i| {
            const s: usize = @as(usize, 1) << @intCast(i + 1);
            var j: usize = (@as(usize, 1) << @intCast(m)) - 1;
            while (j < s) : (j += step) {
                skew[j + s] = skew[j] ^ temp[i];
            }
        }

        temp[m] = GF_MODULUS - el.log[mul(temp[m], el.log[temp[m] ^ 1], el)];

        for (m + 1..GF_BITS - 1) |i| {
            const sum = gf.addMod(el.log[temp[i] ^ 1], temp[m]);
            temp[i] = mul(temp[i], sum, el);
        }
    }

    for (0..GF_MODULUS) |i| {
        skew[i] = el.log[skew[i]];
    }

    return skew;
}

// ── LogWalsh table (for decoding) ──────────────────────────────────────

pub const LogWalsh = [GF_ORDER]GfElement;

pub fn initLogWalsh(el: *const ExpLog) LogWalsh {
    var log_walsh: LogWalsh = undefined;
    @memcpy(&log_walsh, &el.log);
    log_walsh[0] = 0;
    fwht.fwht(&log_walsh, GF_ORDER);
    return log_walsh;
}

// ── Mul128 table for SIMD (8 MiB) ─────────────────────────────────────

pub const Mul128Entry = gf.Mul128;

/// Build a single Mul128 entry for a given log_m using Cantor-basis tables.
pub fn buildMul128Entry(log_m: GfElement, el: *const ExpLog) Mul128Entry {
    var result: Mul128Entry = undefined;
    for (0..16) |nibble| {
        const n: GfElement = @truncate(nibble);
        const v0 = mul(n, log_m, el);
        result.lo[0][nibble] = @truncate(v0);
        result.hi[0][nibble] = @truncate(v0 >> 8);

        const v1 = mul(n << 4, log_m, el);
        result.lo[1][nibble] = @truncate(v1);
        result.hi[1][nibble] = @truncate(v1 >> 8);

        const v2 = mul(n << 8, log_m, el);
        result.lo[2][nibble] = @truncate(v2);
        result.hi[2][nibble] = @truncate(v2 >> 8);

        const v3 = mul(n << 12, log_m, el);
        result.lo[3][nibble] = @truncate(v3);
        result.hi[3][nibble] = @truncate(v3 >> 8);
    }
    return result;
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "Cantor-basis exp/log round-trip" {
    const el = initExpLog();
    // For all non-zero elements, exp[log[x]] == x
    for (1..GF_ORDER) |i| {
        const x: GfElement = @truncate(i);
        try testing.expectEqual(x, el.exp[el.log[x]]);
    }
}

test "Cantor-basis mul properties" {
    const el = initExpLog();
    // Identity
    try testing.expectEqual(@as(GfElement, 42), mul(42, el.log[1], &el));
    // Zero
    try testing.expectEqual(@as(GfElement, 0), mul(0, 100, &el));
    // Commutativity (via log form)
    const a: GfElement = 123;
    const b: GfElement = 456;
    const ab = mul(a, el.log[b], &el);
    const ba = mul(b, el.log[a], &el);
    try testing.expectEqual(ab, ba);
}

test "Skew table initialization" {
    const el = initExpLog();
    const skew = initSkew(&el);
    // Skew[0] = log[0] which is GF_MODULUS in Cantor-basis tables
    try testing.expectEqual(GF_MODULUS, skew[0]);
    // Should have some non-zero values
    var has_nonzero = false;
    for (skew[1..]) |s| {
        if (s != 0) {
            has_nonzero = true;
            break;
        }
    }
    try testing.expect(has_nonzero);
}

test "Mul128 matches scalar" {
    const el = initExpLog();
    const log_m: GfElement = 42;
    const lut = buildMul128Entry(log_m, &el);

    for (0..16) |i| {
        const elem: GfElement = @truncate(i * 1000 + 1);
        const lo_byte: u8 = @truncate(elem);
        const hi_byte: u8 = @truncate(elem >> 8);

        // Manual 4-nibble lookup
        var result_lo: u8 = 0;
        var result_hi: u8 = 0;

        result_lo ^= lut.lo[0][lo_byte & 0x0f];
        result_lo ^= lut.lo[1][(lo_byte >> 4) & 0x0f];
        result_lo ^= lut.lo[2][hi_byte & 0x0f];
        result_lo ^= lut.lo[3][(hi_byte >> 4) & 0x0f];

        result_hi ^= lut.hi[0][lo_byte & 0x0f];
        result_hi ^= lut.hi[1][(lo_byte >> 4) & 0x0f];
        result_hi ^= lut.hi[2][hi_byte & 0x0f];
        result_hi ^= lut.hi[3][(hi_byte >> 4) & 0x0f];

        const got: GfElement = @as(GfElement, result_hi) << 8 | result_lo;
        const expected = mul(elem, log_m, &el);
        try testing.expectEqual(expected, got);
    }
}
