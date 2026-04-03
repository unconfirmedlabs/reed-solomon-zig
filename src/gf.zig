//! GF(2^16) Galois field arithmetic for Leopard-RS.
//!
//! Field: GF(2^16) with irreducible polynomial x^16 + x^5 + x^3 + x^2 + 1 (0x1002D).
//! Uses Cantor basis representation for FFT-friendly operations.
//! All tables are computed at comptime — zero runtime initialization.

const std = @import("std");
const builtin = @import("builtin");

// ── Constants ──────────────────────────────────────────────────────────

pub const GF_BITS: usize = 16;
pub const GF_ORDER: usize = 65536;
pub const GF_MODULUS: u16 = 65535;
pub const GF_POLYNOMIAL: usize = 0x1002D;

pub const GfElement = u16;

pub const CANTOR_BASIS: [GF_BITS]GfElement = .{
    0x0001, 0xACCA, 0x3C0E, 0x163E, 0xC582, 0xED2E, 0x914C, 0x4012,
    0x6C98, 0x10D8, 0x6A72, 0xB900, 0xFDB8, 0xFB34, 0xFF38, 0x991E,
};

// ── Modular arithmetic ────────────────────────────────────────────────

/// Add two values modulo GF_MODULUS (65535).
/// Uses the bit trick: (a + b + carry) where carry = (sum >> 16).
pub inline fn addMod(a: GfElement, b: GfElement) GfElement {
    const sum: u32 = @as(u32, a) + @as(u32, b);
    return @truncate(sum + (sum >> 16));
}

/// Subtract modulo GF_MODULUS.
pub inline fn subMod(a: GfElement, b: GfElement) GfElement {
    const diff: i32 = @as(i32, a) - @as(i32, b);
    // If diff < 0, add GF_MODULUS
    return @intCast(@as(u32, @bitCast(diff + (@as(i32, GF_MODULUS) & (diff >> 31)))));
}

// ── Log/Exp tables (comptime) ──────────────────────────────────────────

/// Exponentiation table: exp[i] = g^i in GF(2^16).
/// Generated from LFSR with GF_POLYNOMIAL.
pub const exp_table: [GF_ORDER]GfElement = blk: {
    @setEvalBranchQuota(200_000);
    var table: [GF_ORDER]GfElement = undefined;
    var state: u32 = 1;
    for (0..GF_MODULUS) |i| {
        table[i] = @truncate(state);
        state <<= 1;
        if (state >= GF_ORDER) {
            state ^= GF_POLYNOMIAL;
        }
    }
    table[GF_MODULUS] = table[0]; // wrap-around
    break :blk table;
};

/// Logarithm table: log[x] = i where g^i = x.
pub const log_table: [GF_ORDER]GfElement = blk: {
    @setEvalBranchQuota(200_000);
    var table: [GF_ORDER]GfElement = undefined;
    table[0] = 0; // log(0) is undefined, use 0
    for (0..GF_MODULUS) |i| {
        table[exp_table[i]] = @truncate(i);
    }
    break :blk table;
};

// ── Scalar operations ──────────────────────────────────────────────────

/// Multiply two GF elements using log/exp tables.
pub inline fn mul(a: GfElement, b: GfElement) GfElement {
    if (a == 0 or b == 0) return 0;
    return exp_table[addMod(log_table[a], log_table[b])];
}

/// Multiplicative inverse. Returns 0 for inv(0) (mathematically undefined).
pub inline fn inv(a: GfElement) GfElement {
    if (a == 0) return 0;
    return exp_table[GF_MODULUS - log_table[a]];
}

/// Multiply by log form: result = x * g^log_m
pub inline fn mulLog(x: GfElement, log_m: GfElement) GfElement {
    if (x == 0) return 0;
    return exp_table[addMod(log_table[x], log_m)];
}

// ── SIMD multiply (split-table technique) ──────────────────────────────
//
// GF(2^16) elements are stored as 2 bytes (lo, hi) in split layout:
// bytes[0..32] = lo bytes of 32 elements
// bytes[32..64] = hi bytes of 32 elements
//
// To multiply all elements by a constant (in log form), use 4-nibble
// decomposition with 16-byte lookup tables:
//   result_lo = tbl(t0_lo, lo & 0x0f) ^ tbl(t1_lo, lo >> 4)
//                ^ tbl(t2_lo, hi & 0x0f) ^ tbl(t3_lo, hi >> 4)
//   result_hi = (same with t0_hi..t3_hi)

/// Precomputed multiply tables for SIMD: 8 × 16-byte tables per log_m.
pub const Mul128 = struct {
    lo: [4][16]u8, // 4 nibble tables for result low byte
    hi: [4][16]u8, // 4 nibble tables for result high byte
};

/// Build SIMD multiply tables for a given log_m value.
pub fn buildMul128(log_m: GfElement) Mul128 {
    var result: Mul128 = undefined;

    for (0..16) |nibble| {
        const n: GfElement = @truncate(nibble);

        // Nibble 0: lo byte, bits 0-3
        const v0 = mulLog(n, log_m);
        result.lo[0][nibble] = @truncate(v0);
        result.hi[0][nibble] = @truncate(v0 >> 8);

        // Nibble 1: lo byte, bits 4-7
        const v1 = mulLog(n << 4, log_m);
        result.lo[1][nibble] = @truncate(v1);
        result.hi[1][nibble] = @truncate(v1 >> 8);

        // Nibble 2: hi byte, bits 0-3
        const v2 = mulLog(n << 8, log_m);
        result.lo[2][nibble] = @truncate(v2);
        result.hi[2][nibble] = @truncate(v2 >> 8);

        // Nibble 3: hi byte, bits 4-7
        const v3 = mulLog(n << 12, log_m);
        result.lo[3][nibble] = @truncate(v3);
        result.hi[3][nibble] = @truncate(v3 >> 8);
    }

    return result;
}

// ── SIMD table lookup ──────────────────────────────────────────────────

/// 16-byte SIMD table lookup.
/// ARM AArch64: tbl instruction (NEON)
/// x86/x86_64 with SSSE3: pshufb instruction
/// Fallback: scalar loop
pub inline fn tblLookup(table: @Vector(16, u8), indices: @Vector(16, u8)) @Vector(16, u8) {
    if (comptime builtin.cpu.arch == .aarch64) {
        return asm ("tbl %[out].16b, {%[tbl].16b}, %[idx].16b"
            : [out] "=w" (-> @Vector(16, u8)),
            : [tbl] "w" (table),
              [idx] "w" (indices),
        );
    } else if (comptime (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86)) {
        // x86 hot path uses AVX2 C FFI (simd_x86.c) for the engine layer.
        // This scalar fallback is only used for tests and non-hot-path code.
        // Zig's x86 inline asm encoder has bugs with pshufb in inlined contexts.
        var result: [16]u8 = undefined;
        const t: [16]u8 = table;
        const idx: [16]u8 = indices;
        inline for (0..16) |i| {
            result[i] = t[idx[i] & 0x0f];
        }
        return result;
    } else {
        var result: [16]u8 = undefined;
        const t: [16]u8 = table;
        const idx: [16]u8 = indices;
        inline for (0..16) |i| {
            result[i] = t[idx[i] & 0x0f];
        }
        return result;
    }
}

/// Multiply 16 GF(2^16) elements (stored as 32 bytes: lo[0..16] + hi[0..16])
/// by a constant using precomputed Mul128 tables.
pub fn mulSimd32(data_lo: @Vector(16, u8), data_hi: @Vector(16, u8), lut: *const Mul128) struct { lo: @Vector(16, u8), hi: @Vector(16, u8) } {
    const mask: @Vector(16, u8) = @splat(0x0f);

    // 4 nibble lookups for result_lo
    var prod_lo = tblLookup(lut.lo[0], data_lo & mask);
    prod_lo ^= tblLookup(lut.lo[1], (data_lo >> @splat(4)) & mask);
    prod_lo ^= tblLookup(lut.lo[2], data_hi & mask);
    prod_lo ^= tblLookup(lut.lo[3], (data_hi >> @splat(4)) & mask);

    // 4 nibble lookups for result_hi
    var prod_hi = tblLookup(lut.hi[0], data_lo & mask);
    prod_hi ^= tblLookup(lut.hi[1], (data_lo >> @splat(4)) & mask);
    prod_hi ^= tblLookup(lut.hi[2], data_hi & mask);
    prod_hi ^= tblLookup(lut.hi[3], (data_hi >> @splat(4)) & mask);

    return .{ .lo = prod_lo, .hi = prod_hi };
}

/// Multiply a 64-byte chunk (32 GF(2^16) elements in split layout) by log_m.
/// Layout: bytes[0..32] = lo bytes, bytes[32..64] = hi bytes.
pub fn mulChunk64(chunk: *[64]u8, lut: *const Mul128) void {
    // Process low 16 elements
    const lo0: @Vector(16, u8) = chunk[0..16].*;
    const hi0: @Vector(16, u8) = chunk[32..48].*;
    const r0 = mulSimd32(lo0, hi0, lut);
    chunk[0..16].* = r0.lo;
    chunk[32..48].* = r0.hi;

    // Process high 16 elements
    const lo1: @Vector(16, u8) = chunk[16..32].*;
    const hi1: @Vector(16, u8) = chunk[48..64].*;
    const r1 = mulSimd32(lo1, hi1, lut);
    chunk[16..32].* = r1.lo;
    chunk[48..64].* = r1.hi;
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "GF(2^16) basic properties" {
    // 1 is multiplicative identity
    try testing.expectEqual(@as(GfElement, 42), mul(42, 1));
    try testing.expectEqual(@as(GfElement, 42), mul(1, 42));

    // 0 annihilates
    try testing.expectEqual(@as(GfElement, 0), mul(0, 42));
    try testing.expectEqual(@as(GfElement, 0), mul(42, 0));

    // Commutativity
    try testing.expectEqual(mul(123, 456), mul(456, 123));

    // Inverse
    const x: GfElement = 12345;
    try testing.expectEqual(@as(GfElement, 1), mul(x, inv(x)));
}

test "GF(2^16) addMod/subMod" {
    try testing.expectEqual(@as(GfElement, 0), addMod(0, 0));
    try testing.expectEqual(@as(GfElement, 100), addMod(50, 50));
    try testing.expectEqual(@as(GfElement, 0), subMod(100, 100));

    // Wrap-around
    const a: GfElement = GF_MODULUS - 10;
    const b: GfElement = 20;
    const sum = addMod(a, b);
    try testing.expectEqual(a, subMod(sum, b));
}

test "GF(2^16) exp/log round-trip" {
    // For all non-zero elements, exp[log[x]] == x
    for (1..GF_ORDER) |i| {
        const x: GfElement = @truncate(i);
        try testing.expectEqual(x, exp_table[log_table[x]]);
    }
}

test "SIMD multiply matches scalar" {
    const log_m: GfElement = 42;
    const lut = buildMul128(log_m);

    // Create 16 test elements
    var lo: [16]u8 = undefined;
    var hi: [16]u8 = undefined;
    for (0..16) |i| {
        const elem: GfElement = @truncate(i * 1000 + 1);
        lo[i] = @truncate(elem);
        hi[i] = @truncate(elem >> 8);
    }

    const result = mulSimd32(lo, hi, &lut);
    const result_lo: [16]u8 = result.lo;
    const result_hi: [16]u8 = result.hi;

    // Verify against scalar
    for (0..16) |i| {
        const elem: GfElement = @as(GfElement, hi[i]) << 8 | lo[i];
        const expected = mulLog(elem, log_m);
        const got: GfElement = @as(GfElement, result_hi[i]) << 8 | result_lo[i];
        try testing.expectEqual(expected, got);
    }
}

test "SIMD mulChunk64 matches scalar" {
    const log_m: GfElement = 9999;
    const lut = buildMul128(log_m);

    // Fill a 64-byte chunk with 32 GF elements in split layout
    var chunk: [64]u8 = undefined;
    var expected: [32]GfElement = undefined;
    for (0..32) |i| {
        const elem: GfElement = @truncate(i * 2000 + 7);
        chunk[i] = @truncate(elem); // lo byte
        chunk[32 + i] = @truncate(elem >> 8); // hi byte
        expected[i] = mulLog(elem, log_m);
    }

    mulChunk64(&chunk, &lut);

    // Verify
    for (0..32) |i| {
        const got: GfElement = @as(GfElement, chunk[32 + i]) << 8 | chunk[i];
        try testing.expectEqual(expected[i], got);
    }
}
