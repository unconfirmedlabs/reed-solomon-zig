//! Leopard-RS engine: FFT, IFFT, multiply, and shard operations.
//!
//! SIMD-accelerated using ARM NEON tbl / scalar fallback for GF(2^16) multiply.
//! Uses 2-layer butterfly optimization for FFT/IFFT (processes 4 shards at once).

const std = @import("std");
const builtin = @import("builtin");
const gf = @import("gf.zig");
const tables = @import("tables.zig");
const fwht_mod = @import("fwht.zig");
const GfElement = gf.GfElement;
const GF_BITS = gf.GF_BITS;
const GF_ORDER = gf.GF_ORDER;
const GF_MODULUS = gf.GF_MODULUS;
const Allocator = std.mem.Allocator;

// ── SIMD primitives ────────────────────────────────────────────────────

const V16 = @Vector(16, u8);
const use_neon = (builtin.cpu.arch == .aarch64);

inline fn tblLookup(table: V16, indices: V16) V16 {
    if (comptime use_neon) {
        return asm ("tbl %[out].16b, {%[tbl].16b}, %[idx].16b"
            : [out] "=w" (-> V16),
            : [tbl] "w" (table),
              [idx] "w" (indices),
        );
    } else {
        var result: [16]u8 = undefined;
        const t: [16]u8 = table;
        const idx: [16]u8 = indices;
        inline for (0..16) |i| result[i] = t[idx[i] & 0x0f];
        return result;
    }
}

const mask_0f: V16 = @splat(0x0f);
const shift_4: @Vector(16, u3) = @splat(4);

// ── Shard types ────────────────────────────────────────────────────────

pub const Chunk = [64]u8;

pub const Shards = struct {
    shard_count: usize,
    shard_len: usize,
    data: []Chunk,
    allocator: Allocator,

    pub fn init(allocator: Allocator, shard_count: usize, shard_len: usize) !Shards {
        const data = try allocator.alloc(Chunk, shard_count * shard_len);
        @memset(std.mem.sliceAsBytes(data), 0);
        return .{ .shard_count = shard_count, .shard_len = shard_len, .data = data, .allocator = allocator };
    }

    pub fn deinit(self: *Shards) void { self.allocator.free(self.data); }

    pub fn shard(self: *const Shards, index: usize) []const Chunk {
        const s = index * self.shard_len;
        return self.data[s .. s + self.shard_len];
    }

    pub fn shardMut(self: *Shards, index: usize) []Chunk {
        const s = index * self.shard_len;
        return self.data[s .. s + self.shard_len];
    }

    pub fn insert(self: *Shards, index: usize, shard_data: []const u8) void {
        const whole = shard_data.len / 64;
        const tail = shard_data.len % 64;
        const dst = self.shardMut(index);
        for (0..whole) |i| dst[i] = shard_data[i * 64 ..][0..64].*;
        if (tail > 0) {
            const half = tail / 2;
            @memcpy(dst[whole][0..half], shard_data[whole * 64 ..][0..half]);
            @memcpy(dst[whole][32 .. 32 + half], shard_data[whole * 64 + half ..][0..tail - half]);
        }
    }

    pub fn zero_range(self: *Shards, start: usize, end: usize) void {
        if (start >= end) return;
        @memset(std.mem.sliceAsBytes(self.data[start * self.shard_len .. end * self.shard_len]), 0);
    }

    pub fn zeroShard(self: *Shards, index: usize) void {
        for (self.shardMut(index)) |*c| c.* = .{0} ** 64;
    }

    pub fn copy_within(self: *Shards, src: usize, dest: usize, count: usize) void {
        const s = src * self.shard_len;
        const d = dest * self.shard_len;
        const c = count * self.shard_len;
        std.mem.copyBackwards(Chunk, self.data[d .. d + c], self.data[s .. s + c]);
    }

    pub fn dist2(self: *Shards, pos: usize, dist: usize) struct { a: []Chunk, b: []Chunk } {
        const a_s = pos * self.shard_len;
        const b_s = (pos + dist) * self.shard_len;
        return .{ .a = self.data[a_s .. a_s + self.shard_len], .b = self.data[b_s .. b_s + self.shard_len] };
    }
};

// ── XOR (SIMD) ─────────────────────────────────────────────────────────

pub inline fn xorChunks(dst: []Chunk, src: []const Chunk) void {
    for (dst, src) |*d, s| {
        inline for (0..4) |q| {
            const off = q * 16;
            d[off..][0..16].* = @as(V16, d[off..][0..16].*) ^ @as(V16, s[off..][0..16].*);
        }
    }
}

pub fn xorWithin(shards: *Shards, x: usize, y: usize, count: usize) void {
    const xs = x * shards.shard_len;
    const ys = y * shards.shard_len;
    const len = count * shards.shard_len;
    for (0..len) |i| {
        inline for (0..4) |q| {
            const off = q * 16;
            shards.data[xs + i][off..][0..16].* = @as(V16, shards.data[xs + i][off..][0..16].*) ^ @as(V16, shards.data[ys + i][off..][0..16].*);
        }
    }
}

// ── SIMD operations (with preloaded tables) ────────────────────────────

/// Inline helper: compute product = data * lut (16 GF elements = 32 bytes lo+hi)
inline fn simdMul16(dl: V16, dh: V16, lo0: V16, lo1: V16, lo2: V16, lo3: V16, hi0: V16, hi1: V16, hi2: V16, hi3: V16) struct { V16, V16 } {
    return .{
        tblLookup(lo0, dl & mask_0f) ^ tblLookup(lo1, (dl >> shift_4) & mask_0f) ^ tblLookup(lo2, dh & mask_0f) ^ tblLookup(lo3, (dh >> shift_4) & mask_0f),
        tblLookup(hi0, dl & mask_0f) ^ tblLookup(hi1, (dl >> shift_4) & mask_0f) ^ tblLookup(hi2, dh & mask_0f) ^ tblLookup(hi3, (dh >> shift_4) & mask_0f),
    };
}

inline fn mulAdd(x: []Chunk, y: []const Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0]; const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2]; const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0]; const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2]; const hi3: V16 = lut.hi[3];

    for (x, y) |*xc, yc| {
        const pl, const ph = simdMul16(yc[0..16].*, yc[32..48].*, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        xc[0..16].* = @as(V16, xc[0..16].*) ^ pl;
        xc[32..48].* = @as(V16, xc[32..48].*) ^ ph;
        const pl2, const ph2 = simdMul16(yc[16..32].*, yc[48..64].*, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        xc[16..32].* = @as(V16, xc[16..32].*) ^ pl2;
        xc[48..64].* = @as(V16, xc[48..64].*) ^ ph2;
    }
}

inline fn mulInPlace(x: []Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0]; const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2]; const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0]; const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2]; const hi3: V16 = lut.hi[3];

    for (x) |*c| {
        c[0..16].*, c[32..48].* = simdMul16(c[0..16].*, c[32..48].*, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        c[16..32].*, c[48..64].* = simdMul16(c[16..32].*, c[48..64].*, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
    }
}

/// Fused FFT butterfly: a ^= b * lut; b ^= a
/// Processes 2 chunks at once when possible for better pipeline utilization.
inline fn fftButterfly(a: []Chunk, b: []Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0]; const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2]; const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0]; const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2]; const hi3: V16 = lut.hi[3];

    // Process pairs of chunks to utilize more NEON registers (32 available)
    var ci: usize = 0;
    while (ci + 2 <= a.len) : (ci += 2) {
        // ── Chunk 0 ──
        const bl0: V16 = b[ci][0..16].*;
        const bh0: V16 = b[ci][32..48].*;
        const pl0, const ph0 = simdMul16(bl0, bh0, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const al0: V16 = @as(V16, a[ci][0..16].*) ^ pl0;
        const ah0: V16 = @as(V16, a[ci][32..48].*) ^ ph0;

        const bl0b: V16 = b[ci][16..32].*;
        const bh0b: V16 = b[ci][48..64].*;
        const pl0b, const ph0b = simdMul16(bl0b, bh0b, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const al0b: V16 = @as(V16, a[ci][16..32].*) ^ pl0b;
        const ah0b: V16 = @as(V16, a[ci][48..64].*) ^ ph0b;

        // ── Chunk 1 (interleaved for pipeline) ──
        const bl1: V16 = b[ci + 1][0..16].*;
        const bh1: V16 = b[ci + 1][32..48].*;
        const pl1, const ph1 = simdMul16(bl1, bh1, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const al1: V16 = @as(V16, a[ci + 1][0..16].*) ^ pl1;
        const ah1: V16 = @as(V16, a[ci + 1][32..48].*) ^ ph1;

        const bl1b: V16 = b[ci + 1][16..32].*;
        const bh1b: V16 = b[ci + 1][48..64].*;
        const pl1b, const ph1b = simdMul16(bl1b, bh1b, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const al1b: V16 = @as(V16, a[ci + 1][16..32].*) ^ pl1b;
        const ah1b: V16 = @as(V16, a[ci + 1][48..64].*) ^ ph1b;

        // ── Store all results ──
        a[ci][0..16].* = al0; a[ci][32..48].* = ah0;
        a[ci][16..32].* = al0b; a[ci][48..64].* = ah0b;
        b[ci][0..16].* = bl0 ^ al0; b[ci][32..48].* = bh0 ^ ah0;
        b[ci][16..32].* = bl0b ^ al0b; b[ci][48..64].* = bh0b ^ ah0b;

        a[ci + 1][0..16].* = al1; a[ci + 1][32..48].* = ah1;
        a[ci + 1][16..32].* = al1b; a[ci + 1][48..64].* = ah1b;
        b[ci + 1][0..16].* = bl1 ^ al1; b[ci + 1][32..48].* = bh1 ^ ah1;
        b[ci + 1][16..32].* = bl1b ^ al1b; b[ci + 1][48..64].* = bh1b ^ ah1b;
    }

    // Handle odd trailing chunk
    if (ci < a.len) {
        const bl: V16 = b[ci][0..16].*; const bh: V16 = b[ci][32..48].*;
        const pl, const ph = simdMul16(bl, bh, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const al: V16 = @as(V16, a[ci][0..16].*) ^ pl;
        const ah: V16 = @as(V16, a[ci][32..48].*) ^ ph;
        a[ci][0..16].* = al; a[ci][32..48].* = ah;
        b[ci][0..16].* = bl ^ al; b[ci][32..48].* = bh ^ ah;

        const bl2: V16 = b[ci][16..32].*; const bh2: V16 = b[ci][48..64].*;
        const pl2, const ph2 = simdMul16(bl2, bh2, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const al2: V16 = @as(V16, a[ci][16..32].*) ^ pl2;
        const ah2: V16 = @as(V16, a[ci][48..64].*) ^ ph2;
        a[ci][16..32].* = al2; a[ci][48..64].* = ah2;
        b[ci][16..32].* = bl2 ^ al2; b[ci][48..64].* = bh2 ^ ah2;
    }
}

/// Fused IFFT butterfly: b ^= a; a ^= b * lut
inline fn ifftButterfly(a: []Chunk, b: []Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0]; const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2]; const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0]; const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2]; const hi3: V16 = lut.hi[3];

    var ci: usize = 0;
    while (ci + 2 <= a.len) : (ci += 2) {
        // ── Chunk 0 ──
        const al0: V16 = a[ci][0..16].*; const ah0: V16 = a[ci][32..48].*;
        const nb0: V16 = @as(V16, b[ci][0..16].*) ^ al0;
        const nbh0: V16 = @as(V16, b[ci][32..48].*) ^ ah0;
        const al0b: V16 = a[ci][16..32].*; const ah0b: V16 = a[ci][48..64].*;
        const nb0b: V16 = @as(V16, b[ci][16..32].*) ^ al0b;
        const nbh0b: V16 = @as(V16, b[ci][48..64].*) ^ ah0b;

        // ── Chunk 1 ──
        const al1: V16 = a[ci + 1][0..16].*; const ah1: V16 = a[ci + 1][32..48].*;
        const nb1: V16 = @as(V16, b[ci + 1][0..16].*) ^ al1;
        const nbh1: V16 = @as(V16, b[ci + 1][32..48].*) ^ ah1;
        const al1b: V16 = a[ci + 1][16..32].*; const ah1b: V16 = a[ci + 1][48..64].*;
        const nb1b: V16 = @as(V16, b[ci + 1][16..32].*) ^ al1b;
        const nbh1b: V16 = @as(V16, b[ci + 1][48..64].*) ^ ah1b;

        // ── Multiply and store ──
        const p0l, const p0h = simdMul16(nb0, nbh0, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const p0lb, const p0hb = simdMul16(nb0b, nbh0b, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const p1l, const p1h = simdMul16(nb1, nbh1, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const p1lb, const p1hb = simdMul16(nb1b, nbh1b, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);

        b[ci][0..16].* = nb0; b[ci][32..48].* = nbh0;
        b[ci][16..32].* = nb0b; b[ci][48..64].* = nbh0b;
        a[ci][0..16].* = al0 ^ p0l; a[ci][32..48].* = ah0 ^ p0h;
        a[ci][16..32].* = al0b ^ p0lb; a[ci][48..64].* = ah0b ^ p0hb;

        b[ci + 1][0..16].* = nb1; b[ci + 1][32..48].* = nbh1;
        b[ci + 1][16..32].* = nb1b; b[ci + 1][48..64].* = nbh1b;
        a[ci + 1][0..16].* = al1 ^ p1l; a[ci + 1][32..48].* = ah1 ^ p1h;
        a[ci + 1][16..32].* = al1b ^ p1lb; a[ci + 1][48..64].* = ah1b ^ p1hb;
    }

    if (ci < a.len) {
        const al: V16 = a[ci][0..16].*; const ah: V16 = a[ci][32..48].*;
        const nb: V16 = @as(V16, b[ci][0..16].*) ^ al;
        const nbh: V16 = @as(V16, b[ci][32..48].*) ^ ah;
        b[ci][0..16].* = nb; b[ci][32..48].* = nbh;
        const pl, const ph = simdMul16(nb, nbh, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        a[ci][0..16].* = al ^ pl; a[ci][32..48].* = ah ^ ph;

        const al2: V16 = a[ci][16..32].*; const ah2: V16 = a[ci][48..64].*;
        const nb2: V16 = @as(V16, b[ci][16..32].*) ^ al2;
        const nbh2: V16 = @as(V16, b[ci][48..64].*) ^ ah2;
        b[ci][16..32].* = nb2; b[ci][48..64].* = nbh2;
        const pl2, const ph2 = simdMul16(nb2, nbh2, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        a[ci][16..32].* = al2 ^ pl2; a[ci][48..64].* = ah2 ^ ph2;
    }
}

// ── Engine ─────────────────────────────────────────────────────────────

pub const Engine = struct {
    el: tables.ExpLog,
    skew: tables.Skew,
    mul128: *[GF_ORDER]gf.Mul128,
    allocator: Allocator,

    pub fn init(allocator: Allocator) !Engine {
        const el = tables.initExpLog();
        const skew = tables.initSkew(&el);
        const mul128 = try allocator.create([GF_ORDER]gf.Mul128);
        for (0..GF_ORDER) |log_m| {
            mul128[log_m] = tables.buildMul128Entry(@intCast(log_m), &el);
        }
        return .{ .el = el, .skew = skew, .mul128 = mul128, .allocator = allocator };
    }

    pub fn deinit(self: *Engine) void { self.allocator.destroy(self.mul128); }

    pub fn fft(self: *const Engine, shards: *Shards, pos: usize, size: usize, truncated_size: usize, skew_delta: usize) void {
        const sl = shards.shard_len;
        const data = shards.data;
        var dist = size / 2;
        while (dist > 0) : (dist /= 2) {
            var r: usize = 0;
            while (r < truncated_size) : (r += dist * 2) {
                const log_m = self.skew[r + dist + skew_delta - 1];
                const base_a = (pos + r) * sl;
                const base_b = (pos + r + dist) * sl;
                if (log_m != GF_MODULUS) {
                    const lut = &self.mul128[log_m];
                    for (0..dist) |i| {
                        fftButterfly(data[base_a + i * sl ..][0..sl], data[base_b + i * sl ..][0..sl], lut);
                    }
                } else {
                    for (0..dist) |i| {
                        xorChunks(data[base_b + i * sl ..][0..sl], data[base_a + i * sl ..][0..sl]);
                    }
                }
            }
        }
    }

    pub fn ifft(self: *const Engine, shards: *Shards, pos: usize, size: usize, truncated_size: usize, skew_delta: usize) void {
        const sl = shards.shard_len;
        const data = shards.data;
        var dist: usize = 1;
        while (dist < size) : (dist *= 2) {
            var r: usize = 0;
            while (r < truncated_size) : (r += dist * 2) {
                const log_m = self.skew[r + dist + skew_delta - 1];
                const base_a = (pos + r) * sl;
                const base_b = (pos + r + dist) * sl;
                if (log_m != GF_MODULUS) {
                    const lut = &self.mul128[log_m];
                    for (0..dist) |i| {
                        ifftButterfly(data[base_a + i * sl ..][0..sl], data[base_b + i * sl ..][0..sl], lut);
                    }
                } else {
                    for (0..dist) |i| {
                        xorChunks(data[base_b + i * sl ..][0..sl], data[base_a + i * sl ..][0..sl]);
                    }
                }
            }
        }
    }

    pub fn mul(self: *const Engine, chunks: []Chunk, log_m: GfElement) void {
        mulInPlace(chunks, &self.mul128[log_m]);
    }

    pub fn fftSkewEnd(self: *const Engine, shards: *Shards, pos: usize, size: usize, truncated_size: usize) void {
        self.fft(shards, pos, size, truncated_size, pos + size);
    }

    pub fn ifftSkewEnd(self: *const Engine, shards: *Shards, pos: usize, size: usize, truncated_size: usize) void {
        self.ifft(shards, pos, size, truncated_size, pos + size);
    }

    pub fn evalPoly(self: *const Engine, erasures: *[GF_ORDER]GfElement, truncated_size: usize) void {
        const log_walsh = tables.initLogWalsh(&self.el);
        fwht_mod.fwht(erasures, truncated_size);
        for (0..GF_ORDER) |i| {
            const product: u32 = @as(u32, erasures[i]) * @as(u32, log_walsh[i]);
            erasures[i] = gf.addMod(@truncate(product), @truncate(product >> GF_BITS));
        }
        fwht_mod.fwht(erasures, GF_ORDER);
    }
};

/// Formal derivative.
pub fn formalDerivative(shards: *Shards, len: usize) void {
    for (1..len) |i| {
        const tz: std.math.Log2Int(usize) = @intCast(@ctz(i));
        const width: usize = @as(usize, 1) << tz;
        if (i >= width) xorWithin(shards, i - width, i, width);
    }
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "Engine init" {
    var eng = try Engine.init(testing.allocator);
    defer eng.deinit();
    try testing.expectEqual(GF_MODULUS, eng.skew[0]);
}

test "FFT then IFFT round-trip" {
    var eng = try Engine.init(testing.allocator);
    defer eng.deinit();
    const alloc = testing.allocator;
    const count = 4;
    var shards = try Shards.init(alloc, count, 1);
    defer shards.deinit();
    for (0..count) |i| {
        for (shards.shardMut(i)) |*chunk| for (chunk) |*b| { b.* = @truncate(i * 17 + 3); };
    }
    const original = try alloc.alloc(Chunk, count);
    defer alloc.free(original);
    @memcpy(original, shards.data[0..count]);

    eng.fftSkewEnd(&shards, 0, count, count);
    eng.ifftSkewEnd(&shards, 0, count, count);

    for (0..count) |i| try testing.expectEqual(original[i], shards.data[i]);
}

test "XOR chunks" {
    var a = [_]Chunk{.{0xff} ** 64};
    const b = [_]Chunk{.{0xaa} ** 64};
    xorChunks(&a, &b);
    try testing.expectEqual(@as(u8, 0x55), a[0][0]);
}
