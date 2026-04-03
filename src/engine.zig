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
const tblLookup = gf.tblLookup;
const use_x86_avx2 = (builtin.cpu.arch == .x86_64 or builtin.cpu.arch == .x86);

const mask_0f: V16 = @splat(0x0f);
const shift_4: @Vector(16, u3) = @splat(4);

// x86 AVX2 C FFI (from simd_x86.c, compiled with -mavx2)
extern "c" fn gf_fft_butterfly_avx2(a: [*]u8, b: [*]const u8, lut: *const gf.Mul128) void;
extern "c" fn gf_ifft_butterfly_avx2(a: [*]u8, b: [*]u8, lut: *const gf.Mul128) void;
extern "c" fn gf_mul_avx2(x: [*]u8, lut: *const gf.Mul128) void;

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

    pub fn deinit(self: *Shards) void {
        self.allocator.free(self.data);
    }

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
            dst[whole] = .{0} ** 64;
            const half = tail / 2;
            @memcpy(dst[whole][0..half], shard_data[whole * 64 ..][0..half]);
            @memcpy(dst[whole][32 .. 32 + half], shard_data[whole * 64 + half ..][0 .. tail - half]);
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

const V64 = @Vector(64, u8);

pub inline fn xorChunks(dst: []Chunk, src: []const Chunk) void {
    for (dst, src) |*d, s| {
        const dv: V64 = @bitCast(d.*);
        const sv: V64 = @bitCast(s);
        d.* = @bitCast(dv ^ sv);
    }
}

pub inline fn xorWithin(shards: *Shards, x: usize, y: usize, count: usize) void {
    const xs = x * shards.shard_len;
    const ys = y * shards.shard_len;
    const len = count * shards.shard_len;
    for (0..len) |i| {
        const dv: V64 = shards.data[xs + i];
        const sv: V64 = shards.data[ys + i];
        shards.data[xs + i] = @bitCast(dv ^ sv);
    }
}

// ── SIMD operations (with preloaded tables) ────────────────────────────

/// Inline helper: compute product = data * lut (16 GF elements = 32 bytes lo+hi)
/// Uses nibble decomposition with 8 table lookups.
inline fn simdMul16(dl: V16, dh: V16, lo0: V16, lo1: V16, lo2: V16, lo3: V16, hi0: V16, hi1: V16, hi2: V16, hi3: V16) struct { V16, V16 } {
    const dl_lo = dl & mask_0f;
    const dl_hi = (dl >> shift_4) & mask_0f;
    const dh_lo = dh & mask_0f;
    const dh_hi = (dh >> shift_4) & mask_0f;
    return .{
        tblLookup(lo0, dl_lo) ^ tblLookup(lo1, dl_hi) ^ tblLookup(lo2, dh_lo) ^ tblLookup(lo3, dh_hi),
        tblLookup(hi0, dl_lo) ^ tblLookup(hi1, dl_hi) ^ tblLookup(hi2, dh_lo) ^ tblLookup(hi3, dh_hi),
    };
}

/// Load a 64-byte chunk as 4 × V16 (enables ldp optimization).
inline fn loadChunk(c: *const Chunk) struct { V16, V16, V16, V16 } {
    return .{ c[0..16].*, c[16..32].*, c[32..48].*, c[48..64].* };
}

/// Store 4 × V16 back to a 64-byte chunk (enables stp optimization).
inline fn storeChunk(c: *Chunk, v0: V16, v1: V16, v2: V16, v3: V16) void {
    c[0..16].* = v0;
    c[16..32].* = v1;
    c[32..48].* = v2;
    c[48..64].* = v3;
}

inline fn mulAdd(x: []Chunk, y: []const Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0];
    const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2];
    const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0];
    const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2];
    const hi3: V16 = lut.hi[3];

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
    if (comptime use_x86_avx2) {
        for (x) |*c| gf_mul_avx2(@ptrCast(c), lut);
        return;
    }
    const lo0: V16 = lut.lo[0];
    const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2];
    const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0];
    const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2];
    const hi3: V16 = lut.hi[3];

    for (x) |*c| {
        c[0..16].*, c[32..48].* = simdMul16(c[0..16].*, c[32..48].*, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        c[16..32].*, c[48..64].* = simdMul16(c[16..32].*, c[48..64].*, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
    }
}

/// Fused FFT butterfly: a ^= b * lut; b ^= a
/// On x86: dispatches to AVX2 C implementation (vpshufb, 256-bit)
/// On ARM: uses Zig NEON tbl (128-bit)
inline fn fftButterfly(a: []Chunk, b: []Chunk, lut: *const gf.Mul128) void {
    if (comptime use_x86_avx2) {
        for (a, b) |*ac, *bc| {
            gf_fft_butterfly_avx2(@ptrCast(ac), @ptrCast(bc), lut);
        }
        return;
    }
    const lo0: V16 = lut.lo[0];
    const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2];
    const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0];
    const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2];
    const hi3: V16 = lut.hi[3];

    for (a, b) |*ac, *bc| {
        // Load both chunks (4 × V16 each → ldp candidates)
        const b0, const b1, const b2, const b3 = loadChunk(bc);
        const a0, const a1, const a2, const a3 = loadChunk(ac);

        // Multiply b (lo=b0,b2 hi=b1,b3 in split layout: [0..16]=lo_lo, [16..32]=lo_hi, [32..48]=hi_lo, [48..64]=hi_hi)
        const pl0, const ph0 = simdMul16(b0, b2, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const pl1, const ph1 = simdMul16(b1, b3, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);

        // a ^= product
        const na0 = a0 ^ pl0;
        const na1 = a1 ^ pl1;
        const na2 = a2 ^ ph0;
        const na3 = a3 ^ ph1;

        // Store a, then b ^= a
        storeChunk(ac, na0, na1, na2, na3);
        storeChunk(bc, b0 ^ na0, b1 ^ na1, b2 ^ na2, b3 ^ na3);
    }
}

/// Fused IFFT butterfly: b ^= a; a ^= b * lut
inline fn ifftButterfly(a: []Chunk, b: []Chunk, lut: *const gf.Mul128) void {
    if (comptime use_x86_avx2) {
        for (a, b) |*ac, *bc| {
            gf_ifft_butterfly_avx2(@ptrCast(ac), @ptrCast(bc), lut);
        }
        return;
    }
    const lo0: V16 = lut.lo[0];
    const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2];
    const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0];
    const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2];
    const hi3: V16 = lut.hi[3];

    for (a, b) |*ac, *bc| {
        const a0, const a1, const a2, const a3 = loadChunk(ac);
        const b0, const b1, const b2, const b3 = loadChunk(bc);

        // b ^= a
        const nb0 = b0 ^ a0;
        const nb1 = b1 ^ a1;
        const nb2 = b2 ^ a2;
        const nb3 = b3 ^ a3;

        // a ^= new_b * lut
        const pl0, const ph0 = simdMul16(nb0, nb2, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);
        const pl1, const ph1 = simdMul16(nb1, nb3, lo0, lo1, lo2, lo3, hi0, hi1, hi2, hi3);

        storeChunk(bc, nb0, nb1, nb2, nb3);
        storeChunk(ac, a0 ^ pl0, a1 ^ pl1, a2 ^ ph0, a3 ^ ph1);
    }
}

// ── Engine ─────────────────────────────────────────────────────────────

const SharedTables = struct {
    skew: tables.Skew,
    log_walsh: tables.LogWalsh,
    mul128: [GF_ORDER]gf.Mul128,
};

var shared_tables: SharedTables = undefined;
var shared_tables_initialized: bool = false;

fn initSharedTables() void {
    const el = tables.initExpLog();
    shared_tables.skew = tables.initSkew(&el);
    shared_tables.log_walsh = tables.initLogWalsh(&el);
    for (0..GF_ORDER) |log_m| {
        shared_tables.mul128[log_m] = tables.buildMul128Entry(@intCast(log_m), &el);
    }
}

fn getSharedTables() *const SharedTables {
    if (!shared_tables_initialized) {
        initSharedTables();
        shared_tables_initialized = true;
    }
    return &shared_tables;
}

pub const Engine = struct {
    skew: *const tables.Skew,
    log_walsh: *const tables.LogWalsh,
    mul128: *const [GF_ORDER]gf.Mul128,

    pub fn init(allocator: Allocator) !Engine {
        _ = allocator;
        const shared = getSharedTables();
        return .{
            .skew = &shared.skew,
            .log_walsh = &shared.log_walsh,
            .mul128 = &shared.mul128,
        };
    }

    pub fn deinit(self: *Engine) void {
        _ = self;
    }

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
        fwht_mod.fwht(erasures, truncated_size);
        for (0..GF_ORDER) |i| {
            const product: u32 = @as(u32, erasures[i]) * @as(u32, self.log_walsh[i]);
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
        for (shards.shardMut(i)) |*chunk| for (chunk) |*b| {
            b.* = @truncate(i * 17 + 3);
        };
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
