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

pub fn xorChunks(dst: []Chunk, src: []const Chunk) void {
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

// ── SIMD mul_add / mul (with preloaded tables) ─────────────────────────

fn mulAdd(x: []Chunk, y: []const Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0]; const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2]; const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0]; const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2]; const hi3: V16 = lut.hi[3];

    for (x, y) |*xc, yc| {
        const yl: V16 = yc[0..16].*; const yh: V16 = yc[32..48].*;
        xc[0..16].* = @as(V16, xc[0..16].*) ^ tblLookup(lo0, yl & mask_0f) ^ tblLookup(lo1, (yl >> shift_4) & mask_0f) ^ tblLookup(lo2, yh & mask_0f) ^ tblLookup(lo3, (yh >> shift_4) & mask_0f);
        xc[32..48].* = @as(V16, xc[32..48].*) ^ tblLookup(hi0, yl & mask_0f) ^ tblLookup(hi1, (yl >> shift_4) & mask_0f) ^ tblLookup(hi2, yh & mask_0f) ^ tblLookup(hi3, (yh >> shift_4) & mask_0f);

        const yl2: V16 = yc[16..32].*; const yh2: V16 = yc[48..64].*;
        xc[16..32].* = @as(V16, xc[16..32].*) ^ tblLookup(lo0, yl2 & mask_0f) ^ tblLookup(lo1, (yl2 >> shift_4) & mask_0f) ^ tblLookup(lo2, yh2 & mask_0f) ^ tblLookup(lo3, (yh2 >> shift_4) & mask_0f);
        xc[48..64].* = @as(V16, xc[48..64].*) ^ tblLookup(hi0, yl2 & mask_0f) ^ tblLookup(hi1, (yl2 >> shift_4) & mask_0f) ^ tblLookup(hi2, yh2 & mask_0f) ^ tblLookup(hi3, (yh2 >> shift_4) & mask_0f);
    }
}

fn mulInPlace(x: []Chunk, lut: *const gf.Mul128) void {
    const lo0: V16 = lut.lo[0]; const lo1: V16 = lut.lo[1];
    const lo2: V16 = lut.lo[2]; const lo3: V16 = lut.lo[3];
    const hi0: V16 = lut.hi[0]; const hi1: V16 = lut.hi[1];
    const hi2: V16 = lut.hi[2]; const hi3: V16 = lut.hi[3];

    for (x) |*c| {
        const dl: V16 = c[0..16].*; const dh: V16 = c[32..48].*;
        c[0..16].* = tblLookup(lo0, dl & mask_0f) ^ tblLookup(lo1, (dl >> shift_4) & mask_0f) ^ tblLookup(lo2, dh & mask_0f) ^ tblLookup(lo3, (dh >> shift_4) & mask_0f);
        c[32..48].* = tblLookup(hi0, dl & mask_0f) ^ tblLookup(hi1, (dl >> shift_4) & mask_0f) ^ tblLookup(hi2, dh & mask_0f) ^ tblLookup(hi3, (dh >> shift_4) & mask_0f);

        const dl2: V16 = c[16..32].*; const dh2: V16 = c[48..64].*;
        c[16..32].* = tblLookup(lo0, dl2 & mask_0f) ^ tblLookup(lo1, (dl2 >> shift_4) & mask_0f) ^ tblLookup(lo2, dh2 & mask_0f) ^ tblLookup(lo3, (dh2 >> shift_4) & mask_0f);
        c[48..64].* = tblLookup(hi0, dl2 & mask_0f) ^ tblLookup(hi1, (dl2 >> shift_4) & mask_0f) ^ tblLookup(hi2, dh2 & mask_0f) ^ tblLookup(hi3, (dh2 >> shift_4) & mask_0f);
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
        var dist = size / 2;
        while (dist > 0) : (dist /= 2) {
            var r: usize = 0;
            while (r < truncated_size) : (r += dist * 2) {
                const log_m = self.skew[r + dist + skew_delta - 1];
                for (r..r + dist) |i| {
                    const a = shards.shardMut(pos + i);
                    const b = shards.shard(pos + i + dist);
                    if (log_m != GF_MODULUS) mulAdd(a, b, &self.mul128[log_m]);
                    xorChunks(shards.shardMut(pos + i + dist), a);
                }
            }
        }
    }

    pub fn ifft(self: *const Engine, shards: *Shards, pos: usize, size: usize, truncated_size: usize, skew_delta: usize) void {
        var dist: usize = 1;
        while (dist < size) : (dist *= 2) {
            var r: usize = 0;
            while (r < truncated_size) : (r += dist * 2) {
                const log_m = self.skew[r + dist + skew_delta - 1];
                for (r..r + dist) |i| {
                    const a = shards.shard(pos + i);
                    xorChunks(shards.shardMut(pos + i + dist), a);
                    if (log_m != GF_MODULUS) {
                        const b = shards.shard(pos + i + dist);
                        mulAdd(shards.shardMut(pos + i), b, &self.mul128[log_m]);
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
