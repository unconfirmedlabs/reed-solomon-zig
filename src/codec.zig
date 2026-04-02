//! Reed-Solomon encoder/decoder using Leopard-RS over GF(2^16).
//!
//! Public API matching the reed_solomon_simd Rust crate.

const std = @import("std");
const engine_mod = @import("engine.zig");
const gf = @import("gf.zig");
const GfElement = gf.GfElement;
const GF_ORDER = gf.GF_ORDER;
const GF_MODULUS = gf.GF_MODULUS;
const Allocator = std.mem.Allocator;
const Shards = engine_mod.Shards;
const Engine = engine_mod.Engine;

pub const Error = error{
    InvalidArgs,
    TooFewShards,
    ShardSizeMismatch,
    ShardSizeNotEven,
    TooManyOriginal,
    TooManyRecovery,
    EncodeFailed,
    DecodeFailed,
    OutOfMemory,
};

/// Find the next power of 2 >= n.
fn nextPow2(n: usize) usize {
    if (n == 0) return 1;
    var v = n - 1;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    return v + 1;
}

/// Round up to next multiple of `chunk_size`.
fn nextPow2Multiple(n: usize, chunk_size: usize) usize {
    return ((n + chunk_size - 1) / chunk_size) * chunk_size;
}

// ── Encoder ────────────────────────────────────────────────────────────

pub const Encoder = struct {
    engine: Engine,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    shard_len_64: usize, // shard size in 64-byte chunks (rounded up)
    shards: Shards,
    added: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, original_count: usize, recovery_count: usize, shard_bytes: usize) Error!Encoder {
        if (original_count == 0 or recovery_count == 0 or shard_bytes == 0)
            return Error.InvalidArgs;
        if (shard_bytes % 2 != 0)
            return Error.ShardSizeNotEven;
        if (original_count + recovery_count > GF_ORDER)
            return Error.TooManyOriginal;

        const shard_len_64 = (shard_bytes + 63) / 64; // round up to 64-byte chunks
        const total = nextPow2(original_count + recovery_count);

        var shards = Shards.init(allocator, total, shard_len_64) catch return Error.OutOfMemory;
        errdefer shards.deinit();
        var eng = Engine.init(allocator) catch return Error.OutOfMemory;
        errdefer eng.deinit();

        return .{
            .engine = eng,
            .original_count = original_count,
            .recovery_count = recovery_count,
            .shard_bytes = shard_bytes,
            .shard_len_64 = shard_len_64,
            .shards = shards,
            .added = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Encoder) void {
        self.shards.deinit();
        self.engine.deinit();
    }

    /// Add an original shard (must be called exactly `original_count` times in order).
    pub fn addOriginal(self: *Encoder, data: []const u8) Error!void {
        if (self.added >= self.original_count) return Error.TooManyOriginal;
        if (data.len != self.shard_bytes) return Error.ShardSizeMismatch;
        self.shards.insert(self.added, data);
        self.added += 1;
    }

    /// Encode recovery shards. Returns recovery data in the output buffer.
    /// `out` must be at least `recovery_count * shard_bytes` bytes.
    pub fn encode(self: *Encoder, out: []u8) Error!void {
        if (self.added != self.original_count) return Error.TooFewShards;
        if (out.len < self.recovery_count * self.shard_bytes) return Error.InvalidArgs;

        const k = self.original_count;

        // Determine if high-rate or low-rate
        if (k >= self.recovery_count) {
            self.encodeHighRate(k);
        } else {
            self.encodeLowRate(k);
        }

        // Extract recovery shards to output buffer
        // High rate: recovery at [0..recovery_count]
        // Low rate: recovery at [0..recovery_count]
        for (0..self.recovery_count) |i| {
            const shard_data = self.shards.shard(i);
            self.extractShard(shard_data, out[i * self.shard_bytes ..]);
        }

        // Reset for reuse
        self.added = 0;
    }

    fn encodeHighRate(self: *Encoder, k: usize) void {
        const chunk_size = nextPow2(self.recovery_count);
        const work_count = nextPow2Multiple(k, chunk_size);

        // Zero padding beyond original data
        self.shards.zero_range(k, work_count);

        // FIRST CHUNK
        const first_count = @min(k, chunk_size);
        self.shards.zero_range(first_count, chunk_size);
        self.engine.ifftSkewEnd(&self.shards, 0, chunk_size, first_count);

        if (k > chunk_size) {
            // FULL CHUNKS
            var chunk_start = chunk_size;
            while (chunk_start + chunk_size <= k) : (chunk_start += chunk_size) {
                self.engine.ifftSkewEnd(&self.shards, chunk_start, chunk_size, chunk_size);
                engine_mod.xorWithin(&self.shards, 0, chunk_start, chunk_size);
            }

            // FINAL PARTIAL CHUNK
            const last_count = k % chunk_size;
            if (last_count > 0) {
                self.shards.zero_range(chunk_start + last_count, work_count);
                self.engine.ifftSkewEnd(&self.shards, chunk_start, chunk_size, last_count);
                engine_mod.xorWithin(&self.shards, 0, chunk_start, chunk_size);
            }
        }

        // FFT to produce recovery shards at [0..recovery_count]
        self.engine.fft(&self.shards, 0, chunk_size, self.recovery_count, 0);
    }

    fn encodeLowRate(self: *Encoder, k: usize) void {
        const chunk_size = nextPow2(k);
        const work_count = nextPow2Multiple(self.recovery_count, chunk_size);

        // Zero-pad original
        self.shards.zero_range(k, chunk_size);

        // IFFT original
        self.engine.ifft(&self.shards, 0, chunk_size, k, 0);

        // Copy IFFT result to each recovery chunk
        var chunk_start = chunk_size;
        while (chunk_start < self.recovery_count) : (chunk_start += chunk_size) {
            self.shards.copy_within(0, chunk_start, chunk_size);
        }

        // FFT each chunk with skew_delta = chunk_start + chunk_size
        chunk_start = 0;
        while (chunk_start + chunk_size <= self.recovery_count) : (chunk_start += chunk_size) {
            self.engine.fftSkewEnd(&self.shards, chunk_start, chunk_size, chunk_size);
        }

        // Final partial chunk
        const last_count = self.recovery_count % chunk_size;
        if (last_count > 0) {
            self.engine.fftSkewEnd(&self.shards, chunk_start, chunk_size, last_count);
        }
        _ = work_count;
    }

    fn extractShard(self: *const Encoder, chunks: []const engine_mod.Chunk, out: []u8) void {
        const whole = self.shard_bytes / 64;
        const tail = self.shard_bytes % 64;

        for (0..whole) |i| {
            @memcpy(out[i * 64 ..][0..64], &chunks[i]);
        }

        if (tail > 0) {
            const half = tail / 2;
            @memcpy(out[whole * 64 ..][0..half], chunks[whole][0..half]);
            @memcpy(out[whole * 64 + half ..][0..half], chunks[whole][32 .. 32 + half]);
        }
    }
};

// ── Decoder ────────────────────────────────────────────────────────────

pub const Decoder = struct {
    engine: Engine,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    shard_len_64: usize,
    high_rate: bool,
    // Memory layout depends on rate:
    //   High rate: work[0..chunk_size] = recovery, work[chunk_size..] = original
    //   Low rate:  work[0..chunk_size] = original, work[chunk_size..] = recovery
    original_base: usize,
    recovery_base: usize,
    work_count: usize,
    shards: Shards,
    received: []bool, // indexed by work position
    original_recv_count: usize,
    recovery_recv_count: usize,
    allocator: Allocator,

    pub fn init(allocator: Allocator, original_count: usize, recovery_count: usize, shard_bytes: usize) Error!Decoder {
        if (original_count == 0 or recovery_count == 0 or shard_bytes == 0)
            return Error.InvalidArgs;
        if (shard_bytes % 2 != 0)
            return Error.ShardSizeNotEven;
        if (original_count >= GF_ORDER or recovery_count >= GF_ORDER)
            return Error.TooManyOriginal;

        const shard_len_64 = (shard_bytes + 63) / 64;

        // Determine rate and layout (matching Rust's use_high_rate logic)
        const high_rate = useHighRate(original_count, recovery_count);

        var original_base: usize = undefined;
        var recovery_base: usize = undefined;
        var work_count: usize = undefined;

        if (high_rate) {
            const chunk_size = nextPow2(recovery_count);
            original_base = chunk_size;
            recovery_base = 0;
            work_count = nextPow2(chunk_size + original_count);
        } else {
            const chunk_size = nextPow2(original_count);
            original_base = 0;
            recovery_base = chunk_size;
            work_count = nextPow2(chunk_size + recovery_count);
        }

        var shards = Shards.init(allocator, work_count, shard_len_64) catch return Error.OutOfMemory;
        errdefer shards.deinit();
        const received = allocator.alloc(bool, work_count) catch return Error.OutOfMemory;
        errdefer allocator.free(received);
        @memset(received, false);
        var eng = Engine.init(allocator) catch return Error.OutOfMemory;
        errdefer eng.deinit();

        return .{
            .engine = eng,
            .original_count = original_count,
            .recovery_count = recovery_count,
            .shard_bytes = shard_bytes,
            .shard_len_64 = shard_len_64,
            .high_rate = high_rate,
            .original_base = original_base,
            .recovery_base = recovery_base,
            .work_count = work_count,
            .shards = shards,
            .received = received,
            .original_recv_count = 0,
            .recovery_recv_count = 0,
            .allocator = allocator,
        };
    }

    pub fn deinit(self: *Decoder) void {
        self.shards.deinit();
        self.engine.deinit();
        self.allocator.free(self.received);
    }

    pub fn addOriginal(self: *Decoder, index: usize, data: []const u8) Error!void {
        if (index >= self.original_count) return Error.InvalidArgs;
        if (data.len != self.shard_bytes) return Error.ShardSizeMismatch;
        const pos = self.original_base + index;
        self.shards.insert(pos, data);
        self.received[pos] = true;
        self.original_recv_count += 1;
    }

    pub fn addRecovery(self: *Decoder, index: usize, data: []const u8) Error!void {
        if (index >= self.recovery_count) return Error.InvalidArgs;
        if (data.len != self.shard_bytes) return Error.ShardSizeMismatch;
        const pos = self.recovery_base + index;
        self.shards.insert(pos, data);
        self.received[pos] = true;
        self.recovery_recv_count += 1;
    }

    /// Decode missing original shards.
    /// Returns the number of restored shards.
    pub fn decode(self: *Decoder, restored_out: []u8, restored_indices: []usize) Error!usize {
        if (self.original_recv_count + self.recovery_recv_count < self.original_count)
            return Error.TooFewShards;

        // Count missing originals
        var missing: usize = 0;
        for (0..self.original_count) |i| {
            if (!self.received[self.original_base + i]) missing += 1;
        }
        if (missing == 0) return 0;

        if (self.high_rate) {
            self.decodeHighRate();
        } else {
            self.decodeLowRate();
        }

        // Extract restored shards
        var count: usize = 0;
        for (0..self.original_count) |i| {
            if (!self.received[self.original_base + i]) {
                const shard_data = self.shards.shard(self.original_base + i);
                self.extractShard(shard_data, restored_out[count * self.shard_bytes ..]);
                restored_indices[count] = i;
                count += 1;
            }
        }

        return count;
    }

    fn decodeHighRate(self: *Decoder) void {
        const chunk_size = nextPow2(self.recovery_count);
        const original_end = chunk_size + self.original_count;

        // 1. ERASURE LOCATIONS
        var erasures: [GF_ORDER]GfElement = .{0} ** GF_ORDER;

        for (0..self.recovery_count) |i| {
            if (!self.received[i]) erasures[i] = 1;
        }
        // Gap between recovery_count and chunk_size
        for (self.recovery_count..chunk_size) |i| {
            erasures[i] = 1;
        }
        for (chunk_size..original_end) |i| {
            if (!self.received[i]) erasures[i] = 1;
        }

        // 2. EVALUATE POLYNOMIAL
        self.engine.evalPoly(&erasures, original_end);

        // 3. MULTIPLY SHARDS by erasure values
        for (0..self.recovery_count) |i| {
            if (self.received[i]) {
                self.engine.mul(self.shards.shardMut(i), erasures[i]);
            } else {
                self.shards.zeroShard(i);
            }
        }
        self.shards.zero_range(self.recovery_count, chunk_size);

        for (chunk_size..original_end) |i| {
            if (self.received[i]) {
                self.engine.mul(self.shards.shardMut(i), erasures[i]);
            } else {
                self.shards.zeroShard(i);
            }
        }
        self.shards.zero_range(original_end, self.work_count);

        // 4. IFFT / FORMAL DERIVATIVE / FFT
        self.engine.ifft(&self.shards, 0, self.work_count, original_end, 0);
        engine_mod.formalDerivative(&self.shards, self.work_count);
        self.engine.fft(&self.shards, 0, self.work_count, original_end, 0);

        // 5. REVEAL ERASURES
        for (chunk_size..original_end) |i| {
            if (!self.received[i]) {
                self.engine.mul(self.shards.shardMut(i), GF_MODULUS -| erasures[i]);
            }
        }
    }

    fn decodeLowRate(self: *Decoder) void {
        const chunk_size = nextPow2(self.original_count);
        const recovery_end = chunk_size + self.recovery_count;

        // 1. ERASURE LOCATIONS
        var erasures: [GF_ORDER]GfElement = .{0} ** GF_ORDER;

        for (0..self.original_count) |i| {
            if (!self.received[i]) erasures[i] = 1;
        }
        for (chunk_size..recovery_end) |i| {
            if (!self.received[i]) erasures[i] = 1;
        }
        // Everything beyond recovery_end is erased
        for (recovery_end..GF_ORDER) |i| {
            erasures[i] = 1;
        }

        // 2. EVALUATE POLYNOMIAL
        self.engine.evalPoly(&erasures, GF_ORDER);

        // 3. MULTIPLY SHARDS by erasure values
        for (0..self.original_count) |i| {
            if (self.received[i]) {
                self.engine.mul(self.shards.shardMut(i), erasures[i]);
            } else {
                self.shards.zeroShard(i);
            }
        }
        self.shards.zero_range(self.original_count, chunk_size);

        for (chunk_size..recovery_end) |i| {
            if (self.received[i]) {
                self.engine.mul(self.shards.shardMut(i), erasures[i]);
            } else {
                self.shards.zeroShard(i);
            }
        }
        self.shards.zero_range(recovery_end, self.work_count);

        // 4. IFFT / FORMAL DERIVATIVE / FFT
        self.engine.ifft(&self.shards, 0, self.work_count, recovery_end, 0);
        engine_mod.formalDerivative(&self.shards, self.work_count);
        self.engine.fft(&self.shards, 0, self.work_count, recovery_end, 0);

        // 5. REVEAL ERASURES
        for (0..self.original_count) |i| {
            if (!self.received[i]) {
                self.engine.mul(self.shards.shardMut(i), GF_MODULUS -| erasures[i]);
            }
        }
    }

    fn extractShard(self: *const Decoder, chunks: []const engine_mod.Chunk, out: []u8) void {
        const whole = self.shard_bytes / 64;
        const tail = self.shard_bytes % 64;

        for (0..whole) |i| {
            @memcpy(out[i * 64 ..][0..64], &chunks[i]);
        }

        if (tail > 0) {
            const half = tail / 2;
            @memcpy(out[whole * 64 ..][0..half], chunks[whole][0..half]);
            @memcpy(out[whole * 64 + half ..][0..half], chunks[whole][32 .. 32 + half]);
        }
    }

    fn useHighRate(original_count: usize, recovery_count: usize) bool {
        const o_pow2 = nextPow2(original_count);
        const r_pow2 = nextPow2(recovery_count);
        if (o_pow2 < r_pow2) return false;
        if (o_pow2 > r_pow2) return true;
        // Equal power-of-two: use "wrong" rate (counter-intuitive optimization)
        return original_count <= recovery_count;
    }
};

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

test "Encoder init and deinit" {
    var enc = try Encoder.init(testing.allocator, 3, 2, 64);
    defer enc.deinit();
    try testing.expectEqual(@as(usize, 3), enc.original_count);
    try testing.expectEqual(@as(usize, 2), enc.recovery_count);
}

test "Encoder add and encode" {
    const original_count = 3;
    const recovery_count = 2;
    const shard_bytes = 64;

    var enc = try Encoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer enc.deinit();

    const shards_data = [original_count][shard_bytes]u8{
        .{0xaa} ** shard_bytes,
        .{0xbb} ** shard_bytes,
        .{0xcc} ** shard_bytes,
    };

    for (&shards_data) |*s| {
        try enc.addOriginal(s);
    }

    var recovery_buf: [recovery_count * shard_bytes]u8 = undefined;
    try enc.encode(&recovery_buf);

    // Recovery shards should be non-trivial
    var non_zero = false;
    for (recovery_buf[0..shard_bytes]) |b| {
        if (b != 0) {
            non_zero = true;
            break;
        }
    }
    try testing.expect(non_zero);
}

test "Full encode → decode round-trip (high rate: 3 original, 2 recovery)" {
    const original_count = 3;
    const recovery_count = 2;
    const shard_bytes = 64;

    // Original data
    const originals = [original_count][shard_bytes]u8{
        .{0xaa} ** shard_bytes,
        .{0xbb} ** shard_bytes,
        .{0xcc} ** shard_bytes,
    };

    // Encode
    var enc = try Encoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer enc.deinit();
    for (&originals) |*s| try enc.addOriginal(s);

    var recovery_buf: [recovery_count * shard_bytes]u8 = undefined;
    try enc.encode(&recovery_buf);

    // Decode — simulate losing shard 0 and shard 2
    var dec = try Decoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer dec.deinit();

    try dec.addOriginal(1, &originals[1]); // only shard 1 survives
    try dec.addRecovery(0, recovery_buf[0..shard_bytes]);
    try dec.addRecovery(1, recovery_buf[shard_bytes .. 2 * shard_bytes]);

    var restored_buf: [original_count * shard_bytes]u8 = undefined;
    var restored_indices: [original_count]usize = undefined;
    const restored_count = try dec.decode(&restored_buf, &restored_indices);

    try testing.expectEqual(@as(usize, 2), restored_count);

    // Verify restored data matches originals
    for (0..restored_count) |i| {
        const idx = restored_indices[i];
        const restored = restored_buf[i * shard_bytes ..][0..shard_bytes];
        try testing.expectEqualSlices(u8, &originals[idx], restored);
    }
}

test "Full encode → decode round-trip (low rate: 2 original, 3 recovery)" {
    const original_count = 2;
    const recovery_count = 3;
    const shard_bytes = 64;

    const originals = [original_count][shard_bytes]u8{
        .{0x11} ** shard_bytes,
        .{0x22} ** shard_bytes,
    };

    var enc = try Encoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer enc.deinit();
    for (&originals) |*s| try enc.addOriginal(s);

    var recovery_buf: [recovery_count * shard_bytes]u8 = undefined;
    try enc.encode(&recovery_buf);

    // Lose both originals, recover from 2 of 3 recovery shards
    var dec = try Decoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer dec.deinit();

    try dec.addRecovery(0, recovery_buf[0..shard_bytes]);
    try dec.addRecovery(1, recovery_buf[shard_bytes .. 2 * shard_bytes]);

    var restored_buf: [original_count * shard_bytes]u8 = undefined;
    var restored_indices: [original_count]usize = undefined;
    const restored_count = try dec.decode(&restored_buf, &restored_indices);

    try testing.expectEqual(@as(usize, 2), restored_count);

    for (0..restored_count) |i| {
        const idx = restored_indices[i];
        const restored = restored_buf[i * shard_bytes ..][0..shard_bytes];
        try testing.expectEqualSlices(u8, &originals[idx], restored);
    }
}

test "nextPow2" {
    try testing.expectEqual(@as(usize, 1), nextPow2(0));
    try testing.expectEqual(@as(usize, 1), nextPow2(1));
    try testing.expectEqual(@as(usize, 2), nextPow2(2));
    try testing.expectEqual(@as(usize, 4), nextPow2(3));
    try testing.expectEqual(@as(usize, 8), nextPow2(5));
    try testing.expectEqual(@as(usize, 16), nextPow2(16));
    try testing.expectEqual(@as(usize, 32), nextPow2(17));
}
