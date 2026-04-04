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
    if (n <= 1) return 1;
    // Use @clz for a portable, branch-free implementation
    const bits = @bitSizeOf(usize);
    const leading = @clz(n - 1);
    return @as(usize, 1) << @intCast(bits - leading);
}

/// Round up to next multiple of `chunk_size`.
fn nextPow2Multiple(n: usize, chunk_size: usize) usize {
    return ((n + chunk_size - 1) / chunk_size) * chunk_size;
}

fn shardLen64(shard_bytes: usize) Error!usize {
    if (shard_bytes == 0) return Error.InvalidArgs;
    if (shard_bytes % 2 != 0) return Error.ShardSizeNotEven;

    const padded = std.math.add(usize, shard_bytes, 63) catch return Error.InvalidArgs;
    return padded / 64;
}

fn checkedByteCount(count: usize, shard_bytes: usize) Error!usize {
    return std.math.mul(usize, count, shard_bytes) catch return Error.InvalidArgs;
}

fn ensureShardStorageFits(shard_count: usize, shard_len_64: usize) Error!void {
    _ = std.math.mul(usize, shard_count, shard_len_64) catch return Error.InvalidArgs;
}

// ── Public helpers ─────────────────────────────────────────────────────

/// Returns true if the given shard counts are supported.
pub fn supports(original_count: usize, recovery_count: usize) bool {
    return supportsShardCounts(original_count, recovery_count);
}

/// Returns the total size in bytes of the recovery data for the given configuration.
pub fn serializedSize(original_count: usize, recovery_count: usize, shard_bytes: usize) ?usize {
    if (!supportsShardCounts(original_count, recovery_count)) return null;
    if (shard_bytes == 0 or shard_bytes % 2 != 0) return null;
    return std.math.mul(usize, recovery_count, shard_bytes) catch null;
}

/// Encode in one call. Allocates encoder internally.
pub fn encode(
    allocator: Allocator,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    originals: []const []const u8,
    recovery_out: []u8,
) Error!void {
    var enc = try Encoder.init(allocator, original_count, recovery_count, shard_bytes);
    defer enc.deinit();
    for (originals) |shard| try enc.addOriginal(shard);
    try enc.encode(recovery_out);
}

/// Decode in one call. Allocates decoder internally.
/// `original_indices` and `original_shards` are the surviving originals.
/// `recovery_indices` and `recovery_shards` are the available recovery shards.
/// Returns the number of restored shards.
pub fn decode(
    allocator: Allocator,
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
    original_indices: []const usize,
    original_shards: []const []const u8,
    recovery_indices: []const usize,
    recovery_shards: []const []const u8,
    restored_out: []u8,
    restored_indices: []usize,
) Error!usize {
    if (original_indices.len != original_shards.len) return Error.InvalidArgs;
    if (recovery_indices.len != recovery_shards.len) return Error.InvalidArgs;

    var dec = try Decoder.init(allocator, original_count, recovery_count, shard_bytes);
    defer dec.deinit();
    for (original_indices, original_shards) |idx, shard| try dec.addOriginal(idx, shard);
    for (recovery_indices, recovery_shards) |idx, shard| try dec.addRecovery(idx, shard);
    return dec.decode(restored_out, restored_indices);
}

// ── Internal helpers ───────────────────────────────────────────────────

fn extractShard(shard_bytes: usize, chunks: []const engine_mod.Chunk, out: []u8) void {
    const whole = shard_bytes / 64;
    const tail = shard_bytes % 64;
    for (0..whole) |i| {
        @memcpy(out[i * 64 ..][0..64], &chunks[i]);
    }
    if (tail > 0) {
        const half = tail / 2;
        @memcpy(out[whole * 64 ..][0..half], chunks[whole][0..half]);
        @memcpy(out[whole * 64 + half ..][0..half], chunks[whole][32 .. 32 + half]);
    }
}

fn supportsShardCounts(original_count: usize, recovery_count: usize) bool {
    if (original_count == 0 or recovery_count == 0) return false;
    if (original_count >= GF_ORDER or recovery_count >= GF_ORDER) return false;

    const o_pow2 = nextPow2(original_count);
    const r_pow2 = nextPow2(recovery_count);
    const smaller_pow2 = @min(o_pow2, r_pow2);
    const larger = @max(original_count, recovery_count);

    return smaller_pow2 <= GF_ORDER - larger;
}

/// Matches the upstream `reed-solomon-simd` default rate-selection heuristic.
fn useHighRate(original_count: usize, recovery_count: usize) bool {
    const o_pow2 = nextPow2(original_count);
    const r_pow2 = nextPow2(recovery_count);

    if (o_pow2 < r_pow2) return false;
    if (o_pow2 > r_pow2) return true;

    // Equal power-of-two buckets intentionally prefer the "wrong" rate when
    // it is typically faster for skewed shard-count ratios.
    return original_count <= recovery_count;
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
        if (original_count == 0 or recovery_count == 0)
            return Error.InvalidArgs;
        if (!supportsShardCounts(original_count, recovery_count))
            return Error.InvalidArgs;

        const shard_len_64 = try shardLen64(shard_bytes);
        const total = nextPow2(original_count + recovery_count);
        try ensureShardStorageFits(total, shard_len_64);

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

    /// Reset the encoder to a new configuration, reusing the engine's shared tables.
    pub fn reset(self: *Encoder, original_count: usize, recovery_count: usize, shard_bytes: usize) Error!void {
        if (original_count == 0 or recovery_count == 0)
            return Error.InvalidArgs;
        if (!supportsShardCounts(original_count, recovery_count))
            return Error.InvalidArgs;

        const shard_len_64 = try shardLen64(shard_bytes);
        const total = nextPow2(original_count + recovery_count);
        try ensureShardStorageFits(total, shard_len_64);

        const new_shards = Shards.init(self.allocator, total, shard_len_64) catch return Error.OutOfMemory;
        var old_shards = self.shards;
        self.shards = new_shards;
        old_shards.deinit();

        self.original_count = original_count;
        self.recovery_count = recovery_count;
        self.shard_bytes = shard_bytes;
        self.shard_len_64 = shard_len_64;
        self.added = 0;
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
        const recovery_bytes = try checkedByteCount(self.recovery_count, self.shard_bytes);

        if (self.added != self.original_count) return Error.TooFewShards;
        if (out.len < recovery_bytes) return Error.InvalidArgs;

        if (useHighRate(self.original_count, self.recovery_count)) {
            self.encodeHighRate(self.original_count);
        } else {
            self.encodeLowRate(self.original_count);
        }

        // Extract recovery shards to output buffer
        // High rate: recovery at [0..recovery_count]
        // Low rate: recovery at [0..recovery_count]
        for (0..self.recovery_count) |i| {
            const shard_data = self.shards.shard(i);
            extractShard(self.shard_bytes, shard_data, out[i * self.shard_bytes ..]);
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
        if (original_count == 0 or recovery_count == 0)
            return Error.InvalidArgs;
        if (!supportsShardCounts(original_count, recovery_count))
            return Error.InvalidArgs;

        const shard_len_64 = try shardLen64(shard_bytes);

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
        try ensureShardStorageFits(work_count, shard_len_64);

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

    /// Reset the decoder to a new configuration, reusing the engine's shared tables.
    pub fn reset(self: *Decoder, original_count: usize, recovery_count: usize, shard_bytes: usize) Error!void {
        if (original_count == 0 or recovery_count == 0)
            return Error.InvalidArgs;
        if (!supportsShardCounts(original_count, recovery_count))
            return Error.InvalidArgs;

        const shard_len_64 = try shardLen64(shard_bytes);
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
        try ensureShardStorageFits(work_count, shard_len_64);

        var new_shards = Shards.init(self.allocator, work_count, shard_len_64) catch return Error.OutOfMemory;
        errdefer new_shards.deinit();

        const new_received = self.allocator.alloc(bool, work_count) catch return Error.OutOfMemory;
        errdefer self.allocator.free(new_received);
        @memset(new_received, false);

        var old_shards = self.shards;
        const old_received = self.received;

        self.shards = new_shards;
        self.received = new_received;

        old_shards.deinit();
        self.allocator.free(old_received);

        self.original_count = original_count;
        self.recovery_count = recovery_count;
        self.shard_bytes = shard_bytes;
        self.shard_len_64 = shard_len_64;
        self.high_rate = high_rate;
        self.original_base = original_base;
        self.recovery_base = recovery_base;
        self.work_count = work_count;
        self.original_recv_count = 0;
        self.recovery_recv_count = 0;
    }

    pub fn addOriginal(self: *Decoder, index: usize, data: []const u8) Error!void {
        if (index >= self.original_count) return Error.InvalidArgs;
        if (data.len != self.shard_bytes) return Error.ShardSizeMismatch;
        const pos = self.original_base + index;
        if (self.received[pos]) return Error.InvalidArgs;
        self.shards.insert(pos, data);
        self.received[pos] = true;
        self.original_recv_count += 1;
    }

    pub fn addRecovery(self: *Decoder, index: usize, data: []const u8) Error!void {
        if (index >= self.recovery_count) return Error.InvalidArgs;
        if (data.len != self.shard_bytes) return Error.ShardSizeMismatch;
        const pos = self.recovery_base + index;
        if (self.received[pos]) return Error.InvalidArgs;
        self.shards.insert(pos, data);
        self.received[pos] = true;
        self.recovery_recv_count += 1;
    }

    /// Decode missing original shards.
    /// Returns the number of restored shards.
    /// `restored_out` must be at least `original_count * shard_bytes` bytes.
    /// `restored_indices` must be at least `original_count` entries.
    pub fn decode(self: *Decoder, restored_out: []u8, restored_indices: []usize) Error!usize {
        const original_bytes = try checkedByteCount(self.original_count, self.shard_bytes);

        if (restored_out.len < original_bytes)
            return Error.InvalidArgs;
        if (restored_indices.len < self.original_count)
            return Error.InvalidArgs;
        if (self.original_recv_count + self.recovery_recv_count < self.original_count)
            return Error.TooFewShards;

        // Count missing originals
        var missing: usize = 0;
        for (0..self.original_count) |i| {
            if (!self.received[self.original_base + i]) missing += 1;
        }
        if (missing == 0) {
            self.resetState();
            return 0;
        }

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
                extractShard(self.shard_bytes, shard_data, restored_out[count * self.shard_bytes ..]);
                restored_indices[count] = i;
                count += 1;
            }
        }

        self.resetState();
        return count;
    }

    fn resetState(self: *Decoder) void {
        @memset(self.received, false);
        self.original_recv_count = 0;
        self.recovery_recv_count = 0;
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
};

// ── Tests ──────────────────────────────────────────────────────────────

const testing = std.testing;

const RandomizedConfig = struct {
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
};

fn shuffleIndices(random: std.Random, indices: []usize) void {
    if (indices.len <= 1) return;

    var i = indices.len;
    while (i > 1) {
        i -= 1;
        const j = random.uintLessThan(usize, i + 1);
        std.mem.swap(usize, &indices[i], &indices[j]);
    }
}

fn randomSupportedConfig(random: std.Random) RandomizedConfig {
    const shard_byte_options = [_]usize{ 2, 4, 6, 62, 64, 66, 96, 128, 254, 256, 510, 512, 1024, 4100 };

    while (true) {
        const original_count = random.uintLessThan(usize, 32) + 1;
        const recovery_count = random.uintLessThan(usize, 32) + 1;
        if (!supports(original_count, recovery_count)) continue;

        return .{
            .original_count = original_count,
            .recovery_count = recovery_count,
            .shard_bytes = shard_byte_options[random.uintLessThan(usize, shard_byte_options.len)],
        };
    }
}

fn allocRandomOriginals(allocator: Allocator, random: std.Random, original_count: usize, shard_bytes: usize) ![][]u8 {
    const originals = try allocator.alloc([]u8, original_count);
    errdefer allocator.free(originals);

    var allocated: usize = 0;
    errdefer {
        for (originals[0..allocated]) |shard| allocator.free(shard);
    }

    for (0..original_count) |i| {
        originals[i] = try allocator.alloc(u8, shard_bytes);
        allocated += 1;
        random.bytes(originals[i]);
    }

    return originals;
}

fn freeOriginals(allocator: Allocator, originals: [][]u8) void {
    for (originals) |shard| allocator.free(shard);
    allocator.free(originals);
}

fn countMissing(missing_flags: []const bool) usize {
    var count: usize = 0;
    for (missing_flags) |is_missing| {
        if (is_missing) count += 1;
    }
    return count;
}

fn verifyRestored(
    originals: [][]u8,
    missing_flags: []const bool,
    restored_count: usize,
    restored_out: []const u8,
    restored_indices: []const usize,
    shard_bytes: usize,
) !void {
    try testing.expectEqual(countMissing(missing_flags), restored_count);

    var seen_missing: usize = 0;
    var restored_seen: [32]bool = .{false} ** 32;

    for (0..restored_count) |i| {
        const idx = restored_indices[i];
        try testing.expect(idx < originals.len);
        try testing.expect(missing_flags[idx]);
        try testing.expect(!restored_seen[idx]);

        restored_seen[idx] = true;
        seen_missing += 1;

        const restored = restored_out[i * shard_bytes ..][0..shard_bytes];
        try testing.expectEqualSlices(u8, originals[idx], restored);
    }

    try testing.expectEqual(countMissing(missing_flags), seen_missing);
}

fn chooseDecodePattern(
    random: std.Random,
    original_count: usize,
    recovery_count: usize,
    missing_flags: *[32]bool,
    original_order: *[32]usize,
    recovery_order: *[32]usize,
) struct { missing_count: usize, recovery_used: usize } {
    @memset(missing_flags, false);

    for (0..original_count) |i| original_order[i] = i;
    for (0..recovery_count) |i| recovery_order[i] = i;

    shuffleIndices(random, original_order[0..original_count]);
    shuffleIndices(random, recovery_order[0..recovery_count]);

    const max_missing = @min(original_count, recovery_count);
    const missing_count = random.uintLessThan(usize, max_missing + 1);
    for (original_order[0..missing_count]) |idx| missing_flags[idx] = true;

    const recovery_used = missing_count + random.uintLessThan(usize, recovery_count - missing_count + 1);
    return .{ .missing_count = missing_count, .recovery_used = recovery_used };
}

fn runRandomRoundTripCase(allocator: Allocator, random: std.Random, enc: *Encoder, dec: *Decoder) !void {
    const cfg = randomSupportedConfig(random);

    try enc.reset(cfg.original_count, cfg.recovery_count, cfg.shard_bytes);
    try dec.reset(cfg.original_count, cfg.recovery_count, cfg.shard_bytes);

    const originals = try allocRandomOriginals(allocator, random, cfg.original_count, cfg.shard_bytes);
    defer freeOriginals(allocator, originals);

    const recovery_buf = try allocator.alloc(u8, try checkedByteCount(cfg.recovery_count, cfg.shard_bytes));
    defer allocator.free(recovery_buf);

    for (originals) |shard| try enc.addOriginal(shard);
    try enc.encode(recovery_buf);

    var missing_flags: [32]bool = .{false} ** 32;
    var original_order: [32]usize = undefined;
    var recovery_order: [32]usize = undefined;
    const pattern = chooseDecodePattern(
        random,
        cfg.original_count,
        cfg.recovery_count,
        &missing_flags,
        &original_order,
        &recovery_order,
    );

    for (original_order[0..cfg.original_count]) |idx| {
        if (!missing_flags[idx]) try dec.addOriginal(idx, originals[idx]);
    }
    for (recovery_order[0..pattern.recovery_used]) |idx| {
        const start = idx * cfg.shard_bytes;
        try dec.addRecovery(idx, recovery_buf[start .. start + cfg.shard_bytes]);
    }

    const restored_out = try allocator.alloc(u8, try checkedByteCount(cfg.original_count, cfg.shard_bytes));
    defer allocator.free(restored_out);
    const restored_indices = try allocator.alloc(usize, cfg.original_count);
    defer allocator.free(restored_indices);

    const restored_count = try dec.decode(restored_out, restored_indices);
    try testing.expectEqual(pattern.missing_count, restored_count);
    try verifyRestored(
        originals,
        missing_flags[0..cfg.original_count],
        restored_count,
        restored_out,
        restored_indices,
        cfg.shard_bytes,
    );
}

fn runRandomWrapperRoundTripCase(allocator: Allocator, random: std.Random) !void {
    const cfg = randomSupportedConfig(random);
    const originals = try allocRandomOriginals(allocator, random, cfg.original_count, cfg.shard_bytes);
    defer freeOriginals(allocator, originals);

    const recovery_buf = try allocator.alloc(u8, try checkedByteCount(cfg.recovery_count, cfg.shard_bytes));
    defer allocator.free(recovery_buf);
    try encode(allocator, cfg.original_count, cfg.recovery_count, cfg.shard_bytes, originals, recovery_buf);

    var missing_flags: [32]bool = .{false} ** 32;
    var original_order: [32]usize = undefined;
    var recovery_order: [32]usize = undefined;
    const pattern = chooseDecodePattern(
        random,
        cfg.original_count,
        cfg.recovery_count,
        &missing_flags,
        &original_order,
        &recovery_order,
    );

    const survivor_count = cfg.original_count - pattern.missing_count;
    const original_indices = try allocator.alloc(usize, survivor_count);
    defer allocator.free(original_indices);
    const original_shards = try allocator.alloc([]const u8, survivor_count);
    defer allocator.free(original_shards);

    var survivor_idx: usize = 0;
    for (original_order[0..cfg.original_count]) |idx| {
        if (missing_flags[idx]) continue;
        original_indices[survivor_idx] = idx;
        original_shards[survivor_idx] = originals[idx];
        survivor_idx += 1;
    }

    const recovery_indices = try allocator.alloc(usize, pattern.recovery_used);
    defer allocator.free(recovery_indices);
    const recovery_shards = try allocator.alloc([]const u8, pattern.recovery_used);
    defer allocator.free(recovery_shards);

    for (recovery_order[0..pattern.recovery_used], 0..) |idx, i| {
        recovery_indices[i] = idx;
        const start = idx * cfg.shard_bytes;
        recovery_shards[i] = recovery_buf[start .. start + cfg.shard_bytes];
    }

    const restored_out = try allocator.alloc(u8, try checkedByteCount(cfg.original_count, cfg.shard_bytes));
    defer allocator.free(restored_out);
    const restored_indices = try allocator.alloc(usize, cfg.original_count);
    defer allocator.free(restored_indices);

    const restored_count = try decode(
        allocator,
        cfg.original_count,
        cfg.recovery_count,
        cfg.shard_bytes,
        original_indices,
        original_shards,
        recovery_indices,
        recovery_shards,
        restored_out,
        restored_indices,
    );

    try testing.expectEqual(pattern.missing_count, restored_count);
    try verifyRestored(
        originals,
        missing_flags[0..cfg.original_count],
        restored_count,
        restored_out,
        restored_indices,
        cfg.shard_bytes,
    );
}

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

test "Encoder can be reused with non-64-byte shards" {
    const original_count = 2;
    const recovery_count = 3;
    const shard_bytes = 66;

    const originals = [original_count][shard_bytes]u8{
        .{0x11} ** shard_bytes,
        .{0x22} ** shard_bytes,
    };

    var enc = try Encoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer enc.deinit();

    var first: [recovery_count * shard_bytes]u8 = undefined;
    var second: [recovery_count * shard_bytes]u8 = undefined;

    for (&originals) |*s| try enc.addOriginal(s);
    try enc.encode(&first);

    for (&originals) |*s| try enc.addOriginal(s);
    try enc.encode(&second);

    try testing.expectEqualSlices(u8, &first, &second);
}

test "Decoder can be reused after successful decode" {
    const original_count = 2;
    const recovery_count = 3;
    const shard_bytes = 66;

    const originals_a = [original_count][shard_bytes]u8{
        .{0x31} ** shard_bytes,
        .{0x42} ** shard_bytes,
    };
    const originals_b = [original_count][shard_bytes]u8{
        .{0x51} ** shard_bytes,
        .{0x62} ** shard_bytes,
    };

    var enc = try Encoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer enc.deinit();

    var dec = try Decoder.init(testing.allocator, original_count, recovery_count, shard_bytes);
    defer dec.deinit();

    var recovery_a: [recovery_count * shard_bytes]u8 = undefined;
    for (&originals_a) |*s| try enc.addOriginal(s);
    try enc.encode(&recovery_a);

    try dec.addRecovery(0, recovery_a[0..shard_bytes]);
    try dec.addRecovery(1, recovery_a[shard_bytes .. 2 * shard_bytes]);

    var restored_a: [original_count * shard_bytes]u8 = undefined;
    var indices_a: [original_count]usize = undefined;
    const restored_count_a = try dec.decode(&restored_a, &indices_a);
    try testing.expectEqual(@as(usize, 2), restored_count_a);
    for (0..restored_count_a) |i| {
        const idx = indices_a[i];
        const restored = restored_a[i * shard_bytes ..][0..shard_bytes];
        try testing.expectEqualSlices(u8, &originals_a[idx], restored);
    }

    var recovery_b: [recovery_count * shard_bytes]u8 = undefined;
    for (&originals_b) |*s| try enc.addOriginal(s);
    try enc.encode(&recovery_b);

    try dec.addRecovery(0, recovery_b[0..shard_bytes]);
    try dec.addRecovery(2, recovery_b[2 * shard_bytes .. 3 * shard_bytes]);

    var restored_b: [original_count * shard_bytes]u8 = undefined;
    var indices_b: [original_count]usize = undefined;
    const restored_count_b = try dec.decode(&restored_b, &indices_b);
    try testing.expectEqual(@as(usize, 2), restored_count_b);
    for (0..restored_count_b) |i| {
        const idx = indices_b[i];
        const restored = restored_b[i * shard_bytes ..][0..shard_bytes];
        try testing.expectEqualSlices(u8, &originals_b[idx], restored);
    }
}

test "randomized round-trips across encoder and decoder resets" {
    var prng = std.Random.DefaultPrng.init(0x5eed_1234_89ab_cdef);
    const random = prng.random();

    var enc = try Encoder.init(testing.allocator, 1, 1, 2);
    defer enc.deinit();
    var dec = try Decoder.init(testing.allocator, 1, 1, 2);
    defer dec.deinit();

    for (0..200) |_| {
        try runRandomRoundTripCase(testing.allocator, random, &enc, &dec);
    }
}

test "randomized one-shot encode/decode wrappers round-trip" {
    var prng = std.Random.DefaultPrng.init(0x1234_5678_dead_beef);
    const random = prng.random();

    for (0..120) |_| {
        try runRandomWrapperRoundTripCase(testing.allocator, random);
    }
}

test "Unsupported shard-count combinations are rejected" {
    try testing.expect(supports(4096, 61440));
    try testing.expect(supports(61440, 4096));
    try testing.expect(!supports(4096, 61441));
    try testing.expect(!supports(61441, 4096));

    try testing.expectError(Error.InvalidArgs, Encoder.init(testing.allocator, 4096, 61441, 64));
    try testing.expectError(Error.InvalidArgs, Decoder.init(testing.allocator, 61441, 4096, 64));
}

test "Decoder rejects duplicate shard indexes" {
    var dec = try Decoder.init(testing.allocator, 2, 3, 64);
    defer dec.deinit();

    const shard = [_]u8{0xaa} ** 64;
    try dec.addRecovery(0, &shard);
    try testing.expectError(Error.InvalidArgs, dec.addRecovery(0, &shard));
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

test "serializedSize rejects overflow" {
    const huge = std.math.maxInt(usize) - 1;
    try testing.expectEqual(@as(?usize, null), serializedSize(1, 2, huge));
}

test "oversized shard sizes are rejected before allocation" {
    const huge = std.math.maxInt(usize) - 1;
    try testing.expectError(Error.InvalidArgs, Encoder.init(testing.allocator, 1, 1, huge));
    try testing.expectError(Error.InvalidArgs, Decoder.init(testing.allocator, 1, 1, huge));
}

test "useHighRate matches upstream heuristic examples" {
    try testing.expect(useHighRate(3, 2));
    try testing.expect(!useHighRate(2, 3));
    try testing.expect(useHighRate(5, 8));
    try testing.expect(!useHighRate(8, 5));
    try testing.expect(useHighRate(9, 16));
    try testing.expect(!useHighRate(16, 9));
}

test "decode wrapper rejects mismatched paired input slices" {
    var restored_out: [2]u8 = undefined;
    var restored_indices: [1]usize = undefined;
    const shard = [_]u8{ 0xaa, 0xbb };

    try testing.expectError(Error.InvalidArgs, decode(
        testing.allocator,
        1,
        1,
        2,
        &.{0},
        &.{},
        &.{},
        &.{},
        &restored_out,
        &restored_indices,
    ));

    try testing.expectError(Error.InvalidArgs, decode(
        testing.allocator,
        1,
        1,
        2,
        &.{},
        &.{&shard},
        &.{},
        &.{},
        &restored_out,
        &restored_indices,
    ));

    try testing.expectError(Error.InvalidArgs, decode(
        testing.allocator,
        1,
        1,
        2,
        &.{},
        &.{},
        &.{0},
        &.{},
        &restored_out,
        &restored_indices,
    ));
}

fn encoderResetHandlesAllocationFailures(allocator: Allocator) !void {
    var enc = try Encoder.init(allocator, 3, 2, 64);
    defer enc.deinit();

    try enc.reset(9, 16, 128);
}

test "Encoder.reset is allocation-failure safe" {
    try testing.checkAllAllocationFailures(testing.allocator, encoderResetHandlesAllocationFailures, .{});
}

fn decoderResetHandlesAllocationFailures(allocator: Allocator) !void {
    var dec = try Decoder.init(allocator, 2, 3, 64);
    defer dec.deinit();

    try dec.reset(9, 16, 128);
}

test "Decoder.reset is allocation-failure safe" {
    try testing.checkAllAllocationFailures(testing.allocator, decoderResetHandlesAllocationFailures, .{});
}
