const std = @import("std");
const codec = @import("reed_solomon").codec;

const Enc = opaque {};
const Dec = opaque {};

extern "c" fn rs_encoder_new(original_count: usize, recovery_count: usize, shard_bytes: usize) ?*Enc;
extern "c" fn rs_encoder_add_original(enc: *Enc, data: [*]const u8, len: usize) i32;
extern "c" fn rs_encoder_encode(enc: *Enc, out: [*]u8) i32;
extern "c" fn rs_encoder_free(enc: *Enc) void;

extern "c" fn rs_decoder_new(original_count: usize, recovery_count: usize, shard_bytes: usize) ?*Dec;
extern "c" fn rs_decoder_add_original(dec: *Dec, index: usize, data: [*]const u8, len: usize) i32;
extern "c" fn rs_decoder_add_recovery(dec: *Dec, index: usize, data: [*]const u8, len: usize) i32;
extern "c" fn rs_decoder_decode(dec: *Dec, out: [*]u8, indices: [*]usize, capacity: usize) isize;
extern "c" fn rs_decoder_free(dec: *Dec) void;

const Config = struct {
    original_count: usize,
    recovery_count: usize,
    shard_bytes: usize,
};

fn checkedByteCount(count: usize, shard_bytes: usize) !usize {
    return std.math.mul(usize, count, shard_bytes);
}

fn shuffleIndices(random: std.Random, indices: []usize) void {
    if (indices.len <= 1) return;

    var i = indices.len;
    while (i > 1) {
        i -= 1;
        const j = random.uintLessThan(usize, i + 1);
        std.mem.swap(usize, &indices[i], &indices[j]);
    }
}

fn randomSupportedConfig(random: std.Random) Config {
    const shard_byte_options = [_]usize{ 2, 4, 6, 62, 64, 66, 96, 128, 254, 256, 510, 512, 1024, 4100 };

    while (true) {
        const original_count = random.uintLessThan(usize, 32) + 1;
        const recovery_count = random.uintLessThan(usize, 32) + 1;
        if (!codec.supports(original_count, recovery_count)) continue;

        return .{
            .original_count = original_count,
            .recovery_count = recovery_count,
            .shard_bytes = shard_byte_options[random.uintLessThan(usize, shard_byte_options.len)],
        };
    }
}

fn allocOriginals(allocator: std.mem.Allocator, random: std.Random, original_count: usize, shard_bytes: usize) ![][]u8 {
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

fn freeOriginals(allocator: std.mem.Allocator, originals: [][]u8) void {
    for (originals) |shard| allocator.free(shard);
    allocator.free(originals);
}

fn findIndex(indices: []const usize, target: usize) ?usize {
    for (indices, 0..) |idx, i| {
        if (idx == target) return i;
    }
    return null;
}

fn fuzzOneCase(allocator: std.mem.Allocator, random: std.Random, iteration: usize, seed: u64) !void {
    const cfg = randomSupportedConfig(random);
    const originals = try allocOriginals(allocator, random, cfg.original_count, cfg.shard_bytes);
    defer freeOriginals(allocator, originals);

    const recovery_bytes = try checkedByteCount(cfg.recovery_count, cfg.shard_bytes);
    const restored_bytes = try checkedByteCount(cfg.original_count, cfg.shard_bytes);

    const rust_recovery = try allocator.alloc(u8, recovery_bytes);
    defer allocator.free(rust_recovery);
    const zig_recovery = try allocator.alloc(u8, recovery_bytes);
    defer allocator.free(zig_recovery);

    const rust_enc = rs_encoder_new(cfg.original_count, cfg.recovery_count, cfg.shard_bytes) orelse return error.RustEncoderInitFailed;
    defer rs_encoder_free(rust_enc);
    for (originals) |shard| {
        if (rs_encoder_add_original(rust_enc, shard.ptr, shard.len) != 0) return error.RustEncoderAddFailed;
    }
    if (rs_encoder_encode(rust_enc, rust_recovery.ptr) != 0) return error.RustEncoderEncodeFailed;

    var zig_enc = try codec.Encoder.init(allocator, cfg.original_count, cfg.recovery_count, cfg.shard_bytes);
    defer zig_enc.deinit();
    for (originals) |shard| try zig_enc.addOriginal(shard);
    try zig_enc.encode(zig_recovery);

    if (!std.mem.eql(u8, zig_recovery, rust_recovery)) {
        std.debug.print(
            "encode mismatch at case {d} seed {d} cfg {d}+{d} shard_bytes={d}\n",
            .{ iteration, seed, cfg.original_count, cfg.recovery_count, cfg.shard_bytes },
        );
        return error.EncodeMismatch;
    }

    var missing_flags: [32]bool = .{false} ** 32;
    var original_order: [32]usize = undefined;
    var recovery_order: [32]usize = undefined;
    for (0..cfg.original_count) |i| original_order[i] = i;
    for (0..cfg.recovery_count) |i| recovery_order[i] = i;
    shuffleIndices(random, original_order[0..cfg.original_count]);
    shuffleIndices(random, recovery_order[0..cfg.recovery_count]);

    const max_missing = @min(cfg.original_count, cfg.recovery_count);
    const missing_count = random.uintLessThan(usize, max_missing + 1);
    for (original_order[0..missing_count]) |idx| missing_flags[idx] = true;
    const recovery_used = missing_count + random.uintLessThan(usize, cfg.recovery_count - missing_count + 1);

    var zig_dec = try codec.Decoder.init(allocator, cfg.original_count, cfg.recovery_count, cfg.shard_bytes);
    defer zig_dec.deinit();
    const rust_dec = rs_decoder_new(cfg.original_count, cfg.recovery_count, cfg.shard_bytes) orelse return error.RustDecoderInitFailed;
    defer rs_decoder_free(rust_dec);

    for (original_order[0..cfg.original_count]) |idx| {
        if (missing_flags[idx]) continue;

        try zig_dec.addOriginal(idx, originals[idx]);
        if (rs_decoder_add_original(rust_dec, idx, originals[idx].ptr, originals[idx].len) != 0) {
            return error.RustDecoderAddOriginalFailed;
        }
    }

    for (recovery_order[0..recovery_used]) |idx| {
        const start = idx * cfg.shard_bytes;
        const shard = rust_recovery[start .. start + cfg.shard_bytes];
        try zig_dec.addRecovery(idx, shard);
        if (rs_decoder_add_recovery(rust_dec, idx, shard.ptr, shard.len) != 0) {
            return error.RustDecoderAddRecoveryFailed;
        }
    }

    const zig_restored = try allocator.alloc(u8, restored_bytes);
    defer allocator.free(zig_restored);
    const zig_indices = try allocator.alloc(usize, cfg.original_count);
    defer allocator.free(zig_indices);

    const rust_restored = try allocator.alloc(u8, restored_bytes);
    defer allocator.free(rust_restored);
    const rust_indices = try allocator.alloc(usize, cfg.original_count);
    defer allocator.free(rust_indices);

    const zig_count = try zig_dec.decode(zig_restored, zig_indices);
    const rust_count_signed = rs_decoder_decode(rust_dec, rust_restored.ptr, rust_indices.ptr, cfg.original_count);
    if (rust_count_signed < 0) return error.RustDecodeFailed;
    const rust_count: usize = @intCast(rust_count_signed);

    if (zig_count != missing_count or rust_count != missing_count or zig_count != rust_count) {
        std.debug.print(
            "count mismatch at case {d} seed {d} cfg {d}+{d} shard_bytes={d} missing={d} recovery_used={d} zig={d} rust={d}\n",
            .{ iteration, seed, cfg.original_count, cfg.recovery_count, cfg.shard_bytes, missing_count, recovery_used, zig_count, rust_count },
        );
        return error.RestoredCountMismatch;
    }

    for (0..cfg.original_count) |idx| {
        if (!missing_flags[idx]) continue;

        const zig_pos = findIndex(zig_indices[0..zig_count], idx) orelse return error.MissingZigRestoredIndex;
        const rust_pos = findIndex(rust_indices[0..rust_count], idx) orelse return error.MissingRustRestoredIndex;

        const zig_shard = zig_restored[zig_pos * cfg.shard_bytes ..][0..cfg.shard_bytes];
        const rust_shard = rust_restored[rust_pos * cfg.shard_bytes ..][0..cfg.shard_bytes];

        if (!std.mem.eql(u8, zig_shard, rust_shard)) {
            std.debug.print(
                "decode mismatch at case {d} seed {d} cfg {d}+{d} shard_bytes={d} index={d}\n",
                .{ iteration, seed, cfg.original_count, cfg.recovery_count, cfg.shard_bytes, idx },
            );
            return error.DecodeMismatch;
        }

        if (!std.mem.eql(u8, zig_shard, originals[idx])) {
            std.debug.print(
                "restored shard mismatch vs original at case {d} seed {d} cfg {d}+{d} shard_bytes={d} index={d}\n",
                .{ iteration, seed, cfg.original_count, cfg.recovery_count, cfg.shard_bytes, idx },
            );
            return error.RestoredOriginalMismatch;
        }
    }
}

pub fn main() !void {
    const allocator = std.heap.page_allocator;
    const args = try std.process.argsAlloc(allocator);
    defer std.process.argsFree(allocator, args);

    const iterations = if (args.len >= 2) try std.fmt.parseInt(usize, args[1], 10) else 10_000;
    const seed = if (args.len >= 3) try std.fmt.parseInt(u64, args[2], 10) else 0x0ddc_0ffe_e123_4567;

    var prng = std.Random.DefaultPrng.init(seed);
    const random = prng.random();

    std.debug.print("starting differential fuzz: iterations={d} seed={d}\n", .{ iterations, seed });
    for (0..iterations) |i| {
        try fuzzOneCase(allocator, random, i, seed);
        if ((i + 1) % 1000 == 0 or i + 1 == iterations) {
            std.debug.print("completed {d}/{d}\n", .{ i + 1, iterations });
        }
    }
    std.debug.print("differential fuzz passed: iterations={d} seed={d}\n", .{ iterations, seed });
}
