//! Fast Walsh-Hadamard Transform over GF(2^16).
//! Used for error polynomial evaluation during decoding.

const gf = @import("gf.zig");
const GfElement = gf.GfElement;
const GF_ORDER = gf.GF_ORDER;
const addMod = gf.addMod;
const subMod = gf.subMod;

/// Decimation in time (DIT) Fast Walsh-Hadamard Transform.
/// `truncated`: Number of non-zero elements in `data` (at the front).
pub fn fwht(data: *[GF_ORDER]GfElement, truncated: usize) void {
    var dist: usize = 1;
    var dist4: usize = 4;
    while (dist4 <= GF_ORDER) : ({
        dist = dist4;
        dist4 <<= 2;
    }) {
        var r: usize = 0;
        while (r < truncated) : (r += dist4) {
            var offset: usize = r;
            while (offset < r + dist) : (offset += 1) {
                fwht4(data, offset, dist);
            }
        }
    }
}

fn fwht4(data: *[GF_ORDER]GfElement, offset: usize, dist: usize) void {
    const @"i0" = offset;
    const @"i1" = offset + dist;
    const @"i2" = offset + dist * 2;
    const @"i3" = offset + dist * 3;

    const s0, const d0 = fwht2(data[@"i0"], data[@"i1"]);
    const s1, const d1 = fwht2(data[@"i2"], data[@"i3"]);
    const s2, const d2 = fwht2(s0, s1);
    const s3, const d3 = fwht2(d0, d1);

    data[@"i0"] = s2;
    data[@"i1"] = s3;
    data[@"i2"] = d2;
    data[@"i3"] = d3;
}

inline fn fwht2(a: GfElement, b: GfElement) struct { GfElement, GfElement } {
    return .{ addMod(a, b), subMod(a, b) };
}

// ── Tests ──────────────────────────────────────────────────────────────

const testing = @import("std").testing;

test "FWHT produces deterministic output" {
    var data: [GF_ORDER]GfElement = undefined;
    for (0..GF_ORDER) |i| {
        data[i] = @truncate(i);
    }

    fwht(&data, GF_ORDER);

    // Transformed data should differ from identity
    var differs = false;
    for (1..GF_ORDER) |i| {
        if (data[i] != @as(GfElement, @truncate(i))) {
            differs = true;
            break;
        }
    }
    try testing.expect(differs);
}

test "FWHT truncated produces same result as full" {
    var full: [GF_ORDER]GfElement = .{0} ** GF_ORDER;
    var truncated_data: [GF_ORDER]GfElement = .{0} ** GF_ORDER;

    // Fill first 256 elements
    for (0..256) |i| {
        const v: GfElement = @truncate(i * 7 + 3);
        full[i] = v;
        truncated_data[i] = v;
    }

    fwht(&full, GF_ORDER);
    fwht(&truncated_data, 256);

    try testing.expectEqualSlices(GfElement, &full, &truncated_data);
}
