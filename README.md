# reed-solomon-zig

Reed-Solomon erasure coding for Zig, compatible with the Leopard-RS / `reed-solomon-simd` format.

Implements Leopard-RS over GF(2^16) with platform-specific SIMD acceleration. In local comparison runs it produced byte-identical output to the Rust [`reed-solomon-simd`](https://github.com/AndersTrier/reed-solomon-simd) crate across representative configurations, and the repository includes vendored differential-fuzz and benchmark tooling for reproducing those checks.

## Usage

```zig
const rs = @import("reed_solomon");

// Encode: 3 original shards → 2 recovery shards
var enc = try rs.Encoder.init(allocator, 3, 2, shard_bytes);
defer enc.deinit();

for (original_shards) |shard| try enc.addOriginal(shard);
try enc.encode(recovery_buffer);

// Decode: recover missing originals from any 3 of 5 shards
var dec = try rs.Decoder.init(allocator, 3, 2, shard_bytes);
defer dec.deinit();

try dec.addOriginal(1, shard_1);     // shard 0 and 2 are lost
try dec.addRecovery(0, recovery_0);
try dec.addRecovery(1, recovery_1);

const restored_count = try dec.decode(restored_buffer, indices_buffer);
```

### Key properties

- **Shard bytes must be even** (Reed-Solomon over GF(2^16) operates on 16-bit elements)
- **Supported shard-count combinations match the same GF(2^16) limits used by `reed-solomon-simd`**
- **Encoder/decoder are reusable** — encode/decode resets internal state automatically
- **Thread-safe shared table initialization** — lookup tables are computed once per process

## Performance

The numbers below are from local runs against `reed-solomon-simd` on the listed machines. The Zig implementation is generally near parity and often modestly faster, but not every configuration is a win.

### ARM — Apple M5 (AArch64 NEON)

| Config | Zig | Rust | Ratio |
|---|---|---|---|
| small (3+2, 1KB) | 0.1 μs | 0.1 μs | 1.01x |
| medium (10+5, 4KB) | 4.6 μs | 4.5 μs | 1.02x |
| large (50+25, 4KB) | 32.9 μs | 34.2 μs | **0.96x** |
| xlarge (100+50, 1KB) | 19.4 μs | 21.7 μs | **0.89x** |
| big shards (10+5, 64KB) | 80.4 μs | 80.9 μs | **1.00x** |

### x86 — AMD Ryzen 9 9950X3D (AVX2)

| Config | Zig | Rust | Ratio |
|---|---|---|---|
| small (3+2, 1KB) | 0.1 μs | 0.1 μs | **0.70x** |
| medium (10+5, 4KB) | 2.6 μs | 3.0 μs | **0.81x** |
| large (50+25, 4KB) | 19.6 μs | 20.1 μs | **0.97x** |
| xlarge (100+50, 1KB) | 11.3 μs | 12.2 μs | **0.93x** |
| big shards (10+5, 64KB) | 46.0 μs | 50.5 μs | **0.91x** |

*Ratio < 1.0 = Zig faster. All configs produce byte-identical output (verified).*

### Implementation notes

- Like the Rust reference, the hot path uses fused butterfly kernels for FFT/IFFT work.
- On AArch64, the lookup-heavy multiply path uses inline `tbl` instructions.
- On x86 targets built with AVX2 enabled, the shuffle-heavy hot path lives in a small C helper (`src/simd_x86.c`).
- Shared lookup tables amortize the one-time initialization cost across encoder/decoder instances.

## Architecture

```
src/
  reed_solomon.zig   Root module — re-exports Encoder, Decoder
  codec.zig          Encoder/Decoder with high-rate + low-rate paths
  engine.zig         FFT/IFFT, butterfly kernels, shard management
  gf.zig             GF(2^16) arithmetic, comptime log/exp tables, SIMD multiply
  fwht.zig           Fast Walsh-Hadamard Transform
  tables.zig         Cantor-basis exp/log, skew table, Mul128 table init
  simd_x86.c         AVX2 intrinsics for x86 (compiled by zig cc)
```

## SIMD support

| Platform | Engine | Instructions |
|---|---|---|
| AArch64 (ARM) | Zig inline asm | NEON `tbl` (128-bit) |
| x86/x86_64 targets built with AVX2 | C helper | `vpshufb` (256-bit) |
| Other | Scalar fallback | Element-by-element lookup |

The x86 AVX2 path uses a small C file (`simd_x86.c`) compiled with `-mavx2` by Zig's built-in C compiler. This keeps the shuffle-heavy hot path simple and reliable while the rest of the implementation stays in Zig.

## Install

Add to `build.zig.zon`:

```zig
.dependencies = .{
    .reed_solomon = .{
        .url = "https://github.com/unconfirmedlabs/reed-solomon-zig/archive/<commit>.tar.gz",
        .hash = "...",
    },
},
```

Then in `build.zig`:

```zig
const rs = b.dependency("reed_solomon", .{ .target = target, .optimize = optimize });
exe.root_module.addImport("reed_solomon", rs.module("reed_solomon"));
```

## Differential fuzzing

The repository includes a vendored Rust reference shim and a Zig differential harness:

```bash
zig build diff-fuzz -Doptimize=ReleaseFast
```

Useful knobs:

```bash
zig build diff-fuzz -Doptimize=ReleaseFast -Ddiff-iters=50000 -Ddiff-seed=1311768467463790320
```

This target:

- builds `tools/rs-ffi` in locked offline mode
- links it into `tools/diff_decode_fuzz.zig`
- compares Zig encode/decode behavior against the vendored Rust `reed-solomon-simd` reference over randomized shard sizes, shard-count pairs, and erasure patterns

Run it with the default host target. It is a runtime validation step, not a cross-compilation target.

## Reproducing benchmarks

Requires: Zig 0.15+, Rust stable, both on the same machine.

### 1. Build the Rust reference FFI wrapper

```bash
cargo build --release --locked --offline --manifest-path tools/rs-ffi/Cargo.toml
```

### 2. Build and run the benchmark

Save this as `bench.zig`:

```zig
const std = @import("std");
const zig_codec = @import("codec");
const Opq = opaque {};
extern "c" fn rs_encoder_new(o: usize, r: usize, s: usize) ?*Opq;
extern "c" fn rs_encoder_add_original(e: *Opq, d: [*]const u8, l: usize) i32;
extern "c" fn rs_encoder_encode(e: *Opq, o: [*]u8) i32;
extern "c" fn rs_encoder_free(e: *Opq) void;

pub fn main() !void {
    const alloc = std.heap.page_allocator;
    const cfgs = [_]struct{o:usize,r:usize,s:usize,it:usize,l:[]const u8}{
        .{.o=3,.r=2,.s=1024,.it=10000,.l="small (3+2, 1KB)"},
        .{.o=10,.r=5,.s=4096,.it=2000,.l="medium (10+5, 4KB)"},
        .{.o=50,.r=25,.s=4096,.it=200,.l="large (50+25, 4KB)"},
        .{.o=100,.r=50,.s=1024,.it=100,.l="xlarge (100+50, 1KB)"},
        .{.o=10,.r=5,.s=65536,.it=200,.l="big shards (10+5, 64KB)"},
    };
    std.debug.print("\n{s:<30} {s:>12} {s:>12} {s:>8}\n", .{"Config","Zig (us)","Rust (us)","Ratio"});
    std.debug.print("{s:<30} {s:>12} {s:>12} {s:>8}\n",
        .{"------------------------------","------------","------------","--------"});
    for (cfgs) |c| {
        const sh = try alloc.alloc([]u8, c.o);
        for (sh,0..) |*s,i| {
            s.*=try alloc.alloc(u8,c.s);
            for(s.*,0..) |*b,j| b.*=@truncate(i*%97+%j*%31);
        }
        const zb=try alloc.alloc(u8,c.r*c.s);
        const rb=try alloc.alloc(u8,c.r*c.s);

        // Zig
        var ze=zig_codec.Encoder.init(alloc,c.o,c.r,c.s) catch continue;
        defer ze.deinit();
        var zt:u64=0;
        for(0..c.it)|_|{
            for(sh[0..c.o])|s| ze.addOriginal(s) catch {};
            var t=try std.time.Timer.start();
            ze.encode(zb) catch {};
            zt+=t.read();
        }

        // Rust
        const re=rs_encoder_new(c.o,c.r,c.s) orelse continue;
        defer rs_encoder_free(re);
        var rt:u64=0;
        for(0..c.it)|_|{
            for(sh[0..c.o])|s| _=rs_encoder_add_original(re,s.ptr,s.len);
            var t=try std.time.Timer.start();
            _=rs_encoder_encode(re,rb.ptr);
            rt+=t.read();
        }

        const eq=std.mem.eql(u8,zb,rb);
        const zu=@as(f64,@floatFromInt(zt/c.it))/1000.0;
        const ru=@as(f64,@floatFromInt(rt/c.it))/1000.0;
        std.debug.print("{s:<30} {d:>12.1} {d:>12.1} {d:>7.2}x {s}\n",
            .{c.l,zu,ru,zu/ru,if(eq)"✓"else"✗"});

        for(sh)|s| alloc.free(s);
        alloc.free(sh); alloc.free(zb); alloc.free(rb);
    }
    std.debug.print("\n(ratio > 1 = Rust faster, < 1 = Zig faster)\n",.{});
}
```

Then build and run:

```bash
cd reed-solomon-zig

# On ARM (macOS):
zig build-exe -OReleaseFast \
  --dep codec -Mroot=bench.zig -Mcodec=src/codec.zig \
  tools/rs-ffi/target/release/librs_ffi.a -lc \
  -femit-bin=bench

# On x86 (Linux):
zig cc -c -mavx2 -mssse3 -O3 src/simd_x86.c -o /tmp/simd_x86.o
zig build-exe -OReleaseFast \
  --dep codec -Mroot=bench.zig -Mcodec=src/codec.zig \
  /tmp/simd_x86.o tools/rs-ffi/target/release/librs_ffi.a -lc -lunwind \
  -femit-bin=bench

./bench
```

> **Note on x86 Linux**: Link `-lunwind` for Rust's static library. The benchmark timer uses `std.time.Timer` (Zig 0.15). On Zig 0.16-dev, replace with `clock_gettime(CLOCK_MONOTONIC)` via `@cImport`.

## License

Apache-2.0
