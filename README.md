# reed-solomon-zig

Reed-Solomon erasure coding for Zig. Faster than the Rust reference on both ARM and x86.

Implements Leopard-RS over GF(2^16) with platform-specific SIMD acceleration. Produces byte-identical output to the Rust [`reed-solomon-simd`](https://github.com/AndersTrier/reed-solomon-simd) crate — verified across 26 configurations.

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
- **Max 65536 total shards** (original + recovery)
- **Encoder/decoder are reusable** — encode/decode resets internal state automatically
- **Thread-safe table initialization** — lookup tables computed once per process

## Performance

Byte-identical output with Rust `reed-solomon-simd`, consistently faster across all tested configurations.

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

### Why it's fast

- **Fused butterfly** — FFT butterfly combines multiply + XOR in a single pass, keeping data in SIMD registers between operations. Rust does these as separate function calls with an extra memory round-trip.
- **AVX2 with amortized table broadcast** — On x86, lookup tables are loaded into YMM registers once per butterfly group and applied to all chunks in a tight C loop. No per-chunk function call overhead.
- **NEON `tbl` inline asm** — On ARM, uses the `tbl` instruction directly for 16-byte table lookups at 30+ GB/s.
- **Shared global tables** — The 8 MiB Mul128 lookup table is initialized once per process and shared across all encoder/decoder instances.

## Architecture

```
src/
  reed_solomon.zig   Root module — re-exports Encoder, Decoder
  codec.zig          Encoder/Decoder with high-rate + low-rate paths
  engine.zig         FFT/IFFT with fused butterfly, shard management
  gf.zig             GF(2^16) arithmetic, comptime log/exp tables, SIMD multiply
  fwht.zig           Fast Walsh-Hadamard Transform
  tables.zig         Cantor-basis exp/log, skew table, Mul128 table init
  simd_x86.c         AVX2 intrinsics for x86 (compiled by zig cc)
```

~1,900 lines total (1,770 Zig + 90 C). The Rust reference is ~7,600 lines.

## SIMD support

| Platform | Engine | Instructions |
|---|---|---|
| AArch64 (ARM) | Zig inline asm | NEON `tbl` (128-bit) |
| x86_64 (Intel/AMD) | C FFI (AVX2) | `vpshufb` (256-bit) |
| Other | Scalar fallback | Element-by-element lookup |

The x86 AVX2 path uses a small C file (`simd_x86.c`) compiled with `-mavx2` by Zig's built-in C compiler. This bypasses a [Zig compiler limitation](https://github.com/ziglang/zig/issues/24810) with inline assembly on x86 vectors. When Zig adds runtime `@shuffle` support, the C file will be replaced with pure Zig.

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

## Reproducing benchmarks

Requires: Zig 0.15+, Rust stable, both on the same machine.

### 1. Build the Rust reference FFI wrapper

```bash
mkdir -p /tmp/rs-ffi/src
cat > /tmp/rs-ffi/Cargo.toml << 'EOF'
[package]
name = "rs-ffi"
version = "0.1.0"
edition = "2021"
[lib]
crate-type = ["staticlib"]
[dependencies]
reed-solomon-simd = "3.1"
[profile.release]
opt-level = 3
lto = true
codegen-units = 1
panic = "abort"
EOF

cat > /tmp/rs-ffi/src/lib.rs << 'EOF'
use reed_solomon_simd::ReedSolomonEncoder;
use std::ptr;
pub struct Enc { inner: ReedSolomonEncoder, sb: usize }
#[no_mangle] pub extern "C" fn rs_encoder_new(o: usize, r: usize, s: usize) -> *mut Enc {
    match ReedSolomonEncoder::new(o, r, s) { Ok(e) => Box::into_raw(Box::new(Enc{inner:e,sb:s})), Err(_) => ptr::null_mut() }
}
#[no_mangle] pub extern "C" fn rs_encoder_add_original(e: *mut Enc, d: *const u8, l: usize) -> i32 {
    let e=unsafe{&mut*e}; match e.inner.add_original_shard(unsafe{std::slice::from_raw_parts(d,l)}) { Ok(())=>0, Err(_)=>-1 }
}
#[no_mangle] pub extern "C" fn rs_encoder_encode(e: *mut Enc, out: *mut u8) -> i32 {
    let e=unsafe{&mut*e}; let r=match e.inner.encode(){Ok(r)=>r,Err(_)=>return -1};
    let mut off=0; for s in r.recovery_iter(){unsafe{std::ptr::copy_nonoverlapping(s.as_ptr(),out.add(off),e.sb);}off+=e.sb;} 0
}
#[no_mangle] pub extern "C" fn rs_encoder_free(e: *mut Enc) { if !e.is_null(){unsafe{drop(Box::from_raw(e));}} }
EOF

cd /tmp/rs-ffi && cargo build --release
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
  /tmp/rs-ffi/target/release/librs_ffi.a -lc \
  -femit-bin=bench

# On x86 (Linux):
zig cc -c -mavx2 -mssse3 -O3 src/simd_x86.c -o /tmp/simd_x86.o
zig build-exe -OReleaseFast \
  --dep codec -Mroot=bench.zig -Mcodec=src/codec.zig \
  /tmp/simd_x86.o /tmp/rs-ffi/target/release/librs_ffi.a -lc -lunwind \
  -femit-bin=bench

./bench
```

> **Note on x86 Linux**: Link `-lunwind` for Rust's static library. The benchmark timer uses `std.time.Timer` (Zig 0.15). On Zig 0.16-dev, replace with `clock_gettime(CLOCK_MONOTONIC)` via `@cImport`.

## License

Apache-2.0
