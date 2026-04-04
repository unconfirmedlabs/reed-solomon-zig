const std = @import("std");

fn configureModule(b: *std.Build, mod: *std.Build.Module, target: std.Build.ResolvedTarget) void {
    const resolved = target.result;
    const is_x86 = resolved.cpu.arch == .x86_64 or resolved.cpu.arch == .x86;
    const has_avx2 = is_x86 and std.Target.x86.featureSetHas(resolved.cpu.features, .avx2);

    if (has_avx2) {
        mod.addCSourceFile(.{
            .file = b.path("src/simd_x86.c"),
            .flags = &.{ "-mavx2", "-mssse3", "-O3" },
        });
        mod.link_libc = true;
    }
}

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});
    const diff_iters = b.option(usize, "diff-iters", "Number of randomized differential fuzz cases to run") orelse 10_000;
    const diff_seed = b.option(u64, "diff-seed", "PRNG seed for differential fuzzing") orelse 0x0ddc_0ffe_e123_4567;

    const mod = b.addModule("reed_solomon", .{
        .root_source_file = b.path("src/reed_solomon.zig"),
        .target = target,
        .optimize = optimize,
    });
    configureModule(b, mod, target);

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Reed-Solomon tests");
    test_step.dependOn(&run_tests.step);

    const cargo_build = b.addSystemCommand(&.{
        "cargo",
        "build",
        "--release",
        "--locked",
        "--offline",
        "--manifest-path",
    });
    cargo_build.addFileArg(b.path("tools/rs-ffi/Cargo.toml"));

    const diff_fuzz_root = b.createModule(.{
        .root_source_file = b.path("tools/diff_decode_fuzz.zig"),
        .target = target,
        .optimize = optimize,
    });
    diff_fuzz_root.addImport("reed_solomon", mod);

    const diff_fuzz = b.addExecutable(.{
        .name = "diff_decode_fuzz",
        .root_module = diff_fuzz_root,
    });
    diff_fuzz.linkLibC();
    if (target.result.os.tag == .linux) {
        diff_fuzz.linkSystemLibrary("unwind");
    }
    diff_fuzz.addObjectFile(b.path("tools/rs-ffi/target/release/librs_ffi.a"));
    diff_fuzz.step.dependOn(&cargo_build.step);

    const run_diff_fuzz = b.addRunArtifact(diff_fuzz);
    run_diff_fuzz.addArg(b.fmt("{d}", .{diff_iters}));
    run_diff_fuzz.addArg(b.fmt("{d}", .{diff_seed}));
    run_diff_fuzz.step.dependOn(&cargo_build.step);

    const diff_fuzz_step = b.step("diff-fuzz", "Run differential fuzzing against vendored reed-solomon-simd");
    diff_fuzz_step.dependOn(&run_diff_fuzz.step);
}
