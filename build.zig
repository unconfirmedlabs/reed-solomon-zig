const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("reed_solomon", .{
        .root_source_file = b.path("src/reed_solomon.zig"),
        .target = target,
        .optimize = optimize,
    });

    // On x86, compile AVX2 C SIMD implementation
    const resolved = target.result;
    if (resolved.cpu.arch == .x86_64 or resolved.cpu.arch == .x86) {
        mod.addCSourceFile(.{
            .file = b.path("src/simd_x86.c"),
            .flags = &.{ "-mavx2", "-mssse3", "-O3" },
        });
        mod.link_libc = true;
    }

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Reed-Solomon tests");
    test_step.dependOn(&run_tests.step);
}
