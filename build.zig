const std = @import("std");

pub fn build(b: *std.Build) void {
    const target = b.standardTargetOptions(.{});
    const optimize = b.standardOptimizeOption(.{});

    const mod = b.addModule("reed_solomon", .{
        .root_source_file = b.path("src/reed_solomon.zig"),
        .target = target,
        .optimize = optimize,
    });

    // On x86 with AVX2, compile the C SIMD implementation
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

    const tests = b.addTest(.{ .root_module = mod });
    const run_tests = b.addRunArtifact(tests);
    const test_step = b.step("test", "Run Reed-Solomon tests");
    test_step.dependOn(&run_tests.step);
}
