// AVX2 GF(2^16) multiply for x86_64.
// Compiled with zig cc -mavx2 -mssse3 and linked into the Zig library.
// This bypasses Zig's inline asm encoder issues with vpshufb.

#include <immintrin.h>
#include <stdint.h>

// Mul128 layout: lo[4][16] + hi[4][16] = 128 bytes
// lo[0..4] are the 4 nibble tables for the low byte of the product
// hi[0..4] are the 4 nibble tables for the high byte of the product
typedef struct {
    uint8_t lo[4][16];
    uint8_t hi[4][16];
} Mul128;

// Process one 64-byte chunk: x ^= y * lut
void gf_muladd_avx2(uint8_t *x, const uint8_t *y, const Mul128 *lut) {
    // Load lookup tables into YMM registers (duplicate across 128-bit lanes)
    __m128i lo0_128 = _mm_loadu_si128((const __m128i*)lut->lo[0]);
    __m128i lo1_128 = _mm_loadu_si128((const __m128i*)lut->lo[1]);
    __m128i lo2_128 = _mm_loadu_si128((const __m128i*)lut->lo[2]);
    __m128i lo3_128 = _mm_loadu_si128((const __m128i*)lut->lo[3]);
    __m128i hi0_128 = _mm_loadu_si128((const __m128i*)lut->hi[0]);
    __m128i hi1_128 = _mm_loadu_si128((const __m128i*)lut->hi[1]);
    __m128i hi2_128 = _mm_loadu_si128((const __m128i*)lut->hi[2]);
    __m128i hi3_128 = _mm_loadu_si128((const __m128i*)lut->hi[3]);

    // Broadcast to 256-bit (duplicate each 128-bit table into both lanes)
    __m256i lo0 = _mm256_broadcastsi128_si256(lo0_128);
    __m256i lo1 = _mm256_broadcastsi128_si256(lo1_128);
    __m256i lo2 = _mm256_broadcastsi128_si256(lo2_128);
    __m256i lo3 = _mm256_broadcastsi128_si256(lo3_128);
    __m256i hi0 = _mm256_broadcastsi128_si256(hi0_128);
    __m256i hi1 = _mm256_broadcastsi128_si256(hi1_128);
    __m256i hi2 = _mm256_broadcastsi128_si256(hi2_128);
    __m256i hi3 = _mm256_broadcastsi128_si256(hi3_128);

    __m256i mask = _mm256_set1_epi8(0x0f);

    // Load data: y[0..32] = lo bytes of 32 GF elements, y[32..64] = hi bytes
    __m256i y_lo = _mm256_loadu_si256((const __m256i*)(y));
    __m256i y_hi = _mm256_loadu_si256((const __m256i*)(y + 32));

    // 4-nibble decomposition
    __m256i y_lo_lo = _mm256_and_si256(y_lo, mask);
    __m256i y_lo_hi = _mm256_and_si256(_mm256_srli_epi16(y_lo, 4), mask);
    __m256i y_hi_lo = _mm256_and_si256(y_hi, mask);
    __m256i y_hi_hi = _mm256_and_si256(_mm256_srli_epi16(y_hi, 4), mask);

    // Product low byte
    __m256i prod_lo = _mm256_shuffle_epi8(lo0, y_lo_lo);
    prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(lo1, y_lo_hi));
    prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(lo2, y_hi_lo));
    prod_lo = _mm256_xor_si256(prod_lo, _mm256_shuffle_epi8(lo3, y_hi_hi));

    // Product high byte
    __m256i prod_hi = _mm256_shuffle_epi8(hi0, y_lo_lo);
    prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(hi1, y_lo_hi));
    prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(hi2, y_hi_lo));
    prod_hi = _mm256_xor_si256(prod_hi, _mm256_shuffle_epi8(hi3, y_hi_hi));

    // XOR into x
    __m256i x_lo = _mm256_loadu_si256((const __m256i*)(x));
    __m256i x_hi = _mm256_loadu_si256((const __m256i*)(x + 32));
    _mm256_storeu_si256((__m256i*)(x), _mm256_xor_si256(x_lo, prod_lo));
    _mm256_storeu_si256((__m256i*)(x + 32), _mm256_xor_si256(x_hi, prod_hi));
}

// Fused FFT butterfly: a ^= b * lut; b ^= a
void gf_fft_butterfly_avx2(uint8_t *a, const uint8_t *b, const Mul128 *lut) {
    __m128i lo0_128 = _mm_loadu_si128((const __m128i*)lut->lo[0]);
    __m128i lo1_128 = _mm_loadu_si128((const __m128i*)lut->lo[1]);
    __m128i lo2_128 = _mm_loadu_si128((const __m128i*)lut->lo[2]);
    __m128i lo3_128 = _mm_loadu_si128((const __m128i*)lut->lo[3]);
    __m128i hi0_128 = _mm_loadu_si128((const __m128i*)lut->hi[0]);
    __m128i hi1_128 = _mm_loadu_si128((const __m128i*)lut->hi[1]);
    __m128i hi2_128 = _mm_loadu_si128((const __m128i*)lut->hi[2]);
    __m128i hi3_128 = _mm_loadu_si128((const __m128i*)lut->hi[3]);

    __m256i lo0 = _mm256_broadcastsi128_si256(lo0_128);
    __m256i lo1 = _mm256_broadcastsi128_si256(lo1_128);
    __m256i lo2 = _mm256_broadcastsi128_si256(lo2_128);
    __m256i lo3 = _mm256_broadcastsi128_si256(lo3_128);
    __m256i hi0 = _mm256_broadcastsi128_si256(hi0_128);
    __m256i hi1 = _mm256_broadcastsi128_si256(hi1_128);
    __m256i hi2 = _mm256_broadcastsi128_si256(hi2_128);
    __m256i hi3 = _mm256_broadcastsi128_si256(hi3_128);
    __m256i mask = _mm256_set1_epi8(0x0f);

    // Load b
    __m256i b_lo = _mm256_loadu_si256((const __m256i*)(b));
    __m256i b_hi = _mm256_loadu_si256((const __m256i*)(b + 32));

    // Multiply b by lut
    __m256i b_lo_lo = _mm256_and_si256(b_lo, mask);
    __m256i b_lo_hi = _mm256_and_si256(_mm256_srli_epi16(b_lo, 4), mask);
    __m256i b_hi_lo = _mm256_and_si256(b_hi, mask);
    __m256i b_hi_hi = _mm256_and_si256(_mm256_srli_epi16(b_hi, 4), mask);

    __m256i prod_lo = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_shuffle_epi8(lo0, b_lo_lo), _mm256_shuffle_epi8(lo1, b_lo_hi)),
        _mm256_xor_si256(_mm256_shuffle_epi8(lo2, b_hi_lo), _mm256_shuffle_epi8(lo3, b_hi_hi)));
    __m256i prod_hi = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_shuffle_epi8(hi0, b_lo_lo), _mm256_shuffle_epi8(hi1, b_lo_hi)),
        _mm256_xor_si256(_mm256_shuffle_epi8(hi2, b_hi_lo), _mm256_shuffle_epi8(hi3, b_hi_hi)));

    // a ^= product
    __m256i a_lo = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a)), prod_lo);
    __m256i a_hi = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(a + 32)), prod_hi);
    _mm256_storeu_si256((__m256i*)(a), a_lo);
    _mm256_storeu_si256((__m256i*)(a + 32), a_hi);

    // b ^= a (a is now the updated value, still in registers)
    _mm256_storeu_si256((__m256i*)(b), _mm256_xor_si256(b_lo, a_lo));
    _mm256_storeu_si256((__m256i*)(b + 32), _mm256_xor_si256(b_hi, a_hi));
}

// Fused IFFT butterfly: b ^= a; a ^= b * lut
void gf_ifft_butterfly_avx2(uint8_t *a, uint8_t *b, const Mul128 *lut) {
    __m128i lo0_128 = _mm_loadu_si128((const __m128i*)lut->lo[0]);
    __m128i lo1_128 = _mm_loadu_si128((const __m128i*)lut->lo[1]);
    __m128i lo2_128 = _mm_loadu_si128((const __m128i*)lut->lo[2]);
    __m128i lo3_128 = _mm_loadu_si128((const __m128i*)lut->lo[3]);
    __m128i hi0_128 = _mm_loadu_si128((const __m128i*)lut->hi[0]);
    __m128i hi1_128 = _mm_loadu_si128((const __m128i*)lut->hi[1]);
    __m128i hi2_128 = _mm_loadu_si128((const __m128i*)lut->hi[2]);
    __m128i hi3_128 = _mm_loadu_si128((const __m128i*)lut->hi[3]);

    __m256i lo0 = _mm256_broadcastsi128_si256(lo0_128);
    __m256i lo1 = _mm256_broadcastsi128_si256(lo1_128);
    __m256i lo2 = _mm256_broadcastsi128_si256(lo2_128);
    __m256i lo3 = _mm256_broadcastsi128_si256(lo3_128);
    __m256i hi0 = _mm256_broadcastsi128_si256(hi0_128);
    __m256i hi1 = _mm256_broadcastsi128_si256(hi1_128);
    __m256i hi2 = _mm256_broadcastsi128_si256(hi2_128);
    __m256i hi3 = _mm256_broadcastsi128_si256(hi3_128);
    __m256i mask = _mm256_set1_epi8(0x0f);

    // Load a and b
    __m256i a_lo = _mm256_loadu_si256((const __m256i*)(a));
    __m256i a_hi = _mm256_loadu_si256((const __m256i*)(a + 32));
    __m256i b_lo = _mm256_loadu_si256((const __m256i*)(b));
    __m256i b_hi = _mm256_loadu_si256((const __m256i*)(b + 32));

    // b ^= a
    __m256i nb_lo = _mm256_xor_si256(b_lo, a_lo);
    __m256i nb_hi = _mm256_xor_si256(b_hi, a_hi);
    _mm256_storeu_si256((__m256i*)(b), nb_lo);
    _mm256_storeu_si256((__m256i*)(b + 32), nb_hi);

    // Multiply new_b by lut
    __m256i nb_lo_lo = _mm256_and_si256(nb_lo, mask);
    __m256i nb_lo_hi = _mm256_and_si256(_mm256_srli_epi16(nb_lo, 4), mask);
    __m256i nb_hi_lo = _mm256_and_si256(nb_hi, mask);
    __m256i nb_hi_hi = _mm256_and_si256(_mm256_srli_epi16(nb_hi, 4), mask);

    __m256i prod_lo = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_shuffle_epi8(lo0, nb_lo_lo), _mm256_shuffle_epi8(lo1, nb_lo_hi)),
        _mm256_xor_si256(_mm256_shuffle_epi8(lo2, nb_hi_lo), _mm256_shuffle_epi8(lo3, nb_hi_hi)));
    __m256i prod_hi = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_shuffle_epi8(hi0, nb_lo_lo), _mm256_shuffle_epi8(hi1, nb_lo_hi)),
        _mm256_xor_si256(_mm256_shuffle_epi8(hi2, nb_hi_lo), _mm256_shuffle_epi8(hi3, nb_hi_hi)));

    // a ^= product
    _mm256_storeu_si256((__m256i*)(a), _mm256_xor_si256(a_lo, prod_lo));
    _mm256_storeu_si256((__m256i*)(a + 32), _mm256_xor_si256(a_hi, prod_hi));
}

// Multiply in-place: x *= lut
void gf_mul_avx2(uint8_t *x, const Mul128 *lut) {
    __m128i lo0_128 = _mm_loadu_si128((const __m128i*)lut->lo[0]);
    __m128i lo1_128 = _mm_loadu_si128((const __m128i*)lut->lo[1]);
    __m128i lo2_128 = _mm_loadu_si128((const __m128i*)lut->lo[2]);
    __m128i lo3_128 = _mm_loadu_si128((const __m128i*)lut->lo[3]);
    __m128i hi0_128 = _mm_loadu_si128((const __m128i*)lut->hi[0]);
    __m128i hi1_128 = _mm_loadu_si128((const __m128i*)lut->hi[1]);
    __m128i hi2_128 = _mm_loadu_si128((const __m128i*)lut->hi[2]);
    __m128i hi3_128 = _mm_loadu_si128((const __m128i*)lut->hi[3]);

    __m256i lo0 = _mm256_broadcastsi128_si256(lo0_128);
    __m256i lo1 = _mm256_broadcastsi128_si256(lo1_128);
    __m256i lo2 = _mm256_broadcastsi128_si256(lo2_128);
    __m256i lo3 = _mm256_broadcastsi128_si256(lo3_128);
    __m256i hi0 = _mm256_broadcastsi128_si256(hi0_128);
    __m256i hi1 = _mm256_broadcastsi128_si256(hi1_128);
    __m256i hi2 = _mm256_broadcastsi128_si256(hi2_128);
    __m256i hi3 = _mm256_broadcastsi128_si256(hi3_128);
    __m256i mask = _mm256_set1_epi8(0x0f);

    __m256i x_lo = _mm256_loadu_si256((const __m256i*)(x));
    __m256i x_hi = _mm256_loadu_si256((const __m256i*)(x + 32));

    __m256i x_lo_lo = _mm256_and_si256(x_lo, mask);
    __m256i x_lo_hi = _mm256_and_si256(_mm256_srli_epi16(x_lo, 4), mask);
    __m256i x_hi_lo = _mm256_and_si256(x_hi, mask);
    __m256i x_hi_hi = _mm256_and_si256(_mm256_srli_epi16(x_hi, 4), mask);

    __m256i prod_lo = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_shuffle_epi8(lo0, x_lo_lo), _mm256_shuffle_epi8(lo1, x_lo_hi)),
        _mm256_xor_si256(_mm256_shuffle_epi8(lo2, x_hi_lo), _mm256_shuffle_epi8(lo3, x_hi_hi)));
    __m256i prod_hi = _mm256_xor_si256(
        _mm256_xor_si256(_mm256_shuffle_epi8(hi0, x_lo_lo), _mm256_shuffle_epi8(hi1, x_lo_hi)),
        _mm256_xor_si256(_mm256_shuffle_epi8(hi2, x_hi_lo), _mm256_shuffle_epi8(hi3, x_hi_hi)));

    _mm256_storeu_si256((__m256i*)(x), prod_lo);
    _mm256_storeu_si256((__m256i*)(x + 32), prod_hi);
}
