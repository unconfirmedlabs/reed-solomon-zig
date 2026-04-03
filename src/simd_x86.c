// AVX2 GF(2^16) multiply for x86_64.
// Compiled with zig cc -mavx2 -mssse3 and linked into the Zig library.
// Processes N chunks per call to amortize table broadcast overhead.

#include <immintrin.h>
#include <stdint.h>
#include <stddef.h>

typedef struct {
    uint8_t lo[4][16];
    uint8_t hi[4][16];
} Mul128;

// Load and broadcast lookup tables (shared across all chunk-processing functions)
#define LOAD_TABLES(lut) \
    __m256i lo0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->lo[0])); \
    __m256i lo1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->lo[1])); \
    __m256i lo2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->lo[2])); \
    __m256i lo3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->lo[3])); \
    __m256i hi0 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->hi[0])); \
    __m256i hi1 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->hi[1])); \
    __m256i hi2 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->hi[2])); \
    __m256i hi3 = _mm256_broadcastsi128_si256(_mm_loadu_si128((const __m128i*)(lut)->hi[3])); \
    __m256i mask = _mm256_set1_epi8(0x0f);

// Multiply: compute product of data by preloaded tables
#define MUL_BODY(d_lo, d_hi, out_lo, out_hi) { \
    __m256i _dl = _mm256_and_si256(d_lo, mask); \
    __m256i _dh = _mm256_and_si256(_mm256_srli_epi16(d_lo, 4), mask); \
    __m256i _el = _mm256_and_si256(d_hi, mask); \
    __m256i _eh = _mm256_and_si256(_mm256_srli_epi16(d_hi, 4), mask); \
    out_lo = _mm256_xor_si256( \
        _mm256_xor_si256(_mm256_shuffle_epi8(lo0, _dl), _mm256_shuffle_epi8(lo1, _dh)), \
        _mm256_xor_si256(_mm256_shuffle_epi8(lo2, _el), _mm256_shuffle_epi8(lo3, _eh))); \
    out_hi = _mm256_xor_si256( \
        _mm256_xor_si256(_mm256_shuffle_epi8(hi0, _dl), _mm256_shuffle_epi8(hi1, _dh)), \
        _mm256_xor_si256(_mm256_shuffle_epi8(hi2, _el), _mm256_shuffle_epi8(hi3, _eh))); \
}

// Fused FFT butterfly over N chunks: a[i] ^= b[i] * lut; b[i] ^= a[i]
// Tables loaded ONCE, applied to all N chunks.
void gf_fft_butterfly_avx2(uint8_t *a, uint8_t *b, const Mul128 *lut, size_t n_chunks) {
    LOAD_TABLES(lut);
    for (size_t i = 0; i < n_chunks; i++) {
        uint8_t *ap = a + i * 64;
        uint8_t *bp = b + i * 64;
        __m256i b_lo = _mm256_loadu_si256((const __m256i*)bp);
        __m256i b_hi = _mm256_loadu_si256((const __m256i*)(bp + 32));
        __m256i prod_lo, prod_hi;
        MUL_BODY(b_lo, b_hi, prod_lo, prod_hi);
        __m256i a_lo = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)ap), prod_lo);
        __m256i a_hi = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(ap + 32)), prod_hi);
        _mm256_storeu_si256((__m256i*)ap, a_lo);
        _mm256_storeu_si256((__m256i*)(ap + 32), a_hi);
        _mm256_storeu_si256((__m256i*)bp, _mm256_xor_si256(b_lo, a_lo));
        _mm256_storeu_si256((__m256i*)(bp + 32), _mm256_xor_si256(b_hi, a_hi));
    }
}

// Fused IFFT butterfly over N chunks: b[i] ^= a[i]; a[i] ^= b[i] * lut
void gf_ifft_butterfly_avx2(uint8_t *a, uint8_t *b, const Mul128 *lut, size_t n_chunks) {
    LOAD_TABLES(lut);
    for (size_t i = 0; i < n_chunks; i++) {
        uint8_t *ap = a + i * 64;
        uint8_t *bp = b + i * 64;
        __m256i a_lo = _mm256_loadu_si256((const __m256i*)ap);
        __m256i a_hi = _mm256_loadu_si256((const __m256i*)(ap + 32));
        __m256i nb_lo = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)bp), a_lo);
        __m256i nb_hi = _mm256_xor_si256(_mm256_loadu_si256((const __m256i*)(bp + 32)), a_hi);
        _mm256_storeu_si256((__m256i*)bp, nb_lo);
        _mm256_storeu_si256((__m256i*)(bp + 32), nb_hi);
        __m256i prod_lo, prod_hi;
        MUL_BODY(nb_lo, nb_hi, prod_lo, prod_hi);
        _mm256_storeu_si256((__m256i*)ap, _mm256_xor_si256(a_lo, prod_lo));
        _mm256_storeu_si256((__m256i*)(ap + 32), _mm256_xor_si256(a_hi, prod_hi));
    }
}

// Multiply N chunks in-place: x[i] *= lut
void gf_mul_avx2(uint8_t *x, const Mul128 *lut, size_t n_chunks) {
    LOAD_TABLES(lut);
    for (size_t i = 0; i < n_chunks; i++) {
        uint8_t *xp = x + i * 64;
        __m256i x_lo = _mm256_loadu_si256((const __m256i*)xp);
        __m256i x_hi = _mm256_loadu_si256((const __m256i*)(xp + 32));
        __m256i prod_lo, prod_hi;
        MUL_BODY(x_lo, x_hi, prod_lo, prod_hi);
        _mm256_storeu_si256((__m256i*)xp, prod_lo);
        _mm256_storeu_si256((__m256i*)(xp + 32), prod_hi);
    }
}
