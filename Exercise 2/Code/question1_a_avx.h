#pragma once

#include <immintrin.h>
#include <math.h>
#include <float.h>

#ifndef WENOEPS
#define WENOEPS 1.e-6f
#endif

static inline void weno_avx(const float * __restrict a,
                          const float * __restrict b,
                          const float * __restrict c,
                          const float * __restrict d,
                          const float * __restrict e,
                          float * __restrict out,
                          const int NENTRIES)
{
    // Pre-compute constants
    const __m256 v_4o3  = _mm256_set1_ps(4.0f/3.0f);
    const __m256 v_19o3 = _mm256_set1_ps(19.0f/3.0f);
    const __m256 v_11o3 = _mm256_set1_ps(11.0f/3.0f);
    const __m256 v_25o3 = _mm256_set1_ps(25.0f/3.0f);
    const __m256 v_31o3 = _mm256_set1_ps(31.0f/3.0f);
    const __m256 v_10o3 = _mm256_set1_ps(10.0f/3.0f);
    const __m256 v_13o3 = _mm256_set1_ps(13.0f/3.0f);
    const __m256 v_5o3  = _mm256_set1_ps(5.0f/3.0f);
    const __m256 v_eps  = _mm256_set1_ps(WENOEPS);
    const __m256 v_one  = _mm256_set1_ps(1.0f);
    const __m256 v_alpha0 = _mm256_set1_ps(0.1f);
    const __m256 v_alpha1 = _mm256_set1_ps(0.6f);
    const __m256 v_alpha2 = _mm256_set1_ps(0.3f);

    #pragma omp parallel for
    for (int i = 0; i < NENTRIES; i += 8) {
        // Load all data at once to improve cache locality
        __m256 va = _mm256_loadu_ps(&a[i]);
        __m256 vb = _mm256_loadu_ps(&b[i]);
        __m256 vc = _mm256_loadu_ps(&c[i]);
        __m256 vd = _mm256_loadu_ps(&d[i]);
        __m256 ve = _mm256_loadu_ps(&e[i]);

        // Compute is0 coefficients
        __m256 is0 = _mm256_mul_ps(va, _mm256_mul_ps(va, v_4o3));
        is0 = _mm256_sub_ps(is0, _mm256_mul_ps(va, _mm256_mul_ps(vb, v_19o3)));
        is0 = _mm256_add_ps(is0, _mm256_mul_ps(va, _mm256_mul_ps(vc, v_11o3)));
        __m256 tmp = _mm256_mul_ps(vb, _mm256_mul_ps(vb, v_25o3));
        tmp = _mm256_sub_ps(tmp, _mm256_mul_ps(vb, _mm256_mul_ps(vc, v_31o3)));
        is0 = _mm256_add_ps(is0, tmp);
        is0 = _mm256_add_ps(is0, _mm256_mul_ps(_mm256_mul_ps(vc, vc), v_10o3));

        // Compute is1 coefficients
        __m256 is1 = _mm256_mul_ps(vb, _mm256_mul_ps(vb, v_4o3));
        is1 = _mm256_sub_ps(is1, _mm256_mul_ps(vb, _mm256_mul_ps(vc, v_13o3)));
        is1 = _mm256_add_ps(is1, _mm256_mul_ps(vb, _mm256_mul_ps(vd, v_5o3)));
        tmp = _mm256_mul_ps(vc, _mm256_mul_ps(vc, v_13o3));
        tmp = _mm256_sub_ps(tmp, _mm256_mul_ps(vc, _mm256_mul_ps(vd, v_13o3)));
        is1 = _mm256_add_ps(is1, tmp);
        is1 = _mm256_add_ps(is1, _mm256_mul_ps(_mm256_mul_ps(vd, vd), v_4o3));

        // Compute is2 coefficients
        __m256 is2 = _mm256_mul_ps(vc, _mm256_mul_ps(vc, v_10o3));
        is2 = _mm256_sub_ps(is2, _mm256_mul_ps(vc, _mm256_mul_ps(vd, v_31o3)));
        is2 = _mm256_add_ps(is2, _mm256_mul_ps(vc, _mm256_mul_ps(ve, v_11o3)));
        tmp = _mm256_mul_ps(vd, _mm256_mul_ps(vd, v_25o3));
        tmp = _mm256_sub_ps(tmp, _mm256_mul_ps(vd, _mm256_mul_ps(ve, v_19o3)));
        is2 = _mm256_add_ps(is2, tmp);
        is2 = _mm256_add_ps(is2, _mm256_mul_ps(_mm256_mul_ps(ve, ve), v_4o3));

        // Add epsilon and compute squares
        is0 = _mm256_add_ps(is0, v_eps);
        is1 = _mm256_add_ps(is1, v_eps);
        is2 = _mm256_add_ps(is2, v_eps);
        is0 = _mm256_mul_ps(is0, is0);
        is1 = _mm256_mul_ps(is1, is1);
        is2 = _mm256_mul_ps(is2, is2);

        // Compute alphas directly
        is0 = _mm256_div_ps(v_alpha0, is0);
        is1 = _mm256_div_ps(v_alpha1, is1);
        is2 = _mm256_div_ps(v_alpha2, is2);

        // Compute weights
        __m256 alpha_sum = _mm256_add_ps(_mm256_add_ps(is0, is1), is2);
        __m256 inv_sum = _mm256_div_ps(v_one, alpha_sum);

        __m256 omega0 = _mm256_mul_ps(is0, inv_sum);
        __m256 omega1 = _mm256_mul_ps(is1, inv_sum);
        __m256 omega2 = _mm256_sub_ps(v_one, _mm256_add_ps(omega0, omega1));

        // Final reconstruction coefficients
        __m256 u0 = _mm256_mul_ps(_mm256_set1_ps(1.0f/3.0f), va);
        u0 = _mm256_sub_ps(u0, _mm256_mul_ps(_mm256_set1_ps(7.0f/6.0f), vb));
        u0 = _mm256_add_ps(u0, _mm256_mul_ps(_mm256_set1_ps(11.0f/6.0f), vc));

        __m256 u1 = _mm256_mul_ps(_mm256_set1_ps(-1.0f/6.0f), vb);
        u1 = _mm256_add_ps(u1, _mm256_mul_ps(_mm256_set1_ps(5.0f/6.0f), vc));
        u1 = _mm256_add_ps(u1, _mm256_mul_ps(_mm256_set1_ps(1.0f/3.0f), vd));

        __m256 u2 = _mm256_mul_ps(_mm256_set1_ps(1.0f/3.0f), vc);
        u2 = _mm256_add_ps(u2, _mm256_mul_ps(_mm256_set1_ps(5.0f/6.0f), vd));
        u2 = _mm256_sub_ps(u2, _mm256_mul_ps(_mm256_set1_ps(1.0f/6.0f), ve));

        // Combine results
        __m256 result = _mm256_mul_ps(omega0, u0);
        result = _mm256_add_ps(result, _mm256_mul_ps(omega1, u1));
        result = _mm256_add_ps(result, _mm256_mul_ps(omega2, u2));

        _mm256_storeu_ps(&out[i], result);
    }
}