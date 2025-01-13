#pragma once

#include <omp.h>
#include "weno.h"

void weno_minus_openmp(const float * const a, const float * const b, const float * const c,
                   const float * const d, const float * const e, float * const out,
                   const int NENTRIES)
{
    #pragma omp parallel for simd
    for(int i = 0; i < NENTRIES; ++i) {
        const float is0 = a[i]*(a[i]*(float)(4./3.)  - b[i]*(float)(19./3.)  + c[i]*(float)(11./3.)) + 
                         b[i]*(b[i]*(float)(25./3.)  - c[i]*(float)(31./3.)) + 
                         c[i]*c[i]*(float)(10./3.);

        const float is1 = b[i]*(b[i]*(float)(4./3.)  - c[i]*(float)(13./3.)  + d[i]*(float)(5./3.))  + 
                         c[i]*(c[i]*(float)(13./3.)  - d[i]*(float)(13./3.)) + 
                         d[i]*d[i]*(float)(4./3.);

        const float is2 = c[i]*(c[i]*(float)(10./3.) - d[i]*(float)(31./3.)  + e[i]*(float)(11./3.)) + 
                         d[i]*(d[i]*(float)(25./3.)  - e[i]*(float)(19./3.)) + 
                         e[i]*e[i]*(float)(4./3.);

        const float is0plus = is0 + (float)WENOEPS;
        const float is1plus = is1 + (float)WENOEPS;
        const float is2plus = is2 + (float)WENOEPS;

        const float alpha0 = (float)(0.1)/((float)(is0plus*is0plus));
        const float alpha1 = (float)(0.6)/((float)(is1plus*is1plus));
        const float alpha2 = (float)(0.3)/((float)(is2plus*is2plus));

        const float alphasum = alpha0 + alpha1 + alpha2;
        const float inv_alpha = (float)1/alphasum;

        const float omega0 = alpha0 * inv_alpha;
        const float omega1 = alpha1 * inv_alpha;
        const float omega2 = (float)1 - omega0 - omega1;

        out[i] = omega0*((float)(1.0/3.)*a[i] - (float)(7./6.)*b[i] + (float)(11./6.)*c[i]) +
                 omega1*(-(float)(1./6.)*b[i] + (float)(5./6.)*c[i] + (float)(1./3.)*d[i]) +
                 omega2*((float)(1./3.)*c[i]  + (float)(5./6.)*d[i] - (float)(1./6.)*e[i]);
    }
}