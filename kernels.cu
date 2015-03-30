#include "kernels.h"

#include <cstdio>

__device__ void fetch(MeshElement *dst, const MeshElement *src) {
    /* We have NFREQ threads
     * sizeof(MeshElement) = K * NFREQ * sizeof(real)
     */
    int i = threadIdx.x;
    const int K = sizeof(MeshElement) / NFREQ / sizeof(real);
    const real *_src = reinterpret_cast<const real *>(src);
    real *_dst = reinterpret_cast<real *>(dst);
    #pragma unroll
    for (int k = 0; k < K; k++)
        _dst[k * NFREQ + i] = _src[k * NFREQ + i];
}

/*
    dim3 block(NFREQ, PTSPERBLOCK);
    dim3 grid((nP + PTSPERBLOCK - 1) / PTSPERBLOCK, ndir);
 */
__global__ void trace_kernel(const int pointLo, const int nP, const int dirLo, const int dirOffs, GPUMeshViewRaw mv, real *Idirs, int *inner, point *ws) {
    __shared__ MeshElement currTets[PTSPERBLOCK];

    int dir = dirLo + dirOffs + blockIdx.y;
    int ifreq = threadIdx.x;
    int blockPoint = threadIdx.y;
    int i = pointLo + blockPoint + blockIdx.x * PTSPERBLOCK;

    const point w = ws[dir];
    real *Idir = Idirs + (dirOffs + blockIdx.y) * nP * NFREQ;

    bool idle = i >= nP || !inner[i];

    int itet = idle ? -1 : mv.anyTet[i];
    point pw;
    point r(idle ? point() : mv.pts[i]);
    real a = 1, b = 0;
    int vout[3];

    do {
        int face;
        if (!idle)
            fetch(currTets + blockPoint, mv.elems + itet);
        __syncthreads();
        if (!idle) {
            const MeshElement &currTet = currTets[blockPoint];

            real len = trace(w, currTet, r, face, mv.pts);

            real delta = len * currTet.kappa[ifreq];
            real q = exp(-delta);
            b += a * currTet.Ip[ifreq] * (1 - q);
            a *= q;
            itet = currTet.neib[face];
            if (itet == NO_TET) {
                for (int j = 0; j < 3; j++)
                    vout[j] = currTet.p[(face + 1 + j) & 3];
                pw = bary(r, mv.pts[vout[0]], mv.pts[vout[1]], mv.pts[vout[2]]);
                real I0 = Idir[vout[0] * NFREQ + ifreq];
                real I1 = Idir[vout[1] * NFREQ + ifreq];
                real I2 = Idir[vout[2] * NFREQ + ifreq];
                real out = b + a * (
                        pw.x * I0 +
                        pw.y * I1 +
                        pw.z * I2
                    );
                Idir[i * NFREQ + ifreq] = out;
                idle = true;
            }
        }
    } while (!__syncthreads_and(idle));
}
