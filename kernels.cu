#include "kernels.h"

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
    __syncthreads();
}

/*
    dim3 block(NFREQ, PTSPERBLOCK);
    dim3 grid((nP + PTSPERBLOCK - 1) / PTSPERBLOCK, ndir);
 */
__global__ void trace_kernel(const int nP, const int lo, const int offs, GPUMeshViewRaw mv, real *Idirs, int *inner, point *ws) {
    __shared__ MeshElement currTets[PTSPERBLOCK];

    int dir = lo + offs + blockIdx.y;
    int ifreq = threadIdx.x;
    int blockPoint = threadIdx.y;
    int i = blockPoint + blockIdx.x * blockDim.x;

    const point w = ws[dir];
    real *Idir = Idirs + (offs + blockIdx.y) * nP * NFREQ;

    if (i >= nP || !inner[i])
        return;

    int itet = mv.anyTet[i];
    point r(mv.pts[i]);
    real a = 1, b = 0;
    int vout[3];
    point pw;

    while (true) {
        int face;
        fetch(currTets + blockPoint, mv.elems + itet);
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
            Idir[i * NFREQ + ifreq] = b + a * (
                    pw.x * Idir[vout[0] * NFREQ + ifreq] +
                    pw.y * Idir[vout[1] * NFREQ + ifreq] +
                    pw.z * Idir[vout[2] * NFREQ + ifreq]
                );
            return;
        }
    }
}
