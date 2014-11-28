#include "kernels.h"

__global__ void trace_kernel(const int nP, const int lo, GPUMeshViewRaw mv, real *Idirs, int *inner, point *ws) {
    int dir = lo + blockIdx.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    const point w = ws[dir];
    real *Idir = Idirs + blockIdx.y * nP;

    if (i >= nP || !inner[i])
        return;

    int itet = mv.anyTet[i];
    point r(mv.pts[i]);
    real a = 1, b = 0;
    int vout[3];
    point pw;

    while (true) {
        int face;
        MeshElement currTet = mv.elems[itet];
        real len = trace(w, currTet, r, face, mv.pts);
        real delta = len * currTet.kappa;
        real q = exp(-delta);
        b += a * currTet.Ip * (1 - q);
        a *= q;
        itet = currTet.neib[face];
        if (itet == NO_TET) {
            for (int j = 0; j < 3; j++)
                vout[j] = currTet.p[(face + 1 + j) & 3];
            pw = bary(r, mv.pts[vout[0]], mv.pts[vout[1]], mv.pts[vout[2]]);
            Idir[i] = b + a * (pw.x * Idir[vout[0]] + pw.y * Idir[vout[1]] + pw.z * Idir[vout[2]]);
            return;
        }
    }
}
