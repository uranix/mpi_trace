#include "kernels.h"

__global__ void trace_kernel(const int nP, const int lo, GPUMeshViewRaw mv, real *Idirs, int *inner, point *ws) {
    int dir = lo + blockIdx.y;
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    const point w = ws[dir];

    if (i >= nP || !inner[i])
        return;

    Idirs[i + blockIdx.y * nP] = i;
}
