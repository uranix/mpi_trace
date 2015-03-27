#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "trace.h"
#include "gpu.h"

__global__ void trace_kernel(const int nP, const int lo, const int offs, GPUMeshViewRaw mv, real *Idirs, int *inner, point *ws);

#endif
