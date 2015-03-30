#ifndef __KERNELS_H__
#define __KERNELS_H__

#include "trace.h"
#include "gpu.h"

__global__ void trace_kernel(const int pointLo, const int nP, const int dirLo, const int dirOffs, GPUMeshViewRaw mv, real *Idirs, int *inner, point *ws);

#endif
