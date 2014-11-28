#ifndef __GPU_H__
#define __GPU_H__

#include "trace.h"

#include <vector>

struct MeshView;

struct GPUMeshView {
    int nP;
    point *pts;
    int   *anyTet;
    tet   *tets;
    real  *kappa;
    real  *Ip;
    GPUMeshView(MeshView &mv);
    ~GPUMeshView();
};

struct GPUAverageSolution {
    int nP;
    std::vector<double> U;
    real *Udev;

    GPUAverageSolution(const MeshView &mv);
    ~GPUAverageSolution();

    void add(real *Idir, const real wei);

    std::vector<double> &retrieve();
};

#endif
