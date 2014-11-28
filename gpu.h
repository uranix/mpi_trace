#ifndef __GPU_H__
#define __GPU_H__

#include "trace.h"

#include <vector>

struct MeshView;

struct GPUMeshViewRaw {
    int nP;
    point *pts;
    int   *anyTet;
    tet   *tets;
    real  *kappa;
    real  *Ip;
};

struct GPUMeshView : public GPUMeshViewRaw {
    GPUMeshView(MeshView &mv);
    ~GPUMeshView();
};

struct GPUAverageSolution {
    int nP;
    std::vector<double> U;
    real *Udev;

    GPUAverageSolution(const MeshView &mv);
    ~GPUAverageSolution();

    template<typename R>
    void add(R *Idir, const R wei);

    std::vector<double> &retrieve();
};

struct GPUMultipleDirectionSolver {
    const int maxDirections;
    const GPUMeshView &mv;
    real *Idirs;
    GPUMultipleDirectionSolver(const int maxDirections, const GPUMeshView &mv);
    ~GPUMultipleDirectionSolver();
    real *Idir(const int direction);
};

#endif
