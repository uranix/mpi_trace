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
    GPUMeshView(int device, MeshView &mv);
    ~GPUMeshView();
};

struct GPUAverageSolution {
    int nP;
    std::vector<double> U;
    real *Udev;

    GPUAverageSolution(const GPUMeshView &gmv);
    ~GPUAverageSolution();

    template<typename R>
    void add(R *Idir, const R wei);

    std::vector<double> &retrieve();
};

struct GPUMultipleDirectionSolver {
    const int maxDirections;
    const GPUMeshView &mv;
    real *Idirs;
    int *inner;
    point *w;
    GPUMultipleDirectionSolver(const int maxDirections, const GPUMeshView &mv, const std::vector<point> &ws);
    ~GPUMultipleDirectionSolver();
    void setBoundary(const int direction, std::vector<real> &Ihostdir, std::vector<int> &isInner);
    real *Idir(const int direction);
    int *innerFlag(const int direction);
};

#endif
