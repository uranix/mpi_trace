#ifndef __GPU_H__
#define __GPU_H__

#include "trace.h"

#include <vector>

struct MeshView;

struct GPUMeshViewRaw {
    int nP;
    point *pts;
    int   *anyTet;
    MeshElement *elems;
};

struct GPUMeshView : public GPUMeshViewRaw {
    GPUMeshView(int rank, int device, MeshView &mv);
    ~GPUMeshView();
};

struct GPUAverageSolution {
    int nP;
    std::vector<real> U;
    real *Udev;

    GPUAverageSolution(const GPUMeshView &gmv);
    ~GPUAverageSolution();

    void add(real *Idir, const real wei);

    const std::vector<real> &retrieve();
};

struct GPUMultipleDirectionSolver {
    const int maxDirections;
    const GPUMeshViewRaw mv;
    real *Idirs;
    int *inner;
    point *w;

    GPUMultipleDirectionSolver(const int maxDirections, const GPUMeshView &mv, const std::vector<point> &ws);
    ~GPUMultipleDirectionSolver();

    real *Idir(const int direction);
    int *innerFlag(const int direction);

    void setBoundary(const int direction, std::vector<real> &Ihostdir, std::vector<int> &isInner);
    void traceInterior(const int startDirection, const int offsDirection, const int dirs);
};

bool alignment_test();

#endif
