#ifndef __MESHVIEW_H__
#define __MESHVIEW_H__

#include <vector>

#include "trace.h"

namespace mesh3d {
    class mesh;
}

struct MeshView {
    std::vector<point> pts;
    std::vector<tet> tets;
    std::vector<real> kappa;
    std::vector<real> Ip;
    std::vector<int> anyTet;

    MeshView(const mesh3d::mesh &);
private:
    void convertMesh(const mesh3d::mesh &);
    void setParams(const mesh3d::mesh &);
};

#endif
