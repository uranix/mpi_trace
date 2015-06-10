#ifndef __MESHVIEW_H__
#define __MESHVIEW_H__

#include <vector>

#include "trace.h"

namespace mesh3d {
    class mesh;
}

struct MHDdata;

struct MeshView {
    std::vector<point> pts;
    std::vector<int> anyTet;
    std::vector<MeshElement> elems;
    const MHDdata &mhd;

    MeshView(const mesh3d::mesh &, const MHDdata &mhd);
private:
    void convertMesh(const mesh3d::mesh &);
    void setParams(const mesh3d::mesh &);
};

#endif
