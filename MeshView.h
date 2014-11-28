#ifndef __MESHVIEW_H__
#define __MESHVIEW_H__

#include <vector>

#include "trace.h"

namespace mesh3d {
    class mesh;
}

struct MeshView {
    std::vector<point> pts;
    std::vector<int> anyTet;
    std::vector<MeshElement> elems;

    MeshView(const mesh3d::mesh &);
private:
    void convertMesh(const mesh3d::mesh &);
    void setParams(const mesh3d::mesh &);
};

#endif
