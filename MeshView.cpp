#include "MeshView.h"

#include <meshProcessor/mesh.h>

using namespace mesh3d;

MeshView::MeshView(const mesh &m) {
    convertMesh(m);
    setParams(m);
}

void MeshView::convertMesh(const mesh &m) {
    int nP = m.vertices().size();
    int nT = m.tets().size();

    pts.resize(nP);
    tets.resize(nT);
    anyTet.resize(nP);

    for (int i = 0; i < nT; i++) {
        const tetrahedron &tet = m.tets(i);
        for (int j = 0; j < 4; j++) {
            tets[i].p[j] = tet.p(j).idx();
            const face &f = tet.f(j).flip();
            if (f.is_border())
                tets[i].neib[j] = NO_TET;
            else
                tets[i].neib[j] = f.tet().idx();
        }
    }

    for (int i = 0; i < nP; i++) {
        const vertex &v = m.vertices(i);
        pts[i].x = v.r().x;
        pts[i].y = v.r().y;
        pts[i].z = v.r().z;
        anyTet[i] = v.tetrahedrons().front().t->idx();
    }
}

void MeshView::setParams(const mesh &m) {
    int nT = m.tets().size();
    kappa.resize(nT);
    Ip   .resize(nT);

    for (int i = 0; i < nT; i++) {
        const tetrahedron &tet = m.tets(i);
        if (tet.color() == 1) {
            kappa[i] = 11;
            Ip[i] = 1;
        } else {
            kappa[i] = 3;
            Ip[i] = 0;
        }
    }
}
