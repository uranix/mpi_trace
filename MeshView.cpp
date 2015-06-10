#include "MeshView.h"
#include "mhd.h"

#include <meshProcessor/mesh.h>

using namespace mesh3d;

MeshView::MeshView(const mesh &m, const MHDdata &mhd) : mhd(mhd) {
    convertMesh(m);
    setParams(m);
}

void MeshView::convertMesh(const mesh &m) {
    int nP = m.vertices().size();
    int nT = m.tets().size();

    pts.resize(nP);
    anyTet.resize(nP);

    elems.resize(nT);

    for (int i = 0; i < nT; i++) {
        const tetrahedron &tet = m.tets(i);
        for (int j = 0; j < 4; j++) {
            elems[i].p[j] = tet.p(j).idx();
            const face &f = tet.f(j).flip();
            if (f.is_border())
                elems[i].neib[j] = NO_TET;
            else
                elems[i].neib[j] = f.tet().idx();
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

    for (int i = 0; i < nT; i++) {
        const tetrahedron &tet = m.tets(i);

        elems[i].Te = 1;
        elems[i].dvstep = 6.f / NFREQ;

        if (tet.color() == 1) {
            elems[i].kappa0 = 1000;
            elems[i].Ip0 = 1;
            for (int i = 0; i < 3; i++)
                elems[i].v[i] = 0;
            elems[i].Teff = 100;
        } else {
            const auto p = tet.center();
            const auto &pos = mhd.getCoordAndSide(p.x, p.y, p.z);

            elems[i].kappa0 = mhd.kappa(pos);
            elems[i].Ip0 = 0;
            for (int i = 0; i < 3; i++)
                elems[i].v[i] = mhd.velocity(pos, i);
            elems[i].Teff = mhd.Teff(pos);
        }
    }
}
