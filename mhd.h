#ifndef __INTERFACE_H__
#define __INTERFACE_H__

#include "array.h"
#include <string>

struct coordAndSide {
	int side, i, j, k;
};

struct MHDdata {
    const int nr, n;
    fort::array<1, float> r;
    fort::array<3, float> x, y, z;
    fort::array<4, float> ro, p, s, u, v, w, hx, hy, hz;

    MHDdata(const int Nr, const int n, const std::string &gridfile, const std::string &datfile);

    coordAndSide getCoordAndSide(float x, float y, float z) const;
    float variation(int i, int j, int k, int s) const;
    float velocity(const coordAndSide &pos, const int component) const;
    float Teff(const coordAndSide &pos) const;
    float kappa(const coordAndSide &pos) const;
};

#endif
