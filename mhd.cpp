#include "mhd.h"
#include <cmath>
#include <fstream>
#include <stdexcept>

using fort::range;

MHDdata::MHDdata(const int nr, const int n,
        const std::string &gridfile,
        const std::string &datfile)
    :
        nr(nr), n(n),
        r(range(-2, nr+2)),
        x(range(-1, n+1), range(-1, n+1), 6),
        y(range(-1, n+1), range(-1, n+1), 6),
        z(range(-1, n+1), range(-1, n+1), 6),
        ro(range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        p (range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        s (range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        u (range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        v (range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        w (range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        hx(range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        hy(range(0, nr+1), range(-1, n+2), range(-1, n+2), 6),
        hz(range(0, nr+1), range(-1, n+2), range(-1, n+2), 6)
{
    std::ifstream grid(gridfile, std::ios::binary);
    if (!grid)
        throw std::invalid_argument("Could not open " + gridfile);

    grid >> x >> y >> z >> r;
    grid.close();

    std::ifstream dat(datfile, std::ios::binary);
    if (!dat)
        throw std::invalid_argument("Could not open" + datfile);

    dat >> ro >> p >> s >> hx >> hy >> hz >> u >> v >> w;
}

coordAndSide MHDdata::getCoordAndSide(float x, float y, float z) const {
	int side = 0, i, j, k;
	float maxXY, maxYZ, maxZX, radVect, phi = 0, psi = 0;

	maxXY = std::max(std::abs(x), std::abs(y));
	maxYZ = std::max(std::abs(y), std::abs(z));
	maxZX = std::max(std::abs(z), std::abs(x));

	double pi = std::atan(1)*4;

	radVect = std::sqrt(x*x + y*y + z*z);
	k = std::lower_bound(&r(0), &r(nr) + 1, radVect) - &r(0);
	if (k < 1)
		k = 1;
	if (k > nr)
		k = nr;

	if (-x >= maxYZ ) {
		side = 1;
		phi = std::atan2(-y,-x);
		psi = std::atan2(z,-x);
	}

	if ( z >= maxXY ) {
		side = 2;
		phi = std::atan2(-y,z);
		psi = std::atan2(x,z);
	}

	if ( y >= maxZX ) {
		side = 3;
		phi = std::atan2(-x,y);
		psi = std::atan2(z,y);
	}

	if (-y >= maxZX ) {
		side = 4;
		phi = std::atan2(x,-y);
		psi = std::atan2(z,-y);
	}

	if ( x >= maxYZ ) {
		side = 5;
		phi = std::atan2(y,x);
		psi = std::atan2(z,x);
	}

	if (-z >= maxXY ) {
		side = 6;
		phi = std::atan2(-y,-z);
		psi = std::atan2(-x,-z);
	}

	i = std::ceil(2*n*phi/pi + n*0.5);
	j = std::ceil(2*n*psi/pi + n*0.5);
    if (i < 1)
        i = 1;
    if (i > n)
        i = n;
    if (j < 1)
        j = 1;
    if (j > n)
        j = n;

	return coordAndSide{side, i, j, k};
}

float MHDdata::variation(int k, int i, int j, int s) const {
    float ss = 0;
    float um, vm, wm;
    um = u(k, i, j, s);
    vm = v(k, i, j, s);
    wm = w(k, i, j, s);
    for (int di = -1; di <= 1; di++)
        for (int dj = -1; dj <= 1; dj++)
            for (int dk = -1; dk <= 1; dk++) {
                ss += std::pow(u(k + dk, i + di, j + dj, s) - um, 2);
                ss += std::pow(v(k + dk, i + di, j + dj, s) - vm, 2);
                ss += std::pow(w(k + dk, i + di, j + dj, s) - wm, 2);
            }
    ss /= 26;
    return ss;
}

float MHDdata::velocity(const coordAndSide &pos, const int component) const {
    if (component == 0)
        return u(pos.k, pos.i, pos.j, pos.side);
    if (component == 1)
        return v(pos.k, pos.i, pos.j, pos.side);
    if (component == 2)
        return w(pos.k, pos.i, pos.j, pos.side);
    return 0;
}

float MHDdata::Teff(const coordAndSide &pos) const {
    const float P = p(pos.k, pos.i, pos.j, pos.side);
    const float Ro = ro(pos.k, pos.i, pos.j, pos.side);

    const float dv2 = variation(pos.k, pos.i, pos.j, pos.side);
    return P / Ro + dv2;
}

float MHDdata::kappa(const coordAndSide &pos) const {
    const float R = 8.314e7; // erg / K / mol
    const float c = 2.998e10; // cm/s
    const float Na = 6.02e23; // mol^{-1}

    const float Tunit = 4.6e6; // K
    const float Lunit = 2.8e11; // cm
    const float Bunit = 40; // G
    const float Runit = Bunit * Bunit / (R * Tunit); // g / cm^3
    const float Nunit = Runit * Na;
    const float v0 = std::sqrt(R * Tunit);

    const float eion = 13.605 * 11604; // K
    const float fnn = .425;
    const float nq = 2;
    const float nu0 = 456.9e12; // Hz
    const float lambda0 = c / nu0; // cm

    const float P = p(pos.k, pos.i, pos.j, pos.side);
    const float Ro = ro(pos.k, pos.i, pos.j, pos.side);
    const float T = Tunit * P / Ro;
    const float N = Nunit * Ro;

    const float kappa = 1.102e-17 * nq * nq * fnn *
        exp(eion / T / (nq * nq)) * N * N / std::pow(T, 1.5);

    return kappa * Lunit * lambda0 / v0;
}
