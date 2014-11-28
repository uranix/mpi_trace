#ifndef __TRACE_H__
#define __TRACE_H__

#include <cmath>

typedef float real;

const int NO_TET = -1;
const real MIN_BARY = 1e-5;

struct tet {
	int p[4];
	int neib[4];
};

struct point {
	real x, y, z;
	point() { }
	point(real x, real y, real z) : x(x), y(y), z(z) { }
	real norm() const {
		return sqrt(x * x + y * y + z * z);
	}
	point &operator+=(const point &p) {
		x += p.x;
		y += p.y;
		z += p.z;
		return *this;
	}
	point &operator-=(const point &p) {
		x -= p.x;
		y -= p.y;
		z -= p.z;
		return *this;
	}
};

struct point4 {
	real x, y, z, w;
	point4(real x, real y, real z, real w)
	: x(x), y(y), z(z), w(w)
	{ }
};

inline point operator+(const point &p1, const point &p2) {
	return point(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
}

inline point operator-(const point &p1, const point &p2) {
	return point(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

inline const point operator*(const real v, const point &p) {
	return point(v * p.x, v * p.y, v * p.z);
}

struct coord {
	real w1, w2, w3, len;
	coord() { }
	coord(real w1, real w2, real w3, real len) : w1(w1), w2(w2), w3(w3), len(len) { }
	real abs() const {
		return fabs(w1) + fabs(w2) + fabs(w3);
	}
};

point bary(const point &rs, const point &r1, const point &r2, const point &r3);
real trace(const point &w, const tet &t, point &r0, int &face, const point pts[]);

#endif
