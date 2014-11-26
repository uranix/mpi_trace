#include <math.h>
#include <ostream>

struct tet {
	int p[4];
	int neib[4];
};

struct point {
	double x, y, z;
	point() { }
	point(double x, double y, double z) : x(x), y(y), z(z) { }
	double norm() const {
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
	friend std::ostream &operator<<(std::ostream &o, const point &p) {
		o.precision(3);
		return o << "{" << p.x << ", " << p.y << ", " << p.z << "}";
	}
};

struct point4 {
	double x, y, z, w;
	point4(double x, double y, double z, double w)
	: x(x), y(y), z(z), w(w)
	{ }
};

inline point operator+(const point &p1, const point &p2) {
	return point(p1.x + p2.x, p1.y + p2.y, p1.z + p2.z);
}

inline point operator-(const point &p1, const point &p2) {
	return point(p1.x - p2.x, p1.y - p2.y, p1.z - p2.z);
}

inline const point operator*(const double v, const point &p) {
	return point(v * p.x, v * p.y, v * p.z);
}

struct coord {
	double w1, w2, w3, len;
	coord() { }
	coord(double w1, double w2, double w3, double len) : w1(w1), w2(w2), w3(w3), len(len) { }
	double abs() const {
		return fabs(w1) + fabs(w2) + fabs(w3);
	}
	friend std::ostream &operator<<(std::ostream &o, const coord &p) {
		o.precision(3);
		return o << "{" << p.w1 << ", " << p.w2 << ", " << p.w3 << "}";
	}
};

point bary(const point &rs, const point &r1, const point &r2, const point &r3);
double trace(const point &w, const tet &t, point &r0, int &face, const point pts[]);
