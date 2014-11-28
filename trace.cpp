#include "trace.h"

point cross(const point &a, const point &b) {
	return point(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

real dot(const point &a, const point &b) {
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

point bary(const point &rs, const point &r1, const point &r2, const point &r3) {
	point p1(r1 - rs);
	point p2(r2 - rs);
	point p3(r3 - rs);
	point S1 = cross(p2, p3);
	point S2 = cross(p3, p1);
	point S3 = cross(p1, p2);
	point S = S1 + S2 + S3;
	real s1 = dot(S1, S);
	real s2 = dot(S2, S);
	real s3 = dot(S3, S);
	real s = dot(S, S);
	return point(s1 / s, s2 / s, s3 / s);
}

point4 bary4(const point &rs, const point r[]) {
	point p1(r[0] - rs);
	point p2(r[1] - rs);
	point p3(r[2] - rs);
	point p4(r[3] - rs);
	real V1 = dot(cross(p2, p3), p4);
	real V2 = dot(cross(p1, p4), p3);
	real V3 = dot(cross(p4, p1), p2);
	real V4 = dot(cross(p3, p2), p1);
	real V = V1 + V2 + V3 + V4;
	return point4(V1 / V, V2 / V, V3 / V, V4 / V);
}

coord trace_face(const point &w, const point &r0, const point &r1, const point &r2, const point &r3) {
	point a = cross(r1 - r3, r2 - r3);
	real num = dot(a, r0 - r3);
	real denom = dot(a, w);
	if (denom == 0) /* strict zero */
		return coord(-1, -1, -1, -1);
	point rstar(r0);
	real len = num / denom;
	rstar -= len * w;

	point bc = bary(rstar, r1, r2, r3);
	return coord(bc.x, bc.y, bc.z, len);
}

real trace(const point &w, const tet &t, point &r0, int &face, const point *pts) {
	coord b[4];
	point p[8];

	for (int j = 0; j < 4; j++) {
		p[j] = pts[t.p[j]];
		p[j + 4] = p[j];
	}

	point4 input = bary4(r0, p);
	real eps = MIN_BARY;
	if (input.x < eps)
		input.x = eps;
	if (input.y < eps)
		input.y = eps;
	if (input.z < eps)
		input.z = eps;
	if (input.w < eps)
		input.w = eps;
	real wsum = input.x + input.y + input.z + input.w;

	input.x /= wsum;
	input.y /= wsum;
	input.z /= wsum;
	input.w /= wsum;

	/* Shift a bit into tet */
	r0 = input.x * p[0] + input.y * p[1] + input.z * p[2] + input.w * p[3];

	for (int j = 0; j < 4; j++)
		b[j] = trace_face(w, r0, p[j + 1], p[j + 2], p[j + 3]);

	int minj = -1;
	real len = -1;
	for (int j = 0; j < 4; j++) {
		if (b[j].abs() < (1 + MIN_BARY) && b[j].len > len) {
			minj = j;
			len = b[j].len;
		}
	}
	face = minj;

	coord out(b[minj]);

	r0 = out.w1 * p[face + 1] + out.w2 * p[face + 2] + out.w3 * p[face + 3];
	return out.len;
}
