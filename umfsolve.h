#ifndef __UMFSOLVE_H__
#define __UMFSOLVE_H__

#include "config.h"

const static int BC_VAR = -1;

struct slae_row {
    double beta[NFREQ];
    double alpha[NFREQ];
    double w[2]; /* w[2] = 1 - w[0] - w[1] */
    int cols[3];
};

#include <vector>

enum UmfSolveStatus {
    OK = 0,
    SYMBOLIC_FAILED,
    NUMERIC_FAILED,
    SOLVE_FAILED
};

UmfSolveStatus umfsolve(const std::vector<slae_row> &slae, std::vector<double> &sol);
double testSlaeSolution(const std::vector<slae_row> &slae, std::vector<double> &sol);

#endif
