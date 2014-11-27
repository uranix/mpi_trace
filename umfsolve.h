#ifndef __UMFSOLVE_H__
#define __UMFSOLVE_H__

const static int BC_VAR = -1;

struct slae_row {
    double beta;
    double alpha[3];
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
