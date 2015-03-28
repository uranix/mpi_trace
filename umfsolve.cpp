#include "umfsolve.h"

#include <umfpack.h>
#include <algorithm>

UmfSolveStatus umfsolve(const std::vector<slae_row> &slae, double *sol, const int ifreq);

UmfSolveStatus umfsolve(const std::vector<slae_row> &slae, std::vector<double> &sol) {
    const size_t m = slae.size();
    UmfSolveStatus ret;
    for (int ifreq = 0; ifreq < NFREQ; ifreq++) {
        ret = umfsolve(slae, &sol[ifreq * m], ifreq);
        if (ret != OK)
            return ret;
    }
    return OK;
}

UmfSolveStatus umfsolve(const std::vector<slae_row> &slae, double *sol, const int ifreq) {
    const size_t m = slae.size();
    std::vector<int> Ap(slae.size() + 1);
    Ap[0] = 0;
    std::vector<double> Ax;
    std::vector<int> Ai;
    for (size_t i = 0; i < m; i++) {
        std::vector<std::pair<int, double>> row;

        row.push_back(std::pair<int, double>(i, -1.0));
        for (int j = 0; j < 3; j++)
            if (slae[i].cols[j] != BC_VAR) {
                double w;
                if (j < 2)
                    w = slae[i].w[j];
                else
                    w = 1 - slae[i].w[0] - slae[i].w[1];
                row.push_back(std::pair<int, double>(slae[i].cols[j], slae[i].alpha[ifreq] * w));
            }

        std::sort(row.begin(), row.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) { return a.first < b.first; } );

        Ap[i + 1] = Ap[i] + row.size();
        for (auto it = row.begin(); it != row.end(); it++) {
            Ai.push_back(it->first);
            Ax.push_back(it->second);
        }
    }
    void *symbolic;
    void *numeric;
    if (umfpack_di_symbolic(m, m, &Ap[0], &Ai[0], 0, &symbolic, 0, 0) != UMFPACK_OK)
        return SYMBOLIC_FAILED;
    if (umfpack_di_numeric(&Ap[0], &Ai[0], &Ax[0], symbolic, &numeric, 0, 0) != UMFPACK_OK) {
        umfpack_di_free_symbolic(&symbolic);
        return NUMERIC_FAILED;
    }
    std::vector<double> rhs(m);
    for (size_t i = 0; i < m; i++)
        rhs[i] = -slae[i].beta[ifreq];
    if (umfpack_di_solve(UMFPACK_At, &Ap[0], &Ai[0], &Ax[0], sol, &rhs[0], numeric, 0, 0) != UMFPACK_OK) {
        umfpack_di_free_numeric(&numeric);
        umfpack_di_free_symbolic(&symbolic);
        return SOLVE_FAILED;
    }
    umfpack_di_free_numeric(&numeric);
    umfpack_di_free_symbolic(&symbolic);
    return OK;
}

double testSlaeSolution(const std::vector<slae_row> &slae, std::vector<double> &sol) {
    const size_t m = slae.size();
    double norm = 0;
    for (int ifreq = 0; ifreq < NFREQ; ifreq++) {
        for (size_t i = 0; i < m; i++) {
            double res = sol[ifreq * m + i] - slae[i].beta[ifreq];
            for (int j = 0; j < 3; j++) {
                int col = slae[i].cols[j];
                double w;
                if (j < 2)
                    w = slae[i].w[j];
                else
                    w = 1 - slae[i].w[0] - slae[i].w[1];
                if (col != BC_VAR)
                    res -= slae[i].alpha[ifreq] * w * sol[ifreq * m + col];
            }

            if (fabs(res) > norm)
                norm = fabs(res);
        }
    }
    return norm;
}
