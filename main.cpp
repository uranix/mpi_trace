#include <meshProcessor/mesh.h>
#include <meshProcessor/vtk_stream.h>

#include "MeshView.h"

#include "LebedevQuad.h"

#include "trace.h"
#include "umfsolve.h"

#include "gpu.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <memory>

#include <fenv.h>

const int OUTER_DOMAIN = -1;

using namespace mesh3d;
typedef mesh3d::index idx;

double volume(const vector &p1, const vector &p2, const vector &p3, const vector &p4) {
    return 1. / 6 * (p3 - p4).dot((p1 - p4) % (p2 - p4));
}

struct VertTetQual {
    idx vertIdx;
    idx tetIdx;
    double qual;

    VertTetQual(idx vertIdx, idx tetIdx, double qual) : vertIdx(vertIdx), tetIdx(tetIdx), qual(qual) { }
};

VertTetQual bestTetWithRay(const vertex &v, const vector &omega) {
    auto &tets = v.tetrahedrons();
    const double BIG_SA = 1e3;
    double minsa = BIG_SA;
    idx best = BAD_INDEX;

    for (auto tetVertex : tets) {
        const tetrahedron &tet = *(tetVertex.t);
        const int localIndex = tetVertex.li;
        const vector rr = tet.p(localIndex).r() - omega;

        double V[4];
        V[0] = volume(rr, tet.p(1).r(), tet.p(2).r(), tet.p(3).r());
        V[1] = volume(tet.p(0).r(), rr, tet.p(2).r(), tet.p(3).r());
        V[2] = volume(tet.p(0).r(), tet.p(1).r(), rr, tet.p(3).r());
        V[3] = volume(tet.p(0).r(), tet.p(1).r(), tet.p(2).r(), rr);

        double Vtot = 0;
        double Vabs = 0;
        for (int j = 0; j < 4; j++)
            if (j != localIndex) {
                Vtot += V[j];
                Vabs += fabs(V[j]);
            }

        double sa = Vabs / Vtot;
        if (sa < 0)
            sa = BIG_SA;

        if (sa < minsa) {
            minsa = sa;
            best = tet.idx();
        }
    }
    return VertTetQual(v.idx(), best, minsa);
}

struct DirectionSolver {
    const int procs, rank;
    const mesh &m;
    const MeshView &meshview;
    const vector omega;

    std::vector<VertTetQual> interface;
    std::vector<int> owner;
    std::unordered_map<idx, int> vertToVarMap;
    std::vector<int> isInner;

    std::vector<int> unknownSize;
    std::vector<int> unknownStarts;

    MPI::Datatype SLAE_ROW_TYPE;
    std::vector<slae_row> slae;
    std::vector<double> sol;
    std::vector<real> Idir;

    DirectionSolver(int size, int rank, const mesh &m, const MeshView &mv, const vector &omega)
    :
        procs(size), rank(rank),
        m(m), meshview(mv), omega(omega)
    {
        if (static_cast<int>(m.domains()) != procs || static_cast<int>(m.domain()) != rank) {
            std::cout << "Mesh partition has wrong number of procs or wrong rank" << std::endl;
            MPI::COMM_WORLD.Abort(0);
        }
        int count = 2;
        MPI::Datatype oldtypes[2] = {MPI::DOUBLE, MPI::INT};
        int blockcounts[2] = {2 + 2 * NFREQ, 3};
        MPI::Aint offsets[2] = {0, (2 + 2 * NFREQ) * sizeof(double)};
        SLAE_ROW_TYPE = MPI::Datatype::Create_struct(count, blockcounts, offsets, oldtypes);
        SLAE_ROW_TYPE.Commit();
    }

    void prepare() {
        extractInterface();
        selectOwners();
        computeVertToVarMap();
    }

    void extractInterface() {
        for (idx i = 0; i < m.vertices().size(); i++)
            if (m.vertices(i).aliases().size() > 0) {
                auto best = bestTetWithRay(m.vertices(i), omega);
                interface.push_back(best);
            }
    }

    void selectOwners() {
        std::vector<int> iface_sizes(procs);
        std::vector<int> iface_starts(procs + 1);
        iface_sizes[rank] = interface.size();
        MPI::COMM_WORLD.Allgather(&iface_sizes[rank], 1, MPI::INT, &iface_sizes[0], 1, MPI::INT);
        iface_starts[0] = 0;
        for (int i = 0; i < procs; i++)
            iface_starts[i + 1] = iface_starts[i] + iface_sizes[i];

        std::vector<int> vertexIndicesPerDomain(iface_starts.back());
        std::vector<double> qualValues(iface_starts.back());

        for (int j = 0; j < iface_sizes[rank]; j++) {
            vertexIndicesPerDomain[iface_starts[rank] + j] = interface[j].vertIdx;
            qualValues            [iface_starts[rank] + j] = interface[j].qual;
        }

        MPI::COMM_WORLD.Allgatherv(
            &vertexIndicesPerDomain[iface_starts[rank]], iface_sizes[rank], MPI::INT, &vertexIndicesPerDomain[0],
            &iface_sizes[0], &iface_starts[0], MPI::INT
        );
        MPI::COMM_WORLD.Allgatherv(
            &qualValues[iface_starts[rank]], iface_sizes[rank], MPI::DOUBLE, &qualValues[0],
            &iface_sizes[0], &iface_starts[0], MPI::DOUBLE
        );

        owner.resize(interface.size());
        for (size_t i = 0; i < interface.size(); i++) {
            const vertex &v = m.vertices(interface[i].vertIdx);
            owner[i] = OUTER_DOMAIN;
            double bestQual = 1 + 1e-8;

            if (interface[i].qual < bestQual) {
                bestQual = interface[i].qual;
                owner[i] = rank;
            }

            for (auto it : v.aliases()) {
                int dom = it.first;
                int rid = it.second;

                int pos = std::lower_bound(
                        &vertexIndicesPerDomain[iface_starts[dom]],
                        &vertexIndicesPerDomain[iface_starts[dom+1]],
                        rid
                    ) - &vertexIndicesPerDomain[0];
                MESH3D_ASSERT(vertexIndicesPerDomain[pos] == rid);
                if (qualValues[pos] < bestQual) {
                    owner[i] = dom;
                    bestQual = qualValues[pos];
                }
            }
            /*
            * If no better match found, use OUTER_DOMAIN as owner domain. That means true boundary.
            */
        }
    }

    void computeVertToVarMap() {
        std::vector<int> unknownVert;
        for (size_t i = 0; i < interface.size(); i++) {
            if (owner[i] == rank)
                unknownVert.push_back(interface[i].vertIdx);
        }

        unknownSize.resize(procs);
        unknownStarts.resize(procs + 1);

        unknownSize[rank] = unknownVert.size();
        MPI::COMM_WORLD.Allgather(&unknownSize[rank], 1, MPI::INT, &unknownSize[0], 1, MPI::INT);
        unknownStarts[0] = 0;
        for (int i = 0; i < procs; i++)
            unknownStarts[i + 1] = unknownStarts[i] + unknownSize[i];

        std::vector<int> allUnknownVert(unknownStarts.back());
        for (size_t i = 0; i < unknownVert.size(); i++)
            allUnknownVert[unknownStarts[rank] + i] = unknownVert[i];

        MPI::COMM_WORLD.Allgatherv(&allUnknownVert[unknownStarts[rank]], unknownSize[rank], MPI::INT, &allUnknownVert[0],
            &unknownSize[0], &unknownStarts[0], MPI::INT);


        for (idx g = 0; g < m.faces().size(); g++) {
            const face &f = m.faces(g);
            if (f.is_border())
                for (int j = 0; j < 3; j++)
                    vertToVarMap[f.p(j).idx()] = BC_VAR;
        }

        for (size_t j = 0; j < interface.size(); j++) {
            int dom = owner[j];
            if (dom == OUTER_DOMAIN)
                continue;
            idx i = interface[j].vertIdx;
            auto &alias = m.vertices(i).aliases();
            MESH3D_ASSERT(alias.size());
            int rid;
            if (dom != rank) {
                auto it = alias.find(dom);
                MESH3D_ASSERT(it != alias.end());
                rid = it->second;
            } else
                rid = i;
            vertToVarMap[i] = std::lower_bound(
                    &allUnknownVert[unknownStarts[dom]],
                    &allUnknownVert[unknownStarts[dom + 1]],
                    rid
                ) - &allUnknownVert[0];
        }
    }

    void traceFromBoundary() {
        point w(omega.x, omega.y, omega.z);
        slae.resize(unknownStarts.back());

        const auto &pts   = meshview.pts;
        const auto &elems  = meshview.elems;

        for (size_t j = 0; j < interface.size(); j++) {
            if (owner[j] != rank)
                continue;
            int itet = interface[j].tetIdx;
            int i = interface[j].vertIdx;
            point r(pts[i]);
            real a[NFREQ];
            real b[NFREQ];

            for (int ifreq = 0; ifreq < NFREQ; ifreq++) {
                a[ifreq] = 1;
                b[ifreq] = 0;
            }

            int vout[3];
            real wei[3];
            while (true) {
                int face;
                const MeshElement &currTet = elems[itet];
                real len = trace(w, currTet, r, face, &pts[0]);
                for (int ifreq = 0; ifreq < NFREQ; ifreq++) {
                    real delta = len * currTet.kappa[ifreq];
                    real q = exp(-delta);
                    b[ifreq] += a[ifreq] * currTet.Ip[ifreq] * (1 - q);
                    a[ifreq] *= q;
                }
                itet = currTet.neib[face];
                if (itet == NO_TET) {
                    vout[0] = currTet.p[(face + 1) & 3];
                    vout[1] = currTet.p[(face + 2) & 3];
                    vout[2] = currTet.p[(face + 3) & 3];
                    point ws = bary(r, pts[vout[0]], pts[vout[1]], pts[vout[2]]);
                    wei[0] = ws.x;
                    wei[1] = ws.y;
                    wei[2] = ws.z;
                    break;
                }
            }

            for (int k = 0; k < 3; k++)
                MESH3D_ASSERT(vertToVarMap.count(vout[k]) > 0);

            int row = vertToVarMap[i];

            for (int ifreq = 0; ifreq < NFREQ; ifreq++) {
                slae[row].beta[ifreq] = b[ifreq];
                slae[row].alpha[ifreq] = a[ifreq];
            }
            slae[row].w[0] = wei[0];
            slae[row].w[1] = wei[1];
            for (int k = 0; k < 3; k++) {
                slae[row].cols[k] = vertToVarMap[vout[k]];
            }
        }
    }

    /*
     * Gather system on rank root
     * */
    void gatherSystem(int root) {
        MPI::COMM_WORLD.Gatherv(&slae[unknownStarts[rank]], unknownSize[rank], SLAE_ROW_TYPE, &slae[0],
            &unknownSize[0], &unknownStarts[0], SLAE_ROW_TYPE, root);
    }
    void solveSystem(int root) {
        if (slae.size() == 0)
            return;
        sol.assign(slae.size() * NFREQ, 0);
        if (root != rank)
            return;
        std::cout << "[rank #" << rank << "] Solving system of " << slae.size() << " x " << NFREQ << " eqns for dir = " << omega << std::endl;
        UmfSolveStatus status = umfsolve(slae, sol);
        if (status != OK) {
            std::cout << "UMFPack solver failed: ";
            if (status == SYMBOLIC_FAILED)
                std::cout << "Could not perform symbolic decomposition.";
            if (status == NUMERIC_FAILED)
                std::cout << "Could not perform numeric decomposition.";
            if (status == SOLVE_FAILED)
                std::cout << "Could not solve decomposed system.";
            std::cout << std::endl;
            MPI::COMM_WORLD.Abort(0);
        }
        double error = testSlaeSolution(slae, sol);

        std::cout << "Error norm : " << error << std::endl;
    }

    /*
     * Bcast sol vector from rank root
     * */
    void bcastSolution(int root) {
        if (sol.size())
            MPI::COMM_WORLD.Bcast(&sol[0], sol.size(), MPI::DOUBLE, root);

        size_t M = sol.size() / NFREQ;
        int nP = m.vertices().size();
        Idir.resize(nP * NFREQ, -1e10);
        isInner.assign(nP, 1);

        for (auto it : vertToVarMap) {
            idx i = it.first;
            int j = it.second;
            if (j >= 0) {
                for (int ifreq = 0; ifreq < NFREQ; ifreq++)
                    Idir[i * NFREQ + ifreq] = sol[ifreq * M + j];
                isInner[i] = 0;
            } else {
                for (int ifreq = 0; ifreq < NFREQ; ifreq++)
                    Idir[i * NFREQ + ifreq] = 0;
            }
        }
    }
};

void saveSolution(
        const std::string &prefix, int rank,
        const mesh &m, const MeshView &mv, const std::vector<real> &U)
{
    vtk_stream v((prefix + "_sol." + std::to_string(rank) + ".vtk").c_str());
    v.write_header(m, "Solution");
    std::vector<real> Ufreq(U.size() / NFREQ);
    for (int ifreq = 0; ifreq < NFREQ; ifreq++) {
        for (size_t i = 0; i < Ufreq.size(); i++)
            Ufreq[i] = U[NFREQ * i + ifreq];
        v.append_point_data(Ufreq.data(), "U" + std::to_string(ifreq));
    }
    v.close();
}

int main(int argc, char **argv) {
    feenableexcept(FE_INVALID | FE_OVERFLOW | FE_DIVBYZERO);
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();

    if (argc != 3) {
        std::cerr << "USAGE: mpirun -np <np> " << argv[0] << " <prefix> <ndir>" << std::endl;
        MPI::Finalize();
        return 1;
    }

    std::fstream meshfile(std::string(argv[1]) + "." + std::to_string(rank) + ".m3d", std::ios::binary | std::ios::in);
    mesh m(meshfile);
    MeshView mv(m);

    const int devmap[] = {0, 1, 2, 3, 4, 5, 6, 7};

    GPUMeshView gmv(rank, devmap[rank], mv);
    GPUAverageSolution gas(gmv);

    if (!rank) {
        std::cout << "Running alignment test" << std::endl;
        bool ret = alignment_test();
        if (!ret) {
            std::cout << "Test failed, aborting" << std::endl;
            MPI::COMM_WORLD.Abort(0);
        }
        std::cout << "Test passed" << std::endl;
    }

    std::cout << "Mesh for domain " << rank << " loaded" << std::endl;

    LebedevQuad quad = LebedevQuadBank::lookupByOrder(atoi(argv[2]));

    if (!rank)
        std::cout << "Using " << quad.order << " directions" << std::endl;

    const int roundSize = 2 * size;
    const int rounds = (quad.order + roundSize - 1) / roundSize;

    std::vector<point> ws;
    for (int i = 0; i < quad.order; i++)
        ws.push_back(point(quad.x[i], quad.y[i], quad.z[i]));

    std::vector<std::unique_ptr<DirectionSolver> > ds(roundSize);
    GPUMultipleDirectionSolver gmds(roundSize, gmv, ws);

    double spent = 0, prepare = 0, boundary = 0, slae = 0, trace = 0;

    for (int round = 0; round < rounds; round++) {
        MPI::COMM_WORLD.Barrier();
        if (!rank)
            std::cout << "------- NEW ROUND --------" << std::endl;

        const int activeDirections = std::min(quad.order - round * roundSize, roundSize);
        if (!rank)
            std::cout << "Active directions = " << activeDirections << std::endl;

        for (int j = 0; j < activeDirections; j++) {
            int i = round * roundSize + j;
            vector omega(quad.x[i], quad.y[i], quad.z[i]);
            ds[j] = std::unique_ptr<DirectionSolver>(new DirectionSolver(size, rank, m, mv, omega));
        }

        double start = MPI::Wtime();

        for (int j = 0; j < activeDirections; j++)
            ds[j]->prepare();

        double mark1 = MPI::Wtime();

        for (int j = 0; j < activeDirections; j++)
            ds[j]->traceFromBoundary();

        double mark2 = MPI::Wtime();

        for (int j = 0; j < activeDirections; j++)
            ds[j]->gatherSystem(j % size);

        for (int j = 0; j < activeDirections; j++)
            ds[j]->solveSystem(j % size);

        for (int j = 0; j < activeDirections; j++)
            ds[j]->bcastSolution(j % size);

        double mark3 = MPI::Wtime();

        for (int j = 0; j < activeDirections; j++)
            gmds.setBoundary(j, ds[j]->Idir, ds[j]->isInner);

#ifdef ALL_AT_ONCE
        gmds.traceInterior(round * roundSize, 0, activeDirections);
#else
        for (int j = 0; j < activeDirections; j++)
            gmds.traceInterior(round * roundSize, j, 1);
#endif

        for (int j = 0; j < activeDirections; j++) {
            int i = round * roundSize + j;
            gas.add(gmds.Idir(j), static_cast<real>(quad.w[i]));
        }

        if (round == rounds - 1)
            gmds.sync();

        double stop = MPI::Wtime();

        spent    += stop  - start;
        prepare  += mark1 - start;
        boundary += mark2 - mark1;
        slae     += mark3 - mark2;
        trace    += stop  - mark3;
    }

    std::cout << "---------------------------------------------------------" << std::endl;
    std::cout << "[" << rank << "] Time spent: " << (spent    * 1e3) << "ms" << std::endl;
    std::cout << "[" << rank << "]   Prepare : " << (prepare  * 1e3) << "ms" << std::endl;
    std::cout << "[" << rank << "]   Boundary: " << (boundary * 1e3) << "ms" << std::endl;
    std::cout << "[" << rank << "]   SLAE    : " << (slae     * 1e3) << "ms" << std::endl;
    std::cout << "[" << rank << "]   TraceAll: " << (trace    * 1e3) << "ms" << std::endl;

    saveSolution(argv[1], rank, m, mv, gas.retrieve());

    MPI::Finalize();
    return 0;
}
