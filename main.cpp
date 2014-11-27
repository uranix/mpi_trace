#include <meshProcessor/mesh.h>
#include <meshProcessor/vtk_stream.h>

#include "trace.h"
#include "umfsolve.h"

#include <mpi.h>
#include <iostream>
#include <fstream>
#include <unordered_map>

const int OUTER_DOMAIN = -1;
const int NO_TET = -1;

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
    const vector omega;

    std::vector<VertTetQual> interface;
    std::vector<int> owner;
    std::unordered_map<idx, int> vertToVarMap;

    std::vector<int> unknownSize;
    std::vector<int> unknownStarts;

    MPI::Datatype SLAE_ROW_TYPE;

    std::vector<slae_row> slae;

    std::vector<point> pts;
    std::vector<tet> tets;
    std::vector<double> kappa;
    std::vector<double> Ip;
    std::vector<double> u;

    DirectionSolver(int size, int rank, const mesh &m, const std::string &prefix, const vector &omega) :
        procs(size), rank(rank),
        m(m), omega(omega)
    {
        if (static_cast<int>(m.domains()) != procs || static_cast<int>(m.domain()) != rank)
            throw std::invalid_argument("MPI rank or size and mesh rank or size mismatched");

        std::cout << "Mesh for domain " << rank << " loaded" << std::endl;

        int count = 2;
        MPI::Datatype oldtypes[2] = {MPI::DOUBLE, MPI::INT};
        int blockcounts[2] = {4, 3};
        MPI::Aint offsets[2] = {0, 4 * sizeof(double)};
        SLAE_ROW_TYPE = MPI::Datatype::Create_struct(count, blockcounts, offsets, oldtypes);
        SLAE_ROW_TYPE.Commit();
    }

    void splitInterface() {
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

                int pos = std::lower_bound(&vertexIndicesPerDomain[iface_starts[dom]], &vertexIndicesPerDomain[iface_starts[dom+1]], rid) - &vertexIndicesPerDomain[0];
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

    size_t computeVertToVarMap() {
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
            vertToVarMap[i] = std::lower_bound(&allUnknownVert[unknownStarts[dom]], &allUnknownVert[unknownStarts[dom + 1]], rid) - &allUnknownVert[0];
        }

        return unknownStarts.back();
    }

    void convertMesh() {
        int nP = m.vertices().size();
        int nT = m.tets().size();

        pts.resize(nP);
        tets .resize(nT);

        for (int i = 0; i < nT; i++) {
            const tetrahedron &tet = m.tets(i);
            for (int j = 0; j < 4; j++) {
                tets[i].p[j] = tet.p(j).idx();
                const face &f = tet.f(j).flip();
                if (f.is_border())
                    tets[i].neib[j] = NO_TET;
                else
                    tets[i].neib[j] = f.tet().idx();
            }
        }

        for (int i = 0; i < nP; i++) {
            const vertex &v = m.vertices(i);
            pts[i].x = v.r().x;
            pts[i].y = v.r().y;
            pts[i].z = v.r().z;
        }
    }

    void setParams() {
        int nT = m.tets().size();
        kappa.resize(nT);
        Ip   .resize(nT);

        for (int i = 0; i < nT; i++) {
            const tetrahedron &tet = m.tets(i);
            if (tet.color() == 1) {
                kappa[i] = 11;
                Ip[i] = 1;
            } else {
                kappa[i] = 0.1;
                Ip[i] = 0;
            }
        }
    }

    void traceFromBoundary() {
        point w(omega.x, omega.y, omega.z);
        slae.resize(unknownStarts.back());

        for (size_t j = 0; j < interface.size(); j++) {
            if (owner[j] != rank)
                continue;
            int itet = interface[j].tetIdx;
            int i = interface[j].vertIdx;
            point r(pts[i]);
            double a = 1, b = 0;
            int vout[3];
            double wei[3];
            while (true) {
                int face;
                double len = trace(w, tets[itet], r, face, &pts[0]);
                double delta = len * kappa[itet];
                double q = exp(-delta);
                b += a * Ip[itet] * (1 - q);
                a *= q;
                const tet &old = tets[itet];
                itet = old.neib[face];
                if (itet == NO_TET) {
                    vout[0] = old.p[(face + 1) & 3];
                    vout[1] = old.p[(face + 2) & 3];
                    vout[2] = old.p[(face + 3) & 3];
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

            slae[row].beta = b;
            for (int k = 0; k < 3; k++) {
                slae[row].alpha[k] = wei[k] * a;
                slae[row].cols[k] = vertToVarMap[vout[k]];
            }
        }
    }

    void gatherAndSolveSystem() {
        MPI::COMM_WORLD.Gatherv(&slae[unknownStarts[rank]], unknownSize[rank], SLAE_ROW_TYPE, &slae[0],
            &unknownSize[0], &unknownStarts[0], SLAE_ROW_TYPE, 0);

        std::vector<double> sol(slae.size());
        if (rank == 0 && slae.size() > 0) {
            umfsolve(slae, sol);
            double error = testSlaeSolution(slae, sol);

            std::cout << "Error norm : " << error << std::endl;
        }
        MPI::COMM_WORLD.Bcast(&sol[0], sol.size(), MPI::DOUBLE, 0);

        int nP = m.vertices().size();
        u.assign(nP, 0);

        for (auto it : vertToVarMap) {
            idx i = it.first;
            int j = it.second;
            if (j >= 0)
                u[i] = sol[j];
        }
    }

    void traceRest() {
        int nP = m.vertices().size();
        point w(omega.x, omega.y, omega.z);

        for (int i = 0; i < nP; i++) {
            if (vertToVarMap.count(i) > 0 && vertToVarMap[i] >= 0)
                continue;
            int itet = m.vertices(i).tetrahedrons().front().t->idx();
            int face;
            point r(pts[i]);
            double a = 1, b = 0;
            int vout[3];
            double wei[3];
            while (true) {
                double len = trace(w, tets[itet], r, face, &pts[0]);
                double delta = len * kappa[itet];
                double q = exp(-delta);
                b += a * Ip[itet] * (1 - q);
                a *= q;
                const tet &old = tets[itet];
                itet = old.neib[face];
                if (itet == NO_TET) {
                    vout[0] = old.p[(face + 1) & 3];
                    vout[1] = old.p[(face + 2) & 3];
                    vout[2] = old.p[(face + 3) & 3];
                    point ws = bary(r, pts[vout[0]], pts[vout[1]], pts[vout[2]]);
                    wei[0] = ws.x;
                    wei[1] = ws.y;
                    wei[2] = ws.z;
                    break;
                }
            }

            u[i] = b;
            for (int k = 0; k < 3; k++)
                u[i] += u[vout[k]] * a * wei[k];
        }
    }

    void saveOutput(const std::string &prefix) {
        vtk_stream v((prefix + "_sol." + std::to_string(rank) + ".vtk").c_str());
        v.write_header(m, "Solution");
        v.append_cell_data(&kappa[0], "kappa");
        v.append_cell_data(&Ip[0], "Ip");
        v.append_point_data(&u[0], "u");
        v.close();
    }

};

int main(int argc, char **argv) {
    MPI::Init(argc, argv);
    int size = MPI::COMM_WORLD.Get_size();
    int rank = MPI::COMM_WORLD.Get_rank();

    if (argc != 2) {
        std::cerr << "USAGE: mpirun -np <np> " << argv[0] << " <prefix>" << std::endl;
        MPI::Finalize();
        return 1;
    }

    vector omega(1, 0, 0);
    omega *= 1 / omega.norm();

    if (!rank)
        std::cout << "omega = " << omega << std::endl;

    std::fstream meshfile(std::string(argv[1]) + "." + std::to_string(rank) + ".m3d", std::ios::binary | std::ios::in);
    mesh m(meshfile);

    DirectionSolver ds(size, rank, m, argv[1], omega);

    double start = MPI::Wtime();

    ds.convertMesh();
    ds.setParams();

    double mark0 = MPI::Wtime();    

    ds.splitInterface();
    ds.selectOwners();
    size_t totalUnknowns = ds.computeVertToVarMap();

    if (!rank)
        std::cout << "Total unknowns: " << totalUnknowns << std::endl;

    double mark1 = MPI::Wtime();
    ds.traceFromBoundary();
    double mark2 = MPI::Wtime();
    ds.gatherAndSolveSystem();
    double mark3 = MPI::Wtime();
    ds.traceRest();
    double stop = MPI::Wtime();

    if (!rank) {
        std::cout << "Time spent: " << ((stop  - start) * 1e3) << "ms" << std::endl;
        std::cout << "  Convert : " << ((mark0 - start) * 1e3) << "ms" << std::endl;
        std::cout << "  Prepare : " << ((mark1 - mark0) * 1e3) << "ms" << std::endl;
        std::cout << "  Boundary: " << ((mark2 - mark1) * 1e3) << "ms" << std::endl;
        std::cout << "  SLAE    : " << ((mark3 - mark2) * 1e3) << "ms" << std::endl;
        std::cout << "  TraceAll: " << ((stop  - mark3) * 1e3) << "ms" << std::endl;
    }

    ds.saveOutput(argv[1]);

    MPI::Finalize();
    return 0;
}
