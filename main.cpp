#include <meshProcessor/mesh.h>
#include <meshProcessor/vtk_stream.h>
#include <tiny/format.h>

#include <umfpack.h>

#include "trace.h"

#include <mpi.h>
#include <iostream>
#include <fstream>

using namespace mesh3d;
typedef mesh3d::index idx;

double volume(const vector &p1, const vector &p2, const vector &p3, const vector &p4) {
	return 1. / 6 * (p3 - p4).dot((p1 - p4) % (p2 - p4));
}

double belong(const vertex &v, const vector &omega, idx &best) {
	auto &tets = v.tetrahedrons();
	double sum = 1e3;
	best = BAD_INDEX;
	for (auto it = tets.begin(); it != tets.end(); it++) {
		const tetrahedron &tet = *(it->t);
		const vector rr = tet.p(it->li).r() - omega;
		double V[4];
		V[0] = volume(rr, tet.p(1).r(), tet.p(2).r(), tet.p(3).r());
		V[1] = volume(tet.p(0).r(), rr, tet.p(2).r(), tet.p(3).r());
		V[2] = volume(tet.p(0).r(), tet.p(1).r(), rr, tet.p(3).r());
		V[3] = volume(tet.p(0).r(), tet.p(1).r(), tet.p(2).r(), rr);
		double Vtot = 0;
		double Vabs = 0;
		for (int j = 0; j < 4; j++)
			if (j != it->li) {
				Vtot += V[j];
				Vabs += fabs(V[j]);
			}
		double sa = Vabs / Vtot;
		if (sa < 0)
			sa = 2;
		if (sa < sum) {
			sum = sa;
			best = tet.idx();
		}
	}
	return sum;
}



int main(int argc, char **argv) {
	MPI::Init(argc, argv);
	int size = MPI::COMM_WORLD.Get_size();
	int rank = MPI::COMM_WORLD.Get_rank();

	if (argc != 2) {
		std::cerr << "USAGE: mpirun -np <np> " << argv[0] << " <prefix>" << std::endl;
		MPI::Finalize();
		return 1;
	}

	std::fstream meshfile(tiny::format("%s.%d.m3d", argv[1], rank).c_str(), std::ios::in | std::ios::binary);
	mesh m(meshfile);
	if (m.domains() != size || m.domain() != rank) {
		std::cerr << "MPI rank or size and mesh rank or size mismatched" << std::endl;
		MPI::Finalize();
		return 2;
	}

	std::cout << "Mesh for domain " << rank << " loaded" << std::endl;

	double start = MPI::Wtime();

	vector omega(1, 2, 3);
	omega *= 1 / omega.norm();
	if (!rank)
		std::cout << "omega = " << omega << std::endl;

	std::vector<std::pair<idx, double>> iface;
	std::vector<idx> intet;

	for (idx i = 0; i < m.vertices().size(); i++) {
		if (m.vertices(i).aliases().size() > 0) {
			idx tet;
			double sa = belong(m.vertices(i), omega, tet);
			iface.push_back(std::pair<idx, double>(i, sa));
			intet.push_back(tet);
		}
	}

	std::vector<int> num_aliases(size);
	num_aliases[rank] = iface.size();
	MPI::COMM_WORLD.Allgather(&num_aliases[rank], 1, MPI::INT, &num_aliases[0], 1, MPI::INT);
	std::vector<int> starts(size + 1);
	starts[0] = 0;
	for (int i = 0; i < size; i++)
		starts[i + 1] = starts[i] + num_aliases[i];
	
	std::vector<int> local_index(starts.back());
	std::vector<double> belong_value(starts.back());

	for (int j = 0; j < num_aliases[rank]; j++) {
		local_index[starts[rank] + j] = iface[j].first;
		belong_value[starts[rank] + j] = iface[j].second;
	}

	MPI::COMM_WORLD.Allgatherv(&local_index[starts[rank]], num_aliases[rank], MPI::INT, &local_index[0], 
		&num_aliases[0], &starts[0], MPI::INT);
	MPI::COMM_WORLD.Allgatherv(&belong_value[starts[rank]], num_aliases[rank], MPI::DOUBLE, &belong_value[0], 
		&num_aliases[0], &starts[0], MPI::DOUBLE);

	std::vector<int> owner(iface.size());
	for (int i = 0; i < iface.size(); i++) {
		const vertex &v = m.vertices(iface[i].first);
		owner[i] = -1;
		double bv = 1 + 1e-8;

		if (iface[i].second < bv) {
			bv = iface[i].second;
			owner[i] = rank;
		}

		for (auto it = v.aliases().begin(); it != v.aliases().end(); it++) {
			int dom = it->first;
			int rid = it->second;

			int pos = std::lower_bound(&local_index[starts[dom]], &local_index[starts[dom+1]], rid) - &local_index[0];
			MESH3D_ASSERT(local_index[pos] == rid);
			if (belong_value[pos] < bv) {
				owner[i] = dom;
				bv = belong_value[pos];
			}
		}
	}

	std::vector<int> myvars;
	for (int i = 0; i < iface.size(); i++) {
		if (owner[i] == rank)
			myvars.push_back(iface[i].first);
	}
	std::vector<int> numvars(size);
	numvars[rank] = myvars.size();
	MPI::COMM_WORLD.Allgather(&numvars[rank], 1, MPI::INT, &numvars[0], 1, MPI::INT);
	starts[0] = 0;
	for (int i = 0; i < size; i++)
		starts[i + 1] = starts[i] + numvars[i];
	std::vector<int> vars(starts.back());
	for (int i = 0; i < myvars.size(); i++)
		vars[starts[rank] + i] = myvars[i];
	
	MPI::COMM_WORLD.Allgatherv(&vars[starts[rank]], numvars[rank], MPI::INT, &vars[0], 
		&numvars[0], &starts[0], MPI::INT);

	int nT = m.tets().size();
	int nP = m.vertices().size();

	std::vector<int> varno(nP, -2);
	std::vector<int> allowner(nP, -1);
	for (idx g = 0; g < m.faces().size(); g++) {
		const face &f = m.faces(g);
		if (f.is_border())
			for (int j = 0; j < 3; j++)
				varno[f.p(j).idx()] = -1;
	}

	for (int j = 0; j < iface.size(); j++) {
		allowner[iface[j].first] = owner[j];
	}

	for (int i = 0; i < nP; i++) {
		if (allowner[i] != -1) {
			int dom = allowner[i];
			auto &alias = m.vertices(i).aliases();
			MESH3D_ASSERT(alias.size());
			int rid;
			if (dom != rank) {
				auto it = alias.find(dom);
				MESH3D_ASSERT(it != alias.end() );
				rid = it->second;
			} else
				rid = i;
			varno[i] = std::lower_bound(&vars[starts[dom]], &vars[starts[dom + 1]], rid) - &vars[0];
		}
	}
	
	if (!rank)
		std::cout << "Total unknowns: " << vars.size() << std::endl;

	std::vector<tet> tets(nT);
	std::vector<point> pts(nP);
	std::vector<double> kappa(nT);
	std::vector<double> Ip(nT);

	for (int i = 0; i < nT; i++) {
		const tetrahedron &tet = m.tets(i);
		if (tet.color() == 1) {
			kappa[i] = 11;
			Ip[i] = 1;
		} else {
			kappa[i] = 0.1;
			Ip[i] = 0;
		}
		for (int j = 0; j < 4; j++) {
			tets[i].p[j] = tet.p(j).idx();
			const face &f = tet.f(j).flip();
			if (f.is_border())
				tets[i].neib[j] = -1;
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

	point w(omega.x, omega.y, omega.z);
	
	struct slae_row {
		double beta;
		double alpha[3];
		int cols[3];
	};

	std::vector<slae_row> slae(vars.size());

	int count = 2;
	MPI::Datatype oldtypes[2] = {MPI::DOUBLE, MPI::INT};
	int blockcounts[2] = {4, 3};
	MPI::Aint offsets[2] = {0, 4 * sizeof(double)};
	MPI::Datatype SLAE_ROW = MPI::Datatype::Create_struct(count, blockcounts, offsets, oldtypes);
	SLAE_ROW.Commit();

	for (int j = 0; j < iface.size(); j++) {
		if (owner[j] != rank)
			continue;
		int itet = intet[j];
		int face;
		int i = iface[j].first;
		point r(pts[i]);
		double a = 1, b = 0;
		int vout[3];
		double wei[3];
		while (true) {
			face = -1;
			double len = trace(w, tets[itet], r, face, &pts[0]);
			double delta = len * kappa[itet];
			double q = exp(-delta);
			b += a * Ip[itet] * (1 - q);
			a *= q;
			const tet &old = tets[itet];
			itet = old.neib[face];
			if (itet == -1) {
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
			MESH3D_ASSERT(varno[vout[k]] > -2);

		int row = varno[i];

		slae[row].beta = b;
		for (int k = 0; k < 3; k++) {
			slae[row].alpha[k] = wei[k] * a;
			slae[row].cols[k] = varno[vout[k]];
		}
	}

	MPI::COMM_WORLD.Gatherv(&slae[starts[rank]], numvars[rank], SLAE_ROW, &slae[0], 
		&numvars[0], &starts[0], SLAE_ROW, 0);

	std::vector<double> sol(slae.size());
	if (!rank && slae.size() > 0) {
		int m = slae.size();
		std::vector<int> Ap(slae.size() + 1);
		Ap[0] = 0;
		std::vector<double> Ax;
		std::vector<int> Ai;
		for (int i = 0; i < m; i++) {
			std::vector<std::pair<int, double>> row;
			row.push_back(std::pair<int, double>(i, -1));
			for (int j = 0; j < 3; j++)
				if (slae[i].cols[j] != -1)
					row.push_back(std::pair<int, double>(slae[i].cols[j], slae[i].alpha[j]));
			std::sort(row.begin(), row.end(), [](const std::pair<int, double> &a, const std::pair<int, double> &b) { return a.first < b.first; } );
			Ap[i + 1] = Ap[i] + row.size();
			for (auto it = row.begin(); it != row.end(); it++) {
				Ai.push_back(it->first);
				Ax.push_back(it->second);
			}
		}
		void *symbolic;
		void *numeric;
		if (umfpack_di_symbolic(m, m, &Ap[0], &Ai[0], 0, &symbolic, 0, 0) != UMFPACK_OK) {
			std::cerr << "Umfpack failed to perform symbolic decomposition" << std::endl;
			MPI::COMM_WORLD.Abort(0);
		}
		if (umfpack_di_numeric(&Ap[0], &Ai[0], &Ax[0], symbolic, &numeric, 0, 0) != UMFPACK_OK) {
			std::cerr << "Umfpack failed to perform numeric decomposition" << std::endl;
			MPI::COMM_WORLD.Abort(0);
		}
		std::vector<double> rhs(m);
		for (int i = 0; i < m; i++)
			rhs[i] = -slae[i].beta;
		if (umfpack_di_solve(UMFPACK_At, &Ap[0], &Ai[0], &Ax[0], &sol[0], &rhs[0], numeric, 0, 0) != UMFPACK_OK) {
			std::cerr << "Umfpack failed to perform numeric decomposition" << std::endl;
			MPI::COMM_WORLD.Abort(0);
		}
		std::cout << "Solve ok!" << std::endl;

		double norm = 0;
		for (int i = 0; i < m; i++) {
			double res = sol[i] - slae[i].beta;
			for (int j = 0; j < 3; j++) {
				int col = slae[i].cols[j];
				if (col != -1)
					res -= slae[i].alpha[j] * sol[col];
			}
			
			if (fabs(res) > norm)
				norm = fabs(res);
		}

		std::cout << "Error norm : " << norm << std::endl;
	}
	MPI::COMM_WORLD.Bcast(&sol[0], sol.size(), MPI::DOUBLE, 0);

	std::vector<double> u(nP, 0);

	for (int i = 0; i < nP; i++) {
		int j = varno[i];
		if (j < 0)
			continue;
		u[i] = sol[j];
	}
	
	for (int i = 0; i < nP; i++) {
		if (varno[i] >= 0)
			continue;
		int itet = m.vertices(i).tetrahedrons().front().t->idx();
		int face;
		point r(pts[i]);
		double a = 1, b = 0;
		int vout[3];
		double wei[3];
		while (true) {
			face = -1;
			double len = trace(w, tets[itet], r, face, &pts[0]);
			double delta = len * kappa[itet];
			double q = exp(-delta);
			b += a * Ip[itet] * (1 - q);
			a *= q;
			const tet &old = tets[itet];
			itet = old.neib[face];
			if (itet == -1) {
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

	double stop = MPI::Wtime();

	if (!rank)
		std::cout << "Time spent: " << ((stop - start) * 1e3) << "ms" << std::endl;

	vtk_stream v(tiny::format("sol.%d.vtk", rank).c_str());
	v.write_header(m, "Solution");
	v.append_cell_data(&kappa[0], "kappa");
	v.append_cell_data(&Ip[0], "Ip");
	v.append_point_data(&u[0], "u");
	v.close();

	MPI::Finalize();
	return 0;
}
