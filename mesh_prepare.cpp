#include <meshProcessor/vol_mesh.h>
#include <meshProcessor/mesh.h>
#include <meshProcessor/mesh_graph.h>
#include <meshProcessor/vtk_stream.h>

#include <tiny/format.h>

#include <iostream>
#include <cstdlib>
#include <fstream>

int main(int argc, char **argv) {
	if (argc != 3) {
		std::cerr << "USAGE: " << argv[0] << " <prefix> <num_parts>\n\tLoads prefix.vol mesh and produces prefix.<id>.m3d parts" << std::endl;
		return 1;
	}
	mesh3d::vol_mesh vm(tiny::format("%s.vol", argv[1]).c_str());
	int numparts = atoi(argv[2]);

	mesh3d::mesh m(vm);
	if (numparts == 1) {
		int i = 0;
		std::ofstream f(tiny::format("%s.%d.m3d", argv[1], i).c_str(), std::ios::out | std::ios::binary);
		bool res = m.check();
		std::cout << "Piece #" << i << " check: " << (res ? "OK" : "failed!") << std::endl;
		m.serialize(f);

		mesh3d::vtk_stream vtk(tiny::format("%s.%d.vtk", argv[1], i).c_str());
		vtk.write_header(m, tiny::format("Piece #%d", 0));
		vtk.close();
		return 0;
	}
	mesh3d::tet_graph tg(m);
	tg.partition(numparts);

	for (int i = 0; i < numparts; i++) {
		std::ofstream f(tiny::format("%s.%d.m3d", argv[1], i).c_str(), std::ios::out | std::ios::binary);
		mesh3d::mesh part(m, i, tg);
		bool res = part.check();
		std::cout << "Piece #" << i << " check: " << (res ? "OK" : "failed!") << std::endl;
		part.serialize(f);

		mesh3d::vtk_stream vtk(tiny::format("%s.%d.vtk", argv[1], i).c_str());
		vtk.write_header(part, tiny::format("Piece #%d", i));
		vtk.close();
	}
	return 0;
}
