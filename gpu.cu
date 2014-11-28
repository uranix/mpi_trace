#include "gpu.h"

#include "MeshView.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>

#define CUDA_CHECK(x) do { cudaError_t __err = (x);\
    if (__err != cudaSuccess) { std::cerr << __FILE__ << ":" << __LINE__ << " CUDA call `" << #x \
        << "' failed with error `" << cudaGetErrorString(__err) << std::endl; abort(); } \
} while (false)

GPUMeshView::GPUMeshView(MeshView &mv) {
    nP = mv.pts.size();
    int nT = mv.tets.size();

    CUDA_CHECK(cudaMalloc(&pts,    nP * sizeof(point)));
    CUDA_CHECK(cudaMalloc(&anyTet, nP * sizeof(int)  ));
    CUDA_CHECK(cudaMalloc(&tets,   nT * sizeof(tet)  ));
    CUDA_CHECK(cudaMalloc(&kappa,  nT * sizeof(real) ));
    CUDA_CHECK(cudaMalloc(&Ip,     nT * sizeof(real) ));
}

GPUMeshView::~GPUMeshView() {
    CUDA_CHECK(cudaFree(pts));
    CUDA_CHECK(cudaFree(anyTet));
    CUDA_CHECK(cudaFree(tets));
    CUDA_CHECK(cudaFree(kappa));
    CUDA_CHECK(cudaFree(Ip));
}

GPUAverageSolution::GPUAverageSolution(const MeshView &mv) : U(nP) {
    nP = mv.pts.size();
    CUDA_CHECK(cudaMalloc(&Udev, nP * sizeof(real)));
}

GPUAverageSolution::~GPUAverageSolution() {
    CUDA_CHECK(cudaFree(Udev));
}

std::vector<double> &GPUAverageSolution::retrieve() {
    std::vector<real> Uhost(nP);
    CUDA_CHECK(cudaMemcpy(Uhost.data(), Udev, nP * sizeof(real), cudaMemcpyDeviceToHost));
    std::copy(Uhost.begin(), Uhost.end(), U.begin());
    return U;
}
