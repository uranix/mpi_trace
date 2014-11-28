#include "gpu.h"
#include "kernels.h"

#include "MeshView.h"

#include <iostream>
#include <cstdlib>
#include <algorithm>
#include <cassert>

#include <cublas.h>

#define CUDA_CHECK(x) do { cudaError_t __err = (x);\
    if (__err != cudaSuccess) { std::cerr << __FILE__ << ":" << __LINE__ << " CUDA call `" << #x \
        << "' failed with error `" << cudaGetErrorString(__err) << "'" << std::endl; abort(); } \
} while (false)

GPUMeshView::GPUMeshView(int device, MeshView &mv) {
    cudaSetDevice(device);

    nP = mv.pts.size();
    int nT = mv.tets.size();

    CUDA_CHECK(cudaMalloc(&pts,    nP * sizeof(point)));
    CUDA_CHECK(cudaMalloc(&anyTet, nP * sizeof(int)  ));
    CUDA_CHECK(cudaMalloc(&tets,   nT * sizeof(tet)  ));
    CUDA_CHECK(cudaMalloc(&kappa,  nT * sizeof(real) ));
    CUDA_CHECK(cudaMalloc(&Ip,     nT * sizeof(real) ));

    CUDA_CHECK(cudaMemcpy(pts,    mv.pts   .data(), nP * sizeof(point), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(anyTet, mv.anyTet.data(), nP * sizeof(int),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(tets,   mv.tets  .data(), nT * sizeof(tet),   cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(kappa,  mv.kappa .data(), nT * sizeof(real),  cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(Ip,     mv.Ip    .data(), nT * sizeof(real),  cudaMemcpyHostToDevice));
}

GPUMeshView::~GPUMeshView() {
    CUDA_CHECK(cudaFree(pts));
    CUDA_CHECK(cudaFree(anyTet));
    CUDA_CHECK(cudaFree(tets));
    CUDA_CHECK(cudaFree(kappa));
    CUDA_CHECK(cudaFree(Ip));
}

GPUAverageSolution::GPUAverageSolution(const GPUMeshView &gmv) : nP(gmv.nP), U(nP) {
    cublasStatus status = cublasInit();
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS init failed with status " << status << std::endl;
        abort();
    }
    CUDA_CHECK(cudaMalloc(&Udev, nP * sizeof(real)));
    CUDA_CHECK(cudaMemset(Udev, 0, nP * sizeof(real)));
}

GPUAverageSolution::~GPUAverageSolution() {
    CUDA_CHECK(cudaFree(Udev));
    cublasShutdown();
}

std::vector<double> &GPUAverageSolution::retrieve() {
    std::vector<real> Uhost(nP);
    CUDA_CHECK(cudaMemcpy(Uhost.data(), Udev, nP * sizeof(real), cudaMemcpyDeviceToHost));
    std::copy(Uhost.begin(), Uhost.end(), U.begin());
    return U;
}

template<>
void GPUAverageSolution::add<float>(float *Idir, const float wei) {
    assert(sizeof(real) == sizeof(float));
    cublasSaxpy(nP, wei, Idir, 1, (float *)Udev, 1);
}

template<>
void GPUAverageSolution::add<double>(double *Idir, const double wei) {
    assert(sizeof(real) == sizeof(double));
    cublasDaxpy(nP, wei, Idir, 1, (double *)Udev, 1);
}

GPUMultipleDirectionSolver::GPUMultipleDirectionSolver(
        const int maxDirections, const GPUMeshView &mv, const std::vector<point> &ws
)
    : maxDirections(maxDirections), mv(mv)
{
    CUDA_CHECK(cudaMalloc(&Idirs, mv.nP * maxDirections * sizeof(real)));
    CUDA_CHECK(cudaMalloc(&inner, mv.nP * maxDirections * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&w, ws.size() * sizeof(point)));
    CUDA_CHECK(cudaMemcpy(w, ws.data(), ws.size() * sizeof(point), cudaMemcpyHostToDevice));
}

real *GPUMultipleDirectionSolver::Idir(const int direction) {
    assert(direction < maxDirections);
    return Idirs + direction * mv.nP;
}

int *GPUMultipleDirectionSolver::innerFlag(const int direction) {
    assert(direction < maxDirections);
    return inner + direction * mv.nP;
}

GPUMultipleDirectionSolver::~GPUMultipleDirectionSolver() {
    CUDA_CHECK(cudaFree(Idirs));
    CUDA_CHECK(cudaFree(w));
}

void GPUMultipleDirectionSolver::setBoundary(
        const int direction, std::vector<real> &Ihostdir, std::vector<int> &isInner)
{
    CUDA_CHECK(cudaMemcpy(Idir(direction), Ihostdir.data(), mv.nP * sizeof(real), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(innerFlag(direction), isInner.data(), mv.nP * sizeof(int), cudaMemcpyHostToDevice));
}

void GPUMultipleDirectionSolver::traceInterior(const int lo, const int ndir) {
    const int nP = mv.nP;

    dim3 block(256);
    dim3 grid((nP + block.x - 1) / block.x, ndir);

    trace_kernel<<<grid, block>>>(nP, lo, mv, Idirs, inner, w);
    CUDA_CHECK(cudaDeviceSynchronize());
}
