#include "config.h"

#include <iostream>

const char *realstr() {
    return sizeof(real) == 4 ? "float" : "double";
}

struct AlignStats {
    int offs_kappa0;
    int offs_Ip0;
    int offs_v;
    int offs_Teff;
    int offs_Te;
    int offs_dvstep;
    int offs_p;
    int offs_neib;
    int size;
};

#define _(x) ret.offs_ ## x = offsetof(MeshElement, x);

__global__ void test_gpu(AlignStats *_ret) {
    AlignStats &ret = *_ret;
    _(kappa0);
    _(Ip0);
    _(v);
    _(Teff);
    _(Te);
    _(dvstep);
    _(p);
    _(neib);
    ret.size = sizeof(MeshElement);
}

void test_host(AlignStats &ret) {
    _(kappa0);
    _(Ip0);
    _(v);
    _(Teff);
    _(Te);
    _(dvstep);
    _(p);
    _(neib);
    ret.size = sizeof(MeshElement);
}

bool alignment_test() {
    AlignStats host_as;
    AlignStats gpu_as;

    test_host(host_as);
    AlignStats *gpu_as_dev;
    cudaMalloc(&gpu_as_dev, sizeof(AlignStats));
    cudaMemset(&gpu_as_dev, -1, sizeof(AlignStats));
    test_gpu<<<1, 1>>>(gpu_as_dev);
    cudaMemcpy(&gpu_as, gpu_as_dev, sizeof(AlignStats), cudaMemcpyDeviceToHost);

    #define TEST(x) do { \
        bool __test = host_as.x == gpu_as.x; \
        std::cout << "Testing " << #x << ", on host = " \
        << host_as.x << ", on dev = " << gpu_as.x \
        << (__test ? " OK" : " FAILED!") << std::endl; \
        ok &= __test; \
    } while (false)

    bool ok = true;

    TEST(offs_kappa0);
    TEST(offs_Ip0);
    TEST(offs_v);
    TEST(offs_Teff);
    TEST(offs_Te);
    TEST(offs_dvstep);
    TEST(offs_p);
    TEST(offs_neib);
    TEST(size);

    return ok;
}
