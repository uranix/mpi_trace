#include "config.h"

#include <iostream>

const char *realstr() {
    return sizeof(real) == 4 ? "float" : "double";
}

struct AlignStats {
    int offs_kappa;
    int offs_Ip;
    int offs_p;
    int offs_neib;
    int offs__padd;
    int size;
    int quot, rem;
};

#define _(x) ret.offs_ ## x = offsetof(MeshElement, x);

__global__ void test_gpu(AlignStats *_ret) {
    AlignStats &ret = *_ret;
    _(kappa);
    _(Ip);
    _(p);
    _(neib);
    _(_padd);
    ret.size = sizeof(MeshElement);
    int Nreal = sizeof(real) * NFREQ;
    ret.quot = ret.size / Nreal;
    ret.rem = ret.size % Nreal;
}

void test_host(AlignStats &ret) {
    _(kappa);
    _(Ip);
    _(p);
    _(neib);
    _(_padd);
    ret.size = sizeof(MeshElement);
    int Nreal = sizeof(real) * NFREQ;
    ret.quot = ret.size / Nreal;
    ret.rem = ret.size % Nreal;
}

bool alignment_test() {
    std::cout << "struct MeshElement {\n";
    std::cout << "\t" << realstr() << " kappa[" << NFREQ << "];\n";
    std::cout << "\t" << realstr() << " Ip[" << NFREQ << "];\n";
    std::cout << "\tint p[4];\n";
    std::cout << "\tint neib[4];\n";
    std::cout << "\tchar _padd[" << sizeof(MeshElement::_padd) << "];\n";
    std::cout << "};\n";

    AlignStats host_as;
    AlignStats gpu_as;

    test_host(host_as);
    AlignStats *gpu_as_dev;
    cudaMalloc(&gpu_as_dev, sizeof(AlignStats));
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

    TEST(offs_kappa);
    TEST(offs_Ip);
    TEST(offs_p);
    TEST(offs_neib);
    TEST(offs__padd);
    TEST(size);
    TEST(quot);
    TEST(rem);

    return ok;
}
