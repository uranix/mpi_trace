__global__ void foobar(float *a) {
    a[threadIdx.x] = blockIdx.x;
}
