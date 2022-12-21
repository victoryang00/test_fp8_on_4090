#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <cuda.h>
#include <cuda_runtime.h>
#include <climits>
#include <cstdint>
#include <vector>
#include <cuda_fp8.h>

#define FLOAT_TO_BITS(x) (*reinterpret_cast<unsigned int*>(x))
#define BITS_TO_FLOAT(x) (*reinterpret_cast<float*>(x))

__device__ __forceinline__ unsigned int round_bitwise_nearest(__nv_fp8_storage_t target) {
    int man_bits = 5;
    unsigned int mask = (1 << (23 - man_bits)) - 1;
    unsigned int rand_prob = 1 << (23 - man_bits - 1);
    unsigned int add_r = target + rand_prob;
    unsigned int quantized = add_r & ~mask;
    return quantized;
}

__device__ __forceinline__ unsigned int clip_exponent(__nv_fp8_storage_t old_num,
                                                      __nv_fp8_storage_t quantized_num) {
    int man_bits = 5;
    int round_bitwise_nearest = 2;
    int quantized_exponent_store = quantized_num << 1 >> 1 >> 23; // 1 sign bit, 23 mantissa bits
    // int min_exponent_store = -((1 << (exp_bits-1))-1) + 127;
    // int max_exponent_store = ((1 << (exp_bits-1))-1) + 127;
    int min_exponent_store = -(24) + 127;
    int max_exponent_store = (7) + 127;
    if (quantized_exponent_store > max_exponent_store) {
        unsigned int max_man = (unsigned int) -1 << 9 >> 9 >> (23 - man_bits)
                                                 << (23 - man_bits); // 1 sign bit, 8 exponent bits, 1 virtual bit
        unsigned int max_num = ((unsigned int) max_exponent_store << 23) | max_man;
        unsigned int old_sign = old_num >> 31 << 31;
        quantized_num = old_sign | max_num;
    } else if (quantized_exponent_store < min_exponent_store) {
        unsigned int min_num = ((unsigned int) min_exponent_store << 23);
        unsigned int old_sign = old_num >> 31 << 31;
        quantized_num = old_sign | min_num;
    }
    return quantized_num;
}

    __global__ void float_nearest_kernel(float *__restrict__ a, float *o, int size) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < size) {
        auto old_num = __nv_cvt_float_to_fp8(a[index], __NV_NOSAT, __NV_E5M2);
//        unsigned int old_num = FLOAT_TO_BITS(&a[index]);
        unsigned int quantize = round_bitwise_nearest(old_num);
        quantize = clip_exponent(old_num, quantize);
        float quantize_float = BITS_TO_FLOAT(&quantize);
        printf("%d %d %f\n", old_num, quantize, quantize_float);

        o[index] = quantize_float;
    }
}


float *float_quantize_nearest_cuda(float *a) {
    auto o = reinterpret_cast<float *>( malloc(1024 * sizeof(float)));
    float *h_O;
    cudaMalloc((void **) &h_O, 1024);
    cudaMemcpy(o, h_O, 1024, cudaMemcpyHostToDevice);
    int size = 1024;
    int blockSize = 1024;
    int blockNums = (size + blockSize - 1) / blockSize;

    float_nearest_kernel<<<blockNums, blockSize>>>(a,
                                                   o,
                                                   size);
    cudaMemcpy(h_O, o, 1024, cudaMemcpyDeviceToHost);
    cudaFree(h_O);
    return o;

}

int main() {
    auto b = __nv_cvt_float_to_fp8(0.1, __NV_NOSAT, __NV_E5M2);

    auto a = reinterpret_cast<float *> (malloc(1024 * sizeof(float)));
    float *h_A;
    for (int i = 0; i < 1024; i++) {
        a[i] = float(i);
    }
    cudaMalloc((void **) &h_A, 1024);
    cudaMemcpy(a, h_A, 1024, cudaMemcpyHostToDevice);

    auto o = float_quantize_nearest_cuda(a);
    for (int i = 0; i < 1024; i++) {
        printf("%f\n", o[i] );
    }
    cudaFree(h_A);

}