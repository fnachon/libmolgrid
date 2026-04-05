/*
 * common.h
 *
 * Utility functions and definitions.
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <string>

#include "libmolgrid/config.h"

#if LIBMOLGRID_USE_CUDA
#include <cuda_runtime_api.h>
#include <cuda.h>
#endif

#if LIBMOLGRID_USE_CUDA && defined(__CUDACC__)
#define CUDA_CALLABLE_MEMBER __host__ __device__
#define CUDA_DEVICE_MEMBER __device__
#else
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE_MEMBER
#endif

#if !LIBMOLGRID_USE_CUDA
struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct int2 { int x, y; };
struct int3 { int x, y, z; };
struct uint2 { unsigned int x, y; };
struct uint3 { unsigned int x, y, z; };
struct double3 { double x, y, z; };

inline float2 make_float2(float x, float y) { return {x, y}; }
inline float3 make_float3(float x, float y, float z) { return {x, y, z}; }
inline float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }
inline int2 make_int2(int x, int y) { return {x, y}; }
inline int3 make_int3(int x, int y, int z) { return {x, y, z}; }
inline uint2 make_uint2(unsigned x, unsigned y) { return {x, y}; }
inline uint3 make_uint3(unsigned x, unsigned y, unsigned z) { return {x, y, z}; }
inline double3 make_double3(double x, double y, double z) { return {x, y, z}; }
#endif

#if LIBMOLGRID_USE_CUDA
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); i += blockDim.x * gridDim.x)
#else
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = 0; i < static_cast<int>(n); i++)
#endif

#define LMG_CUDA_NUM_THREADS 512
#define LMG_CUDA_BLOCKDIM 8
#define LOG2_WARP_SIZE 5U
#define WARP_SIZE (1U << LOG2_WARP_SIZE)

#define LMG_GET_BLOCKS(N) ((unsigned(N) + LMG_CUDA_NUM_THREADS - 1) / LMG_CUDA_NUM_THREADS)
#define LMG_GET_THREADS(N) ((unsigned(N) < LMG_CUDA_NUM_THREADS) ? unsigned(N) : LMG_CUDA_NUM_THREADS)

#if LIBMOLGRID_USE_CUDA && !defined(__CUDA_ARCH__)
#define LMG_CUDA_CHECK(condition) \
  do { \
    cudaError_t error = condition; \
    if(error != cudaSuccess) { \
      std::cerr << __FILE__ << ":" << __LINE__ << ": " << cudaGetErrorString(error); \
      throw std::runtime_error(std::string("CUDA Error: ") + cudaGetErrorString(error)); \
    } \
  } while (0)
#elif LIBMOLGRID_USE_CUDA
#define LMG_CUDA_CHECK(condition) condition
#else
#define LMG_CUDA_CHECK(condition) (condition)
#endif

inline std::string itoa(size_t v) { return std::to_string(v); }

#endif /* COMMON_H_ */
