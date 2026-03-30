/*
 * common.h
 *
 * Utility functions and definitions for Apple Silicon MPS backend.
 * Replaces the previous CUDA-based common.h.
 */

#ifndef COMMON_H_
#define COMMON_H_

#include <memory>
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <cmath>

// ---- Vector types (replaces CUDA built-ins) ----
// These match the CUDA naming convention so the rest of the code is unchanged.

#ifndef __METAL_VERSION__

struct float2 { float x, y; };
struct float3 { float x, y, z; };
struct float4 { float x, y, z, w; };
struct int2   { int x, y; };
struct int3   { int x, y, z; };
struct uint2  { unsigned int x, y; };
struct uint3  { unsigned int x, y, z; };
struct double3 { double x, y, z; };

inline float2  make_float2(float x, float y)                        { return {x, y}; }
inline float3  make_float3(float x, float y, float z)               { return {x, y, z}; }
inline float4  make_float4(float x, float y, float z, float w)      { return {x, y, z, w}; }
inline int2    make_int2(int x, int y)                               { return {x, y}; }
inline int3    make_int3(int x, int y, int z)                        { return {x, y, z}; }
inline uint2   make_uint2(unsigned x, unsigned y)                    { return {x, y}; }
inline uint3   make_uint3(unsigned x, unsigned y, unsigned z)        { return {x, y, z}; }
inline double3 make_double3(double x, double y, double z)            { return {x, y, z}; }

#endif // __METAL_VERSION__

// ---- Callable member annotation (was __host__ __device__) ----
// On Apple Silicon, all code runs on CPU or Metal GPU; no annotation needed.
#define CUDA_CALLABLE_MEMBER
#define CUDA_DEVICE_MEMBER

// ---- Thread / block size constants (shared with Metal shaders via headers) ----
#define LMG_CUDA_NUM_THREADS 512
#define LMG_CUDA_BLOCKDIM    8
#define LOG2_WARP_SIZE       5U
#define WARP_SIZE            (1U << LOG2_WARP_SIZE)

// ---- Block/thread count helpers ----
#define LMG_GET_BLOCKS(N)  ((unsigned(N) + LMG_CUDA_NUM_THREADS - 1) / LMG_CUDA_NUM_THREADS)
#define LMG_GET_THREADS(N) (unsigned(N) < LMG_CUDA_NUM_THREADS ? unsigned(N) : LMG_CUDA_NUM_THREADS)

// ---- Kernel loop macro (used in CPU fallback paths) ----
// In Metal this pattern is handled by thread_position_in_grid.
#define LMG_CUDA_KERNEL_LOOP(i, n) \
  for (int i = 0; i < static_cast<int>(n); i++)

// ---- Error check (was LMG_CUDA_CHECK) ----
// Metal errors are handled via NSError in metal_context.mm.
// This macro is kept as a no-op placeholder so call sites still compile.
#define LMG_CUDA_CHECK(condition) (condition)

// ---- itoa helper used in error messages ----
#include <string>
inline std::string itoa(size_t v) { return std::to_string(v); }

#endif /* COMMON_H_ */
