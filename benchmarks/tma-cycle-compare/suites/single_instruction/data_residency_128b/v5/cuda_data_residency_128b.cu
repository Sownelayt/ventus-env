// ColdThenHotV1 payload controls.  One untimed TMA primes the exact TensorMap
// before both measured commands, so cold/hot here refers only to the 128B line.

#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

using Barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(expr) do { cudaError_t s_ = (expr); if (s_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA_ERROR,%s,%d,%s\n", #expr, __LINE__, cudaGetErrorString(s_)); std::exit(1); } } while (0)
#define CU_CHECK(expr) do { CUresult s_ = (expr); if (s_ != CUDA_SUCCESS) { \
  const char* t_ = nullptr; cuGetErrorString(s_, &t_); std::fprintf(stderr, "CU_ERROR,%s,%d,%s\n", #expr, __LINE__, t_ ? t_ : "unknown"); std::exit(1); } } while (0)

namespace {
constexpr int kBytes = 128;
constexpr int kThreads = 128;
constexpr int kRepeats = 2;
constexpr int kMapPageBytes = 4096;
constexpr unsigned char kValue = 0x6d;
constexpr size_t kMaxTrash = 256ull * 1024 * 1024;

struct Sample {
  unsigned long long cycles;
  unsigned long long issue;
  unsigned long long prep;
  unsigned int polls;
  unsigned int errors;
  unsigned int smid;
};

uint64_t fingerprint64(const void* data, size_t bytes) {
  const auto* input = static_cast<const unsigned char*>(data);
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= input[index];
    value *= 1099511628211ull;
  }
  return value;
}

__device__ __forceinline__ unsigned int load_cg(const void* pointer) {
  unsigned int value;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
  return value;
}
__device__ __forceinline__ unsigned int smid() { unsigned int v; asm("mov.u32 %0, %%smid;" : "=r"(v)); return v; }
__device__ __forceinline__ void acquire_map(const CUtensorMap* map) { asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"(map) : "memory"); }

__device__ __forceinline__ unsigned long long arrive_token(Barrier* barrier) {
  unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned long long token;
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];" : "=l"(token) : "r"(address) : "memory");
  return token;
}
__device__ __forceinline__ bool test_wait(Barrier* barrier, unsigned long long token) {
  unsigned int address = static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned int done;
  asm volatile("{ .reg .pred p; mbarrier.test_wait.acquire.cta.shared::cta.b64 p, [%1], %2; selp.b32 %0, 1, 0, p; }"
               : "=r"(done) : "r"(address), "l"(token) : "memory");
  return done != 0;
}

template <bool Tensor, bool G2S>
__global__ void residency_kernel(
    const CUtensorMap* map, const unsigned char* source,
    unsigned char* destination, const unsigned int* trash,
    size_t trash_words, int scenario, Sample* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tile = storage;
  auto* barrier = reinterpret_cast<Barrier*>(storage + kBytes);
  auto* sums = reinterpret_cast<unsigned long long*>(storage + kBytes + sizeof(Barrier));
  for (int byte = threadIdx.x; byte < kBytes; byte += blockDim.x)
    tile[byte] = G2S ? 0 : kValue;
  if (threadIdx.x == 0 && G2S) init(barrier, 1);
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();

  if constexpr (Tensor) {
    if (threadIdx.x == 0) {
      sums[0] = 0;
      acquire_map(map);
      const auto* words = reinterpret_cast<const unsigned int*>(map);
#pragma unroll
      for (int word = 0; word < 32; ++word)
        sums[0] += load_cg(words + word);
      if constexpr (G2S) {
        cuda::device::barrier_expect_tx(*barrier, kBytes);
        int32_t coordinates[2] = {0, 0};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_shared, cuda::ptx::space_global, tile, map,
            coordinates, cuda::device::barrier_native_handle(*barrier));
        const auto token = arrive_token(barrier);
        while (!test_wait(barrier, token)) {}
      } else {
        int32_t coordinates[2] = {0, 0};
        cuda::ptx::cp_async_bulk_tensor(
            cuda::ptx::space_global, cuda::ptx::space_shared, map,
            coordinates, tile);
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      }
    }
  }
  __syncthreads();

  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    // scenario 0: cold then immediate hot; 1: cold before both; 2: hot before both.
    const bool make_cold = scenario == 1 || (scenario == 0 && repeat == 0);
    const bool make_hot = scenario == 2;
    unsigned long long sum = 0;
    if (make_cold)
      for (size_t word = threadIdx.x; word < trash_words; word += blockDim.x)
        sum = sum * 0x9e3779b97f4a7c15ull + load_cg(trash + word);
    if (make_hot) {
      const auto* target = reinterpret_cast<const unsigned int*>(G2S ? source : destination);
      for (int word = threadIdx.x; word < kBytes / 4; word += blockDim.x)
        sum += load_cg(target + word);
    }
    sums[threadIdx.x] = sum;
    __syncthreads();

    Sample current{};
    if (threadIdx.x == 0) {
      for (int thread = 0; thread < blockDim.x; ++thread)
        current.prep ^= sums[thread] + thread;
      const auto begin = clock64();
      if constexpr (G2S) {
        cuda::device::barrier_expect_tx(*barrier, kBytes);
        if constexpr (Tensor) {
          int32_t coordinates[2] = {0, 0};
          cuda::ptx::cp_async_bulk_tensor(
              cuda::ptx::space_shared, cuda::ptx::space_global, tile, map,
              coordinates, cuda::device::barrier_native_handle(*barrier));
        } else {
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared, cuda::ptx::space_global, tile, source,
              kBytes, cuda::device::barrier_native_handle(*barrier));
        }
        const auto issued = clock64();
        const auto token = arrive_token(barrier);
        do { ++current.polls; } while (!test_wait(barrier, token));
        current.issue = issued - begin;
      } else {
        if constexpr (Tensor) {
          int32_t coordinates[2] = {0, 0};
          cuda::ptx::cp_async_bulk_tensor(
              cuda::ptx::space_global, cuda::ptx::space_shared, map,
              coordinates, tile);
        } else {
          cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                   cuda::ptx::space_shared, destination,
                                   tile, kBytes);
        }
        const auto issued = clock64();
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        current.issue = issued - begin;
      }
      current.cycles = clock64() - begin;
      if constexpr (G2S)
        for (int byte = 0; byte < kBytes; ++byte)
          current.errors += tile[byte] != kValue;
      current.smid = smid();
      output[repeat] = current;
    }
    __syncthreads();
  }
}

CUtensorMap encode(unsigned char* base) {
  CUtensorMap map{};
  const cuuint64_t dims[2] = {16, 8};
  const cuuint64_t strides[1] = {16};
  const cuuint32_t box[2] = {16, 8};
  const cuuint32_t element[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box,
      element, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}
}  // namespace

int main(int argc, char** argv) {
  const char* selected_path = argc > 1 ? argv[1] : "all";
  const char* selected_direction = argc > 2 ? argv[2] : "all";
  if (std::strcmp(selected_path, "all") != 0 &&
      std::strcmp(selected_path, "bulk") != 0 &&
      std::strcmp(selected_path, "tensor") != 0) return 2;
  if (std::strcmp(selected_direction, "all") != 0 &&
      std::strcmp(selected_direction, "g2s") != 0 &&
      std::strcmp(selected_direction, "s2g") != 0) return 2;
  CUDA_CHECK(cudaSetDevice(0));
  CU_CHECK(cuInit(0));
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  const size_t path_count = std::strcmp(selected_path, "all") == 0 ? 2 : 1;
  const size_t direction_count =
      std::strcmp(selected_direction, "all") == 0 ? 2 : 1;
  const size_t samples = path_count * direction_count * 3;
  unsigned char *source_pool = nullptr, *destination_pool = nullptr;
  CUDA_CHECK(cudaMalloc(&source_pool, samples * kBytes));
  CUDA_CHECK(cudaMalloc(&destination_pool, samples * kBytes));
  CUDA_CHECK(cudaMemset(source_pool, kValue, samples * kBytes));
  const size_t map_bytes = samples * kMapPageBytes + kMapPageBytes - 1;
  unsigned char* raw_maps = nullptr;
  CUDA_CHECK(cudaMalloc(&raw_maps, map_bytes));
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw_maps) + kMapPageBytes - 1) &
      ~(static_cast<uintptr_t>(kMapPageBytes) - 1);
  auto* map_pages = reinterpret_cast<unsigned char*>(aligned);
  const size_t trash_bytes = std::min(
      kMaxTrash, std::max<size_t>(64ull << 20,
      static_cast<size_t>(property.l2CacheSize) * 2));
  unsigned int* trash = nullptr;
  CUDA_CHECK(cudaMalloc(&trash, trash_bytes));
  CUDA_CHECK(cudaMemset(trash, 0x3c, trash_bytes));
  Sample* device_sample = nullptr;
  CUDA_CHECK(cudaMalloc(&device_sample, kRepeats * sizeof(Sample)));

  std::puts("gpu,cc,case_id,path,direction,wait,scenario,data_state,l2_bytes,repeat,pair_role,total_cycles,issue_cycles,polls,errors,smid,preparation_checksum,descriptor_state,descriptor_va,descriptor_fingerprint64,prior_tma_use,descriptor_l2_warm_method,tma_prefetch_before_issue");
  size_t sample_index = 0;
  const size_t shared = kBytes + sizeof(Barrier) +
                        kThreads * sizeof(unsigned long long);
  for (const char* path : {"bulk", "tensor"})
    for (const char* direction : {"g2s", "s2g"})
      for (int scenario = 0; scenario < 3; ++scenario)
      {
          if (std::strcmp(selected_path, "all") != 0 &&
              std::strcmp(selected_path, path) != 0) continue;
          if (std::strcmp(selected_direction, "all") != 0 &&
              std::strcmp(selected_direction, direction) != 0) continue;
          const bool tensor = std::strcmp(path, "tensor") == 0;
          const bool g2s = std::strcmp(direction, "g2s") == 0;
          unsigned char* source = source_pool + sample_index * kBytes;
          unsigned char* destination = destination_pool + sample_index * kBytes;
          CUDA_CHECK(cudaMemset(destination, 0, kBytes));
          CUtensorMap* map = nullptr;
          uint64_t fingerprint = 0;
          if (tensor) {
            CUtensorMap host_map = encode(g2s ? source : destination);
            fingerprint = fingerprint64(&host_map, sizeof(host_map));
            map = reinterpret_cast<CUtensorMap*>(
                map_pages + sample_index * kMapPageBytes);
            CUDA_CHECK(cudaMemcpy(map, &host_map, sizeof(host_map),
                                  cudaMemcpyHostToDevice));
          }
          if (tensor && g2s)
            residency_kernel<true, true><<<1, kThreads, shared>>>(
                map, source, destination, trash, trash_bytes / 4, scenario, device_sample);
          else if (tensor)
            residency_kernel<true, false><<<1, kThreads, shared>>>(
                map, source, destination, trash, trash_bytes / 4, scenario, device_sample);
          else if (g2s)
            residency_kernel<false, true><<<1, kThreads, shared>>>(
                nullptr, source, destination, trash, trash_bytes / 4, scenario, device_sample);
          else
            residency_kernel<false, false><<<1, kThreads, shared>>>(
                nullptr, source, destination, trash, trash_bytes / 4, scenario, device_sample);
          CUDA_CHECK(cudaGetLastError());
          CUDA_CHECK(cudaDeviceSynchronize());
          std::array<Sample, kRepeats> results{};
          CUDA_CHECK(cudaMemcpy(results.data(), device_sample, sizeof(results),
                                cudaMemcpyDeviceToHost));
          if (!g2s) {
            unsigned char host[kBytes];
            CUDA_CHECK(cudaMemcpy(host, destination, kBytes,
                                  cudaMemcpyDeviceToHost));
            for (unsigned char value : host) results.back().errors += value != kValue;
          }
          for (int repeat = 0; repeat < kRepeats; ++repeat) {
          const Sample& result = results[repeat];
          const char* scenario_name = scenario == 0 ? "cold_hot" : scenario == 1 ? "cold_cold" : "hot_hot";
          const bool cold = scenario == 1 || (scenario == 0 && repeat == 0);
          std::printf("%s,%d.%d,residency_%s_%s_%s,%s,%s,%s,%s,%s,%zu,%d,%s,%llu,%llu,%u,%u,%u,%llu,%s,0x%016" PRIx64 ",0x%016" PRIx64 ",%d,%s,%d\n",
                      property.name, property.major, property.minor, path, direction, scenario_name, path,
                      direction, g2s ? "test_wait" : "wait_group", scenario_name,
                      cold ? "cold" : "hot", trash_bytes, repeat,
                      cold ? "cold" : "hot", result.cycles, result.issue,
                      result.polls, result.errors, result.smid, result.prep,
                      tensor ? "tmau_hot_after_untimed_tma_primer" : "not_applicable",
                      reinterpret_cast<uint64_t>(map), fingerprint, tensor ? repeat + 1 : repeat,
                      tensor ? "ld.global.cg+untimed_same_map_tma_primer" : "not_applicable",
                      0);
          }
          ++sample_index;
      }
  if (sample_index != samples) return 2;
  CUDA_CHECK(cudaFree(device_sample));
  CUDA_CHECK(cudaFree(trash));
  CUDA_CHECK(cudaFree(raw_maps));
  CUDA_CHECK(cudaFree(destination_pool));
  CUDA_CHECK(cudaFree(source_pool));
  return 0;
}
