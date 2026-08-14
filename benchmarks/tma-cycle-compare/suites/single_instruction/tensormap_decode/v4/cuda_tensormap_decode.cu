// TensorMap lookup/decode differential probe for H100 and B200.
//
// Every case owns one previously unused 4 KiB TensorMap page.  Two measured
// commands execute consecutively in the same kernel: repeat 0 is the first
// TMA use, and repeat 1 immediately reuses the same address and contents.
// Ordinary ld.global.cg loads make the descriptor bytes L2-hot without
// inserting them into the TMA unit.  Explicit prefetch is a separate control.

#include <cuda/barrier>
#include <cuda/ptx>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cinttypes>
#include <string>
#include <vector>

using Barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(expr) do {                                                \
  cudaError_t status_ = (expr);                                              \
  if (status_ != cudaSuccess) {                                              \
    std::fprintf(stderr, "CUDA_ERROR,%s,%d,%s\n", #expr, __LINE__,         \
                 cudaGetErrorString(status_));                               \
    std::exit(EXIT_FAILURE);                                                 \
  }                                                                          \
} while (0)

#define CU_CHECK(expr) do {                                                  \
  CUresult status_ = (expr);                                                 \
  if (status_ != CUDA_SUCCESS) {                                             \
    const char* text_ = nullptr;                                             \
    cuGetErrorString(status_, &text_);                                       \
    std::fprintf(stderr, "CU_ERROR,%s,%d,%s\n", #expr, __LINE__,           \
                 text_ ? text_ : "unknown");                               \
    std::exit(EXIT_FAILURE);                                                 \
  }                                                                          \
} while (0)

namespace {

constexpr int kBytes = 128;
constexpr int kThreads = 128;
constexpr int kRepeats = 2;
constexpr int kMapPageBytes = 4096;
constexpr unsigned char kValue = 0xa5;

struct Sample {
  unsigned long long total;
  unsigned long long issue;
  unsigned long long preparation;
  unsigned int polls;
  unsigned int errors;
  unsigned int smid;
};

struct Case {
  int rank;
  const char* scenario;
  unsigned long long prefetch_gap;
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

__device__ __forceinline__ unsigned int read_smid() {
  unsigned int value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ unsigned int load_cg(const void* pointer) {
  unsigned int value;
  asm volatile("ld.global.cg.u32 %0, [%1];" : "=r"(value) : "l"(pointer) : "memory");
  return value;
}

__device__ __forceinline__ unsigned long long reheat_128(
    const void* pointer, unsigned long long seed) {
  const auto* words = static_cast<const unsigned int*>(pointer);
#pragma unroll
  for (int word = 0; word < 32; ++word)
    seed = seed * 0x9e3779b97f4a7c15ull + load_cg(words + word);
  return seed;
}

__device__ __forceinline__ void acquire_map(const CUtensorMap* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;" :: "l"(map) : "memory");
}

__device__ __forceinline__ void prefetch_map(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

__device__ __forceinline__ void wait_cycles(unsigned long long cycles) {
  const unsigned long long begin = clock64();
  while (clock64() - begin < cycles) asm volatile("" ::: "memory");
}

__device__ __forceinline__ unsigned long long arrive_token(Barrier* barrier) {
  const unsigned int address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned long long token;
  asm volatile("mbarrier.arrive.release.cta.shared::cta.b64 %0, [%1];"
               : "=l"(token) : "r"(address) : "memory");
  return token;
}

__device__ __forceinline__ bool test_wait(
    Barrier* barrier, unsigned long long token) {
  const unsigned int address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned int complete;
  asm volatile("{ .reg .pred p;\n\t"
               "mbarrier.test_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p; }"
               : "=r"(complete) : "r"(address), "l"(token) : "memory");
  return complete != 0;
}

template <int Rank>
__device__ __forceinline__ void tensor_load(
    const CUtensorMap* map, void* shared, Barrier& barrier) {
  int32_t coordinates[Rank] = {};
  cuda::ptx::cp_async_bulk_tensor(
      cuda::ptx::space_shared, cuda::ptx::space_global, shared, map,
      coordinates, cuda::device::barrier_native_handle(barrier));
}

template <int Rank>
__device__ __forceinline__ void tensor_store(
    const CUtensorMap* map, const void* shared) {
  int32_t coordinates[Rank] = {};
  cuda::ptx::cp_async_bulk_tensor(
      cuda::ptx::space_global, cuda::ptx::space_shared, map, coordinates,
      shared);
}

template <int Rank, bool Tensor, bool G2S, bool Prefetch>
__global__ void probe_kernel(
    const CUtensorMap* map, const unsigned char* source,
    unsigned char* destination, unsigned long long prefetch_gap,
    Sample* samples) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tile = storage;
  auto* barrier = reinterpret_cast<Barrier*>(storage + kBytes);

  for (int byte = threadIdx.x; byte < kBytes; byte += blockDim.x)
    tile[byte] = G2S ? 0 : kValue;
  if (threadIdx.x == 0 && G2S) init(barrier, 1);
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();

  if (threadIdx.x == 0) {
    unsigned long long preparation = 0;
    if constexpr (Tensor) {
      acquire_map(map);
      preparation = reheat_128(map, preparation);
    }
    preparation = reheat_128(G2S ? static_cast<const void*>(source)
                                  : static_cast<const void*>(destination),
                                preparation);
    if constexpr (Tensor && Prefetch) {
      prefetch_map(map);
      wait_cycles(prefetch_gap);
    }

    for (int iteration = 0; iteration < kRepeats; ++iteration) {
      Sample sample{};
      sample.preparation = preparation;
      const unsigned long long begin = clock64();
      if constexpr (G2S) {
        cuda::device::barrier_expect_tx(*barrier, kBytes);
        if constexpr (Tensor)
          tensor_load<Rank>(map, tile, *barrier);
        else
          cuda::ptx::cp_async_bulk(
              cuda::ptx::space_shared, cuda::ptx::space_global, tile,
              source, kBytes, cuda::device::barrier_native_handle(*barrier));
        const unsigned long long issued = clock64();
        const unsigned long long token = arrive_token(barrier);
        do { ++sample.polls; } while (!test_wait(barrier, token));
        sample.issue = issued - begin;
      } else {
        if constexpr (Tensor)
          tensor_store<Rank>(map, tile);
        else
          cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                                   cuda::ptx::space_shared, destination,
                                   tile, kBytes);
        const unsigned long long issued = clock64();
        cuda::ptx::cp_async_bulk_commit_group();
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        sample.issue = issued - begin;
      }
      sample.total = clock64() - begin;
      if constexpr (G2S) {
        const volatile unsigned char* observed = tile;
        for (int byte = 0; byte < kBytes; ++byte)
          sample.errors += observed[byte] != kValue;
      }
      sample.smid = read_smid();
      samples[iteration] = sample;
    }
  }
}

CUtensorMap encode_map(unsigned char* base, int rank) {
  CUtensorMap map{};
  cuuint64_t dimensions[5] = {128, 1, 1, 1, 1};
  cuuint64_t strides[4] = {128, 128, 128, 128};
  cuuint32_t box[5] = {128, 1, 1, 1, 1};
  cuuint32_t element_strides[5] = {1, 1, 1, 1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, rank, base, dimensions, strides,
      box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

template <bool Tensor, bool G2S, bool Prefetch = false>
void launch_rank(
    int rank, const CUtensorMap* map,
    const unsigned char* source, unsigned char* destination,
    unsigned long long gap, Sample* samples) {
  const size_t shared = kBytes + sizeof(Barrier);
  switch (rank) {
    case 1: probe_kernel<1, Tensor, G2S, Prefetch><<<1, kThreads, shared>>>(map, source, destination, gap, samples); break;
    case 2: probe_kernel<2, Tensor, G2S, Prefetch><<<1, kThreads, shared>>>(map, source, destination, gap, samples); break;
    case 3: probe_kernel<3, Tensor, G2S, Prefetch><<<1, kThreads, shared>>>(map, source, destination, gap, samples); break;
    case 4: probe_kernel<4, Tensor, G2S, Prefetch><<<1, kThreads, shared>>>(map, source, destination, gap, samples); break;
    case 5: probe_kernel<5, Tensor, G2S, Prefetch><<<1, kThreads, shared>>>(map, source, destination, gap, samples); break;
  }
}

}  // namespace

int main() {
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  CU_CHECK(cuInit(0));

  std::vector<Case> cases;
  for (int rank = 1; rank <= 5; ++rank) {
    cases.push_back({rank, "cold_then_hot", ~0ull});
    cases.push_back({rank, "explicit_prefetch", 512});
  }
  for (unsigned long long gap : {0ull, 32ull, 64ull, 96ull, 128ull,
                                 192ull, 256ull, 384ull, 512ull})
    cases.push_back({2, "prefetch_lead", gap});

  const size_t tensor_samples = 2 * cases.size();
  const size_t all_samples = 2 * (cases.size() + 1);
  unsigned char* source_pool = nullptr;
  unsigned char* destination_pool = nullptr;
  unsigned char* descriptor_allocation = nullptr;
  unsigned char* descriptor_pages = nullptr;
  CUDA_CHECK(cudaMalloc(&source_pool, all_samples * kBytes));
  CUDA_CHECK(cudaMalloc(&destination_pool, all_samples * kBytes));
  CUDA_CHECK(cudaMalloc(&descriptor_allocation,
                        tensor_samples * kMapPageBytes + kMapPageBytes - 1));
  const uintptr_t descriptor_aligned =
      (reinterpret_cast<uintptr_t>(descriptor_allocation) +
       kMapPageBytes - 1) & ~(static_cast<uintptr_t>(kMapPageBytes) - 1);
  descriptor_pages = reinterpret_cast<unsigned char*>(descriptor_aligned);
  CUDA_CHECK(cudaMemset(source_pool, kValue, all_samples * kBytes));
  Sample* device_samples = nullptr;
  CUDA_CHECK(cudaMalloc(&device_samples, kRepeats * sizeof(Sample)));

  std::puts("gpu,cc,l2_bytes,direction,wait,rank,scenario,prefetch_gap_cycles,repeat,pair_role,total_cycles,issue_cycles,polls,errors,smid,preparation_checksum,descriptor_state,descriptor_va,descriptor_fingerprint64,prior_tma_use,descriptor_l2_warm_method,tma_prefetch_before_issue");
  auto emit = [&](const char* direction, const char* wait, int rank,
                  const char* scenario, unsigned long long gap,
                  int repeat, const char* pair_role, const Sample& sample,
                  const CUtensorMap* map, uint64_t fingerprint,
                  int prior_tma_use, int prefetched) {
      const char* descriptor_state = !map ? "not_applicable"
          : prefetched ? (repeat == 0 ? "prefetched_first_use" : "tmau_hot_reuse")
                       : (repeat == 0 ? "tmau_cold_first_use_l2_hot" : "tmau_hot_reuse");
      std::printf("%s,%d.%d,%zu,%s,%s,%d,%s,%llu,%d,%s,%llu,%llu,%u,%u,%u,%llu,%s,0x%016" PRIx64 ",0x%016" PRIx64 ",%d,%s,%d\n",
                  property.name, property.major, property.minor,
                  static_cast<size_t>(property.l2CacheSize), direction,
                  wait, rank, scenario, gap, repeat, pair_role, sample.total,
                  sample.issue, sample.polls, sample.errors, sample.smid,
                  sample.preparation, descriptor_state, reinterpret_cast<uint64_t>(map),
                  fingerprint, prior_tma_use,
                  map ? "ld.global.cg" : "not_applicable", prefetched);
  };
  size_t data_index = 0, descriptor_index = 0;
  for (const char* direction : {"g2s", "s2g"}) {
    const bool g2s = std::strcmp(direction, "g2s") == 0;
    {
      unsigned char* source = source_pool + data_index * kBytes;
      unsigned char* destination = destination_pool + data_index++ * kBytes;
      CUDA_CHECK(cudaMemset(destination, 0, kBytes));
      if (g2s) launch_rank<false, true>(2, nullptr, source, destination, ~0ull, device_samples);
      else launch_rank<false, false>(2, nullptr, source, destination, ~0ull, device_samples);
      CUDA_CHECK(cudaGetLastError());
      CUDA_CHECK(cudaDeviceSynchronize());
      std::array<Sample, kRepeats> samples{};
      CUDA_CHECK(cudaMemcpy(samples.data(), device_samples, sizeof(samples), cudaMemcpyDeviceToHost));
      for (int repeat = 0; repeat < kRepeats; ++repeat)
      emit(direction, g2s ? "test_wait" : "wait_group", 2, "bulk_hot", 0,
           repeat, repeat == 0 ? "first" : "second", samples[repeat],
           nullptr, 0, repeat, 0);
    }

    for (const Case& test : cases) {
        unsigned char* source = source_pool + data_index * kBytes;
        unsigned char* destination = destination_pool + data_index++ * kBytes;
        CUDA_CHECK(cudaMemset(destination, 0, kBytes));
        CUtensorMap host_map = encode_map(g2s ? source : destination, test.rank);
        const uint64_t fingerprint = fingerprint64(&host_map, sizeof(host_map));
        auto* map = reinterpret_cast<CUtensorMap*>(
            descriptor_pages + descriptor_index * kMapPageBytes);
        ++descriptor_index;
        CUDA_CHECK(cudaMemcpy(map, &host_map, sizeof(host_map), cudaMemcpyHostToDevice));
        const bool prefetched = test.prefetch_gap != ~0ull;
        if (g2s && prefetched)
          launch_rank<true, true, true>(test.rank, map, source, destination,
                                        test.prefetch_gap, device_samples);
        else if (g2s)
          launch_rank<true, true, false>(test.rank, map, source, destination,
                                         test.prefetch_gap, device_samples);
        else if (prefetched)
          launch_rank<true, false, true>(test.rank, map, source, destination,
                                         test.prefetch_gap, device_samples);
        else
          launch_rank<true, false, false>(test.rank, map, source, destination,
                                          test.prefetch_gap, device_samples);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        std::array<Sample, kRepeats> samples{};
        CUDA_CHECK(cudaMemcpy(samples.data(), device_samples, sizeof(samples), cudaMemcpyDeviceToHost));
        if (!g2s) {
          unsigned char host[kBytes];
          CUDA_CHECK(cudaMemcpy(host, destination, kBytes, cudaMemcpyDeviceToHost));
          for (unsigned char value : host) samples.back().errors += value != kValue;
        }
        for (int repeat = 0; repeat < kRepeats; ++repeat)
          emit(direction, g2s ? "test_wait" : "wait_group", test.rank,
               test.scenario, test.prefetch_gap == ~0ull ? 0 : test.prefetch_gap,
               repeat, prefetched ? (repeat == 0 ? "prefetched" : "hot")
                                  : (repeat == 0 ? "cold" : "hot"),
               samples[repeat], map, fingerprint, repeat,
               prefetched && repeat == 0);
    }
  }

  if (descriptor_index != tensor_samples || data_index != all_samples) {
    std::fprintf(stderr, "fresh resource mismatch: descriptor=%zu/%zu data=%zu/%zu\n",
                 descriptor_index, tensor_samples, data_index, all_samples);
    return 2;
  }

  CUDA_CHECK(cudaFree(device_samples));
  CUDA_CHECK(cudaFree(descriptor_allocation));
  CUDA_CHECK(cudaFree(destination_pool));
  CUDA_CHECK(cudaFree(source_pool));
  return 0;
}
