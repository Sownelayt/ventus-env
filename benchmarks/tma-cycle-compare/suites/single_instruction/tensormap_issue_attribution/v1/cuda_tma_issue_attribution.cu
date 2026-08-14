// Attribute the first-versus-second issue latency of a fixed 2-D TMA command.
//
// Each process creates a fresh CUDA context and launches the same kernel symbol
// exactly once.  The measured TensorMap is made L2-hot and explicitly prefetched
// into the TMA unit before either measured command.  A runtime mode optionally
// executes, at the exact same static PTX instruction site:
//
//   * a predicate-false PTX TMA (front-end/control-flow only), or
//   * a real TMA using a different, already-prefetched TensorMap (full path
//     priming without executing the measured descriptor).
//
// The following two iterations execute the measured descriptor and report the
// issue-instruction and completion latencies independently.

#include <cuda/barrier>
#include <cuda.h>
#include <cuda_runtime.h>

#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

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
constexpr int kMapPageBytes = 4096;
constexpr unsigned long long kPrefetchLead = 1024;
constexpr unsigned char kValue = 0xa5;

enum PrimeMode : int {
  kNoPrime = 0,
  kPredicatedOff = 1,
  kOtherTensorMap = 2,
};

struct Sample {
  unsigned long long issue_cycles;
  unsigned long long completion_cycles;
  unsigned int polls;
  unsigned int errors;
  unsigned int smid;
};

__device__ __forceinline__ unsigned int read_smid() {
  unsigned int value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ unsigned int load_cg(const void* pointer) {
  unsigned int value;
  asm volatile("ld.global.cg.u32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
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
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;"
               :: "l"(map) : "memory");
}

__device__ __forceinline__ void prefetch_map(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

__device__ __forceinline__ void wait_cycles(unsigned long long cycles) {
  const unsigned long long begin = clock64();
  while (clock64() - begin < cycles) asm volatile("" ::: "memory");
}

__device__ __forceinline__ unsigned long long arrive_expect_tx(
    Barrier* barrier, unsigned int bytes) {
  const unsigned int address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned long long token;
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
      : "=l"(token) : "r"(address), "r"(bytes) : "memory");
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

// These raw forms make the PTX predicate explicit and keep one fixed-rank
// static instruction site.  ptxas currently lowers the predicate to a branch
// around one unpredicated UTMALD/UTMAST.  Thus the false iteration warms the
// surrounding instruction-fetch/control-flow path but intentionally does not
// activate the TMAU issue endpoint.  The loop containing the call is not
// unrolled; the PTX/SASS audit verifies both properties after compilation.
__device__ __forceinline__ void issue_tensor_g2s_predicated(
    unsigned int enable, const CUtensorMap* map, void* shared,
    Barrier* barrier) {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared));
  const unsigned int barrier_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  const int coordinate = 0;
  asm volatile(
      "{ .reg .pred issue;\n\t"
      "setp.ne.u32 issue, %5, 0;\n\t"
      "@issue cp.async.bulk.tensor.2d.shared::cta.global.tile."
      "mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];\n\t}"
      :: "r"(shared_address), "l"(map), "r"(coordinate), "r"(coordinate),
         "r"(barrier_address), "r"(enable)
      : "memory");
}

__device__ __forceinline__ void issue_tensor_s2g_predicated(
    unsigned int enable, const CUtensorMap* map, const void* shared) {
  const unsigned int shared_address =
      static_cast<unsigned int>(__cvta_generic_to_shared(shared));
  const int coordinate = 0;
  asm volatile(
      "{ .reg .pred issue;\n\t"
      "setp.ne.u32 issue, %4, 0;\n\t"
      "@issue cp.async.bulk.tensor.2d.global.shared::cta.tile.bulk_group "
      "[%0, {%1, %2}], [%3];\n\t}"
      :: "l"(map), "r"(coordinate), "r"(coordinate), "r"(shared_address),
         "r"(enable)
      : "memory");
}

template <bool G2S>
__global__ void issue_attribution_kernel(
    const CUtensorMap* measured_map, const CUtensorMap* prime_map,
    const unsigned char* measured_source, const unsigned char* prime_source,
    unsigned char* measured_destination, unsigned char* prime_destination,
    int prime_mode, Sample* samples, unsigned long long* preparation_sink) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tile = storage;
  Barrier* barrier = reinterpret_cast<Barrier*>(storage + kBytes);

  for (int byte = threadIdx.x; byte < kBytes; byte += blockDim.x)
    tile[byte] = G2S ? 0 : kValue;
  __syncthreads();
  if (threadIdx.x != 0) return;

  acquire_map(measured_map);
  acquire_map(prime_map);
  unsigned long long preparation = reheat_128(measured_map, 0);
  preparation = reheat_128(prime_map, preparation);
  preparation = reheat_128(G2S ? static_cast<const void*>(measured_source)
                               : static_cast<const void*>(measured_destination),
                             preparation);
  preparation = reheat_128(G2S ? static_cast<const void*>(prime_source)
                               : static_cast<const void*>(prime_destination),
                             preparation);
  prefetch_map(measured_map);
  prefetch_map(prime_map);
  wait_cycles(kPrefetchLead);
  *preparation_sink = preparation;

  // Runtime trip count retains one static instruction site in every mode.
  const int start = prime_mode == kNoPrime ? 0 : -1;
#pragma unroll 1
  for (int iteration = start; iteration < 2; ++iteration) {
    const bool is_prime = iteration < 0;
    const unsigned int enable =
        is_prime && prime_mode == kPredicatedOff ? 0u : 1u;
    const CUtensorMap* selected_map =
        is_prime && prime_mode == kOtherTensorMap ? prime_map : measured_map;
    Sample sample{};

    unsigned long long token = 0;
    if constexpr (G2S) {
      init(barrier, 1);
      if (enable) token = arrive_expect_tx(barrier, kBytes);
    }

    const unsigned long long begin = clock64();
    if constexpr (G2S)
      issue_tensor_g2s_predicated(enable, selected_map, tile, barrier);
    else
      issue_tensor_s2g_predicated(enable, selected_map, tile);
    const unsigned long long issued = clock64();

    if (enable) {
      if constexpr (G2S) {
        do { ++sample.polls; } while (!test_wait(barrier, token));
      } else {
        asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        asm volatile("cp.async.bulk.wait_group 0;" ::: "memory");
      }
    }
    sample.issue_cycles = issued - begin;
    sample.completion_cycles = clock64() - begin;
    sample.smid = read_smid();

    if (is_prime) {
      samples[2] = sample;
    } else {
      if constexpr (G2S) {
        const volatile unsigned char* observed = tile;
        for (int byte = 0; byte < kBytes; ++byte)
          sample.errors += observed[byte] != kValue;
      }
      samples[iteration] = sample;
    }
  }
}

CUtensorMap encode_map(unsigned char* base) {
  CUtensorMap map{};
  cuuint64_t dimensions[2] = {128, 1};
  cuuint64_t strides[1] = {128};
  cuuint32_t box[2] = {128, 1};
  cuuint32_t element_strides[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(
      &map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dimensions, strides,
      box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE,
      CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE,
      CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE));
  return map;
}

const char* mode_name(int mode) {
  switch (mode) {
    case kNoPrime: return "none";
    case kPredicatedOff: return "predicated_off_same_site";
    case kOtherTensorMap: return "other_tensormap_same_site";
    default: return "invalid";
  }
}

int parse_mode(int argc, char** argv) {
  if (argc != 2) {
    std::fprintf(stderr, "usage: %s none|predicated_off|other_tensormap\n", argv[0]);
    std::exit(2);
  }
  if (!std::strcmp(argv[1], "none")) return kNoPrime;
  if (!std::strcmp(argv[1], "predicated_off")) return kPredicatedOff;
  if (!std::strcmp(argv[1], "other_tensormap")) return kOtherTensorMap;
  std::fprintf(stderr, "invalid mode: %s\n", argv[1]);
  std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
  const int prime_mode = parse_mode(argc, argv);
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  CU_CHECK(cuInit(0));

  unsigned char* source = nullptr;
  unsigned char* destination = nullptr;
  unsigned char* descriptor_allocation = nullptr;
  Sample* device_samples = nullptr;
  unsigned long long* device_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&source, 2 * kBytes));
  CUDA_CHECK(cudaMalloc(&destination, 2 * kBytes));
  CUDA_CHECK(cudaMemset(source, kValue, 2 * kBytes));
  CUDA_CHECK(cudaMemset(destination, 0, 2 * kBytes));
  CUDA_CHECK(cudaMalloc(&descriptor_allocation,
                        2 * kMapPageBytes + kMapPageBytes - 1));
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(descriptor_allocation) + kMapPageBytes - 1) &
      ~(static_cast<uintptr_t>(kMapPageBytes) - 1);
  auto* descriptor_pages = reinterpret_cast<unsigned char*>(aligned);
  CUDA_CHECK(cudaMalloc(&device_samples, 3 * sizeof(Sample)));
  CUDA_CHECK(cudaMalloc(&device_sink, sizeof(unsigned long long)));

  std::puts("gpu,cc,direction,prime_mode,stage,repeat,issue_cycles,completion_cycles,polls,errors,smid,descriptor_state,prefetch_lead_cycles,prior_measured_map_tma_use,static_rank,bytes");
  for (const char* direction : {"g2s", "s2g"}) {
    const bool g2s = !std::strcmp(direction, "g2s");
    CUtensorMap host_measured = encode_map(g2s ? source : destination);
    CUtensorMap host_prime = encode_map(
        g2s ? source + kBytes : destination + kBytes);
    auto* measured_map = reinterpret_cast<CUtensorMap*>(descriptor_pages);
    auto* prime_map = reinterpret_cast<CUtensorMap*>(
        descriptor_pages + kMapPageBytes);
    CUDA_CHECK(cudaMemcpy(measured_map, &host_measured, sizeof(host_measured),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(prime_map, &host_prime, sizeof(host_prime),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(device_samples, 0, 3 * sizeof(Sample)));
    const size_t shared_bytes = kBytes + sizeof(Barrier);
    if (g2s) {
      issue_attribution_kernel<true><<<1, kThreads, shared_bytes>>>(
          measured_map, prime_map, source, source + kBytes, destination,
          destination + kBytes, prime_mode, device_samples, device_sink);
    } else {
      issue_attribution_kernel<false><<<1, kThreads, shared_bytes>>>(
          measured_map, prime_map, source, source + kBytes, destination,
          destination + kBytes, prime_mode, device_samples, device_sink);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::array<Sample, 3> samples{};
    CUDA_CHECK(cudaMemcpy(samples.data(), device_samples, sizeof(samples),
                          cudaMemcpyDeviceToHost));

    if (!g2s) {
      std::array<unsigned char, 2 * kBytes> observed{};
      CUDA_CHECK(cudaMemcpy(observed.data(), destination, observed.size(),
                            cudaMemcpyDeviceToHost));
      for (int byte = 0; byte < kBytes; ++byte)
        samples[1].errors += observed[byte] != kValue;
    }

    auto emit = [&](const char* stage, int repeat, const Sample& sample,
                    int prior_use) {
      std::printf("%s,%d.%d,%s,%s,%s,%d,%llu,%llu,%u,%u,%u,explicit_prefetch_hot,%llu,%d,2,%d\n",
                  property.name, property.major, property.minor, direction,
                  mode_name(prime_mode), stage, repeat, sample.issue_cycles,
                  sample.completion_cycles, sample.polls, sample.errors,
                  sample.smid, kPrefetchLead, prior_use, kBytes);
    };
    if (prime_mode != kNoPrime)
      emit("prime", -1, samples[2], 0);
    emit("measured", 0, samples[0], 0);
    emit("measured", 1, samples[1], 1);
  }

  CUDA_CHECK(cudaFree(device_sink));
  CUDA_CHECK(cudaFree(device_samples));
  CUDA_CHECK(cudaFree(descriptor_allocation));
  CUDA_CHECK(cudaFree(destination));
  CUDA_CHECK(cudaFree(source));
  return 0;
}
