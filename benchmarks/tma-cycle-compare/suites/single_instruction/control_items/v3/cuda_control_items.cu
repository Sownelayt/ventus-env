// Protocol-correct CUDA TMA parity benchmark for the Ventus comparison study.
//
// Static tensor maps are passed by value in grid-constant parameter space.  In
// particular, this file deliberately does not copy a CUtensorMap to ordinary
// global memory: doing so would require an explicit tensormap proxy acquire at
// every consuming CTA.  Keeping every descriptor in grid-constant parameter
// space makes the static-map cases both simpler and protocol-correct.

#include <cuda/barrier>
#include <cuda/ptx>
// Frozen control-items v1, derived from tma-refer
// src/tma_ventus_parity_bench_v2.cu.  The smoke entry below deliberately
// executes only the old cooperative-copy and compute-overlap controls whose
// measurements are not covered by the formal common/multi-context matrices.
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <numeric>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using BlockBarrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t cuda_check_status = (expr);                                \
    if (cuda_check_status != cudaSuccess) {                                      \
      std::fprintf(stderr, "CUDA_ERROR,%s,%d,%s,%s\n", #expr, __LINE__,         \
                   cudaGetErrorName(cuda_check_status),                          \
                   cudaGetErrorString(cuda_check_status));                       \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

#define CU_CHECK(expr)                                                          \
  do {                                                                          \
    const CUresult cu_check_status = (expr);                                     \
    if (cu_check_status != CUDA_SUCCESS) {                                       \
      const char *cu_check_name = nullptr;                                       \
      const char *cu_check_text = nullptr;                                       \
      cuGetErrorName(cu_check_status, &cu_check_name);                            \
      cuGetErrorString(cu_check_status, &cu_check_text);                          \
      std::fprintf(stderr, "CU_ERROR,%s,%d,%s,%s\n", #expr, __LINE__,           \
                   cu_check_name ? cu_check_name : "unknown",                   \
                   cu_check_text ? cu_check_text : "unknown");                  \
      std::exit(EXIT_FAILURE);                                                   \
    }                                                                            \
  } while (0)

namespace {

constexpr int kMaxRank = 5;
constexpr int kVectorBytes = 16;
constexpr int kOverlapBytes = 4096;
constexpr int kOverlapOps = 1024;

struct DeviceCoords {
  int32_t value[kMaxRank];
};

struct Options {
  int warmups = 20;
  int repeats = 200;
  std::vector<uint32_t> seeds{101u, 202u, 303u};
  std::string suite = "core";
  bool csv = false;
};

struct CaseSpec {
  std::string id;
  std::string layout = "contiguous";
  int rank = 2;
  int bytes = 4096;
  int element_bytes = 1;
  CUtensorMapDataType data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  std::array<cuuint64_t, kMaxRank> dims{};
  std::array<cuuint64_t, kMaxRank - 1> strides{};
  std::array<cuuint32_t, kMaxRank> box{};
  std::array<cuuint32_t, kMaxRank> element_strides{};
  std::array<int32_t, kMaxRank> coords{};
  std::array<int32_t, kMaxRank> block_steps{};
  CUtensorMapInterleave interleave = CU_TENSOR_MAP_INTERLEAVE_NONE;
  CUtensorMapSwizzle swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  int ctas = 1;
  bool shared_is_linear = true;
};

struct Context {
  Options options;
  cudaDeviceProp prop{};
  std::string gpu;
  std::string cc;
};

__host__ __device__ inline uint8_t pattern_byte(uint32_t seed,
                                                uint64_t index) {
  uint32_t x = static_cast<uint32_t>(index) ^ seed ^
               static_cast<uint32_t>(index >> 32);
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  x ^= x >> 16;
  return static_cast<uint8_t>((x ^ (x >> 8) ^ (x >> 24)) & 0xffu);
}

__device__ __forceinline__ BlockBarrier *barrier_after(uint8_t *smem,
                                                       size_t data_bytes) {
  const uintptr_t raw = reinterpret_cast<uintptr_t>(smem + data_bytes);
  const uintptr_t aligned =
      (raw + alignof(BlockBarrier) - 1) & ~(alignof(BlockBarrier) - 1);
  return reinterpret_cast<BlockBarrier *>(aligned);
}

template <int Rank>
__device__ __forceinline__ void issue_tma_load(
    const CUtensorMap *map, void *smem, const DeviceCoords &coords,
    BlockBarrier &barrier) {
  int32_t c[Rank];
#pragma unroll
  for (int i = 0; i < Rank; ++i) c[i] = coords.value[i];
  cuda::ptx::cp_async_bulk_tensor(
      cuda::ptx::space_shared, cuda::ptx::space_global, smem, map, c,
      cuda::device::barrier_native_handle(barrier));
}

template <int Rank>
__device__ __forceinline__ void issue_tma_store(
    const CUtensorMap *map, const void *smem, const DeviceCoords &coords) {
  int32_t c[Rank];
#pragma unroll
  for (int i = 0; i < Rank; ++i) c[i] = coords.value[i];
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global,
                                  cuda::ptx::space_shared, map, c, smem);
}

__device__ __forceinline__ DeviceCoords block_coords(
    DeviceCoords base, DeviceCoords step) {
#pragma unroll
  for (int i = 0; i < kMaxRank; ++i)
    base.value[i] += static_cast<int32_t>(blockIdx.x) * step.value[i];
  return base;
}

template <int Rank>
__global__ void tma_g2s_kernel(
    const __grid_constant__ CUtensorMap map, int bytes, int iterations,
    DeviceCoords base, DeviceCoords step, unsigned long long *cycles,
    uint8_t *capture) {
  extern __shared__ __align__(128) uint8_t smem[];
  BlockBarrier *barrier = barrier_after(smem, static_cast<size_t>(bytes));
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  const DeviceCoords coords = block_coords(base, step);

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      const unsigned long long begin = clock64();
      cuda::device::barrier_expect_tx(*barrier, bytes);
      issue_tma_load<Rank>(&map, smem, coords, *barrier);
      barrier->arrive_and_wait();
      const unsigned long long end = clock64();
      if (cycles)
        cycles[static_cast<size_t>(blockIdx.x) * iterations + iteration] =
            end - begin;
    }
    __syncthreads();
  }

  if (capture) {
    uint8_t *block_capture =
        capture + static_cast<size_t>(blockIdx.x) * bytes;
    for (int i = threadIdx.x; i < bytes; i += blockDim.x)
      block_capture[i] = smem[i];
  }
}

template <int Rank>
__global__ void tma_s2g_kernel(
    const __grid_constant__ CUtensorMap map, int bytes, int iterations,
    uint32_t seed, DeviceCoords base, DeviceCoords step,
    unsigned long long *cycles) {
  extern __shared__ __align__(128) uint8_t smem[];
  const uint64_t block_base = static_cast<uint64_t>(blockIdx.x) * bytes;
  for (int i = threadIdx.x; i < bytes; i += blockDim.x)
    smem[i] = pattern_byte(seed, block_base + static_cast<uint64_t>(i));
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
  const DeviceCoords coords = block_coords(base, step);

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      const unsigned long long begin = clock64();
      issue_tma_store<Rank>(&map, smem, coords);
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      const unsigned long long end = clock64();
      if (cycles)
        cycles[static_cast<size_t>(blockIdx.x) * iterations + iteration] =
            end - begin;
    }
    __syncthreads();
  }
}

template <int Rank>
__global__ void tma_roundtrip_kernel(
    const __grid_constant__ CUtensorMap source_map,
    const __grid_constant__ CUtensorMap destination_map, int bytes,
    int iterations, DeviceCoords base, DeviceCoords step,
    unsigned long long *cycles) {
  extern __shared__ __align__(128) uint8_t smem[];
  BlockBarrier *barrier = barrier_after(smem, static_cast<size_t>(bytes));
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  const DeviceCoords coords = block_coords(base, step);

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      const unsigned long long begin = clock64();
      cuda::device::barrier_expect_tx(*barrier, bytes);
      issue_tma_load<Rank>(&source_map, smem, coords, *barrier);
      barrier->arrive_and_wait();
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
      issue_tma_store<Rank>(&destination_map, smem, coords);
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      const unsigned long long end = clock64();
      if (cycles)
        cycles[static_cast<size_t>(blockIdx.x) * iterations + iteration] =
            end - begin;
    }
    __syncthreads();
  }
}

template <int Threads, int Direction>
__global__ void cooperative_kernel(const uint8_t *source, uint8_t *destination,
                                   int bytes, int iterations, uint32_t seed,
                                   unsigned long long *cycles,
                                   uint8_t *capture) {
  static_assert(Threads == 32 || Threads == 128, "supported baselines");
  static_assert(Direction >= 0 && Direction <= 2, "direction");
  extern __shared__ __align__(128) uint8_t smem[];
  const size_t block_base = static_cast<size_t>(blockIdx.x) * bytes;

  if constexpr (Direction == 1) {
    for (int i = threadIdx.x; i < bytes; i += Threads)
      smem[i] = pattern_byte(seed, block_base + static_cast<size_t>(i));
    __syncthreads();
  }

  for (int iteration = 0; iteration < iterations; ++iteration) {
    const unsigned long long begin = clock64();
    if constexpr (Direction == 0 || Direction == 2) {
      for (int i = threadIdx.x * kVectorBytes; i < bytes;
           i += Threads * kVectorBytes) {
        *reinterpret_cast<uint4 *>(smem + i) =
            *reinterpret_cast<const uint4 *>(source + block_base + i);
      }
      __syncthreads();
    }
    if constexpr (Direction == 1 || Direction == 2) {
      for (int i = threadIdx.x * kVectorBytes; i < bytes;
           i += Threads * kVectorBytes) {
        *reinterpret_cast<uint4 *>(destination + block_base + i) =
            *reinterpret_cast<const uint4 *>(smem + i);
      }
      __syncthreads();
    }
    const unsigned long long end = clock64();
    if (threadIdx.x == 0 && cycles)
      cycles[static_cast<size_t>(blockIdx.x) * iterations + iteration] =
          end - begin;
    __syncthreads();
  }

  if constexpr (Direction == 0) {
    if (capture) {
      uint8_t *block_capture = capture + block_base;
      for (int i = threadIdx.x; i < bytes; i += Threads)
        block_capture[i] = smem[i];
    }
  }
}

template <int Outstanding>
__global__ void tma_outstanding_g2s_kernel(
    const __grid_constant__ CUtensorMap map, int bytes, int rows_per_tile,
    int iterations, unsigned long long *cycles, uint8_t *capture) {
  static_assert(Outstanding == 1 || Outstanding == 2 || Outstanding == 4 ||
                    Outstanding == 8,
                "outstanding set");
  extern __shared__ __align__(128) uint8_t smem[];
  const size_t payload = static_cast<size_t>(bytes) * Outstanding;
  BlockBarrier *barriers = barrier_after(smem, payload);
  if (threadIdx.x == 0) {
#pragma unroll
    for (int q = 0; q < Outstanding; ++q) init(&barriers[q], 1);
  }
  __syncthreads();

  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      const unsigned long long begin = clock64();
#pragma unroll
      for (int q = 0; q < Outstanding; ++q) {
        cuda::device::barrier_expect_tx(barriers[q], bytes);
        DeviceCoords coords{};
        coords.value[1] = q * rows_per_tile;
        issue_tma_load<2>(&map, smem + static_cast<size_t>(q) * bytes, coords,
                          barriers[q]);
      }
#pragma unroll
      for (int q = 0; q < Outstanding; ++q)
        barriers[q].arrive_and_wait();
      const unsigned long long end = clock64();
      if (cycles) cycles[iteration] = end - begin;
    }
    __syncthreads();
  }

  if (capture) {
    for (size_t i = threadIdx.x; i < payload; i += blockDim.x)
      capture[i] = smem[i];
  }
}

__device__ __noinline__ uint32_t overlap_work(uint32_t value) {
#pragma unroll 1
  for (int i = 0; i < kOverlapOps; ++i) {
    value = value * 1664525u + 1013904223u;
    asm volatile("" : "+r"(value));
  }
  return value;
}

__global__ void overlap_compute_kernel(int iterations,
                                       unsigned long long *cycles,
                                       uint32_t *sink) {
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      uint32_t value = static_cast<uint32_t>(iteration) + 123u;
      const unsigned long long begin = clock64();
      value = overlap_work(value);
      const unsigned long long end = clock64();
      if (cycles) cycles[iteration] = end - begin;
      sink[iteration] = value;
    }
    __syncthreads();
  }
}

template <bool Overlap>
__global__ void overlap_tma_kernel(
    const CUtensorMap *map, int iterations,
    unsigned long long *cycles, uint32_t *sink, uint8_t *capture) {
  extern __shared__ __align__(128) uint8_t smem[];
  BlockBarrier *barrier = barrier_after(smem, kOverlapBytes);
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  DeviceCoords coords{};

  if (threadIdx.x == 0)
    asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;"
                 :: "l"(map) : "memory");
  __syncthreads();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      uint32_t value = static_cast<uint32_t>(iteration) + 123u;
      const unsigned long long begin = clock64();
      cuda::device::barrier_expect_tx(*barrier, kOverlapBytes);
      issue_tma_load<2>(map, smem, coords, *barrier);
      if constexpr (Overlap) value = overlap_work(value);
      barrier->arrive_and_wait();
      if constexpr (!Overlap) value = overlap_work(value);
      const unsigned long long end = clock64();
      if (cycles) cycles[iteration] = end - begin;
      sink[iteration] = value + smem[iteration & (kOverlapBytes - 1)];
    }
    __syncthreads();
  }
  if (capture) {
    for (int i = threadIdx.x; i < kOverlapBytes; i += blockDim.x)
      capture[i] = smem[i];
  }
}

size_t descriptor_allocation_bytes(const CaseSpec &spec) {
  if (spec.rank == 1)
    return static_cast<size_t>(spec.dims[0]) * spec.element_bytes;
  return static_cast<size_t>(spec.strides[spec.rank - 2]) *
         static_cast<size_t>(spec.dims[spec.rank - 1]);
}

size_t barrier_shared_bytes(size_t payload_bytes, int barriers = 1) {
  const size_t aligned =
      (payload_bytes + alignof(BlockBarrier) - 1) &
      ~(static_cast<size_t>(alignof(BlockBarrier)) - 1);
  return aligned + static_cast<size_t>(barriers) * sizeof(BlockBarrier);
}

DeviceCoords device_coords(const std::array<int32_t, kMaxRank> &coords) {
  DeviceCoords result{};
  std::copy(coords.begin(), coords.end(), result.value);
  return result;
}

CUresult encode_map(CUtensorMap *map, const CaseSpec &spec, void *base) {
  return cuTensorMapEncodeTiled(
      map, spec.data_type, static_cast<cuuint32_t>(spec.rank), base,
      spec.dims.data(), spec.strides.data(), spec.box.data(),
      spec.element_strides.data(), spec.interleave, spec.swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

std::string cu_error_string(CUresult status) {
  const char *name = nullptr;
  const char *text = nullptr;
  cuGetErrorName(status, &name);
  cuGetErrorString(status, &text);
  return std::string(name ? name : "unknown") + ":" +
         (text ? text : "unknown");
}

CaseSpec make_contiguous_case(std::string id, int bytes, int ctas = 1) {
  CaseSpec spec;
  spec.id = std::move(id);
  spec.bytes = bytes;
  spec.ctas = ctas;
  spec.rank = 2;
  spec.element_bytes = 1;
  spec.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  const int width = std::min(128, bytes);
  const int rows = bytes / width;
  spec.dims[0] = width;
  spec.dims[1] = static_cast<cuuint64_t>(rows) * ctas;
  spec.strides[0] = width;
  spec.box[0] = width;
  spec.box[1] = rows;
  spec.element_strides.fill(1);
  spec.block_steps[1] = rows;
  return spec;
}

CaseSpec make_fp32_tile_case(int rows, int cols) {
  CaseSpec spec;
  // Ventus test reports and the study matrix name tiles as rows x columns.
  // CUDA Tensor Map dimension 0 is the contiguous column dimension, so keep
  // the public case id in rows x cols order while encoding cols x rows.
  spec.id = "tile_" + std::to_string(rows) + "x" +
            std::to_string(cols);
  spec.layout = "contiguous";
  spec.rank = 2;
  spec.bytes = rows * cols * 4;
  spec.element_bytes = 4;
  spec.data_type = CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
  spec.dims[0] = cols;
  spec.dims[1] = rows;
  spec.strides[0] = static_cast<cuuint64_t>(cols) * 4;
  spec.box[0] = cols;
  spec.box[1] = rows;
  spec.element_strides.fill(1);
  spec.block_steps[1] = rows;
  return spec;
}

void set_contiguous_strides(CaseSpec *spec) {
  cuuint64_t stride = spec->dims[0] * spec->element_bytes;
  for (int dimension = 1; dimension < spec->rank; ++dimension) {
    spec->strides[dimension - 1] = stride;
    stride *= spec->dims[dimension];
  }
}

CaseSpec make_rank_case(int rank) {
  CaseSpec spec;
  spec.id = "rank" + std::to_string(rank) + "_contiguous";
  spec.layout = "contiguous";
  spec.rank = rank;
  spec.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  spec.element_bytes = 1;
  spec.element_strides.fill(1);
  if (rank == 1) {
    spec.dims[0] = spec.box[0] = 128;
    spec.bytes = 128;
  } else if (rank == 2) {
    spec.dims[0] = spec.box[0] = 128;
    spec.dims[1] = spec.box[1] = 32;
    spec.bytes = 4096;
  } else if (rank == 3) {
    spec.dims[0] = spec.box[0] = 128;
    spec.dims[1] = spec.box[1] = 4;
    spec.dims[2] = spec.box[2] = 8;
    spec.bytes = 4096;
  } else if (rank == 4) {
    spec.dims[0] = spec.box[0] = 128;
    spec.dims[1] = spec.box[1] = 2;
    spec.dims[2] = spec.box[2] = 4;
    spec.dims[3] = spec.box[3] = 4;
    spec.bytes = 4096;
  } else {
    spec.dims[0] = spec.box[0] = 128;
    spec.dims[1] = spec.box[1] = 2;
    spec.dims[2] = spec.box[2] = 2;
    spec.dims[3] = spec.box[3] = 2;
    spec.dims[4] = spec.box[4] = 4;
    spec.bytes = 4096;
  }
  set_contiguous_strides(&spec);
  return spec;
}

CaseSpec make_row_stride_case() {
  CaseSpec spec = make_contiguous_case("rank2_row_stride2", 4096);
  spec.layout = "row_stride2";
  spec.strides[0] = 256;
  return spec;
}

CaseSpec make_element_stride_case() {
  CaseSpec spec = make_contiguous_case("rank2_element_stride2", 4096);
  spec.layout = "element_stride2";
  spec.dims[1] = 64;
  spec.box[1] = 64;
  spec.element_strides[1] = 2;
  spec.block_steps[1] = 64;
  return spec;
}

CaseSpec make_oob_case(int percent) {
  CaseSpec spec = make_contiguous_case("oob_" + std::to_string(percent), 4096);
  spec.layout = "oob" + std::to_string(percent);
  spec.coords[0] = -(128 * percent / 100);
  return spec;
}

CaseSpec make_swizzle_case(int bytes) {
  CaseSpec spec = make_contiguous_case("swizzle" + std::to_string(bytes),
                                       4096);
  spec.layout = "swizzle" + std::to_string(bytes);
  spec.dims[0] = spec.box[0] = bytes;
  spec.dims[1] = spec.box[1] = 4096 / bytes;
  spec.strides[0] = bytes;
  spec.block_steps[1] = 4096 / bytes;
  spec.shared_is_linear = false;
  spec.swizzle = bytes == 32   ? CU_TENSOR_MAP_SWIZZLE_32B
                 : bytes == 64 ? CU_TENSOR_MAP_SWIZZLE_64B
                               : CU_TENSOR_MAP_SWIZZLE_128B;
  return spec;
}

CaseSpec make_interleave_case(int bytes) {
  CaseSpec spec;
  spec.id = "interleave" + std::to_string(bytes);
  spec.layout = spec.id;
  spec.rank = 3;
  spec.bytes = 4096;
  spec.data_type = CU_TENSOR_MAP_DATA_TYPE_UINT8;
  spec.element_bytes = 1;
  spec.element_strides.fill(1);
  if (bytes == 16) {
    spec.dims[0] = spec.box[0] = 16;
    spec.dims[1] = spec.box[1] = 8;
    spec.dims[2] = spec.box[2] = 32;
    spec.interleave = CU_TENSOR_MAP_INTERLEAVE_16B;
    spec.swizzle = CU_TENSOR_MAP_SWIZZLE_NONE;
  } else {
    spec.dims[0] = spec.box[0] = 32;
    spec.dims[1] = spec.box[1] = 4;
    spec.dims[2] = spec.box[2] = 32;
    spec.interleave = CU_TENSOR_MAP_INTERLEAVE_32B;
    spec.swizzle = CU_TENSOR_MAP_SWIZZLE_32B;
  }
  set_contiguous_strides(&spec);
  spec.shared_is_linear = false;
  return spec;
}

std::vector<uint8_t> patterned_bytes(size_t bytes, uint32_t seed) {
  std::vector<uint8_t> data(bytes);
  for (size_t i = 0; i < bytes; ++i) data[i] = pattern_byte(seed, i);
  return data;
}

std::vector<uint8_t> gather_expected(const CaseSpec &spec,
                                     const std::vector<uint8_t> &source) {
  std::vector<uint8_t> expected(static_cast<size_t>(spec.bytes) * spec.ctas,
                                0);
  std::array<size_t, kMaxRank> counts{};
  size_t elements = 1;
  for (int d = 0; d < spec.rank; ++d) {
    const cuuint32_t stride =
        (d == 0 && spec.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE)
            ? 1
            : spec.element_strides[d];
    counts[d] = (spec.box[d] + stride - 1) / stride;
    elements *= counts[d];
  }
  if (elements * static_cast<size_t>(spec.element_bytes) !=
      static_cast<size_t>(spec.bytes)) {
    std::fprintf(stderr, "INTERNAL,gather_size,%s,%zu,%d\n", spec.id.c_str(),
                 elements * spec.element_bytes, spec.bytes);
    std::exit(EXIT_FAILURE);
  }

  for (int block = 0; block < spec.ctas; ++block) {
    for (size_t linear = 0; linear < elements; ++linear) {
      size_t remaining = linear;
      bool in_bounds = true;
      int64_t source_offset = 0;
      for (int d = 0; d < spec.rank; ++d) {
        const size_t local = remaining % counts[d];
        remaining /= counts[d];
        const int64_t element_stride =
            (d == 0 && spec.interleave == CU_TENSOR_MAP_INTERLEAVE_NONE)
                ? 1
                : spec.element_strides[d];
        const int64_t coordinate =
            static_cast<int64_t>(spec.coords[d]) +
            static_cast<int64_t>(block) * spec.block_steps[d] +
            static_cast<int64_t>(local) * element_stride;
        if (coordinate < 0 ||
            coordinate >= static_cast<int64_t>(spec.dims[d]))
          in_bounds = false;
        if (d == 0)
          source_offset += coordinate * spec.element_bytes;
        else
          source_offset += coordinate * static_cast<int64_t>(spec.strides[d - 1]);
      }
      if (in_bounds) {
        const size_t output_offset =
            static_cast<size_t>(block) * spec.bytes +
            linear * spec.element_bytes;
        for (int byte = 0; byte < spec.element_bytes; ++byte)
          expected[output_offset + byte] =
              source[static_cast<size_t>(source_offset) + byte];
      }
    }
  }
  return expected;
}

uint64_t mismatch_count(const std::vector<uint8_t> &actual,
                        const std::vector<uint8_t> &expected) {
  if (actual.size() != expected.size())
    return static_cast<uint64_t>(std::max(actual.size(), expected.size()));
  uint64_t errors = 0;
  for (size_t i = 0; i < actual.size(); ++i)
    errors += actual[i] != expected[i];
  return errors;
}

uint64_t fingerprint64(const uint8_t *data, size_t bytes) {
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= data[index];
    value *= 1099511628211ull;
  }
  return value;
}

std::vector<unsigned long long> average_block_cycles(
    const std::vector<unsigned long long> &raw, int ctas, int repeats) {
  std::vector<unsigned long long> result(repeats);
  for (int repeat = 0; repeat < repeats; ++repeat) {
    unsigned long long sum = 0;
    for (int block = 0; block < ctas; ++block)
      sum += raw[static_cast<size_t>(block) * repeats + repeat];
    result[repeat] = (sum + static_cast<unsigned long long>(ctas / 2)) /
                     static_cast<unsigned long long>(ctas);
  }
  return result;
}

float timed_launch(const std::function<void()> &launch) {
  cudaEvent_t begin = nullptr;
  cudaEvent_t end = nullptr;
  CUDA_CHECK(cudaEventCreate(&begin));
  CUDA_CHECK(cudaEventCreate(&end));
  CUDA_CHECK(cudaEventRecord(begin));
  launch();
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaEventRecord(end));
  CUDA_CHECK(cudaEventSynchronize(end));
  float milliseconds = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&milliseconds, begin, end));
  CUDA_CHECK(cudaEventDestroy(end));
  CUDA_CHECK(cudaEventDestroy(begin));
  return milliseconds;
}

void emit_rows(const Context &context, std::string_view direction,
               std::string_view method, const CaseSpec &spec, int outstanding,
               int batch, const std::vector<unsigned long long> &cycles,
               double event_ns, uint64_t errors,
               const std::vector<uint64_t> &descriptor_addresses = {},
               const std::vector<uint64_t> &descriptor_fingerprints = {}) {
  const std::string measurement_id =
      spec.id + "__" + std::string(method.begin(), method.end());
  for (int repeat = 0; repeat < static_cast<int>(cycles.size()); ++repeat) {
    std::printf(
        "%s,%s,%.*s,%.*s,%s,%d,%d,%s,%d,%d,%d,%d,%llu,%.3f,%llu,"
        "0x%016llx,0x%016llx,%d,%s,0,%s,%s\n",
        context.gpu.c_str(), context.cc.c_str(),
        static_cast<int>(direction.size()), direction.data(),
        static_cast<int>(method.size()), method.data(), measurement_id.c_str(),
        spec.bytes, spec.rank, spec.layout.c_str(), outstanding, spec.ctas,
        batch, repeat, cycles[repeat], event_ns,
        static_cast<unsigned long long>(errors),
        static_cast<unsigned long long>(
            repeat < static_cast<int>(descriptor_addresses.size())
                ? descriptor_addresses[repeat] : 0),
        static_cast<unsigned long long>(
            repeat < static_cast<int>(descriptor_fingerprints.size())
                ? descriptor_fingerprints[repeat] : 0),
        descriptor_addresses.empty() ? 0 : repeat,
        descriptor_addresses.empty() ? "not_applicable" : "cuda_memcpy_l2_unspecified",
        descriptor_addresses.empty() ? "not_applicable"
          : repeat == 0 ? "tmau_cold_first_use" : "tmau_hot_reuse",
        descriptor_addresses.empty() ? (repeat == 0 ? "first" : "second")
                                     : (repeat == 0 ? "cold" : "hot"));
  }
}

void launch_tma_g2s(int rank, dim3 grid, dim3 block, size_t shared,
                    const CUtensorMap &map, int bytes, int iterations,
                    DeviceCoords base, DeviceCoords step,
                    unsigned long long *cycles, uint8_t *capture) {
  switch (rank) {
    case 1:
      tma_g2s_kernel<1><<<grid, block, shared>>>(map, bytes, iterations, base,
                                                step, cycles, capture);
      break;
    case 2:
      tma_g2s_kernel<2><<<grid, block, shared>>>(map, bytes, iterations, base,
                                                step, cycles, capture);
      break;
    case 3:
      tma_g2s_kernel<3><<<grid, block, shared>>>(map, bytes, iterations, base,
                                                step, cycles, capture);
      break;
    case 4:
      tma_g2s_kernel<4><<<grid, block, shared>>>(map, bytes, iterations, base,
                                                step, cycles, capture);
      break;
    case 5:
      tma_g2s_kernel<5><<<grid, block, shared>>>(map, bytes, iterations, base,
                                                step, cycles, capture);
      break;
    default:
      std::fprintf(stderr, "INTERNAL,unsupported_rank,%d\n", rank);
      std::exit(EXIT_FAILURE);
  }
}

void launch_tma_s2g(int rank, dim3 grid, dim3 block, size_t shared,
                    const CUtensorMap &map, int bytes, int iterations,
                    uint32_t seed, DeviceCoords base, DeviceCoords step,
                    unsigned long long *cycles) {
  switch (rank) {
    case 1:
      tma_s2g_kernel<1><<<grid, block, shared>>>(map, bytes, iterations, seed,
                                                base, step, cycles);
      break;
    case 2:
      tma_s2g_kernel<2><<<grid, block, shared>>>(map, bytes, iterations, seed,
                                                base, step, cycles);
      break;
    case 3:
      tma_s2g_kernel<3><<<grid, block, shared>>>(map, bytes, iterations, seed,
                                                base, step, cycles);
      break;
    case 4:
      tma_s2g_kernel<4><<<grid, block, shared>>>(map, bytes, iterations, seed,
                                                base, step, cycles);
      break;
    case 5:
      tma_s2g_kernel<5><<<grid, block, shared>>>(map, bytes, iterations, seed,
                                                base, step, cycles);
      break;
  }
}

void launch_tma_roundtrip(int rank, dim3 grid, dim3 block, size_t shared,
                          const CUtensorMap &source_map,
                          const CUtensorMap &destination_map, int bytes,
                          int iterations, DeviceCoords base, DeviceCoords step,
                          unsigned long long *cycles) {
  switch (rank) {
    case 1:
      tma_roundtrip_kernel<1><<<grid, block, shared>>>(
          source_map, destination_map, bytes, iterations, base, step, cycles);
      break;
    case 2:
      tma_roundtrip_kernel<2><<<grid, block, shared>>>(
          source_map, destination_map, bytes, iterations, base, step, cycles);
      break;
    case 3:
      tma_roundtrip_kernel<3><<<grid, block, shared>>>(
          source_map, destination_map, bytes, iterations, base, step, cycles);
      break;
    case 4:
      tma_roundtrip_kernel<4><<<grid, block, shared>>>(
          source_map, destination_map, bytes, iterations, base, step, cycles);
      break;
    case 5:
      tma_roundtrip_kernel<5><<<grid, block, shared>>>(
          source_map, destination_map, bytes, iterations, base, step, cycles);
      break;
  }
}

template <typename T>
class DeviceArray {
 public:
  explicit DeviceArray(size_t count = 0) : count_(count) {
    if (count_) CUDA_CHECK(cudaMalloc(&data_, count_ * sizeof(T)));
  }
  ~DeviceArray() {
    if (data_) cudaFree(data_);
  }
  DeviceArray(const DeviceArray &) = delete;
  DeviceArray &operator=(const DeviceArray &) = delete;
  T *get() const { return data_; }
  size_t size() const { return count_; }
  size_t bytes() const { return count_ * sizeof(T); }

 private:
  T *data_ = nullptr;
  size_t count_ = 0;
};

void complete_warmup() {
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());
}

void launch_cooperative(int threads, int direction, dim3 grid, size_t shared,
                        const uint8_t *source, uint8_t *destination, int bytes,
                        int iterations, uint32_t seed,
                        unsigned long long *cycles, uint8_t *capture) {
  if (threads == 32) {
    if (direction == 0)
      cooperative_kernel<32, 0><<<grid, 32, shared>>>(
          source, destination, bytes, iterations, seed, cycles, capture);
    else if (direction == 1)
      cooperative_kernel<32, 1><<<grid, 32, shared>>>(
          source, destination, bytes, iterations, seed, cycles, capture);
    else
      cooperative_kernel<32, 2><<<grid, 32, shared>>>(
          source, destination, bytes, iterations, seed, cycles, capture);
  } else {
    if (direction == 0)
      cooperative_kernel<128, 0><<<grid, 128, shared>>>(
          source, destination, bytes, iterations, seed, cycles, capture);
    else if (direction == 1)
      cooperative_kernel<128, 1><<<grid, 128, shared>>>(
          source, destination, bytes, iterations, seed, cycles, capture);
    else
      cooperative_kernel<128, 2><<<grid, 128, shared>>>(
          source, destination, bytes, iterations, seed, cycles, capture);
  }
}

const char *direction_name(int direction) {
  return direction == 0 ? "g2s" : direction == 1 ? "s2g" : "roundtrip";
}

void log_encode_skip(const CaseSpec &spec, std::string_view direction,
                     CUresult status) {
  std::fprintf(stderr, "SKIP,encode,%s,%.*s,%s\n", spec.id.c_str(),
               static_cast<int>(direction.size()), direction.data(),
               cu_error_string(status).c_str());
}

void run_tma_direction(const Context &context, const CaseSpec &spec,
                       int direction) {
  if (direction == 0 && !spec.shared_is_linear) {
    std::fprintf(stderr, "SKIP,nonlinear_capture,%s,g2s\n", spec.id.c_str());
    return;
  }
  if (direction == 1 && spec.layout != "contiguous") {
    std::fprintf(stderr, "SKIP,noncontiguous_store_validation,%s,s2g\n",
                 spec.id.c_str());
    return;
  }

  const size_t allocation = descriptor_allocation_bytes(spec);
  const size_t captured = static_cast<size_t>(spec.bytes) * spec.ctas;
  DeviceArray<uint8_t> source(allocation);
  DeviceArray<uint8_t> destination(allocation);
  DeviceArray<uint8_t> capture(direction == 0 ? captured : 0);
  DeviceArray<unsigned long long> device_cycles(
      static_cast<size_t>(spec.ctas) * context.options.repeats);

  CUtensorMap source_map{};
  CUtensorMap destination_map{};
  if (direction != 1) {
    const CUresult status = encode_map(&source_map, spec, source.get());
    if (status != CUDA_SUCCESS) {
      log_encode_skip(spec, direction_name(direction), status);
      return;
    }
  }
  if (direction != 0) {
    const CUresult status =
        encode_map(&destination_map, spec, destination.get());
    if (status != CUDA_SUCCESS) {
      log_encode_skip(spec, direction_name(direction), status);
      return;
    }
  }

  const DeviceCoords base = device_coords(spec.coords);
  const DeviceCoords step = device_coords(spec.block_steps);
  const dim3 grid(spec.ctas);
  const dim3 block(128);
  const size_t shared = direction == 1
                            ? static_cast<size_t>(spec.bytes)
                            : barrier_shared_bytes(spec.bytes);

  for (int batch = 0;
       batch < static_cast<int>(context.options.seeds.size()); ++batch) {
    const uint32_t seed = context.options.seeds[batch];
    const std::vector<uint8_t> host_source = patterned_bytes(allocation, seed);
    CUDA_CHECK(cudaMemcpy(source.get(), host_source.data(), allocation,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(destination.get(), 0xa5, allocation));

    if (direction == 0)
      launch_tma_g2s(spec.rank, grid, block, shared, source_map, spec.bytes,
                     context.options.warmups, base, step, nullptr, nullptr);
    else if (direction == 1)
      launch_tma_s2g(spec.rank, grid, block, shared, destination_map,
                     spec.bytes, context.options.warmups, seed, base, step,
                     nullptr);
    else
      launch_tma_roundtrip(spec.rank, grid, block, shared, source_map,
                           destination_map, spec.bytes, context.options.warmups,
                           base, step, nullptr);
    complete_warmup();

    if (direction != 0)
      CUDA_CHECK(cudaMemset(destination.get(), 0xa5, allocation));
    const float elapsed_ms = timed_launch([&] {
      if (direction == 0)
        launch_tma_g2s(spec.rank, grid, block, shared, source_map, spec.bytes,
                       context.options.repeats, base, step,
                       device_cycles.get(), nullptr);
      else if (direction == 1)
        launch_tma_s2g(spec.rank, grid, block, shared, destination_map,
                       spec.bytes, context.options.repeats, seed, base, step,
                       device_cycles.get());
      else
        launch_tma_roundtrip(spec.rank, grid, block, shared, source_map,
                             destination_map, spec.bytes,
                             context.options.repeats, base, step,
                             device_cycles.get());
    });

    std::vector<unsigned long long> raw_cycles(
        static_cast<size_t>(spec.ctas) * context.options.repeats);
    CUDA_CHECK(cudaMemcpy(raw_cycles.data(), device_cycles.get(),
                          raw_cycles.size() * sizeof(raw_cycles[0]),
                          cudaMemcpyDeviceToHost));
    const auto cycles = average_block_cycles(raw_cycles, spec.ctas,
                                             context.options.repeats);
    uint64_t errors = 0;
    if (direction == 0) {
      launch_tma_g2s(spec.rank, grid, block, shared, source_map, spec.bytes, 1,
                     base, step, nullptr, capture.get());
      complete_warmup();
      std::vector<uint8_t> actual(captured);
      CUDA_CHECK(cudaMemcpy(actual.data(), capture.get(), captured,
                            cudaMemcpyDeviceToHost));
      errors = mismatch_count(actual, gather_expected(spec, host_source));
    } else {
      std::vector<uint8_t> actual(allocation);
      CUDA_CHECK(cudaMemcpy(actual.data(), destination.get(), allocation,
                            cudaMemcpyDeviceToHost));
      if (direction == 1) {
        std::vector<uint8_t> expected(allocation, 0xa5);
        for (int block_index = 0; block_index < spec.ctas; ++block_index)
          for (int i = 0; i < spec.bytes; ++i)
            expected[static_cast<size_t>(block_index) * spec.bytes + i] =
                pattern_byte(seed,
                             static_cast<uint64_t>(block_index) * spec.bytes +
                                 static_cast<uint64_t>(i));
        errors = mismatch_count(actual, expected);
      } else {
        errors = mismatch_count(actual, host_source);
      }
    }
    const double event_ns =
        static_cast<double>(elapsed_ms) * 1.0e6 / context.options.repeats;
    emit_rows(context, direction_name(direction), "tma", spec, 1, batch,
              cycles, event_ns, errors);
  }
}

void run_cooperative_direction(const Context &context, const CaseSpec &spec,
                               int direction, int threads) {
  const size_t allocation = static_cast<size_t>(spec.bytes) * spec.ctas;
  DeviceArray<uint8_t> source(allocation);
  DeviceArray<uint8_t> destination(allocation);
  DeviceArray<uint8_t> capture(direction == 0 ? allocation : 0);
  DeviceArray<unsigned long long> device_cycles(
      static_cast<size_t>(spec.ctas) * context.options.repeats);
  const dim3 grid(spec.ctas);
  const std::string method =
      threads == 32 ? "cooperative32" : "cooperative128";

  for (int batch = 0;
       batch < static_cast<int>(context.options.seeds.size()); ++batch) {
    const uint32_t seed = context.options.seeds[batch];
    const std::vector<uint8_t> host_source = patterned_bytes(allocation, seed);
    CUDA_CHECK(cudaMemcpy(source.get(), host_source.data(), allocation,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(destination.get(), 0xa5, allocation));
    launch_cooperative(threads, direction, grid, spec.bytes, source.get(),
                       destination.get(), spec.bytes, context.options.warmups,
                       seed, nullptr, nullptr);
    complete_warmup();
    if (direction != 0)
      CUDA_CHECK(cudaMemset(destination.get(), 0xa5, allocation));

    const float elapsed_ms = timed_launch([&] {
      launch_cooperative(threads, direction, grid, spec.bytes, source.get(),
                         destination.get(), spec.bytes,
                         context.options.repeats, seed, device_cycles.get(),
                         nullptr);
    });
    std::vector<unsigned long long> raw_cycles(
        static_cast<size_t>(spec.ctas) * context.options.repeats);
    CUDA_CHECK(cudaMemcpy(raw_cycles.data(), device_cycles.get(),
                          raw_cycles.size() * sizeof(raw_cycles[0]),
                          cudaMemcpyDeviceToHost));
    const auto cycles = average_block_cycles(raw_cycles, spec.ctas,
                                             context.options.repeats);

    uint64_t errors = 0;
    if (direction == 0) {
      launch_cooperative(threads, direction, grid, spec.bytes, source.get(),
                         destination.get(), spec.bytes, 1, seed, nullptr,
                         capture.get());
      complete_warmup();
      std::vector<uint8_t> actual(allocation);
      CUDA_CHECK(cudaMemcpy(actual.data(), capture.get(), allocation,
                            cudaMemcpyDeviceToHost));
      errors = mismatch_count(actual, host_source);
    } else {
      std::vector<uint8_t> actual(allocation);
      CUDA_CHECK(cudaMemcpy(actual.data(), destination.get(), allocation,
                            cudaMemcpyDeviceToHost));
      if (direction == 1) {
        std::vector<uint8_t> expected = patterned_bytes(allocation, seed);
        errors = mismatch_count(actual, expected);
      } else {
        errors = mismatch_count(actual, host_source);
      }
    }
    const double event_ns =
        static_cast<double>(elapsed_ms) * 1.0e6 / context.options.repeats;
    emit_rows(context, direction_name(direction), method, spec, 1, batch,
              cycles, event_ns, errors);
  }
}

void run_all_methods(const Context &context, const CaseSpec &spec,
                     int direction) {
  run_tma_direction(context, spec, direction);
  run_cooperative_direction(context, spec, direction, 32);
  run_cooperative_direction(context, spec, direction, 128);
}

void launch_outstanding(int outstanding, const CUtensorMap &map, int bytes,
                        int rows, int iterations, size_t shared,
                        unsigned long long *cycles, uint8_t *capture) {
  if (outstanding == 1)
    tma_outstanding_g2s_kernel<1><<<1, 128, shared>>>(
        map, bytes, rows, iterations, cycles, capture);
  else if (outstanding == 2)
    tma_outstanding_g2s_kernel<2><<<1, 128, shared>>>(
        map, bytes, rows, iterations, cycles, capture);
  else if (outstanding == 4)
    tma_outstanding_g2s_kernel<4><<<1, 128, shared>>>(
        map, bytes, rows, iterations, cycles, capture);
  else
    tma_outstanding_g2s_kernel<8><<<1, 128, shared>>>(
        map, bytes, rows, iterations, cycles, capture);
}

void run_outstanding_case(const Context &context, int outstanding) {
  CaseSpec map_spec = make_contiguous_case(
      "outstanding_" + std::to_string(outstanding), 4096, outstanding);
  map_spec.ctas = 1;  // One CTA issues N independent requests.
  CaseSpec output_spec = map_spec;
  const size_t payload = static_cast<size_t>(map_spec.bytes) * outstanding;
  DeviceArray<uint8_t> source(payload);
  DeviceArray<uint8_t> capture(payload);
  DeviceArray<unsigned long long> device_cycles(context.options.repeats);
  CUtensorMap map{};
  const CUresult status = encode_map(&map, map_spec, source.get());
  if (status != CUDA_SUCCESS) {
    log_encode_skip(map_spec, "g2s", status);
    return;
  }
  const int rows = map_spec.bytes / 128;
  const size_t shared = barrier_shared_bytes(payload, outstanding);

  for (int batch = 0;
       batch < static_cast<int>(context.options.seeds.size()); ++batch) {
    const uint32_t seed = context.options.seeds[batch];
    const std::vector<uint8_t> host_source = patterned_bytes(payload, seed);
    CUDA_CHECK(cudaMemcpy(source.get(), host_source.data(), payload,
                          cudaMemcpyHostToDevice));
    launch_outstanding(outstanding, map, map_spec.bytes, rows,
                       context.options.warmups, shared, nullptr, nullptr);
    complete_warmup();
    const float elapsed_ms = timed_launch([&] {
      launch_outstanding(outstanding, map, map_spec.bytes, rows,
                         context.options.repeats, shared, device_cycles.get(),
                         nullptr);
    });
    std::vector<unsigned long long> cycles(context.options.repeats);
    CUDA_CHECK(cudaMemcpy(cycles.data(), device_cycles.get(),
                          cycles.size() * sizeof(cycles[0]),
                          cudaMemcpyDeviceToHost));
    launch_outstanding(outstanding, map, map_spec.bytes, rows, 1, shared,
                       nullptr, capture.get());
    complete_warmup();
    std::vector<uint8_t> actual(payload);
    CUDA_CHECK(cudaMemcpy(actual.data(), capture.get(), payload,
                          cudaMemcpyDeviceToHost));
    const uint64_t errors = mismatch_count(actual, host_source);
    const double event_ns =
        static_cast<double>(elapsed_ms) * 1.0e6 / context.options.repeats;
    emit_rows(context, "g2s", "tma", output_spec, outstanding, batch, cycles,
              event_ns, errors);
  }
}

void run_overlap_cases(const Context &context) {
  CaseSpec spec = make_contiguous_case("overlap_4k", kOverlapBytes);
  const size_t tensor_samples = 2;
  DeviceArray<uint8_t> sources(
      std::max<size_t>(1, tensor_samples) * kOverlapBytes);
  DeviceArray<uint8_t> capture(kOverlapBytes);
  DeviceArray<unsigned long long> device_cycles(context.options.repeats);
  DeviceArray<uint32_t> sink(
      std::max(context.options.warmups, context.options.repeats));
  DeviceArray<uint8_t> raw_maps(
      std::max<size_t>(1, tensor_samples) * 4096 + 4095);
  const uintptr_t aligned_map_address =
      (reinterpret_cast<uintptr_t>(raw_maps.get()) + 4095) &
      ~static_cast<uintptr_t>(4095);
  auto *map_pages = reinterpret_cast<uint8_t *>(aligned_map_address);
  const size_t shared = barrier_shared_bytes(kOverlapBytes);

  for (int batch = 0;
       batch < static_cast<int>(context.options.seeds.size()); ++batch) {
    const uint32_t seed = context.options.seeds[batch];
    const std::vector<uint8_t> host_source = patterned_bytes(kOverlapBytes, seed);
    for (int mode = 0; mode < 3; ++mode) {
      std::vector<const CUtensorMap *> maps(context.options.repeats, nullptr);
      std::vector<uint64_t> descriptor_addresses;
      std::vector<uint64_t> descriptor_fingerprints;
      if (mode != 0) {
        descriptor_addresses.resize(context.options.repeats);
        descriptor_fingerprints.resize(context.options.repeats);
        const size_t mode_index = static_cast<size_t>(mode - 1);
        const size_t resource = mode_index;
          uint8_t *source = sources.get() + resource * kOverlapBytes;
          CUDA_CHECK(cudaMemcpy(source, host_source.data(), host_source.size(),
                                cudaMemcpyHostToDevice));
          CUtensorMap host_map{};
          const CUresult status = encode_map(&host_map, spec, source);
          if (status != CUDA_SUCCESS) {
            log_encode_skip(spec, "overlap", status);
            return;
          }
          auto *map = reinterpret_cast<CUtensorMap *>(
              map_pages + resource * 4096);
          CUDA_CHECK(cudaMemcpy(map, &host_map, sizeof(host_map),
                                cudaMemcpyHostToDevice));
          for (int repeat = 0; repeat < context.options.repeats; ++repeat) {
            maps[repeat] = map;
            descriptor_addresses[repeat] = reinterpret_cast<uint64_t>(map);
            descriptor_fingerprints[repeat] = fingerprint64(
                reinterpret_cast<const uint8_t *>(&host_map), sizeof(host_map));
          }
      }
      const float elapsed_ms = timed_launch([&] {
        if (mode == 0)
          overlap_compute_kernel<<<1, 128>>>(
              context.options.repeats, device_cycles.get(), sink.get());
        else if (mode == 1)
          overlap_tma_kernel<false><<<1, 128, shared>>>(
              maps[0], context.options.repeats, device_cycles.get(),
              sink.get(), capture.get());
        else
          overlap_tma_kernel<true><<<1, 128, shared>>>(
              maps[0], context.options.repeats, device_cycles.get(),
              sink.get(), capture.get());
      });
      std::vector<unsigned long long> cycles(context.options.repeats);
      CUDA_CHECK(cudaMemcpy(cycles.data(), device_cycles.get(),
                            cycles.size() * sizeof(cycles[0]),
                            cudaMemcpyDeviceToHost));
      uint64_t errors = 0;
      if (mode != 0) {
        std::vector<uint8_t> actual(kOverlapBytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), capture.get(), actual.size(),
                              cudaMemcpyDeviceToHost));
        errors = mismatch_count(actual, host_source);
      }
      const char *method = mode == 0   ? "compute_only"
                           : mode == 1 ? "tma_serial"
                                       : "tma_overlap";
      const double event_ns =
          static_cast<double>(elapsed_ms) * 1.0e6 / context.options.repeats;
      emit_rows(context, "overlap", method, spec, 1, batch, cycles, event_ns,
                errors, descriptor_addresses, descriptor_fingerprints);
    }
  }
}

[[noreturn]] void usage(const char *program, int status) {
  std::fprintf(
      status == 0 ? stdout : stderr,
      "Usage: %s [--warmups N] [--repeats N] [--seeds A,B,C] "
      "[--suite smoke|core|full] [--csv]\n"
      "\n"
      "stdout is always the fixed 15-column sample CSV.  Environment and "
      "skip records are written to stderr as META/SKIP records.\n",
      program);
  std::exit(status);
}

int parse_nonnegative(const char *text, const char *option, bool allow_zero) {
  char *end = nullptr;
  const long value = std::strtol(text, &end, 10);
  if (!text[0] || !end || *end || value < (allow_zero ? 0 : 1) ||
      value > std::numeric_limits<int>::max()) {
    std::fprintf(stderr, "ARG_ERROR,%s,%s\n", option, text);
    std::exit(EXIT_FAILURE);
  }
  return static_cast<int>(value);
}

std::vector<uint32_t> parse_seeds(const char *text) {
  std::vector<uint32_t> seeds;
  const std::string input(text);
  size_t begin = 0;
  while (begin <= input.size()) {
    const size_t comma = input.find(',', begin);
    const std::string token =
        input.substr(begin, comma == std::string::npos ? std::string::npos
                                                       : comma - begin);
    if (token.empty()) {
      std::fprintf(stderr, "ARG_ERROR,--seeds,%s\n", text);
      std::exit(EXIT_FAILURE);
    }
    char *end = nullptr;
    const unsigned long long value = std::strtoull(token.c_str(), &end, 0);
    if (!end || *end || value > std::numeric_limits<uint32_t>::max()) {
      std::fprintf(stderr, "ARG_ERROR,--seeds,%s\n", text);
      std::exit(EXIT_FAILURE);
    }
    seeds.push_back(static_cast<uint32_t>(value));
    if (comma == std::string::npos) break;
    begin = comma + 1;
  }
  if (seeds.empty() || seeds.size() > 32) {
    std::fprintf(stderr, "ARG_ERROR,--seeds,count=%zu\n", seeds.size());
    std::exit(EXIT_FAILURE);
  }
  return seeds;
}

Options parse_options(int argc, char **argv) {
  Options options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view argument(argv[i]);
    const auto require_value = [&](const char *option) -> const char * {
      if (++i >= argc) {
        std::fprintf(stderr, "ARG_ERROR,%s,missing_value\n", option);
        std::exit(EXIT_FAILURE);
      }
      return argv[i];
    };
    if (argument == "--warmups")
      options.warmups =
          parse_nonnegative(require_value("--warmups"), "--warmups", true);
    else if (argument == "--repeats")
      options.repeats =
          parse_nonnegative(require_value("--repeats"), "--repeats", false);
    else if (argument == "--seeds")
      options.seeds = parse_seeds(require_value("--seeds"));
    else if (argument == "--suite")
      options.suite = require_value("--suite");
    else if (argument == "--csv")
      options.csv = true;
    else if (argument == "--help" || argument == "-h")
      usage(argv[0], 0);
    else {
      std::fprintf(stderr, "ARG_ERROR,unknown,%s\n", argv[i]);
      usage(argv[0], 2);
    }
  }
  if (options.suite != "smoke" && options.suite != "core" &&
      options.suite != "full") {
    std::fprintf(stderr, "ARG_ERROR,--suite,%s\n", options.suite.c_str());
    std::exit(EXIT_FAILURE);
  }
  return options;
}

std::string csv_safe_gpu_name(const char *name) {
  std::string result(name ? name : "unknown");
  std::replace(result.begin(), result.end(), ',', ' ');
  return result;
}

void emit_metadata(const Context &context) {
  int driver_version = 0;
  int runtime_version = 0;
  int clock_khz = 0;
  CUDA_CHECK(cudaDriverGetVersion(&driver_version));
  CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
  CUDA_CHECK(cudaDeviceGetAttribute(&clock_khz, cudaDevAttrClockRate, 0));
  std::fprintf(stderr, "META,schema_version,2\n");
  std::fprintf(stderr, "META,gpu,%s\n", context.gpu.c_str());
  std::fprintf(stderr, "META,cc,%s\n", context.cc.c_str());
  std::fprintf(stderr, "META,sm_count,%d\n",
               context.prop.multiProcessorCount);
  std::fprintf(stderr, "META,clock_khz,%d\n", clock_khz);
  std::fprintf(stderr, "META,warmups,%d\n", context.options.warmups);
  std::fprintf(stderr, "META,repeats,%d\n", context.options.repeats);
  std::fprintf(stderr, "META,seeds,");
  for (size_t i = 0; i < context.options.seeds.size(); ++i)
    std::fprintf(stderr, "%s%u", i ? ";" : "", context.options.seeds[i]);
  std::fprintf(stderr, "\n");
  std::fprintf(stderr,
               "META,toolchain,CUDART_VERSION=%d;CUDA_VERSION=%d;driver=%d;"
               "runtime=%d\n",
               CUDART_VERSION, CUDA_VERSION, driver_version, runtime_version);
  std::fprintf(stderr, "META,suite,%s\n", context.options.suite.c_str());
  std::fprintf(stderr, "META,descriptor,fresh_4k_global_pointer_per_repeat\n");
  std::fprintf(stderr, "META,validation,bytewise_nonuniform\n");
  std::fprintf(stderr, "META,event_ns,per_grid_iteration\n");
  std::fprintf(stderr, "META,cycles,mean_cta_latency_per_repeat\n");
}

void run_smoke_suite(const Context &context) {
  const CaseSpec spec = make_contiguous_case("size_4096", 4096);
  for (int direction = 0; direction < 3; ++direction) {
    run_cooperative_direction(context, spec, direction, 32);
    run_cooperative_direction(context, spec, direction, 128);
  }
  run_overlap_cases(context);
}

void run_core_suite(const Context &context) {
  const std::array<int, 9> movement_bytes{
      128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768};
  for (const int bytes : movement_bytes) {
    const CaseSpec spec =
        make_contiguous_case("size_" + std::to_string(bytes), bytes);
    for (int direction = 0; direction < 3; ++direction)
      run_all_methods(context, spec, direction);
  }

  const std::array<std::pair<int, int>, 5> fp32_tiles{
      std::pair{16, 16}, std::pair{32, 16}, std::pair{32, 32},
      std::pair{64, 32}, std::pair{64, 64}};
  for (const auto [rows, cols] : fp32_tiles) {
    const CaseSpec spec = make_fp32_tile_case(rows, cols);
    for (int direction = 0; direction < 3; ++direction)
      run_all_methods(context, spec, direction);
  }

  for (int rank = 1; rank <= 5; ++rank)
    run_tma_direction(context, make_rank_case(rank), 0);
  run_tma_direction(context, make_row_stride_case(), 0);
  run_tma_direction(context, make_element_stride_case(), 0);
  for (const int percent : {25, 50, 100})
    run_tma_direction(context, make_oob_case(percent), 0);
  for (const int bytes : {32, 64, 128})
    run_tma_direction(context, make_swizzle_case(bytes), 2);
  for (const int bytes : {16, 32})
    run_tma_direction(context, make_interleave_case(bytes), 2);

  for (const int outstanding : {1, 2, 4, 8})
    run_outstanding_case(context, outstanding);

  for (const int bytes : {4096, 16384, 32768}) {
    const std::array<std::pair<const char *, int>, 4> scales{
        std::pair{"1cta", 1},
        std::pair{"1xsm", context.prop.multiProcessorCount},
        std::pair{"2xsm", 2 * context.prop.multiProcessorCount},
        std::pair{"4xsm", 4 * context.prop.multiProcessorCount}};
    for (const auto &[scale, ctas] : scales) {
      CaseSpec spec = make_contiguous_case(
          "unique_" + std::to_string(bytes) + "_" + scale, bytes, ctas);
      run_all_methods(context, spec, 0);
    }
  }

  run_overlap_cases(context);
}

}  // namespace

int main(int argc, char **argv) {
  Context context;
  context.options = parse_options(argc, argv);
  CU_CHECK(cuInit(0));
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaGetDeviceProperties(&context.prop, 0));
  context.gpu = csv_safe_gpu_name(context.prop.name);
  context.cc = std::to_string(context.prop.major) + "." +
               std::to_string(context.prop.minor);
  emit_metadata(context);

  std::puts("gpu,cc,direction,method,case_id,bytes,rank,layout,outstanding,ctas,"
            "batch,repeat,cycles,event_ns,errors,descriptor_va,"
            "descriptor_fingerprint64,prior_tma_use,descriptor_l2_warm_method,"
            "tma_prefetch_before_issue,descriptor_state,pair_role");
  if (context.prop.major < 9) {
    std::fprintf(stderr, "FATAL,TMA_requires_cc_9_or_newer,cc=%s\n",
                 context.cc.c_str());
    return 2;
  }

  if (context.options.suite == "smoke")
    run_smoke_suite(context);
  else
    run_core_suite(context);  // full currently adds no unverified cases.
  CUDA_CHECK(cudaDeviceSynchronize());
  return 0;
}
