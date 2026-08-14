// S2G payload-size sweep derived from the frozen v4 multi-context probe.
// Tensor/same-map/batched/intra-CTA and command counts remain fixed; only the
// per-command byte length changes. Commands retain disjoint shared regions.

#include <cuda/barrier>
#include <cuda/ptx>
#include <cooperative_groups.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string_view>
#include <vector>

namespace cg = cooperative_groups;
using Barrier = cuda::barrier<cuda::thread_scope_block>;

#define CUDA_CHECK(expr) do { cudaError_t s_ = (expr); if (s_ != cudaSuccess) { \
  std::fprintf(stderr, "CUDA_ERROR,%s,%d,%s\n", #expr, __LINE__, cudaGetErrorString(s_)); std::exit(1); } } while (0)
#define CU_CHECK(expr) do { CUresult s_ = (expr); if (s_ != CUDA_SUCCESS) { \
  const char* t_ = nullptr; cuGetErrorString(s_, &t_); std::fprintf(stderr, "CU_ERROR,%s,%d,%s\n", #expr, __LINE__, t_ ? t_ : "unknown"); std::exit(1); } } while (0)

namespace {
constexpr int kThreads = 128;
constexpr int kMaxContexts = 32;
constexpr int kRepeats = 2;
constexpr int kWidth = 128;
constexpr int kMapPageBytes = 4096;
constexpr unsigned char kValue = 0x59;

enum Path { kBulk, kTensor };
enum Direction { kG2S, kS2G };
enum Mode { kSerial, kBatched };
enum MapMode { kSameMap, kDistinctMap };
enum Order { kAlternating, kG2SThenS2G, kS2GThenG2S };

struct Sample {
  unsigned long long issue_cycles;
  unsigned long long total_cycles;
  unsigned long long issue_global_ns;
  unsigned long long global_ns;
  unsigned int errors;
  unsigned int smid;
};

__device__ __forceinline__ unsigned long long global_timer() {
  unsigned long long value;
  asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(value));
  return value;
}
__device__ __forceinline__ unsigned int read_smid() { unsigned int value; asm("mov.u32 %0, %%smid;" : "=r"(value)); return value; }

__device__ __forceinline__ void tensor_load(
    const CUtensorMap* map, int row, void* shared, Barrier& barrier) {
  int32_t coordinates[2] = {0, row};
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_shared,
      cuda::ptx::space_global, shared, map, coordinates,
      cuda::device::barrier_native_handle(barrier));
}
__device__ __forceinline__ void tensor_store(
    const CUtensorMap* map, int row, const void* shared) {
  int32_t coordinates[2] = {0, row};
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global,
      cuda::ptx::space_shared, map, coordinates, shared);
}

__device__ __forceinline__ const CUtensorMap* select_map(
    const CUtensorMap* maps, int map_mode, int index, int contexts,
    int repeat) {
  (void)contexts;
  (void)repeat;
  const int slot = map_mode == kDistinctMap ? index : 0;
  return reinterpret_cast<const CUtensorMap*>(
      reinterpret_cast<const unsigned char*>(maps) +
      static_cast<size_t>(slot) * kMapPageBytes);
}

__device__ __forceinline__ void issue_g2s(
    int path, const CUtensorMap* maps, int map_mode, int index, int contexts,
    int repeat,
    int rows, const unsigned char* source, unsigned char* shared,
    Barrier& barrier, int bytes) {
  cuda::device::barrier_expect_tx(barrier, bytes);
  if (path == kTensor) {
    const CUtensorMap* map =
        select_map(maps, map_mode, index, contexts, repeat);
    tensor_load(map, map_mode == kSameMap ? index * rows : 0, shared, barrier);
  } else {
    cuda::ptx::cp_async_bulk(cuda::ptx::space_shared,
        cuda::ptx::space_global, shared,
        source + (static_cast<size_t>(repeat) * contexts + index) * bytes,
        bytes,
        cuda::device::barrier_native_handle(barrier));
  }
}

__device__ __forceinline__ void issue_s2g(
    int path, const CUtensorMap* maps, int map_mode, int index, int contexts,
    int repeat,
    int rows, unsigned char* destination, const unsigned char* shared,
    int bytes) {
  if (path == kTensor) {
    const CUtensorMap* map =
        select_map(maps, map_mode, index, contexts, repeat);
    tensor_store(map, map_mode == kSameMap ? index * rows : 0, shared);
  } else {
    cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
        cuda::ptx::space_shared,
        destination + (static_cast<size_t>(repeat) * contexts + index) * bytes,
        shared, bytes);
  }
  cuda::ptx::cp_async_bulk_commit_group();
}

__global__ void same_cta_kernel(
    int path, int direction, int mode, int map_mode, int contexts, int bytes,
    const CUtensorMap* maps, const unsigned char* source,
    unsigned char* destination, Sample* samples) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tiles = storage;
  auto* barriers = reinterpret_cast<Barrier*>(storage + static_cast<size_t>(contexts) * bytes);
  for (size_t byte = threadIdx.x; byte < static_cast<size_t>(contexts) * bytes; byte += blockDim.x)
    tiles[byte] = direction == kG2S ? 0 : kValue;
  if (threadIdx.x == 0 && direction == kG2S)
    for (int index = 0; index < contexts; ++index) init(&barriers[index], 1);
  __syncthreads(); cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); __syncthreads();
  const int rows = bytes / kWidth;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    if (threadIdx.x == 0) {
      Sample sample{};
      const auto begin = clock64();
      if (mode == kBatched) {
        for (int index = 0; index < contexts; ++index) {
          if (direction == kG2S)
            issue_g2s(path, maps, map_mode, index, contexts, repeat, rows, source,
                      tiles + static_cast<size_t>(index) * bytes,
                      barriers[index], bytes);
          else
            issue_s2g(path, maps, map_mode, index, contexts, repeat, rows, destination,
                      tiles + static_cast<size_t>(index) * bytes, bytes);
        }
        sample.issue_cycles = clock64() - begin;
        if (direction == kG2S)
          for (int index = 0; index < contexts; ++index) barriers[index].arrive_and_wait();
        else
          cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      } else {
        for (int index = 0; index < contexts; ++index) {
          if (direction == kG2S) {
            issue_g2s(path, maps, map_mode, index, contexts, repeat, rows, source,
                      tiles + static_cast<size_t>(index) * bytes,
                      barriers[index], bytes);
            barriers[index].arrive_and_wait();
          } else {
            issue_s2g(path, maps, map_mode, index, contexts, repeat, rows, destination,
                      tiles + static_cast<size_t>(index) * bytes, bytes);
            cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
          }
        }
        sample.issue_cycles = clock64() - begin;
      }
      sample.total_cycles = clock64() - begin;
      if (direction == kG2S)
        for (size_t byte = 0; byte < static_cast<size_t>(contexts) * bytes; ++byte)
          sample.errors += tiles[byte] != kValue;
      sample.smid = read_smid();
      samples[repeat] = sample;
    }
    __syncthreads();
  }
}

__global__ void mixed_cta_kernel(
    int path, int mode, int map_mode, int order, int pairs, int bytes,
    const CUtensorMap* load_maps, const CUtensorMap* store_maps,
    const unsigned char* source, unsigned char* destination, Sample* samples) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* load_tiles = storage;
  unsigned char* store_tiles = storage + static_cast<size_t>(pairs) * bytes;
  auto* barriers = reinterpret_cast<Barrier*>(storage + static_cast<size_t>(pairs) * bytes * 2);
  for (size_t byte = threadIdx.x; byte < static_cast<size_t>(pairs) * bytes; byte += blockDim.x) {
    load_tiles[byte] = 0; store_tiles[byte] = kValue;
  }
  if (threadIdx.x == 0) for (int index = 0; index < pairs; ++index) init(&barriers[index], 1);
  __syncthreads(); cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); __syncthreads();
  const int rows = bytes / kWidth;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    if (threadIdx.x == 0) {
      Sample sample{}; const auto begin = clock64();
      auto g2s = [&](int index) {
        issue_g2s(path, load_maps, map_mode, index, pairs, repeat, rows, source,
                  load_tiles + static_cast<size_t>(index) * bytes,
                  barriers[index], bytes);
      };
      auto s2g = [&](int index) {
        issue_s2g(path, store_maps, map_mode, index, pairs, repeat, rows, destination,
                  store_tiles + static_cast<size_t>(index) * bytes, bytes);
      };
      if (mode == kBatched) {
        if (order == kAlternating) for (int index = 0; index < pairs; ++index) { g2s(index); s2g(index); }
        else if (order == kG2SThenS2G) { for (int index = 0; index < pairs; ++index) g2s(index); for (int index = 0; index < pairs; ++index) s2g(index); }
        else { for (int index = 0; index < pairs; ++index) s2g(index); for (int index = 0; index < pairs; ++index) g2s(index); }
        sample.issue_cycles = clock64() - begin;
        cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
        for (int index = 0; index < pairs; ++index) barriers[index].arrive_and_wait();
      } else {
        for (int index = 0; index < pairs; ++index) {
          if (order == kS2GThenG2S) { s2g(index); cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{}); g2s(index); barriers[index].arrive_and_wait(); }
          else { g2s(index); barriers[index].arrive_and_wait(); s2g(index); cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{}); }
        }
        sample.issue_cycles = clock64() - begin;
      }
      sample.total_cycles = clock64() - begin;
      for (size_t byte = 0; byte < static_cast<size_t>(pairs) * bytes; ++byte) sample.errors += load_tiles[byte] != kValue;
      sample.smid = read_smid(); samples[repeat] = sample;
    }
    __syncthreads();
  }
}

__global__ void cooperative_same_kernel(
    int path, int direction, int map_mode, int bytes,
    const CUtensorMap* maps, const unsigned char* source,
    unsigned char* destination, unsigned long long* begins,
    unsigned long long* issue_ends, unsigned long long* ends,
    Sample* aggregate) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tile = storage;
  auto* barrier = reinterpret_cast<Barrier*>(storage + bytes);
  for (int byte = threadIdx.x; byte < bytes; byte += blockDim.x) tile[byte] = direction == kG2S ? 0 : kValue;
  if (threadIdx.x == 0 && direction == kG2S) init(barrier, 1);
  __syncthreads(); cuda::ptx::fence_proxy_async(cuda::ptx::space_shared); grid.sync();
  const int rows = bytes / kWidth;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    if (threadIdx.x == 0) {
      const auto global_begin = global_timer(); const auto local_begin = clock64();
      if (direction == kG2S) { issue_g2s(path, maps, map_mode, blockIdx.x, gridDim.x, repeat, rows, source, tile, *barrier, bytes); const auto issued = clock64(); issue_ends[repeat * gridDim.x + blockIdx.x] = global_timer(); barrier->arrive_and_wait(); aggregate[repeat * gridDim.x + blockIdx.x].issue_cycles = issued - local_begin; }
      else { issue_s2g(path, maps, map_mode, blockIdx.x, gridDim.x, repeat, rows, destination, tile, bytes); const auto issued = clock64(); issue_ends[repeat * gridDim.x + blockIdx.x] = global_timer(); cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{}); aggregate[repeat * gridDim.x + blockIdx.x].issue_cycles = issued - local_begin; }
      const auto global_end = global_timer(); begins[repeat * gridDim.x + blockIdx.x] = global_begin; ends[repeat * gridDim.x + blockIdx.x] = global_end;
      aggregate[repeat * gridDim.x + blockIdx.x].total_cycles = clock64() - local_begin; aggregate[repeat * gridDim.x + blockIdx.x].smid = read_smid();
      if (direction == kG2S) for (int byte = 0; byte < bytes; ++byte) aggregate[repeat * gridDim.x + blockIdx.x].errors += tile[byte] != kValue;
    }
    grid.sync();
  }
}

__global__ void cooperative_mixed_kernel(
    int path, int map_mode, int order, int bytes,
    const CUtensorMap* load_maps, const CUtensorMap* store_maps,
    const unsigned char* source, unsigned char* destination,
    unsigned long long* begins, unsigned long long* issue_ends,
    unsigned long long* ends,
    Sample* aggregate) {
  cg::grid_group grid = cg::this_grid();
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* load_tile = storage;
  unsigned char* store_tile = storage + bytes;
  auto* barrier = reinterpret_cast<Barrier*>(storage + 2 * bytes);
  for (int byte = threadIdx.x; byte < bytes; byte += blockDim.x) {
    load_tile[byte] = 0;
    store_tile[byte] = kValue;
  }
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  grid.sync();
  const int rows = bytes / kWidth;
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    if (threadIdx.x == 0) {
      Sample& sample = aggregate[repeat * gridDim.x + blockIdx.x];
      const auto global_begin = global_timer();
      const auto local_begin = clock64();
      if (order == kS2GThenG2S) {
        issue_s2g(path, store_maps, map_mode, blockIdx.x, gridDim.x, repeat, rows,
                  destination, store_tile, bytes);
        issue_g2s(path, load_maps, map_mode, blockIdx.x, gridDim.x, repeat, rows,
                  source, load_tile, *barrier, bytes);
      } else {
        issue_g2s(path, load_maps, map_mode, blockIdx.x, gridDim.x, repeat, rows,
                  source, load_tile, *barrier, bytes);
        issue_s2g(path, store_maps, map_mode, blockIdx.x, gridDim.x, repeat, rows,
                  destination, store_tile, bytes);
      }
      sample.issue_cycles = clock64() - local_begin;
      issue_ends[repeat * gridDim.x + blockIdx.x] = global_timer();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      barrier->arrive_and_wait();
      const auto global_end = global_timer();
      begins[repeat * gridDim.x + blockIdx.x] = global_begin;
      ends[repeat * gridDim.x + blockIdx.x] = global_end;
      sample.total_cycles = clock64() - local_begin;
      sample.smid = read_smid();
      for (int byte = 0; byte < bytes; ++byte)
        sample.errors += load_tile[byte] != kValue;
    }
    grid.sync();
  }
}

CUtensorMap encode_map(unsigned char* base, int bytes, int contexts, bool same) {
  CUtensorMap map{}; const int rows = bytes / kWidth;
  cuuint64_t dims[2] = {kWidth, static_cast<cuuint64_t>(rows * (same ? contexts : 1))};
  const cuuint64_t strides[1] = {kWidth}; const cuuint32_t box[2] = {kWidth, static_cast<cuuint32_t>(rows)}; const cuuint32_t element[2] = {1, 1};
  CU_CHECK(cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_UINT8, 2, base, dims, strides, box, element, CU_TENSOR_MAP_INTERLEAVE_NONE, CU_TENSOR_MAP_SWIZZLE_NONE, CU_TENSOR_MAP_L2_PROMOTION_NONE, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE)); return map;
}

uint64_t fingerprint64(const unsigned char* data, size_t bytes) {
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= data[index];
    value *= 1099511628211ull;
  }
  return value;
}

__device__ __forceinline__ uint32_t load_descriptor_cg(const void* pointer) {
  uint32_t value;
  asm volatile("ld.global.cg.u32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
  return value;
}

__global__ void warm_descriptor_pages(
    const CUtensorMap* maps, int pages, unsigned long long* sink) {
  if (threadIdx.x != 0) return;
  unsigned long long value = 0;
  const auto* base = reinterpret_cast<const unsigned char*>(maps);
  for (int page = 0; page < pages; ++page) {
    const auto* words = reinterpret_cast<const uint32_t*>(
        base + static_cast<size_t>(page) * kMapPageBytes);
    for (int word = 0; word < 32; word += 8)
      value = value * 0x9e3779b97f4a7c15ull +
              load_descriptor_cg(words + word);
  }
  *sink = value;
}

struct MapAllocation {
  CUtensorMap* pointer = nullptr;
  unsigned char* raw = nullptr;
  int per_repeat = 0;
  std::vector<uint64_t> fingerprints;
};
MapAllocation make_maps(unsigned char* base, int bytes, int contexts, int map_mode, std::vector<CUtensorMap*>& keep) {
  const int per_repeat = map_mode == kSameMap ? 1 : contexts;
  const int count = per_repeat;
  std::vector<CUtensorMap> host(count);
  uint64_t bank_fingerprint = 1469598103934665603ull;
    for (int index = 0; index < per_repeat; ++index) {
      CUtensorMap& map = host[index];
      map = encode_map(
          base + (map_mode == kDistinctMap ? index : 0) * bytes,
          bytes, contexts, map_mode == kSameMap);
      const uint64_t map_hash = fingerprint64(
          reinterpret_cast<const unsigned char*>(&map), sizeof(map));
      bank_fingerprint ^= map_hash;
      bank_fingerprint *= 1099511628211ull;
    }
  std::vector<uint64_t> fingerprints(kRepeats, bank_fingerprint);
  unsigned char* raw = nullptr;
  CUDA_CHECK(cudaMalloc(&raw,
                        static_cast<size_t>(count) * kMapPageBytes +
                            kMapPageBytes - 1));
  const uintptr_t aligned =
      (reinterpret_cast<uintptr_t>(raw) + kMapPageBytes - 1) &
      ~(static_cast<uintptr_t>(kMapPageBytes) - 1);
  auto* device = reinterpret_cast<CUtensorMap*>(aligned);
  for (int slot = 0; slot < count; ++slot)
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<unsigned char*>(device) +
            static_cast<size_t>(slot) * kMapPageBytes,
        &host[slot], sizeof(host[slot]), cudaMemcpyHostToDevice));
  keep.push_back(reinterpret_cast<CUtensorMap*>(raw));
  return {device, raw, per_repeat, fingerprints};
}

const char* path_name(int value) { return value == kTensor ? "tensor" : "bulk"; }
const char* direction_name(int value) { return value == kG2S ? "g2s" : "s2g"; }
const char* mode_name(int value) { return value == kBatched ? "batched" : "serial"; }
const char* map_name(int value) { return value == kDistinctMap ? "distinct_map" : "same_map"; }
const char* order_name(int value) { return value == kAlternating ? "alternating" : value == kG2SThenS2G ? "g2s_then_s2g" : "s2g_then_g2s"; }
}  // namespace

int main(int argc, char** argv) {
  std::string_view project = "same";
  std::string_view direction_filter = "all";
  for (int index = 1; index < argc; ++index) {
    if (std::string_view(argv[index]) == "--project" && index + 1 < argc)
      project = argv[++index];
    else if (std::string_view(argv[index]) == "--direction" && index + 1 < argc)
      direction_filter = argv[++index];
  }
  if (project != "same" && project != "mixed") { std::fprintf(stderr, "--project must be same or mixed\n"); return 2; }
  if (direction_filter != "all" && direction_filter != "g2s" && direction_filter != "s2g") {
    std::fprintf(stderr, "--direction must be all, g2s, or s2g\n"); return 2;
  }
  CUDA_CHECK(cudaSetDevice(0)); CU_CHECK(cuInit(0)); cudaDeviceProp property{}; CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  const size_t max_bytes =
      static_cast<size_t>(kRepeats) * kMaxContexts * 4096;
  unsigned char *source = nullptr, *destination = nullptr; CUDA_CHECK(cudaMalloc(&source, max_bytes)); CUDA_CHECK(cudaMalloc(&destination, max_bytes)); CUDA_CHECK(cudaMemset(source, kValue, max_bytes));
  Sample* samples = nullptr; CUDA_CHECK(cudaMalloc(&samples, kMaxContexts * kRepeats * sizeof(Sample)));
  unsigned long long *begins = nullptr, *issue_ends = nullptr, *ends = nullptr; CUDA_CHECK(cudaMalloc(&begins, kMaxContexts * kRepeats * sizeof(unsigned long long))); CUDA_CHECK(cudaMalloc(&issue_ends, kMaxContexts * kRepeats * sizeof(unsigned long long))); CUDA_CHECK(cudaMalloc(&ends, kMaxContexts * kRepeats * sizeof(unsigned long long)));
  unsigned long long* descriptor_warm_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&descriptor_warm_sink, sizeof(*descriptor_warm_sink)));
  std::vector<CUtensorMap*> maps_to_free;
  auto warm_maps = [&](const MapAllocation& maps) {
    warm_descriptor_pages<<<1, 32>>>(
        maps.pointer, maps.per_repeat, descriptor_warm_sink);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
  };
  std::puts("gpu,cc,l2_bytes,project,level,path,direction,order,mode,map_mode,bytes,contexts,commands,repeat,pair_role,issue_cycles,total_cycles,issue_span_ns,completion_span_ns,errors,smid,source_descriptor_bank_va,destination_descriptor_bank_va,source_descriptor_bank_fingerprint64,destination_descriptor_bank_fingerprint64,descriptor_pages_per_repeat,repeat_descriptor_bank_unique,prior_tma_use,descriptor_l2_warm_method,tma_prefetch_before_issue,descriptor_state");
  auto emit = [&](const char* project_name, const char* level, int path, const char* direction, const char* order, int mode, int map_mode, int bytes, int contexts, int commands, const std::vector<Sample>& host, unsigned int extra_errors, const MapAllocation* load_maps = nullptr, const MapAllocation* store_maps = nullptr) {
    for (int repeat = 0; repeat < kRepeats; ++repeat) {
      const Sample& row = host[repeat];
      const auto bank_address = [&](const MapAllocation* maps) -> uint64_t {
        return maps ? reinterpret_cast<uint64_t>(
            reinterpret_cast<const unsigned char*>(maps->pointer)) : 0;
      };
      const auto bank_fingerprint = [&](const MapAllocation* maps) -> uint64_t {
        return maps ? maps->fingerprints[repeat] : 0;
      };
      const int pages = load_maps ? load_maps->per_repeat
                        : store_maps ? store_maps->per_repeat : 0;
      std::printf("%s,%d.%d,%zu,%s,%s,%s,%s,%s,%s,%s,%d,%d,%d,%d,%s,%llu,%llu,%llu,%llu,%u,%u,0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64 ",%d,%d,%d,%s,%d,%s\n", property.name, property.major, property.minor, static_cast<size_t>(property.l2CacheSize), project_name, level, path_name(path), direction, order, mode_name(mode), path == kBulk ? "na" : map_name(map_mode), bytes, contexts, commands, repeat, repeat == 0 ? "cold" : "hot", row.issue_cycles, row.total_cycles, row.issue_global_ns, row.global_ns, row.errors + extra_errors, row.smid, bank_address(load_maps), bank_address(store_maps), bank_fingerprint(load_maps), bank_fingerprint(store_maps), pages, 0, path == kTensor ? repeat : 0, path == kTensor ? "ld.global.cg" : "not_applicable", 0, path == kTensor ? (repeat == 0 ? "tmau_cold_bank_first_use" : "tmau_hot_bank_reuse") : "not_applicable");
    }
  };
  std::vector<Sample> host(kMaxContexts * kRepeats);

  if (project == "same") {
    constexpr int path = kTensor;
    constexpr int direction = kS2G;
    constexpr int map_mode = kSameMap;
    constexpr int mode = kBatched;
    for (int bytes : {128, 256, 512, 1024, 1536, 2048, 2560, 3072,
                      3584, 4096}) {
      for (int contexts : {1, 2, 4, 8, 16, 32}) {
        MapAllocation maps = make_maps(
            destination, bytes, contexts, map_mode, maps_to_free);
        warm_maps(maps);
        CUDA_CHECK(cudaMemset(
            destination, 0,
            static_cast<size_t>(kRepeats) * contexts * bytes));
        CUDA_CHECK(cudaMemset(samples, 0, kRepeats * sizeof(Sample)));
        const size_t shared =
            static_cast<size_t>(contexts) * bytes +
            contexts * sizeof(Barrier);
        CUDA_CHECK(cudaFuncSetAttribute(
            same_cta_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(shared)));
        same_cta_kernel<<<1, kThreads, shared>>>(
            path, direction, mode, map_mode, contexts, bytes, maps.pointer,
            source, destination, samples);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host.data(), samples,
                              kRepeats * sizeof(Sample),
                              cudaMemcpyDeviceToHost));
        const size_t result_bytes =
            static_cast<size_t>(contexts) * bytes;
        std::vector<unsigned char> actual(result_bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), destination, actual.size(),
                              cudaMemcpyDeviceToHost));
        unsigned int errors = 0;
        for (auto value : actual) errors += value != kValue;
        emit("s2g_size_sweep", "intra_cta", path, "s2g", "na", mode,
             map_mode, bytes, contexts, contexts, host, errors, nullptr,
             &maps);
      }
    }
  } else {
    for (int path : {kTensor, kBulk}) for (int bytes : {128, 4096}) for (int pairs : {1, 2, 4, 8, 16}) for (int map_mode : {kSameMap, kDistinctMap}) {
      if (path == kBulk && map_mode == kDistinctMap) continue;
      for (int order : {kAlternating, kG2SThenS2G, kS2GThenG2S}) for (int mode : {kSerial, kBatched}) {
        // Order and issue mode are independent cold experiments.  Never let
        // one variant populate the TMAU entry consumed by the next variant.
        MapAllocation load_maps{};
        MapAllocation store_maps{};
        if (path == kTensor) {
          load_maps = make_maps(source, bytes, pairs, map_mode, maps_to_free);
          store_maps =
              make_maps(destination, bytes, pairs, map_mode, maps_to_free);
          warm_maps(load_maps);
          warm_maps(store_maps);
        }
        CUDA_CHECK(cudaMemset(destination, 0, static_cast<size_t>(kRepeats) * pairs * bytes)); CUDA_CHECK(cudaMemset(samples, 0, kRepeats * sizeof(Sample)));
        const size_t shared = static_cast<size_t>(pairs) * bytes * 2 + pairs * sizeof(Barrier); CUDA_CHECK(cudaFuncSetAttribute(mixed_cta_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(shared)));
        mixed_cta_kernel<<<1, kThreads, shared>>>(path, mode, map_mode, order, pairs, bytes, load_maps.pointer, store_maps.pointer, source, destination, samples); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize()); CUDA_CHECK(cudaMemcpy(host.data(), samples, kRepeats * sizeof(Sample), cudaMemcpyDeviceToHost));
        const size_t result_bytes = static_cast<size_t>(path == kTensor ? 1 : kRepeats) * pairs * bytes; std::vector<unsigned char> actual(result_bytes); CUDA_CHECK(cudaMemcpy(actual.data(), destination, actual.size(), cudaMemcpyDeviceToHost)); unsigned int errors = 0; for (auto value : actual) errors += value != kValue;
        emit("mixed_direction", "intra_cta", path, "mixed", order_name(order), mode, map_mode, bytes, pairs, pairs * 2, host, errors, path == kTensor ? &load_maps : nullptr, path == kTensor ? &store_maps : nullptr);
      }
    }
    for (int path : {kTensor, kBulk}) for (int bytes : {128, 4096})
        for (int map_mode : {kSameMap, kDistinctMap})
        for (int order : {kAlternating, kG2SThenS2G, kS2GThenG2S}) {
      if (path == kBulk && map_mode == kDistinctMap) continue;
      for (int contexts : {1, 2, 4, 8, 16}) {
        int blocks_per_sm = 0;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks_per_sm, cooperative_mixed_kernel, kThreads,
            2 * bytes + sizeof(Barrier)));
        if (contexts > blocks_per_sm * property.multiProcessorCount) continue;
        MapAllocation load_maps{};
        MapAllocation store_maps{};
        if (path == kTensor) {
          load_maps =
              make_maps(source, bytes, contexts, map_mode, maps_to_free);
          store_maps = make_maps(destination, bytes, contexts, map_mode,
                                 maps_to_free);
          warm_maps(load_maps);
          warm_maps(store_maps);
        }
        CUDA_CHECK(cudaMemset(samples, 0, contexts * kRepeats * sizeof(Sample)));
        CUDA_CHECK(cudaMemset(destination, 0, static_cast<size_t>(kRepeats) * contexts * bytes));
        void* arguments[] = {&path, &map_mode, &order, &bytes,
                             &load_maps.pointer, &store_maps.pointer,
                             &source, &destination, &begins, &issue_ends,
                             &ends, &samples};
        CUDA_CHECK(cudaLaunchCooperativeKernel(
            reinterpret_cast<void*>(cooperative_mixed_kernel), contexts,
            kThreads, arguments, 2 * bytes + sizeof(Barrier)));
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(host.data(), samples,
                              contexts * kRepeats * sizeof(Sample),
                              cudaMemcpyDeviceToHost));
        std::vector<unsigned long long> hb(contexts * kRepeats), hi(contexts * kRepeats), he(contexts * kRepeats);
        CUDA_CHECK(cudaMemcpy(hb.data(), begins, hb.size() * sizeof(*begins), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(hi.data(), issue_ends, hi.size() * sizeof(*issue_ends), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(he.data(), ends, he.size() * sizeof(*ends), cudaMemcpyDeviceToHost));
        const size_t result_bytes = static_cast<size_t>(path == kTensor ? 1 : kRepeats) * contexts * bytes;
        std::vector<unsigned char> actual(result_bytes);
        CUDA_CHECK(cudaMemcpy(actual.data(), destination, actual.size(), cudaMemcpyDeviceToHost));
        unsigned int errors = 0;
        for (auto value : actual) errors += value != kValue;
        std::vector<Sample> aggregate(kRepeats);
        for (int repeat = 0; repeat < kRepeats; ++repeat) {
          auto begin_it = hb.begin() + repeat * contexts;
          auto issue_it = hi.begin() + repeat * contexts;
          auto end_it = he.begin() + repeat * contexts;
          const auto first_begin = *std::min_element(begin_it, begin_it + contexts);
          aggregate[repeat].issue_global_ns =
              *std::max_element(issue_it, issue_it + contexts) - first_begin;
          aggregate[repeat].global_ns =
              *std::max_element(end_it, end_it + contexts) -
              first_begin;
          for (int block = 0; block < contexts; ++block) {
            const Sample& row = host[repeat * contexts + block];
            aggregate[repeat].issue_cycles =
                std::max(aggregate[repeat].issue_cycles, row.issue_cycles);
            aggregate[repeat].total_cycles =
                std::max(aggregate[repeat].total_cycles, row.total_cycles);
            aggregate[repeat].errors += row.errors;
          }
        }
        emit("mixed_direction", "multi_cta", path, "mixed",
             order_name(order), kBatched, map_mode, bytes, contexts,
             contexts * 2, aggregate, errors,
             path == kTensor ? &load_maps : nullptr,
             path == kTensor ? &store_maps : nullptr);
      }
    }
  }
  for (CUtensorMap* map : maps_to_free) CUDA_CHECK(cudaFree(map)); CUDA_CHECK(cudaFree(descriptor_warm_sink)); CUDA_CHECK(cudaFree(ends)); CUDA_CHECK(cudaFree(issue_ends)); CUDA_CHECK(cudaFree(begins)); CUDA_CHECK(cudaFree(samples)); CUDA_CHECK(cudaFree(destination)); CUDA_CHECK(cudaFree(source)); return 0;
}
