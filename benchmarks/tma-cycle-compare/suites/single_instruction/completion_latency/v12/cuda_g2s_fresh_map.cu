// G2S completion probe with one cold-then-hot TensorMap pair per delay point.
//
// V12 performs one non-adaptive scan.  Eighty statically selected kernel banks
// each contain eight predetermined delay positions.  The bank is selected
// before kernel launch, outside the timed region; inside the kernel each next
// position adds one predicate, one uniform branch and one retained PM-event.
// H100/B200 clock64 remains the authoritative physical timing record.
//
// Each delay point receives a descriptor at a unique, randomly permuted 4 KiB
// page and a unique global base.  One kernel first uses that descriptor, waits
// for completion, then immediately issues the same command with the exact same
// descriptor address and contents.  The no-prefetch entry contains no
// prefetch.tensormap instruction.

#define main tma_reference_main_disabled_v8
#include "tma_ventus_parity_bench_v2.cu"
#undef main

#include "tma_comprehensive_matrix.h"
#include "tma_model.cc"

#include <algorithm>
#include <array>
#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <set>
#include <string>
#include <string_view>
#include <vector>

#include "delay_ladder_v12.inc"

namespace {

namespace matrix = tma_comprehensive;
using NeutralCase = matrix::CaseSpec;
using ventus::tma::DecodeAndPlan;
using ventus::tma::Descriptor;
using ventus::tma::Request;
using ventus::tma::Status;

constexpr int kRepeats = 2;
constexpr int kDelayPoints = V12_DELAY_LADDER_POINTS;
constexpr int kRequestedStepCycles = V12_DELAY_LADDER_STEP;
constexpr int kMapPageBytes = 4096;
constexpr int kPrefetchLeadCycles = 1024;
constexpr unsigned char kPayloadValue = 0xa5;
constexpr unsigned int kCleanupSuspendHintNs = 4096;
constexpr uint64_t kPermutationSeed = 0x56385f4652455348ull;

enum class DescriptorMode : int {
  kNotApplicable = 0,
  kTmauFirstUseL2Hot = 1,
  kTmauPrefetchedL2Hot = 2,
};

struct ProbeSample {
  unsigned long long issue_instruction;
  unsigned long long probe_begin_from_issue_begin;
  unsigned long long probe_end_from_issue_begin;
  unsigned long long probe_instruction;
  unsigned long long payload_checksum;
  unsigned int delay_iterations;
  unsigned int probe_success;
  unsigned int cleanup_tries;
  unsigned int errors;
  unsigned int smid;
};

struct ProbeSeries {
  NeutralCase spec;
  DescriptorMode mode;
  size_t first_resource = 0;
};

struct CommandResource {
  size_t source_offset = 0;
  size_t source_span = 0;
  int map_slot = -1;
  uint64_t descriptor_fingerprint = 0;
};

const char* descriptor_mode_name(DescriptorMode value) {
  switch (value) {
    case DescriptorMode::kNotApplicable: return "not_applicable";
    case DescriptorMode::kTmauFirstUseL2Hot:
      return "tmau_first_use_l2_hot";
    case DescriptorMode::kTmauPrefetchedL2Hot:
      return "tmau_prefetched_l2_hot";
  }
  return "invalid";
}

const char* interleave_name(matrix::Interleave value) {
  switch (value) {
    case matrix::Interleave::kNone: return "none";
    case matrix::Interleave::k16B: return "16";
    case matrix::Interleave::k32B: return "32";
  }
  return "invalid";
}

const char* swizzle_name(matrix::Swizzle value) {
  switch (value) {
    case matrix::Swizzle::kNone: return "none";
    case matrix::Swizzle::k32B: return "32";
    case matrix::Swizzle::k64B: return "64";
    case matrix::Swizzle::k128B: return "128";
  }
  return "invalid";
}

CUtensorMapDataType cuda_dtype(matrix::DType value) {
  switch (value) {
    case matrix::DType::kU8: return CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case matrix::DType::kU16: return CU_TENSOR_MAP_DATA_TYPE_UINT16;
    case matrix::DType::kU32: return CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case matrix::DType::kS32: return CU_TENSOR_MAP_DATA_TYPE_INT32;
    case matrix::DType::kU64: return CU_TENSOR_MAP_DATA_TYPE_UINT64;
    case matrix::DType::kS64: return CU_TENSOR_MAP_DATA_TYPE_INT64;
    case matrix::DType::kF16: return CU_TENSOR_MAP_DATA_TYPE_FLOAT16;
    case matrix::DType::kF32: return CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case matrix::DType::kF32Ftz: return CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ;
    case matrix::DType::kF64: return CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    case matrix::DType::kBf16: return CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case matrix::DType::kTf32: return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32;
    case matrix::DType::kTf32Ftz: return CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ;
    case matrix::DType::kU4Align8:
      return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
    case matrix::DType::kU4Align16:
      return CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B;
    case matrix::DType::kU6Align16:
      return CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B;
  }
  std::abort();
}

CUtensorMapInterleave cuda_interleave(matrix::Interleave value) {
  switch (value) {
    case matrix::Interleave::kNone: return CU_TENSOR_MAP_INTERLEAVE_NONE;
    case matrix::Interleave::k16B: return CU_TENSOR_MAP_INTERLEAVE_16B;
    case matrix::Interleave::k32B: return CU_TENSOR_MAP_INTERLEAVE_32B;
  }
  std::abort();
}

CUtensorMapSwizzle cuda_swizzle(matrix::Swizzle value) {
  switch (value) {
    case matrix::Swizzle::kNone: return CU_TENSOR_MAP_SWIZZLE_NONE;
    case matrix::Swizzle::k32B: return CU_TENSOR_MAP_SWIZZLE_32B;
    case matrix::Swizzle::k64B: return CU_TENSOR_MAP_SWIZZLE_64B;
    case matrix::Swizzle::k128B: return CU_TENSOR_MAP_SWIZZLE_128B;
  }
  std::abort();
}

uint8_t model_dtype(matrix::DType value) {
  switch (value) {
    case matrix::DType::kU8: return VENTUS_TMA_DTYPE_U8;
    case matrix::DType::kU16: return VENTUS_TMA_DTYPE_U16;
    case matrix::DType::kU32: return VENTUS_TMA_DTYPE_U32;
    case matrix::DType::kS32: return VENTUS_TMA_DTYPE_S32;
    case matrix::DType::kU64: return VENTUS_TMA_DTYPE_U64;
    case matrix::DType::kS64: return VENTUS_TMA_DTYPE_S64;
    case matrix::DType::kF16: return VENTUS_TMA_DTYPE_FP16;
    case matrix::DType::kF32: return VENTUS_TMA_DTYPE_FP32;
    case matrix::DType::kF32Ftz: return VENTUS_TMA_DTYPE_FP32_FTZ;
    case matrix::DType::kF64: return VENTUS_TMA_DTYPE_FP64;
    case matrix::DType::kBf16: return VENTUS_TMA_DTYPE_BF16;
    case matrix::DType::kTf32: return VENTUS_TMA_DTYPE_TF32;
    case matrix::DType::kTf32Ftz: return VENTUS_TMA_DTYPE_TF32_FTZ;
    case matrix::DType::kU4Align8: return VENTUS_TMA_DTYPE_B4X16;
    case matrix::DType::kU4Align16: return VENTUS_TMA_DTYPE_B4X16_P64;
    case matrix::DType::kU6Align16: return VENTUS_TMA_DTYPE_B6;
  }
  std::abort();
}

CaseSpec cuda_spec(const NeutralCase& input) {
  CaseSpec output;
  output.id = input.id;
  output.layout = input.family;
  output.rank = input.rank;
  output.bytes = static_cast<int>(input.transfer_bytes);
  output.element_bytes = std::max(1, static_cast<int>(input.element_bits) / 8);
  output.data_type = cuda_dtype(input.dtype);
  output.dims = input.global_dims;
  output.strides = input.global_strides_bytes;
  output.box = input.box_dims;
  output.element_strides = input.element_strides;
  output.coords = input.coordinates;
  output.interleave = cuda_interleave(input.interleave);
  output.swizzle = cuda_swizzle(input.swizzle);
  output.shared_is_linear = input.shared_is_linear;
  return output;
}

size_t global_span(const NeutralCase& spec) {
  if (spec.path == matrix::Path::kBulk) return spec.transfer_bytes;
  if (spec.rank == 1)
    return static_cast<size_t>(
        (spec.global_dims[0] * spec.element_bits + 7) / 8);
  return static_cast<size_t>(spec.global_strides_bytes[spec.rank - 2]) *
         static_cast<size_t>(spec.global_dims[spec.rank - 1]);
}

CUresult encode_map(CUtensorMap* map, const NeutralCase& neutral, void* base) {
  const CaseSpec spec = cuda_spec(neutral);
  return cuTensorMapEncodeTiled(
      map, spec.data_type, static_cast<cuuint32_t>(spec.rank), base,
      spec.dims.data(), spec.strides.data(), spec.box.data(),
      spec.element_strides.data(), spec.interleave, spec.swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      neutral.oob_fill == matrix::OobFill::kNan
          ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
          : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

Descriptor model_descriptor(const NeutralCase& spec) {
  Descriptor descriptor{};
  descriptor.words[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
  descriptor.words[VENTUS_TMA_V2_WORD_CONTROL] =
      model_dtype(spec.dtype) | (static_cast<uint32_t>(spec.rank) << 5) |
      (static_cast<uint32_t>(spec.interleave) << 8) |
      (static_cast<uint32_t>(spec.swizzle) << 10) |
      (static_cast<uint32_t>(spec.oob_fill) << 18);
  for (int d = 0; d < spec.rank; ++d) {
    descriptor.words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + d] =
        static_cast<uint32_t>(spec.global_dims[d]);
    descriptor.words[VENTUS_TMA_V2_WORD_BOX_DIMS + d] = spec.box_dims[d];
    descriptor.words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + d] =
        spec.element_strides[d];
    if (d + 1 < spec.rank) {
      const uint64_t stride = spec.global_strides_bytes[d];
      const int word = VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + d * 2;
      descriptor.words[word] = static_cast<uint32_t>(stride);
      descriptor.words[word + 1] = static_cast<uint32_t>(stride >> 32);
    }
  }
  return descriptor;
}

ventus::tma::Result model_plan(const NeutralCase& spec) {
  Request request{};
  request.direction = VENTUS_TMA_G2S;
  request.shared_base = 0;
  request.coordinates = spec.coordinates;
  return DecodeAndPlan(model_descriptor(spec), request);
}

uint32_t tf32_rna(uint32_t bits) {
  const uint32_t exponent = bits & 0x7f800000u;
  const uint32_t fraction = bits & 0x007fffffu;
  if (exponent == 0x7f800000u) {
    if (fraction == 0) return bits;
    return 0x7fffe000u;
  }
  return (bits + 0x00001000u) & 0xffffe000u;
}

uint64_t expected_checksum(const NeutralCase& spec) {
  if (spec.path == matrix::Path::kBulk)
    return static_cast<uint64_t>(spec.transfer_bytes) * kPayloadValue;
  const auto plan = model_plan(spec);
  if (plan.status != Status::kOk) {
    std::fprintf(stderr, "MODEL_REJECTED,%s,%s\n", spec.id.c_str(),
                 plan.detail.c_str());
    std::exit(EXIT_FAILURE);
  }
  std::vector<uint8_t> shared(spec.transfer_bytes, 0);
  for (const auto& atom : plan.atoms) {
    for (unsigned shared_lane = 0; shared_lane < 16; ++shared_lane) {
      if (((atom.shared_mask >> shared_lane) & 1u) == 0) continue;
      const size_t offset = atom.shared_atom + shared_lane;
      if (!atom.fill) {
        shared[offset] = kPayloadValue;
      }
    }
  }
  if (spec.dtype == matrix::DType::kTf32 ||
      spec.dtype == matrix::DType::kTf32Ftz) {
    for (size_t offset = 0; offset + 4 <= shared.size(); offset += 4) {
      uint32_t word = static_cast<uint32_t>(shared[offset]) |
                      (static_cast<uint32_t>(shared[offset + 1]) << 8) |
                      (static_cast<uint32_t>(shared[offset + 2]) << 16) |
                      (static_cast<uint32_t>(shared[offset + 3]) << 24);
      word = tf32_rna(word);
      for (unsigned byte = 0; byte < 4; ++byte)
        shared[offset + byte] = static_cast<uint8_t>(word >> (8 * byte));
    }
  }
  if (spec.oob_fill == matrix::OobFill::kNan) {
    // CUDA materializes its documented 16-bit special NaN pattern after the
    // TF32 movement conversion.  Keeping this order matches the established
    // common-matrix byte-exact validator.
    for (const auto& atom : plan.atoms) {
      if (!atom.fill) continue;
      for (unsigned lane = 0; lane < 16; ++lane) {
        if (((atom.shared_mask >> lane) & 1u) == 0) continue;
        const size_t offset = atom.shared_atom + lane;
        shared[offset] = (offset & 1u) == 0 ? 0xf7 : 0x7f;
      }
    }
  }
  return std::accumulate(shared.begin(), shared.end(), uint64_t{0});
}

uint64_t fingerprint64(const void* pointer, size_t bytes) {
  const auto* data = static_cast<const uint8_t*>(pointer);
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= data[index];
    value *= 1099511628211ull;
  }
  return value;
}

size_t align_up(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

__device__ __forceinline__ unsigned int read_smid_v8() {
  unsigned int value;
  asm volatile("mov.u32 %0, %%smid;" : "=r"(value));
  return value;
}

__device__ __forceinline__ unsigned int load_cg_v8(const void* pointer) {
  unsigned int value;
  asm volatile("ld.global.cg.u32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
  return value;
}

__device__ __forceinline__ void acquire_map_v8(const CUtensorMap* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;"
               :: "l"(map) : "memory");
}

__device__ __forceinline__ void prefetch_map_v8(const CUtensorMap* map) {
  asm volatile("prefetch.tensormap [%0];" :: "l"(map) : "memory");
}

__device__ __forceinline__ void wait_cycles_v8(unsigned long long cycles) {
  const unsigned long long begin = clock64();
  while (clock64() - begin < cycles) asm volatile("" ::: "memory");
}

__device__ __forceinline__ unsigned long long arrive_expect_tx_v8(
    BlockBarrier* barrier, unsigned int bytes) {
  const unsigned int address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned long long token;
  asm volatile(
      "mbarrier.arrive.expect_tx.release.cta.shared::cta.b64 %0, [%1], %2;"
      : "=l"(token) : "r"(address), "r"(bytes) : "memory");
  return token;
}

__device__ __forceinline__ bool raw_test_wait_v8(
    BlockBarrier* barrier, unsigned long long token) {
  const unsigned int address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned int complete;
  asm volatile("{ .reg .pred p;\n\t"
               "mbarrier.test_wait.acquire.cta.shared::cta.b64 p, [%1], %2;\n\t"
               "selp.b32 %0, 1, 0, p; }"
               : "=r"(complete) : "r"(address), "l"(token) : "memory");
  return complete != 0;
}

__device__ __forceinline__ bool cleanup_try_wait_v8(
    BlockBarrier* barrier, unsigned long long token) {
  const unsigned int address =
      static_cast<unsigned int>(__cvta_generic_to_shared(barrier));
  unsigned int complete;
  asm volatile("{ .reg .pred p;\n\t"
               "mbarrier.try_wait.acquire.cta.shared::cta.b64 "
               "p, [%1], %2, %3;\n\t"
               "selp.b32 %0, 1, 0, p; }"
               : "=r"(complete)
               : "r"(address), "l"(token), "r"(kCleanupSuspendHintNs)
               : "memory");
  return complete != 0;
}

__device__ __forceinline__ unsigned int cleanup_after_false_v8(
    BlockBarrier* barrier, unsigned long long token) {
  unsigned int tries = 0;
  bool complete = false;
  do {
    ++tries;
    complete = cleanup_try_wait_v8(barrier, token);
  } while (!complete);
  return tries;
}

template <int Bank>
__device__ __forceinline__ unsigned int static_delay_ladder_v12(
    unsigned int slot, unsigned int value) {
  // Bank is fixed by the kernel symbol and slot is host-validated to [0, 8).
  // Host-side bank selection is outside issue_begin/probe_begin.  None of the
  // retained PM-events read or write L2/shared/TensorMap state, and no result
  // ever changes which of the 640 predetermined points is executed.
#define V12_INVOKE_DELAY_BANK(N)                                             \
  if constexpr (Bank == N)                                                  \
    asm volatile(V12_DELAY_BANK_##N##_PTX                                   \
                 : "+r"(value) : "r"(slot) : "memory");
  V12_FOR_EACH_DELAY_BANK(V12_INVOKE_DELAY_BANK)
#undef V12_INVOKE_DELAY_BANK
  return value;
}

__device__ __forceinline__ BlockBarrier* dependent_barrier_v8(
    BlockBarrier* barrier, unsigned int delay_result,
    unsigned int runtime_zero) {
  unsigned int offset;
  // runtime_zero is passed as zero by the host, so the address is unchanged.
  // Keeping it as a kernel parameter prevents ptxas from proving that the
  // delay result is dead and deleting the instruction loop from final SASS.
  asm volatile("and.b32 %0, %1, %2;"
               : "=r"(offset) : "r"(delay_result), "r"(runtime_zero));
  return reinterpret_cast<BlockBarrier*>(
      reinterpret_cast<unsigned char*>(barrier) + offset);
}

__device__ __forceinline__ void issue_tensor_rank_v8(
    int rank, const CUtensorMap* map, void* shared,
    const DeviceCoords& coords, BlockBarrier& barrier) {
  switch (rank) {
    case 1: issue_tma_load<1>(map, shared, coords, barrier); break;
    case 2: issue_tma_load<2>(map, shared, coords, barrier); break;
    case 3: issue_tma_load<3>(map, shared, coords, barrier); break;
    case 4: issue_tma_load<4>(map, shared, coords, barrier); break;
    case 5: issue_tma_load<5>(map, shared, coords, barrier); break;
    default: asm volatile("trap;");
  }
}

__global__ void warm_payload_and_descriptor_v8(
    const unsigned char* source, int source_span, const CUtensorMap* map,
    int warm_descriptor, unsigned long long* sink) {
  if (threadIdx.x != 0) return;
  unsigned long long value = 0;
  for (int offset = 0; offset < source_span; offset += 32)
    value = value * 0x9e3779b97f4a7c15ull + load_cg_v8(source + offset);
  if (warm_descriptor) {
    const auto* words = reinterpret_cast<const unsigned int*>(map);
    for (int word = 0; word < 32; word += 8)
      value = value * 0x9e3779b97f4a7c15ull + load_cg_v8(words + word);
  }
  *sink = value;
}

__global__ void evict_l2_v8(
    const unsigned char* data, size_t bytes, unsigned long long* sink) {
  unsigned long long value = 0;
  for (size_t offset = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
       offset * 32 < bytes;
       offset += static_cast<size_t>(gridDim.x) * blockDim.x)
    value += load_cg_v8(data + offset * 32);
  if (value && blockIdx.x == 0 && threadIdx.x == 0) *sink = value;
}

template <bool DoPrefetch, int Bank>
__device__ __forceinline__ void tensor_probe_body_v8(
    const CUtensorMap* map, int rank, DeviceCoords coords, int bytes,
    unsigned int delay_iterations, unsigned long long expected_checksum,
    unsigned int runtime_zero, ProbeSample* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tile = storage;
  BlockBarrier* barrier = barrier_after(tile, static_cast<size_t>(bytes));
  for (int byte = threadIdx.x; byte < bytes; byte += blockDim.x)
    tile[byte] = 0;
  __syncthreads();
  if (threadIdx.x != 0) return;
  // Establish proxy coherency exactly once.  Repeating this acquire before
  // repeat 1 emits a second UTMACCTL.IV and can destroy the TMAU-hot state we
  // are explicitly trying to measure.
  acquire_map_v8(map);
  if constexpr (DoPrefetch) {
    prefetch_map_v8(map);
    wait_cycles_v8(kPrefetchLeadCycles);
  }
  // Keep one static instruction site and execute it exactly twice.  This
  // avoids a subroutine-call overhead before the earliest probe while the
  // output index and host contract prove the cold-then-hot runtime pair.
#pragma unroll 1
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    init(barrier, 1);
    const unsigned long long token =
        arrive_expect_tx_v8(barrier, static_cast<unsigned int>(bytes));
    const unsigned long long issue_begin = clock64();
    issue_tensor_rank_v8(rank, map, tile, coords, *barrier);
    const unsigned long long issue_end = clock64();
    const unsigned int phase_result = static_delay_ladder_v12<Bank>(
        delay_iterations, static_cast<unsigned int>(expected_checksum));
    BlockBarrier* observed_barrier = dependent_barrier_v8(
        barrier, phase_result, runtime_zero);
    const unsigned long long probe_begin = clock64();
    const bool complete = raw_test_wait_v8(observed_barrier, token);
    const unsigned long long probe_end = clock64();

    ProbeSample sample{};
    sample.delay_iterations =
        Bank * V12_DELAY_LADDER_BANK_SIZE + delay_iterations;
    sample.issue_instruction = issue_end - issue_begin;
    sample.probe_begin_from_issue_begin = probe_begin - issue_begin;
    sample.probe_end_from_issue_begin = probe_end - issue_begin;
    sample.probe_instruction = probe_end - probe_begin;
    sample.probe_success = complete;
    if (!complete)
      sample.cleanup_tries = cleanup_after_false_v8(observed_barrier, token);
    for (int byte = 0; byte < bytes; ++byte)
      sample.payload_checksum += static_cast<volatile unsigned char*>(tile)[byte];
    sample.errors = sample.payload_checksum != expected_checksum;
    sample.smid = read_smid_v8();
    output[repeat] = sample;
  }
}

template <int Bank>
__device__ __forceinline__ void bulk_probe_body_v12(
    const unsigned char* source, int bytes, unsigned int delay_iterations,
    unsigned long long expected_checksum, unsigned int runtime_zero,
    ProbeSample* output) {
  extern __shared__ __align__(128) unsigned char storage[];
  unsigned char* tile = storage;
  BlockBarrier* barrier = barrier_after(tile, static_cast<size_t>(bytes));
  for (int byte = threadIdx.x; byte < bytes; byte += blockDim.x)
    tile[byte] = 0;
  __syncthreads();
  if (threadIdx.x != 0) return;
#pragma unroll 1
  for (int repeat = 0; repeat < kRepeats; ++repeat) {
    init(barrier, 1);
    const unsigned long long token =
        arrive_expect_tx_v8(barrier, static_cast<unsigned int>(bytes));
    const unsigned long long issue_begin = clock64();
    cuda::ptx::cp_async_bulk(
        cuda::ptx::space_shared, cuda::ptx::space_global, tile, source,
        static_cast<unsigned int>(bytes),
        cuda::device::barrier_native_handle(*barrier));
    const unsigned long long issue_end = clock64();
    const unsigned int phase_result = static_delay_ladder_v12<Bank>(
        delay_iterations, static_cast<unsigned int>(expected_checksum));
    BlockBarrier* observed_barrier = dependent_barrier_v8(
        barrier, phase_result, runtime_zero);
    const unsigned long long probe_begin = clock64();
    const bool complete = raw_test_wait_v8(observed_barrier, token);
    const unsigned long long probe_end = clock64();

    ProbeSample sample{};
    sample.delay_iterations =
        Bank * V12_DELAY_LADDER_BANK_SIZE + delay_iterations;
    sample.issue_instruction = issue_end - issue_begin;
    sample.probe_begin_from_issue_begin = probe_begin - issue_begin;
    sample.probe_end_from_issue_begin = probe_end - issue_begin;
    sample.probe_instruction = probe_end - probe_begin;
    sample.probe_success = complete;
    if (!complete)
      sample.cleanup_tries = cleanup_after_false_v8(observed_barrier, token);
    for (int byte = 0; byte < bytes; ++byte)
      sample.payload_checksum += static_cast<volatile unsigned char*>(tile)[byte];
    sample.errors = sample.payload_checksum != expected_checksum;
    sample.smid = read_smid_v8();
    output[repeat] = sample;
  }
}

template <int Bank>
__global__ void completion_tensor_no_prefetch_probe_v12(
    const CUtensorMap* map, int rank, DeviceCoords coords, int bytes,
    unsigned int delay_iterations, unsigned long long expected_checksum,
    unsigned int runtime_zero, ProbeSample* output) {
  tensor_probe_body_v8<false, Bank>(
      map, rank, coords, bytes, delay_iterations, expected_checksum,
      runtime_zero, output);
}

template <int Bank>
__global__ void completion_tensor_explicit_prefetch_probe_v12(
    const CUtensorMap* map, int rank, DeviceCoords coords, int bytes,
    unsigned int delay_iterations, unsigned long long expected_checksum,
    unsigned int runtime_zero, ProbeSample* output) {
  tensor_probe_body_v8<true, Bank>(
      map, rank, coords, bytes, delay_iterations, expected_checksum,
      runtime_zero, output);
}

template <int Bank>
__global__ void completion_bulk_probe_v12(
    const unsigned char* source, int bytes, unsigned int delay_iterations,
    unsigned long long expected_checksum, unsigned int runtime_zero,
    ProbeSample* output) {
  bulk_probe_body_v12<Bank>(
      source, bytes, delay_iterations, expected_checksum, runtime_zero,
      output);
}

std::vector<NeutralCase> selected_cases(std::string_view project) {
  std::vector<NeutralCase> output;
  if (project == "common") {
    const std::set<std::string> safe_shards{
        "bulk", "tensor_capacity", "tensor_geometry", "dtype", "layout",
        "interleave16"};
    for (const NeutralCase& spec : matrix::make_cases()) {
      if (spec.direction != matrix::Direction::kG2s ||
          (spec.path != matrix::Path::kBulk &&
           spec.path != matrix::Path::kTensor) ||
          !safe_shards.count(spec.shard))
        continue;
      output.push_back(spec);
    }
  } else if (project == "tensor_2d") {
    for (const NeutralCase& spec : matrix::make_tensor_2d_length_probe_cases())
      if (spec.direction == matrix::Direction::kG2s)
        output.push_back(spec);
  } else if (project == "smoke") {
    const std::set<std::string> smoke_ids{
        "bulk_capacity_g2s_b128", "tensor_capacity_g2s_b128",
        "dtype_g2s_tf32_b128"};
    for (const NeutralCase& spec : matrix::make_cases())
      if (smoke_ids.count(spec.id)) output.push_back(spec);
  } else {
    std::fprintf(stderr, "unknown project: %.*s\n",
                 static_cast<int>(project.size()), project.data());
    std::exit(2);
  }
  const size_t expected = project == "common" ? 130 :
                          project == "tensor_2d" ? 91 : 3;
  if (output.size() != expected) {
    std::fprintf(stderr, "case-count mismatch: %zu expected %zu\n",
                 output.size(), expected);
    std::exit(2);
  }
  return output;
}

std::vector<ProbeSeries> make_series(std::string_view project) {
  std::vector<ProbeSeries> output;
  for (const NeutralCase& spec : selected_cases(project)) {
    if (spec.path == matrix::Path::kBulk) {
      output.push_back({spec, DescriptorMode::kNotApplicable, 0});
    } else {
      output.push_back({spec, DescriptorMode::kTmauFirstUseL2Hot, 0});
      output.push_back({spec, DescriptorMode::kTmauPrefetchedL2Hot, 0});
    }
  }
  const size_t expected = project == "common" ? 243 :
                          project == "tensor_2d" ? 182 : 5;
  if (output.size() != expected) {
    std::fprintf(stderr, "series-count mismatch: %zu expected %zu\n",
                 output.size(), expected);
    std::exit(2);
  }
  return output;
}

void print_list(std::string_view project) {
  std::puts("project,base_case_id,family,path,dtype,rank,logical_bytes,descriptor_mode");
  for (const ProbeSeries& series : make_series(project))
    std::printf("%.*s,%s,%s,%s,%s,%d,%u,%s\n",
                static_cast<int>(project.size()), project.data(),
                series.spec.id.c_str(), series.spec.family.c_str(),
                matrix::path_name(series.spec.path),
                matrix::dtype_name(series.spec.dtype), series.spec.rank,
                series.spec.transfer_bytes, descriptor_mode_name(series.mode));
}

}  // namespace

int main(int argc, char** argv) {
  static_assert(sizeof(CUtensorMap) == 128,
                "TensorMap must occupy one 128-byte cache line");
  std::string project = "common";
  int shard_index = 0;
  int shard_count = 1;
  bool list_only = false;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    if (argument == "--project" && index + 1 < argc)
      project = argv[++index];
    else if (argument == "--shard-index" && index + 1 < argc)
      shard_index = std::atoi(argv[++index]);
    else if (argument == "--shard-count" && index + 1 < argc)
      shard_count = std::atoi(argv[++index]);
    else if (argument == "--list")
      list_only = true;
    else {
      std::fprintf(stderr, "unknown argument: %s\n", argv[index]);
      return 2;
    }
  }
  if (list_only) {
    print_list(project);
    return 0;
  }
  if (shard_count < 1 || shard_index < 0 || shard_index >= shard_count) {
    std::fprintf(stderr, "invalid shard %d/%d\n", shard_index, shard_count);
    return 2;
  }
  CU_CHECK(cuInit(0));
  CUDA_CHECK(cudaSetDevice(0));
  cudaDeviceProp property{};
  CUDA_CHECK(cudaGetDeviceProperties(&property, 0));
  if (property.major < 9) return 2;

  const std::vector<ProbeSeries> all_series = make_series(project);
  std::vector<ProbeSeries> series;
  for (size_t index = 0; index < all_series.size(); ++index)
    if (static_cast<int>(index % static_cast<size_t>(shard_count)) ==
        shard_index)
      series.push_back(all_series[index]);
  const size_t commands_per_series = kDelayPoints;
  std::vector<CommandResource> resources(series.size() * commands_per_series);
  size_t source_bytes = 0;
  size_t tensor_command_count = 0;
  for (size_t series_index = 0; series_index < series.size(); ++series_index) {
    ProbeSeries& current = series[series_index];
    current.first_resource = series_index * commands_per_series;
    const size_t span = global_span(current.spec);
    for (size_t local = 0; local < commands_per_series; ++local) {
      CommandResource& resource = resources[current.first_resource + local];
      source_bytes = align_up(source_bytes, 128);
      resource.source_offset = source_bytes;
      resource.source_span = span;
      source_bytes += align_up(span + current.spec.global_base_mod128, 128);
      if (current.spec.path == matrix::Path::kTensor)
        resource.map_slot = static_cast<int>(tensor_command_count++);
    }
  }

  std::vector<int> permutation(tensor_command_count);
  std::iota(permutation.begin(), permutation.end(), 0);
  const uint64_t shard_seed =
      kPermutationSeed ^ (static_cast<uint64_t>(shard_index) << 56);
  std::mt19937_64 generator(shard_seed);
  std::shuffle(permutation.begin(), permutation.end(), generator);
  for (CommandResource& resource : resources)
    if (resource.map_slot >= 0)
      resource.map_slot = permutation[resource.map_slot];
  {
    std::set<int> map_slots;
    for (const CommandResource& resource : resources)
      if (resource.map_slot >= 0) map_slots.insert(resource.map_slot);
    if (map_slots.size() != tensor_command_count) {
      std::fprintf(stderr, "TensorMap address allocation is not one-to-one\n");
      return 2;
    }
  }

  unsigned char* device_source = nullptr;
  unsigned char* device_map_allocation = nullptr;
  unsigned char* device_map_pages = nullptr;
  ProbeSample* device_samples = nullptr;
  unsigned long long* device_sink = nullptr;
  CUDA_CHECK(cudaMalloc(&device_source, source_bytes));
  CUDA_CHECK(cudaMemset(device_source, kPayloadValue, source_bytes));
  CUDA_CHECK(cudaMalloc(
      &device_map_allocation,
      tensor_command_count * kMapPageBytes + kMapPageBytes - 1));
  const uintptr_t device_map_aligned =
      (reinterpret_cast<uintptr_t>(device_map_allocation) +
       kMapPageBytes - 1) & ~(static_cast<uintptr_t>(kMapPageBytes) - 1);
  device_map_pages = reinterpret_cast<unsigned char*>(device_map_aligned);
  CUDA_CHECK(cudaMalloc(
      &device_samples, resources.size() * kRepeats * sizeof(ProbeSample)));
  CUDA_CHECK(cudaMemset(
      device_samples, 0,
      resources.size() * kRepeats * sizeof(ProbeSample)));
  CUDA_CHECK(cudaMalloc(&device_sink, sizeof(unsigned long long)));

  void* host_map_storage = nullptr;
  if (posix_memalign(&host_map_storage, kMapPageBytes,
                     tensor_command_count * kMapPageBytes) != 0)
    return 2;
  std::memset(host_map_storage, 0, tensor_command_count * kMapPageBytes);
  auto* host_map_pages = static_cast<unsigned char*>(host_map_storage);
  for (const ProbeSeries& current : series) {
    if (current.spec.path != matrix::Path::kTensor) continue;
    for (size_t local = 0; local < commands_per_series; ++local) {
      CommandResource& resource = resources[current.first_resource + local];
      auto* map = reinterpret_cast<CUtensorMap*>(
          host_map_pages + static_cast<size_t>(resource.map_slot) *
                               kMapPageBytes);
      void* source = device_source + resource.source_offset +
                     current.spec.global_base_mod128;
      const CUresult status = encode_map(map, current.spec, source);
      if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "ENCODE_REJECTED,%s,%d\n",
                     current.spec.id.c_str(), static_cast<int>(status));
        return 1;
      }
      resource.descriptor_fingerprint = fingerprint64(map, sizeof(*map));
    }
  }
  {
    std::set<uint64_t> descriptor_fingerprints;
    std::set<size_t> global_bases;
    for (const CommandResource& resource : resources) {
      global_bases.insert(resource.source_offset);
      if (resource.map_slot >= 0)
        descriptor_fingerprints.insert(resource.descriptor_fingerprint);
    }
    if (global_bases.size() != resources.size() ||
        descriptor_fingerprints.size() != tensor_command_count) {
      std::fprintf(stderr,
                   "per-command global base or TensorMap contents are reused\n");
      return 2;
    }
  }
  CUDA_CHECK(cudaMemcpy(
      device_map_pages, host_map_pages,
      tensor_command_count * kMapPageBytes, cudaMemcpyHostToDevice));

  const size_t eviction_bytes = align_up(
      static_cast<size_t>(property.l2CacheSize) * 2 + (1u << 20), 128);
  unsigned char* eviction = nullptr;
  CUDA_CHECK(cudaMalloc(&eviction, eviction_bytes));
  CUDA_CHECK(cudaMemset(eviction, 0x3c, eviction_bytes));
  evict_l2_v8<<<256, 256>>>(eviction, eviction_bytes, device_sink);
  CUDA_CHECK(cudaGetLastError());
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<unsigned char> prior_tma_uses(tensor_command_count, 0);
  for (size_t series_index = 0; series_index < series.size(); ++series_index) {
    const ProbeSeries& current = series[series_index];
    const CaseSpec spec = cuda_spec(current.spec);
    const DeviceCoords coords = device_coords(spec.coords);
    const size_t shared_bytes = barrier_shared_bytes(current.spec.transfer_bytes);
    const uint64_t checksum = expected_checksum(current.spec);
    // Execute the one fixed scan in descending order.  The first, longest
    // ladder walk brings the measurement code itself into the instruction
    // cache before we approach any completion boundary.  It does not touch a
    // TensorMap other than the unique map belonging to that measurement point.
    for (int delay = kDelayPoints - 1; delay >= 0; --delay) {
        const int delay_bank = delay / V12_DELAY_LADDER_BANK_SIZE;
        const unsigned int delay_slot =
            static_cast<unsigned int>(delay % V12_DELAY_LADDER_BANK_SIZE);
        const size_t resource_index =
            current.first_resource + static_cast<size_t>(delay);
        const CommandResource& resource = resources[resource_index];
        const auto* source = device_source + resource.source_offset +
                             current.spec.global_base_mod128;
        const CUtensorMap* map = resource.map_slot < 0
            ? nullptr
            : reinterpret_cast<const CUtensorMap*>(
                  device_map_pages + static_cast<size_t>(resource.map_slot) *
                                         kMapPageBytes);
        if (resource.map_slot >= 0 && prior_tma_uses[resource.map_slot] != 0) {
          std::fprintf(stderr, "TensorMap was already used by TMA: slot=%d\n",
                       resource.map_slot);
          return 2;
        }
        // This is an ordinary global load, not a TensorMap prefetch.  It makes
        // the descriptor an L2 hit while leaving it completely unseen by the
        // TMA unit.  Every command uses a different descriptor address and
        // contents.  The following probe kernel first-uses this TensorMap and
        // then immediately reuses it for the hot sample.
        const int warm_descriptor =
            current.spec.path == matrix::Path::kTensor;
        warm_payload_and_descriptor_v8<<<1, 32>>>(
            source, static_cast<int>(resource.source_span), map,
            warm_descriptor, device_sink);
        if (current.spec.path == matrix::Path::kBulk) {
#define V12_LAUNCH_BULK_BANK(N)                                              \
          case N:                                                            \
            completion_bulk_probe_v12<N><<<1, 32, shared_bytes>>>(           \
                source, current.spec.transfer_bytes, delay_slot, checksum,    \
                0u, device_samples + resource_index * kRepeats);             \
            break;
          switch (delay_bank) {
            V12_FOR_EACH_DELAY_BANK(V12_LAUNCH_BULK_BANK)
            default: std::abort();
          }
#undef V12_LAUNCH_BULK_BANK
        } else if (current.mode == DescriptorMode::kTmauPrefetchedL2Hot) {
#define V12_LAUNCH_PREFETCH_BANK(N)                                          \
          case N:                                                            \
            completion_tensor_explicit_prefetch_probe_v12<N>                \
                <<<1, 32, shared_bytes>>>(                                   \
                    map, spec.rank, coords, spec.bytes, delay_slot, checksum, \
                    0u, device_samples + resource_index * kRepeats);         \
            break;
          switch (delay_bank) {
            V12_FOR_EACH_DELAY_BANK(V12_LAUNCH_PREFETCH_BANK)
            default: std::abort();
          }
#undef V12_LAUNCH_PREFETCH_BANK
        } else {
#define V12_LAUNCH_TENSOR_BANK(N)                                            \
          case N:                                                            \
            completion_tensor_no_prefetch_probe_v12<N>                      \
                <<<1, 32, shared_bytes>>>(                                   \
                    map, spec.rank, coords, spec.bytes, delay_slot, checksum, \
                    0u, device_samples + resource_index * kRepeats);         \
            break;
          switch (delay_bank) {
            V12_FOR_EACH_DELAY_BANK(V12_LAUNCH_TENSOR_BANK)
            default: std::abort();
          }
#undef V12_LAUNCH_TENSOR_BANK
        }
        CUDA_CHECK(cudaGetLastError());
        if (resource.map_slot >= 0) prior_tma_uses[resource.map_slot] = 2;
    }
  }
  CUDA_CHECK(cudaDeviceSynchronize());

  std::vector<ProbeSample> samples(resources.size() * kRepeats);
  CUDA_CHECK(cudaMemcpy(samples.data(), device_samples,
                        samples.size() * sizeof(ProbeSample),
                        cudaMemcpyDeviceToHost));
  std::puts(
      "gpu,cc,l2_bytes,project,family,base_case_id,case_id,path,direction,dtype,"
      "dtype_bits,rank,logical_bytes,global_span,global_offset,interleave,"
      "swizzle,oob_fill,descriptor_mode,payload_state,wait_method,repeat,"
      "delay_iterations,scan_index,requested_delay_cycles,"
      "cuda_context_shard,issue_instruction_cycles,"
      "probe_begin_from_issue_begin_cycles,probe_end_from_issue_begin_cycles,"
      "probe_instruction_cycles,probe_success,cleanup_tries,errors,smid,"
      "descriptor_va,global_base_va,descriptor_fingerprint64,prior_tma_use,"
      "pair_role,descriptor_state,descriptor_l2_warm_method,"
      "tma_prefetch_before_issue,map_page_bytes,"
      "permutation_seed,expected_payload_checksum,payload_checksum");
  for (const ProbeSeries& current : series) {
    const uint64_t expected = expected_checksum(current.spec);
    for (int delay = 0; delay < kDelayPoints; ++delay) {
      for (int repeat = 0; repeat < kRepeats; ++repeat) {
        const size_t resource_index =
            current.first_resource + static_cast<size_t>(delay);
        const CommandResource& resource = resources[resource_index];
        const ProbeSample& sample =
            samples[resource_index * kRepeats + repeat];
        const uint64_t descriptor_va = resource.map_slot < 0 ? 0 :
            reinterpret_cast<uint64_t>(device_map_pages) +
                static_cast<uint64_t>(resource.map_slot) * kMapPageBytes;
        const uint64_t global_base_va =
            reinterpret_cast<uint64_t>(device_source) + resource.source_offset +
            current.spec.global_base_mod128;
        const std::string case_id = current.spec.id + "__" +
            descriptor_mode_name(current.mode) + "__scan" +
            std::to_string(delay);
        const int requested_delay_cycles = delay * kRequestedStepCycles;
        std::printf(
            "%s,%d.%d,%zu,%s,%s,%s,%s,%s,g2s,%s,%u,%d,%u,%zu,%u,%s,%s,%s,"
            "%s,l2_hot,single_test_wait_cold_then_hot_v12,%d,%u,%d,%d,%d,%llu,%llu,%llu,"
            "%llu,%u,%u,%u,%u,0x%016" PRIx64 ",0x%016" PRIx64
            ",0x%016" PRIx64 ",%d,%s,%s,%s,%d,%d,0x%016" PRIx64
            ",%" PRIu64 ",%llu\n",
            property.name, property.major, property.minor,
            static_cast<size_t>(property.l2CacheSize), project.c_str(),
            current.spec.family.c_str(), current.spec.id.c_str(),
            case_id.c_str(), matrix::path_name(current.spec.path),
            matrix::dtype_name(current.spec.dtype), current.spec.element_bits,
            current.spec.rank, current.spec.transfer_bytes,
            global_span(current.spec), current.spec.global_base_mod128,
            interleave_name(current.spec.interleave),
            swizzle_name(current.spec.swizzle),
            matrix::oob_name(current.spec.oob_fill),
            descriptor_mode_name(current.mode), repeat,
            sample.delay_iterations, delay, requested_delay_cycles,
            shard_index,
            sample.issue_instruction,
            sample.probe_begin_from_issue_begin,
            sample.probe_end_from_issue_begin, sample.probe_instruction,
            sample.probe_success, sample.cleanup_tries, sample.errors,
            sample.smid, descriptor_va, global_base_va,
            resource.descriptor_fingerprint, repeat,
            current.mode == DescriptorMode::kTmauPrefetchedL2Hot
                ? (repeat == 0 ? "prefetched" : "hot")
                : (repeat == 0 ? "cold" : "hot"),
            current.spec.path == matrix::Path::kTensor
                ? (current.mode == DescriptorMode::kTmauPrefetchedL2Hot
                       ? (repeat == 0 ? "tmau_prefetched_first_use"
                                      : "tmau_hot_reuse")
                       : (repeat == 0 ? "tmau_cold_first_use_l2_hot"
                                      : "tmau_hot_reuse"))
                : "not_applicable",
            current.spec.path == matrix::Path::kTensor
                ? "ld.global.cg" : "not_applicable",
            current.mode == DescriptorMode::kTmauPrefetchedL2Hot && repeat == 0,
            kMapPageBytes,
            shard_seed, expected, sample.payload_checksum);
      }
    }
  }

  std::free(host_map_storage);
  CUDA_CHECK(cudaFree(eviction));
  CUDA_CHECK(cudaFree(device_sink));
  CUDA_CHECK(cudaFree(device_samples));
  CUDA_CHECK(cudaFree(device_map_allocation));
  CUDA_CHECK(cudaFree(device_source));
  return 0;
}
