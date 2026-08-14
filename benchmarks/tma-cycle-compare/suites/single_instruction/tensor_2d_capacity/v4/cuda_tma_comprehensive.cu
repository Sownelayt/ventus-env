// Directed comprehensive CUDA half of the Ventus/CUDA TMA comparison.
//
// This file reuses tma-refer's protocol-correct rank-dispatched Tensor kernels
// while supplying an independently shared matrix, legal base-address phases,
// byte-exact C-model validation, Bulk and all common Reduce operations.  The
// executable is intentionally one-process/one-shard so Modal can put a short
// timeout around each fault domain.

#define main tma_reference_main_disabled
#include "tma_ventus_parity_bench_v2.cu"
#undef main

#include "tma_comprehensive_matrix.h"
#include "tma_model.cc"

#include <cinttypes>
#include <map>
#include <set>

namespace {

namespace matrix = tma_comprehensive;
using NeutralCase = matrix::CaseSpec;
using ventus::tma::DecodeAndPlan;
using ventus::tma::Descriptor;
using ventus::tma::Request;
using ventus::tma::Status;

constexpr uint32_t kComprehensiveSeed = 101u;
constexpr uint8_t kSentinel = 0xa5u;
constexpr size_t kTensorMapPageBytes = 4096;

struct ComprehensiveOptions {
  int warmups = 1;
  int repeats = 1;
  std::string shard = "all";
  std::string single_case;
  bool list = false;
};

size_t align_up_fresh(size_t value, size_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

uint64_t fingerprint64(const void* data, size_t bytes) {
  const auto* input = static_cast<const uint8_t*>(data);
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= input[index];
    value *= 1099511628211ull;
  }
  return value;
}

class FreshDescriptorPool {
 public:
  explicit FreshDescriptorPool(size_t entries) : entries_(entries) {
    CUDA_CHECK(cudaMalloc(&allocation_,
                          entries_ * kTensorMapPageBytes +
                              kTensorMapPageBytes - 1));
    const uintptr_t aligned =
        (reinterpret_cast<uintptr_t>(allocation_) + kTensorMapPageBytes - 1) &
        ~(static_cast<uintptr_t>(kTensorMapPageBytes) - 1);
    storage_ = reinterpret_cast<uint8_t*>(aligned);
  }

  ~FreshDescriptorPool() {
    if (allocation_) cudaFree(allocation_);
  }

  const CUtensorMap* put(const CUtensorMap& map) {
    if (next_ >= entries_) {
      std::fprintf(stderr, "fresh TensorMap pool exhausted\n");
      std::exit(2);
    }
    auto* destination = storage_ + next_ * kTensorMapPageBytes;
    ++next_;
    CUDA_CHECK(cudaMemcpy(destination, &map, sizeof(map),
                          cudaMemcpyHostToDevice));
    return reinterpret_cast<const CUtensorMap*>(destination);
  }

  size_t used() const { return next_; }

 private:
  uint8_t* storage_ = nullptr;
  uint8_t* allocation_ = nullptr;
  size_t entries_ = 0;
  size_t next_ = 0;
};

class FreshDataPool {
 public:
  explicit FreshDataPool(size_t bytes) : bytes_(bytes) {
    CUDA_CHECK(cudaMalloc(&storage_, bytes_));
  }

  ~FreshDataPool() {
    if (storage_) cudaFree(storage_);
  }

  uint8_t* allocate(size_t bytes, size_t phase) {
    next_ = align_up_fresh(next_, 128);
    const size_t end = next_ + phase + bytes;
    if (end > bytes_) {
      std::fprintf(stderr, "fresh global-data pool exhausted\n");
      std::exit(2);
    }
    uint8_t* result = storage_ + next_ + phase;
    next_ = align_up_fresh(end, 128);
    return result;
  }

 private:
  uint8_t* storage_ = nullptr;
  size_t bytes_ = 0;
  size_t next_ = 0;
};

FreshDescriptorPool* fresh_descriptor_pool = nullptr;
FreshDataPool* fresh_data_pool = nullptr;

struct FreshEvidence {
  uint64_t source_descriptor_va = 0;
  uint64_t destination_descriptor_va = 0;
  uint64_t source_descriptor_fingerprint = 0;
  uint64_t destination_descriptor_fingerprint = 0;
  const char* descriptor_state = "not_applicable";
  const char* pair_role = "not_applicable";
  int prior_tma_use = 0;
};

const char* interleave_name(matrix::Interleave value) {
  switch (value) {
    case matrix::Interleave::kNone: return "none";
    case matrix::Interleave::k16B: return "16";
    case matrix::Interleave::k32B: return "32";
  }
  return "unknown";
}

const char* swizzle_name(matrix::Swizzle value) {
  switch (value) {
    case matrix::Swizzle::kNone: return "none";
    case matrix::Swizzle::k32B: return "32";
    case matrix::Swizzle::k64B: return "64";
    case matrix::Swizzle::k128B: return "128";
  }
  return "unknown";
}

const char* risk_name(matrix::Risk value) {
  switch (value) {
    case matrix::Risk::kLow: return "low";
    case matrix::Risk::kMedium: return "medium";
    case matrix::Risk::kHigh: return "high";
  }
  return "unknown";
}

CUtensorMapDataType cuda_dtype(matrix::DType value) {
  // Do not cast the neutral/Ventus dtype number: CUDA differs at FP64,
  // BF16 and FP32_FTZ.
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

uint8_t ventus_dtype(matrix::DType value) {
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

int ventus_reduce(matrix::ReduceOp value) {
  switch (value) {
    case matrix::ReduceOp::kAdd: return VENTUS_TMA_V2_REDUCE_ADD;
    case matrix::ReduceOp::kMin: return VENTUS_TMA_V2_REDUCE_MIN;
    case matrix::ReduceOp::kMax: return VENTUS_TMA_V2_REDUCE_MAX;
    case matrix::ReduceOp::kAnd: return VENTUS_TMA_V2_REDUCE_AND;
    case matrix::ReduceOp::kOr: return VENTUS_TMA_V2_REDUCE_OR;
    case matrix::ReduceOp::kXor: return VENTUS_TMA_V2_REDUCE_XOR;
    case matrix::ReduceOp::kNone: break;
  }
  return VENTUS_TMA_V2_REDUCE_COPY;
}

CaseSpec to_cuda_spec(const NeutralCase& input) {
  CaseSpec spec;
  spec.id = input.id;
  spec.layout = input.family;
  spec.rank = input.rank;
  spec.bytes = static_cast<int>(input.transfer_bytes);
  spec.element_bytes = std::max(1, static_cast<int>(input.element_bits) / 8);
  spec.data_type = cuda_dtype(input.dtype);
  spec.dims = input.global_dims;
  spec.strides = input.global_strides_bytes;
  spec.box = input.box_dims;
  spec.element_strides = input.element_strides;
  spec.coords = input.coordinates;
  spec.interleave = cuda_interleave(input.interleave);
  spec.swizzle = cuda_swizzle(input.swizzle);
  spec.shared_is_linear = input.shared_is_linear;
  return spec;
}

Descriptor model_descriptor(const NeutralCase& spec) {
  Descriptor descriptor{};
  descriptor.words[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
  descriptor.words[VENTUS_TMA_V2_WORD_CONTROL] =
      ventus_dtype(spec.dtype) | (static_cast<uint32_t>(spec.rank) << 5) |
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

ventus::tma::Result model_plan(const NeutralCase& spec, int direction) {
  Request request{};
  request.direction = direction == 0 ? VENTUS_TMA_G2S : VENTUS_TMA_S2G;
  request.shared_base = 0;
  request.coordinates = spec.coordinates;
  return DecodeAndPlan(model_descriptor(spec), request);
}

size_t global_span(const NeutralCase& spec) {
  if (spec.path == matrix::Path::kBulk) return spec.transfer_bytes;
  if (spec.rank == 1) {
    return static_cast<size_t>(
        (spec.global_dims[0] * spec.element_bits + 7) / 8);
  }
  return static_cast<size_t>(spec.global_strides_bytes[spec.rank - 2]) *
         static_cast<size_t>(spec.global_dims[spec.rank - 1]);
}

std::vector<uint8_t> expected_g2s(
    const ventus::tma::Result& plan, const std::vector<uint8_t>& global,
    size_t shared_bytes) {
  std::vector<uint8_t> shared(shared_bytes, 0);
  for (const auto& atom : plan.atoms) {
    for (unsigned shared_lane = 0; shared_lane < 16; ++shared_lane) {
      if (((atom.shared_mask >> shared_lane) & 1u) == 0) continue;
      if (atom.fill) {
        shared[atom.shared_atom + shared_lane] = atom.fill_bytes[shared_lane];
        continue;
      }
      for (unsigned global_lane = 0; global_lane < 16; ++global_lane) {
        if (((atom.global_mask >> global_lane) & 1u) != 0 &&
            atom.global_to_shared[global_lane] == shared_lane) {
          shared[atom.shared_atom + shared_lane] =
              global[atom.global_atom + global_lane];
        }
      }
    }
  }
  return shared;
}

uint32_t tf32_rna(uint32_t bits) {
  const uint32_t exponent = bits & 0x7f800000u;
  const uint32_t fraction = bits & 0x007fffffu;
  // CUDA's __float_to_tf32 lowers to cvt.rna.tf32.f32.  Preserve infinities
  // and canonicalize NaNs while retaining the TF32 requirement that the low
  // 13 representation bits are zero.  TMA movement gives TF32_FTZ the same
  // bit result as TF32; it does not pre-flush an f32 subnormal here.
  if (exponent == 0x7f800000u) {
    if (fraction == 0) return bits;
    return 0x7fffe000u;
  }
  return (bits + 0x00001000u) & 0xffffe000u;
}

uint32_t swizzle_mask(matrix::Swizzle mode) {
  switch (mode) {
    case matrix::Swizzle::k32B: return 1;
    case matrix::Swizzle::k64B: return 3;
    case matrix::Swizzle::k128B: return 7;
    case matrix::Swizzle::kNone: return 0;
  }
  return 0;
}

uint32_t model_swizzle_offset(uint32_t logical, matrix::Swizzle mode) {
  if (mode == matrix::Swizzle::kNone) return logical;
  const uint32_t mask = swizzle_mask(mode);
  const uint32_t span = (mask + 1) * 16;
  const uint32_t row = logical / span;
  return (logical & ~(span - 1)) |
         (((((logical >> 4) & mask) ^ (row & mask))) << 4) |
         (logical & 15);
}

uint32_t cuda_swizzle_offset(uint32_t logical, matrix::Swizzle mode) {
  if (mode == matrix::Swizzle::kNone) return logical;
  const uint32_t mask = swizzle_mask(mode);
  // CUDA defines y as the 128B shared-memory row for every swizzle width;
  // 32B/64B merely limit the XOR to the low 1/2 chunk-index bits.
  const uint32_t y = logical >> 7;
  const uint32_t x = (logical >> 4) & 7;
  return (logical & ~127u) | ((x ^ (y & mask)) << 4) | (logical & 15);
}

std::vector<uint8_t> model_shared_to_cuda_shared(
    const NeutralCase& spec, const std::vector<uint8_t>& model_shared) {
  if (spec.swizzle == matrix::Swizzle::kNone) return model_shared;
  std::vector<uint8_t> cuda_shared(model_shared.size(), 0);
  for (uint32_t logical = 0; logical < model_shared.size(); ++logical) {
    const uint32_t old_offset = model_swizzle_offset(logical, spec.swizzle);
    const uint32_t new_offset = cuda_swizzle_offset(logical, spec.swizzle);
    cuda_shared[new_offset] = model_shared[old_offset];
  }
  return cuda_shared;
}

std::vector<uint8_t> cuda_shared_to_model_shared(
    const NeutralCase& spec, const std::vector<uint8_t>& cuda_shared) {
  if (spec.swizzle == matrix::Swizzle::kNone) return cuda_shared;
  std::vector<uint8_t> model_shared(cuda_shared.size(), 0);
  for (uint32_t logical = 0; logical < cuda_shared.size(); ++logical) {
    const uint32_t old_offset = model_swizzle_offset(logical, spec.swizzle);
    const uint32_t new_offset = cuda_swizzle_offset(logical, spec.swizzle);
    model_shared[old_offset] = cuda_shared[new_offset];
  }
  return model_shared;
}

std::vector<uint8_t> cuda_expected_g2s(
    const NeutralCase& spec, const ventus::tma::Result& plan,
    const std::vector<uint8_t>& global, size_t shared_bytes) {
  std::vector<uint8_t> shared = model_shared_to_cuda_shared(
      spec, expected_g2s(plan, global, shared_bytes));
  const bool tf32 = spec.dtype == matrix::DType::kTf32 ||
                    spec.dtype == matrix::DType::kTf32Ftz;
  if (tf32) {
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
    // CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA is the documented
    // special NaN constant, not the conventional quiet-NaN payload used by
    // the Ventus functional C-model.  Hopper materializes the 16-bit pattern
    // 0x7ff7 repeatedly, independent of the floating element width.
    for (const auto& atom : plan.atoms) {
      if (!atom.fill) continue;
      for (unsigned lane = 0; lane < 16; ++lane)
        if (((atom.shared_mask >> lane) & 1u) != 0)
          shared[atom.shared_atom + lane] =
              ((atom.shared_atom + lane) & 1u) == 0 ? 0xf7 : 0x7f;
    }
  }
  return shared;
}

void log_mismatches(const NeutralCase& spec,
                    const std::vector<uint8_t>& actual,
                    const std::vector<uint8_t>& expected) {
  unsigned logged = 0;
  const size_t size = std::min(actual.size(), expected.size());
  for (size_t offset = 0; offset < size && logged < 8; ++offset) {
    if (actual[offset] == expected[offset]) continue;
    std::fprintf(stderr, "MISMATCH,%s,%zu,%02x,%02x\n", spec.id.c_str(),
                 offset, static_cast<unsigned>(actual[offset]),
                 static_cast<unsigned>(expected[offset]));
    ++logged;
  }
}

std::vector<uint8_t> expected_s2g(
    const ventus::tma::Result& plan, const std::vector<uint8_t>& shared,
    size_t bytes) {
  std::vector<uint8_t> global(bytes, kSentinel);
  for (const auto& atom : plan.atoms) {
    if (atom.fill) continue;
    for (unsigned global_lane = 0; global_lane < 16; ++global_lane) {
      if (((atom.global_mask >> global_lane) & 1u) == 0) continue;
      const unsigned shared_lane = atom.global_to_shared[global_lane];
      if (shared_lane < 16)
        global[atom.global_atom + global_lane] =
            shared[atom.shared_atom + shared_lane];
    }
  }
  return global;
}

std::vector<uint8_t> cuda_expected_s2g(
    const NeutralCase& spec, const ventus::tma::Result& plan,
    const std::vector<uint8_t>& shared, size_t bytes) {
  return expected_s2g(plan, cuda_shared_to_model_shared(spec, shared), bytes);
}

std::vector<uint8_t> patterned(size_t bytes, uint32_t seed) {
  return patterned_bytes(bytes, seed);
}

const char* alignment_class(const NeutralCase& spec) {
  return spec.global_base_mod128 == 0 ? "aligned128" : "phase_mod128";
}

std::string stride_class(const NeutralCase& spec) {
  if (spec.path == matrix::Path::kBulk || spec.rank < 2) return "contiguous";
  uint64_t expected =
      (spec.global_dims[0] * static_cast<uint64_t>(spec.element_bits) + 7) / 8;
  for (int d = 1; d < spec.rank; ++d) {
    if (spec.global_strides_bytes[d - 1] != expected) return "pitched";
    expected *= spec.global_dims[d];
  }
  return "contiguous";
}

void emit_header() {
  std::puts(
      "gpu,cc,l2_bytes,shard,family,case_id,path,direction,dtype,dtype_bits,rank,"
      "logical_bytes,global_span,global_offset,alignment_class,stride_class,"
      "interleave,swizzle,oob_fill,reduce_op,descriptor_mode,risk,repeat,"
      "cycles,event_ns,errors,status,descriptor_state,payload_state,"
      "source_descriptor_va,destination_descriptor_va,"
      "source_descriptor_fingerprint64,destination_descriptor_fingerprint64,"
      "prior_tma_use,descriptor_l2_warm_method,tma_prefetch_before_issue,pair_role");
}

void emit_row(const Context& context, const NeutralCase& spec, int repeat,
              unsigned long long cycles, double event_ns, uint64_t errors,
              const char* status,
              const FreshEvidence& evidence = FreshEvidence{}) {
  std::printf(
      "%s,%s,%zu,%s,%s,%s,%s,%s,%s,%u,%d,%u,%zu,%u,%s,%s,%s,%s,%s,%s,"
      "direct,%s,%d,%llu,%.3f,%llu,%s,%s,l2_hot,0x%016" PRIx64
      ",0x%016" PRIx64 ",0x%016" PRIx64 ",0x%016" PRIx64
      ",%d,%s,0,%s\n",
      context.gpu.c_str(), context.cc.c_str(),
      static_cast<size_t>(context.prop.l2CacheSize), spec.shard.c_str(),
      spec.family.c_str(), spec.id.c_str(), matrix::path_name(spec.path),
      matrix::direction_name(spec.direction), matrix::dtype_name(spec.dtype),
      static_cast<unsigned>(spec.element_bits), spec.rank,
      spec.transfer_bytes, global_span(spec),
      static_cast<unsigned>(spec.global_base_mod128), alignment_class(spec),
      stride_class(spec).c_str(), interleave_name(spec.interleave),
      swizzle_name(spec.swizzle), matrix::oob_name(spec.oob_fill),
      matrix::reduce_name(spec.reduce_op),
      risk_name(spec.expected.risk), repeat, cycles, event_ns,
      static_cast<unsigned long long>(errors), status,
      evidence.descriptor_state, evidence.source_descriptor_va,
      evidence.destination_descriptor_va,
      evidence.source_descriptor_fingerprint,
      evidence.destination_descriptor_fingerprint,
      evidence.prior_tma_use,
      spec.path == matrix::Path::kTensor ? "ld.global.cg" : "not_applicable",
      evidence.pair_role);
  std::fflush(stdout);
}

CUresult encode_neutral_map(CUtensorMap* map, const NeutralCase& neutral,
                            void* base) {
  const CaseSpec spec = to_cuda_spec(neutral);
  return cuTensorMapEncodeTiled(
      map, spec.data_type, static_cast<cuuint32_t>(spec.rank), base,
      spec.dims.data(), spec.strides.data(), spec.box.data(),
      spec.element_strides.data(), spec.interleave, spec.swizzle,
      CU_TENSOR_MAP_L2_PROMOTION_NONE,
      neutral.oob_fill == matrix::OobFill::kNan
          ? CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
          : CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
}

bool encode_rejection_is_allowed(const NeutralCase& spec,
                                 const Context& context) {
  const matrix::Support support = context.prop.major >= 10
                                      ? spec.expected.b200
                                      : spec.expected.h100;
  return support != matrix::Support::kExpectedPass;
}

__device__ __forceinline__ uint32_t fresh_load_cg(const void* pointer) {
  uint32_t value;
  asm volatile("ld.global.cg.u32 %0, [%1];"
               : "=r"(value) : "l"(pointer) : "memory");
  return value;
}

__device__ __forceinline__ void fresh_acquire_map(const CUtensorMap* map) {
  asm volatile("fence.proxy.tensormap::generic.acquire.sys [%0], 128;"
               :: "l"(map) : "memory");
}

__global__ void warm_fresh_inputs(
    const uint8_t* payload, size_t payload_bytes,
    const CUtensorMap* source_map, const CUtensorMap* destination_map,
    unsigned long long* sink) {
  if (threadIdx.x != 0) return;
  unsigned long long value = 0;
  if (payload)
    for (size_t offset = 0; offset < payload_bytes; offset += 32)
      value = value * 0x9e3779b97f4a7c15ull +
              fresh_load_cg(payload + offset);
  if (source_map) {
    const auto* words = reinterpret_cast<const uint32_t*>(source_map);
    for (int word = 0; word < 32; word += 8)
      value = value * 0x9e3779b97f4a7c15ull + fresh_load_cg(words + word);
  }
  if (destination_map) {
    const auto* words = reinterpret_cast<const uint32_t*>(destination_map);
    for (int word = 0; word < 32; word += 8)
      value = value * 0x9e3779b97f4a7c15ull + fresh_load_cg(words + word);
  }
  *sink = value;
}

__global__ void acquire_fresh_maps_once(
    const CUtensorMap* source_map, const CUtensorMap* destination_map) {
  if (threadIdx.x != 0) return;
  if (source_map) fresh_acquire_map(source_map);
  if (destination_map) fresh_acquire_map(destination_map);
}

template <int Rank>
__global__ void fresh_tensor_g2s(
    const CUtensorMap* map, int bytes, DeviceCoords coords,
    int iterations, unsigned long long* cycles, uint8_t* capture) {
  extern __shared__ __align__(128) uint8_t shared[];
  BlockBarrier* barrier = barrier_after(shared, static_cast<size_t>(bytes));
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  for (int iteration = 0; iteration < iterations; ++iteration) {
    if (threadIdx.x == 0) {
      cuda::device::barrier_expect_tx(*barrier, bytes);
      const unsigned long long begin = clock64();
      issue_tma_load<Rank>(map, shared, coords, *barrier);
      barrier->arrive_and_wait();
      cycles[iteration] = clock64() - begin;
    }
    __syncthreads();
    for (int byte = threadIdx.x; byte < bytes; byte += blockDim.x)
      capture[static_cast<size_t>(iteration) * bytes + byte] = shared[byte];
    __syncthreads();
  }
}

template <int Rank>
__global__ void fresh_tensor_s2g(
    const CUtensorMap* map, int bytes, DeviceCoords coords,
    int iterations, unsigned long long* cycles) {
  extern __shared__ __align__(128) uint8_t shared[];
  for (int byte = threadIdx.x; byte < bytes; byte += blockDim.x)
    shared[byte] = pattern_byte(kComprehensiveSeed, byte);
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const unsigned long long begin = clock64();
      issue_tma_store<Rank>(map, shared, coords);
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      cycles[iteration] = clock64() - begin;
    }
  }
}

template <int Rank>
__global__ void fresh_tensor_roundtrip(
    const CUtensorMap* source_map, const CUtensorMap* destination_map,
    int bytes, DeviceCoords coords, int iterations,
    unsigned long long* cycles) {
  extern __shared__ __align__(128) uint8_t shared[];
  BlockBarrier* barrier = barrier_after(shared, static_cast<size_t>(bytes));
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      cuda::device::barrier_expect_tx(*barrier, bytes);
      const unsigned long long begin = clock64();
      issue_tma_load<Rank>(source_map, shared, coords, *barrier);
      barrier->arrive_and_wait();
      cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
      issue_tma_store<Rank>(destination_map, shared, coords);
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      cycles[iteration] = clock64() - begin;
    }
  }
}

void launch_fresh_tensor(
    int direction, int rank, const CUtensorMap* source_map,
    const CUtensorMap* destination_map, int bytes, size_t shared,
    DeviceCoords coords, int iterations, unsigned long long* cycles,
    uint8_t* capture) {
#define LAUNCH_FRESH(N)                                                        \
  case N:                                                                     \
    if (direction == 0)                                                       \
      fresh_tensor_g2s<N><<<1, 128, shared>>>(                                \
          source_map, bytes, coords, iterations, cycles, capture);            \
    else if (direction == 1)                                                  \
      fresh_tensor_s2g<N><<<1, 128, shared>>>(                                \
          destination_map, bytes, coords, iterations, cycles);                \
    else                                                                       \
      fresh_tensor_roundtrip<N><<<1, 128, shared>>>(                          \
          source_map, destination_map, bytes, coords, iterations, cycles);    \
    break
  switch (rank) {
    LAUNCH_FRESH(1); LAUNCH_FRESH(2); LAUNCH_FRESH(3);
    LAUNCH_FRESH(4); LAUNCH_FRESH(5);
    default: std::abort();
  }
#undef LAUNCH_FRESH
}

__global__ void comprehensive_bulk_g2s(
    const uint8_t* source, uint8_t* capture, int bytes, int iterations,
    unsigned long long* cycles) {
  extern __shared__ __align__(128) uint8_t shared[];
  BlockBarrier* barrier = barrier_after(shared, static_cast<size_t>(bytes));
  if (threadIdx.x == 0) init(barrier, 1);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const auto begin = clock64();
      cuda::device::barrier_expect_tx(*barrier, bytes);
      cuda::ptx::cp_async_bulk(
          cuda::ptx::space_shared, cuda::ptx::space_global, shared, source,
          static_cast<uint32_t>(bytes),
          cuda::device::barrier_native_handle(*barrier));
      barrier->arrive_and_wait();
      if (cycles) cycles[iteration] = clock64() - begin;
    }
  }
  __syncthreads();
  if (capture)
    for (int i = threadIdx.x; i < bytes; i += blockDim.x) capture[i] = shared[i];
}

__global__ void comprehensive_bulk_s2g(
    const uint8_t* seed, uint8_t* destination, int bytes, int iterations,
    unsigned long long* cycles) {
  extern __shared__ __align__(128) uint8_t shared[];
  for (int i = threadIdx.x; i < bytes; i += blockDim.x) shared[i] = seed[i];
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const auto begin = clock64();
      cuda::ptx::cp_async_bulk(cuda::ptx::space_global,
                               cuda::ptx::space_shared, destination, shared,
                               static_cast<uint32_t>(bytes));
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      if (cycles) cycles[iteration] = clock64() - begin;
    }
  }
}

bool run_bulk(const Context& context, const NeutralCase& spec) {
  const size_t bytes = spec.transfer_bytes;
  const size_t phase = spec.global_base_mod128;
  DeviceArray<uint8_t> source_storage(bytes + 128);
  DeviceArray<uint8_t> destination_storage(bytes + 128);
  DeviceArray<uint8_t> capture(bytes);
  DeviceArray<unsigned long long> device_cycles(context.options.repeats);
  uint8_t* source = source_storage.get() + phase;
  uint8_t* destination = destination_storage.get() + phase;
  const auto host = patterned(bytes, kComprehensiveSeed);
  const std::vector<uint8_t> sentinel(bytes, kSentinel);
  CUDA_CHECK(cudaMemcpy(source, host.data(), bytes, cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(destination, sentinel.data(), bytes,
                        cudaMemcpyHostToDevice));
  const size_t shared = spec.direction == matrix::Direction::kG2s
                            ? barrier_shared_bytes(bytes)
                            : bytes;
  const auto launch = [&](int iterations, unsigned long long* cycles) {
    if (spec.direction == matrix::Direction::kG2s)
      comprehensive_bulk_g2s<<<1, 128, shared>>>(
          source, nullptr, static_cast<int>(bytes), iterations, cycles);
    else
      comprehensive_bulk_s2g<<<1, 128, shared>>>(
          source, destination, static_cast<int>(bytes), iterations, cycles);
  };
  if (context.options.warmups) {
    launch(context.options.warmups, nullptr);
    complete_warmup();
    if (spec.direction == matrix::Direction::kS2g)
      CUDA_CHECK(cudaMemcpy(destination, sentinel.data(), bytes,
                            cudaMemcpyHostToDevice));
  }
  const float elapsed = timed_launch(
      [&] { launch(context.options.repeats, device_cycles.get()); });
  std::vector<unsigned long long> cycles(context.options.repeats);
  CUDA_CHECK(cudaMemcpy(cycles.data(), device_cycles.get(),
                        cycles.size() * sizeof(cycles[0]),
                        cudaMemcpyDeviceToHost));
  uint64_t errors = 0;
  if (spec.direction == matrix::Direction::kG2s) {
    comprehensive_bulk_g2s<<<1, 128, shared>>>(
        source, capture.get(), static_cast<int>(bytes), 1, nullptr);
    complete_warmup();
    std::vector<uint8_t> actual(bytes);
    CUDA_CHECK(cudaMemcpy(actual.data(), capture.get(), bytes,
                          cudaMemcpyDeviceToHost));
    errors = mismatch_count(actual, host);
  } else {
    std::vector<uint8_t> actual(bytes);
    CUDA_CHECK(cudaMemcpy(actual.data(), destination, bytes,
                          cudaMemcpyDeviceToHost));
    errors = mismatch_count(actual, host);
  }
  for (int repeat = 0; repeat < context.options.repeats; ++repeat)
    emit_row(context, spec, repeat, cycles[repeat],
             elapsed * 1.0e6 / context.options.repeats, errors,
             errors ? "validation_error" : "ok");
  return errors == 0;
}

bool run_tensor(const Context& context, const NeutralCase& neutral) {
  const int direction = neutral.direction == matrix::Direction::kG2s
                            ? 0
                            : neutral.direction == matrix::Direction::kS2g ? 1 : 2;
  const auto g2s_plan = model_plan(neutral, 0);
  const auto s2g_plan = model_plan(neutral, 1);
  if ((direction != 1 && g2s_plan.status != Status::kOk) ||
      (direction != 0 && s2g_plan.status != Status::kOk)) {
    std::fprintf(stderr, "MODEL_ERROR,%s,%s,%s\n", neutral.id.c_str(),
                 g2s_plan.detail.c_str(), s2g_plan.detail.c_str());
    emit_row(context, neutral, -1, 0, 0, 1, "model_rejected");
    return false;
  }

  const CaseSpec spec = to_cuda_spec(neutral);
  const size_t allocation = global_span(neutral);
  const size_t phase = neutral.global_base_mod128;
  const int repeats = context.options.repeats;
  DeviceArray<uint8_t> capture(
      direction == 0 ? neutral.transfer_bytes * repeats : 0);
  DeviceArray<unsigned long long> device_cycles(repeats);
  DeviceArray<unsigned long long> warm_sink(1);
  std::vector<uint8_t*> sources(repeats, nullptr);
  std::vector<uint8_t*> destinations(repeats, nullptr);
  std::vector<const CUtensorMap*> source_maps(repeats, nullptr);
  std::vector<const CUtensorMap*> destination_maps(repeats, nullptr);
  std::vector<FreshEvidence> evidence(repeats);
  const auto host_source = patterned(allocation, kComprehensiveSeed);
  const std::vector<uint8_t> sentinel(allocation, kSentinel);

  if (direction != 1) {
      sources[0] = fresh_data_pool->allocate(allocation, phase);
      CUDA_CHECK(cudaMemcpy(sources[0], host_source.data(), allocation,
                            cudaMemcpyHostToDevice));
      CUtensorMap host_map{};
      const CUresult status =
          encode_neutral_map(&host_map, neutral, sources[0]);
      if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "ENCODE_REJECTED,%s,%s\n", neutral.id.c_str(),
                     cu_error_string(status).c_str());
        emit_row(context, neutral, -1, 0, 0, 0, "encode_rejected");
        return encode_rejection_is_allowed(neutral, context);
      }
      source_maps[0] = fresh_descriptor_pool->put(host_map);
      evidence[0].source_descriptor_va =
          reinterpret_cast<uint64_t>(source_maps[0]);
      evidence[0].source_descriptor_fingerprint =
          fingerprint64(&host_map, sizeof(host_map));
  }
  if (direction != 0) {
      destinations[0] = fresh_data_pool->allocate(allocation, phase);
      CUDA_CHECK(cudaMemcpy(destinations[0], sentinel.data(), allocation,
                            cudaMemcpyHostToDevice));
      CUtensorMap host_map{};
      const CUresult status =
          encode_neutral_map(&host_map, neutral, destinations[0]);
      if (status != CUDA_SUCCESS) {
        std::fprintf(stderr, "ENCODE_REJECTED,%s,%s\n", neutral.id.c_str(),
                     cu_error_string(status).c_str());
        emit_row(context, neutral, -1, 0, 0, 0, "encode_rejected");
        return encode_rejection_is_allowed(neutral, context);
      }
      destination_maps[0] = fresh_descriptor_pool->put(host_map);
      evidence[0].destination_descriptor_va =
          reinterpret_cast<uint64_t>(destination_maps[0]);
      evidence[0].destination_descriptor_fingerprint =
          fingerprint64(&host_map, sizeof(host_map));
  }
  for (int repeat = 0; repeat < repeats; ++repeat) {
    sources[repeat] = sources[0];
    destinations[repeat] = destinations[0];
    source_maps[repeat] = source_maps[0];
    destination_maps[repeat] = destination_maps[0];
    evidence[repeat] = evidence[0];
    evidence[repeat].descriptor_state =
        repeat == 0 ? "tmau_cold_first_use_l2_hot" : "tmau_hot_reuse";
    evidence[repeat].pair_role = repeat == 0 ? "cold" : "hot";
    evidence[repeat].prior_tma_use = repeat;
  }
  warm_fresh_inputs<<<1, 32>>>(
      sources[0], direction == 1 ? 0 : allocation,
      source_maps[0], destination_maps[0], warm_sink.get());
  complete_warmup();
  acquire_fresh_maps_once<<<1, 1>>>(source_maps[0], destination_maps[0]);
  complete_warmup();
  const DeviceCoords base = device_coords(spec.coords);
  const size_t shared = direction == 1
                            ? neutral.transfer_bytes
                            : barrier_shared_bytes(neutral.transfer_bytes);
  const float elapsed = timed_launch([&] {
    launch_fresh_tensor(
        direction, spec.rank, source_maps[0], destination_maps[0],
        spec.bytes, shared, base, repeats, device_cycles.get(),
        direction == 0 ? capture.get() : nullptr);
  });
  std::vector<unsigned long long> cycles(repeats);
  CUDA_CHECK(cudaMemcpy(cycles.data(), device_cycles.get(),
                        cycles.size() * sizeof(cycles[0]),
                        cudaMemcpyDeviceToHost));

  uint64_t errors = 0;
  if (direction == 0) {
    std::vector<uint8_t> actual(neutral.transfer_bytes * repeats);
    CUDA_CHECK(cudaMemcpy(actual.data(), capture.get(), actual.size(),
                          cudaMemcpyDeviceToHost));
    const auto expected = cuda_expected_g2s(
        neutral, g2s_plan, host_source, neutral.transfer_bytes);
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto begin = actual.begin() +
                         static_cast<ptrdiff_t>(repeat * neutral.transfer_bytes);
      const std::vector<uint8_t> sample(
          begin, begin + static_cast<ptrdiff_t>(neutral.transfer_bytes));
      const uint64_t sample_errors = mismatch_count(sample, expected);
      errors += sample_errors;
      if (sample_errors) log_mismatches(neutral, sample, expected);
    }
  } else {
    for (int repeat = 0; repeat < repeats; ++repeat) {
      std::vector<uint8_t> actual(allocation);
      CUDA_CHECK(cudaMemcpy(actual.data(), destinations[repeat], allocation,
                            cudaMemcpyDeviceToHost));
      std::vector<uint8_t> expected;
      if (direction == 1) {
        expected = cuda_expected_s2g(
            neutral, s2g_plan,
            patterned(neutral.transfer_bytes, kComprehensiveSeed), allocation);
      } else if (neutral.dtype == matrix::DType::kU6Align16) {
        expected.assign(allocation, kSentinel);
        for (size_t offset = 0; offset < allocation; ++offset)
          if ((offset & 15u) < 12u) expected[offset] = host_source[offset];
      } else {
        const auto shared_bytes = cuda_expected_g2s(
            neutral, g2s_plan, host_source, neutral.transfer_bytes);
        expected =
            cuda_expected_s2g(neutral, s2g_plan, shared_bytes, allocation);
      }
      const uint64_t sample_errors = mismatch_count(actual, expected);
      errors += sample_errors;
      if (sample_errors) log_mismatches(neutral, actual, expected);
    }
  }
  for (int repeat = 0; repeat < repeats; ++repeat)
    emit_row(context, neutral, repeat, cycles[repeat],
             elapsed * 1.0e6 / repeats, errors,
             errors ? "validation_error" : "ok", evidence[repeat]);
  return errors == 0;
}

__host__ __device__ uint32_t reduce_operand(uint32_t index) {
  return 0x01020304u ^ (index * 0x1021u);
}

template <int Rank, int Op>
__device__ __forceinline__ void issue_reduce(const CUtensorMap* map,
                                              const DeviceCoords& coords,
                                              const void* shared) {
  int32_t c[Rank];
#pragma unroll
  for (int i = 0; i < Rank; ++i) c[i] = coords.value[i];
  if constexpr (Op == VENTUS_TMA_V2_REDUCE_ADD)
    cuda::ptx::cp_reduce_async_bulk_tensor(cuda::ptx::space_global,
        cuda::ptx::space_shared, cuda::ptx::op_add, map, c, shared);
  else if constexpr (Op == VENTUS_TMA_V2_REDUCE_MIN)
    cuda::ptx::cp_reduce_async_bulk_tensor(cuda::ptx::space_global,
        cuda::ptx::space_shared, cuda::ptx::op_min, map, c, shared);
  else if constexpr (Op == VENTUS_TMA_V2_REDUCE_MAX)
    cuda::ptx::cp_reduce_async_bulk_tensor(cuda::ptx::space_global,
        cuda::ptx::space_shared, cuda::ptx::op_max, map, c, shared);
  else if constexpr (Op == VENTUS_TMA_V2_REDUCE_AND)
    cuda::ptx::cp_reduce_async_bulk_tensor(cuda::ptx::space_global,
        cuda::ptx::space_shared, cuda::ptx::op_and_op, map, c, shared);
  else if constexpr (Op == VENTUS_TMA_V2_REDUCE_OR)
    cuda::ptx::cp_reduce_async_bulk_tensor(cuda::ptx::space_global,
        cuda::ptx::space_shared, cuda::ptx::op_or_op, map, c, shared);
  else if constexpr (Op == VENTUS_TMA_V2_REDUCE_XOR)
    cuda::ptx::cp_reduce_async_bulk_tensor(cuda::ptx::space_global,
        cuda::ptx::space_shared, cuda::ptx::op_xor_op, map, c, shared);
}

template <int Rank, int Op>
__global__ void comprehensive_reduce(
                                     const __grid_constant__ CUtensorMap map,
                                     int bytes,
                                     int iterations,
                                     DeviceCoords coords,
                                     unsigned long long* cycles) {
  extern __shared__ __align__(128) uint32_t reduce_shared[];
  for (int i = threadIdx.x; i < bytes / 4; i += blockDim.x)
    reduce_shared[i] = reduce_operand(i);
  __syncthreads();
  cuda::ptx::fence_proxy_async(cuda::ptx::space_shared);
  __syncthreads();
  if (threadIdx.x == 0) {
    for (int iteration = 0; iteration < iterations; ++iteration) {
      const auto begin = clock64();
      issue_reduce<Rank, Op>(&map, coords, reduce_shared);
      cuda::ptx::cp_async_bulk_commit_group();
      cuda::ptx::cp_async_bulk_wait_group(cuda::ptx::n32_t<0>{});
      if (cycles) cycles[iteration] = clock64() - begin;
    }
  }
}

template <int Op>
void launch_reduce_rank(int rank, const CUtensorMap& map, int bytes,
                        int iterations, DeviceCoords coords,
                        unsigned long long* cycles) {
#define LAUNCH_RANK(N) \
  case N: comprehensive_reduce<N, Op><<<1, 128, bytes>>>(map, bytes, iterations, coords, cycles); break
  switch (rank) {
    LAUNCH_RANK(1); LAUNCH_RANK(2); LAUNCH_RANK(3); LAUNCH_RANK(4);
    LAUNCH_RANK(5);
    default: std::abort();
  }
#undef LAUNCH_RANK
}

void launch_reduce(int op, int rank, const CUtensorMap& map, int bytes,
                   int iterations, DeviceCoords coords,
                   unsigned long long* cycles) {
  switch (op) {
    case VENTUS_TMA_V2_REDUCE_ADD:
      launch_reduce_rank<VENTUS_TMA_V2_REDUCE_ADD>(rank, map, bytes, iterations, coords, cycles); break;
    case VENTUS_TMA_V2_REDUCE_MIN:
      launch_reduce_rank<VENTUS_TMA_V2_REDUCE_MIN>(rank, map, bytes, iterations, coords, cycles); break;
    case VENTUS_TMA_V2_REDUCE_MAX:
      launch_reduce_rank<VENTUS_TMA_V2_REDUCE_MAX>(rank, map, bytes, iterations, coords, cycles); break;
    case VENTUS_TMA_V2_REDUCE_AND:
      launch_reduce_rank<VENTUS_TMA_V2_REDUCE_AND>(rank, map, bytes, iterations, coords, cycles); break;
    case VENTUS_TMA_V2_REDUCE_OR:
      launch_reduce_rank<VENTUS_TMA_V2_REDUCE_OR>(rank, map, bytes, iterations, coords, cycles); break;
    case VENTUS_TMA_V2_REDUCE_XOR:
      launch_reduce_rank<VENTUS_TMA_V2_REDUCE_XOR>(rank, map, bytes, iterations, coords, cycles); break;
    default: std::abort();
  }
}

uint32_t apply_reduce(int op, bool signed_dtype, uint32_t old_value,
                      uint32_t operand) {
  switch (op) {
    case VENTUS_TMA_V2_REDUCE_ADD: return old_value + operand;
    case VENTUS_TMA_V2_REDUCE_MIN:
      return signed_dtype
                 ? (static_cast<int32_t>(old_value) < static_cast<int32_t>(operand)
                        ? old_value : operand)
                 : std::min(old_value, operand);
    case VENTUS_TMA_V2_REDUCE_MAX:
      return signed_dtype
                 ? (static_cast<int32_t>(old_value) > static_cast<int32_t>(operand)
                        ? old_value : operand)
                 : std::max(old_value, operand);
    case VENTUS_TMA_V2_REDUCE_AND: return old_value & operand;
    case VENTUS_TMA_V2_REDUCE_OR: return old_value | operand;
    case VENTUS_TMA_V2_REDUCE_XOR: return old_value ^ operand;
  }
  return old_value;
}

bool run_reduce(const Context& context, const NeutralCase& neutral) {
  const CaseSpec spec = to_cuda_spec(neutral);
  const size_t allocation = global_span(neutral);
  const size_t phase = neutral.global_base_mod128;
  DeviceArray<uint8_t> destination_storage(allocation + 128);
  DeviceArray<unsigned long long> device_cycles(context.options.repeats);
  auto* destination = destination_storage.get() + phase;
  std::vector<uint32_t> initial(allocation / 4);
  for (size_t i = 0; i < initial.size(); ++i)
    initial[i] = 0x91e10da5u ^ static_cast<uint32_t>(i * 0x9e37u);
  CUDA_CHECK(cudaMemcpy(destination, initial.data(), allocation,
                        cudaMemcpyHostToDevice));
  CUtensorMap map{};
  const CUresult status = encode_neutral_map(&map, neutral, destination);
  if (status != CUDA_SUCCESS) {
    emit_row(context, neutral, -1, 0, 0, 0, "encode_rejected");
    return encode_rejection_is_allowed(neutral, context);
  }
  const int op = ventus_reduce(neutral.reduce_op);
  const DeviceCoords coords = device_coords(spec.coords);
  if (context.options.warmups) {
    launch_reduce(op, spec.rank, map, spec.bytes, context.options.warmups,
                  coords, nullptr);
    complete_warmup();
    CUDA_CHECK(cudaMemcpy(destination, initial.data(), allocation,
                          cudaMemcpyHostToDevice));
  }
  const float elapsed = timed_launch([&] {
    launch_reduce(op, spec.rank, map, spec.bytes, context.options.repeats,
                  coords, device_cycles.get());
  });
  std::vector<unsigned long long> cycles(context.options.repeats);
  CUDA_CHECK(cudaMemcpy(cycles.data(), device_cycles.get(),
                        cycles.size() * sizeof(cycles[0]),
                        cudaMemcpyDeviceToHost));
  std::vector<uint32_t> actual(initial.size());
  CUDA_CHECK(cudaMemcpy(actual.data(), destination, allocation,
                        cudaMemcpyDeviceToHost));
  std::vector<uint32_t> expected = initial;
  const auto reduce_plan = model_plan(neutral, 1);
  if (reduce_plan.status != Status::kOk) {
    emit_row(context, neutral, -1, 0, 0, 1, "model_rejected");
    return false;
  }
  std::vector<uint8_t> shared_bytes(neutral.transfer_bytes);
  for (size_t i = 0; i < shared_bytes.size() / 4; ++i) {
    const uint32_t operand = reduce_operand(i);
    std::memcpy(shared_bytes.data() + i * 4, &operand, sizeof(operand));
  }
  const auto mapped_bytes =
      expected_s2g(reduce_plan, shared_bytes, allocation);
  std::vector<uint8_t> mapped_words(expected.size(), 0);
  for (const auto& atom : reduce_plan.atoms)
    for (unsigned lane = 0; lane < 16; ++lane)
      if ((atom.global_mask >> lane) & 1u)
        mapped_words[(atom.global_atom + lane) / 4] = 1;
  for (int repeat = 0; repeat < context.options.repeats; ++repeat)
    for (size_t i = 0; i < expected.size(); ++i) {
      if (!mapped_words[i]) continue;
      uint32_t operand = 0;
      std::memcpy(&operand, mapped_bytes.data() + i * 4, sizeof(operand));
      expected[i] = apply_reduce(op, neutral.dtype == matrix::DType::kS32,
                                 expected[i], operand);
    }
  uint64_t errors = 0;
  for (size_t i = 0; i < actual.size(); ++i) errors += actual[i] != expected[i];
  for (int repeat = 0; repeat < context.options.repeats; ++repeat)
    emit_row(context, neutral, repeat, cycles[repeat],
             elapsed * 1.0e6 / context.options.repeats, errors,
             errors ? "validation_error" : "ok");
  return errors == 0;
}

ComprehensiveOptions parse_comprehensive_options(int argc, char** argv) {
  ComprehensiveOptions options;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    const auto value = [&](const char* name) -> const char* {
      if (++i >= argc) {
        std::fprintf(stderr, "missing value for %s\n", name);
        std::exit(2);
      }
      return argv[i];
    };
    if (arg == "--shard") options.shard = value("--shard");
    else if (arg == "--case") options.single_case = value("--case");
    else if (arg == "--warmups") options.warmups = std::atoi(value("--warmups"));
    else if (arg == "--repeats") options.repeats = std::atoi(value("--repeats"));
    else if (arg == "--list") options.list = true;
    else if (arg == "--csv") {}
    else {
      std::fprintf(stderr, "unknown option: %s\n", argv[i]);
      std::exit(2);
    }
  }
  if (!matrix::valid_shard(options.shard) || options.warmups < 0 ||
      options.warmups > 1 || options.repeats < 1 || options.repeats > 3) {
    std::fprintf(stderr, "invalid shard/warmups/repeats\n");
    std::exit(2);
  }
  return options;
}

void list_manifest(const std::vector<NeutralCase>& cases) {
  std::puts("case_id,shard,family,path,direction,dtype,dtype_bits,rank,bytes,global_offset,interleave,swizzle,oob_fill,reduce_op,risk");
  for (const auto& spec : cases)
    std::printf("%s,%s,%s,%s,%s,%s,%u,%d,%u,%u,%s,%s,%s,%s,%s\n",
                spec.id.c_str(), spec.shard.c_str(), spec.family.c_str(),
                matrix::path_name(spec.path), matrix::direction_name(spec.direction),
                matrix::dtype_name(spec.dtype), spec.element_bits, spec.rank,
                spec.transfer_bytes, spec.global_base_mod128,
                interleave_name(spec.interleave), swizzle_name(spec.swizzle),
                matrix::oob_name(spec.oob_fill),
                matrix::reduce_name(spec.reduce_op), risk_name(spec.expected.risk));
}

}  // namespace

int main(int argc, char** argv) {
  const ComprehensiveOptions options = parse_comprehensive_options(argc, argv);
  std::vector<NeutralCase> cases = matrix::cases_for_shard(options.shard);
  if (!options.single_case.empty()) {
    const NeutralCase* selected = matrix::find_case(cases, options.single_case);
    if (!selected) {
      std::fprintf(stderr, "case not found: %s\n", options.single_case.c_str());
      return 2;
    }
    cases = {*selected};
  }
  if (options.list) {
    list_manifest(cases);
    return 0;
  }

  std::setvbuf(stdout, nullptr, _IONBF, 0);
  Context context;
  context.options.warmups = options.warmups;
  context.options.repeats = options.repeats;
  context.options.seeds = {kComprehensiveSeed};
  CU_CHECK(cuInit(0));
  CUDA_CHECK(cudaSetDevice(0));
  CUDA_CHECK(cudaGetDeviceProperties(&context.prop, 0));
  context.gpu = csv_safe_gpu_name(context.prop.name);
  context.cc = std::to_string(context.prop.major) + "." +
               std::to_string(context.prop.minor);
  size_t descriptor_entries = 0;
  size_t data_bytes = 0;
  for (const auto& spec : cases) {
    if (spec.path != matrix::Path::kTensor) continue;
    const size_t copies =
        spec.direction == matrix::Direction::kRoundtrip ? 2 : 1;
    // One new TensorMap per direction and case.  The two measured commands
    // deliberately reuse that exact entry as cold then hot.
    descriptor_entries += copies;
    for (size_t copy = 0; copy < copies; ++copy) {
      data_bytes = align_up_fresh(data_bytes, 128);
      data_bytes += spec.global_base_mod128 + global_span(spec);
      data_bytes = align_up_fresh(data_bytes, 128);
    }
  }
  FreshDescriptorPool descriptor_pool(std::max<size_t>(1, descriptor_entries));
  FreshDataPool data_pool(std::max<size_t>(128, data_bytes));
  fresh_descriptor_pool = &descriptor_pool;
  fresh_data_pool = &data_pool;
  emit_header();
  int failures = 0;
  for (const auto& spec : cases) {
    std::fprintf(stderr, "CASE_START,%s,%s\n", spec.shard.c_str(),
                 spec.id.c_str());
    std::fflush(stderr);
    bool passed = false;
    if (spec.path == matrix::Path::kBulk) passed = run_bulk(context, spec);
    else if (spec.path == matrix::Path::kReduce) passed = run_reduce(context, spec);
    else passed = run_tensor(context, spec);
    failures += !passed;
  }
  CUDA_CHECK(cudaDeviceSynchronize());
  if (fresh_descriptor_pool->used() != descriptor_entries) {
    std::fprintf(stderr, "fresh TensorMap allocation mismatch: %zu/%zu\n",
                 fresh_descriptor_pool->used(), descriptor_entries);
    return 2;
  }
  fresh_descriptor_pool = nullptr;
  fresh_data_pool = nullptr;
  return failures ? 1 : 0;
}
