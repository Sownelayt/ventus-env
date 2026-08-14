#ifndef VENTUS_BENCHMARKS_TMA_COMPREHENSIVE_MATRIX_H_
#define VENTUS_BENCHMARKS_TMA_COMPREHENSIVE_MATRIX_H_

// Shared, implementation-neutral case matrix for the CUDA and Ventus TMA
// cycle benchmarks.  Keep CUDA and Ventus numeric enum mappings in their
// respective harnesses: in particular, their dtype numbers are not identical.
//
// The matrix deliberately uses paired sweeps instead of a Cartesian product.
// A pair_id groups a feature case with its plain/reference case (or groups a
// one-dimensional capacity/rank/dtype sweep).  This keeps a one-attempt H100
// or B200 run useful without making it unnecessarily expensive.  Risky
// descriptor families have their own shards so an encode failure or GPU fault
// cannot discard the inexpensive, well-established cases.

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tma_comprehensive {

constexpr int kMaxRank = 5;

enum class Path { kBulk, kTensor, kReduce };
enum class Direction { kG2s, kS2g, kRoundtrip };

// Neutral names, not ABI values.  A harness must map every value explicitly.
enum class DType {
  kU8,
  kU16,
  kU32,
  kS32,
  kU64,
  kS64,
  kF16,
  kF32,
  kF32Ftz,
  kF64,
  kBf16,
  kTf32,
  kTf32Ftz,
  kU4Align8,
  kU4Align16,
  kU6Align16,
};

enum class Interleave { kNone, k16B, k32B };
enum class Swizzle { kNone, k32B, k64B, k128B };
enum class OobFill { kZero, kNan };
enum class ReduceOp { kNone, kAdd, kMin, kMax, kAnd, kOr, kXor };

enum class Support {
  kExpectedPass,
  kProbe,               // Legal/interesting, but not yet a frozen positive control.
  kExpectedUnsupported  // The harness should record an encode/feature skip.
};

enum class Risk { kLow, kMedium, kHigh };

struct ExpectedSupport {
  Support h100 = Support::kExpectedPass;
  Support b200 = Support::kExpectedPass;
  Risk risk = Risk::kLow;
  std::string note;
};

struct CaseSpec {
  std::string id;       // Globally unique; suitable for --case and CSV joins.
  std::string family;   // Fine-grained report category.
  std::string shard;    // Cost/fault-containment execution shard.
  std::string pair_id;  // Reference/feature or one-dimensional sweep group.
  bool is_reference = false;

  Path path = Path::kTensor;
  Direction direction = Direction::kG2s;
  DType dtype = DType::kU8;
  ReduceOp reduce_op = ReduceOp::kNone;
  int rank = 2;

  // transfer_bytes is the TMA payload footprint.  For packed formats it is
  // the padded shared/global footprint, while element_bits remains 4 or 6.
  std::uint32_t transfer_bytes = 0;
  std::uint8_t element_bits = 8;
  std::uint8_t global_base_mod128 = 0;
  std::uint8_t shared_base_mod128 = 0;

  std::array<std::uint64_t, kMaxRank> global_dims{};
  std::array<std::uint64_t, kMaxRank - 1> global_strides_bytes{};
  std::array<std::uint32_t, kMaxRank> box_dims{};
  std::array<std::uint32_t, kMaxRank> element_strides{{1, 1, 1, 1, 1}};
  std::array<std::int32_t, kMaxRank> coordinates{};

  Interleave interleave = Interleave::kNone;
  Swizzle swizzle = Swizzle::kNone;
  OobFill oob_fill = OobFill::kZero;
  bool shared_is_linear = true;
  ExpectedSupport expected;
};

inline const char* path_name(Path value) {
  switch (value) {
    case Path::kBulk: return "bulk";
    case Path::kTensor: return "tensor";
    case Path::kReduce: return "reduce";
  }
  return "unknown";
}

inline const char* direction_name(Direction value) {
  switch (value) {
    case Direction::kG2s: return "g2s";
    case Direction::kS2g: return "s2g";
    case Direction::kRoundtrip: return "roundtrip";
  }
  return "unknown";
}

inline const char* dtype_name(DType value) {
  switch (value) {
    case DType::kU8: return "u8";
    case DType::kU16: return "u16";
    case DType::kU32: return "u32";
    case DType::kS32: return "s32";
    case DType::kU64: return "u64";
    case DType::kS64: return "s64";
    case DType::kF16: return "f16";
    case DType::kF32: return "f32";
    case DType::kF32Ftz: return "f32_ftz";
    case DType::kF64: return "f64";
    case DType::kBf16: return "bf16";
    case DType::kTf32: return "tf32";
    case DType::kTf32Ftz: return "tf32_ftz";
    case DType::kU4Align8: return "u4_align8";
    case DType::kU4Align16: return "u4_align16";
    case DType::kU6Align16: return "u6_align16";
  }
  return "unknown";
}

inline const char* reduce_name(ReduceOp value) {
  switch (value) {
    case ReduceOp::kNone: return "none";
    case ReduceOp::kAdd: return "add";
    case ReduceOp::kMin: return "min";
    case ReduceOp::kMax: return "max";
    case ReduceOp::kAnd: return "and";
    case ReduceOp::kOr: return "or";
    case ReduceOp::kXor: return "xor";
  }
  return "unknown";
}

inline const char* oob_name(OobFill value) {
  return value == OobFill::kZero ? "zero" : "nan";
}

inline std::uint8_t dtype_bits(DType value) {
  switch (value) {
    case DType::kU8: return 8;
    case DType::kU16:
    case DType::kF16:
    case DType::kBf16: return 16;
    case DType::kU32:
    case DType::kS32:
    case DType::kF32:
    case DType::kF32Ftz:
    case DType::kTf32:
    case DType::kTf32Ftz: return 32;
    case DType::kU64:
    case DType::kS64:
    case DType::kF64: return 64;
    case DType::kU4Align8:
    case DType::kU4Align16: return 4;
    case DType::kU6Align16: return 6;
  }
  return 0;
}

inline void set_normal_contiguous_shape(
    CaseSpec* spec, int rank,
    const std::array<std::uint64_t, kMaxRank>& dims) {
  spec->rank = rank;
  spec->global_dims = dims;
  spec->box_dims.fill(1);
  spec->element_strides.fill(1);
  for (int d = 0; d < rank; ++d)
    spec->box_dims[d] = static_cast<std::uint32_t>(dims[d]);

  if (spec->element_bits < 8 || (spec->element_bits % 8) != 0)
    throw std::logic_error("packed dtype needs a dedicated physical layout");
  std::uint64_t stride = dims[0] * (spec->element_bits / 8);
  for (int d = 1; d < rank; ++d) {
    spec->global_strides_bytes[d - 1] = stride;
    stride *= dims[d];
  }
  spec->transfer_bytes = static_cast<std::uint32_t>(stride);
}

inline CaseSpec make_tensor_base(const std::string& family,
                                 const std::string& shard,
                                 const std::string& id,
                                 Direction direction, DType dtype = DType::kU8) {
  CaseSpec spec;
  spec.id = id;
  spec.family = family;
  spec.shard = shard;
  spec.pair_id = id;
  spec.path = Path::kTensor;
  spec.direction = direction;
  spec.dtype = dtype;
  spec.element_bits = dtype_bits(dtype);
  return spec;
}

inline CaseSpec make_normal_rank_case(const std::string& family,
                                      const std::string& shard,
                                      const std::string& id,
                                      Direction direction, DType dtype,
                                      int rank,
                                      const std::array<std::uint64_t,
                                                       kMaxRank>& dims) {
  CaseSpec spec = make_tensor_base(family, shard, id, direction, dtype);
  set_normal_contiguous_shape(&spec, rank, dims);
  return spec;
}

inline std::array<std::uint64_t, kMaxRank> rank_shape(int rank,
                                                      int bytes) {
  std::array<std::uint64_t, kMaxRank> dims{};
  if (bytes == 128) {
    if (rank == 1) dims = {{128, 0, 0, 0, 0}};
    if (rank == 2) dims = {{32, 4, 0, 0, 0}};
    if (rank == 3) dims = {{16, 2, 4, 0, 0}};
    if (rank == 4) dims = {{16, 2, 2, 2, 0}};
    if (rank == 5) dims = {{16, 2, 2, 2, 1}};
  } else if (bytes == 256) {
    if (rank == 1) dims = {{256, 0, 0, 0, 0}};
    if (rank == 2) dims = {{64, 4, 0, 0, 0}};
    if (rank == 3) dims = {{16, 4, 4, 0, 0}};
    if (rank == 4) dims = {{16, 2, 4, 2, 0}};
    if (rank == 5) dims = {{16, 2, 2, 2, 2}};
  } else if (bytes == 4096) {
    if (rank == 2) dims = {{128, 32, 0, 0, 0}};
    if (rank == 3) dims = {{64, 8, 8, 0, 0}};
    if (rank == 4) dims = {{32, 4, 4, 8, 0}};
    if (rank == 5) dims = {{16, 4, 4, 4, 4}};
  } else if (bytes == 32768) {
    if (rank == 2) dims = {{128, 256, 0, 0, 0}};
    if (rank == 3) dims = {{128, 16, 16, 0, 0}};
    if (rank == 4) dims = {{64, 8, 8, 8, 0}};
    if (rank == 5) dims = {{32, 4, 8, 8, 4}};
  }
  return dims;
}

inline void append_bulk_capacity(std::vector<CaseSpec>* cases) {
  const std::array<int, 12> sizes{{16, 32, 64, 128, 256, 512, 1024,
                                   2048, 4096, 8192, 16384, 32768}};
  for (Direction direction : {Direction::kG2s, Direction::kS2g}) {
    for (int bytes : sizes) {
      CaseSpec spec;
      spec.id = std::string("bulk_capacity_") + direction_name(direction) +
                "_b" + std::to_string(bytes);
      spec.family = "bulk_capacity";
      spec.shard = "bulk";
      spec.pair_id = std::string("bulk_capacity_") + direction_name(direction);
      spec.is_reference = bytes == sizes.front();
      spec.path = Path::kBulk;
      spec.direction = direction;
      spec.transfer_bytes = static_cast<std::uint32_t>(bytes);
      cases->push_back(std::move(spec));
    }
  }
}

inline void append_tensor_capacity(std::vector<CaseSpec>* cases) {
  const std::array<int, 9> sizes{{128, 256, 512, 1024, 2048, 4096, 8192,
                                  16384, 32768}};
  for (Direction direction :
       {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
    for (int bytes : sizes) {
      const int width = std::min(128, bytes);
      CaseSpec spec = make_normal_rank_case(
          "tensor_capacity", "tensor_capacity",
          std::string("tensor_capacity_") + direction_name(direction) +
              "_b" + std::to_string(bytes),
          direction, DType::kU8, 2,
          {{static_cast<std::uint64_t>(width),
            static_cast<std::uint64_t>(bytes / width), 0, 0, 0}});
      spec.pair_id =
          std::string("tensor_capacity_") + direction_name(direction);
      spec.is_reference = bytes == sizes.front();
      cases->push_back(std::move(spec));
    }
  }
}

// Cost-bounded, opt-in 2D length probe.  Every selected footprint is on a
// 32-byte grid, but the sweep is deliberately stratified instead of covering
// all 512 points through 16 KiB.  The dense small-size region and the +/-32 B
// points around power-of-two boundaries are intended to expose transaction
// quantization without turning one short Modal shard into a Cartesian sweep.
inline std::vector<int> tensor_2d_length_probe_sizes() {
  std::vector<int> sizes;
  for (int bytes = 32; bytes <= 512; bytes += 32) sizes.push_back(bytes);
  for (int bytes = 640; bytes <= 2048; bytes += 128) sizes.push_back(bytes);
  for (int bytes = 2560; bytes <= 8192; bytes += 512) sizes.push_back(bytes);
  for (int bytes : {992, 1056, 2016, 2080, 4064, 4128, 8160})
    sizes.push_back(bytes);
  std::sort(sizes.begin(), sizes.end());
  sizes.erase(std::unique(sizes.begin(), sizes.end()), sizes.end());
  return sizes;
}

inline CaseSpec make_tensor_2d_length_case(
    Direction direction, DType dtype, int row_bytes, int transfer_bytes,
    const std::string& format) {
  const int bits = dtype_bits(dtype);
  if (bits < 8 || (bits % 8) != 0)
    throw std::logic_error("2D length probe requires a byte-addressed dtype");
  const int element_bytes = bits / 8;
  if (row_bytes <= 0 || transfer_bytes <= 0 ||
      (row_bytes % element_bytes) != 0 ||
      (transfer_bytes % row_bytes) != 0)
    throw std::logic_error("invalid 2D length-probe shape");
  const std::string prefix =
      std::string("tensor_2d_length_") + direction_name(direction) + "_" +
      format;
  CaseSpec spec = make_normal_rank_case(
      std::string("tensor_2d_length_") + format, "tensor_2d_length",
      prefix + "_b" + std::to_string(transfer_bytes), direction, dtype, 2,
      {{static_cast<std::uint64_t>(row_bytes / element_bytes),
        static_cast<std::uint64_t>(transfer_bytes / row_bytes), 0, 0, 0}});
  spec.pair_id = prefix;
  spec.is_reference = transfer_bytes == 128;
  if (spec.transfer_bytes != static_cast<std::uint32_t>(transfer_bytes))
    throw std::logic_error("2D length probe byte-count mismatch");
  return spec;
}

inline void append_tensor_2d_length_probe(std::vector<CaseSpec>* cases) {
  const std::vector<int> row32_sizes = tensor_2d_length_probe_sizes();
  std::vector<int> row64_sizes;
  for (int bytes = 8192; bytes <= 16384; bytes += 512)
    row64_sizes.push_back(bytes);
  for (int bytes : {8128, 8256, 16320}) row64_sizes.push_back(bytes);
  std::sort(row64_sizes.begin(), row64_sizes.end());
  row64_sizes.erase(
      std::unique(row64_sizes.begin(), row64_sizes.end()), row64_sizes.end());
  const std::array<int, 17> row128_sizes{{
      128, 256, 512, 1024, 2048, 4096, 8192, 16256, 16384,
      16512, 17408, 18432, 20480, 22528, 24576, 28672, 32768}};
  const std::array<int, 5> u16_row32_controls{{
      128, 256, 1024, 4096, 8192}};
  const std::array<int, 2> u16_row128_controls{{16384, 32768}};
  for (Direction direction : {Direction::kG2s, Direction::kS2g}) {
    for (int bytes : row32_sizes)
      cases->push_back(make_tensor_2d_length_case(
          direction, DType::kU8, 32, bytes, "u8_row32"));
    for (int bytes : row64_sizes)
      cases->push_back(make_tensor_2d_length_case(
          direction, DType::kU8, 64, bytes, "u8_row64"));
    for (int bytes : row128_sizes)
      cases->push_back(make_tensor_2d_length_case(
          direction, DType::kU8, 128, bytes, "u8_row128"));
    for (int bytes : u16_row32_controls)
      cases->push_back(make_tensor_2d_length_case(
          direction, DType::kU16, 32, bytes, "u16_row32"));
    for (int bytes : u16_row128_controls)
      cases->push_back(make_tensor_2d_length_case(
          direction, DType::kU16, 128, bytes, "u16_row128"));
  }
}

inline void append_rank_sweep(std::vector<CaseSpec>* cases) {
  for (Direction direction :
       {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
    for (int bytes : {128, 256, 4096, 32768}) {
      const int first_rank = bytes <= 256 ? 1 : 2;
      for (int rank = first_rank; rank <= 5; ++rank) {
        CaseSpec spec = make_normal_rank_case(
            "rank", "tensor_geometry",
            std::string("rank_") + direction_name(direction) + "_r" +
                std::to_string(rank) + "_b" + std::to_string(bytes),
            direction, DType::kU8, rank, rank_shape(rank, bytes));
        spec.pair_id = std::string("rank_") + direction_name(direction) +
                       "_b" + std::to_string(bytes);
        spec.is_reference = rank == 2;
        cases->push_back(std::move(spec));
      }
    }
  }
}

inline void append_base_alignment(std::vector<CaseSpec>* cases) {
  const std::array<int, 5> offsets{{0, 16, 32, 64, 112}};
  for (Path path : {Path::kBulk, Path::kTensor}) {
    const std::array<Direction, 3> directions{{
        Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}};
    const int direction_count = path == Path::kBulk ? 2 : 3;
    for (int di = 0; di < direction_count; ++di) {
      const Direction direction = directions[di];
      for (int offset : offsets) {
        const std::string prefix =
            std::string(path_name(path)) + "_base_" + direction_name(direction);
        CaseSpec spec;
        if (path == Path::kTensor) {
          spec = make_normal_rank_case(
              "base_alignment", "tensor_geometry",
              prefix + "_mod128_" + std::to_string(offset), direction,
              DType::kU8, 2, {{128, 32, 0, 0, 0}});
        } else {
          spec.id = prefix + "_mod128_" + std::to_string(offset);
          spec.family = "base_alignment";
          spec.shard = "bulk";
          spec.path = Path::kBulk;
          spec.direction = direction;
          spec.transfer_bytes = 4096;
        }
        spec.pair_id = prefix;
        spec.is_reference = offset == 0;
        spec.global_base_mod128 = static_cast<std::uint8_t>(offset);
        if (offset != 0) {
          spec.expected.risk = Risk::kMedium;
          spec.expected.note =
              "legal 16-byte alignment crossing a 128-byte line boundary";
        }
        cases->push_back(std::move(spec));
      }
    }
  }
}

inline void append_dtype_sweep(std::vector<CaseSpec>* cases) {
  const std::array<DType, 13> dtypes{{
      DType::kU8, DType::kU16, DType::kU32, DType::kS32, DType::kU64,
      DType::kS64, DType::kF16, DType::kF32, DType::kF32Ftz,
      DType::kF64, DType::kBf16, DType::kTf32, DType::kTf32Ftz}};
  for (Direction direction :
       {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
    for (int bytes : {128, 4096}) {
      for (DType dtype : dtypes) {
        const std::uint64_t elements_per_row = 128 / (dtype_bits(dtype) / 8);
        CaseSpec spec = make_normal_rank_case(
            "dtype", "dtype",
            std::string("dtype_") + direction_name(direction) + "_" +
                dtype_name(dtype) + "_b" + std::to_string(bytes),
            direction, dtype, 2,
            {{elements_per_row, static_cast<std::uint64_t>(bytes / 128),
              0, 0, 0}});
        spec.pair_id = std::string("dtype_") + direction_name(direction) +
                       "_b" + std::to_string(bytes);
        spec.is_reference = dtype == DType::kU8;
        cases->push_back(std::move(spec));
      }
    }
  }
}

inline void append_nan_oob_sweep(std::vector<CaseSpec>* cases) {
  const std::array<DType, 7> floating{{
      DType::kF16, DType::kF32, DType::kF32Ftz, DType::kF64,
      DType::kBf16, DType::kTf32, DType::kTf32Ftz}};
  for (DType dtype : floating) {
    const std::uint64_t elements_per_row = 128 / (dtype_bits(dtype) / 8);
    const std::string pair = std::string("oob_nan_") + dtype_name(dtype);
    for (OobFill fill : {OobFill::kZero, OobFill::kNan}) {
      CaseSpec spec = make_normal_rank_case(
          "oob_nan", "dtype",
          pair + "_" + oob_name(fill), Direction::kG2s, dtype, 2,
          {{elements_per_row, 32, 0, 0, 0}});
      spec.pair_id = pair;
      spec.is_reference = fill == OobFill::kZero;
      spec.coordinates[0] = -static_cast<int32_t>(elements_per_row / 2);
      spec.oob_fill = fill;
      cases->push_back(std::move(spec));
    }
  }
}

inline void append_subbyte(std::vector<CaseSpec>* cases) {
  struct PackedPoint {
    DType dtype;
    Direction direction;
    int dim0;
    int rows;
    int base_mod128;
  };
  // These are three dedicated legal descriptor points, not a generic packed
  // Cartesian sweep.  ALIGN16 rows include their documented physical padding.
  const std::array<PackedPoint, 3> points{{
      {DType::kU4Align8, Direction::kRoundtrip, 256, 8, 0},
      {DType::kU4Align16, Direction::kG2s, 128, 8, 32},
      {DType::kU6Align16, Direction::kRoundtrip, 128, 8, 32},
  }};
  for (const PackedPoint& point : points) {
    CaseSpec spec = make_tensor_base(
        "subbyte", "subbyte",
        std::string("subbyte_") + direction_name(point.direction) + "_" +
            dtype_name(point.dtype),
        point.direction, point.dtype);
    spec.pair_id = "subbyte_legal_points";
    spec.rank = 2;
    spec.transfer_bytes = 1024;
    spec.global_base_mod128 =
        static_cast<std::uint8_t>(point.base_mod128);
    spec.global_dims = {{static_cast<std::uint64_t>(point.dim0),
                         static_cast<std::uint64_t>(point.rows), 0, 0, 0}};
    spec.box_dims = {{static_cast<std::uint32_t>(point.dim0),
                      static_cast<std::uint32_t>(point.rows), 1, 1, 1}};
    spec.global_strides_bytes[0] = 128;
    spec.expected.h100 = Support::kProbe;
    spec.expected.b200 = Support::kExpectedPass;
    spec.expected.risk = Risk::kHigh;
    spec.expected.note =
        "packed TensorMap support is isolated from the main paid run";
    cases->push_back(std::move(spec));
  }
}

inline void append_stride_sweep(std::vector<CaseSpec>* cases) {
  const std::array<int, 4> strides{{128, 144, 256, 512}};
  for (Direction direction :
       {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
    for (int stride : strides) {
      CaseSpec spec = make_normal_rank_case(
          "stride", "tensor_geometry",
          std::string("stride_") + direction_name(direction) + "_s" +
              std::to_string(stride),
          direction, DType::kU8, 2, {{128, 32, 0, 0, 0}});
      spec.global_strides_bytes[0] = stride;
      spec.pair_id = std::string("stride_") + direction_name(direction);
      spec.is_reference = stride == strides.front();
      cases->push_back(std::move(spec));
    }
  }
}

inline void append_subbox_sweep(std::vector<CaseSpec>* cases) {
  struct Shape {
    const char* name;
    int rank;
    std::array<std::uint64_t, kMaxRank> global;
    std::array<std::uint64_t, kMaxRank> box;
    std::array<std::int32_t, kMaxRank> coords;
  };
  const std::array<Shape, 3> shapes{{
      {"small2d", 2, {{128, 32, 0, 0, 0}}, {{32, 4, 0, 0, 0}},
       // Keep the effective global address 16B aligned on CUDA.  An x=17
       // origin encodes successfully but raises XID 13 when issued on H100.
       {{16, 7, 0, 0, 0}}},
      {"large2d", 2, {{256, 64, 0, 0, 0}}, {{128, 32, 0, 0, 0}},
       {{64, 16, 0, 0, 0}}},
      {"large3d", 3, {{128, 16, 16, 0, 0}}, {{64, 8, 8, 0, 0}},
       {{32, 4, 4, 0, 0}}},
  }};
  for (Direction direction :
       {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
    for (const Shape& shape : shapes) {
      const std::string pair = std::string("subbox_") +
                               direction_name(direction) + "_" + shape.name;
      for (bool subbox : {false, true}) {
        CaseSpec spec = make_normal_rank_case(
            "subbox", "tensor_geometry",
            pair + (subbox ? "_offset" : "_plain"), direction, DType::kU8,
            shape.rank, subbox ? shape.global : shape.box);
        if (subbox) {
          spec.global_dims = shape.global;
          spec.coordinates = shape.coords;
          // Recompute strides from the full global allocation, not the box.
          std::uint64_t stride = shape.global[0];
          for (int d = 1; d < shape.rank; ++d) {
            spec.global_strides_bytes[d - 1] = stride;
            stride *= shape.global[d];
          }
        }
        spec.box_dims.fill(1);
        std::uint64_t bytes = 1;
        for (int d = 0; d < shape.rank; ++d) {
          spec.box_dims[d] = static_cast<std::uint32_t>(shape.box[d]);
          bytes *= shape.box[d];
        }
        spec.transfer_bytes = static_cast<std::uint32_t>(bytes);
        spec.pair_id = pair;
        spec.is_reference = !subbox;
        cases->push_back(std::move(spec));
      }
    }
  }
}

inline void append_oob_sweep(std::vector<CaseSpec>* cases) {
  struct OobPoint {
    const char* name;
    int x;
    int y;
    bool reference;
  };
  const std::array<OobPoint, 7> points{{
      {"plain", 0, 0, true},
      // Hopper accepts signed OOB coordinates, but an effective first-byte
      // address outside the 16B TMA phase faults with XID 13 (for example
      // x=-1, just as an in-bounds x=17 does).  Use the smallest aligned
      // negative origin so this is an OOB-fill measurement, not an invalid
      // instruction-parameter probe.
      {"left12p5", -16, 0, false},
      {"left25", -32, 0, false}, {"left50", -64, 0, false},
      {"left100", -128, 0, false}, {"right25", 32, 0, false},
      {"outer75", 0, 24, false},
  }};
  for (Direction direction : {Direction::kG2s, Direction::kS2g}) {
    for (const OobPoint& point : points) {
      // Ventus follows the S2G suppress contract only for a non-negative
      // origin.  Negative-origin fill is a G2S-only feature.
      if (direction == Direction::kS2g && point.x < 0) continue;
      CaseSpec spec = make_normal_rank_case(
          "oob", "tensor_geometry",
          std::string("oob_") + direction_name(direction) + "_" + point.name,
          direction, DType::kU8, 2, {{128, 32, 0, 0, 0}});
      spec.pair_id = std::string("oob_") + direction_name(direction);
      spec.is_reference = point.reference;
      spec.coordinates[0] = point.x;
      spec.coordinates[1] = point.y;
      cases->push_back(std::move(spec));
    }
  }
}

inline void append_swizzle_sweep(std::vector<CaseSpec>* cases) {
  struct SwizzlePoint { int span; Swizzle mode; };
  const std::array<SwizzlePoint, 3> modes{{
      {32, Swizzle::k32B}, {64, Swizzle::k64B}, {128, Swizzle::k128B}}};
  for (const SwizzlePoint& point : modes) {
    for (int bytes : {1024, 4096, 16384}) {
      const int middle = bytes == 1024 ? 1 : (bytes == 4096 ? 8 : 16);
      const int outer = bytes / (point.span * middle);
      const int rank = middle == 1 ? 2 : 3;
      const std::array<std::uint64_t, kMaxRank> dims{{
          static_cast<std::uint64_t>(point.span),
          static_cast<std::uint64_t>(middle == 1 ? outer : middle),
          static_cast<std::uint64_t>(middle == 1 ? 0 : outer), 0, 0}};
      for (Direction direction :
           {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
        const std::string pair = "swizzle" + std::to_string(point.span) +
                                 "_" + direction_name(direction) + "_b" +
                                 std::to_string(bytes);
        for (bool enabled : {false, true}) {
          CaseSpec spec = make_normal_rank_case(
              "swizzle", "layout", pair + (enabled ? "_swz" : "_plain"),
              direction, DType::kU8, rank, dims);
          spec.pair_id = pair;
          spec.is_reference = !enabled;
          if (enabled) {
            spec.swizzle = point.mode;
            spec.shared_is_linear = false;
          }
          cases->push_back(std::move(spec));
        }
      }
    }
  }
}

inline void append_interleave_sweep(std::vector<CaseSpec>* cases) {
  for (int atom : {16, 32}) {
    for (int bytes : {1024, 4096, 16384}) {
      const int channels = atom / 2;  // UINT16 whole-atom positive control.
      const int height = atom == 16 ? 8 : 4;
      const int outer = bytes / (atom * height);
      const std::array<std::uint64_t, kMaxRank> dims{{
          static_cast<std::uint64_t>(channels),
          static_cast<std::uint64_t>(height),
          static_cast<std::uint64_t>(outer), 0, 0}};
      for (Direction direction :
           {Direction::kG2s, Direction::kS2g, Direction::kRoundtrip}) {
        const std::string pair = "interleave" + std::to_string(atom) + "_" +
                                 direction_name(direction) + "_b" +
                                 std::to_string(bytes);
        for (bool enabled : {false, true}) {
          CaseSpec spec = make_normal_rank_case(
              "interleave", atom == 16 ? "interleave16" : "interleave32",
              pair + (enabled ? "_int" : "_plain"), direction, DType::kU16,
              3, dims);
          spec.pair_id = pair;
          spec.is_reference = !enabled;
          if (enabled) {
            spec.interleave = atom == 16 ? Interleave::k16B : Interleave::k32B;
            spec.swizzle = atom == 32 ? Swizzle::k32B : Swizzle::kNone;
            spec.shared_is_linear = false;
            if (atom == 16) {
              spec.expected.risk = Risk::kMedium;
              spec.expected.note =
                  "validated UINT16 whole-atom control; partial/U8 shapes excluded";
            } else {
              spec.expected.h100 = Support::kProbe;
              spec.expected.b200 = Support::kProbe;
              spec.expected.risk = Risk::kHigh;
              spec.expected.note =
                  "UINT16 interleave32 qualification; isolated after all safe shards";
            }
          }
          cases->push_back(std::move(spec));
        }
      }
    }
  }
}

inline CaseSpec make_reduce_case(const std::string& family,
                                 const std::string& id, DType dtype,
                                 ReduceOp op, int rank,
                                 const std::array<std::uint64_t,
                                                  kMaxRank>& dims) {
  CaseSpec spec = make_normal_rank_case(family, "reduce", id,
                                       Direction::kS2g, dtype, rank, dims);
  spec.path = Path::kReduce;
  spec.reduce_op = op;
  return spec;
}

inline void append_reduce_sweeps(std::vector<CaseSpec>* cases) {
  const std::array<DType, 2> dtypes{{DType::kU32, DType::kS32}};
  const std::array<ReduceOp, 6> ops{{ReduceOp::kAdd, ReduceOp::kMin,
                                     ReduceOp::kMax, ReduceOp::kAnd,
                                     ReduceOp::kOr, ReduceOp::kXor}};
  for (DType dtype : dtypes) {
    for (ReduceOp op : ops) {
      CaseSpec spec = make_reduce_case(
          "reduce_ops",
          std::string("reduce_ops_") + dtype_name(dtype) + "_" +
              reduce_name(op) + "_b4096",
          dtype, op, 2, {{32, 32, 0, 0, 0}});
      spec.pair_id = std::string("reduce_ops_") + dtype_name(dtype);
      spec.is_reference = op == ReduceOp::kAdd;
      cases->push_back(std::move(spec));
    }

    for (int bytes : {128, 512, 1024, 16384, 32768}) {
      CaseSpec spec = make_reduce_case(
          "reduce_capacity",
          std::string("reduce_capacity_") + dtype_name(dtype) + "_add_b" +
              std::to_string(bytes),
          dtype, ReduceOp::kAdd, 2,
          {{32, static_cast<std::uint64_t>(bytes / 128), 0, 0, 0}});
      spec.pair_id =
          std::string("reduce_capacity_") + dtype_name(dtype) + "_add";
      spec.is_reference = bytes == 128;
      cases->push_back(std::move(spec));
    }

    const std::array<std::array<std::uint64_t, kMaxRank>, 5> shapes{{
        {{64, 0, 0, 0, 0}}, {{32, 32, 0, 0, 0}},
        {{16, 8, 8, 0, 0}}, {{8, 4, 4, 8, 0}}, {{4, 4, 4, 4, 4}}}};
    for (int rank = 1; rank <= 5; ++rank) {
      CaseSpec spec = make_reduce_case(
          "reduce_rank",
          std::string("reduce_rank_") + dtype_name(dtype) + "_add_r" +
              std::to_string(rank),
          dtype, ReduceOp::kAdd, rank, shapes[rank - 1]);
      spec.pair_id =
          std::string("reduce_rank_") + dtype_name(dtype) + "_add";
      spec.is_reference = rank == 2;
      cases->push_back(std::move(spec));
    }
  }

  for (int offset : {0, 16, 32, 64, 112}) {
    CaseSpec spec = make_reduce_case(
        "reduce_alignment", "reduce_alignment_u32_add_mod128_" +
                                std::to_string(offset),
        DType::kU32, ReduceOp::kAdd, 2, {{32, 32, 0, 0, 0}});
    spec.pair_id = "reduce_alignment_u32_add";
    spec.is_reference = offset == 0;
    spec.global_base_mod128 = static_cast<std::uint8_t>(offset);
    cases->push_back(std::move(spec));
  }
  for (int stride : {128, 144, 256, 512}) {
    CaseSpec spec = make_reduce_case(
        "reduce_stride", "reduce_stride_u32_add_s" + std::to_string(stride),
        DType::kU32, ReduceOp::kAdd, 2, {{32, 32, 0, 0, 0}});
    spec.global_strides_bytes[0] = stride;
    spec.pair_id = "reduce_stride_u32_add";
    spec.is_reference = stride == 128;
    cases->push_back(std::move(spec));
  }
  for (const auto& [name, x, y] :
       {std::tuple{"right50", 16, 0}, std::tuple{"right25", 8, 0},
        std::tuple{"outer75", 0, 24}}) {
    CaseSpec spec = make_reduce_case(
        "reduce_oob", std::string("reduce_oob_u32_add_") + name,
        DType::kU32, ReduceOp::kAdd, 2, {{32, 32, 0, 0, 0}});
    spec.coordinates[0] = x;
    spec.coordinates[1] = y;
    spec.pair_id = "reduce_oob_u32_add";
    cases->push_back(std::move(spec));
  }
}

inline void validate_cases(const std::vector<CaseSpec>& cases) {
  std::unordered_set<std::string> ids;
  for (const CaseSpec& spec : cases) {
    if (spec.id.empty() || spec.family.empty() || spec.shard.empty() ||
        spec.pair_id.empty())
      throw std::logic_error("TMA comprehensive case has an empty key");
    if (!ids.insert(spec.id).second)
      throw std::logic_error("duplicate TMA comprehensive case id: " + spec.id);
    if ((spec.global_base_mod128 % 16) != 0)
      throw std::logic_error("global base offset is not 16-byte aligned: " +
                             spec.id);
    if (spec.path != Path::kBulk && (spec.rank < 1 || spec.rank > kMaxRank))
      throw std::logic_error("invalid TensorMap rank: " + spec.id);
    if (spec.rank == 1 && spec.transfer_bytes > 256 &&
        spec.family == "rank")
      throw std::logic_error("rank-1 sweep exceeds 256 bytes: " + spec.id);
  }
}

inline std::vector<CaseSpec> make_cases() {
  std::vector<CaseSpec> cases;
  cases.reserve(420);
  append_bulk_capacity(&cases);
  append_tensor_capacity(&cases);
  append_rank_sweep(&cases);
  append_base_alignment(&cases);
  append_dtype_sweep(&cases);
  append_nan_oob_sweep(&cases);
  append_subbyte(&cases);
  append_stride_sweep(&cases);
  append_subbox_sweep(&cases);
  append_oob_sweep(&cases);
  append_swizzle_sweep(&cases);
  append_interleave_sweep(&cases);
  append_reduce_sweeps(&cases);
  validate_cases(cases);
  return cases;
}

inline std::vector<CaseSpec> make_tensor_2d_length_probe_cases() {
  std::vector<CaseSpec> cases;
  cases.reserve(184);
  append_tensor_2d_length_probe(&cases);
  validate_cases(cases);
  return cases;
}

inline std::vector<std::string> shards() {
  return {"bulk", "tensor_capacity", "tensor_geometry", "dtype", "layout",
          "interleave16", "subbyte", "reduce", "interleave32",
          "tensor_2d_length"};
}

inline bool valid_shard(std::string_view shard) {
  if (shard == "all") return true;
  const std::vector<std::string> names = shards();
  return std::find(names.begin(), names.end(), shard) != names.end();
}

inline std::vector<CaseSpec> cases_for_shard(std::string_view shard) {
  if (!valid_shard(shard))
    throw std::invalid_argument("unknown TMA comprehensive shard: " +
                                std::string(shard));
  // The dense probe is explicit opt-in.  Keep the historical meaning and
  // cost of --shard all fixed at the original 399-case comprehensive matrix.
  if (shard == "tensor_2d_length")
    return make_tensor_2d_length_probe_cases();
  const std::vector<CaseSpec> all = make_cases();
  if (shard == "all") return all;
  std::vector<CaseSpec> selected;
  for (const CaseSpec& spec : all)
    if (spec.shard == shard) selected.push_back(spec);
  return selected;
}

inline std::vector<std::string> list_cases(std::string_view shard = "all") {
  const std::vector<CaseSpec> selected = cases_for_shard(shard);
  std::vector<std::string> ids;
  ids.reserve(selected.size());
  for (const CaseSpec& spec : selected) ids.push_back(spec.id);
  return ids;
}

inline const CaseSpec* find_case(const std::vector<CaseSpec>& cases,
                                 std::string_view id) {
  const auto it = std::find_if(cases.begin(), cases.end(),
                               [id](const CaseSpec& spec) {
                                 return spec.id == id;
                               });
  return it == cases.end() ? nullptr : &*it;
}

}  // namespace tma_comprehensive

#endif  // VENTUS_BENCHMARKS_TMA_COMPREHENSIVE_MATRIX_H_
