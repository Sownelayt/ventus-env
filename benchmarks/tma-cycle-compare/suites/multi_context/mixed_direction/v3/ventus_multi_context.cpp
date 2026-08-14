// One-case host launcher for the Ventus V3.10 multi-context probe.

#include <CL/cl.h>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <string>
#include <string_view>
#include <vector>

#include "ventus_opencl_test.h"
#include "ventus_tma_v2_spec.h"

namespace {
constexpr size_t kWorkgroup = 32;
constexpr uint32_t kDescriptorWords = 32;
constexpr uint32_t kDescriptorSlots = 32;
constexpr uint32_t kResultWords = 4;
constexpr size_t kVentusLocalBytes = 64 * 1024;

struct Options {
  std::string project = "same";
  std::string level = "intra_cta";
  std::string path = "tensor";
  std::string direction = "g2s";
  std::string order = "alternating";
  std::string map_mode = "same_map";
  std::string mode = "batched";
  uint32_t bytes = 128;
  uint32_t contexts = 1;
};

struct Environment {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel patch = nullptr;
  cl_kernel operation = nullptr;

  bool build(const std::filesystem::path& source_path,
             const Options& selected, uint32_t descriptor_base) {
    if (operation) clReleaseKernel(operation);
    if (patch) clReleaseKernel(patch);
    if (program) clReleaseProgram(program);
    operation = nullptr;
    patch = nullptr;
    program = nullptr;
    cl_int error = CL_SUCCESS;
    size_t source_size = 0;
    char* source = ventus_read_text_file(source_path.c_str(), &source_size);
    if (!source) return false;
    const char* sources[] = {source};
    const size_t sizes[] = {source_size};
    program = clCreateProgramWithSource(context, 1, sources, sizes, &error);
    std::free(source);
    if (error != CL_SUCCESS) return false;
    const char* root_value = std::getenv("VENTUS_ENV_PATH");
    const std::filesystem::path root =
        root_value ? root_value : std::filesystem::current_path();
    std::string build_options =
        "-I" + (root / "testcases/_get_case/common").string() +
        " -DVENTUS_MULTI_SPECIALIZED=1" +
        " -DVENTUS_MULTI_PATH=" +
            std::to_string(selected.path == "tensor" ? 1 : 0) +
        " -DVENTUS_MULTI_DIRECTION=" +
            std::to_string(selected.direction == "s2g" ? 1 : 0) +
        " -DVENTUS_MULTI_DISTINCT=" +
            std::to_string(selected.map_mode == "distinct_map" ? 1 : 0) +
        " -DVENTUS_MULTI_BATCHED=" +
            std::to_string(selected.mode == "batched" ? 1 : 0) +
        " -DVENTUS_MULTI_BYTES=" + std::to_string(selected.bytes) +
        " -DVENTUS_MULTI_CONTEXTS=" + std::to_string(selected.contexts);
    if (descriptor_base != 0)
      build_options += " -DVENTUS_DESCRIPTOR_BASE=" +
                       std::to_string(descriptor_base);
    error = clBuildProgram(program, 1, &device, build_options.c_str(), nullptr,
                           nullptr);
    if (error != CL_SUCCESS) {
      ventus_print_build_log(program, device);
      return false;
    }
    const auto make = [&](const char* name) {
      cl_int status = CL_SUCCESS;
      cl_kernel kernel = clCreateKernel(program, name, &status);
      return status == CL_SUCCESS ? kernel : nullptr;
    };
    patch = make("patch_descriptor_bases");
    std::string operation_name;
    if (selected.project == "same")
      operation_name = selected.level == "multi_cta"
                           ? "same_direction_multi_kernel"
                           : "same_direction_intra_" +
                                 std::to_string(selected.contexts) + "_kernel";
    else
      operation_name = "mixed_direction_" + selected.path +
                       (selected.path == "tensor" ? "_" + selected.map_mode : "") +
                       (selected.level == "multi_cta"
                            ? "_multi_" + selected.order + "_" + selected.mode + "_kernel"
                            : "_intra_" + std::to_string(selected.contexts) +
                                  "_" + selected.order + "_" + selected.mode + "_kernel");
    operation = make(operation_name.c_str());
    return patch && operation;
  }

  bool initialize(const std::filesystem::path& source_path,
                  const Options& selected) {
    cl_int error = ventus_get_default_device(&context, &device, &queue, nullptr);
    return error == CL_SUCCESS && build(source_path, selected, 0);
  }

  ~Environment() {
    if (operation) clReleaseKernel(operation);
    if (patch) clReleaseKernel(patch);
    if (program) clReleaseProgram(program);
    if (queue) clReleaseCommandQueue(queue);
    if (context) clReleaseContext(context);
  }
};

struct Buffers {
  std::vector<cl_mem> values;
  ~Buffers() {
    for (auto it = values.rbegin(); it != values.rend(); ++it)
      if (*it) clReleaseMemObject(*it);
  }
};

template <typename T>
cl_mem make_buffer(Environment& env, Buffers& buffers, cl_mem_flags flags,
                   std::vector<T>& values) {
  cl_int error = CL_SUCCESS;
  cl_mem result = clCreateBuffer(env.context, flags | CL_MEM_COPY_HOST_PTR,
                                 values.size() * sizeof(T), values.data(),
                                 &error);
  if (error != CL_SUCCESS) return nullptr;
  buffers.values.push_back(result);
  return result;
}

bool set_arg(cl_kernel kernel, cl_uint index, size_t size, const void* value) {
  return clSetKernelArg(kernel, index, size, value) == CL_SUCCESS;
}

bool launch(Environment& env, cl_kernel kernel, size_t global,
            size_t local = kWorkgroup) {
  return clEnqueueNDRangeKernel(env.queue, kernel, 1, nullptr, &global, &local,
                                0, nullptr, nullptr) == CL_SUCCESS &&
         clFinish(env.queue) == CL_SUCCESS;
}

template <typename T>
bool read(Environment& env, cl_mem object, std::vector<T>& values) {
  return clEnqueueReadBuffer(env.queue, object, CL_TRUE, 0,
                             values.size() * sizeof(T), values.data(), 0,
                             nullptr, nullptr) == CL_SUCCESS;
}

uint8_t pattern(size_t index) {
  uint32_t value = static_cast<uint32_t>(index) ^ 0x5a19u;
  value ^= value >> 15;
  value *= 0x7feb352du;
  value ^= value >> 16;
  return static_cast<uint8_t>(value);
}

uint64_t fingerprint64(const uint8_t* data, size_t bytes) {
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= data[index];
    value *= 1099511628211ull;
  }
  return value;
}

std::vector<uint32_t> descriptors(const Options& options) {
  const uint32_t count = options.map_mode == "distinct_map" ? options.contexts : 1;
  const uint32_t rows = options.bytes / 128;
  std::vector<uint32_t> result(static_cast<size_t>(count) * kDescriptorWords, 0);
  for (uint32_t index = 0; index < count; ++index) {
    uint32_t* words = result.data() + index * kDescriptorWords;
    words[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
    words[VENTUS_TMA_V2_WORD_CONTROL] =
        VENTUS_TMA_DTYPE_U8 | (2u << 5) |
        (static_cast<uint32_t>(VENTUS_TMA_INTERLEAVE_NONE) << 8) |
        (static_cast<uint32_t>(VENTUS_TMA_SWIZZLE_NONE) << 10) |
        (static_cast<uint32_t>(VENTUS_TMA_OOB_ZERO) << 18);
    words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS] = 128;
    words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + 1] =
        options.map_mode == "distinct_map" ? rows : rows * options.contexts;
    words[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES] = 128;
    words[VENTUS_TMA_V2_WORD_BOX_DIMS] = 128;
    words[VENTUS_TMA_V2_WORD_BOX_DIMS + 1] = rows;
    words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES] = 1;
    words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + 1] = 1;
  }
  return result;
}

Options parse(int argc, char** argv) {
  Options result;
  for (int index = 1; index < argc; ++index) {
    const std::string_view argument(argv[index]);
    const auto next = [&](const char* name) {
      if (++index >= argc) {
        std::cerr << "missing " << name << '\n';
        std::exit(2);
      }
      return argv[index];
    };
    if (argument == "--project") result.project = next("--project");
    else if (argument == "--level") result.level = next("--level");
    else if (argument == "--path") result.path = next("--path");
    else if (argument == "--direction") result.direction = next("--direction");
    else if (argument == "--order") result.order = next("--order");
    else if (argument == "--map-mode") result.map_mode = next("--map-mode");
    else if (argument == "--mode") result.mode = next("--mode");
    else if (argument == "--bytes") result.bytes = std::strtoul(next("--bytes"), nullptr, 0);
    else if (argument == "--contexts") result.contexts = std::strtoul(next("--contexts"), nullptr, 0);
    else {
      std::cerr << "unknown option: " << argument << '\n';
      std::exit(2);
    }
  }
  const bool valid =
      (result.project == "same" || result.project == "mixed") &&
      (result.level == "intra_cta" || result.level == "multi_cta") &&
      (result.path == "bulk" || result.path == "tensor") &&
      (result.direction == "g2s" || result.direction == "s2g" ||
       result.direction == "mixed") &&
      (result.order == "alternating" || result.order == "g2s_then_s2g" ||
       result.order == "s2g_then_g2s") &&
      (result.map_mode == "same_map" || result.map_mode == "distinct_map") &&
      (result.mode == "serial" || result.mode == "batched") &&
      (result.bytes == 128 || result.bytes == 4096) &&
      result.contexts >= 1 && result.contexts <= 32;
  if (!valid || (result.project == "same" && result.direction == "mixed") ||
      (result.project == "mixed" && result.direction != "mixed")) {
    std::cerr << "invalid multi-context case\n";
    std::exit(2);
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  const Options options = parse(argc, argv);
  const auto here = std::filesystem::path(argv[0]).parent_path();
  Environment env;
  if (!env.initialize(here / "ventus_multi_context.cl", options)) return 1;
  const size_t payload_bytes =
      static_cast<size_t>(options.contexts) * options.bytes;
  std::vector<uint8_t> source(2 * payload_bytes);
  for (size_t index = 0; index < source.size(); ++index)
    source[index] = pattern(index % payload_bytes);
  std::vector<uint8_t> destination(2 * payload_bytes, 0);
  std::vector<uint8_t> capture(payload_bytes, 0);
  std::vector<uint8_t> mixed_output(
      options.project == "mixed" ? 3 * payload_bytes : 1, 0);
  std::vector<int32_t> coordinates(static_cast<size_t>(options.contexts) * 32, 0);
  const int32_t rows = options.bytes / 128;
  if (options.map_mode == "same_map")
    for (uint32_t index = 0; index < options.contexts; ++index)
      coordinates[index * 32 + 1] = index * rows;
  std::vector<uint32_t> load_descriptor_data = descriptors(options);
  std::vector<uint32_t> descriptor_data(
      2u * kDescriptorSlots * kDescriptorWords, 0);
  for (uint32_t bank = 0; bank < 2; ++bank)
    std::copy(load_descriptor_data.begin(), load_descriptor_data.end(),
              descriptor_data.begin() +
                  bank * kDescriptorSlots * kDescriptorWords);
  const uint32_t result_entities = options.level == "multi_cta" ? options.contexts : 1;
  std::vector<uint32_t> results(static_cast<size_t>(2 * result_entities) * kResultWords, 0);
  std::vector<uint32_t> descriptor_identity(1, 0);
  Buffers buffers;
  cl_mem source_buffer = make_buffer(env, buffers, CL_MEM_READ_WRITE, source);
  cl_mem destination_buffer = make_buffer(env, buffers, CL_MEM_READ_WRITE, destination);
  cl_mem capture_buffer = make_buffer(env, buffers, CL_MEM_READ_WRITE, capture);
  cl_mem mixed_output_buffer =
      make_buffer(env, buffers, CL_MEM_READ_WRITE, mixed_output);
  cl_mem coordinate_buffer = make_buffer(env, buffers, CL_MEM_READ_ONLY, coordinates);
  cl_mem descriptor_buffer =
      make_buffer(env, buffers, CL_MEM_READ_WRITE, descriptor_data);
  cl_mem result_buffer = make_buffer(env, buffers, CL_MEM_READ_WRITE, results);
  cl_mem descriptor_identity_buffer =
      make_buffer(env, buffers, CL_MEM_READ_WRITE, descriptor_identity);
  if (!source_buffer || !destination_buffer || !capture_buffer ||
      !mixed_output_buffer || !coordinate_buffer || !descriptor_buffer ||
      !result_buffer || !descriptor_identity_buffer) return 1;
  const uint32_t descriptor_count =
      options.map_mode == "distinct_map" ? options.contexts : 1;
  const uint32_t distinct = options.map_mode == "distinct_map";
  const auto patch = [&](uint32_t descriptor_start, cl_mem base,
                         uint32_t base_offset) {
    return set_arg(env.patch, 0, sizeof(descriptor_buffer), &descriptor_buffer) &&
           set_arg(env.patch, 1, sizeof(descriptor_start), &descriptor_start) &&
           set_arg(env.patch, 2, sizeof(base), &base) &&
           set_arg(env.patch, 3, sizeof(descriptor_count), &descriptor_count) &&
           set_arg(env.patch, 4, sizeof(options.bytes), &options.bytes) &&
           set_arg(env.patch, 5, sizeof(distinct), &distinct) &&
           set_arg(env.patch, 6, sizeof(base_offset), &base_offset) &&
           set_arg(env.patch, 7, sizeof(descriptor_identity_buffer),
                   &descriptor_identity_buffer) &&
           launch(env, env.patch, kWorkgroup);
  };
  const cl_mem primary_base = options.direction == "s2g"
                                  ? destination_buffer : source_buffer;
  if (!patch(0, primary_base, 0) ||
      (options.project == "mixed" &&
       !patch(kDescriptorSlots, mixed_output_buffer, 0))) return 1;
  if (!read(env, descriptor_buffer, descriptor_data) ||
      !read(env, descriptor_identity_buffer, descriptor_identity) ||
      descriptor_identity[0] == 0) return 1;
  if (!env.build(here / "ventus_multi_context.cl", options,
                 descriptor_identity[0])) return 1;
  std::array<uint64_t, 2> descriptor_bank_fingerprints{};
  const uint64_t descriptor_bank_fingerprint = fingerprint64(
      reinterpret_cast<const uint8_t*>(descriptor_data.data()),
      descriptor_data.size() * sizeof(uint32_t));
  descriptor_bank_fingerprints.fill(descriptor_bank_fingerprint);

  const uint32_t path = options.path == "tensor";
  const uint32_t direction = options.direction == "s2g";
  const uint32_t multi_cta = options.level == "multi_cta";
  const uint32_t batched = options.mode == "batched";
  bool configured = true;
  cl_kernel kernel = env.operation;
  const size_t local_contexts = multi_cta ? 1 : options.contexts;
  const size_t payload_local_bytes = local_contexts * options.bytes + 127;
  const size_t barrier_local_bytes = local_contexts * 2 * sizeof(uint32_t);
  const size_t total_local_bytes =
      options.project == "mixed"
          ? 2 * payload_local_bytes + barrier_local_bytes
          : payload_local_bytes +
                (options.direction == "g2s" ? barrier_local_bytes
                                             : 2 * sizeof(uint32_t));
  if (total_local_bytes > kVentusLocalBytes) {
    std::cerr << "RESOURCE_LIMIT,required_local_bytes=" << total_local_bytes
              << ",available_local_bytes=" << kVentusLocalBytes << '\n';
    return 3;
  }
  if (options.project == "same") {
    configured &= set_arg(kernel, 0, sizeof(descriptor_identity[0]),
                          &descriptor_identity[0]);
    const std::array<cl_mem, 5> objects{{
        coordinate_buffer, source_buffer,
        destination_buffer, capture_buffer, result_buffer}};
    for (cl_uint index = 0; index < objects.size(); ++index)
      configured &= set_arg(kernel, index + 1, sizeof(objects[index]), &objects[index]);
    configured &= set_arg(kernel, 6, sizeof(path), &path);
    configured &= set_arg(kernel, 7, sizeof(direction), &direction);
    configured &= set_arg(kernel, 8, sizeof(options.bytes), &options.bytes);
    configured &= set_arg(kernel, 9, sizeof(options.contexts), &options.contexts);
    configured &= set_arg(kernel, 10, sizeof(distinct), &distinct);
    configured &= set_arg(kernel, 11, sizeof(batched), &batched);
  } else {
    const uint32_t order = options.order == "alternating" ? 0u
                         : options.order == "g2s_then_s2g" ? 1u : 2u;
    configured &= set_arg(kernel, 0, sizeof(descriptor_identity[0]),
                          &descriptor_identity[0]);
    const std::array<cl_mem, 4> objects{{
        coordinate_buffer, source_buffer,
        mixed_output_buffer, result_buffer}};
    for (cl_uint index = 0; index < objects.size(); ++index)
      configured &= set_arg(kernel, index + 1, sizeof(objects[index]), &objects[index]);
    configured &= set_arg(kernel, 5, sizeof(options.bytes), &options.bytes);
  }
  const size_t global = static_cast<size_t>(result_entities) * kWorkgroup;
  if (!configured || !launch(env, kernel, global) ||
      !read(env, result_buffer, results) ||
      (options.project == "mixed"
           ? !read(env, mixed_output_buffer, mixed_output)
           : (!read(env, destination_buffer, destination) ||
              !read(env, capture_buffer, capture)))) return 1;
  if (options.project == "mixed") {
    std::copy_n(mixed_output.begin(), 2 * payload_bytes, destination.begin());
    std::copy_n(mixed_output.begin() + 2 * payload_bytes, payload_bytes,
                capture.begin());
  }
  uint64_t errors = 0;
  uint32_t printed_mismatches = 0;
  if (options.direction != "s2g")
    for (size_t index = 0; index < payload_bytes; ++index) {
      if (capture[index] == source[payload_bytes + index]) continue;
      ++errors;
      if (printed_mismatches++ < 16)
        std::cerr << "G2S_MISMATCH,index=" << index
                  << ",expected=" << static_cast<uint32_t>(source[payload_bytes + index])
                  << ",actual=" << static_cast<uint32_t>(capture[index]) << '\n';
    }
  if (options.direction != "g2s") {
    const size_t checked_payload_bytes =
        options.path == "tensor" ? payload_bytes : 2 * payload_bytes;
    for (size_t index = 0; index < checked_payload_bytes; ++index) {
      if (destination[index] == source[index]) continue;
      ++errors;
      if (printed_mismatches++ < 16)
        std::cerr << "S2G_MISMATCH,index=" << index
                  << ",expected=" << static_cast<uint32_t>(source[index])
                  << ",actual=" << static_cast<uint32_t>(destination[index]) << '\n';
    }
  }
  for (uint32_t row = 0; row < 2 * result_entities; ++row) {
    const uint32_t observed_status = results[row * kResultWords + 3];
    if (observed_status == VENTUS_TMA_STATUS_OK) continue;
    ++errors;
    std::cerr << "STATUS_MISMATCH,row=" << row
              << ",status=" << observed_status << '\n';
  }
  if (errors && options.direction != "s2g") {
    std::cerr << "G2S_FIRST32";
    for (size_t index = 0; index < std::min<size_t>(32, capture.size()); ++index)
      std::cerr << ',' << static_cast<uint32_t>(capture[index]);
    std::cerr << '\n';
  }
  if (errors && options.direction == "s2g") {
    std::cerr << "S2G_LOCAL_FIRST32";
    for (size_t index = 0; index < std::min<size_t>(32, capture.size()); ++index)
      std::cerr << ',' << static_cast<uint32_t>(capture[index]);
    std::cerr << '\n';
  }

  std::cout << "device,project,level,path,direction,order,mode,map_mode,bytes,contexts,commands,repeat,pair_role,issue_cycles,total_cycles,issue_span,completion_span,errors,status,descriptor_bank_word_offset,descriptor_bank_spacing_bytes,descriptor_bank_fingerprint64,repeat_descriptor_bank_unique,prior_tma_use,descriptor_l2_warm_method,tma_prefetch_before_issue,descriptor_state,descriptor_va\n";
  for (uint32_t repeat = 0; repeat < 2; ++repeat) {
    uint32_t minimum_begin = std::numeric_limits<uint32_t>::max();
    uint32_t maximum_issue = 0;
    uint32_t maximum_end = 0;
    uint32_t maximum_issue_cycles = 0;
    uint32_t maximum_total_cycles = 0;
    uint32_t aggregate_status = VENTUS_TMA_STATUS_OK;
    for (uint32_t entity = 0; entity < result_entities; ++entity) {
      const uint32_t* row = results.data() +
          (repeat * result_entities + entity) * kResultWords;
      minimum_begin = std::min(minimum_begin, row[0]);
      maximum_issue = std::max(maximum_issue, row[1]);
      maximum_end = std::max(maximum_end, row[2]);
      maximum_issue_cycles = std::max(maximum_issue_cycles, row[1] - row[0]);
      maximum_total_cycles = std::max(maximum_total_cycles, row[2] - row[0]);
      if (row[3] != VENTUS_TMA_STATUS_OK) aggregate_status = row[3];
    }
    std::cout << "Ventus," << (options.project == "same" ? "same_direction" : "mixed_direction")
              << ',' << options.level << ',' << options.path << ','
              << options.direction << ','
              << (options.project == "same" ? "na" : options.order) << ','
              << options.mode << ','
              << (options.path == "bulk" ? "na" : options.map_mode) << ','
              << options.bytes << ',' << options.contexts << ','
              << options.contexts * (options.project == "mixed" ? 2 : 1)
              << ',' << repeat << ',' << (repeat == 0 ? "cold" : "hot")
              << ',' << maximum_issue_cycles << ','
              << maximum_total_cycles << ',' << maximum_issue - minimum_begin
              << ',' << maximum_end - minimum_begin << ',' << errors << ','
              << aggregate_status << ','
              << 0 << ',' << 0
              << ',' << descriptor_bank_fingerprints[repeat]
              << ",0," << repeat << ','
              << (options.path == "tensor" ? "patch_descriptor_write" : "not_applicable")
              << ",0,"
              << (options.path == "tensor"
                      ? (repeat == 0 ? "tmau_cold_bank_first_use"
                                     : "tmau_hot_bank_reuse")
                      : "not_applicable")
              << ",0x" << std::hex << descriptor_identity[0]
              << std::dec << '\n';
  }
  return errors == 0 ? 0 : 1;
}
