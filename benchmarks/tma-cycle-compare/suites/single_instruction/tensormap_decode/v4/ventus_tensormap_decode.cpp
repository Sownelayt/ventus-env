// Host launcher for the Ventus V3.10 TensorMap PMU probe.

#include <CL/cl.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>
#include <vector>

#include "ventus_opencl_test.h"
#include "ventus_tma_v2_spec.h"

namespace {
constexpr size_t kWorkgroup = 32;
constexpr size_t kBytes = 128;
constexpr uint8_t kValue = 0x59;

uint64_t fingerprint64(const void* data, size_t bytes) {
  const auto* input = static_cast<const uint8_t*>(data);
  uint64_t value = 1469598103934665603ull;
  for (size_t index = 0; index < bytes; ++index) {
    value ^= input[index];
    value *= 1099511628211ull;
  }
  return value;
}

struct Options {
  int rank = 2;
  std::string direction = "g2s";
  std::string scenario = "cold_then_hot";
};

struct Environment {
  cl_context context = nullptr;
  cl_device_id device = nullptr;
  cl_command_queue queue = nullptr;
  cl_program program = nullptr;
  cl_kernel patch = nullptr;
  cl_kernel g2s = nullptr;
  cl_kernel s2g = nullptr;

  bool initialize(const std::filesystem::path& source_path) {
    cl_int error = ventus_get_default_device(&context, &device, &queue, nullptr);
    if (error != CL_SUCCESS) return false;
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
    const std::string build_options =
        "-I" + (root / "testcases/_get_case/common").string();
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
    patch = make("patch_descriptor_base");
    g2s = make("bench_decode_g2s");
    s2g = make("bench_decode_s2g");
    return patch && g2s && s2g;
  }

  ~Environment() {
    if (s2g) clReleaseKernel(s2g);
    if (g2s) clReleaseKernel(g2s);
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
cl_mem buffer(Environment& env, Buffers& buffers, cl_mem_flags flags,
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

bool launch(Environment& env, cl_kernel kernel) {
  const size_t global = kWorkgroup;
  const size_t local = kWorkgroup;
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

std::array<uint32_t, 32> descriptor(int rank) {
  std::array<uint32_t, 32> words{};
  words[VENTUS_TMA_V2_WORD_MAGIC] = VENTUS_TMA_V2_MAGIC;
  words[VENTUS_TMA_V2_WORD_CONTROL] =
      VENTUS_TMA_DTYPE_U8 | (static_cast<uint32_t>(rank) << 5) |
      (static_cast<uint32_t>(VENTUS_TMA_INTERLEAVE_NONE) << 8) |
      (static_cast<uint32_t>(VENTUS_TMA_SWIZZLE_NONE) << 10) |
      (static_cast<uint32_t>(VENTUS_TMA_OOB_ZERO) << 18);
  for (int dimension = 0; dimension < rank; ++dimension) {
    words[VENTUS_TMA_V2_WORD_GLOBAL_DIMS + dimension] =
        dimension == 0 ? kBytes : 1;
    words[VENTUS_TMA_V2_WORD_BOX_DIMS + dimension] =
        dimension == 0 ? kBytes : 1;
    words[VENTUS_TMA_V2_WORD_ELEMENT_STRIDES + dimension] = 1;
    if (dimension + 1 < rank) {
      words[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + dimension * 2] = kBytes;
      words[VENTUS_TMA_V2_WORD_GLOBAL_STRIDES + dimension * 2 + 1] = 0;
    }
  }
  return words;
}

Options options(int argc, char** argv) {
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
    if (argument == "--rank") result.rank = std::atoi(next("--rank"));
    else if (argument == "--direction") result.direction = next("--direction");
    else if (argument == "--scenario") result.scenario = next("--scenario");
    else {
      std::cerr << "unknown option: " << argument << '\n';
      std::exit(2);
    }
  }
  if (result.rank < 1 || result.rank > 5 ||
      (result.direction != "g2s" && result.direction != "s2g") ||
      (result.scenario != "cold_then_hot" &&
       result.scenario != "explicit_prefetch")) {
    std::cerr << "invalid rank/direction/scenario\n";
    std::exit(2);
  }
  return result;
}
}  // namespace

int main(int argc, char** argv) {
  const Options selected = options(argc, argv);
  const auto here = std::filesystem::path(argv[0]).parent_path();
  Environment env;
  if (!env.initialize(here / "ventus_tensormap_decode.cl")) {
    std::cerr << "OpenCL initialization failed\n";
    return 1;
  }
  std::vector<int32_t> coordinates(32, 0);
  Buffers buffers;
  cl_mem coordinate_buffer = buffer(env, buffers, CL_MEM_READ_ONLY,
                                    coordinates);
  if (!coordinate_buffer) {
    std::cerr << "buffer allocation failed\n";
    return 1;
  }
  cl_kernel kernel = selected.direction == "g2s" ? env.g2s : env.s2g;
  const uint32_t prefetch = selected.scenario == "explicit_prefetch";
  const uint32_t prefetch_lead = prefetch ? 32u : 0u;
  const uint32_t iterations = 2;
  std::cout << "device,direction,rank,scenario,prefetch_lead_cycles,repeat,pair_role,cycles,errors,status,descriptor_state,descriptor_va,descriptor_fingerprint64,prior_tma_use,descriptor_l2_warm_method,tma_prefetch_before_issue\n";
  bool all_ok = true;
    auto descriptor_words = descriptor(selected.rank);
    std::vector<uint32_t> descriptor_data(descriptor_words.begin(),
                                          descriptor_words.end());
    std::vector<uint8_t> source(kBytes, kValue);
    std::vector<uint8_t> destination(kBytes, 0);
    std::vector<uint8_t> capture(kBytes, 0);
    std::vector<uint32_t> cycles(2, 0);
    std::vector<uint32_t> status(1, ~0u);
    std::vector<uint32_t> identity(2, 0);
    cl_mem descriptor_buffer = buffer(env, buffers, CL_MEM_READ_WRITE,
                                      descriptor_data);
    cl_mem source_buffer = buffer(env, buffers, CL_MEM_READ_WRITE, source);
    cl_mem destination_buffer = buffer(env, buffers, CL_MEM_READ_WRITE,
                                       destination);
    cl_mem capture_buffer = buffer(env, buffers, CL_MEM_READ_WRITE, capture);
    cl_mem cycle_buffer = buffer(env, buffers, CL_MEM_READ_WRITE, cycles);
    cl_mem status_buffer = buffer(env, buffers, CL_MEM_READ_WRITE, status);
    cl_mem identity_buffer = buffer(env, buffers, CL_MEM_READ_WRITE, identity);
    if (!descriptor_buffer || !source_buffer || !destination_buffer ||
        !capture_buffer || !cycle_buffer || !status_buffer ||
        !identity_buffer) {
      std::cerr << "sample buffer allocation failed\n";
      return 1;
    }
    cl_mem descriptor_base = selected.direction == "g2s" ? source_buffer
                                                           : destination_buffer;
    if (!set_arg(env.patch, 0, sizeof(descriptor_buffer), &descriptor_buffer) ||
        !set_arg(env.patch, 1, sizeof(descriptor_base), &descriptor_base) ||
        !set_arg(env.patch, 2, sizeof(identity_buffer), &identity_buffer) ||
        !launch(env, env.patch)) {
      std::cerr << "descriptor patch failed\n";
      return 1;
    }
    const std::array<cl_mem, 7> objects{{
        descriptor_buffer, coordinate_buffer, source_buffer,
        destination_buffer, capture_buffer, cycle_buffer, status_buffer}};
    bool configured = true;
    for (cl_uint index = 0; index < objects.size(); ++index)
      configured &= set_arg(kernel, index, sizeof(objects[index]), &objects[index]);
    configured &= set_arg(kernel, 7, sizeof(prefetch), &prefetch);
    configured &= set_arg(kernel, 8, sizeof(prefetch_lead), &prefetch_lead);
    configured &= set_arg(kernel, 9, sizeof(iterations), &iterations);
    if (!configured || !launch(env, kernel) ||
        !read(env, cycle_buffer, cycles) ||
        !read(env, status_buffer, status) ||
        !read(env, identity_buffer, identity) ||
        !read(env, descriptor_buffer, descriptor_data)) {
      std::cerr << "measurement failed\n";
      return 1;
    }
    uint64_t errors = 0;
    if (selected.direction == "g2s") {
      if (!read(env, capture_buffer, capture)) return 1;
      for (uint8_t value : capture) errors += value != kValue;
    } else {
      if (!read(env, destination_buffer, destination)) return 1;
      for (uint8_t value : destination) errors += value != kValue;
    }
    const uint64_t fingerprint = fingerprint64(
        descriptor_data.data(), descriptor_data.size() * sizeof(uint32_t));
    const bool ok = errors == 0 && status[0] == VENTUS_TMA_STATUS_OK;
    all_ok &= ok;
    for (int repeat = 0; repeat < 2; ++repeat) {
      const bool cold = repeat == 0 && !prefetch;
      std::cout << "Ventus," << selected.direction << ',' << selected.rank << ','
                << selected.scenario << ',' << prefetch_lead << ',' << repeat
                << ',' << (repeat == 0 ? (prefetch ? "prefetched" : "cold") : "hot")
                << ',' << cycles[repeat] << ',' << errors << ',' << status[0]
                << ',' << (cold ? "tmau_cold_first_use_fresh_allocation"
                                 : repeat == 0 ? "prefetched_first_use"
                                               : "tmau_hot_reuse")
                << ",0x" << std::hex << identity[0] << ",0x" << fingerprint
                << std::dec << ',' << repeat
                << ",patch_descriptor_write,"
                << (prefetch && repeat == 0) << '\n';
    }
  return all_ok ? 0 : 1;
}
