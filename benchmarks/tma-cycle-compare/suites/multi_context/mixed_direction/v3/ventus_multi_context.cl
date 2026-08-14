/* Ventus V3.10 same- and mixed-direction command scaling probe.
 *
 * One workgroup models an issuing warp with N commands.  N workgroups model
 * independent CTA issuers without a grid barrier; absolute cycle timestamps
 * let the host compute the actual launch/completion span safely.
 */

#include "ventus_tma_v2_opencl.h"

#define WG_SIZE 32u
#define RESULT_WORDS 4u
#define MAX_BYTES_PER_COMMAND 4096u
#define SAME_INTRA_PAYLOAD_BYTES (8u * MAX_BYTES_PER_COMMAND + 127u)
#define MIXED_INTRA_PAYLOAD_BYTES (4u * MAX_BYTES_PER_COMMAND + 127u)
#define MULTI_CTA_PAYLOAD_BYTES (MAX_BYTES_PER_COMMAND + 127u)
#define MAX_BARRIER_WORDS (2u * 32u)
#define DESCRIPTOR_SET_WORDS (32u * 32u)

#ifdef VENTUS_MULTI_SPECIALIZED
#define VENTUS_CASE_PATH VENTUS_MULTI_PATH
#define VENTUS_CASE_DIRECTION VENTUS_MULTI_DIRECTION
#define VENTUS_CASE_DISTINCT VENTUS_MULTI_DISTINCT
#define VENTUS_CASE_BATCHED VENTUS_MULTI_BATCHED
#define VENTUS_CASE_BYTES VENTUS_MULTI_BYTES
#define VENTUS_CASE_CONTEXTS VENTUS_MULTI_CONTEXTS
#define VENTUS_CASE_PAIRS(multi_value, context_value) VENTUS_MULTI_CONTEXTS
#else
#define VENTUS_CASE_PATH path
#define VENTUS_CASE_DIRECTION direction
#define VENTUS_CASE_DISTINCT distinct_map
#define VENTUS_CASE_BATCHED batched
#define VENTUS_CASE_BYTES bytes
#define VENTUS_CASE_CONTEXTS contexts
#define VENTUS_CASE_PAIRS(multi_value, context_value) \
  ((multi_value) ? get_num_groups(0) : (context_value))
#endif

#ifdef VENTUS_DESCRIPTOR_BASE
#define VENTUS_CASE_DESCRIPTOR_BASE ((uint)VENTUS_DESCRIPTOR_BASE)
#else
#define VENTUS_CASE_DESCRIPTOR_BASE descriptor_base
#endif

static uint read_cycle_lo(void) {
  uint value;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(value) :: "memory");
  return value;
}

static uint read_group_x_scalar(void) {
  uint value;
  __asm__ volatile("csrr %0, 0x808\n\t" : "=r"(value) :: "memory");
  return value;
}

static __local uchar *align_local_128(__local uchar *base) {
  return (__local uchar *)(((uint)base + 127u) & ~127u);
}

#define LOAD_COORD_INDEX(coords_arg, index_arg) do {                         \
  uint _coordinates = (uint)(coords_arg);                                    \
  uint _coordinate_index = (uint)(index_arg);                                \
  __asm__ volatile(                                                          \
    "slli t0, %[index], 7\n\t"                                             \
    "add t0, %[coords], t0\n\t"                                           \
    "vid.v v12\n\t"                                                       \
    "vsll.vi v12, v12, 2\n\t"                                             \
    "vadd.vx v12, v12, t0\n\t"                                           \
    "vlw12.v v12, 0(v12)\n\t"                                            \
    : : [coords] "r"(_coordinates), [index] "r"(_coordinate_index)        \
    : "t0", "memory");                                                    \
} while (0)

#define LOAD_COORD_GROUP(coords_arg) do {                                    \
  uint _coordinates = (uint)(coords_arg);                                    \
  __asm__ volatile(                                                          \
    "csrr t0, 0x808\n\t"                                                 \
    "slli t0, t0, 7\n\t"                                                \
    "add t0, %[coords], t0\n\t"                                          \
    "vid.v v12\n\t"                                                     \
    "vsll.vi v12, v12, 2\n\t"                                           \
    "vadd.vx v12, v12, t0\n\t"                                         \
    "vlw12.v v12, 0(v12)\n\t"                                          \
    : : [coords] "r"(_coordinates) : "t0", "memory");                  \
} while (0)

#define MULTI_BULK_G2S(shared_arg, global_base_arg, bytes_arg) do {          \
  uint _shared = (uint)(shared_arg);                                         \
  uint _base = (uint)(global_base_arg);                                      \
  uint _bytes = (uint)(bytes_arg);                                           \
  __asm__ volatile(                                                          \
    "csrr t0, 0x808\n\t"                                                 \
    "mul t0, t0, %[bytes]\n\t"                                           \
    "add t0, t0, %[base]\n\t"                                            \
    ".insn r 0x42, 1, 0, %[shared], t0, %[bytes]\n\t"                    \
    : : [shared] "r"(_shared), [base] "r"(_base),                       \
        [bytes] "r"(_bytes) : "t0", "memory");                         \
} while (0)

#define MULTI_BULK_S2G(global_base_arg, shared_arg, bytes_arg) do {          \
  uint _shared = (uint)(shared_arg);                                         \
  uint _base = (uint)(global_base_arg);                                      \
  uint _bytes = (uint)(bytes_arg);                                           \
  __asm__ volatile(                                                          \
    "csrr t0, 0x808\n\t"                                                 \
    "mul t0, t0, %[bytes]\n\t"                                           \
    "add t0, t0, %[base]\n\t"                                            \
    ".insn r 0x42, 3, 0, t0, %[shared], %[bytes]\n\t"                    \
    : : [shared] "r"(_shared), [base] "r"(_base),                       \
        [bytes] "r"(_bytes) : "t0", "memory");                         \
} while (0)

#define MULTI_TENSOR_G2S(shared_arg, descriptor_base_arg) do {             \
  uint _shared = (uint)(shared_arg);                                         \
  uint _descriptor = (uint)(descriptor_base_arg);                            \
  __asm__ volatile(                                                          \
    "csrr t0, 0x808\n\t"                                                 \
    "slli t0, t0, 7\n\t"                                                \
    "add t0, %[descriptor], t0\n\t"                                      \
    ".insn r 0x42, 2, 0, %[shared], t0, x12\n\t"                        \
    : : [shared] "r"(_shared), [descriptor] "r"(_descriptor)            \
    : "t0", "memory");                                                   \
} while (0)

#define MULTI_TENSOR_S2G(shared_arg, descriptor_base_arg) do {             \
  uint _shared = (uint)(shared_arg);                                         \
  uint _descriptor = (uint)(descriptor_base_arg);                            \
  __asm__ volatile(                                                          \
    "csrr t0, 0x808\n\t"                                                 \
    "slli t0, t0, 7\n\t"                                                \
    "add t0, %[descriptor], t0\n\t"                                      \
    ".insn r 0x42, 4, 0, %[shared], t0, x12\n\t"                        \
    : : [shared] "r"(_shared), [descriptor] "r"(_descriptor)            \
    : "t0", "memory");                                                   \
} while (0)

#define MULTI_TENSOR_SAME_G2S(shared_arg, descriptor_arg) do {              \
  uint _shared = (uint)(shared_arg);                                         \
  uint _descriptor = (uint)(descriptor_arg);                                \
  __asm__ volatile(                                                          \
    ".insn r 0x42, 2, 0, %[shared], %[descriptor], x12\n\t"             \
    : : [shared] "r"(_shared), [descriptor] "r"(_descriptor)            \
    : "memory");                                                           \
} while (0)

#define MULTI_TENSOR_SAME_S2G(shared_arg, descriptor_arg) do {              \
  uint _shared = (uint)(shared_arg);                                         \
  uint _descriptor = (uint)(descriptor_arg);                                \
  __asm__ volatile(                                                          \
    ".insn r 0x42, 4, 0, %[shared], %[descriptor], x12\n\t"             \
    : : [shared] "r"(_shared), [descriptor] "r"(_descriptor)            \
    : "memory");                                                           \
} while (0)

kernel void patch_descriptor_bases(__global uint *descriptors,
                                   uint descriptor_start,
                                   __global uchar *base, uint count,
                                   uint bytes, uint distinct,
                                   uint base_offset,
                                   __global uint *identity) {
  uint index = get_global_id(0);
  if (index == 0u) identity[0] = (uint)descriptors;
  if (index < count) {
    descriptors[(descriptor_start + index) * 32u + 2u] =
        (uint)(base + base_offset + (distinct ? index * bytes : 0u));
    descriptors[(descriptor_start + index) * 32u + 3u] = 0u;
  }
}

static void record_result(__global uint *results, uint result_index,
                          uint begin, uint issue_end, uint end, uint status) {
  __global uint *row = results + result_index * RESULT_WORDS;
  row[0] = begin;
  row[1] = issue_end;
  row[2] = end;
  row[3] = status;
}

static inline __attribute__((always_inline)) void same_direction_impl(
    uint descriptor_base, __global const int *coordinates,
    __global const uchar *source, __global uchar *destination,
    __global uchar *capture, __global uint *results,
    __local uchar *storage_raw, __local uint *barriers,
    uint path, uint direction, uint bytes, uint contexts,
    uint distinct_map, uint multi_cta, uint batched) {
  __local uchar *storage = align_local_128(storage_raw);
  uint lid = get_local_id(0);
  uint first = multi_cta ? get_group_id(0) : 0u;
  uint issue_first = multi_cta ? read_group_x_scalar() : 0u;
  uint count = multi_cta ? 1u : contexts;
  __local uint *storage_words = (__local uint *)storage;
  __global const uint *source_words = (__global const uint *)source;
  for (uint word = lid; word < count * bytes / 4u;
       word += get_local_size(0))
    storage_words[word] = direction == 0u ? 0u :
        source_words[first * bytes / 4u + word];
  barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll 2
  for (uint repeat = 0u; repeat < 2u; ++repeat) {
    /* The first batch is the descriptor's first TMAU use.  The second batch
     * immediately reuses the exact same descriptor bank in this kernel. */
    uint repeat_descriptors = descriptor_base;
    if (lid == 0u) {
      VENTUS_TMA_STATUS_CLEAR();
      if (direction == 0u) {
#pragma unroll 32
        for (uint command = 0u; command < 32u; ++command)
          if (command < count)
            VENTUS_TMA_MBARRIER_INIT(barriers + command * 2u, 1u);
      }
      else
        VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint begin = 0u;
    if (lid == 0u) begin = read_cycle_lo();
    #pragma unroll 32
    for (uint command = 0u; command < 32u; ++command) {
      if (command >= count) continue;
      uint index = issue_first + command;
      if (path != 0u) {
        if (multi_cta) LOAD_COORD_GROUP(coordinates);
        else LOAD_COORD_INDEX(coordinates, index);
      }
      if (lid == 0u) {
        __local uchar *tile = storage + command * bytes;
        if (direction == 0u) {
          VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(
              barriers + command * 2u, bytes);
          if (path == 0u) {
            if (multi_cta)
              MULTI_BULK_G2S(tile,
                             source + repeat * contexts * bytes, bytes);
            else
              VENTUS_TMA_BULK_G2S(
                  tile, source + (repeat * contexts + index) * bytes, bytes);
          }
          else
            if (multi_cta && distinct_map)
              MULTI_TENSOR_G2S(tile, repeat_descriptors);
            else if (distinct_map)
              MULTI_TENSOR_G2S(tile, repeat_descriptors + index * 128u);
            else if (multi_cta)
              MULTI_TENSOR_SAME_G2S(tile, repeat_descriptors);
            else
              MULTI_TENSOR_G2S(tile, repeat_descriptors);
        } else {
          if (path == 0u) {
            if (multi_cta)
              MULTI_BULK_S2G(
                  destination + repeat * contexts * bytes, tile, bytes);
            else
              VENTUS_TMA_BULK_S2G(
                  destination + (repeat * contexts + index) * bytes,
                  tile, bytes);
          }
          else
            if (multi_cta && distinct_map)
              MULTI_TENSOR_S2G(tile, repeat_descriptors);
            else if (distinct_map)
              MULTI_TENSOR_S2G(tile, repeat_descriptors + index * 128u);
            else if (multi_cta)
              MULTI_TENSOR_SAME_S2G(tile, repeat_descriptors);
            else
              MULTI_TENSOR_S2G(tile, repeat_descriptors);
        }
      }
      if (!batched && lid == 0u) {
        if (direction == 0u)
          VENTUS_TMA_MBARRIER_WAIT(barriers + command * 2u, 0u);
        else {
          VENTUS_TMA_S2G_COMMIT_GROUP();
          VENTUS_TMA_S2G_WAIT_GROUP0();
        }
      }
    }
    uint issue_end = 0u;
    if (lid == 0u) {
      if (batched && direction != 0u) VENTUS_TMA_S2G_COMMIT_GROUP();
      issue_end = read_cycle_lo();
      if (batched && direction == 0u) {
#pragma unroll 32
        for (uint command = 0u; command < 32u; ++command)
          if (command < count)
            VENTUS_TMA_MBARRIER_WAIT(barriers + command * 2u, 0u);
      }
      else if (batched)
        VENTUS_TMA_S2G_WAIT_GROUP0();
      uint status = ~0u;
      VENTUS_TMA_STATUS_READ(status);
      uint end = read_cycle_lo();
      uint result_index = repeat * (multi_cta ? contexts : 1u) +
                          (multi_cta ? get_group_id(0) : 0u);
      record_result(results, result_index, begin, issue_end, end, status);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  __global uint *capture_words = (__global uint *)capture;
  for (uint word = lid; word < count * bytes / 4u;
       word += get_local_size(0))
    capture_words[first * bytes / 4u + word] = storage_words[word];
}

#define ISSUE_MIXED_COMMAND(command_value, direction_value) do {             \
  const uint issue_command = (command_value);                               \
  const uint issue_direction = (direction_value);                           \
  const uint index = issue_first + issue_command;                           \
  if (path != 0u && (issue_direction == 0u || order != 0u)) {               \
    if (multi_cta) LOAD_COORD_GROUP(coordinates);                           \
    else LOAD_COORD_INDEX(coordinates, index);                              \
  }                                                                         \
  if (lid == 0u) {                                                          \
    if (issue_direction == 0u) {                                            \
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(                                 \
          barriers + issue_command * 2u, bytes);                            \
      if (path == 0u) {                                                     \
        if (multi_cta)                                                      \
          MULTI_BULK_G2S(load_storage + issue_command * bytes,              \
                         source + repeat * pairs * bytes, bytes);           \
        else                                                                \
          VENTUS_TMA_BULK_G2S(load_storage + issue_command * bytes,         \
                              source + (repeat * pairs + index) * bytes,     \
                              bytes);                                       \
      } else if (multi_cta && distinct_map)                                 \
        MULTI_TENSOR_G2S(load_storage + issue_command * bytes,              \
                         repeat_load_descriptors);                          \
      else if (distinct_map)                                                \
        MULTI_TENSOR_G2S(load_storage + issue_command * bytes,              \
                         repeat_load_descriptors + index * 128u);           \
      else if (multi_cta)                                                   \
        MULTI_TENSOR_SAME_G2S(load_storage + issue_command * bytes,         \
                              repeat_load_descriptors);                     \
      else                                                                  \
        MULTI_TENSOR_G2S(load_storage + issue_command * bytes,              \
                         repeat_load_descriptors);                          \
    } else {                                                               \
      if (path == 0u) {                                                     \
        if (multi_cta)                                                      \
          MULTI_BULK_S2G(destination + repeat * pairs * bytes,              \
                         store_storage + issue_command * bytes,              \
                         bytes);                                             \
        else                                                                \
          VENTUS_TMA_BULK_S2G(                                             \
                              destination +                                \
                                  (repeat * pairs + index) * bytes,         \
                              store_storage + issue_command * bytes, bytes); \
      } else if (multi_cta && distinct_map)                                 \
        MULTI_TENSOR_S2G(store_storage + issue_command * bytes,             \
                         repeat_store_descriptors);                         \
      else if (distinct_map)                                                \
        MULTI_TENSOR_S2G(store_storage + issue_command * bytes,             \
                         repeat_store_descriptors + index * 128u);          \
      else if (multi_cta)                                                   \
        MULTI_TENSOR_SAME_S2G(store_storage + issue_command * bytes,        \
                              repeat_store_descriptors);                    \
      else                                                                  \
        MULTI_TENSOR_S2G(store_storage + issue_command * bytes,             \
                         repeat_store_descriptors);                         \
    }                                                                       \
  }                                                                         \
  if (!batched && lid == 0u) {                                              \
    if (issue_direction == 0u)                                              \
      VENTUS_TMA_MBARRIER_WAIT(barriers + issue_command * 2u, 0u);         \
    else {                                                                  \
      VENTUS_TMA_S2G_COMMIT_GROUP();                                        \
      VENTUS_TMA_S2G_WAIT_GROUP0();                                         \
    }                                                                       \
  }                                                                         \
} while (0)

static inline __attribute__((always_inline)) void mixed_direction_impl(
    uint load_descriptor_base, uint store_descriptor_base,
    __global const int *coordinates, __global const uchar *source,
    __global uchar *destination, __global uchar *capture,
    __global uint *results, __local uchar *load_raw,
    __local uchar *store_raw, __local uint *barriers,
    uint path, uint bytes, uint pairs,
    uint distinct_map, uint order, uint multi_cta, uint batched) {
  __local uchar *load_storage = align_local_128(load_raw);
  __local uchar *store_storage = align_local_128(store_raw);
  uint lid = get_local_id(0);
  uint first = multi_cta ? get_group_id(0) : 0u;
  uint issue_first = multi_cta ? read_group_x_scalar() : 0u;
  uint count = multi_cta ? 1u : pairs;
  __local uint *load_words = (__local uint *)load_storage;
  __local uint *store_words = (__local uint *)store_storage;
  __global const uint *source_words = (__global const uint *)source;
  for (uint word = lid; word < count * bytes / 4u;
       word += get_local_size(0)) {
    load_words[word] = 0u;
    store_words[word] = source_words[first * bytes / 4u + word];
  }
  barrier(CLK_LOCAL_MEM_FENCE);

#pragma unroll 2
  for (uint repeat = 0u; repeat < 2u; ++repeat) {
    uint repeat_load_descriptors = load_descriptor_base;
    uint repeat_store_descriptors = store_descriptor_base;
    if (lid == 0u) {
      VENTUS_TMA_STATUS_CLEAR();
#pragma unroll 16
      for (uint command = 0u; command < 16u; ++command)
        if (command < count)
          VENTUS_TMA_MBARRIER_INIT(barriers + command * 2u, 1u);
      VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    uint begin = 0u;
    if (lid == 0u) begin = read_cycle_lo();
    if (order == 0u) {
#pragma unroll 16
      for (uint command = 0u; command < 16u; ++command)
        if (command < count) {
          ISSUE_MIXED_COMMAND(command, 0u);
          ISSUE_MIXED_COMMAND(command, 1u);
        }
    } else if (order == 1u) {
#pragma unroll 16
      for (uint command = 0u; command < 16u; ++command)
        if (command < count) ISSUE_MIXED_COMMAND(command, 0u);
#pragma unroll 16
      for (uint command = 0u; command < 16u; ++command)
        if (command < count) ISSUE_MIXED_COMMAND(command, 1u);
    } else {
#pragma unroll 16
      for (uint command = 0u; command < 16u; ++command)
        if (command < count) ISSUE_MIXED_COMMAND(command, 1u);
#pragma unroll 16
      for (uint command = 0u; command < 16u; ++command)
        if (command < count) ISSUE_MIXED_COMMAND(command, 0u);
    }
    uint issue_end = 0u;
    if (lid == 0u) {
      if (batched) VENTUS_TMA_S2G_COMMIT_GROUP();
      issue_end = read_cycle_lo();
      if (batched) {
        VENTUS_TMA_S2G_WAIT_GROUP0();
#pragma unroll 16
        for (uint command = 0u; command < 16u; ++command)
          if (command < count)
            VENTUS_TMA_MBARRIER_WAIT(barriers + command * 2u, 0u);
      }
      uint status = ~0u;
      VENTUS_TMA_STATUS_READ(status);
      uint end = read_cycle_lo();
      uint result_index = repeat * (multi_cta ? pairs : 1u) +
                          (multi_cta ? get_group_id(0) : 0u);
      record_result(results, result_index, begin, issue_end, end, status);
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  __global uint *capture_words = (__global uint *)capture;
  for (uint word = lid; word < count * bytes / 4u;
       word += get_local_size(0))
    capture_words[first * bytes / 4u + word] = load_words[word];
}

#undef ISSUE_MIXED_COMMAND

/*
 * Dynamic __local kernel arguments become vector-valued pointers in the
 * current Ventus compiler.  Moving those pointers back to a scalar issuing
 * lane selects an unsupported fsgnj.d encoding.  Separate static LDS layouts
 * also preserve the point of the multi-CTA experiment: the multi-CTA kernels
 * reserve only one command's storage per CTA instead of the intra-CTA maximum.
 */
#define SAME_WRAPPER(name, payload_bytes, multi_value, context_value)        \
kernel void name(                                                            \
    uint descriptor_base, __global const int *coordinates,                   \
    __global const uchar *source, __global uchar *destination,               \
    __global uchar *capture, __global uint *results,                         \
    uint path, uint direction, uint bytes, uint contexts,                    \
    uint distinct_map, uint batched) {                                       \
  __local uchar storage_raw[payload_bytes];                                  \
  __local uint barriers[MAX_BARRIER_WORDS];                                  \
  same_direction_impl(VENTUS_CASE_DESCRIPTOR_BASE, coordinates, source,     \
                      destination,                                          \
                      capture, results, storage_raw, barriers,               \
                      VENTUS_CASE_PATH, VENTUS_CASE_DIRECTION,               \
                      VENTUS_CASE_BYTES, VENTUS_CASE_CONTEXTS,               \
                      VENTUS_CASE_DISTINCT,                                  \
                      multi_value, VENTUS_CASE_BATCHED);                     \
}

SAME_WRAPPER(same_direction_intra_1_kernel, SAME_INTRA_PAYLOAD_BYTES, 0u, 1u)
SAME_WRAPPER(same_direction_intra_2_kernel, SAME_INTRA_PAYLOAD_BYTES, 0u, 2u)
SAME_WRAPPER(same_direction_intra_4_kernel, SAME_INTRA_PAYLOAD_BYTES, 0u, 4u)
SAME_WRAPPER(same_direction_intra_8_kernel, SAME_INTRA_PAYLOAD_BYTES, 0u, 8u)
SAME_WRAPPER(same_direction_intra_16_kernel, SAME_INTRA_PAYLOAD_BYTES, 0u, 16u)
SAME_WRAPPER(same_direction_intra_32_kernel, SAME_INTRA_PAYLOAD_BYTES, 0u, 32u)
SAME_WRAPPER(same_direction_multi_kernel, MULTI_CTA_PAYLOAD_BYTES, 1u, 1u)

#define MIXED_WRAPPER(name, payload_bytes, path_value, distinct_value,       \
                      order_value, batched_value, multi_value,              \
                      context_value)                                        \
kernel void name(                                                            \
    uint descriptor_base, __global const int *coordinates,                   \
    __global const uchar *source, __global uchar *output,                    \
    __global uint *results, uint bytes) {                                    \
  __local uchar shared_raw[2u * ((payload_bytes) - 127u) + 127u];            \
  __local uchar *load_raw = align_local_128(shared_raw);                     \
  __local uchar *store_raw = load_raw + ((payload_bytes) - 127u);            \
  __local uint barriers[MAX_BARRIER_WORDS];                                  \
  uint _pairs = VENTUS_CASE_PAIRS(multi_value, context_value);              \
  mixed_direction_impl(VENTUS_CASE_DESCRIPTOR_BASE,                         \
                       VENTUS_CASE_DESCRIPTOR_BASE +                         \
                           DESCRIPTOR_SET_WORDS * 4u,                        \
                       coordinates, source, output,                          \
                       output + 2u * _pairs * bytes,                         \
                       results, load_raw,                                    \
                       store_raw, barriers, (path_value), VENTUS_CASE_BYTES, \
                       _pairs,                                               \
                       (distinct_value), (order_value), multi_value,         \
                       (batched_value));                                     \
}

#define MIXED_VARIANTS(stem, payload, path, distinct, multi, contexts)       \
  MIXED_WRAPPER(stem##_alternating_serial_kernel, payload, path, distinct, 0u, 0u, multi, contexts) \
  MIXED_WRAPPER(stem##_alternating_batched_kernel, payload, path, distinct, 0u, 1u, multi, contexts) \
  MIXED_WRAPPER(stem##_g2s_then_s2g_serial_kernel, payload, path, distinct, 1u, 0u, multi, contexts) \
  MIXED_WRAPPER(stem##_g2s_then_s2g_batched_kernel, payload, path, distinct, 1u, 1u, multi, contexts) \
  MIXED_WRAPPER(stem##_s2g_then_g2s_serial_kernel, payload, path, distinct, 2u, 0u, multi, contexts) \
  MIXED_WRAPPER(stem##_s2g_then_g2s_batched_kernel, payload, path, distinct, 2u, 1u, multi, contexts)

MIXED_VARIANTS(mixed_direction_bulk_intra_1, MIXED_INTRA_PAYLOAD_BYTES, 0u, 0u, 0u, 1u)
MIXED_VARIANTS(mixed_direction_bulk_intra_2, MIXED_INTRA_PAYLOAD_BYTES, 0u, 0u, 0u, 2u)
MIXED_VARIANTS(mixed_direction_bulk_intra_4, MIXED_INTRA_PAYLOAD_BYTES, 0u, 0u, 0u, 4u)
MIXED_VARIANTS(mixed_direction_bulk_intra_8, MIXED_INTRA_PAYLOAD_BYTES, 0u, 0u, 0u, 8u)
MIXED_VARIANTS(mixed_direction_bulk_intra_16, MIXED_INTRA_PAYLOAD_BYTES, 0u, 0u, 0u, 16u)
MIXED_VARIANTS(mixed_direction_bulk_multi, MULTI_CTA_PAYLOAD_BYTES, 0u, 0u, 1u, 1u)
MIXED_VARIANTS(mixed_direction_tensor_same_map_intra_1, MIXED_INTRA_PAYLOAD_BYTES, 1u, 0u, 0u, 1u)
MIXED_VARIANTS(mixed_direction_tensor_same_map_intra_2, MIXED_INTRA_PAYLOAD_BYTES, 1u, 0u, 0u, 2u)
MIXED_VARIANTS(mixed_direction_tensor_same_map_intra_4, MIXED_INTRA_PAYLOAD_BYTES, 1u, 0u, 0u, 4u)
MIXED_VARIANTS(mixed_direction_tensor_same_map_intra_8, MIXED_INTRA_PAYLOAD_BYTES, 1u, 0u, 0u, 8u)
MIXED_VARIANTS(mixed_direction_tensor_same_map_intra_16, MIXED_INTRA_PAYLOAD_BYTES, 1u, 0u, 0u, 16u)
MIXED_VARIANTS(mixed_direction_tensor_same_map_multi, MULTI_CTA_PAYLOAD_BYTES, 1u, 0u, 1u, 1u)
MIXED_VARIANTS(mixed_direction_tensor_distinct_map_intra_1, MIXED_INTRA_PAYLOAD_BYTES, 1u, 1u, 0u, 1u)
MIXED_VARIANTS(mixed_direction_tensor_distinct_map_intra_2, MIXED_INTRA_PAYLOAD_BYTES, 1u, 1u, 0u, 2u)
MIXED_VARIANTS(mixed_direction_tensor_distinct_map_intra_4, MIXED_INTRA_PAYLOAD_BYTES, 1u, 1u, 0u, 4u)
MIXED_VARIANTS(mixed_direction_tensor_distinct_map_intra_8, MIXED_INTRA_PAYLOAD_BYTES, 1u, 1u, 0u, 8u)
MIXED_VARIANTS(mixed_direction_tensor_distinct_map_intra_16, MIXED_INTRA_PAYLOAD_BYTES, 1u, 1u, 0u, 16u)
MIXED_VARIANTS(mixed_direction_tensor_distinct_map_multi, MULTI_CTA_PAYLOAD_BYTES, 1u, 1u, 1u, 1u)

#undef MIXED_VARIANTS

#undef MIXED_WRAPPER
#undef SAME_WRAPPER
#undef VENTUS_CASE_DESCRIPTOR_BASE
#undef VENTUS_CASE_BATCHED
#undef VENTUS_CASE_DISTINCT
#undef VENTUS_CASE_DIRECTION
#undef VENTUS_CASE_PATH
#undef VENTUS_CASE_CONTEXTS
#undef VENTUS_CASE_BYTES
#undef VENTUS_CASE_PAIRS

#undef LOAD_COORD_INDEX
#undef MULTI_TENSOR
#undef MULTI_TENSOR_SAME_S2G
#undef MULTI_TENSOR_SAME_G2S
#undef MULTI_BULK_S2G
#undef MULTI_BULK_G2S
#undef LOAD_COORD_GROUP
#undef DESCRIPTOR_SET_WORDS
