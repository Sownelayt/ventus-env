/*
 * Ventus half of the tma-refer non-Reduce feature parity matrix.
 *
 * Every timing window matches tma_ventus_parity_bench_v2.cu:
 *   g2s       = expect-tx + issue + mbarrier completion
 *   s2g       = issue + commit + wait-group 0
 *   roundtrip = complete g2s + proxy fence + complete s2g
 *
 * The host launches a separate warmup grid and measurement grid.  Each grid
 * loops in-kernel, matching the CUDA reference's descriptor/cache lifetime.
 */

#include "ventus_tma_v2_opencl.h"

#define WG_SIZE 32u
#define MAX_BYTES 32768u

static uint
read_cycle_lo(void)
{
  uint value;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(value) :: "memory");
  return value;
}

static __local uchar *
align_local_1024(__local uchar *base)
{
  return (__local uchar *)(((uint)base + 1023u) & ~1023u);
}

static __local uint *
align_local_8(__local uint *base)
{
  return (__local uint *)(((uint)base + 7u) & ~7u);
}

kernel void
patch_descriptor_base(__global uint *descriptor, __global uchar *base)
{
  if (get_global_id(0) == 0u) {
    descriptor[2] = (uint)base;
    descriptor[3] = 0u;
  }
}

kernel void
patch_descriptor_bases(__global uint *descriptors, __global uchar *base,
                       uint count)
{
  uint index = get_global_id(0);
  for (; index < count; index += WG_SIZE) {
    descriptors[index * 32u + 2u] = (uint)base;
    descriptors[index * 32u + 3u] = 0u;
  }
}

#define FEATURE_ARGUMENTS                                                    \
  __global uint *source_descriptor, __global uint *destination_descriptor,  \
      __global const int *coordinates, __global const uchar *source,         \
      __global uchar *destination, __global uchar *capture,                  \
      __global uint *cycles, __global uint *status, uint bytes,              \
      uint iterations

kernel void
bench_feature_g2s(FEATURE_ARGUMENTS)
{
  __local uchar shared_raw[MAX_BYTES + 1024u];
  __local uint barrier_raw[4];
  __local uchar *shared = align_local_1024(shared_raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  (void)destination_descriptor;
  (void)destination;

  VENTUS_TMA_LOAD_COORDS_V12(coordinates);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      uint begin = read_cycle_lo();
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
      VENTUS_TMA_TENSOR_G2S(shared, source_descriptor);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      uint end = read_cycle_lo();
      if (cycles) cycles[iteration] = end - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
  if (capture) {
    __local uint *shared_words = (__local uint *)shared;
    __global uint *capture_words = (__global uint *)capture;
    for (uint index = lid; index < bytes / 4u; index += WG_SIZE)
      capture_words[index] = shared_words[index];
  }
}

kernel void
bench_feature_s2g(FEATURE_ARGUMENTS)
{
  __local uchar shared_raw[MAX_BYTES + 1024u];
  __local uchar *shared = align_local_1024(shared_raw);
  uint lid = get_local_id(0);
  (void)source_descriptor;
  (void)capture;

  __local uint *shared_words = (__local uint *)shared;
  __global const uint *source_words = (__global const uint *)source;
  for (uint index = lid; index < bytes / 4u; index += WG_SIZE)
    shared_words[index] = source_words[index];
  VENTUS_TMA_LOAD_COORDS_V12(coordinates);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      uint begin = read_cycle_lo();
      VENTUS_TMA_TENSOR_S2G(shared, destination_descriptor);
      VENTUS_TMA_S2G_COMMIT_GROUP();
      VENTUS_TMA_S2G_WAIT_GROUP0();
      uint end = read_cycle_lo();
      if (cycles) cycles[iteration] = end - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
}

kernel void
bench_feature_roundtrip(FEATURE_ARGUMENTS)
{
  __local uchar shared_raw[MAX_BYTES + 1024u];
  __local uint barrier_raw[4];
  __local uchar *shared = align_local_1024(shared_raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  (void)source;
  (void)capture;

  VENTUS_TMA_LOAD_COORDS_V12(coordinates);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      uint begin = read_cycle_lo();
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
      VENTUS_TMA_TENSOR_G2S(shared, source_descriptor);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
      VENTUS_TMA_TENSOR_S2G(shared, destination_descriptor);
      VENTUS_TMA_S2G_COMMIT_GROUP();
      VENTUS_TMA_S2G_WAIT_GROUP0();
      uint end = read_cycle_lo();
      if (cycles) cycles[iteration] = end - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
}

#undef FEATURE_ARGUMENTS

#define UARCH_TILE_BYTES 4096u
#define UARCH_MAX_OUTSTANDING 8u

#define INIT_OUTSTANDING(barrier_word)                                       \
  VENTUS_TMA_MBARRIER_INIT(mbarriers + (barrier_word), 1u)
#define ISSUE_OUTSTANDING(request, barrier_word)                             \
  do {                                                                       \
    VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(                                   \
        mbarriers + (barrier_word), UARCH_TILE_BYTES);                      \
    VENTUS_TMA_TENSOR_G2S(                                                  \
        shared + (request) * UARCH_TILE_BYTES, descriptor);                 \
  } while (0)
#define WAIT_OUTSTANDING(barrier_word)                                       \
  VENTUS_TMA_MBARRIER_WAIT(mbarriers + (barrier_word), 0u)

#define DEFINE_OUTSTANDING_KERNEL(kernel_name, depth)                        \
kernel void                                                                  \
kernel_name(__global uint *descriptor, __global const int *coordinates,     \
            __global uchar *capture, __global uint *cycles,                 \
            __global uint *status, uint warmups)                            \
{                                                                            \
  __local uchar shared_raw[UARCH_TILE_BYTES * (depth) + 1024u];             \
  __local uint barrier_raw[(depth) * 2u + 2u];                              \
  __local uchar *shared = align_local_1024(shared_raw);                     \
  __local uint *mbarriers = align_local_8(barrier_raw);                    \
  uint lid = get_local_id(0);                                               \
  VENTUS_TMA_LOAD_COORDS_V12(coordinates);                                  \
  if (lid == 0u) {                                                          \
    VENTUS_TMA_STATUS_CLEAR();                                              \
    INIT_OUTSTANDING(0u);                                                   \
    if ((depth) >= 2u) INIT_OUTSTANDING(2u);                               \
    if ((depth) >= 3u) INIT_OUTSTANDING(4u);                               \
    if ((depth) >= 4u) INIT_OUTSTANDING(6u);                               \
    if ((depth) >= 5u) INIT_OUTSTANDING(8u);                               \
    if ((depth) >= 6u) INIT_OUTSTANDING(10u);                              \
    if ((depth) >= 7u) INIT_OUTSTANDING(12u);                              \
    if ((depth) >= 8u) INIT_OUTSTANDING(14u);                              \
  }                                                                          \
  barrier(CLK_LOCAL_MEM_FENCE);                                             \
  for (uint iteration = 0u; iteration <= warmups; ++iteration) {           \
    if (lid == 0u) {                                                        \
      INIT_OUTSTANDING(0u);                                                \
      if ((depth) >= 2u) INIT_OUTSTANDING(2u);                             \
      if ((depth) >= 3u) INIT_OUTSTANDING(4u);                             \
      if ((depth) >= 4u) INIT_OUTSTANDING(6u);                             \
      if ((depth) >= 5u) INIT_OUTSTANDING(8u);                             \
      if ((depth) >= 6u) INIT_OUTSTANDING(10u);                            \
      if ((depth) >= 7u) INIT_OUTSTANDING(12u);                            \
      if ((depth) >= 8u) INIT_OUTSTANDING(14u);                            \
      uint begin = 0u;                                                      \
      if (iteration == warmups) begin = read_cycle_lo();                   \
      ISSUE_OUTSTANDING(0u, 0u);                                           \
      if ((depth) >= 2u) ISSUE_OUTSTANDING(1u, 2u);                        \
      if ((depth) >= 3u) ISSUE_OUTSTANDING(2u, 4u);                        \
      if ((depth) >= 4u) ISSUE_OUTSTANDING(3u, 6u);                        \
      if ((depth) >= 5u) ISSUE_OUTSTANDING(4u, 8u);                        \
      if ((depth) >= 6u) ISSUE_OUTSTANDING(5u, 10u);                       \
      if ((depth) >= 7u) ISSUE_OUTSTANDING(6u, 12u);                       \
      if ((depth) >= 8u) ISSUE_OUTSTANDING(7u, 14u);                       \
      WAIT_OUTSTANDING(0u);                                                \
      if ((depth) >= 2u) WAIT_OUTSTANDING(2u);                             \
      if ((depth) >= 3u) WAIT_OUTSTANDING(4u);                             \
      if ((depth) >= 4u) WAIT_OUTSTANDING(6u);                             \
      if ((depth) >= 5u) WAIT_OUTSTANDING(8u);                             \
      if ((depth) >= 6u) WAIT_OUTSTANDING(10u);                            \
      if ((depth) >= 7u) WAIT_OUTSTANDING(12u);                            \
      if ((depth) >= 8u) WAIT_OUTSTANDING(14u);                            \
      if (iteration == warmups) cycles[0] = read_cycle_lo() - begin;       \
    }                                                                        \
    barrier(CLK_LOCAL_MEM_FENCE);                                           \
  }                                                                          \
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);                         \
  __local uint *shared_words = (__local uint *)shared;                     \
  __global uint *capture_words = (__global uint *)capture;                 \
  for (uint index = lid;                                                    \
       index < (depth) * UARCH_TILE_BYTES / 4u; index += WG_SIZE)          \
    capture_words[index] = shared_words[index];                             \
}

DEFINE_OUTSTANDING_KERNEL(bench_outstanding_g2s_1, 1u)
DEFINE_OUTSTANDING_KERNEL(bench_outstanding_g2s_2, 2u)
DEFINE_OUTSTANDING_KERNEL(bench_outstanding_g2s_4, 4u)
DEFINE_OUTSTANDING_KERNEL(bench_outstanding_g2s_8, 8u)

#undef DEFINE_OUTSTANDING_KERNEL
#undef WAIT_OUTSTANDING
#undef ISSUE_OUTSTANDING
#undef INIT_OUTSTANDING

kernel void
bench_descriptor_set(__global uint *descriptors,
                     __global const int *coordinates,
                     __global uchar *capture, __global uint *cycles,
                     __global uint *status, uint prefetch, uint warmups)
{
  __local uchar shared_raw[UARCH_TILE_BYTES + 1024u];
  __local uint barrier_raw[4];
  __local uchar *shared = align_local_1024(shared_raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);

  VENTUS_TMA_LOAD_COORDS_V12(coordinates);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);

  for (uint iteration = 0u; iteration <= warmups; ++iteration) {
    if (lid == 0u) {
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      if (prefetch) VENTUS_TMA_PREFETCH_TENSORMAP(descriptors);
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, UARCH_TILE_BYTES);
      uint begin = 0u;
      if (iteration == warmups) begin = read_cycle_lo();
      VENTUS_TMA_TENSOR_G2S(shared, descriptors);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      if (iteration == warmups) cycles[0] = read_cycle_lo() - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }

  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *capture_words = (__global uint *)capture;
  for (uint index = lid; index < UARCH_TILE_BYTES / 4u; index += WG_SIZE)
    capture_words[index] = shared_words[index];
}

#undef UARCH_TILE_BYTES
#undef UARCH_MAX_OUTSTANDING
