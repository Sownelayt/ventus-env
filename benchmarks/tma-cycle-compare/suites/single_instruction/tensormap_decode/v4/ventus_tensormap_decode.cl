/* Ventus V3.10 TensorMap compiler/binder probe.
 *
 * Each process handles one rank/direction/scenario so the TESTCASE TOTAL PMU
 * lines belong to exactly two consecutive commands.  A cold-demand run uses
 * repeat 0 as the compiler miss and repeat 1 as the compiled-store hit.  Both
 * commands execute in this same kernel invocation with one descriptor.
 */

#include "ventus_tma_v2_opencl.h"

#define WG_SIZE 32u
#define TILE_BYTES 128u

static uint read_cycle_lo(void) {
  uint value;
  __asm__ volatile("csrr %0, 0xB00\n\t" : "=r"(value) :: "memory");
  return value;
}

static __local uchar *align_local_128(__local uchar *base) {
  return (__local uchar *)(((uint)base + 127u) & ~127u);
}

static __local uint *align_local_8(__local uint *base) {
  return (__local uint *)(((uint)base + 7u) & ~7u);
}

kernel void patch_descriptor_base(__global uint *descriptor,
                                  __global uchar *base,
                                  __global uint *identity) {
  if (get_global_id(0) == 0u) {
    descriptor[2] = (uint)base;
    descriptor[3] = 0u;
    identity[0] = (uint)descriptor;
    identity[1] = (uint)base;
  }
}

#define DECODE_ARGUMENTS                                                     \
  __global uint *descriptor, __global const int *coordinates,               \
      __global const uchar *source, __global uchar *destination,             \
      __global uchar *capture, __global uint *cycles,                        \
      __global uint *status, uint prefetch, uint prefetch_lead,              \
      uint iterations

kernel void bench_decode_g2s(DECODE_ARGUMENTS) {
  __local uchar shared_raw[TILE_BYTES + 128u];
  __local uint barrier_raw[4];
  __local uchar *shared = align_local_128(shared_raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  (void)source;
  (void)destination;
  VENTUS_TMA_LOAD_COORDS_V12(coordinates);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
    if (prefetch) VENTUS_TMA_PREFETCH_TENSORMAP(descriptor);
    for (uint delay = 0u; delay < prefetch_lead; ++delay)
      __asm__ volatile("nop\n\t" ::: "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      uint begin = read_cycle_lo();
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, TILE_BYTES);
      VENTUS_TMA_TENSOR_G2S(shared, descriptor);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      cycles[iteration] = read_cycle_lo() - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *capture_words = (__global uint *)capture;
  for (uint word = lid; word < TILE_BYTES / 4u; word += WG_SIZE)
    capture_words[word] = shared_words[word];
}

kernel void bench_decode_s2g(DECODE_ARGUMENTS) {
  __local uchar shared_raw[TILE_BYTES + 128u];
  __local uchar *shared = align_local_128(shared_raw);
  uint lid = get_local_id(0);
  (void)capture;
  __local uint *shared_words = (__local uint *)shared;
  __global const uint *source_words = (__global const uint *)source;
  for (uint word = lid; word < TILE_BYTES / 4u; word += WG_SIZE)
    shared_words[word] = source_words[word];
  VENTUS_TMA_LOAD_COORDS_V12(coordinates);
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
    if (prefetch) VENTUS_TMA_PREFETCH_TENSORMAP(descriptor);
    for (uint delay = 0u; delay < prefetch_lead; ++delay)
      __asm__ volatile("nop\n\t" ::: "memory");
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      uint begin = read_cycle_lo();
      VENTUS_TMA_TENSOR_S2G(shared, descriptor);
      VENTUS_TMA_S2G_COMMIT_GROUP();
      VENTUS_TMA_S2G_WAIT_GROUP0();
      cycles[iteration] = read_cycle_lo() - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
}

#undef DECODE_ARGUMENTS
