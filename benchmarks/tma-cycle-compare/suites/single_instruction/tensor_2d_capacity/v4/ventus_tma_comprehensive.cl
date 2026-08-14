/* Comprehensive Ventus TMA kernels.
 *
 * Tensor kernels are the already protocol-correct feature kernels.  This file
 * adds a descriptor-base phase patcher, Bulk and six Reduce entry points with
 * the same complete-operation timing boundaries.
 */

#include "ventus_tma_feature_parity.cl"

kernel void
patch_descriptor_base_offset(__global uint *descriptor,
                             __global uchar *base, uint offset,
                             __global uint *identity)
{
  if (get_global_id(0) == 0u) {
    descriptor[2] = (uint)(base + offset);
    descriptor[3] = 0u;
    identity[0] = (uint)descriptor;
    identity[1] = (uint)(base + offset);
  }
}

#define COMPREHENSIVE_ARGUMENTS                                              \
  __global uint *descriptor, __global const int *coordinates,               \
      __global const uchar *source, __global uchar *destination,             \
      __global uchar *capture, __global uint *cycles,                        \
      __global uint *status, uint bytes, uint iterations, uint global_offset

kernel void
bench_comprehensive_bulk_g2s(COMPREHENSIVE_ARGUMENTS)
{
  __local uchar shared_raw[MAX_BYTES + 1024u];
  __local uint barrier_raw[4];
  __local uchar *shared = align_local_1024(shared_raw);
  __local uint *mbarrier = align_local_8(barrier_raw);
  uint lid = get_local_id(0);
  (void)descriptor;
  (void)coordinates;
  (void)destination;

  if (lid == 0u) {
    VENTUS_TMA_STATUS_CLEAR();
    VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
  }
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      /* Give every measured command a fresh completion generation. */
      VENTUS_TMA_MBARRIER_INIT(mbarrier, 1u);
      uint begin = read_cycle_lo();
      VENTUS_TMA_MBARRIER_ARRIVE_EXPECT_TX(mbarrier, bytes);
      VENTUS_TMA_BULK_G2S(shared, source + global_offset, bytes);
      VENTUS_TMA_MBARRIER_WAIT(mbarrier, 0u);
      cycles[iteration] = read_cycle_lo() - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
  __local uint *shared_words = (__local uint *)shared;
  __global uint *capture_words = (__global uint *)capture;
  for (uint index = lid; index < bytes / 4u; index += WG_SIZE)
    capture_words[index] = shared_words[index];
}

kernel void
bench_comprehensive_bulk_s2g(COMPREHENSIVE_ARGUMENTS)
{
  __local uchar shared_raw[MAX_BYTES + 1024u];
  __local uchar *shared = align_local_1024(shared_raw);
  uint lid = get_local_id(0);
  (void)descriptor;
  (void)coordinates;
  (void)capture;

  __local uint *shared_words = (__local uint *)shared;
  __global const uint *source_words = (__global const uint *)source;
  for (uint index = lid; index < bytes / 4u; index += WG_SIZE)
    shared_words[index] = source_words[index];
  barrier(CLK_LOCAL_MEM_FENCE);
  if (lid == 0u) VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();
  barrier(CLK_LOCAL_MEM_FENCE);
  for (uint iteration = 0u; iteration < iterations; ++iteration) {
    if (lid == 0u) {
      VENTUS_TMA_STATUS_CLEAR();
      uint begin = read_cycle_lo();
      VENTUS_TMA_BULK_S2G(destination + global_offset, shared, bytes);
      VENTUS_TMA_S2G_COMMIT_GROUP();
      VENTUS_TMA_S2G_WAIT_GROUP0();
      cycles[iteration] = read_cycle_lo() - begin;
    }
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);
  }
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);
}

#define DEFINE_REDUCE_KERNEL(kernel_name, instruction)                       \
kernel void                                                                   \
kernel_name(COMPREHENSIVE_ARGUMENTS)                                         \
{                                                                             \
  __local uchar shared_raw[MAX_BYTES + 1024u];                               \
  __local uchar *shared = align_local_1024(shared_raw);                      \
  __local uint *shared_words = (__local uint *)shared;                       \
  uint lid = get_local_id(0);                                                 \
  (void)source; (void)destination; (void)capture; (void)global_offset;       \
  for (uint index = lid; index < bytes / 4u; index += WG_SIZE)               \
    shared_words[index] = 0x01020304u ^ (index * 0x1021u);                  \
  VENTUS_TMA_LOAD_COORDS_V12(coordinates);                                   \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  if (lid == 0u) VENTUS_TMA_FENCE_PROXY_ASYNC_SHARED();                     \
  barrier(CLK_LOCAL_MEM_FENCE);                                              \
  for (uint iteration = 0u; iteration < iterations; ++iteration) {           \
    if (lid == 0u) {                                                          \
      VENTUS_TMA_STATUS_CLEAR();                                             \
      uint begin = read_cycle_lo();                                          \
      instruction(shared, descriptor);                                       \
      VENTUS_TMA_S2G_COMMIT_GROUP();                                         \
      VENTUS_TMA_S2G_WAIT_GROUP0();                                          \
      cycles[iteration] = read_cycle_lo() - begin;                           \
    }                                                                         \
    barrier(CLK_LOCAL_MEM_FENCE | CLK_GLOBAL_MEM_FENCE);                     \
  }                                                                           \
  if (lid == 0u) VENTUS_TMA_STATUS_READ(status[0]);                          \
}

DEFINE_REDUCE_KERNEL(bench_comprehensive_reduce_add,
                     VENTUS_TMA_TENSOR_REDUCE_ADD)
DEFINE_REDUCE_KERNEL(bench_comprehensive_reduce_min,
                     VENTUS_TMA_TENSOR_REDUCE_MIN)
DEFINE_REDUCE_KERNEL(bench_comprehensive_reduce_max,
                     VENTUS_TMA_TENSOR_REDUCE_MAX)
DEFINE_REDUCE_KERNEL(bench_comprehensive_reduce_and,
                     VENTUS_TMA_TENSOR_REDUCE_AND)
DEFINE_REDUCE_KERNEL(bench_comprehensive_reduce_or,
                     VENTUS_TMA_TENSOR_REDUCE_OR)
DEFINE_REDUCE_KERNEL(bench_comprehensive_reduce_xor,
                     VENTUS_TMA_TENSOR_REDUCE_XOR)

#undef DEFINE_REDUCE_KERNEL
#undef COMPREHENSIVE_ARGUMENTS
