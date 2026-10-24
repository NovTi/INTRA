// INTRA: custom gather/scatter copies used by the token-group tile loaders in gmem_tile.h.
// Each call moves one 128-dim bf16 row (2 x 128-bit transfers) between a non-contiguous global
// address and a packed shared-memory/register tile.

#pragma once

#include <cute/tensor.hpp>
#include <cute/arch/copy_sm80.hpp>

namespace cute
{

template <class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
custom_copy_kv(char const *src, Tensor<DstEngine, DstLayout>& dst, int idx)
{
    using DstType = typename DstEngine::element_type;

    // load 128 bits / 16 bytes each time
    using DstVecType = conditional_t<is_volatile_v<DstType>, uint_bit_t<128> volatile, uint_bit_t<128>>;

    // const uint4* src_v = reinterpret_cast<const uint4*>(src);
    Tensor dst_v = recast<DstVecType>(dst);

#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)

      uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&dst_v(idx));
      const uint4* src_v = reinterpret_cast<const uint4*>(src);

      // first half of the 128 head dim
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
          :: "r"(smem_int_ptr),
            "l"(src_v),
            "n"(16));

      // second half of the 128 head dim
      smem_int_ptr = cast_smem_ptr_to_uint(&dst_v(idx+4));
      src_v = reinterpret_cast<const uint4*>(src) + 8;
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
          :: "r"(smem_int_ptr),
            "l"(src_v),
            "n"(16));
#else
    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");
#endif

}



template <class SrcEngine, class SrcLayout>
CUTE_HOST_DEVICE
void
custom_copy_o_r2g(Tensor<SrcEngine, SrcLayout>& src, char *dst, int idx)
{
    // First copy
    *reinterpret_cast<uint4*>(dst) = reinterpret_cast<const uint4&>(src(idx));
    
    // Second copy (offset by 128 bytes, 64 bfloat16 elements)
    *reinterpret_cast<uint4*>(dst + 128) = reinterpret_cast<const uint4&>(src(idx + 32));
}



template <class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
custom_copy_qo(char const *src, Tensor<DstEngine, DstLayout>& dst, int idx)
{
    using DstType = typename DstEngine::element_type;

    // load 128 bits / 16 bytes each time
    using DstVecType = conditional_t<is_volatile_v<DstType>, uint_bit_t<128> volatile, uint_bit_t<128>>;

    // const uint4* src_v = reinterpret_cast<const uint4*>(src);
    Tensor dst_v = recast<DstVecType>(dst);

#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)

      uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&dst_v(idx));
      const uint4* src_v = reinterpret_cast<const uint4*>(src);

      // first half of the 128 head dim
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
          :: "r"(smem_int_ptr),
            "l"(src_v),
            "n"(16));

      // second half of the 128 head dim
      smem_int_ptr = cast_smem_ptr_to_uint(&dst_v(idx+4));
      src_v = reinterpret_cast<const uint4*>(src) + 8;
      asm volatile("cp.async.ca.shared.global.L2::128B [%0], [%1], %2;\n"
          :: "r"(smem_int_ptr),
            "l"(src_v),
            "n"(16));
#else
    CUTE_INVALID_CONTROL_PATH("Support for cp.async instructions has not been enabled");
#endif

}

// Zero-fill the smem slots custom_copy_kv / custom_copy_qo would have written. The tile loaders
// skip the copy for out-of-range tokens in a partial tile; clearing (instead of leaving stale
// shared memory) keeps Inf/NaN out of the P*V gemm, the same contract as flash::copy<Clear_OOB_MN>.
template <class DstEngine, class DstLayout>
CUTE_HOST_DEVICE
void
custom_zero_kv(Tensor<DstEngine, DstLayout>& dst, int idx)
{
    using DstType = typename DstEngine::element_type;
    using DstVecType = conditional_t<is_volatile_v<DstType>, uint_bit_t<128> volatile, uint_bit_t<128>>;
    Tensor dst_v = recast<DstVecType>(dst);

    const uint32_t z = 0;
    uint32_t smem_int_ptr = cast_smem_ptr_to_uint(&dst_v(idx));
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n"
        :: "r"(smem_int_ptr), "r"(z));
    smem_int_ptr = cast_smem_ptr_to_uint(&dst_v(idx + 4));
    asm volatile("st.shared.v4.u32 [%0], {%1, %1, %1, %1};\n"
        :: "r"(smem_int_ptr), "r"(z));
}

} // namespace cute
