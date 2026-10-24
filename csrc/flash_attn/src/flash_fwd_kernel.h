/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"
#include "dropout.h"
#include "rotary.h"

namespace flash {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename ElementAccum, typename Params, int kBlockM, bool Is_even_MN>
__forceinline__ __device__ auto get_lse_tile(const Params &params, const int bidb, const int bidh, const int m_block, const BlockInfo</*Varlen=*/!Is_even_MN> &binfo) {
        // When params.unpadded_lse is false, LSE is written as (b, h, seqlen_q) - this is non-variable seqlen path.
        // Otherwise, when params.seqlenq_ngroups_swapped is true, it is written as (h, seqlen_q, b) to account for seqlen_q <-> h swapping trick.
        // Otherwise, it's written as (h, b, seqlen_q).
        const bool varlen_q = params.unpadded_lse && !params.seqlenq_ngroups_swapped;
        auto lse_offset = varlen_q ? binfo.q_offset(params.seqlen_q, 1, bidb) : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);

        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = params.seqlenq_ngroups_swapped ? make_stride(1, params.seqlen_q * params.b, params.b) : (
            params.unpadded_lse ? make_stride(params.h * params.total_q, params.total_q, 1) :  make_stride(params.h * params.seqlen_q, params.seqlen_q, 1)
            );

        auto lse_layout = make_layout(lse_shape, lse_stride);
        Tensor mLSE = make_tensor(gmem_ptr_lse, lse_layout);
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        return local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
// inline __device__ void compute_attn_1rowblock(const Params &params, const int bidb, const int bidh, const int m_block) {

//     using Element = typename Kernel_traits::Element;
//     using ElementAccum = typename Kernel_traits::ElementAccum;
//     using index_t = typename Kernel_traits::index_t;

//     // Shared memory.
//     extern __shared__ char smem_[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     constexpr int kBlockM = Kernel_traits::kBlockM;
//     constexpr int kBlockN = Kernel_traits::kBlockN;
//     constexpr int kHeadDim = Kernel_traits::kHeadDim;
//     constexpr int kNWarps = Kernel_traits::kNWarps;

//     auto seed_offset = at::cuda::philox::unpack(params.philox_args);
//     flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
//                            bidb, bidh, tidx, params.h);

//     // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
//     // exit early and no one saves the rng states.
//     if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//         params.rng_state[0] = std::get<0>(seed_offset);
//         params.rng_state[1] = std::get<1>(seed_offset);
//     }

//     const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
//     if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

//     const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
//     int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
//     if (Is_causal || Is_local) {
//         n_block_max = std::min(n_block_max,
//                                cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
//         // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
//         //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
//         // }
//     }
//     // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
//     // Otherwise we might read OOB elements from gK and gV.
//     if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
//         Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//                                               + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
//                                 make_shape(binfo.actual_seqlen_q, params.h, params.d),
//                                 make_stride(params.o_row_stride, params.o_head_stride, _1{}));
//         Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                               make_coord(m_block, 0));  // (kBlockM, kHeadDim)

//         Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

//         typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
//         auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//         Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
//         Tensor tOrO = make_tensor<Element>(shape(tOgO));
//         clear(tOrO);
//         // Construct identity layout for sO
//         Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//         // Repeat the partitioning with identity layouts
//         Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
//         Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
//         if (!Is_even_K) {
//             #pragma unroll
//             for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
//         }
//         // Clear_OOB_K must be false since we don't want to write zeros to gmem
//         flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//             gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
//         );
//         #pragma unroll
//         for (int m = 0; m < size<1>(tOgO); ++m) {
//             const int row = get<0>(tOcO(0, m, 0));
//             if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
//         }
//         return;
//     }
//     // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

//     // We iterate over the blocks in reverse order. This is because the last block is the only one
//     // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
//     // might save us 1 register (we just need n_block instead of both n_block and n_block_max).
    
//     const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
//         + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

//     Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//                             make_shape(binfo.actual_seqlen_q, params.h, params.d),
//                             make_stride(params.q_row_stride, params.q_head_stride, _1{}));
//     Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                         make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//     //     printf("Is_causal: %d | Is_local: %d | Is_even_MN: %d\n", Is_causal, Is_local, Is_even_MN);
//     // }

//     Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
//                                           + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
//                             make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
//                             make_stride(params.k_row_stride, params.k_head_stride, _1{}));
//     Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                            make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
//     Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
//                                           + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
//                             make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
//                             make_stride(params.v_row_stride, params.v_head_stride, _1{}));
//     Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                            make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
//     Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
//                             Shape<Int<kBlockM>, Int<kBlockN>>{},
//                             make_stride(params.seqlen_k_rounded, _1{}));

//     Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
//                             typename Kernel_traits::SmemLayoutQ{});
//     // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
//     Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
//                             typename Kernel_traits::SmemLayoutKV{});
//     Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
//     Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
//     Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

//     typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
//     auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

//     Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
//     Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
//     Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
//     Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
//     Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
//     Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

//     typename Kernel_traits::TiledMma tiled_mma;
//     auto thr_mma = tiled_mma.get_thread_slice(tidx);
//     Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
//     Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
//     Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

//     Tensor tSgS  = thr_mma.partition_C(gP);

//     Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//     //     // ((_8,_1),_8,_1,32):((_1,_0),10240,_0,81920)
//     //     // cute::print_tensor(tSgS);

//     //     // kBlockN: 128 | kBlockM: 128
//     //     printf("kBlockM: %d | kBlockN: %d |  kHeadDim: %d | actual_q: %d | m_block: %d\n", kBlockM, kBlockN, kHeadDim, binfo.actual_seqlen_q, m_block);
//     //     printf("sum_s_q: %d | sum_s_k: %d\n", binfo.sum_s_q, binfo.sum_s_k);

//     //     // printf("kBlockM: %d | kHeadDim: %d | actual_seqlen_q: %d\n", kBlockM, kHeadDim, binfo.actual_seqlen_q);
//     //     const int L = sqrtf(binfo.actual_seqlen_q);
        
//     //     // Layout layout = make_layout(
//     //     //     make_shape(params.d, L/2, L/2, 2, 2, params.h),
//     //     //     make_stride(1, 2*head_dim*params.h, 2*L*params.d*params.h, L*params.d*params.h, params.d*params.h, params.d)
//     //     // );

//     //     Tensor mQ_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//     //                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //                             make_shape(params.d, L/2, L/2, 2, 2, params.h),
//     //                             make_stride(1, 2*params.d*params.h, 2*L*params.d*params.h, L*params.d*params.h, params.d*params.h, params.d));
//     //                             // local_tile(A(_, _, _, _, _, bidh), Shape<Int<4>, Int<4>, Int<2>>{}, make_coord(0, 0, 0, 0, 0));
//     //     Tensor gQ_test = local_tile(mQ_test(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<32>, Int<2>>{},
//     //                         make_coord(0, 0, 0, 0, 0));  // (kBlockM, kHeadDim)

//     //     Tensor mK_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)),
//     //                             make_shape(params.d, L/2, L/2, 2, 2, params.h),
//     //                             make_stride(1, 2*params.d*params.h, 2*L*params.d*params.h, L*params.d*params.h, params.d*params.h, params.d));
//     //     Tensor gK_test = local_tile(mK_test(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<32>, Int<8>>{},
//     //                         make_coord(0, 0, 0, 0, 0));  // (kBlockM, kHeadDim)

//     //     using GmemLayoutAtom = Layout<Shape <_8, _16>,
//     //                                 Stride<_1, _8>>;

//     //     printf(
//     //         "kBlockM: %d | kHeadDim: %d | actual_seqlen_q: %d | k_offset: %ld | q_offset: %ld\n",
//     //         kBlockM, kHeadDim, binfo.actual_seqlen_q,
//     //         binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
//     //         binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
//     //     );

//     //     printf(
//     //         "GridDim: (%d, %d, %d) | BlockDim: (%d, %d, %d) | row_offset_p: %ld\n",
//     //         gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, row_offset_p
//     //     );


//     //     // We use CACHEGLOBAL instead of CACHEALWAYS for both Q and K/V, since we won't be reading
//     //     // from the same address by the same threadblock. This is slightly faster.
//     //     using Gmem_copy_struct = std::conditional_t<
//     //         true,
//     //         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
//     //         DefaultCopy
//     //     >;

//     //     TiledCopy tle_copy_test = make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
//     //                         GmemLayoutAtom{},
//     //                         Layout<Shape<_8, _1>>{});  // Val layout, 8 vals per read

//     //     auto gmem_thr_copy_QKV_test = tle_copy_test.get_thread_slice(tidx);

//     //     // Tensor tQsQ_test = gmem_thr_copy_QKV.partition_D(sQ);
//     //     // Tensor tQgQ_test = gmem_thr_copy_QKV_test.partition_S(gQ_test);
//     //     Tensor tKgK_test = gmem_thr_copy_QKV_test.partition_S(gK_test);

//     //     // cute::copy(tQgQ_test(_, 0, 0, 0), tQsQ_test(_, 0, 0));
//     //     // cute::copy(tQgQ_test(_, 0, 1, 0), tQsQ_test(_, 1, 0));
//     //     // cute::copy(tQgQ_test(_, 0, 0, 1), tQsQ_test(_, 2, 0));
//     //     // cute::copy(tQgQ_test(_, 0, 1, 1), tQsQ_test(_, 3, 0));

//     //     // // (_8,_1),_1,_2,_2)
//     //     // cute::print_tensor(tQgQ_test);
//     //     // cute::print_tensor(tQsQ_test);
//     //     // cute::print_tensor(tKgK_test);

//     //     printf(
//     //         "kNThreads: %d | kGmemThreadsPerRow: %d | Has_cp_async: %d | Is_even_MN: %d | kBlockN: %d\n",
//     //         Kernel_traits::kNThreads, Kernel_traits::kGmemThreadsPerRow, Kernel_traits::Has_cp_async, Is_even_MN, kBlockN
//     //     );
//     // }

    
//     //
//     // Copy Atom retiling
//     //

//     auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
//     auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
//     // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
//     Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
//     // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

//     auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
//     auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
//     Tensor tSsK = smem_thr_copy_K.partition_S(sK);

//     auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
//     auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
//     Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

//     //
//     // PREDICATES
//     //

//     // // Allocate predicate tensors for m and n
//     // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
//     // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

//     // Construct identity layout for sQ and sK
//     Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
//     // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
//     // if (cute::thread0()) {
//     //     print(tScQ.layout()); printf("\n");
//     //     for (int i = 0; i < size(tScQ); ++i) {
//     //         printf("%d ", get<0>(tScQ(i)));
//     //     }
//     //     printf("\n");
//     //     for (int i = 0; i < size(tScQ); ++i) {
//     //         printf("%d ", get<1>(tScQ(i)));
//     //     }
//     //     printf("\n");
//     // }

//     // Repeat the partitioning with identity layouts
//     Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
//     Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

//     // Allocate predicate tensors for k
//     Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
//     Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

//     // Set predicates for k bounds
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
//         #pragma unroll
//         for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
//     }

//     // Prologue

//     // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
//     flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
//                                        binfo.actual_seqlen_q - m_block * kBlockM);
//     if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

//     // // if (cute::thread(1, 0)) { print(tQsQ); }
//     // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
//     // // if (cute::thread0()) { print(sQNoSwizzle); }

//     if (Kernel_traits::Share_Q_K_smem) {
//         flash::cp_async_wait<0>();
//         __syncthreads();
//         Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
//         CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
//         cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
//         __syncthreads();
//     }

//     int n_block = n_block_max - 1;
//     // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
//     flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
//                                        binfo.actual_seqlen_k - n_block * kBlockN);
//     cute::cp_async_fence();
//     // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
//     // __syncthreads();

//     if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
//         flash::cp_async_wait<1>();
//         __syncthreads();
//         Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
//         CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
//         cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
//     }

//     clear(acc_o);

//     flash::Softmax<2 * size<1>(acc_o)> softmax;

//     const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
//     flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

//     // For performance reason, we separate out two kinds of iterations:
//     // those that need masking on S, and those that don't.
//     // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
//     // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
//     // We will have at least 1 "masking" iteration.

//     // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
//     // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
//     constexpr int n_masking_steps = (!Is_causal && !Is_local)
//         ? 1
//         : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
//     #pragma unroll
//     for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         flash::cp_async_wait<0>();
//         __syncthreads();

//         // Advance gV
//         if (masking_step > 0) {
//             flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
//         } else {
//             // Clear the smem tiles to account for predicated off loads
//             flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
//                 gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
//             );
//         }
//         cute::cp_async_fence();

//         flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
//             acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
//             smem_thr_copy_Q, smem_thr_copy_K
//         );
//         // if (cute::thread0()) { print(acc_s); }
//         if constexpr (Is_softcap){
//             flash::apply_softcap(acc_s, params.softcap);
//         }

//         mask.template apply_mask<Is_causal, Is_even_MN>(
//             acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
//         );

//         flash::cp_async_wait<0>();
//         __syncthreads();
//         if (n_block > n_block_min) {
//             flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
//             // This cp_async_fence needs to be in the if block, otherwise the synchronization
//             // isn't right and we get race conditions.
//             cute::cp_async_fence();
//         }

//         // TODO: when we have key_padding_mask we'll need to Check_inf
//         masking_step == 0
//             ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
//             : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

//         // Convert acc_s from fp32 to fp16/bf16
//         Tensor rP = flash::convert_type<Element>(acc_s);
//         int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
//         int block_col_idx = n_block * (kBlockN / 32);
//         if (Return_softmax) {
//             Tensor rP_drop = make_fragment_like(rP);
//             cute::copy(rP, rP_drop);
//             dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
//                 rP_drop, block_row_idx, block_col_idx, kNWarps
//             );
//             cute::copy(rP_drop, tSgS);
//             tSgS.data() = tSgS.data() + (-kBlockN);
//         }
//         if (Is_dropout) {
//             dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
//         }

//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
//         Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
//         // if (cute::thread0()) { print(tOrP); }
//         flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
//         // if (cute::thread0()) { print(scores); }

//         // This check is at the end of the loop since we always have at least 1 iteration
//         if (n_masking_steps > 1 && n_block <= n_block_min) {
//             --n_block;
//             break;
//         }
//     }

//     // These are the iterations where we don't need masking on S
//     for (; n_block >= n_block_min; --n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         flash::cp_async_wait<0>();
//         __syncthreads();
//         flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
//         cute::cp_async_fence();

//         flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
//             acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
//             smem_thr_copy_Q, smem_thr_copy_K
//         );
//         if constexpr (Is_softcap){
//             flash::apply_softcap(acc_s, params.softcap);
//         }

//         flash::cp_async_wait<0>();
//         __syncthreads();
//         if (n_block > n_block_min) {
//             flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
//             // This cp_async_fence needs to be in the if block, otherwise the synchronization
//             // isn't right and we get race conditions.
//             cute::cp_async_fence();
//         }

//         mask.template apply_mask</*Causal_mask=*/false>(
//             acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
//         );

//         softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

//         Tensor rP = flash::convert_type<Element>(acc_s);
//         int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
//         int block_col_idx = n_block * (kBlockN / 32);
//         if (Return_softmax) {
//             Tensor rP_drop = make_fragment_like(rP);
//             cute::copy(rP, rP_drop);
//             dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
//                 rP_drop, block_row_idx, block_col_idx, kNWarps
//             );
//             cute::copy(rP_drop, tSgS);
//             tSgS.data() = tSgS.data() + (-kBlockN);
//         }
//         if (Is_dropout) {
//             dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
//         }

//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
//         Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
//         flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
//     }

//     // Epilogue

//     Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

//     // Convert acc_o from fp32 to fp16/bf16
//     Tensor rO = flash::convert_type<Element>(acc_o);
//     Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
//     // Partition sO to match the accumulator partitioning
//     auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
//     auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
//     Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     // sO has the same size as sQ, so we don't need to sync here.
//     if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

//     cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

//     Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//                                           + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
//                             make_shape(binfo.actual_seqlen_q, params.h, params.d),
//                             make_stride(params.o_row_stride, params.o_head_stride, _1{}));
//     Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

//     // Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

//     typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
//     auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

//     __syncthreads();

//     Tensor tOrO = make_tensor<Element>(shape(tOgO));
//     cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

//     Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
//     static_assert(decltype(size<0>(taccOcO))::value == 4);
//     // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
//     Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);

//     // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
//     // if (get<1>(taccOcO_row(0)) == 0) {
//     //     #pragma unroll
//     //     for (int mi = 0; mi < size(lse); ++mi) {
//     //         const int row = get<0>(taccOcO_row(mi));
//     //         if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
//     //     }
//     // }

//     // Construct identity layout for sO
//     Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     // Repeat the partitioning with identity layouts
//     Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
//     Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
//     }
//     // Clear_OOB_K must be false since we don't want to write zeros to gmem
//     flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//         gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
//     );
// }

// template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
// inline __device__ void compute_attn_1rowblock_sparse_mask0(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

//     using Element = typename Kernel_traits::Element;
//     using ElementAccum = typename Kernel_traits::ElementAccum;
//     using index_t = typename Kernel_traits::index_t;

//     // Shared memory.
//     extern __shared__ char smem_[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     constexpr int kBlockM = Kernel_traits::kBlockM;
//     constexpr int kBlockN = Kernel_traits::kBlockN;
//     constexpr int kHeadDim = Kernel_traits::kHeadDim;
//     constexpr int kNWarps = Kernel_traits::kNWarps;

//     constexpr int sparseTileX = Kernel_traits::sparseTileX;
//     constexpr int sparseTileMY = Kernel_traits::sparseTileMY;
//     constexpr int sparseTileNY = Kernel_traits::sparseTileNY;

//     auto seed_offset = at::cuda::philox::unpack(params.philox_args);
//     flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
//                            bidb, bidh, tidx, params.h);

//     // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
//     // exit early and no one saves the rng states.
//     if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//         params.rng_state[0] = std::get<0>(seed_offset);
//         params.rng_state[1] = std::get<1>(seed_offset);
//     }

//     const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
//     if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

//     const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
//     int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
//     if (Is_causal || Is_local) {
//         n_block_max = std::min(n_block_max,
//                                cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
//         // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
//         //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
//         // }
//     }
//     // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
//     // Otherwise we might read OOB elements from gK and gV.
//     if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
//         Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//                                               + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
//                                 make_shape(binfo.actual_seqlen_q, params.h, params.d),
//                                 make_stride(params.o_row_stride, params.o_head_stride, _1{}));
//         Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                               make_coord(m_block, 0));  // (kBlockM, kHeadDim)

//         Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

//         typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
//         auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//         Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
//         Tensor tOrO = make_tensor<Element>(shape(tOgO));
//         clear(tOrO);
//         // Construct identity layout for sO
//         Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//         // Repeat the partitioning with identity layouts
//         Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
//         Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
//         if (!Is_even_K) {
//             #pragma unroll
//             for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
//         }
//         // Clear_OOB_K must be false since we don't want to write zeros to gmem
//         flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//             gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
//         );
//         #pragma unroll
//         for (int m = 0; m < size<1>(tOgO); ++m) {
//             const int row = get<0>(tOcO(0, m, 0));
//             if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
//         }
//         return;
//     }
//     // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

//     // We iterate over the blocks in reverse order. This is because the last block is the only one
//     // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
//     // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

//     const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
//         + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

//     // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//     //                                       + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
//     //                         make_stride(params.q_row_stride, params.q_head_stride, _1{}));
//     // Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//     //                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
//     // Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
//     //                                       + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
//     //                         make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
//     //                         make_stride(params.k_row_stride, params.k_head_stride, _1{}));
//     // Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//     //                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
//     // Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
//     //                                       + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
//     //                         make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
//     //                         make_stride(params.v_row_stride, params.v_head_stride, _1{}));
//     // Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
//     //                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)


//     ////////////////// Sparse //////////////////

//     // ////////////////// Sparse Mask 0 //////////////////
//     Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     Tensor gQ = local_tile(mQ(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileMY>>{},
//                         make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));  // (kBlockM, kHeadDim)

//     Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
//                                           + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
//                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     Tensor gK = local_tile(mK(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileNY>>{},
//                         make_coord(0, 0, _, outgroup_id, inside_outgroup_id));

//     Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
//                                           + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
//                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     Tensor gV = local_tile(mV(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileNY>>{},
//                            make_coord(0, 0, _, outgroup_id, inside_outgroup_id));


//     // //////////////// Sparse Mask 1 //////////////////
//     // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//     //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
//     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
//     // Tensor gQ = local_tile(mQ(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<ssparseTileMY>>{},
//     //                     make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

//     // Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
//     //                                       + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
//     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
//     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
//     // Tensor gK = local_tile(mK(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileNYMask2>>{},
//     //                     make_coord(0, 0, 0, 0, _, outgroup_id, inside_outgroup_id));

//     // Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
//     //                                       + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
//     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
//     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
//     // Tensor gV = local_tile(mV(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileNYMask2>>{},
//     //                        make_coord(0, 0, 0, 0, _, outgroup_id, inside_outgroup_id));


//     Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
//                             Shape<Int<kBlockM>, Int<kBlockN>>{},
//                             make_stride(params.seqlen_k_rounded, _1{}));

//     Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
//                             typename Kernel_traits::SmemLayoutQ{});
//     // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
//     Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
//                             typename Kernel_traits::SmemLayoutKV{});
//     Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
//     Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
//     Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

//     typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
//     auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

//     // Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
//     // Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
//     // Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
//     // Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
//     // Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
//     // Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

//     ////////////////// Sparse //////////////////

//     typename Kernel_traits::GmemTiledCopyQKVSparse gmem_tiled_copy_QKV_Sparse;
//     auto gmem_thr_copy_QKV_Sparse = gmem_tiled_copy_QKV_Sparse.get_thread_slice(tidx);

//     Tensor tQgQ = gmem_thr_copy_QKV_Sparse.partition_S(gQ);
//     Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

//     Tensor tKgK = gmem_thr_copy_QKV_Sparse.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
//     Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

//     Tensor tVgV = gmem_thr_copy_QKV_Sparse.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
//     Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0){
//     //     // Tensor mQ_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//     //     //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
//     //     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     //     // Tensor gQ_test = local_tile(mQ_test(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<ssparseTileMY>>{},
//     //     //                     make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

//     //     // Tensor mK_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
//     //     //                                     + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
//     //     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
//     //     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     //     // Tensor gK_test = local_tile(mK_test(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileNYMask2>>{},
//     //     //                     make_coord(0, 0, 0, 0, _, outgroup_id, inside_outgroup_id));





//     //     ////// Mask 1 for 1024 //////
//     //     using GmemLayoutAtom = Layout<Shape <_8, _8, _2>,
//     //                                 Stride<_1, _8, _64>>;

//     //     using Gmem_copy_struct = std::conditional_t<
//     //         true,
//     //         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
//     //         DefaultCopy
//     //     >;

//     //     TiledCopy tle_copy_test = make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
//     //                         GmemLayoutAtom{},
//     //                         Layout<Shape<_8, _1, _1>>{});  // Val layout, 8 vals per read

//     //     auto gmem_thr_copy_QKV_test = tle_copy_test.get_thread_slice(tidx);

//     //     Tensor tQgQ_test = gmem_thr_copy_QKV_test.partition_S(gQ);
//     //     cute::print_tensor(tQgQ);
//     //     cute::print_tensor(tQgQ_test);
//     //     // cute::print_tensor(tQgQ_test(_, 0, 0, _, _, _));
//     //     ////// Mask 1 for 1024 //////

        

//     //     // gQ: 4096: (_64,_16,_2,_2,_2):(1,2560,640,40960,163840)  1024: (_64,_8,_2,_2,_4):(1,2560,640,20480,81920)
//     //     // tQgQ: ((_8,_1),_1,_1,_2,_2,_2):((1,_0),_0,_0,640,40960,163840)

//     //     // gK: (_64,_16,_2,_2,_2,8):(1,2560,640,40960,163840,327680)
//     //     // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
//     //     // cute::print_tensor(tQgQ);

//     //     // cute::print_tensor(tKgK_test);

//     //     // tQgQ: 4096: ((_8,_1),_1,_2,_4):((1,_0),_0,20480,81920)  1024: ((_8,_1),_1,_1,_8):((1,_0),_0,_0,40960)
//     //     // tVgV:  ((_8,_1),_1,_2,_4):((1,_0),_0,20480,81920)

//     //     // // tQsQ: 4096: ((_8,_1),_8,_1):((_1,_0),_1024,_0)  1024: ((_8,_1),_8,_1):((_1,_0),_1024,_0)
//     // }


//     ///////////////////////////////////////////////////////////////////////////

//     typename Kernel_traits::TiledMma tiled_mma;
//     auto thr_mma = tiled_mma.get_thread_slice(tidx);
//     Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
//     Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
//     Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

//     Tensor tSgS  = thr_mma.partition_C(gP);

//     Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//     //     // ((_8,_1),_8,_1,32):((_1,_0),10240,_0,81920)
//     //     // cute::print_tensor(tSgS);

//     //     // kBlockN: 128 | kBlockM: 128
//     //     printf(
//     //         "kBlockM: %d | kBlockN: %d | kHeadDim: %d | actual_q: %d | m_block: %d | sparseTileX: %d | sparseTileMY: %d | sparseTileNY: %d\n",
//     //         kBlockM, kBlockN, kHeadDim, binfo.actual_seqlen_q, m_block, sparseTileX, sparseTileMY, sparseTileNY
//     //     );
//     //     printf("sum_s_q: %d | sum_s_k: %d | n_block_max: %d | n_block_min: %d\n", binfo.sum_s_q, binfo.sum_s_k, n_block_max, n_block_min);

//     //     printf("kBlockM: %d | kHeadDim: %d | actual_seqlen_q: %d\n", kBlockM, kHeadDim, binfo.actual_seqlen_q);

//     //     Tensor mQ_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//     //                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//     //                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     //     Tensor gQ_test = local_tile(mQ_test(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileMY>>{},
//     //                         make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));  // (kBlockM, kHeadDim)

//     //     Tensor mK_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)),
//     //                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//     //                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     //     Tensor gK_test = local_tile(mK_test(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<32>, Int<4>>{},
//     //                         make_coord(0, 0, _, outgroup_id, inside_outgroup_id));  // (kBlockM, kHeadDim)

//     //     using GmemLayoutAtom = Layout<Shape <_8, _16>,
//     //                                 Stride<_1, _8>>;

//     //     using Gmem_copy_struct = std::conditional_t<
//     //         true,
//     //         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
//     //         DefaultCopy
//     //     >;

//     //     TiledCopy tle_copy_test = make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
//     //                         GmemLayoutAtom{},
//     //                         Layout<Shape<_8, _1>>{});  // Val layout, 8 vals per read

//     //     auto gmem_thr_copy_QKV_test = tle_copy_test.get_thread_slice(tidx);

//     //     Tensor tQsQ_test = gmem_thr_copy_QKV.partition_D(sQ);
//     //     // Tensor tKsK_test = gmem_thr_copy_QKV.partition_D(sK);
//     //     Tensor tQgQ_test = gmem_thr_copy_QKV_test.partition_S(gQ_test);
//     //     // Tensor tKgK_test = gmem_thr_copy_QKV_test.partition_S(gK_test);

//     //     cute::print_tensor(tQgQ_test);

//     //     printf("sQ\n");
//     //     cute::print_tensor(tQsQ_test);

//     //     flash::copy_sparse<Is_even_MN, Is_even_K>(tQgQ_test, tQsQ_test);
        
//     //     cute::print_tensor(tQsQ_test);

//     //     // printf(
//     //     //     "kBlockM: %d | kHeadDim: %d | actual_seqlen_q: %d | k_offset: %ld | q_offset: %ld\n",
//     //     //     kBlockM, kHeadDim, binfo.actual_seqlen_q,
//     //     //     binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
//     //     //     binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)
//     //     // );

//     //     // printf(
//     //     //     "GridDim: (%d, %d, %d) | BlockDim: (%d, %d, %d) | row_offset_p: %ld\n",
//     //     //     gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, row_offset_p
//     //     // );


//     //     printf(
//     //         "kNThreads: %d | kGmemThreadsPerRow: %d | Has_cp_async: %d | Is_even_MN: %d | Is_even_K: %d\n",
//     //         Kernel_traits::kNThreads, Kernel_traits::kGmemThreadsPerRow, Kernel_traits::Has_cp_async, Is_even_MN, Is_even_K
//     //     );
//     // }

//     //
//     // Copy Atom retiling
//     //

//     auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
//     auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
//     // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
//     Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
//     // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

//     auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
//     auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
//     Tensor tSsK = smem_thr_copy_K.partition_S(sK);

//     auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
//     auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
//     Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

//     //
//     // PREDICATES
//     //

//     // // Allocate predicate tensors for m and n
//     // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
//     // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

//     // Construct identity layout for sQ and sK
//     Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
//     // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
//     // if (cute::thread0()) {
//     //     print(tScQ.layout()); printf("\n");
//     //     for (int i = 0; i < size(tScQ); ++i) {
//     //         printf("%d ", get<0>(tScQ(i)));
//     //     }
//     //     printf("\n");
//     //     for (int i = 0; i < size(tScQ); ++i) {
//     //         printf("%d ", get<1>(tScQ(i)));
//     //     }
//     //     printf("\n");
//     // }

//     // Repeat the partitioning with identity layouts
//     Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
//     Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

//     // Allocate predicate tensors for k
//     Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
//     Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

//     // Set predicates for k bounds
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
//         #pragma unroll
//         for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
//     }

//     // Prologue

//     // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
//     // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
//     //                                    binfo.actual_seqlen_q - m_block * kBlockM);
//     flash::copy_sparse(tQgQ, tQsQ);

    
//     // // if (cute::thread(1, 0)) { print(tQsQ); }
//     // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
//     // // if (cute::thread0()) { print(sQNoSwizzle); }

//     if (Kernel_traits::Share_Q_K_smem) {
//         flash::cp_async_wait<0>();
//         __syncthreads();
//         Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
//         CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
//         cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
//         __syncthreads();
//     }

//     int n_block = n_block_max - 1;
//     // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
//     // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
//     //                                    binfo.actual_seqlen_k - n_block * kBlockN);
//     /////// Mask 0 ///////
//     flash::copy_sparse(tKgK(_, _, _, _, n_block), tKsK);
//     ////// Mask 0 ///////

//     // ////// Mask 1 ///////
//     // // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
//     // flash::copy_sparse(tKgK(_, _, _, _, _, _, n_block), tKsK);
//     // ////// Mask 1 ///////


    
//     cute::cp_async_fence();
//     // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
//     // __syncthreads();

//     if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
//         flash::cp_async_wait<1>();
//         __syncthreads();
//         Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
//         CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
//         cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
//     }

//     clear(acc_o);

//     flash::Softmax<2 * size<1>(acc_o)> softmax;

//     const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
//     flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

//     // For performance reason, we separate out two kinds of iterations:
//     // those that need masking on S, and those that don't.
//     // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
//     // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
//     // We will have at least 1 "masking" iteration.

//     // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
//     // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
//     constexpr int n_masking_steps = (!Is_causal && !Is_local)
//         ? 1
//         : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
//     #pragma unroll
//     for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         flash::cp_async_wait<0>();
//         __syncthreads();

//         // // Advance gV
//         // if (masking_step > 0) {
//         //     // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);
//         //     flash::copy_sparse(tVgV(_, _, _, _, n_block), tVsV);
//         // } else {
//         //     // // Clear the smem tiles to account for predicated off loads
//         //     // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
//         //     //     gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
//         //     // );
//         //     flash::copy_sparse(tVgV(_, _, _, _, n_block), tVsV);
//         // }

//         /////// Mask 0 ///////
//         flash::copy_sparse(tVgV(_, _, _, _, n_block), tVsV);
//         ////// Mask 0 ///////

//         // ////// Mask 1 ///////
//         // // tVgV: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
//         // flash::copy_sparse(tVgV(_, _, _, _, _, _, n_block), tVsV);
//         // ////// Mask 1 ///////

//         cute::cp_async_fence();

//         flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
//             acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
//             smem_thr_copy_Q, smem_thr_copy_K
//         );
//         // if (cute::thread0()) { print(acc_s); }
//         if constexpr (Is_softcap){
//             flash::apply_softcap(acc_s, params.softcap);
//         }

//         mask.template apply_mask<Is_causal, Is_even_MN>(
//             acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
//         );

//         flash::cp_async_wait<0>();
//         __syncthreads();
//         if (n_block > n_block_min) {
//             // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
//             // flash::copy_sparse(tKgK(_, _, _, _, n_block - 1), tKsK);

//             /////// Mask 0 ///////
//             flash::copy_sparse(tKgK(_, _, _, _, n_block - 1), tKsK);
//             ////// Mask 0 ///////

//             // ////// Mask 1 ///////
//             // // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
//             // flash::copy_sparse(tKgK(_, _, _, _, _, _, n_block - 1), tKsK);
//             // ////// Mask 1 ///////

//             // This cp_async_fence needs to be in the if block, otherwise the synchronization
//             // isn't right and we get race conditions.
//             cute::cp_async_fence();
//         }

//         // TODO: when we have key_padding_mask we'll need to Check_inf
//         masking_step == 0
//             ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
//             : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

//         // Convert acc_s from fp32 to fp16/bf16
//         Tensor rP = flash::convert_type<Element>(acc_s);
//         int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
//         int block_col_idx = n_block * (kBlockN / 32);
//         if (Return_softmax) {
//             Tensor rP_drop = make_fragment_like(rP);
//             cute::copy(rP, rP_drop);
//             dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
//                 rP_drop, block_row_idx, block_col_idx, kNWarps
//             );
//             cute::copy(rP_drop, tSgS);
//             tSgS.data() = tSgS.data() + (-kBlockN);
//         }
//         if (Is_dropout) {
//             dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
//         }

//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
//         Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
//         // if (cute::thread0()) { print(tOrP); }
//         flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
//         // if (cute::thread0()) { print(scores); }

//         // This check is at the end of the loop since we always have at least 1 iteration
//         if (n_masking_steps > 1 && n_block <= n_block_min) {
//             --n_block;
//             break;
//         }
//     }

//     // These are the iterations where we don't need masking on S
//     for (; n_block >= n_block_min; --n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         flash::cp_async_wait<0>();
//         __syncthreads();

//         // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

//         /////// Mask 0 ///////
//         flash::copy_sparse(tVgV(_, _, _, _, n_block), tVsV);
//         ////// Mask 0 ///////

//         // ////// Mask 1 ///////
//         // // tVgV: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
//         // flash::copy_sparse(tVgV(_, _, _, _, _, _, n_block), tVsV);
//         // ////// Mask 1 ///////

//         cute::cp_async_fence();

//         flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
//             acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
//             smem_thr_copy_Q, smem_thr_copy_K
//         );
//         if constexpr (Is_softcap){
//             flash::apply_softcap(acc_s, params.softcap);
//         }

//         flash::cp_async_wait<0>();
//         __syncthreads();
//         if (n_block > n_block_min) {
//             // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
//             // flash::copy_sparse(tKgK(_, _, _, _, n_block-1), tKsK);
            
//             /////// Mask 0 ///////
//             flash::copy_sparse(tKgK(_, _, _, _, n_block-1), tKsK);
//             ////// Mask 0 ///////

//             // ////// Mask 1 ///////
//             // // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
//             // flash::copy_sparse(tKgK(_, _, _, _, _, _, n_block-1), tKsK);
//             // ////// Mask 1 ///////

//             // This cp_async_fence needs to be in the if block, otherwise the synchronization
//             // isn't right and we get race conditions.
//             cute::cp_async_fence();
//         }

//         mask.template apply_mask</*Causal_mask=*/false>(
//             acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
//         );

//         softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

//         Tensor rP = flash::convert_type<Element>(acc_s);
//         int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
//         int block_col_idx = n_block * (kBlockN / 32);
//         if (Return_softmax) {
//             Tensor rP_drop = make_fragment_like(rP);
//             cute::copy(rP, rP_drop);
//             dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
//                 rP_drop, block_row_idx, block_col_idx, kNWarps
//             );
//             cute::copy(rP_drop, tSgS);
//             tSgS.data() = tSgS.data() + (-kBlockN);
//         }
//         if (Is_dropout) {
//             dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
//         }

//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
//         Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
//         flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
//     }

//     // Epilogue

//     Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

//     // Convert acc_o from fp32 to fp16/bf16
//     Tensor rO = flash::convert_type<Element>(acc_o);
//     Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
//     // Partition sO to match the accumulator partitioning
//     auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
//     auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
//     Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
//     Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     // sO has the same size as sQ, so we don't need to sync here.
//     if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

//     cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);


//     ///////////// Org /////////////

//     // Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//     //                                       + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
//     //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
//     //                         make_stride(params.o_row_stride, params.o_head_stride, _1{}));
//     // Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//     //                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)

//     // // Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

//     // // typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
//     // // auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//     // // Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     // // Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

//     // // __syncthreads();

//     // // Tensor tOrO = make_tensor<Element>(shape(tOgO));
//     // // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);


//     ///////////// Sparse /////////////
//     ////////////////// Sparse Mask 0 //////////////////
//     Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     Tensor gO = local_tile(mO(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileMY>>{},
//                         make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));

//     // ////////////////// Sparse Mask 1 //////////////////
//     // Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//     //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
//     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
//     // Tensor gO = local_tile(mO(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<ssparseTileMY>>{},
//     //                     make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

//     typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
//     // typename Kernel_traits::GmemTiledCopyOSparse gmem_tiled_copy_O_Sparse;

//     auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
//     // auto gmem_thr_copy_O_Sparse = gmem_tiled_copy_O_Sparse.get_thread_slice(tidx);

//     Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)

//     // Tensor tOgO_Org = gmem_thr_copy_O.partition_D(gO);
//     Tensor tOgOSparse = gmem_thr_copy_QKV_Sparse.partition_D(gO);

//     __syncthreads();

//     // Tensor tOrO = make_tensor<Element>(shape(tOgO_Org));
//     Tensor tOrO = make_tensor<Element>(shape(tOgOSparse));

//     // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
//     flash::copy_sparse_smem2reg(tOsO, tOrO);

//     // tOgO mask 0: 4096: ((_8,_1),_1,_2,_4):((_1,_0),_0,_8,_16)    1024: ((_8,_1),_1,_1,_8):((_1,_0),_0,_0,_8)
//     // tOgO mask 1: 4096: ((_8,_1),_1,_1,_2,_2,_2):((1,_0),_0,_0,640,40960,163840)

//     // tOsO: 4096: ((_1,_8),_8,_1):((_0,_1),10240,_0)    1024: ((_1,_8),_8,_1):((_1,_0),_1024,_0)

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//     //     // printf("q row: %ld | q head: %ld | o row: %ld | o head: %ld\n", params.q_row_stride, params.q_head_stride, params.o_row_stride, params.o_head_stride);

//     //     Tensor mO_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
//     //                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
//     //                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
//     //     Tensor gO_test = local_tile(mO_test(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<32>, Int<4>>{},
//     //                         make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));  // (kBlockM, kHeadDim)

//     //     using GmemLayoutAtom = Layout<Shape <_8, _16>,
//     //                                 Stride<_1, _8>>;

//     //     using Gmem_copy_struct = std::conditional_t<
//     //         true,
//     //         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
//     //         DefaultCopy
//     //     >;

//     //     TiledCopy tle_copy_test = make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
//     //                         GmemLayoutAtom{},
//     //                         Layout<Shape<_8, _1>>{});  // Val layout, 8 vals per read

//     //     auto gmem_thr_copy_QKV_test = tle_copy_test.get_thread_slice(tidx);

//     //     Tensor tQgO_test = gmem_thr_copy_QKV_test.partition_D(gO_test);
        
//     //     cute::print_tensor(tQgO_test);

//     //     flash::copy_sparse_smem2gmem(tOrO, tQgO_test);

        
//     //     // // ((_1,_8),_2,_4):((_0,640),40960,_16)
//     //     // cute::print_tensor(tOgOSparse);

//     //     // cute::print_tensor(tOrO);

//     //     // flash::copy_sparse_smem2gmem(tOrO, tOgOSparse);

//     //     // cute::print_tensor(tOgOSparse);
        
//     //     // tQsQ_test: ((_8,_1),_8,_1):((_1,_0),_1024,_0)

//     //     // tOsO: ((_1,_8),_8,_1):((_0,_1),_1024,_0)

//     //     // tQgO_test: ((_8,_1),_1,_2,_4):((1,_0),_0,20480,81920)

//     //     // tOgO: ((_1,_8),_8,_1):((_0,_1),10240,_0)

//     //     // tOrO: ((_1,_8),_8,_1):((_0,_1),10240,_0):

//     //     // cute::copy(tOrO(_, 0, 0), tQgO_test(_, 0, 0, 0));
//     //     // flash::copy_sparse_smem2gmem(tOrO, tQgO_test);

//     // }

//     // Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     // Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
//     // static_assert(decltype(size<0>(taccOcO))::value == 4);
//     // // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
//     // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
//     // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
//     // if (get<1>(taccOcO_row(0)) == 0) {
//     //     #pragma unroll
//     //     for (int mi = 0; mi < size(lse); ++mi) {
//     //         const int row = get<0>(taccOcO_row(mi));
//     //         if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
//     //     }
//     // }

//     // // Construct identity layout for sO
//     // Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     // // Repeat the partitioning with identity layouts
//     // Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
//     // Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
//     // if (!Is_even_K) {
//     //     #pragma unroll
//     //     for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
//     // }

//     // // Clear_OOB_K must be false since we don't want to write zeros to gmem
//     // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//     //     gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
//     // );


//     flash::copy_sparse_r2gmem(tOrO, tOgOSparse);
// }

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_mask1(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    constexpr int sparseTileX = Kernel_traits::sparseTileX;
    constexpr int sparseTileMY = Kernel_traits::sparseTileMY;
    constexpr int sparseTileNY = Kernel_traits::sparseTileNY;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    const index_t row_offset_p = ((bidb * params.h + bidh) * params.seqlen_q_rounded
        + m_block * kBlockM) * params.seqlen_k_rounded + (n_block_max - 1) * kBlockN;

    // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
    //                                       + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                         make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    // Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)
    // Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
    //                                       + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
    //                         make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    // Tensor gK = local_tile(mK(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
    //                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)
    // Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
    //                                       + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
    //                         make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    // Tensor gV = local_tile(mV(_, bidh / params.h_h_k_ratio, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
    //                        make_coord(_, 0));  // (kBlockN, kHeadDim, nblocksN)


    ////////////////// Sparse //////////////////
    //////////////// Sparse Mask 1 //////////////////
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
                            make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
    Tensor gQ = local_tile(mQ(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileMY>>{},
                        make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
                            make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
    Tensor gK = local_tile(mK(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileNY>>{},
                        make_coord(0, 0, 0, 0, _, outgroup_id, inside_outgroup_id));

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
                            make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
    Tensor gV = local_tile(mV(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileNY>>{},
                           make_coord(0, 0, 0, 0, _, outgroup_id, inside_outgroup_id));


    Tensor gP = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.p_ptr) + row_offset_p),
                            Shape<Int<kBlockM>, Int<kBlockN>>{},
                            make_stride(params.seqlen_k_rounded, _1{}));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    // Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    // Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    // Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    // Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    // Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    ////////////////// Sparse //////////////////

    typename Kernel_traits::GmemTiledCopyQKVSparse gmem_tiled_copy_QKV_Sparse;
    auto gmem_thr_copy_QKV_Sparse = gmem_tiled_copy_QKV_Sparse.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV_Sparse.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = gmem_thr_copy_QKV_Sparse.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVgV = gmem_thr_copy_QKV_Sparse.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0){
    //     // Tensor mQ_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
    //     //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
    //     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
    //     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    //     // Tensor gQ_test = local_tile(mQ_test(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileMY>>{},
    //     //                     make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

    //     // Tensor mK_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
    //     //                                     + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
    //     //                         make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
    //     //                         make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    //     // Tensor gK_test = local_tile(mK_test(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileNY>>{},
    //     //                     make_coord(0, 0, 0, 0, _, outgroup_id, inside_outgroup_id));

    //     // cute::print_tensor(gQ);

    //     cute::print_tensor(tQgQ);


    //     // ////// Mask 1 for 1024 //////
    //     // using GmemLayoutAtom = Layout<Shape <_8, _8, _2>,
    //     //                             Stride<_1, _8, _64>>;

    //     // using Gmem_copy_struct = std::conditional_t<
    //     //     true,
    //     //     SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    //     //     DefaultCopy
    //     // >;

    //     // TiledCopy tle_copy_test = make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
    //     //                     GmemLayoutAtom{},
    //     //                     Layout<Shape<_8, _1, _1>>{});  // Val layout, 8 vals per read

    //     // auto gmem_thr_copy_QKV_test = tle_copy_test.get_thread_slice(tidx);

    //     // Tensor tQgQ_test = gmem_thr_copy_QKV_test.partition_S(gQ);
    //     // cute::print_tensor(tQgQ_test);
    //     // // cute::print_tensor(tQgQ_test(_, 0, 0, _, _, _));
    //     // ////// Mask 1 for 1024 //////

        

    //     // gQ: 4096: (_64,_16,_2,_2,_2):(1,2560,640,40960,163840)  1024: (_64,_8,_2,_2,_4):(1,2560,640,20480,81920)
    //     // tQgQ: ((_8,_1),_1,_1,_2,_2,_2):((1,_0),_0,_0,640,40960,163840)

    //     // gK: (_64,_16,_2,_2,_2,8):(1,2560,640,40960,163840,327680)
    //     // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
    //     // cute::print_tensor(tQgQ);

    //     // cute::print_tensor(tKgK_test);

    //     // tQgQ: 4096: ((_8,_1),_1,_2,_4):((1,_0),_0,20480,81920)  1024: ((_8,_1),_1,_1,_8):((1,_0),_0,_0,40960)
    //     // tVgV:  ((_8,_1),_1,_2,_4):((1,_0),_0,20480,81920)

    //     // // tQsQ: 4096: ((_8,_1),_8,_1):((_1,_0),_1024,_0)  1024: ((_8,_1),_8,_1):((_1,_0),_1024,_0)
    // }


    ///////////////////////////////////////////////////////////////////////////

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor tSgS  = thr_mma.partition_C(gP);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);
    flash::copy_sparse_mask1(tQgQ, tQsQ);

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    ////// Mask 1 ///////
    // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
    flash::copy_sparse_mask1(tKgK(_, _, _, _, _, _, n_block), tKsK);
    ////// Mask 1 ///////


    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        ////// Mask 1 ///////
        // tVgV: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
        flash::copy_sparse_mask1(tVgV(_, _, _, _, _, _, n_block), tVsV);
        ////// Mask 1 ///////

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse(tKgK(_, _, _, _, n_block - 1), tKsK);

            ////// Mask 1 ///////
            // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
            flash::copy_sparse_mask1(tKgK(_, _, _, _, _, _, n_block - 1), tKsK);
            ////// Mask 1 ///////

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        ////// Mask 1 ///////
        // tVgV: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
        flash::copy_sparse_mask1(tVgV(_, _, _, _, _, _, n_block), tVsV);
        ////// Mask 1 ///////

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse(tKgK(_, _, _, _, n_block-1), tKsK);

            ////// Mask 1 ///////
            // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
            flash::copy_sparse_mask1(tKgK(_, _, _, _, _, _, n_block-1), tKsK);
            ////// Mask 1 ///////

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);
        int block_row_idx = m_block * (kBlockM / 16) + tidx / 32;
        int block_col_idx = n_block * (kBlockN / 32);
        if (Return_softmax) {
            Tensor rP_drop = make_fragment_like(rP);
            cute::copy(rP, rP_drop);
            dropout.template apply_dropout</*encode_dropout_in_sign_bit=*/true>(
                rP_drop, block_row_idx, block_col_idx, kNWarps
            );
            cute::copy(rP_drop, tSgS);
            tSgS.data() = tSgS.data() + (-kBlockN);
        }
        if (Is_dropout) {
            dropout.apply_dropout(rP, block_row_idx, block_col_idx, kNWarps);
        }

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);


    ///////////// Org /////////////

    // Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
    //                                       + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                         make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    // Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                        make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    // // Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    // // typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    // // auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    // // Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
    // // Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    // // __syncthreads();

    // // Tensor tOrO = make_tensor<Element>(shape(tOgO));
    // // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);


    ///////////// Sparse /////////////
    ////////////////// Sparse Mask 1 //////////////////
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
                            make_stride(1, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
    Tensor gO = local_tile(mO(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileMY>>{},
                        make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    // typename Kernel_traits::GmemTiledCopyOSparse gmem_tiled_copy_O_Sparse;

    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    // auto gmem_thr_copy_O_Sparse = gmem_tiled_copy_O_Sparse.get_thread_slice(tidx); // gmem_tiled_copy_QKV_Sparse

    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);        // ((Atom,AtomNum),ATOM_M,ATOM_N)

    // Tensor tOgO_Org = gmem_thr_copy_O.partition_D(gO);
    Tensor tOgOSparse = gmem_thr_copy_QKV_Sparse.partition_D(gO);

    __syncthreads();

    // Tensor tOrO = make_tensor<Element>(shape(tOgO_Org));
    Tensor tOrO = make_tensor<Element>(shape(tOgOSparse));

    // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    flash::copy_sparse_smem2reg_mask1(tOsO, tOrO);

    // tOgO mask 0: 4096: ((_8,_1),_1,_2,_4):((_1,_0),_0,_8,_16)    1024: ((_8,_1),_1,_1,_8):((_1,_0),_0,_0,_8)
    // tOgO mask 1: 4096: ((_8,_1),_1,_1,_2,_2,_2):((1,_0),_0,_0,640,40960,163840)

    // tOsO: 4096: ((_1,_8),_8,_1):((_0,_1),10240,_0)    1024: ((_1,_8),_8,_1):((_1,_0),_1024,_0)

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     // printf("q row: %ld | q head: %ld | o row: %ld | o head: %ld\n", params.q_row_stride, params.q_head_stride, params.o_row_stride, params.o_head_stride);

    //     Tensor mO_test = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
    //                                         + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
    //                             make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
    //                             make_stride(1, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    //     Tensor gO_test = local_tile(mO_test(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<32>, Int<4>>{},
    //                         make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));  // (kBlockM, kHeadDim)

    //     using GmemLayoutAtom = Layout<Shape <_8, _16>,
    //                                 Stride<_1, _8>>;

    //     using Gmem_copy_struct = std::conditional_t<
    //         true,
    //         SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>,
    //         DefaultCopy
    //     >;

    //     TiledCopy tle_copy_test = make_tiled_copy(Copy_Atom<Gmem_copy_struct, cutlass::half_t>{},
    //                         GmemLayoutAtom{},
    //                         Layout<Shape<_8, _1>>{});  // Val layout, 8 vals per read

    //     auto gmem_thr_copy_QKV_test = tle_copy_test.get_thread_slice(tidx);

    //     Tensor tQgO_test = gmem_thr_copy_QKV_test.partition_D(gO_test);
        
    //     cute::print_tensor(tQgO_test);

    //     flash::copy_sparse_smem2gmem(tOrO, tQgO_test);

        
    //     // // ((_1,_8),_2,_4):((_0,640),40960,_16)
    //     // cute::print_tensor(tOgOSparse);

    //     // cute::print_tensor(tOrO);

    //     // flash::copy_sparse_smem2gmem(tOrO, tOgOSparse);

    //     // cute::print_tensor(tOgOSparse);
        
    //     // tQsQ_test: ((_8,_1),_8,_1):((_1,_0),_1024,_0)

    //     // tOsO: ((_1,_8),_8,_1):((_0,_1),_1024,_0)

    //     // tQgO_test: ((_8,_1),_1,_2,_4):((1,_0),_0,20480,81920)

    //     // tOgO: ((_1,_8),_8,_1):((_0,_1),10240,_0)

    //     // tOrO: ((_1,_8),_8,_1):((_0,_1),10240,_0):

    //     // cute::copy(tOrO(_, 0, 0), tQgO_test(_, 0, 0, 0));
    //     // flash::copy_sparse_smem2gmem(tOrO, tQgO_test);

    // }

    // Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
    // static_assert(decltype(size<0>(taccOcO))::value == 4);
    // // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
    // Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    // CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
    // if (get<1>(taccOcO_row(0)) == 0) {
    //     #pragma unroll
    //     for (int mi = 0; mi < size(lse); ++mi) {
    //         const int row = get<0>(taccOcO_row(mi));
    //         if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
    //     }
    // }

    // // Construct identity layout for sO
    // Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    // // Repeat the partitioning with identity layouts
    // Tensor tOcO = gmem_thr_copy_O.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    // Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    // if (!Is_even_K) {
    //     #pragma unroll
    //     for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    // }

    // // Clear_OOB_K must be false since we don't want to write zeros to gmem
    // flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    //     gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    // );


    flash::copy_sparse_r2gmem_mask1(tOrO, tOgOSparse);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_mask0_text_front_text(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    constexpr int sparseTileX = Kernel_traits::sparseTileX;
    constexpr int sparseTileMY = Kernel_traits::sparseTileMY;
    constexpr int sparseTileNY = Kernel_traits::sparseTileNY;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max_text = cute::ceil_div(params.text_length, kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN) - n_block_max_text;
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(_4{}, binfo.actual_seqlen_q/4, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_row_stride*4, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(outgroup_id*2+inside_outgroup_id, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));
    
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
                                            + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) + params.text_length*params.k_row_stride),
                            make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
                            make_stride(_1{}, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    Tensor gK = local_tile(mK(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileNY>>{},
                        make_coord(0, 0, _, outgroup_id, inside_outgroup_id));

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                            + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) + params.text_length*params.v_row_stride),
                            make_shape(params.d, binfo.width/2, binfo.length/2, 2, 2, params.h),
                            make_stride(_1{}, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    Tensor gV = local_tile(mV(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileNY>>{},
                            make_coord(0, 0, _, outgroup_id, inside_outgroup_id));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopyQKVSparse gmem_tiled_copy_QKV_Sparse;
    auto gmem_thr_copy_QKV_Sparse = gmem_tiled_copy_QKV_Sparse.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = gmem_thr_copy_QKV_Sparse.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVgV = gmem_thr_copy_QKV_Sparse.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy_org_cleaned_flux(tQgQ, tQsQ);
    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);
    /////// Mask 0 ///////
    flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block), tKsK);
    ////// Mask 0 ///////
    
    cute::cp_async_fence();


    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        /////// Mask 0 ///////
        flash::copy_sparse_flux_mask0(tVgV(_, _, _, _, n_block), tVsV);
        ////// Mask 0 ///////

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            /////// Mask 0 ///////
            flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);
            ////// Mask 0 ///////

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        /////// Mask 0 ///////
        flash::copy_sparse_flux_mask0(tVgV(_, _, _, _, n_block), tVsV);
        ////// Mask 0 ///////

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // if (m_block == 0 && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && n_block == 0) {
        //     cute::print_tensor(tQsQ);
        //     cute::print_tensor(tKsK);
        //     cute::print_tensor(acc_s);
        // }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
            
            /////// Mask 0 ///////
            flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
            ////// Mask 0 ///////

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }


    // reshape the mK and mV for text tokens
    Tensor mK_text = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));

    Tensor gK_text = local_tile(mK_text(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    Tensor mV_text = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV_text = local_tile(mV_text(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // tKgK_text / tVgV_text: ((_8,_1),_8,_1, text_blocks)
    Tensor tKgK_text = gmem_thr_copy_QKV.partition_S(gK_text);
    Tensor tVgV_text = gmem_thr_copy_QKV.partition_D(gV_text);

    // Text loop
    int n_block_text = n_block_max_text - 1;
    // ((_8,_1),_2,_2,48):((_1,_0),20480,_64,40960)
    flash::copy_org_cleaned_flux(tKgK_text(_, _, _, n_block_text), tKsK);
    cute::cp_async_fence();

    for (; n_block_text >= n_block_min; --n_block_text) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        flash::copy_org_cleaned_flux(tVgV_text(_, _, _, n_block_text), tVsV);

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block_text > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            
            flash::copy_org_cleaned_flux(tKgK_text(_, _, _, n_block_text-1), tKsK);

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(4, binfo.actual_seqlen_q/4, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_row_stride*4, params.q_head_stride, _1{}));
    Tensor gO = local_tile(mO(outgroup_id*2+inside_outgroup_id, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));

    // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    cute::copy(tOsO, tOrO);

    // ((_1,_8),_8,_2):((_0,_1),_8,_64)
    flash::copy_org_cleaned_flux(tOrO, tOgO);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_mask0_text_front_image(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    constexpr int sparseTileX = Kernel_traits::sparseTileX;
    constexpr int sparseTileMY = Kernel_traits::sparseTileMY;
    constexpr int sparseTileNY = Kernel_traits::sparseTileNY;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
    int n_block_max_text = cute::ceil_div(params.text_length, kBlockN);
    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN) - n_block_max_text;
    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    //  + params.text_length * params.q_row_stride

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + params.text_length * params.q_row_stride),
                            make_shape(params.d, binfo.width/2, binfo.length/2, _2{}, _2{}, params.h),
                            make_stride(_1{}, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    Tensor gQ = local_tile(mQ(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileMY>>{},
                        make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));
    
    Tensor mK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr) + 
                                            + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb) + params.text_length * params.k_row_stride),
                            make_shape(params.d, binfo.width/2, binfo.length/2, _2{}, _2{}, params.h),
                            make_stride(_1{}, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    Tensor gK = local_tile(mK(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileNY>>{},
                        make_coord(0, 0, _, outgroup_id, inside_outgroup_id));

    Tensor mV = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                            + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb) + params.text_length * params.v_row_stride),
                            make_shape(params.d, binfo.width/2, binfo.length/2, _2{}, _2{}, params.h),
                            make_stride(_1{}, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    Tensor gV = local_tile(mV(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileNY>>{},
                            make_coord(0, 0, _, outgroup_id, inside_outgroup_id));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopyQKVSparse gmem_tiled_copy_QKV_Sparse;
    auto gmem_thr_copy_QKV_Sparse = gmem_tiled_copy_QKV_Sparse.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV_Sparse.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKgK = gmem_thr_copy_QKV_Sparse.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVgV = gmem_thr_copy_QKV_Sparse.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    ///////////////////////////////////////////////////////////////////////////

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);
    // flash::copy_sparse(tQgQ, tQsQ);
    flash::copy_sparse_flux_o(tQgQ, tQsQ);

    // if (m_block == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     // ((_8,_1),_2,_2,_4):((_1,_0),_64,40960,163840)
    //     cute::print_tensor(tQgQ);

    //     // ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     cute::print_tensor(tQsQ);

    //     // ((_8,_1),_2,_2,_1):((_1,_0),_64,40960,_0)
    //     cute::print_tensor(tKgK(_, _, _, _, 31));

    //     // ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
    //     cute::print_tensor(tKsK);
    // }

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);
    /////// Mask 0 ///////
    flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block), tKsK);
    ////// Mask 0 ///////

    // ////// Mask 1 ///////
    // // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
    // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, _, _, n_block), tKsK);
    // ////// Mask 1 ///////

    
    cute::cp_async_fence();
    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        /////// Mask 0 ///////
        flash::copy_sparse_flux_mask0(tVgV(_, _, _, _, n_block), tVsV);
        ////// Mask 0 ///////

        // ////// Mask 1 ///////
        // // tVgV: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
        // flash::copy_sparse_flux_mask0(tVgV(_, _, _, _, _, _, n_block), tVsV);
        // ////// Mask 1 ///////

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            /////// Mask 0 ///////
            flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);
            ////// Mask 0 ///////

            // ////// Mask 1 ///////
            // // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, _, _, n_block - 1), tKsK);
            // ////// Mask 1 ///////

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        /////// Mask 0 ///////
        flash::copy_sparse_flux_mask0(tVgV(_, _, _, _, n_block), tVsV);
        ////// Mask 0 ///////

        // ////// Mask 1 ///////
        // // tVgV: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
        // flash::copy_sparse_flux_mask0(tVgV(_, _, _, _, _, _, n_block), tVsV);
        // ////// Mask 1 ///////

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // if (m_block == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.x == 0 && n_block == 0) {
        //     cute::print_tensor(tQgQ);
        //     cute::print_tensor(tSsQ);
        //     // cute::print_tensor(tSrQ);
            
        //     cute::print_tensor(acc_s);
        // }

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse(tKgK(_, _, _, _, n_block-1), tKsK);
            
            /////// Mask 0 ///////
            flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
            ////// Mask 0 ///////

            // ////// Mask 1 ///////
            // // tKgK: ((_8,_1),_1,_1,_2,_2,_2, 8):((1,_0),_0,_0,640,40960,163840, 327680)
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, _, _, n_block-1), tKsK);
            // ////// Mask 1 ///////

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }


    // reshape the mK and mV for text tokens
    Tensor mK_text = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.k_ptr)
                                          + binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.k_row_stride, params.k_head_stride, _1{}));
    Tensor gK_text = local_tile(mK_text(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    Tensor mV_text = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.v_ptr)
                                          + binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb)),
                            make_shape(binfo.actual_seqlen_k, params.h_k, params.d),
                            make_stride(params.v_row_stride, params.v_head_stride, _1{}));
    Tensor gV_text = local_tile(mV_text(_, bidh, _), Shape<Int<kBlockN>, Int<kHeadDim>>{},
                           make_coord(_, 0));

    // tKgK_text / tVgV_text: ((_8,_1),_8,_1, text_blocks)
    Tensor tKgK_text = gmem_thr_copy_QKV.partition_S(gK_text);
    Tensor tVgV_text = gmem_thr_copy_QKV.partition_D(gV_text);

    // Text loop
    int n_block_text = n_block_max_text - 1;
    flash::copy_org_cleaned_flux(tKgK_text(_, _, _, n_block_text), tKsK);
    cute::cp_async_fence();
    for (; n_block_text >= n_block_min; --n_block_text) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        flash::copy_org_cleaned_flux(tVgV_text(_, _, _, n_block_text), tVsV);

        cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block_text > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            
            flash::copy_org_cleaned_flux(tKgK_text(_, _, _, n_block_text-1), tKsK);

            // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // isn't right and we get race conditions.
            cute::cp_async_fence();
        }

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    ///////////// Sparse /////////////
    ////////////////// Sparse Mask 0 //////////////////
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + params.text_length*params.q_row_stride),
                            make_shape(params.d, binfo.width/2, binfo.length/2, _2{}, _2{}, params.h),
                            make_stride(_1{}, 2*params.d*params.h, 2*binfo.width*params.d*params.h, binfo.width*params.d*params.h, params.d*params.h, params.d));
    Tensor gO = local_tile(mO(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<sparseTileMY>>{},
                        make_coord(0, 0, m_block, outgroup_id, inside_outgroup_id));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    // typename Kernel_traits::GmemTiledCopyOSparse gmem_tiled_copy_O_Sparse;

    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    // auto gmem_thr_copy_O_Sparse = gmem_tiled_copy_O_Sparse.get_thread_slice(tidx);

    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

    // Tensor tOgO_Org = gmem_thr_copy_O.partition_D(gO);
    Tensor tOgOSparse = gmem_thr_copy_QKV_Sparse.partition_D(gO);

    __syncthreads();

    // Tensor tOrO = make_tensor<Element>(shape(tOgO_Org));
    Tensor tOrO = make_tensor<Element>(shape(tOgOSparse));

    // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    flash::copy_sparse_smem2reg_flux_mask0(tOsO, tOrO);

    flash::copy_sparse_r2gmem_flux_mask0(tOrO, tOgOSparse);
}


////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_mask1_text_front_text(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    constexpr int sparseTileX = Kernel_traits::sparseTileX;
    constexpr int sparseTileMY = Kernel_traits::sparseTileMY;
    constexpr int sparseTileNY = Kernel_traits::sparseTileNY;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);


    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        outgroup_id, inside_outgroup_id, 16, bidh
    );

    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
        outgroup_id, inside_outgroup_id, 16, bidh
    );


    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);


    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(_4{}, binfo.actual_seqlen_q/4, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_row_stride*4, params.q_head_stride, _1{}));
    Tensor gQ = local_tile(mQ(outgroup_id*2+inside_outgroup_id, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopyQKVSparse gmem_tiled_copy_QKV_Sparse;
    auto gmem_thr_copy_QKV_Sparse = gmem_tiled_copy_QKV_Sparse.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    // Tensor tKgK = gmem_thr_copy_QKV_Sparse.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    // Tensor tVgV = gmem_thr_copy_QKV_Sparse.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     // gQ: (_128,_128):(5120,_1)
    //     // cute::print_tensor(gQ);

    //     // tQgQ: ((_8,_1),_8,_2):((_1,_0),81920,_64)
    //     // cute::print_tensor(tQgQ);

    //     // tQsQ: ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     // cute::print_tensor(tQsQ);


    //     // (128,32,32,2,2,10):(_1,2560,163840,81920,1280,128)
    //     // cute::print_tensor(mK);

    //     // (_128,_32,_1,32):(_1,2560,_0,163840)
    //     // cute::print_tensor(gK);

    //     // ((_8,_1),_2,_2,_1,32):((_1,_0),_64,20480,_0,81920)
    //     // cute::print_tensor(tKgK);
        
    //     // ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
    //     // cute::print_tensor(tKsK);
    // }

    // if (m_block == 0 && threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {



    //     gmem_tile_k.load(17, tKsK);

    //     cute::print_tensor(tKsK);

    //     // cute::custom_copy(reinterpret_cast<char *>(params.k_ptr), tKsK, 0);
    //     // cute::custom_copy(params.k_ptr, tKsK, 0);

    //     // flash::copy_org_cleaned_flux(tQgQ, tQsQ);

    //     // Tensor tOrO = make_tensor<Element>(shape(tQgQ));

    //     // // ((_8,_1),_8,_2)
    //     // cute::copy(tQsQ, tOrO);

    //     // tQsQ: ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     // cute::custom_copy(params.q_ptr, tQsQ, 0);

    //     // flash::copy_org_cleaned_flux(tOrO, tQgQ);
    // }

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy_org_cleaned_flux(tQgQ, tQsQ);

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    gmem_tile_k.load(n_block, tKsK);


    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
                            make_shape(4, binfo.actual_seqlen_q/4, params.h, params.d),
                            make_stride(params.q_row_stride, params.q_row_stride*4, params.q_head_stride, _1{}));
    Tensor gO = local_tile(mO(outgroup_id*2+inside_outgroup_id, _, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                        make_coord(m_block, 0));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));

    // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    cute::copy(tOsO, tOrO);

    // ((_1,_8),_8,_2):((_0,_1),_8,_64)
    flash::copy_org_cleaned_flux(tOrO, tOgO);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_mask1_text_front_image(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    constexpr int sparseTileX = Kernel_traits::sparseTileX;
    constexpr int sparseTileMY = Kernel_traits::sparseTileMY;
    constexpr int sparseTileNY = Kernel_traits::sparseTileNY;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);


    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        outgroup_id, inside_outgroup_id, 16, bidh
    );

    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
        outgroup_id, inside_outgroup_id, 16, bidh
    );

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);



    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + params.text_length * params.q_row_stride),
                            make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
                            make_stride(_1{}, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
    Tensor gQ = local_tile(mQ(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileMY>>{},
                        make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    typename Kernel_traits::GmemTiledCopyQKVSparse gmem_tiled_copy_QKV_Sparse;
    auto gmem_thr_copy_QKV_Sparse = gmem_tiled_copy_QKV_Sparse.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV_Sparse.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    // Tensor tKgK = gmem_thr_copy_QKV_Sparse.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K, nblocksN)
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    // Tensor tVgV = gmem_thr_copy_QKV_Sparse.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K, nblocksN)
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    flash::copy_sparse_mask1_flux(tQgQ, tQsQ);

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    gmem_tile_k.load(n_block, tKsK);


    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
    
        // cute::cp_async_fence();


        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);


    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                        + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb) + params.text_length * params.q_row_stride),
                            make_shape(params.d, binfo.width/4, 2, 2, binfo.length/4, 2, 2, params.h),
                            make_stride(_1{}, 4*params.d*params.h, params.d*params.h, binfo.width*params.d*params.h, 4*binfo.width*params.d*params.h, 2*binfo.width*params.d*params.h, 2*params.d*params.h, params.d));
    Tensor gO = local_tile(mO(_, _, _, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<sparseTileX>, Int<2>, Int<2>, Int<sparseTileMY>>{},
                        make_coord(0, 0, 0, 0, m_block, outgroup_id, inside_outgroup_id));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    // Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    Tensor tOgO = gmem_thr_copy_QKV_Sparse.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));

    // // cute::copy(gmem_tiled_copy_O, tOsO, tOrO);
    flash::copy_sparse_smem2reg_mask1_flux(tOsO, tOrO);

    // // ((_1,_8),_8,_2):((_0,_1),_8,_64)
    flash::copy_sparse_r2gmem_mask1_flux(tOrO, tOgO);
}


////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_masksw_text_front_text(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q_sw;
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o_sw;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k_8group;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v_8group;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    // binfo.actual_seqlen_k = 1024;
    // Text queries keep their Scatter rows: all text keys + the image tokens of their group.
    const int text_seqlen_k = params.text_length + params.width * params.length / 8;
    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + text_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    int n_block_max = cute::ceil_div(text_seqlen_k, kBlockN);


    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        outgroup_id, inside_outgroup_id, 8, bidh
    );

    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
        outgroup_id, inside_outgroup_id, 8, bidh
    );

    Gmem_tile_q gmem_tile_q(params.q_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        8, m_block, bidh
    );


    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    // if (blockIdx.x == 1 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {

    //     cute::print_tensor(tQsQ);

    //     gmem_tile_q.load(m_block, tQsQ);
        
    //     cute::print_tensor(tQsQ);

    //     printf("n_block_max: %d\n", n_block_max);


    //     // gQ: (_128,_128):(5120,_1)
    //     // cute::print_tensor(gQ);

    //     // tQgQ: ((_8,_1),_8,_2):((_1,_0),81920,_64)
    //     // cute::print_tensor(tQgQ);

    //     // tQsQ: ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     // cute::print_tensor(tQsQ);


    //     // (128,32,32,2,2,10):(_1,2560,163840,81920,1280,128)
    //     // cute::print_tensor(mK);

    //     // (_128,_32,_1,32):(_1,2560,_0,163840)
    //     // cute::print_tensor(gK);

    //     // ((_8,_1),_2,_2,_1,32):((_1,_0),_64,20480,_0,81920)
    //     // cute::print_tensor(tKgK);
        
    //     // ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
    //     // cute::print_tensor(tKsK);
    // }

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    // flash::copy_org_cleaned_flux(tQgQ, tQsQ);
    gmem_tile_q.load(m_block, tQsQ);

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    gmem_tile_k.load(n_block, tKsK);


    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(text_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Gmem_tile_o gmem_tile_o(params.o_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        8, m_block, bidh
    );

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(make_shape(make_shape(_1{}, _8{}), _4{}, _2{}));

    cute::copy(tOsO, tOrO);

    gmem_tile_o.store(m_block, tOrO);
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_masksw_text_front_image(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q_sw;
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o_sw;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k_sw;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v_sw;

    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        8, 8, m_block, bidh
    );

    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
        8, 8, m_block, bidh
    );

    Gmem_tile_q gmem_tile_q(params.q_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        8, m_block, bidh
    );


    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    // if (m_block == 8 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {

    //     Gmem_tile_q gmem_tile_q(params.q_ptr, params, tidx,
    //         binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
    //         8, m_block, bidh
    //     );

    //     Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
    //         binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
    //         8, 8, m_block, bidh
    //     );

    //     Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
    //         binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
    //         8, 8, m_block, bidh
    //     );


    //     gmem_tile_q.load(m_block, tQsQ);
        
    //     cute::print_tensor(tQsQ);

    //     gmem_tile_k.load(8, tKsK);

    //     cute::print_tensor(tKsK);


    //     // gQ: (_128,_128):(5120,_1)
    //     // cute::print_tensor(gQ);

    //     // tQgQ: ((_8,_1),_8,_2):((_1,_0),81920,_64)
    //     // cute::print_tensor(tQgQ);

    //     // tQsQ: ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     // cute::print_tensor(tQsQ);


    //     // (128,32,32,2,2,10):(_1,2560,163840,81920,1280,128)
    //     // cute::print_tensor(mK);

    //     // (_128,_32,_1,32):(_1,2560,_0,163840)
    //     // cute::print_tensor(gK);

    //     // ((_8,_1),_2,_2,_1,32):((_1,_0),_64,20480,_0,81920)
    //     // cute::print_tensor(tKgK);
        
    //     // ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
    //     // cute::print_tensor(tKsK);
    // }

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    // flash::copy_org_cleaned_flux(tQgQ, tQsQ);
    gmem_tile_q.load(m_block, tQsQ);

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);
    gmem_tile_k.load(n_block, tKsK);


    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Gmem_tile_o gmem_tile_o(params.o_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        8, m_block, bidh
    );

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(make_shape(make_shape(_1{}, _8{}), _4{}, _2{}));

    cute::copy(tOsO, tOrO);

    gmem_tile_o.store(m_block, tOrO);
}


////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_mask0_8group_text_front(const Params &params, const int bidb, const int bidh, const int m_block, const int outgroup_id, const int inside_outgroup_id) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q_8group;
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o_8group;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k_8group;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v_8group;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        outgroup_id, inside_outgroup_id, 8, bidh
    );

    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
        outgroup_id, inside_outgroup_id, 8, bidh
    );

    Gmem_tile_q gmem_tile_q(params.q_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        outgroup_id, inside_outgroup_id, 1, bidh
    );


    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }
    // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // Otherwise we might read OOB elements from gK and gV.
    if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
                                              + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
                                make_shape(binfo.actual_seqlen_q, params.h, params.d),
                                make_stride(params.o_row_stride, params.o_head_stride, _1{}));
        Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
                              make_coord(m_block, 0));  // (kBlockM, kHeadDim)

        Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        // Construct identity layout for sO
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
        // Repeat the partitioning with identity layouts
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        // Clear_OOB_K must be false since we don't want to write zeros to gmem
        flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
        );
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO(0, m, 0));
            if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }
    // if (tidx == 0) { printf("m_block = %d, n_block_min = %d, n_block_max = %d\n", m_block, n_block_min, n_block_max); }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
    //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                         make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    // Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                     make_coord(m_block, 0));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        
    //     cute::print_tensor(tKsK);

    //     gmem_tile_k.load(8, tKsK);

    //     cute::print_tensor(tKsK);

    //     printf("n_block_max: %d\n", n_block_max);


    //     // gQ: (_128,_128):(5120,_1)
    //     // cute::print_tensor(gQ);

    //     // tQgQ: ((_8,_1),_8,_2):((_1,_0),81920,_64)
    //     // cute::print_tensor(tQgQ);

    //     // tQsQ: ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     // cute::print_tensor(tQsQ);


    //     // (128,32,32,2,2,10):(_1,2560,163840,81920,1280,128)
    //     // cute::print_tensor(mK);

    //     // (_128,_32,_1,32):(_1,2560,_0,163840)
    //     // cute::print_tensor(gK);

    //     // ((_8,_1),_2,_2,_1,32):((_1,_0),_64,20480,_0,81920)
    //     // cute::print_tensor(tKgK);
        
    //     // ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
    //     // cute::print_tensor(tKsK);
    // }

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    // flash::copy_org_cleaned_flux(tQgQ, tQsQ);
    gmem_tile_q.load(m_block, tQsQ);

    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    gmem_tile_k.load(n_block, tKsK);


    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Gmem_tile_o gmem_tile_o(params.o_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        outgroup_id, inside_outgroup_id, 1, bidh
    );

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(make_shape(make_shape(_1{}, _8{}), _4{}, _2{}));

    cute::copy(tOsO, tOrO);

    gmem_tile_o.store(m_block, tOrO);
}


////////////////////////////////////////////////////////////////////////////////////////////////////


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sparse_16group_llm(const Params &params, const int bidb, const int bidh, const int group_id, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo_llm</*Varlen=*/!Is_even_MN> binfo(params, bidb, group_id, Kernel_traits::maskType);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        group_id, binfo.actual_seqlen_k, bidh / params.h_h_k_ratio
    );

    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb), 
        group_id, binfo.actual_seqlen_k, bidh / params.h_h_k_ratio
    );

    Gmem_tile_q gmem_tile_q(params.q_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        group_id, binfo.actual_seqlen_q, bidh
    );


    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
        // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
        //     printf("m_block = %d, n_block_max = %d\n", m_block, n_block_max);
        // }
    }

    // // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // // Otherwise we might read OOB elements from gK and gV.
    // if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
    //     Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
    //                                           + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
    //                             make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                             make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    //     Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    //     Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    //     typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    //     auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    //     Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    //     Tensor tOrO = make_tensor<Element>(shape(tOgO));
    //     clear(tOrO);
    //     // Construct identity layout for sO
    //     Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    //     // Repeat the partitioning with identity layouts
    //     Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    //     Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    //     if (!Is_even_K) {
    //         #pragma unroll
    //         for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    //     }
    //     // Clear_OOB_K must be false since we don't want to write zeros to gmem
    //     flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    //         gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    //     );
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(tOgO); ++m) {
    //         const int row = get<0>(tOcO(0, m, 0));
    //         if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
    //     }
    //     return;
    // }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
    //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                         make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    // Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                     make_coord(m_block, 0));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    // flash::copy_org_cleaned_flux(tQgQ, tQsQ);
    gmem_tile_q.load(m_block, tQsQ);
    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    gmem_tile_k.load(n_block, tKsK);

    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

    // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     printf(
    //         "n_masking_steps = %d | Is_causal: %d | Is_even_MN: %d | ceil_div: %d | n_block: %d | n_block_min: %d\n",
    //         n_masking_steps, Is_causal, Is_even_MN, cute::ceil_div(kBlockM, kBlockN), n_block, n_block_min
    //     );
    // }

    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // if(blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        //     cute::print_tensor(acc_s);
        // }

        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Gmem_tile_o gmem_tile_o(params.o_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        group_id, binfo.actual_seqlen_q, bidh
    );

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(make_shape(make_shape(_1{}, _8{}), _4{}, _2{}));

    cute::copy(tOsO, tOrO);

    gmem_tile_o.store(m_block, tOrO);
}




template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_1rowblock_sw_llm(const Params &params, const int bidb, const int bidh, const int group_id, const int m_block, const int start) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    using Gmem_tile_q = typename Kernel_traits::Gmem_tile_q;
    using Gmem_tile_o = typename Kernel_traits::Gmem_tile_o;
    using Gmem_tile_k = typename Kernel_traits::Gmem_tile_k;
    using Gmem_tile_v = typename Kernel_traits::Gmem_tile_v;


    // Shared memory.
    extern __shared__ char smem_[];

    // The thread index.
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    auto seed_offset = at::cuda::philox::unpack(params.philox_args);
    flash::Dropout dropout(std::get<0>(seed_offset), std::get<1>(seed_offset), params.p_dropout_in_uint8_t,
                           bidb, bidh, tidx, params.h);

    // Save seed and offset for backward, before any early exiting. Otherwise the 0-th thread block might
    // exit early and no one saves the rng states.
    if (Is_dropout && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
        params.rng_state[0] = std::get<0>(seed_offset);
        params.rng_state[1] = std::get<1>(seed_offset);
    }

    const BlockInfo_llm</*Varlen=*/!Is_even_MN> binfo(params, bidb, group_id, Kernel_traits::maskType);
    if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

    const int n_block_min = !Is_local ? 0 : std::max(0, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);

    int n_block_max = cute::ceil_div(binfo.actual_seqlen_k, kBlockN);

    Gmem_tile_k gmem_tile_k(params.k_ptr, params, tidx,
        binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb),
        start, binfo.org_seqlen_k, bidh / params.h_h_k_ratio
    );

    // INTRA fix (Bug A2): Gmem_tile_kv_llm_sw's 5th ctor argument is the window
    // START offset (as passed to gmem_tile_k above), not a group id.  Passing
    // group_id made V be read from token `group_id` (0..7) instead of `start`,
    // so every head that was computed used the wrong values.
    Gmem_tile_v gmem_tile_v(params.v_ptr, params, tidx,
        binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb),
        start, binfo.org_seqlen_k, bidh / params.h_h_k_ratio
    );

    Gmem_tile_q gmem_tile_q(params.q_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        m_block, binfo.actual_seqlen_q, bidh
    );

    if (Is_causal || Is_local) {
        n_block_max = std::min(n_block_max,
                               cute::ceil_div((m_block + 1) * kBlockM + binfo.org_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
    }

    // // We exit early and write 0 to gO and gLSE. This also covers the case where actual_seqlen_k == 0.
    // // Otherwise we might read OOB elements from gK and gV.
    // if ((Is_causal || Is_local || !Is_even_MN) && n_block_max <= n_block_min) {
    //     Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr)
    //                                           + binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)),
    //                             make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                             make_stride(params.o_row_stride, params.o_head_stride, _1{}));
    //     Tensor gO = local_tile(mO(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                           make_coord(m_block, 0));  // (kBlockM, kHeadDim)

    //     Tensor gLSE = get_lse_tile<ElementAccum, Params, kBlockM, Is_even_MN>(params, bidb, bidh, m_block, binfo);

    //     typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    //     auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    //     Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
    //     Tensor tOrO = make_tensor<Element>(shape(tOgO));
    //     clear(tOrO);
    //     // Construct identity layout for sO
    //     Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    //     // Repeat the partitioning with identity layouts
    //     Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    //     Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    //     if (!Is_even_K) {
    //         #pragma unroll
    //         for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    //     }
    //     // Clear_OOB_K must be false since we don't want to write zeros to gmem
    //     flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
    //         gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
    //     );
    //     #pragma unroll
    //     for (int m = 0; m < size<1>(tOgO); ++m) {
    //         const int row = get<0>(tOcO(0, m, 0));
    //         if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
    //     }
    //     return;
    // }

    // We iterate over the blocks in reverse order. This is because the last block is the only one
    // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
    // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

    // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
    //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
    //                         make_shape(binfo.actual_seqlen_q, params.h, params.d),
    //                         make_stride(params.q_row_stride, params.q_head_stride, _1{}));
    // Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
    //                     make_coord(m_block, 0));

    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    // Careful we're using the same smem for sQ and sK | sV if Share_Q_K_smem;
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    // Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);

    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);

    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
    Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

    //
    // Copy Atom retiling
    //

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    // if (cute::thread0()) {smem_thr_copy_Q.print_all();}
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    // if (cute::thread0()) {print(tSsQ.layout()); printf("\n");}

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    //
    // PREDICATES
    //

    // // Allocate predicate tensors for m and n
    // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
    // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

    // Construct identity layout for sQ and sK
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)
    // Tensor tScQ = thr_mma.partition_A(cQ);                           // (MMA,MMA_M,MMA_K)
    // if (cute::thread0()) {
    //     print(tScQ.layout()); printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<0>(tScQ(i)));
    //     }
    //     printf("\n");
    //     for (int i = 0; i < size(tScQ); ++i) {
    //         printf("%d ", get<1>(tScQ(i)));
    //     }
    //     printf("\n");
    // }

    // Repeat the partitioning with identity layouts
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

    // Allocate predicate tensors for k
    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

    // Set predicates for k bounds
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // Prologue

    // // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
    //                                    binfo.actual_seqlen_q - m_block * kBlockM);

    // flash::copy_org_cleaned_flux(tQgQ, tQsQ);
    gmem_tile_q.load(m_block, tQsQ);
    
    // // if (cute::thread(1, 0)) { print(tQsQ); }
    // // Tensor sQNoSwizzle = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)), typename Kernel_traits::SmemLayoutQNoSwizzle{});
    // // if (cute::thread0()) { print(sQNoSwizzle); }

    if (Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    int n_block = n_block_max - 1;
    // // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
    // flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block), tKsK, tKVcKV, tKVpKV,
    //                                    binfo.actual_seqlen_k - n_block * kBlockN);

    gmem_tile_k.load(n_block, tKsK);

    // if (blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     printf("m_block: %d | n_block_max: %d | start: %d\n", m_block, n_block_max, start);
    // }

    // if (blockIdx.x == 9 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     cute::print_tensor(tKsK);

    //     cute::print_tensor(tQsQ);

    //     // gmem_tile_k.load(8, tKsK);

    //     // cute::print_tensor(tKsK);

    //     // printf("n_block_max: %d\n", n_block_max);

    //     // gQ: (_128,_128):(5120,_1)
    //     // cute::print_tensor(gQ);

    //     // tQgQ: ((_8,_1),_8,_2):((_1,_0),81920,_64)
    //     // cute::print_tensor(tQgQ);

    //     // tQsQ: ((_8,_1),_8,_2):((_1,_0),_1024,_8192)
    //     // cute::print_tensor(tQsQ);


    //     // (128,32,32,2,2,10):(_1,2560,163840,81920,1280,128)
    //     // cute::print_tensor(mK);

    //     // (_128,_32,_1,32):(_1,2560,_0,163840)
    //     // cute::print_tensor(gK);

    //     // ((_8,_1),_2,_2,_1,32):((_1,_0),_64,20480,_0,81920)
    //     // cute::print_tensor(tKgK);
        
    //     // ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
    //     // cute::print_tensor(tKsK);
    // }

    // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z < 2) { print(tKgK); }
    // __syncthreads();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        flash::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));            // M
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    flash::Softmax<2 * size<1>(acc_o)> softmax;

    const float alibi_slope = !Has_alibi || params.alibi_slopes_ptr == nullptr ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
    flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.org_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

    // For performance reason, we separate out two kinds of iterations:
    // those that need masking on S, and those that don't.
    // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
    // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
    // We will have at least 1 "masking" iteration.

    // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
    // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
    constexpr int n_masking_steps = (!Is_causal && !Is_local)
        ? 1
        : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);    
    #pragma unroll
    for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        gmem_tile_v.load(n_block, tVsV);
        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );
        // if (cute::thread0()) { print(acc_s); }
        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        mask.template apply_mask<Is_causal, Is_even_MN>(
            acc_s, start + n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block - 1), tKsK);

            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        // TODO: when we have key_padding_mask we'll need to Check_inf
        masking_step == 0
            ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2)
            : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        // Convert acc_s from fp32 to fp16/bf16
        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        // if (cute::thread0()) { print(tOrP); }
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
        // if (cute::thread0()) { print(scores); }

        // This check is at the end of the loop since we always have at least 1 iteration
        if (n_masking_steps > 1 && n_block <= n_block_min) {
            --n_block;
            break;
        }
    }

    // These are the iterations where we don't need masking on S
    for (; n_block >= n_block_min; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
        clear(acc_s);
        flash::cp_async_wait<0>();
        __syncthreads();

        // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV(_, _, _, n_block), tVsV, tKVcKV, tKVpKV);

        gmem_tile_v.load(n_block, tVsV);

        // cute::cp_async_fence();

        flash::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        if constexpr (Is_softcap){
            flash::apply_softcap(acc_s, params.softcap);
        }

        flash::cp_async_wait<0>();
        __syncthreads();
        if (n_block > n_block_min) {
            // // flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK(_, _, _, n_block - 1), tKsK, tKVcKV, tKVpKV);
            // flash::copy_sparse_flux_mask0(tKgK(_, _, _, _, n_block-1), tKsK);
        
            gmem_tile_k.load(n_block-1, tKsK);

            // // This cp_async_fence needs to be in the if block, otherwise the synchronization
            // // isn't right and we get race conditions.
            // cute::cp_async_fence();
        }

        mask.template apply_mask</*Causal_mask=*/false>(
            acc_s, start + n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
        );

        softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

        Tensor rP = flash::convert_type<Element>(acc_s);

        // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
        // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
        Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));
        flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

    }

    __syncthreads();

    // Epilogue

    Tensor lse = softmax.template normalize_softmax_lse<Is_dropout>(acc_o, params.scale_softmax, params.rp_dropout);

    // Convert acc_o from fp32 to fp16/bf16
    Tensor rO = flash::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});    // (SMEM_M,SMEM_N)
    // Partition sO to match the accumulator partitioning
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

    // sO has the same size as sQ, so we don't need to sync here.
    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    Gmem_tile_o gmem_tile_o(params.o_ptr, params, tidx,
        binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb),
        m_block, binfo.actual_seqlen_q, bidh
    );

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(make_shape(make_shape(_1{}, _8{}), _4{}, _2{}));

    // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
    //     cute::print_tensor(tOsO);
    // }

    cute::copy(tOsO, tOrO);

    cute::cp_async_fence();

    gmem_tile_o.store(m_block, tOrO);
}


////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
// inline __device__ void compute_attn_1rowblock_splitkv(const Params &params, const int bidb, const int bidh, const int m_block, const int n_split_idx, const int num_n_splits) {

//     using Element = typename Kernel_traits::Element;
//     using ElementAccum = typename Kernel_traits::ElementAccum;
//     using index_t = typename Kernel_traits::index_t;

//     // Shared memory.
//     extern __shared__ char smem_[];

//     // The thread index.
//     const int tidx = threadIdx.x;

//     constexpr int kBlockM = Kernel_traits::kBlockM;
//     constexpr int kBlockN = Kernel_traits::kBlockN;
//     constexpr int kHeadDim = Kernel_traits::kHeadDim;
//     constexpr int kNWarps = Kernel_traits::kNWarps;

//     using GmemTiledCopyO = std::conditional_t<
//         !Split,
//         typename Kernel_traits::GmemTiledCopyO,
//         typename Kernel_traits::GmemTiledCopyOaccum
//     >;
//     using ElementO = std::conditional_t<!Split, Element, ElementAccum>;

//     const BlockInfo</*Varlen=*/!Is_even_MN> binfo(params, bidb);
//     // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("Is_even_MN = %d, is_cumulativ = %d, seqlen_k_cache = %d, actual_seqlen_k = %d\n", Is_even_MN, params.is_seqlens_k_cumulative, binfo.seqlen_k_cache, binfo.actual_seqlen_k); }
//     // if (threadIdx.x == 0 && blockIdx.y == 1 && blockIdx.z == 0) { printf("params.knew_ptr = %p, seqlen_k_cache + seqlen_knew = %d\n", params.knew_ptr, binfo.seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew)); }
//     if (m_block * kBlockM >= binfo.actual_seqlen_q) return;

//     const int n_blocks_per_split = ((params.seqlen_k + kBlockN - 1) / kBlockN + num_n_splits - 1) / num_n_splits;
//     const int n_block_min = !Is_local
//         ? n_split_idx * n_blocks_per_split
//         : std::max(n_split_idx * n_blocks_per_split, (m_block * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q - params.window_size_left) / kBlockN);
//     int n_block_max = std::min(cute::ceil_div(binfo.actual_seqlen_k, kBlockN), (n_split_idx + 1) * n_blocks_per_split);
//     if (Is_causal || Is_local) {
//         n_block_max = std::min(n_block_max,
//                                cute::ceil_div((m_block + 1) * kBlockM + binfo.actual_seqlen_k - binfo.actual_seqlen_q + params.window_size_right, kBlockN));
//     }
//     if (n_block_min >= n_block_max) {  // This also covers the case where n_block_max <= 0
//         // We exit early and write 0 to gOaccum and -inf to gLSEaccum.
//         // Otherwise we might read OOB elements from gK and gV,
//         // or get wrong results when we combine gOaccum from different blocks.
//         const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
//             + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
//         const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
//             + m_block * kBlockM) * params.d_rounded;
//         const index_t row_offset_lseaccum = ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q + m_block * kBlockM;
//         Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
//                                       Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                                      make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
//         Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
//                                       Shape<Int<kBlockM>>{}, Stride<_1>{});

//         GmemTiledCopyO gmem_tiled_copy_Oaccum;
//         auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
//         Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);
//         Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
//         clear(tOrOaccum);
//         // Construct identity layout for sO
//         Tensor cO = make_identity_tensor(make_shape(size<0>(gOaccum), size<1>(gOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//         // Repeat the partitioning with identity layouts
//         Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);
//         Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
//         if (!Is_even_K) {
//             #pragma unroll
//             for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
//         }
//         // Clear_OOB_K must be false since we don't want to write zeros to gmem
//         flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//             gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
//         );
//         #pragma unroll
//         for (int m = 0; m < size<1>(tOgOaccum); ++m) {
//             const int row = get<0>(tOcO(0, m, 0));
//             if (row < binfo.actual_seqlen_q - m_block * kBlockM && get<1>(tOcO(0, m, 0)) == 0) { gLSEaccum(row) = Split ? -INFINITY : INFINITY; }
//         }
//         return;
//     }

//     // We iterate over the blocks in reverse order. This is because the last block is the only one
//     // that needs masking when we read K and V from global memory. Moreover, iterating in reverse
//     // might save us 1 register (we just need n_block instead of both n_block and n_block_max).

//     // We move K and V to the last block.
//     const int bidb_cache = params.cache_batch_idx == nullptr ? bidb : params.cache_batch_idx[bidb];
//     const int *block_table = params.block_table == nullptr ? nullptr : params.block_table + bidb * params.block_table_batch_stride;
//     const int block_table_idx = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN / params.page_block_size;
//     const int block_table_offset = block_table == nullptr ? 0 : (n_block_max - 1) * kBlockN - block_table_idx * params.page_block_size;
//     const index_t row_offset_k = block_table == nullptr
//         ? binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache)
//           + (n_block_max - 1) * kBlockN * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride
//         : block_table[block_table_idx] * params.k_batch_stride + block_table_offset * params.k_row_stride + (bidh / params.h_h_k_ratio) * params.k_head_stride;
//     const index_t row_offset_v = block_table == nullptr
//         ? binfo.k_offset(params.v_batch_stride, params.v_row_stride, bidb_cache)
//           + (n_block_max - 1) * kBlockN * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride
//         : block_table[block_table_idx] * params.v_batch_stride + block_table_offset * params.v_row_stride + (bidh / params.h_h_k_ratio) * params.v_head_stride;

//     if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0){
//         // n_split_idx: 0 | num_n_splits: 1
//         // nblocks: number of blocks used in k seq length iteration
//         printf(
//             "num_n_splits: %d | Is_local: %d | Is_causal: %d | n_split_idx: %d | n_block_min: %d | n_block_max: %d\n",
//             num_n_splits, Is_local, Is_causal, n_split_idx, n_block_min, n_block_max
//         );

//         printf("h_h_k_ratio: %d | k_row_stride: %ld\n", params.h_h_k_ratio, params.k_row_stride);
//     }

//     Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//                             make_shape(binfo.actual_seqlen_q, params.h, params.d),
//                             make_stride(params.q_row_stride, params.q_head_stride, _1{}));
//     Tensor gQ = local_tile(mQ(_, bidh, _), Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                            make_coord(m_block, 0));  // (kBlockM, kHeadDim)

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0) {
//     //     // printf(
//     //     //     "kBlockM: %d | kHeadDim: %d | actual_seqlen_q: %d | is_true: %d | k_offset: %d | n_block_max: %d | kBlockN: %d| block_table: %d | k_batch_stride: %d\n",
//     //     //     kBlockM, kHeadDim, binfo.actual_seqlen_q, block_table == nullptr,
//     //     //     binfo.k_offset(params.k_batch_stride, params.k_row_stride, bidb_cache),
//     //     //     n_block_max, kBlockN, block_table[block_table_idx], params.k_batch_stride
//     //     // );

//     //     // const int L = sqrtf(binfo.actual_seqlen_q);
        
//     //     // // Layout layout = make_layout(
//     //     // //     make_shape(params.d, L/2, L/2, 2, 2, params.h),
//     //     // //     make_stride(1, 2*head_dim*params.h, 2*L*params.d*params.h, L*params.d*params.h, params.d*params.h, params.d)
//     //     // // );

//     //     // Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr)
//     //     //                                     + binfo.q_offset(params.q_batch_stride, params.q_row_stride, bidb)),
//     //     //                         make_shape(params.d, L/2, L/2, 2, 2, params.h),
//     //     //                         make_stride(1, 2*params.d*params.h, 2*L*params.d*params.h, L*params.d*params.h, params.d*params.h, params.d));
//     //     //                         // local_tile(A(_, _, _, _, _, bidh), Shape<Int<4>, Int<4>, Int<2>>{}, make_coord(0, 0, 0, 0, 0));
//     //     // Tensor gQ = local_tile(mQ(_, _, _, _, _, bidh), Shape<Int<kHeadDim>, Int<16>, Int<4>>{},
//     //     //                     make_coord(0, 0, 0, 0, 0));  // (kBlockM, kHeadDim)
//     // }

//     Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.k_ptr) + row_offset_k),
//                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                             make_stride(params.k_row_stride, _1{}));
//     // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("k_ptr = %p, row_offset_k = %d, gK_ptr = %p\n", params.k_ptr, row_offset_k, gK.data()); }
//     Tensor gV = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.v_ptr) + row_offset_v),
//                             Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                             make_stride(params.v_row_stride, _1{}));

//     Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
//                             typename Kernel_traits::SmemLayoutQ{});
//     Tensor sK = make_tensor(sQ.data() + size(sQ), typename Kernel_traits::SmemLayoutKV{});
//     Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
//     Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
//     Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

//     typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
//     auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

//     // S: source | D: desination
//     Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
//     Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
//     Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);  // (KCPY, KCPY_N, KCPY_K)
//     Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
//     Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);  // (VCPY, VCPY_N, VCPY_K)
//     Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

//     typename Kernel_traits::TiledMma tiled_mma;
//     auto thr_mma = tiled_mma.get_thread_slice(tidx);
//     Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);                           // (MMA,MMA_M,MMA_K)
//     Tensor tSrK  = thr_mma.partition_fragment_B(sK);                           // (MMA,MMA_N,MMA_K)
//     Tensor tOrVt  = thr_mma.partition_fragment_B(sVtNoSwizzle);                // (MMA, MMA_K,MMA_N)

//     Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});  // MMA, MMA_M, MMA_K

//     //
//     // Copy Atom retiling
//     //

//     auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
//     auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
//     Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

//     auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
//     auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
//     Tensor tSsK = smem_thr_copy_K.partition_S(sK);

//     auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
//     auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
//     Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

//     // PREDICATES
//     //

//     // // Allocate predicate tensors for m and n
//     // Tensor tQpQ = make_tensor<bool>(make_shape(size<1>(tQsQ), size<2>(tQsQ)), Stride<_1,_0>{});
//     // Tensor tKVpKV = make_tensor<bool>(make_shape(size<1>(tKsK), size<2>(tKsK)), Stride<_1,_0>{});

//     // Construct identity layout for sQ and sK
//     Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));    // (BLK_N,BLK_K) -> (blk_n,blk_k)

//     // Repeat the partitioning with identity layouts
//     Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);       // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
//     Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);   // (BCPY,BCPY_N,BCPY_K) -> (blk_n,blk_k)

//     // Allocate predicate tensors for k
//     Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
//     Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));

//     // Set predicates for k bounds
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
//         #pragma unroll
//         for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
//     }

//     // Prologue

//     ///////// Append KV Cache /////////
//     // Copy from Knew to K, optionally apply rotary embedding.
//     typename Kernel_traits::GmemTiledCopyRotcossin gmem_tiled_copy_rotary;
//     auto gmem_thr_copy_rotary = gmem_tiled_copy_rotary.get_thread_slice(tidx);
//     typename Kernel_traits::GmemTiledCopyRotcossinCont gmem_tiled_copy_rotary_cont;
//     auto gmem_thr_copy_rotary_cont = gmem_tiled_copy_rotary_cont.get_thread_slice(tidx);
//     if constexpr (Append_KV) {
//         // Even if we have MQA / GQA, all threadblocks responsible for the same KV head are writing to
//         // gmem. Technically it's a race condition, but they all write the same content anyway, and it's safe.
//         // We want to do this so that all threadblocks can proceed right after they finish writing the KV cache.
//         const index_t row_offset_cossin = ((n_block_max - 1) * kBlockN + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])) * (params.rotary_dim / 2);
//         Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
//                                   Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
//                                   make_stride(params.rotary_dim / 2, _1{}));
//         Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
//                                   Shape<Int<kBlockN>, Int<kHeadDim / 2>>{},
//                                   make_stride(params.rotary_dim / 2, _1{}));
//         Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
//                                       Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                                       make_stride(params.rotary_dim / 2, _1{}));
//         Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
//                                       Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                                       make_stride(params.rotary_dim / 2, _1{}));
//         Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
//         Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
//         Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
//         Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
//         // if (cute::thread(0, 0)) { printf("rotary_cos_ptr = %p, gCos.data() = %p, tRgCos.data() = %p, rotary_dim = %d\n", params.rotary_cos_ptr, gCos.data(), tRgCos.data(), params.rotary_dim); }
//         // if (cute::thread(8, 0)) { print_tensor(gCos); }
//         // if (cute::thread(0, 0)) { print_tensor(tRgCos); }

//         // const index_t row_offset_knew = binfo.k_offset(params.knew_batch_stride, params.knew_row_stride, bidb)
//         const index_t row_offset_knew = bidb * params.knew_batch_stride
//             + ((n_block_max - 1) * kBlockN) * params.knew_row_stride + (bidh / params.h_h_k_ratio) * params.knew_head_stride;
//         // const index_t row_offset_vnew = binfo.k_offset(params.vnew_batch_stride, params.vnew_row_stride, bidb)
//         const index_t row_offset_vnew = bidb * params.vnew_batch_stride
//             + ((n_block_max - 1) * kBlockN) * params.vnew_row_stride + (bidh / params.h_h_k_ratio) * params.vnew_head_stride;
//         // Subtract seqlen_k_cache * row stride so that conceptually gK and gKnew "line up". When we access them,
//         // e.g. if gK has 128 rows and gKnew has 64 rows, we access gK[:128] and gKNew[128:128 + 64].
//         // This maps to accessing the first 64 rows of knew_ptr.
//         Tensor gKnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.knew_ptr)
//                                                 + row_offset_knew - binfo.seqlen_k_cache * params.knew_row_stride),
//                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                                   make_stride(params.knew_row_stride, _1{}));
//         // if (threadIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) { printf("knew_ptr = %p, row_offset_knew = %d, gKnew_ptr = %p\n", params.knew_ptr, row_offset_knew, gKnew.data()); }
//         Tensor gVnew = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.vnew_ptr)
//                                                 + row_offset_vnew - binfo.seqlen_k_cache * params.vnew_row_stride),
//                                   Shape<Int<kBlockN>, Int<kHeadDim>>{},
//                                   make_stride(params.vnew_row_stride, _1{}));
//         Tensor tKgKnew = gmem_thr_copy_QKV.partition_S(gKnew);  // (KCPY, KCPY_N, KCPY_K)
//         Tensor tVgVnew = gmem_thr_copy_QKV.partition_S(gVnew);  // (VCPY, VCPY_N, VCPY_K)

//         const int n_block_copy_min = std::max(n_block_min, binfo.seqlen_k_cache / kBlockN);
//         auto tKgK_data = tKgK.data();
//         auto tVgV_data = tVgV.data();
//         for (int n_block = n_block_max - 1; n_block >= n_block_copy_min; n_block--) {
//             flash::copy_w_min_idx<Is_even_K>(
//                 tVgVnew, tVgV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
//             );
//             tVgVnew.data() = tVgVnew.data() + (-int(kBlockN * params.vnew_row_stride));
//             if (params.rotary_dim == 0) {
//                 flash::copy_w_min_idx<Is_even_K>(
//                     tKgKnew, tKgK, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN, binfo.seqlen_k_cache - n_block * kBlockN
//                 );
//             } else {
//                 if (params.is_rotary_interleaved) {
//                     // Don't clear OOB_K because we're writing to global memory
//                     flash::copy_rotary_interleaved<Is_even_K, /*Clear_OOB_K=*/false>(
//                         tKgKnew, tKgK, tRgCos, tRgSin, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
//                         binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
//                     );
//                     tRgCos.data() = tRgCos.data() + (-int(kBlockN * params.rotary_dim / 2));
//                     tRgSin.data() = tRgSin.data() + (-int(kBlockN * params.rotary_dim / 2));
//                 } else {
//                     // Don't clear OOB_K because we're writing to global memory
//                     flash::copy_rotary_contiguous<Is_even_K, /*Clear_OOB_K=*/false>(
//                         tKgKnew, tKgK, tRgCosCont, tRgSinCont, tKVcKV, binfo.actual_seqlen_k - n_block * kBlockN,
//                         binfo.seqlen_k_cache - n_block * kBlockN, params.d, params.rotary_dim
//                     );
//                     tRgCosCont.data() = tRgCosCont.data() + (-int(kBlockN * params.rotary_dim / 2));
//                     tRgSinCont.data() = tRgSinCont.data() + (-int(kBlockN * params.rotary_dim / 2));

//                 }
//             }
//             tKgKnew.data() = tKgKnew.data() + (-int(kBlockN * params.knew_row_stride));
//             if (block_table == nullptr) {
//                 tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
//                 tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
//             } else {
//                 if (n_block > n_block_copy_min) {
//                     const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
//                     const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
//                     const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
//                     const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
//                     const int table_diff = block_table[block_table_idx_next] - block_table[block_table_idx_cur];
//                     const int offset_diff = block_table_offset_next - block_table_offset_cur;
//                     tVgV.data() = tVgV.data() + table_diff * params.v_batch_stride + offset_diff * params.v_row_stride;
//                     tKgK.data() = tKgK.data() + table_diff * params.k_batch_stride + offset_diff * params.k_row_stride;
//                 }
//             }
//         }
//         // Need this before we can read in K again, so that we'll see the updated K values.
//         __syncthreads();
//         tKgK.data() = tKgK_data;
//         tVgV.data() = tVgV_data;
//     }

//     ///////// Rotary /////////
//     // Read Q from gmem to smem, optionally apply rotary embedding.
//     if (!Append_KV || params.rotary_dim == 0) {
//         // We don't need to clear the sQ smem tiles since we'll only write out the valid outputs
//         flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
//                                            binfo.actual_seqlen_q - m_block * kBlockM);
//     } else {
//         const index_t row_offset_cossin = (binfo.seqlen_k_cache + (params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb]) + (Is_causal || Is_local ? m_block * kBlockM : 0)) * (params.rotary_dim / 2);
//         // If not causal, all the queries get the same the cos/sin, taken at location seqlen_k_cache.
//         // We do this by setting the row stride of gCos / gSin to 0.
//         Tensor gCos = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
//                                   Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
//                                   make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
//         Tensor gSin = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
//                                   Shape<Int<kBlockM>, Int<kHeadDim / 2>>{},
//                                   make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
//         Tensor gCosCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_cos_ptr) + row_offset_cossin),
//                                   Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                                   make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
//         Tensor gSinCont = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.rotary_sin_ptr) + row_offset_cossin),
//                                   Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                                   make_stride(Is_causal || Is_local ? params.rotary_dim / 2 : 0, _1{}));
//         Tensor tRgCos = gmem_thr_copy_rotary.partition_S(gCos);
//         Tensor tRgSin = gmem_thr_copy_rotary.partition_S(gSin);
//         Tensor tRgCosCont = gmem_thr_copy_rotary_cont.partition_S(gCosCont);
//         Tensor tRgSinCont = gmem_thr_copy_rotary_cont.partition_S(gSinCont);
//         if (params.is_rotary_interleaved) {
//             flash::copy_rotary_interleaved<Is_even_K>(
//                 tQgQ, tQsQ, tRgCos, tRgSin, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
//                 0, params.d, params.rotary_dim
//             );
//         } else {
//             flash::copy_rotary_contiguous<Is_even_K>(
//                 tQgQ, tQsQ, tRgCosCont, tRgSinCont, tQcQ, binfo.actual_seqlen_q - m_block * kBlockM,
//                 0, params.d, params.rotary_dim
//             );
//         }
//     }

//     int n_block = n_block_max - 1;
//     // We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
//     flash::copy<Is_even_MN, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV,
//                                        binfo.actual_seqlen_k - n_block * kBlockN, print);

//     cute::cp_async_fence();

//     // flash::cp_async_wait<0>();
//     // __syncthreads();
//     // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tKsK); }
//     // __syncthreads();

//     clear(acc_o);

//     flash::Softmax<2 * size<1>(acc_o)> softmax;

//     const float alibi_slope = !Has_alibi ? 0.0f : reinterpret_cast<float *>(params.alibi_slopes_ptr)[bidb * params.alibi_slopes_batch_stride + bidh] / params.scale_softmax;
//     flash::Mask<Is_causal, Is_local, Has_alibi> mask(binfo.actual_seqlen_k, binfo.actual_seqlen_q, params.window_size_left, params.window_size_right, alibi_slope);

//     // For performance reason, we separate out two kinds of iterations:
//     // those that need masking on S, and those that don't.
//     // We need masking on S for the very last block when K and V has length not multiple of kBlockN.
//     // We also need masking on S if it's causal, for the last ceil_div(kBlockM, kBlockN) blocks.
//     // We will have at least 1 "masking" iteration.

//     // If not even_N, then seqlen_k might end in the middle of a block. In that case we need to
//     // mask 2 blocks (e.g. when kBlockM == kBlockN), not just 1.
//     constexpr int n_masking_steps = (!Is_causal && !Is_local)
//         ? 1
//         : ((Is_even_MN && Is_causal) ? cute::ceil_div(kBlockM, kBlockN) : cute::ceil_div(kBlockM, kBlockN) + 1);

//     if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx == 0){
//         printf("n_masking_steps: %d | rotary: %d | Is_even_MN: %d | Is_even_K: %d\n", n_masking_steps, params.rotary_dim, Is_even_MN, Is_even_K);
//     }

//     #pragma unroll
//     for (int masking_step = 0; masking_step < n_masking_steps; ++masking_step, --n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         flash::cp_async_wait<0>();
//         __syncthreads();

//         // Advance gV
//         if (masking_step > 0) {
//             if (block_table == nullptr) {
//                 tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
//             } else {
//                 const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
//                 const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
//                 const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
//                 const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
//                 tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
//             }
//             flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
//         } else {
//             // Clear the smem tiles to account for predicated off loads
//             flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/true>(
//                 gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, binfo.actual_seqlen_k - n_block * kBlockN
//             );
//         }
//         cute::cp_async_fence();

//         flash::gemm(
//             acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
//             smem_thr_copy_Q, smem_thr_copy_K
//         );
//         // if (cute::thread0()) { print(acc_s); }
//         if constexpr (Is_softcap){
//             flash::apply_softcap(acc_s, params.softcap);
//         }


//         mask.template apply_mask<Is_causal, Is_even_MN>(
//             acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
//         );

//         flash::cp_async_wait<0>();
//         __syncthreads();
//         // if (tidx == 0 && blockIdx.y == 0 && blockIdx.z == 0) { print(tVsV); }
//         // __syncthreads();

//         if (n_block > n_block_min) {
//             // Advance gK
//             if (block_table == nullptr) {
//                 tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
//             } else {
//                 const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
//                 const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
//                 const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
//                 const int block_table_offset_next =(n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
//                 tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
//             }
//             flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
//             // This cp_async_fence needs to be in the if block, otherwise the synchronization
//             // isn't right and we get race conditions.
//             cute::cp_async_fence();
//         }

//         // We have key_padding_mask so we'll need to Check_inf
//         masking_step == 0
//             ? softmax.template softmax_rescale_o</*Is_first=*/true,  /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2)
//             : softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal || Is_local || !Is_even_MN>(acc_s, acc_o, params.scale_softmax_log2);
//         // if (cute::thread0()) { print(scores_max); print(scores_sum); print(scores); }

//         // Convert acc_s from fp32 to fp16/bf16
//         Tensor rP = flash::convert_type<Element>(acc_s);
//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
//         Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

//         flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);

//         // This check is at the end of the loop since we always have at least 1 iteration
//         if (n_masking_steps > 1 && n_block <= n_block_min) {
//             --n_block;
//             break;
//         }
//     }

//     // These are the iterations where we don't need masking on S
//     for (; n_block >= n_block_min; --n_block) {
//         Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});  // (MMA=4, MMA_M, MMA_N)
//         clear(acc_s);
//         flash::cp_async_wait<0>();
//         __syncthreads();
//         // Advance gV
//         if (block_table == nullptr) {
//             tVgV.data() = tVgV.data() + (-int(kBlockN * params.v_row_stride));
//         } else {
//             const int block_table_idx_cur = (n_block + 1) * kBlockN / params.page_block_size;
//             const int block_table_offset_cur = (n_block + 1) * kBlockN - block_table_idx_cur * params.page_block_size;
//             const int block_table_idx_next = n_block * kBlockN / params.page_block_size;
//             const int block_table_offset_next = n_block * kBlockN - block_table_idx_next * params.page_block_size;
//             tVgV.data() = tVgV.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.v_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.v_row_stride;
//         }
//         flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV);
//         cute::cp_async_fence();

//         flash::gemm(
//             acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
//             smem_thr_copy_Q, smem_thr_copy_K
//         );
//         if constexpr (Is_softcap){
//             flash::apply_softcap(acc_s, params.softcap);
//         }

//         flash::cp_async_wait<0>();
//         __syncthreads();
//         if (n_block > n_block_min) {
//             // Advance gK
//             if (block_table == nullptr) {
//                 tKgK.data() = tKgK.data() + (-int(kBlockN * params.k_row_stride));
//             } else {
//                 const int block_table_idx_cur = n_block * kBlockN / params.page_block_size;
//                 const int block_table_offset_cur = n_block * kBlockN - block_table_idx_cur * params.page_block_size;
//                 const int block_table_idx_next = (n_block - 1) * kBlockN / params.page_block_size;
//                 const int block_table_offset_next = (n_block - 1) * kBlockN - block_table_idx_next * params.page_block_size;
//                 tKgK.data() = tKgK.data() + (block_table[block_table_idx_next] - block_table[block_table_idx_cur]) * params.k_batch_stride + (block_table_offset_next - block_table_offset_cur) * params.k_row_stride;
//             }
//             flash::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV);
//             // This cp_async_fence needs to be in the if block, otherwise the synchronization
//             // isn't right and we get race conditions.
//             cute::cp_async_fence();
//         }

//         mask.template apply_mask</*Causal_mask=*/false>(
//             acc_s, n_block * kBlockN, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
//         );
//         softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_local>(acc_s, acc_o, params.scale_softmax_log2);

//         Tensor rP = flash::convert_type<Element>(acc_s);
//         // Reshape rP from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
//         // if using m16n8k16 or (4, MMA_M, MMA_N) if using m16n8k8.
//         Tensor tOrP = make_tensor(rP.data(), flash::convert_layout_acc_Aregs<Kernel_traits::TiledMma>(rP.layout()));

//         flash::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
//     }

//     // Epilogue

//     Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false, Split>(acc_o, params.scale_softmax);
//     // if (cute::thread0()) { print(lse); }

//     Tensor sOaccum = make_tensor(make_smem_ptr(reinterpret_cast<ElementO *>(smem_)), typename Kernel_traits::SmemLayoutO{}); // (SMEM_M,SMEM_N)
//     // Partition sO to match the accumulator partitioning
//     using SmemTiledCopyO = std::conditional_t<
//         !Split,
//         typename Kernel_traits::SmemCopyAtomO,
//         typename Kernel_traits::SmemCopyAtomOaccum
//     >;
//     auto smem_tiled_copy_Oaccum = make_tiled_copy_C(SmemTiledCopyO{}, tiled_mma);
//     auto smem_thr_copy_Oaccum = smem_tiled_copy_Oaccum.get_thread_slice(tidx);
//     Tensor rO = flash::convert_type<ElementO>(acc_o);
//     Tensor taccOrOaccum = smem_thr_copy_Oaccum.retile_S(rO);        // ((Atom,AtomNum), MMA_M, MMA_N)
//     Tensor taccOsOaccum = smem_thr_copy_Oaccum.partition_D(sOaccum);     // ((Atom,AtomNum),PIPE_M,PIPE_N)

//     // sOaccum is larger than sQ, so we need to syncthreads here
//     // TODO: allocate enough smem for sOaccum
//     if constexpr (Split) { __syncthreads(); }

//     cute::copy(smem_tiled_copy_Oaccum, taccOrOaccum, taccOsOaccum);

//     const index_t row_offset_o = binfo.q_offset(params.o_batch_stride, params.o_row_stride, bidb)
//         + m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
//     const index_t row_offset_oaccum = (((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q
//                                          + m_block * kBlockM) * params.d_rounded;
//     const index_t row_offset_lseaccum = (Split || !params.unpadded_lse ?
//             ((n_split_idx * params.b + bidb) * params.h + bidh) * params.seqlen_q : bidh * params.total_q + binfo.q_offset(params.seqlen_q, 1, bidb)
//         ) + m_block * kBlockM;

//     // if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && tidx < 10){
//     //     printf("tidx: %d | row_offset_o: %ld | row_offset_lseaccum: %ld\n", tidx, row_offset_o, row_offset_lseaccum);
//     // }

//     Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO *>(Split ? params.oaccum_ptr : params.o_ptr) + (Split ? row_offset_oaccum : row_offset_o)),
//                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                                  make_stride(Split ? kHeadDim : params.o_row_stride, _1{}));
//     Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(Split ? params.softmax_lseaccum_ptr : params.softmax_lse_ptr) + row_offset_lseaccum),
//                                    Shape<Int<kBlockM>>{}, Stride<_1>{});
//     // if (tidx == 0) { printf("row_offset_o = %d, bidh = %d, gOaccum = %p\n", row_offset_o, bidh, gOaccum.data()); }

//     GmemTiledCopyO gmem_tiled_copy_Oaccum;
//     auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
//     Tensor tOsOaccum = gmem_thr_copy_Oaccum.partition_S(sOaccum);        // ((Atom,AtomNum),ATOM_M,ATOM_N)
//     Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_D(gOaccum);

//     __syncthreads();

//     Tensor tOrOaccum = make_tensor<ElementO>(shape(tOgOaccum));
//     cute::copy(gmem_tiled_copy_Oaccum, tOsOaccum, tOrOaccum);

//     Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     Tensor taccOcO = thr_mma.partition_C(caccO);                           // (MMA,MMA_M,MMA_K)
//     static_assert(decltype(size<0>(taccOcO))::value == 4);
//     // Convert to ((2, 2), MMA_M, MMA_K) then take only the row indices.
//     Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
//     CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));                     // MMA_M
//     if (get<1>(taccOcO_row(0)) == 0) {
//         #pragma unroll
//         for (int mi = 0; mi < size(lse); ++mi) {
//             const int row = get<0>(taccOcO_row(mi));
//             if (row < binfo.actual_seqlen_q - m_block * kBlockM) { gLSEaccum(row) = lse(mi); }
//         }
//     }

//     // Construct identity layout for sO
//     Tensor cO = make_identity_tensor(make_shape(size<0>(sOaccum), size<1>(sOaccum)));    // (BLK_M,BLK_K) -> (blk_m,blk_k)
//     // Repeat the partitioning with identity layouts
//     Tensor tOcO = gmem_thr_copy_Oaccum.partition_D(cO);                           // (ACPY,ACPY_M,ACPY_K) -> (blk_m,blk_k)
//     Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
//     }
//     // Clear_OOB_K must be false since we don't want to write zeros to gmem
//     flash::copy<Is_even_MN, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
//         gmem_tiled_copy_Oaccum, tOrOaccum, tOgOaccum, tOcO, tOpO, binfo.actual_seqlen_q - m_block * kBlockM
//     );
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
// inline __device__ void compute_attn(const Params &params) {
//     const int m_block = blockIdx.x;
//     // The block index for the batch.
//     const int bidb = blockIdx.y;
//     // The block index for the head.
//     const int bidh = blockIdx.z;

//     // We want the fwd and bwd to generate the same dropout pattern (RNG), without restricting
//     // them to have the same number of threads or have to traverse the attention matrix
//     // in the same order.
//     // In the Philox RNG, we use the offset to store the batch, head, and the lane id
//     // (within a warp). We use the subsequence to store the location of the 16 x 32 blocks within
//     // the attention matrix. This way, as long as we have the batch, head, and the location of
//     // the 16 x 32 block within the attention matrix, we can generate the exact same dropout pattern.

//     flash::compute_attn_1rowblock<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(params, bidb, bidh, m_block);
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
// inline __device__ void compute_attn_sparse_mask0(const Params &params) {
//     const int m_block = blockIdx.x;
//     // The block index for the batch.
//     const int bidb = blockIdx.y;
//     // The block index for the head.
//     const int bidh = blockIdx.z / 4;

//     const int outgroup_id = (blockIdx.z >> 1) & 1; // outer group index
//     const int inside_outgroup_id = blockIdx.z & 1; // inside the outer group index


//     flash::compute_attn_1rowblock_sparse_mask0<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
//         params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
//     );
// }


// template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
// inline __device__ void compute_attn_sparse_mask1(const Params &params) {
//     const int m_block = blockIdx.x;
//     // The block index for the batch.
//     const int bidb = blockIdx.y;
//     // The block index for the head.
//     const int bidh = blockIdx.z / 4;

//     const int outgroup_id = (blockIdx.z >> 1) & 1; // outer group index
//     const int inside_outgroup_id = blockIdx.z & 1; // inside the outer group index

//     flash::compute_attn_1rowblock_sparse_mask1<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
//         params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
//     );
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_sparse_mask0_text_front(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z / 4;

    const int text_blocks = (params.text_length / 4) / Kernel_traits::kBlockM;
    const bool is_text = m_block < text_blocks;
    const int image_m_block = m_block - text_blocks;

    const int outgroup_id = (blockIdx.z >> 1) & 1; // outer group index
    const int inside_outgroup_id = blockIdx.z & 1; // inside the outer group index


    // if (is_text) {
    //     flash::compute_attn_1rowblock_sparse_mask0_text_front_text<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
    //         params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
    //     );
    // }

    if (is_text) {
        flash::compute_attn_1rowblock_sparse_mask0_text_front_text<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
        );
    } else {
        flash::compute_attn_1rowblock_sparse_mask0_text_front_image<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, image_m_block, outgroup_id, inside_outgroup_id
        );
    }
}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_sparse_mask1_text_front(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z / 4;

    const int text_blocks = (params.text_length / 4) / Kernel_traits::kBlockM;
    const bool is_text = m_block < text_blocks;
    const int image_m_block = m_block - text_blocks;

    const int outgroup_id = (blockIdx.z >> 1) & 1; // outer group index
    const int inside_outgroup_id = blockIdx.z & 1; // inside the outer group index

    if (is_text) {
        flash::compute_attn_1rowblock_sparse_mask1_text_front_text<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
        );
    } else {
        flash::compute_attn_1rowblock_sparse_mask1_text_front_image<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, image_m_block, outgroup_id, inside_outgroup_id
        );
    }
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_sparse_masksw_text_front(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z;

    const int text_blocks = params.text_length / Kernel_traits::kBlockM;
    const bool is_text = m_block < text_blocks;

    const int outgroup_id = m_block / 4; // outer group index
    const int inside_outgroup_id = m_block % 4; // inside the outer group index

    if (is_text) {
        flash::compute_attn_1rowblock_sparse_masksw_text_front_text<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
        );
    } else {
        flash::compute_attn_1rowblock_sparse_masksw_text_front_image<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
            params, bidb, bidh, m_block
        );
    }
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_sparse_mask0_8group_text_front(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z / 8;

    const int text_blocks = (params.text_length / 8) / Kernel_traits::kBlockM;
    const bool is_text = m_block < text_blocks;
    const int image_m_block = m_block - text_blocks;

    const int outgroup_id = (blockIdx.z % 8) / 4; // outer group index
    const int inside_outgroup_id = blockIdx.z % 4; // inside the outer group index

    flash::compute_attn_1rowblock_sparse_mask0_8group_text_front<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
        params, bidb, bidh, m_block, outgroup_id, inside_outgroup_id
    );
}



template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_sparse_16group_llm(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // The block index for the head.
    const int bidh = blockIdx.z / params.group_num;

    const int group_id = blockIdx.z % params.group_num;

    int actual_seqlen_q = (Kernel_traits::maskType == 1) 
                ? (group_id < params.increment_groups ? params.seqlen_q + params.group_num 
                    : (group_id == params.increment_groups ? params.seqlen_q + params.special_group_increment 
                    : params.seqlen_q))
                : (group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_q);
    
    if (actual_seqlen_q % 64 == 0){
        flash::compute_attn_1rowblock_sparse_16group_llm<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, true, true, Is_softcap, Return_softmax>(
            params, bidb, bidh, group_id, m_block
        );
    } else{
        flash::compute_attn_1rowblock_sparse_16group_llm<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, false, false, Is_softcap, Return_softmax>(
            params, bidb, bidh, group_id, m_block
        );
    }

    // flash::compute_attn_1rowblock_sparse_16group_llm<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
    //     params, bidb, bidh, group_id, m_block
    // );

}


template<typename Kernel_traits, bool Is_dropout, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Return_softmax, typename Params>
inline __device__ void compute_attn_sparse_sw_llm(const Params &params) {
    const int m_block = blockIdx.x;
    // The block index for the batch.
    const int bidb = blockIdx.y;
    // INTRA fix (Bug A1): the sliding window has NO group partitioning -- every
    // (m_block, batch, head) tile is independent and reads absolute token
    // positions.  run_flash_fwd_sparse_sw launches dim3(num_m_block, b, h), so
    // grid.z indexes heads directly.  The Scatter kernel's
    // `bidh = blockIdx.z / group_num` / `group_id = blockIdx.z % group_num`
    // decode is only valid there, where the launch is h * group_num.  Applying
    // it here made only heads [0, h/group_num) get written (h/group_num = 4 of
    // 32 for LLaMA-3.1-8B) with group_num CTAs racing on each of them.
    const int bidh = blockIdx.z;

    // Lookback distance is fixed by the host: mha_fwd_llm sets seqlen_k = 576
    // for mask=4, i.e. kBlockM current block + (seqlen_k - kBlockM) of history.
    constexpr int kBlockM = Kernel_traits::kBlockM;
    const int start = max(m_block * kBlockM - (params.seqlen_k - kBlockM), 0);

    flash::compute_attn_1rowblock_sw_llm<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
        params, bidb, bidh, /*group_id=*/0, m_block, start
    );


    // flash::compute_attn_1rowblock_sparse_16group_llm<Kernel_traits, Is_dropout, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Return_softmax>(
    //     params, bidb, bidh, group_id, m_block
    // );

}


////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV, typename Params>
// inline __device__ void compute_attn_splitkv(const Params &params) {
//     const int m_block = blockIdx.x;
//     // The block index for the batch.
//     const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
//     // The block index for the head.
//     const int bidh = Split ? blockIdx.z - bidb * params.h : blockIdx.z;
//     const int n_split_idx = Split ? blockIdx.y : 0;
//     const int num_n_splits = Split ? gridDim.y : 1;
//     flash::compute_attn_1rowblock_splitkv<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, bidh, m_block, n_split_idx, num_n_splits);
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// template<typename Kernel_traits, int kBlockM, int Log_max_splits, bool Is_even_K, typename Params>
// inline __device__ void combine_attn_seqk_parallel(const Params &params) {
//     using Element = typename Kernel_traits::Element;
//     using ElementAccum = typename Kernel_traits::ElementAccum;
//     using index_t = typename Kernel_traits::index_t;
//     constexpr int kMaxSplits = 1 << Log_max_splits;
//     constexpr int kHeadDim = Kernel_traits::kHeadDim;
//     constexpr int kNThreads = Kernel_traits::kNThreads;

//     static_assert(kMaxSplits <= 128, "kMaxSplits must be <= 128");
//     static_assert(kBlockM == 4 || kBlockM == 8 || kBlockM == 16 || kBlockM == 32, "kBlockM must be 4, 8, 16 or 32");
//     static_assert(kNThreads == 128, "We assume that each block has 128 threads");

//     // Shared memory.
//     // kBlockM + 1 instead of kBlockM to reduce bank conflicts.
//     __shared__ ElementAccum sLSE[kMaxSplits][kBlockM + 1];

//     // The thread and block index.
//     const int tidx = threadIdx.x;
//     const int bidx = blockIdx.x;

//     const index_t lse_size = params.b * params.h * params.seqlen_q;

//     const index_t row_offset_lse = bidx * kBlockM;
//     Tensor gLSEaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lseaccum_ptr) + row_offset_lse),
//                                    Shape<Int<kMaxSplits>, Int<kBlockM>>{},
//                                    make_stride(lse_size, _1{}));

//     // LSE format is different depending on params.unpadded_lse and params.seqlenq_ngroups_swapped, see comment in get_lse_tile.
//     // This tensor's layout maps row_offset_lse to {bidb, bidh, q_offset}.
//     Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
//                               Shape<Int<kBlockM>>{}, Stride<_1>{});

//     // This layout maps row_offset_lse to {bidh, q_offset, bidb} or {bidh, bidb, q_offset}.
//     Layout flat_layout = make_layout(lse_size);
//     Layout orig_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b));
//     auto transposed_stride = params.seqlenq_ngroups_swapped ? make_stride(params.b, params.seqlen_q * params.b, 1) : make_stride(1, params.seqlen_q * params.b, params.seqlen_q);
//     Layout remapped_layout = make_layout(make_shape(params.seqlen_q, params.h, params.b), transposed_stride);
//     Layout final_layout = cute::composition(remapped_layout, cute::composition(orig_layout, flat_layout));

//     Tensor gLSE_unpadded = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr)), final_layout);

//     constexpr int kNLsePerThread = (kMaxSplits * kBlockM + kNThreads - 1) / kNThreads;

//     // Read the LSE values from gmem and store them in shared memory, then transpose them.
//     constexpr int kRowsPerLoadLSE = kNThreads / kBlockM;
//     #pragma unroll
//     for (int l = 0; l < kNLsePerThread; ++l) {
//         const int row = l * kRowsPerLoadLSE + tidx / kBlockM;
//         const int col = tidx % kBlockM;
//         ElementAccum lse = (row < params.num_splits && col < lse_size - bidx * kBlockM) ? gLSEaccum(row, col) : -INFINITY;
//         if (row < kMaxSplits) { sLSE[row][col] = lse; }
//         // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse); }
//     }
//     // if (bidx == 1 && tidx < 32) { printf("tidx = %d, row_offset_lse = %d, lse = %f\n", tidx, row_offset_lse, lse_accum(0)); }
//     __syncthreads();
//     Tensor lse_accum = make_tensor<ElementAccum>(Shape<Int<kNLsePerThread>>{});
//     constexpr int kRowsPerLoadTranspose = std::min(kRowsPerLoadLSE, kMaxSplits);
//     // To make sure that kMaxSplits is within 1 warp: we decide how many elements within kMaxSplits
//     // each thread should hold. If kMaxSplits = 16, then each thread holds 2 elements (128 threads,
//     // kBlockM rows, so each time we load we can load 128 / kBlockM rows).
//     // constexpr int kThreadsPerSplit = kMaxSplits / kRowsPerLoadTranspose;
//     // static_assert(kThreadsPerSplit <= 32);
//     static_assert(kRowsPerLoadTranspose <= 32);
//     static_assert(kNLsePerThread * kRowsPerLoadTranspose <= kMaxSplits);
//     #pragma unroll
//     for (int l = 0; l < kNLsePerThread; ++l) {
//         const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
//         const int col = tidx / kRowsPerLoadTranspose;
//         lse_accum(l) = (row < kMaxSplits && col < kBlockM) ? sLSE[row][col] : -INFINITY;
//         // if (bidx == 0 && tidx < 32) { printf("tidx = %d, row = %d, col = %d, lse = %f\n", tidx, row, col, lse_accum(l)); }
//     }

//     // Compute the logsumexp of the LSE along the split dimension.
//     ElementAccum lse_max = lse_accum(0);
//     #pragma unroll
//     for (int l = 1; l < kNLsePerThread; ++l) { lse_max = max(lse_max, lse_accum(l)); }
//     MaxOp<float> max_op;
//     lse_max = Allreduce<kRowsPerLoadTranspose>::run(lse_max, max_op);
//     lse_max = lse_max == -INFINITY ? 0.0f : lse_max;  // In case all local LSEs are -inf
//     float lse_sum = expf(lse_accum(0) - lse_max);
//     #pragma unroll
//     for (int l = 1; l < kNLsePerThread; ++l) { lse_sum += expf(lse_accum(l) - lse_max); }
//     SumOp<float> sum_op;
//     lse_sum = Allreduce<kRowsPerLoadTranspose>::run(lse_sum, sum_op);
//     // For the case where all local lse == -INFINITY, we want to set lse_logsum to INFINITY. Otherwise
//     // lse_logsum is log(0.0) = -INFINITY and we get NaN when we do lse_accum(l) - lse_logsum.
//     ElementAccum lse_logsum = (lse_sum == 0.f || lse_sum != lse_sum) ? INFINITY : logf(lse_sum) + lse_max;
//     // if (bidx == 0 && tidx < 32) { printf("tidx = %d, lse = %f, lse_max = %f, lse_logsum = %f\n", tidx, lse_accum(0), lse_max, lse_logsum); }
//     if (tidx % kRowsPerLoadTranspose == 0 && tidx / kRowsPerLoadTranspose < kBlockM) {
//         if (params.unpadded_lse) {
//             const index_t lse_offset = row_offset_lse + tidx / kRowsPerLoadTranspose;
//             if (lse_offset < lse_size) {
//                 gLSE_unpadded(lse_offset) = lse_logsum;
//             }
//         } else {
//             gLSE(tidx / kRowsPerLoadTranspose) = lse_logsum;
//         }
//     }
//     // Store the scales exp(lse - lse_logsum) in shared memory.
//     #pragma unroll
//     for (int l = 0; l < kNLsePerThread; ++l) {
//         const int row = l * kRowsPerLoadTranspose + tidx % kRowsPerLoadTranspose;
//         const int col = tidx / kRowsPerLoadTranspose;
//         if (row < params.num_splits && col < kBlockM) { sLSE[row][col] = expf(lse_accum(l) - lse_logsum); }
//     }
//     __syncthreads();

//     const index_t row_offset_oaccum = bidx * kBlockM * params.d_rounded;
//     Tensor gOaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.oaccum_ptr) + row_offset_oaccum),
//                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
//                                  Stride<Int<kHeadDim>, _1>{});
//     constexpr int kBlockN = kNThreads / kBlockM;
//     using GmemLayoutAtomOaccum = Layout<Shape<Int<kBlockM>, Int<kBlockN>>, Stride<Int<kBlockN>, _1>>;
//     using GmemTiledCopyOaccum = decltype(
//         make_tiled_copy(Copy_Atom<DefaultCopy, ElementAccum>{},
//                         GmemLayoutAtomOaccum{},
//                         Layout<Shape < _1, _4>>{}));  // Val layout, 4 vals per store
//     GmemTiledCopyOaccum gmem_tiled_copy_Oaccum;
//     auto gmem_thr_copy_Oaccum = gmem_tiled_copy_Oaccum.get_thread_slice(tidx);
//     Tensor tOgOaccum = gmem_thr_copy_Oaccum.partition_S(gOaccum);
//     Tensor tOrO = make_tensor<ElementAccum>(shape(tOgOaccum));
//     Tensor tOrOaccum = make_tensor<ElementAccum>(shape(tOgOaccum));
//     clear(tOrO);

//     // Predicates
//     Tensor cOaccum = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
//     // Repeat the partitioning with identity layouts
//     Tensor tOcOaccum = gmem_thr_copy_Oaccum.partition_S(cOaccum);
//     Tensor tOpOaccum = make_tensor<bool>(make_shape(size<2>(tOgOaccum)));
//     if (!Is_even_K) {
//         #pragma unroll
//         for (int k = 0; k < size(tOpOaccum); ++k) { tOpOaccum(k) = get<1>(tOcOaccum(0, 0, k)) < params.d; }
//     }
//     // Load Oaccum in then scale and accumulate to O
//     for (int split = 0; split < params.num_splits; ++split) {
//         flash::copy</*Is_even_MN=*/false, Is_even_K>(
//             gmem_tiled_copy_Oaccum, tOgOaccum, tOrOaccum, tOcOaccum, tOpOaccum, params.b * params.h * params.seqlen_q - bidx * kBlockM
//         );
//         #pragma unroll
//         for (int m = 0; m < size<1>(tOrOaccum); ++m) {
//             int row = get<0>(tOcOaccum(0, m, 0));
//             ElementAccum lse_scale = sLSE[split][row];
//             #pragma unroll
//             for (int k = 0; k < size<2>(tOrOaccum); ++k) {
//                 #pragma unroll
//                 for (int i = 0; i < size<0>(tOrOaccum); ++i) {
//                     tOrO(i, m, k) += lse_scale * tOrOaccum(i, m, k);
//                 }
//             }
//         // if (cute::thread0()) { printf("lse_scale = %f, %f\n", sLSE[split][0], sLSE[split][1]); print(tOrOaccum); }
//         }
//         tOgOaccum.data() = tOgOaccum.data() + params.b * params.h * params.seqlen_q * params.d_rounded;
//     }
//     // if (cute::thread0()) { print_tensor(tOrO); }

//     Tensor rO = flash::convert_type<Element>(tOrO);
//     // Write to gO
//     #pragma unroll
//     for (int m = 0; m < size<1>(rO); ++m) {
//         const int idx = bidx * kBlockM + get<0>(tOcOaccum(0, m, 0));
//         if (idx < params.b * params.h * params.seqlen_q) {
//             const int batch_idx = idx / (params.h * params.seqlen_q);
//             const int head_idx = (idx - batch_idx * (params.h * params.seqlen_q)) / params.seqlen_q;
//             // The index to the rows of Q
//             const int row = idx - batch_idx * (params.h * params.seqlen_q) - head_idx * params.seqlen_q;
//             auto o_ptr = reinterpret_cast<Element *>(params.o_ptr) + batch_idx * params.o_batch_stride
//                 + head_idx * params.o_head_stride + row * params.o_row_stride;
//             #pragma unroll
//             for (int k = 0; k < size<2>(rO); ++k) {
//                 if (Is_even_K || tOpOaccum(k)) {
//                     const int col = get<1>(tOcOaccum(0, m, k));
//                     Tensor gO = make_tensor(make_gmem_ptr(o_ptr + col),
//                                             Shape<Int<decltype(size<0>(rO))::value>>{}, Stride<_1>{});
//                     // TODO: Should check if this is using vectorized store, but it seems pretty fast
//                     copy(rO(_, m, k), gO);
//                     // if (bidx == 0 && tidx == 0) { printf("tidx = %d, idx = %d, batch_idx = %d, head_idx = %d, row = %d, col = %d\n", tidx, idx, batch_idx, head_idx, row, col); print(rO(_, m, k)); print(gO); }
//                     // reinterpret_cast<uint64_t *>(o_ptr)[col / 4] = recast<uint64_t>(rO)(0, m, k);
//                 }
//             }
//         }
//     }
// }

} // namespace flash
