/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/

#pragma once

namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template<bool Varlen=true>
struct BlockInfo {

    template<typename Params>
    __device__ BlockInfo(const Params &params, const int bidb)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        , length(params.length)
        , width(params.width)
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    const int length;
    const int width;
    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int leftpad_k;
    const int seqlen_k_cache;
    const int actual_seqlen_k;
};


template<bool Varlen=true>
struct BlockInfo_llm {

    template<typename Params>
    __device__ BlockInfo_llm(const Params &params, const int bidb, const int group_id, const int mask_type)
        : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
        , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
        // , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
        // , actual_seqlen_q(group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_q)
        // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
        // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
        , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
        , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
        // , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
        // , actual_seqlen_k(group_id < params.increment_groups ? params.seqlen_k + 1 : params.seqlen_k)
        // , actual_seqlen_q(
        //     (mask_type == 1) 
        //         ? (group_id < params.increment_groups ? params.seqlen_q + params.group_num 
        //             : (group_id == params.increment_groups ? params.seqlen_q + params.special_group_increment 
        //             : params.seqlen_q))
        //         : (group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_q))
        // , actual_seqlen_k(
        //     (mask_type == 1) 
        //         ? (group_id < params.increment_groups ? params.seqlen_k + params.group_num 
        //             : (group_id == params.increment_groups ? params.seqlen_k + params.special_group_increment 
        //             : params.seqlen_k))
        //         : (group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_k))
        , actual_seqlen_q(
            (mask_type == 4) 
                ? params.seqlen_q
                : (group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_q))
        , actual_seqlen_k(
            (mask_type == 4) 
                ? params.seqlen_k
                : (group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_k))
        , org_seqlen_k(params.org_seqlen_k)
        {
        }

    template <typename index_t>
    __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
    }

    template <typename index_t>
    __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
        return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
    }

    const int sum_s_q;
    const int sum_s_k;
    const int actual_seqlen_q;
    // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
    const int leftpad_k;
    const int seqlen_k_cache;
    const int actual_seqlen_k;
    const int org_seqlen_k;
};


// template<bool Varlen=true>
// struct BlockInfo_llm_mask1 {

//     template<typename Params>
//     __device__ BlockInfo_llm(const Params &params, const int bidb, const int group_id)
//         : sum_s_q(!Varlen || params.cu_seqlens_q == nullptr ? -1 : params.cu_seqlens_q[bidb])
//         , sum_s_k(!Varlen || params.cu_seqlens_k == nullptr || !params.is_seqlens_k_cumulative ? -1 : params.cu_seqlens_k[bidb])
//         // , actual_seqlen_q(!Varlen || params.cu_seqlens_q == nullptr ? params.seqlen_q : params.cu_seqlens_q[bidb + 1] - sum_s_q)
//         // , actual_seqlen_q(group_id < params.increment_groups ? params.seqlen_q + 1 : params.seqlen_q)
//         // If is_seqlens_k_cumulative, then seqlen_k is cu_seqlens_k[bidb + 1] - cu_seqlens_k[bidb].
//         // Otherwise it's cu_seqlens_k[bidb], i.e., we use cu_seqlens_k to store the sequence lengths of K.
//         , leftpad_k(params.leftpad_k == nullptr ? 0 : params.leftpad_k[bidb])
//         , seqlen_k_cache((!Varlen || params.cu_seqlens_k == nullptr ? params.seqlen_k : (params.is_seqlens_k_cumulative ? params.cu_seqlens_k[bidb + 1] - sum_s_k : params.cu_seqlens_k[bidb])) - leftpad_k)
//         // , actual_seqlen_k(params.seqused_k ? params.seqused_k[bidb] - leftpad_k : seqlen_k_cache + (params.knew_ptr == nullptr ? 0 : params.seqlen_knew))
//         // , actual_seqlen_k(group_id < params.increment_groups ? params.seqlen_k + 1 : params.seqlen_k)
//         , length(params.length)
//         , width(params.width)
//         {
//         }

//     template <typename index_t>
//     __forceinline__ __device__ index_t q_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
//         return sum_s_q == -1 ? bidb * batch_stride : uint32_t(sum_s_q) * row_stride;
//     }

//     template <typename index_t>
//     __forceinline__ __device__ index_t k_offset(const index_t batch_stride, const index_t row_stride, const int bidb) const {
//         return sum_s_k == -1 ? bidb * batch_stride + leftpad_k * row_stride : uint32_t(sum_s_k + leftpad_k) * row_stride;
//     }

//     const int length;
//     const int width;
//     const int sum_s_q;
//     const int sum_s_k;
//     const int actual_seqlen_q;
//     // We have to have seqlen_k_cache declared before actual_seqlen_k, otherwise actual_seqlen_k is set to 0.
//     const int leftpad_k;
//     const int seqlen_k_cache;
//     const int actual_seqlen_k;
// };



////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
