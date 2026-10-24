/******************************************************************************
    Adapted from: flash-attentionv1/csrc/flash_attn/src/fmha/gmem_tile.h
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

#include "intra_copy.h"

namespace flash {

using namespace cute;

////////////////////////////////// Token Group Loader /////////////////////////////////////////

template<
    int WIDTH,
    int kBlockN_
>
struct Gmem_tile_kv {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    static constexpr int SUBBLOCK_NUM = WIDTH;
    static constexpr int SUBGROUP_NUM = WIDTH / 2;
    static constexpr int SUBGROUP_SINGLE_ROW_NUM = WIDTH / 4;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv(void *ptr_, const Params &params, const int tidx, int batch_offset, const int outgroup_id, const int ingroup_id, const int text_blocks, const int bidh)
        : row_stride(params.k_row_stride)
        , text_length(params.text_length)
        , text_blocks(text_blocks)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = (outgroup_id * 2 * WIDTH + ingroup_id * 2) * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
        
        // ptr += (uint32_t)((outgroup_id * 2 * WIDTH + ingroup_id * 2) * row_stride + batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        if (n_block < text_blocks){
            const char *new_ptr;
            int token_idx = n_block * kBlockN + token_increment;

            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            // D: tKsK - ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
            cute::custom_copy_kv(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            cute::custom_copy_kv(new_ptr, D, 1);
            
            cute::cp_async_fence();

        } else{
            const char *new_ptr;
            // first half of the nblock
            int token_idx = (n_block - text_blocks) * kBlockN + token_increment;
            // int token_idx = n_block * kBlockN + token_increment;

            int in_block_token_idx = token_idx % SUBBLOCK_NUM;
            int in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;

            int token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            // printf("token_idx: %d | token_offset: %d | in_block_token_idx: %d | in_group_token_idx: %d | n_block: %d\n", token_idx, token_offset, in_block_token_idx, in_group_token_idx, n_block);

            cute::custom_copy_kv(new_ptr, D, 0);

            // second half of the nblock
            token_idx += 16;
            in_block_token_idx = token_idx % SUBBLOCK_NUM;
            in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;
            token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy_kv(new_ptr, D, 1);

            cute::cp_async_fence();
        }
    }

    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    const int text_blocks;
    int gourp_offset;

    // The pointer.
    char *ptr;

    int text_length;
};


template<
    int WIDTH,
    int kBlockN_
>
struct Gmem_tile_qo {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    static constexpr int SUBBLOCK_NUM = WIDTH;
    static constexpr int SUBGROUP_NUM = WIDTH / 2;
    static constexpr int SUBGROUP_SINGLE_ROW_NUM = WIDTH / 4;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo(void *ptr_, const Params &params, const int tidx, int batch_offset, const int outgroup_id, const int ingroup_id, const int text_blocks, const int bidh)
        : row_stride(params.k_row_stride)
        , text_length(params.text_length)
        , text_blocks(text_blocks)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = (outgroup_id * 2 * WIDTH + ingroup_id * 2) * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
        
        // ptr += (uint32_t)((outgroup_id * 2 * WIDTH + ingroup_id * 2) * row_stride + batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        if (n_block < text_blocks){
            const char *new_ptr;
            int token_idx = n_block * kBlockN + token_increment;

            // printf("token_idx: %d | n_block: %d | kBlockN: %d | token_increment: %d | dim_increment: %d\n", token_idx, n_block, kBlockN, token_increment, dim_increment);

            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            // D: tKsK - ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
            cute::custom_copy_qo(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            cute::custom_copy_qo(new_ptr, D, 1);
            
            cute::cp_async_fence();

        } else{
            const char *new_ptr;
            // first half of the nblock
            int token_idx = (n_block - text_blocks) * kBlockN + token_increment;
            // int token_idx = n_block * kBlockN + token_increment;

            int in_block_token_idx = token_idx % SUBBLOCK_NUM;
            int in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;

            int token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy_qo(new_ptr, D, 0);

            // second half of the nblock
            token_idx += 16;
            in_block_token_idx = token_idx % SUBBLOCK_NUM;
            in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;
            token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy_qo(new_ptr, D, 1);

            cute::cp_async_fence();
        }
    }

    template <typename Engine, typename Layout>
    inline __device__ void store(int n_block, Tensor<Engine, Layout> &D) {
        if (n_block < text_blocks){
            const char *new_ptr;
            int token_idx = n_block * kBlockN + token_increment;

            // printf("token_idx: %d | n_block: %d | kBlockN: %d | token_increment: %d | dim_increment: %d\n", token_idx, n_block, kBlockN, token_increment, dim_increment);

            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            // D: tKsK - ((_8,_1),_2,_2):((_1,_0),_1024,_2048)
            cute::custom_copy_kv(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            cute::custom_copy_kv(new_ptr, D, 1);
            
            cute::cp_async_fence();

        } else{
            const char *new_ptr;
            // first half of the nblock
            int token_idx = (n_block - text_blocks) * kBlockN + token_increment;
            // int token_idx = n_block * kBlockN + token_increment;

            int in_block_token_idx = token_idx % SUBBLOCK_NUM;
            int in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;

            int token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy_kv(new_ptr, D, 0);

            // second half of the nblock
            token_idx += 16;
            in_block_token_idx = token_idx % SUBBLOCK_NUM;
            in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;
            token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy_kv(new_ptr, D, 1);

            cute::cp_async_fence();
        }
    }

    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    const int text_blocks;
    int gourp_offset;

    // The pointer.
    char *ptr;
    // const int tidx_;

    int text_length;
};


////////////////////////////////// 8 Groups /////////////////////////////////////////


template<
    int WIDTH,
    int kBlockN_
>
struct Gmem_tile_kv_8group {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    static constexpr int group_w = 4;
    static constexpr int group_h = 2;
    static constexpr int NUM_ELEMENT_PER_ROW = WIDTH / group_w;
    static_assert(NUM_ELEMENT_PER_ROW % 16 == 0, "each 16-token chunk must stay within one image row");

    // Image offset of the g-th token of a group. A group is a 2 x 4 sub-lattice, i.e. WIDTH / 4
    // tokens on every second image row; the group's (outgroup, ingroup) origin is gourp_offset.
    static inline __device__ int image_offset(int g) {
        return (g / NUM_ELEMENT_PER_ROW) * group_h * WIDTH + (g % NUM_ELEMENT_PER_ROW) * group_w;
    }

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv_8group(void *ptr_, const Params &params, const int tidx, int batch_offset, const int outgroup_id, const int ingroup_id, const int text_blocks, const int bidh)
        : row_stride(params.k_row_stride)
        , text_length(params.text_length)
        , text_blocks(text_blocks)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {
        
        // 8 groups： 2 x 4
        gourp_offset = (outgroup_id * WIDTH + ingroup_id) * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        if (n_block < text_blocks){
            const char *new_ptr;
            int token_idx = n_block * kBlockN + token_increment;

            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);

            new_ptr = ptr + (uint32_t)((token_idx + 32) * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);

            new_ptr = ptr + (uint32_t)((token_idx + 48) * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);

            cute::cp_async_fence();

        } else{
            const char *new_ptr;
            // first half of the nblock
            // int token_idx = (n_block - text_blocks) * kBlockN + token_increment;
            // int token_offset = (token_idx / NUM_ELEMENT_PER_ROW) * group_h + (token_idx % NUM_ELEMENT_PER_ROW) * group_w;

            // 4 chunks of 16 group tokens
            const int g = (n_block - text_blocks) * kBlockN + token_increment;
            #pragma unroll
            for (int i = 0; i < 4; i++){
                new_ptr = ptr + (uint32_t)((image_offset(g + 16 * i) + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
                cute::custom_copy_kv(new_ptr, D, i);
            }

            cute::cp_async_fence();
        }
    }

    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    const int text_blocks;
    int gourp_offset;

    // The pointer.
    char *ptr;

    int text_length;
};



template<
    int WIDTH,
    int kBlockM_
>
struct Gmem_tile_qo_8group {

    static constexpr int kBlockM = kBlockM_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    static constexpr int group_w = 4;
    static constexpr int group_h = 2;
    static constexpr int NUM_ELEMENT_PER_ROW = WIDTH / group_w;
    static_assert(NUM_ELEMENT_PER_ROW % 16 == 0, "each 16-token chunk must stay within one image row");

    // Image offset of the g-th token of a group. A group is a 2 x 4 sub-lattice, i.e. WIDTH / 4
    // tokens on every second image row; the group's (outgroup, ingroup) origin is gourp_offset.
    static inline __device__ int image_offset(int g) {
        return (g / NUM_ELEMENT_PER_ROW) * group_h * WIDTH + (g % NUM_ELEMENT_PER_ROW) * group_w;
    }

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo_8group(void *ptr_, const Params &params, const int tidx, int batch_offset, const int outgroup_id, const int ingroup_id, const int text_blocks, const int bidh)
        : row_stride(params.k_row_stride)
        , text_length(params.text_length)
        , text_blocks(text_blocks)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {
        
        // 8 groups： 2 x 4
        text_group_offset = (outgroup_id * group_w + ingroup_id) * 64 * row_stride;
        gourp_offset = (outgroup_id * WIDTH + ingroup_id) * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int m_block, Tensor<Engine, Layout> &D) {
        if (m_block < text_blocks){
            const char *new_ptr;

            int token_idx = m_block * kBlockM + token_increment;
            new_ptr = ptr + (uint32_t)(token_idx * row_stride + text_group_offset + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 0);

            #pragma unroll
            for (int i=1; i < 4; i++){
                token_idx += 16;
                new_ptr = ptr + (uint32_t)(token_idx * row_stride + text_group_offset + dim_increment) * BYTES_PER_ELEMENT;
                cute::custom_copy_qo(new_ptr, D, i);
            }

            cute::cp_async_fence();

        } else{
            const char *new_ptr;

            // 4 chunks of 16 group tokens
            const int g = (m_block - text_blocks) * kBlockM + token_increment;
            #pragma unroll
            for (int i = 0; i < 4; i++){
                new_ptr = ptr + (uint32_t)((image_offset(g + 16 * i) + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
                cute::custom_copy_qo(new_ptr, D, i);
            }

            cute::cp_async_fence();
        }
    }

    template <typename Engine, typename Layout>
    inline __device__ void store(int m_block, Tensor<Engine, Layout> &S) {
        if (m_block < text_blocks){
            char *new_ptr;

            int token_idx = m_block * kBlockM + token_increment;
            new_ptr = ptr + (uint32_t)(token_idx * row_stride + text_group_offset + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);

            #pragma unroll
            for (int i=1; i < 4; i++){
                token_idx += 16;
                new_ptr = ptr + (uint32_t)(token_idx * row_stride + text_group_offset + dim_increment) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }

            cute::cp_async_fence();

        } else{
            char *new_ptr;

            // 4 chunks of 16 group tokens
            const int g = (m_block - text_blocks) * kBlockM + token_increment;
            #pragma unroll
            for (int i = 0; i < 4; i++){
                new_ptr = ptr + (uint32_t)((image_offset(g + 16 * i) + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }

            cute::cp_async_fence();
        }
    }

    

    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    const int text_blocks;
    int gourp_offset;
    int text_group_offset;

    // The pointer.
    char *ptr;

    int text_length;
};

////////////////////////////////// 8 Groups /////////////////////////////////////////



////////////////////////////////// llm Loader /////////////////////////////////////////


template<
    int kBlockN_
>
struct Gmem_tile_kv_llm_mask0 {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv_llm_mask0(void *ptr_, const Params &params, const int tidx, int batch_offset, const int group_id, const int seq_length_, const int bidh)
        : row_stride(params.k_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , group_num(params.group_num)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8) {
        
        // 8 groups： 2 x 4
        gourp_offset = group_id * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int token_offset = n_block * kBlockN + token_increment;

        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 0);
        } else {
            cute::custom_zero_kv(D, 0);
        }

        token_offset += 16;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        } else {
            cute::custom_zero_kv(D, 1);
        }

        token_offset += 16;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        } else {
            cute::custom_zero_kv(D, 2);
        }


        token_offset += 16;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        } else {
            cute::custom_zero_kv(D, 3);
        }

        cute::cp_async_fence();
    }

    const uint32_t row_stride;
    const int group_num;
    const int token_increment;
    const int dim_increment;
    int gourp_offset;
    int seq_length;

    // The pointer.
    char *ptr;

};


template<
    int kBlockN_
>
struct Gmem_tile_kv_llm_mask1 {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv_llm_mask1(void *ptr_, const Params &params, const int tidx, int batch_offset, const int group_id, const int seq_length_, const int bidh)
        : row_stride(params.k_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , group_num(params.group_num)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = group_id * group_num * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int ingroup_offset = n_block * kBlockN + token_increment;
        // 1024 = 256 * 4 | only work for group_num = 16 since single load if 4 time of 256
        int real_offset = n_block * 1024 + token_increment;
        new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
        cute::custom_copy_kv(new_ptr, D, 0);

        ingroup_offset += 16;
        // 256 = 16 * 16 only for group_num = 16
        real_offset += 256;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        }

        ingroup_offset += 16;
        real_offset += 256;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        }

        ingroup_offset += 16;
        real_offset += 256;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        }

        cute::cp_async_fence();
    }

    const uint32_t row_stride;
    const int group_num;
    const int token_increment;
    const int dim_increment;
    int gourp_offset;
    int seq_length;

    // The pointer.
    char *ptr;

};



template<
    int kBlockM_
>
struct Gmem_tile_qo_llm_mask0 {
    static constexpr int kBlockM = kBlockM_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo_llm_mask0(void *ptr_, const Params &params, const int tidx, int batch_offset, const int group_id, const int seq_length_, const int bidh)
        : row_stride(params.q_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , group_num(params.group_num)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = group_id * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int m_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int token_offset = m_block * kBlockM + token_increment;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 0);
        } else {
            cute::custom_zero_kv(D, 0);
        }

        token_offset += 16;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        } else {
            cute::custom_zero_kv(D, 1);
        }

        token_offset += 16;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        } else {
            cute::custom_zero_kv(D, 2);
        }

        token_offset += 16;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        } else {
            cute::custom_zero_kv(D, 3);
        }

        cute::cp_async_fence();
    }


    template <typename Engine, typename Layout>
    inline __device__ void store(int m_block, Tensor<Engine, Layout> &S) {
        char *new_ptr;

        int token_offset = m_block * kBlockM + token_increment;
        if (token_offset < seq_length){
            new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);
        }

        #pragma unroll
        for (int i=1; i<4; i++){
            token_offset += 16;

            if (token_offset < seq_length){
                new_ptr = ptr + (uint32_t)(token_offset * group_num * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }
            
        }

        cute::cp_async_fence();
    }

    const uint32_t row_stride;
    const int group_num;
    const int token_increment;
    const int dim_increment;
    int gourp_offset;
    int seq_length;

    // The pointer.
    char *ptr;

};



template<
    int kBlockM_
>
struct Gmem_tile_qo_llm_mask1 {

    static constexpr int kBlockM = kBlockM_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo_llm_mask1(void *ptr_, const Params &params, const int tidx, int batch_offset, const int group_id, const int seq_length_, const int bidh)
        : row_stride(params.q_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , group_num(params.group_num)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = group_id * group_num * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int m_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int ingroup_offset = m_block * kBlockM + token_increment;
        // 1024 = 256 * 4 | only work for group_num = 16 since single load if 4 time of 256
        int real_offset = m_block * 1024 + token_increment;
        new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
        cute::custom_copy_kv(new_ptr, D, 0);

        ingroup_offset += 16;
        // 256 = 16 * 16 only for group_num = 16
        real_offset += 256;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        }

        ingroup_offset += 16;
        real_offset += 256;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        }

        ingroup_offset += 16;
        real_offset += 256;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        }

        cute::cp_async_fence();
    }

    template <typename Engine, typename Layout>
    inline __device__ void store(int m_block, Tensor<Engine, Layout> &S) {
        char *new_ptr;

        int ingroup_offset = m_block * kBlockM + token_increment;
        int real_offset = m_block * 1024 + token_increment;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);
        }

        #pragma unroll
        for (int i=1; i<4; i++){
            ingroup_offset += 16;
            real_offset += 256;

            if (ingroup_offset < seq_length){
                new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }
            
        }

        cute::cp_async_fence();
    }


    const uint32_t row_stride;
    const int group_num;
    const int token_increment;
    const int dim_increment;
    int gourp_offset;
    int seq_length;

    // The pointer.
    char *ptr;

};



////////////////////////////////// 8 Groups /////////////////////////////////////////


template<
    int kBlockN_
>
struct Gmem_tile_kv_llm_mask1_8groups {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv_llm_mask1_8groups(void *ptr_, const Params &params, const int tidx, int batch_offset, const int group_id, const int seq_length_, const int bidh)
        : row_stride(params.k_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , group_num(params.group_num)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = group_id * group_num * row_stride;
        ptr += (uint32_t)(batch_offset + params.d * bidh + ((tidx / 64) * 64 + (tidx % 64) / 8) * row_stride) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int ingroup_offset = n_block * kBlockN + token_increment;
        // 512 = 128 * 4 | only work for group_num = 8 since single load is 4 time of 128
        // each load consists of two 8 element chunks, each chunk is of size 64, so two chunks are 128
        int real_offset = n_block * 512;
        new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
        cute::custom_copy_kv(new_ptr, D, 0);

        ingroup_offset += 16;
        // 256 = 16 * 16 only for group_num = 16
        real_offset += 128;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        }

        ingroup_offset += 16;
        real_offset += 128;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        }

        ingroup_offset += 16;
        real_offset += 128;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        }

        cute::cp_async_fence();
    }

    const uint32_t row_stride;
    const int group_num;
    const int token_increment;
    const int dim_increment;
    int gourp_offset;
    int seq_length;

    // The pointer.
    char *ptr;

};



template<
    int kBlockM_
>
struct Gmem_tile_qo_llm_mask1_8groups {

    static constexpr int kBlockM = kBlockM_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo_llm_mask1_8groups(void *ptr_, const Params &params, const int tidx, int batch_offset, const int group_id, const int seq_length_, const int bidh)
        : row_stride(params.q_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , group_num(params.group_num)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8) {
        
        gourp_offset = group_id * group_num * row_stride;
        // (tidx / 64) * 64 * row_stride is for 8 groups.
        // Each time loads 16 elements but we only have the 8 coutigous elements.
        // So each load we have to split the tidx into two parts.

        // (tidx % 64) / 8 is the token increment within the 8 element chunk
        ptr += (uint32_t)(batch_offset + params.d * bidh + ((tidx / 64) * 64 + (tidx % 64) / 8) * row_stride) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int m_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int ingroup_offset = m_block * kBlockM + token_increment;
        // 512 = 128 * 4 | only work for group_num = 8 since single load is 4 time of 128
        // each load consists of two 8 element chunks, each chunk is of size 64, so two chunks are 128
        int real_offset = m_block * 512;
        new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
        cute::custom_copy_kv(new_ptr, D, 0);

        ingroup_offset += 16;
        // 128 = 8 * 8 * 2 only for group_num = 8
        real_offset += 128;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        }

        ingroup_offset += 16;
        real_offset += 128;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        }

        ingroup_offset += 16;
        real_offset += 128;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        }

        cute::cp_async_fence();
    }

    template <typename Engine, typename Layout>
    inline __device__ void store(int m_block, Tensor<Engine, Layout> &S) {
        char *new_ptr;

        int ingroup_offset = m_block * kBlockM + token_increment;
        int real_offset = m_block * 512;
        if (ingroup_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);
        }

        #pragma unroll
        for (int i=1; i<4; i++){
            ingroup_offset += 16;
            real_offset += 128;

            if (ingroup_offset < seq_length){
                new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }
            
        }

        cute::cp_async_fence();
    }


    const uint32_t row_stride;
    const int group_num;
    const int token_increment;
    const int dim_increment;
    int gourp_offset;
    int seq_length;

    // The pointer.
    char *ptr;

};




template<
    int kBlockN_
>
struct Gmem_tile_kv_llm_sw {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv_llm_sw(void *ptr_, const Params &params, const int tidx, int batch_offset, const int start, const int seq_length_, const int bidh)
        : row_stride(params.k_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , seq_length(seq_length_)
        , dim_increment((tidx % 8) * 8)
        , start(start) {
        
        
        ptr += (uint32_t)(batch_offset + params.d * bidh + start * row_stride) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        const char *new_ptr;

        int offset = n_block * kBlockN + token_increment;
        int real_offset = start + offset;
        // a single load spans across 16 elements since each thread account for 8 elements
        // The head dim is 128, split to half we get 8 threads account for one element
        // 128 block size loads 16 elements in total

        // INTRA fix (Bug B): chunk 0 was loaded unguarded.  When the window
        // runs past the end of the sequence (S not a multiple of kBlockM) this
        // read is out of range, so guard it like the other three chunks and
        // zero-fill instead of leaving stale shared memory.
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 0);
        } else {
            cute::custom_zero_kv(D, 0);
        }

        offset += 16;
        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);
        } else {
            cute::custom_zero_kv(D, 1);
        }

        offset += 16;
        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);
        } else {
            cute::custom_zero_kv(D, 2);
        }

        offset += 16;
        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
        } else {
            cute::custom_zero_kv(D, 3);
        }

        cute::cp_async_fence();
    }

    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    int seq_length;
    int start;

    // The pointer.
    char *ptr;

};



template<
    int kBlockM_
>
struct Gmem_tile_qo_llm_sw {

    static constexpr int kBlockM = kBlockM_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo_llm_sw(void *ptr_, const Params &params, const int tidx, int batch_offset, const int m_block, const int seq_length_, const int bidh)
        : row_stride(params.q_row_stride)
        , ptr(reinterpret_cast<char *>(ptr_))
        , seq_length(seq_length_)
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {

        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int m_block, Tensor<Engine, Layout> &D) {
        char *new_ptr;
        int real_offset = kBlockM * m_block + token_increment;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 0);
        } else {
            cute::custom_zero_kv(D, 0);
        }

        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 1);
        } else {
            cute::custom_zero_kv(D, 1);
        }

        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 2);
        } else {
            cute::custom_zero_kv(D, 2);
        }

        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 3);
        } else {
            cute::custom_zero_kv(D, 3);
        }

        cute::cp_async_fence();
    }

    template <typename Engine, typename Layout>
    inline __device__ void store(int m_block, Tensor<Engine, Layout> &S) {
        char *new_ptr;
        int real_offset = kBlockM * m_block + token_increment;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);
        }

        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 8);
        }

        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 16);
        }


        real_offset += 16;
        if (real_offset < seq_length){
            new_ptr = ptr + (uint32_t)(real_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 24);
        }

        cute::cp_async_fence();
    }


    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    int seq_length;

    // The pointer.
    char *ptr;
};



////////////////////////////////// Sliding Window Loader /////////////////////////////////////////

template<
    int WIDTH,
    int kBlockN_
>
struct Gmem_tile_kv_sw {

    static constexpr int kBlockN = kBlockN_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    static constexpr int q_tile_w = 8;
    static constexpr int q_tile_h = 8;
    static constexpr int window_w = 16;
    static constexpr int window_h = 16;
    static constexpr int expand_size = 4;
    static constexpr int num_m_blocks_row = WIDTH / q_tile_w;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_kv_sw(void *ptr_, const Params &params, const int tidx, int batch_offset, const int text_blocks, const int text_blocks_q, const int m_block, const int bidh)
        : row_stride(params.k_row_stride)
        , text_blocks(text_blocks)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {
        
        if (m_block >= text_blocks){
            // square latent grid: WIDTH x WIDTH
            int total_q_row = WIDTH / q_tile_h;
            int total_q_col = WIDTH / q_tile_w;

            int q_block_row = (m_block - text_blocks_q) / num_m_blocks_row;
            int q_block_col = (m_block - text_blocks_q) % num_m_blocks_row;

            int offset_y = q_block_row * q_tile_h;
            int offset_x = q_block_col * q_tile_w;

            // get the window x/y offset
            if (q_block_row == 0) {
                ;
            } else if (q_block_row == total_q_row - 1) {
                offset_y -= expand_size*2;
            } else {
                offset_y -= expand_size;
            }

            if (q_block_col == 0) {
                ;
            } else if (q_block_col == total_q_col - 1) {
                offset_x -= expand_size * 2;
            } else {
                offset_x -= expand_size;
            }

            window_offset = (params.text_length + offset_y * WIDTH + offset_x) * row_stride;
        }


        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
        
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int n_block, Tensor<Engine, Layout> &D) {
        if (n_block < text_blocks){
            const char *new_ptr;
            int token_idx = n_block * kBlockN + token_increment;

            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);

            new_ptr = ptr + (uint32_t)((token_idx + 32) * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);

            new_ptr = ptr + (uint32_t)((token_idx + 48) * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);
            
            cute::cp_async_fence();

        } else{
            const char *new_ptr;
            // single n_block loads across 4 rows of the feature matrix
            int token_offset = (n_block - text_blocks) * WIDTH * 4 + token_increment;

            new_ptr = ptr + (uint32_t)(window_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 0);

            token_offset += WIDTH;
            new_ptr = ptr + (uint32_t)(window_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 1);

            token_offset += WIDTH;
            new_ptr = ptr + (uint32_t)(window_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 2);

            token_offset += WIDTH;
            new_ptr = ptr + (uint32_t)(window_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_kv(new_ptr, D, 3);

            cute::cp_async_fence();
        }
    }

    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    const int text_blocks;
    int window_offset;

    // The pointer.
    char *ptr;
};


template<
    int WIDTH,
    int kBlockM_
>
struct Gmem_tile_qo_sw {

    static constexpr int kBlockM = kBlockM_;
    static constexpr int BYTES_PER_ELEMENT = 2;

    static constexpr int tile_w = 8;
    static constexpr int tile_h = 8;
    static constexpr int num_m_blocks_row = WIDTH / tile_w;

    // Ctor.
    template<typename Params>
    inline __device__ Gmem_tile_qo_sw(void *ptr_, const Params &params, const int tidx, int batch_offset, const int text_blocks, const int m_block, const int bidh)
        : row_stride(params.q_row_stride)
        , text_blocks(text_blocks)
        , ptr(reinterpret_cast<char *>(ptr_))
        , token_increment(tidx / 8)
        , dim_increment((tidx % 8) * 8) {
        
        if (m_block >= text_blocks){
            int offset_y = ((m_block - text_blocks) / num_m_blocks_row) * tile_h;
            int offset_x = ((m_block - text_blocks) % num_m_blocks_row) * tile_w;

            int q_block_row = (m_block - text_blocks) / num_m_blocks_row;
            int q_block_col = (m_block - text_blocks) % num_m_blocks_row;

            tile_offset = (params.text_length + offset_y * WIDTH + offset_x) * row_stride;
        }

        ptr += (uint32_t)(batch_offset + params.d * bidh) * BYTES_PER_ELEMENT;
        
    }

    template <typename Engine, typename Layout>
    inline __device__ void load(int m_block, Tensor<Engine, Layout> &D) {
        if (m_block < text_blocks){
            const char *new_ptr;

            int token_idx = m_block * 64 + token_increment;
            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 0);

            #pragma unroll
            for (int i=1; i < 4; i++){
                token_idx += 16;
                new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;
                cute::custom_copy_qo(new_ptr, D, i);
            }

            cute::cp_async_fence();

        } else{
            const char *new_ptr;
            int token_offset = (token_increment / tile_w) * WIDTH + (token_increment % tile_w);
            new_ptr = ptr + (uint32_t)(tile_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_qo(new_ptr, D, 0);

            #pragma unroll
            for (int i=1; i<4; i++){
                token_offset += WIDTH * 2;
                new_ptr = ptr + (uint32_t)(tile_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
                cute::custom_copy_qo(new_ptr, D, i);
            }

            cute::cp_async_fence();
        }
    }

    template <typename Engine, typename Layout>
    inline __device__ void store(int m_block, Tensor<Engine, Layout> &S) {
        if (m_block < text_blocks){
            char *new_ptr;

            int token_idx = m_block * 64 + token_increment;
            new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);

            #pragma unroll
            for (int i=1; i < 4; i++){
                token_idx += 16;
                new_ptr = ptr + (uint32_t)(token_idx * row_stride + dim_increment) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }

            cute::cp_async_fence();

        } else{
            char *new_ptr;
            int token_offset = (token_increment / tile_w) * WIDTH + (token_increment % tile_w);
            new_ptr = ptr + (uint32_t)(tile_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
            cute::custom_copy_o_r2g(S, new_ptr, 0);

            #pragma unroll
            for (int i=1; i<4; i++){
                token_offset += WIDTH * 2;
                new_ptr = ptr + (uint32_t)(tile_offset + token_offset * row_stride + dim_increment) * BYTES_PER_ELEMENT;
                cute::custom_copy_o_r2g(S, new_ptr, i*8);
            }

            cute::cp_async_fence();
        }
    }


    const uint32_t row_stride;
    const int token_increment;
    const int dim_increment;
    const int text_blocks;
    int tile_offset;

    // The pointer.
    char *ptr;
};


////////////////////////////////////////////////////////////////////////////////////////////////////


}  // namespace flash