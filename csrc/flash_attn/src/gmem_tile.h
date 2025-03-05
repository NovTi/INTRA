/******************************************************************************
    Adapted from: flash-attentionv1/csrc/flash_attn/src/fmha/gmem_tile.h
 ******************************************************************************/

#pragma once

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace flash {

using namespace cute;

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
            cute::custom_copy(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            cute::custom_copy(new_ptr, D, 1);
            
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

            cute::custom_copy(new_ptr, D, 0);

            // second half of the nblock
            token_idx += 16;
            in_block_token_idx = token_idx % SUBBLOCK_NUM;
            in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;
            token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy(new_ptr, D, 1);

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
            cute::custom_copy(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            cute::custom_copy(new_ptr, D, 1);
            
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

            cute::custom_copy(new_ptr, D, 0);

            // second half of the nblock
            token_idx += 16;
            in_block_token_idx = token_idx % SUBBLOCK_NUM;
            in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;
            token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy(new_ptr, D, 1);

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
            cute::custom_copy(new_ptr, D, 0);

            new_ptr = ptr + (uint32_t)((token_idx + 16) * row_stride + dim_increment) * BYTES_PER_ELEMENT;

            cute::custom_copy(new_ptr, D, 1);
            
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

            cute::custom_copy(new_ptr, D, 0);

            // second half of the nblock
            token_idx += 16;
            in_block_token_idx = token_idx % SUBBLOCK_NUM;
            in_group_token_idx = in_block_token_idx % SUBGROUP_NUM;
            token_offset = (token_idx / SUBBLOCK_NUM) * WIDTH * 4
                            + (in_block_token_idx / SUBGROUP_NUM)
                            + (in_group_token_idx / SUBGROUP_SINGLE_ROW_NUM) * WIDTH
                            + (in_group_token_idx % SUBGROUP_SINGLE_ROW_NUM) * 4;

            new_ptr = ptr + (uint32_t)((token_offset + text_length) * row_stride + dim_increment + gourp_offset) * BYTES_PER_ELEMENT;

            cute::custom_copy(new_ptr, D, 1);

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

////////////////////////////////////////////////////////////////////////////////////////////////////


}  // namespace flash