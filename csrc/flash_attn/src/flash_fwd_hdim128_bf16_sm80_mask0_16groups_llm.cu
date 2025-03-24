// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_sparse_llm_<cutlass::bfloat16_t, 128, false, 0>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_16group_llm<cutlass::bfloat16_t, false, 0>(params, stream);
}
