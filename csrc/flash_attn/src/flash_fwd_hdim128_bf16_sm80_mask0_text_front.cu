// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_sparse_text_front_<cutlass::bfloat16_t, 128, false, 0>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim128_mask0_text_front<cutlass::bfloat16_t, false>(params, stream);
}
