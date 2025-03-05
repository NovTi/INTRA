// Copyright (c) 2024, Tri Dao.
// Splitting the different head dimensions to different files to speed up compilation.
// This file is auto-generated. See "generate_kernels.py"

#include "flash_fwd_launch_template.h"

template<>
void run_mha_fwd_sparse_text_front_<float, 64, false, 0>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_hdim64_mask0_text_front<float, false>(params, stream);
}
