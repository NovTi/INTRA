import torch
import torch.nn as nn
import torch.nn.functional as F
import sys

import time
import math
import numpy as np
from einops import rearrange, repeat
from tqdm import tqdm

from flash_attn import flash_attn_func



def _init_qkv():
    qkv = torch.zeros(60 * 512, dtype=torch.float16, device='cuda')
    for i in range(60):
        for j in range(512):
            row_bin = bin(i)[2:].rjust(6, '0')
            col_bin = bin(j)[2:].rjust(9, '0')
            
            # # Example bit patterns for float16 values
            # bit_patterns = [0b0000000100000001]  # These are just examples
    
            # # Convert bit patterns to float16 using numpy
            # float16_values = np.array(bit_patterns, dtype=np.uint16).view(np.float16)

            # # Assign these float16 values to the torch tensor
            # fp16_array[:] = torch.tensor(float16_values, dtype=torch.float16)

            # Step 1: Convert the string to an integer
            bit_pattern_str = f'0b0{col_bin}{row_bin}'
            bit_pattern = int(bit_pattern_str, 2)

            # Step 2: Convert the bit pattern to float16 using numpy
            float16_value = np.array([bit_pattern], dtype=np.uint16).view(np.float16)

            # Step 3: Assign the float16 value to a torch tensor
            qkv[i*512 + j] = torch.tensor(float16_value, dtype=torch.float16)[0]
            # print(f"{qkv[i*128 + j]:.20f}")

    return qkv[:20*512].view(20, 4, 128), qkv[20*512:40*512].view(20, 4, 128), qkv[40*512:].view(20, 4, 128)

def _init_qkv_mma():
    q = torch.zeros((1, 4096, 1, 64), dtype=torch.float16, device='cuda')
    k = torch.zeros((1, 4096, 1, 64), dtype=torch.float16, device='cuda')
    v = torch.zeros((1, 4096, 1, 64), dtype=torch.float16, device='cuda')

    for i in range(64):
        k[0][-64*8+i][0][i] = 1

    for i in range(64):
        for j in range(512):
            col_bin = bin(i)[2:].rjust(6, '0')
            row_bin = bin(j)[2:].rjust(9, '0')

            # Step 1: Convert the string to an integer
            bit_pattern_str = f'0b0{row_bin}{col_bin}'
            bit_pattern = int(bit_pattern_str, 2)

            # Step 2: Convert the bit pattern to float16 using numpy
            float16_value = np.array([bit_pattern], dtype=np.uint16).view(np.float16)

            # Step 3: Assign the float16 value to a torch tensor
            q[0, j, 0, i] = torch.tensor(float16_value, dtype=torch.float16)[0]

    # print(f"{q[0, 0, 0, 0]:.20f}")

    return q, k, v


def _init_qkv_max_test_k32_helper_b0(k):
    # block 0
    # 7 
    k[2][0][0] = 2
    k[7][0][0] = 3
    # 44 (12 + 32)
    k[12][0][1] = 5
    k[9][0][1] = 4
    # 87 (23 + 32 * 2)
    k[5][0][2] = 6
    k[23][0][2] = 7
    # 117 (21 + 32 * 3)
    k[30][0][3] = 8
    k[21][0][3] = 9
    # 132 (4 + 32 * 4)
    k[4][0][4] = 11
    k[13][0][4] = 10
    # 182 (22 + 32 * 5)
    k[19][0][5] = 12
    k[22][0][5] = 13
    # 219 (27 + 32 * 6)
    k[16][0][6] = 14
    k[27][0][6] = 15
    # 255 (31 + 32 * 7)
    k[14][0][7] = 16
    k[31][0][7] = 17
    # 285 (29 + 32 * 8)
    k[17][0][8] = 18
    k[29][0][8] = 19
    # 313 (25 + 32 * 9) 
    k[18][0][9] = 20
    k[25][0][9] = 21
    # 348 (28 + 32 * 10)
    k[20][0][10] = 22
    k[28][0][10] = 23
    # 376 (24 + 32 * 11)
    k[24][0][11] = 25
    k[26][0][11] = 24

def _init_qkv_max_test_k32_helper_b1(k):
    # block 1
    # 4100 (4 + 32 * 0 + 128*32)
    k[4][1][0] = 27
    k[0][1][0] = 26
    # 4145 (17 + 32 * 1 + 128*32)
    k[17][1][1] = 29
    k[21][1][1] = 28
    # 4175 (15 + 32 * 2 + 128*32)
    k[8][1][2] = 30
    k[15][1][2] = 31
    # 4218 (26 + 32 * 3 + 128*32)
    k[31][1][3] = 32
    k[26][1][3] = 33
    # 4234 (10 + 32 * 4 + 128*32)
    k[7][1][4] = 34
    k[10][1][4] = 35
    # 4278 (22 + 32 * 5 + 128*32)
    k[13][1][5] = 36
    k[22][1][5] = 37

def _init_qkv_max_test_k32():
    q = torch.zeros((512, 2, 128), dtype=torch.float16, device='cuda')
    for i in range(128):
        q[i][0][i] = 1

    k = torch.zeros((32, 2, 128), dtype=torch.float16, device='cuda')

    v = torch.zeros((32, 2, 128), dtype=torch.float16, device='cuda')

    _init_qkv_max_test_k32_helper_b0(k)

    _init_qkv_max_test_k32_helper_b1(k)

    return q, k, v


def _init_qkv_max_test_k64():
    q = torch.zeros((128, 2, 128), dtype=torch.float16, device='cuda')
    for i in range(128):
        q[i][0][i] = 1

    k = torch.zeros((128, 2, 128), dtype=torch.float16, device='cuda')

    v = torch.zeros((128, 2, 128), dtype=torch.float16, device='cuda')

    k[2][0][0] = 2
    k[7][0][0] = 3

    k[12][0][1] = 5
    k[9][0][1] = 4

    k[5][0][2] = 6
    k[23][0][2] = 7

    k[30][0][3] = 8
    k[21][0][3] = 9

    k[4][0][4] = 11
    k[13][0][4] = 10

    k[19][0][5] = 12
    k[22][0][5] = 13

    k[16][0][6] = 12
    k[27][0][6] = 13

    k[14][0][6] = 14
    k[39][0][6] = 15

    return q, k, v


def pos(torch_array):
    np_uint16 = torch_array.cpu().numpy().view(np.uint16)
    binary_string = np.binary_repr(np_uint16, width=16)

    col = int(binary_string[1:10], 2)
    row = int(binary_string[-6:], 2)

    return (row, col)


def attention_ref(
    q,
    k,
    v,
    mask=None,
    query_padding_mask=None,
    key_padding_mask=None,
    attn_bias=None,
    dropout_p=0.0,
    dropout_mask=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite window size
    softcap=0.0,
    upcast=True,
    reorder_ops=False,
    key_leftpad=None,
):
    if causal:
        window_size = (window_size[0], 0)
    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]

    scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)

    scores_check = torch.einsum("bthd,bshd->bhts", q, k)
    scores_check = scores_check + mask

    # scores_check[0, 0, 0, 0]
    # scores_check[0, 0, 0, 16]
    # import pdb; pdb.set_trace()

    if mask is not None:
        scores = scores + mask

    attention = torch.softmax(scores, dim=-1).to(v.dtype)
    # Some rows might be completely masked out so we fill them with zero instead of NaN
    if window_size[0] >= 0 or window_size[1] >= 0:
        attention = attention.masked_fill(torch.all(local_mask, dim=-1, keepdim=True), 0.0)
    # We want to mask here so that the attention matrix doesn't have any NaNs
    # Otherwise we'll get NaN in dV
    if query_padding_mask is not None:
        attention = attention.masked_fill(rearrange(~query_padding_mask, "b s -> b 1 s 1"), 0.0)
    dropout_scaling = 1.0 / (1 - dropout_p)
    # attention_drop = attention.masked_fill(~dropout_mask, 0.0) * dropout_scaling
    # output = torch.einsum('bhts,bshd->bthd', attention_drop , v)
    if dropout_mask is not None:
        attention_drop = attention.masked_fill(~dropout_mask, 0.0)
    else:
        attention_drop = attention
    output = torch.einsum("bhts,bshd->bthd", attention_drop, v * dropout_scaling)

    if query_padding_mask is not None:
        output.masked_fill_(rearrange(~query_padding_mask, "b s -> b s 1 1"), 0.0)

    return output.to(dtype=dtype_og)


def build_mask(mask):
    sequence_length = mask.shape[-1]
    h = int(sequence_length ** 0.5)
    patch_size = 1

    # assign circle
    for patch_row in range(h):
        if patch_row % 2 == 0:
            for patch_col in range(h):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, 0, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign cross
    for patch_row in range(h):
        if patch_row % 2 == 0:
            for patch_col in range(h):
                if patch_col % 2 == 1:
                    for idx in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign triangle
    for patch_row in range(h):
        if patch_row % 2 == 1:
            for patch_col in range(h):
                if patch_col % 2 == 0:
                    for idx in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+0, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign square
    for patch_row in range(h):
        if patch_row % 2 == 1:
            for patch_col in range(h):
                if patch_col % 2 == 1:
                    for idx in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+patch_size, within_patch_row_start:within_patch_row_start+patch_size] = 0

    for idx in range(sequence_length):
        if idx not in [0, patch_size, patch_size*h+0, patch_size*h+patch_size]:
            patch_row = idx // h
            patch_col = idx % h

            if patch_row % 2 == 0 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, 0, :]
            elif patch_row % 2 == 0 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+0, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+patch_size, :]


def make_mask_0(sequence_length, mask):
    h = int(sequence_length ** 0.5)
    patch_size = 1

    # assign circle
    for patch_row in range(h):
        if patch_row % 2 == 0:
            for patch_col in range(h):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, 0, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign cross
    for patch_row in range(h):
        if patch_row % 2 == 0:
            for patch_col in range(h):
                if patch_col % 2 == 1:
                    for idx in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign triangle
    for patch_row in range(h):
        if patch_row % 2 == 1:
            for patch_col in range(h):
                if patch_col % 2 == 0:
                    for idx in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+0, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign square
    for patch_row in range(h):
        if patch_row % 2 == 1:
            for patch_col in range(h):
                if patch_col % 2 == 1:
                    for idx in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+patch_size, within_patch_row_start:within_patch_row_start+patch_size] = 0

    for idx in range(sequence_length):
        if idx not in [0, patch_size, patch_size*h+0, patch_size*h+patch_size]:
            patch_row = idx // h
            patch_col = idx % h

            if patch_row % 2 == 0 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, 0, :]
            elif patch_row % 2 == 0 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+0, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+patch_size, :]

    return mask


def make_mask_1(sequence_length, mask):
    h = int(sequence_length ** 0.5)
    patch_size = 2

    # assign circle
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 0:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, 0, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign cross
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 0:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 1:
                    for row_increment in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign triangle
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 1:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+0, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign square
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 1:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 1:
                    for row_increment in range(patch_size):
                        within_patch_row_start = (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+patch_size, within_patch_row_start:within_patch_row_start+patch_size] = 0

    for idx in range(sequence_length):
        if idx not in [0, patch_size, patch_size*h+0, patch_size*h+patch_size]:
            patch_row = (idx // h) // patch_size
            patch_col = (idx % h) // patch_size

            if patch_row % 2 == 0 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, 0, :]
            elif patch_row % 2 == 0 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+0, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+patch_size, :]

    return mask



def make_mask_0_flux(sequence_length, mask):
    # W = int((sequence_length // 2) ** 0.5)
    W = 64
    L = 64
    text_length = 512

    patch_size = 1

    # all tokens attend to all prompts
    mask[:, :, :, :512] = 0

    # assign circle
    for patch_row in range(L):
        if patch_row % 2 == 0:
            for patch_col in range(W):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*W+patch_col*patch_size
                        mask[:, :, 0+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0
    

    # assign cross
    for patch_row in range(L):
        if patch_row % 2 == 0:
            for patch_col in range(W):
                if patch_col % 2 == 1:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*W+patch_col*patch_size
                        mask[:, :, patch_size+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign triangle
    for patch_row in range(L):
        if patch_row % 2 == 1:
            for patch_col in range(W):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*W+patch_col*patch_size
                        mask[:, :, patch_size*W+0+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign square
    for patch_row in range(L):
        if patch_row % 2 == 1:
            for patch_col in range(W):
                if patch_col % 2 == 1:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*W+patch_col*patch_size
                        mask[:, :, patch_size*W+patch_size+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    for idx in range(4096):
        if idx not in [0, patch_size, patch_size*W+0, patch_size*W+patch_size]:
            patch_row = idx // W
            patch_col = idx % W

            if patch_row % 2 == 0 and patch_col % 2 == 0:
                mask[:, :, idx+text_length, :] = mask[:, :, 0+text_length, :]
            elif patch_row % 2 == 0 and patch_col % 2 == 1:
                mask[:, :, idx+text_length, :] = mask[:, :, patch_size+text_length, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 0:
                mask[:, :, idx+text_length, :] = mask[:, :, patch_size*W+0+text_length, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 1:
                mask[:, :, idx+text_length, :] = mask[:, :, patch_size*W+patch_size+text_length, :]

    for idx in range(text_length):
        if idx % 4 == 0:
            mask[:, :, idx, :] = mask[:, :, 0+text_length, :]
        elif idx % 4 == 1:
            mask[:, :, idx, :] = mask[:, :, patch_size+text_length, :]
        elif idx % 4 == 2:
            mask[:, :, idx, :] = mask[:, :, patch_size*W+0+text_length, :]
        elif idx % 4 == 3:
            mask[:, :, idx, :] = mask[:, :, patch_size*W+patch_size+text_length, :]

    return mask


def make_mask_1_flux(sequence_length, mask):
    h = 64
    W = 64
    patch_size = 2

    text_length = 512
    mask[:, :, :, :512] = 0

    # assign circle
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 0:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, 0+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign cross
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 0:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 1:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign triangle
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 1:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 0:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+0+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign square
    for patch_row in range(int(h / patch_size)):
        if patch_row % 2 == 1:
            for patch_col in range(int(h / patch_size)):
                if patch_col % 2 == 1:
                    for row_increment in range(patch_size):
                        within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                        mask[:, :, patch_size*h+patch_size+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    for idx in range(4096):
        if idx not in [0, patch_size, patch_size*h+0, patch_size*h+patch_size]:
            patch_row = (idx // h) // patch_size
            patch_col = (idx % h) // patch_size

            idx = idx + text_length
            if patch_row % 2 == 0 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, 0+text_length, :]
            elif patch_row % 2 == 0 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size+text_length, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+0+text_length, :]
            elif patch_row % 2 == 1 and patch_col % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+patch_size+text_length, :]

    for idx in range(text_length):
        if idx % 4 == 0:
            mask[:, :, idx, :] = mask[:, :, 0+text_length, :]
        elif idx % 4 == 1:
            mask[:, :, idx, :] = mask[:, :, patch_size+text_length, :]
        elif idx % 4 == 2:
            mask[:, :, idx, :] = mask[:, :, patch_size*W+0+text_length, :]
        elif idx % 4 == 3:
            mask[:, :, idx, :] = mask[:, :, patch_size*W+patch_size+text_length, :]

    return mask



def make_mask_0_8groups_cont_text_chunk(mask):
    h = 64
    w = 64
    patch_size = 1

    text_length = 512
    mask[:, :, :, :512] = 0

    # assign first line four groups
    for group in range(4):
        for patch_row in range(int(h / patch_size)):
            if patch_row % 2 == 0:
                for patch_col in range(int(h / patch_size)):
                    if patch_col % 4 == group:
                        for row_increment in range(patch_size):
                            within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                            mask[:, :, group*patch_size+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0

    # assign second line four groups
    for group in range(4):
        for patch_row in range(int(h / patch_size)):
            if patch_row % 2 == 1:
                for patch_col in range(int(h / patch_size)):
                    if patch_col % 4 == group:
                        for row_increment in range(patch_size):
                            within_patch_row_start = text_length + (patch_row*patch_size+row_increment)*h+patch_col*patch_size
                            mask[:, :, patch_size*h+group*patch_size+text_length, within_patch_row_start:within_patch_row_start+patch_size] = 0


    for idx in range(4096):
        patch_row = (idx // w) // patch_size
        patch_col = (idx % w) // patch_size

        if patch_row > 1 or patch_col > 3:

            idx = idx + text_length

            if patch_row % 2 == 0:
                mask[:, :, idx, :] = mask[:, :, (patch_col%4)*patch_size+text_length, :]

            if patch_row % 2 == 1:
                mask[:, :, idx, :] = mask[:, :, patch_size*h+(patch_col%4)*patch_size+text_length, :]

    for idx in range(text_length):
        group = idx // 64

        if group < 4:
            mask[:, :, idx, :] = mask[:, :, group+text_length, :]
        else:
            mask[:, :, idx, :] = mask[:, :, w+(group-4)+text_length, :]

    return mask



def make_mask_4_16x16(mask):
    h = 64
    w = 64
    patch_h = 8
    patch_w = 8

    mask_h = 16
    mask_w = 16

    text_length = 512
    mask[:, :, :, :512] = 0

    mask[:, :, text_length:, text_length:] = float('-inf')

    patches_y = h // patch_h
    patches_x = w // patch_w

    for patch_row in range(patches_y):
        for patch_col in range(patches_x):
            position = text_length + (patch_row*patch_h)*w+patch_col*patch_w

            if patch_row == 0:
                kv_start = position

                if patch_col == 0:
                    pass
                elif patch_col == (patches_x - 1):
                    kv_start -= (mask_w - patch_w)
                else:
                    kv_start -= ((mask_w - patch_w) // 2)
                    
            elif patch_row == patches_y - 1:
                kv_start = position - (mask_h - patch_h) * w
                if patch_col == 0:
                    pass
                elif patch_col == (patches_x - 1):
                    kv_start -= (mask_w - patch_w)
                else:
                    kv_start -= ((mask_w - patch_w) // 2)

            elif patch_col == 0:
                kv_start = position - ((mask_h - patch_h) // 2) * w

            elif patch_col == (patches_x - 1):
                kv_start = position - ((mask_h - patch_h) // 2) * w - (mask_w - patch_w)

            else:
                kv_start = position - ((mask_h - patch_h) // 2) * w - ((mask_w - patch_w) // 2)
            
            # assign single q in the q patch
            for i in range(mask_h):
                mask[:, :, position, kv_start+i*w:kv_start+i*w+mask_w] = 0

            # disminate single q mask to all q in the patch
            for i in range(patch_h):
                for j in range(patch_w):
                    mask[:, :, position+i*w+j, :] = mask[:, :, position, :]

    return mask




# mask0 = make_mask_0(4608, torch.full((1, 24, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()
# mask1 = make_mask_1(4608, torch.full((1, 24, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()



# ##### Single Output Check ######
# torch.manual_seed(0)

# dtype = torch.bfloat16
# seq_length = 4096 + 512


# # dtype = torch.float16
# # seq_length = 4096

# width = 64 
# length = 64

# head = 10
# batch_size = 2
# head_dim = 128

# q = torch.randn(batch_size, seq_length, head, head_dim, dtype=dtype, device='cuda')
# k = torch.randn(batch_size, seq_length, head, head_dim, dtype=dtype, device='cuda')
# v = torch.randn(batch_size, seq_length, head, head_dim, dtype=dtype, device='cuda')

# # k[:, :512] = 0
# # v[:, :512] = 0

# # for i in range(16):
# #     k[:, 512 + i*2+1:512 + (i+1)*2+1] = 0
# #     v[:, 512 + i*2+1:512 + (i+1)*2+1] = 0

# # k[:, 512+16*2:] = 0
# # v[:, 512+16*2:] = 0



# out = flash_attn_func(q, k, v, length=length, width=length, mask=0)

# mask = make_mask_0_8groups_cont_text_chunk(torch.full((1, 1, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()
# make_mask_4_16x16(mask)

# out_ref = attention_ref(q, k, v, mask=mask)


# # result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-03)
# # # result = torch.allclose(out_ref[:, :512], out[:, :512], rtol=1e-02, atol=1e-03)
# # # result = torch.allclose(out_ref[:, 512:], out[:, 512:], rtol=1e-02, atol=1e-03)

# # # total_not_pass = 0
# # # for i in tqdm(range(100)):
# # #     out = flash_attn_func(q, k, v, length=length, width=length, mask=0)
# # #     result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-03)

# # #     if not result:
# # #         total_not_pass += 1
# # #         print(abs(out_ref - out).max())
# # # print(total_not_pass)

# # # abs(out_ref[:, :512] - out[:, :512]).max()
# # # abs(out_ref[0, 1, :] - out[0, 1, :]).max()
# # # abs(out_ref[0, 2, :] - out[0, 2, :]).max()
# # # abs(out_ref[0, 3, :] - out[0, 3, :]).max()

# # # abs(out_ref[:, 512:] - out[:, 512:]).max()
# # # abs(out_ref[:, 512:] - out[:, 512:]).max()

# import pdb; pdb.set_trace()





###################### benchmark time batch ######################
def benchmark_batch(seq_length, width, head, mask_type, func_type, dtype=torch.float16, device='cuda:0'):
    time_lst = []
    batch_lst = [2, 2, 2, 2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
    # batch_lst = [2, 2, 2]
    for i in tqdm(batch_lst):
        start_time = time.time()
        batch = i

        q = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)
        k = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)
        v = torch.randn(batch, seq_length, head, 128, dtype=dtype, device=device)


        q_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)
        k_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)
        v_sdpa = torch.randn(batch, head, seq_length, 128, dtype=dtype, device=device)

        flash_attn_func(q, k, v, mask=mask_type, width=width, length=width)
        
        flash_attn_func(q, k, v, length=width, width=width, mask=0)

        # time.perf_counter()
        if func_type == 'org':
            start_time = time.time()
            for i in range(2000):
                F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            end_time = time.time()

        else:
            start_time = time.time()
            for i in range(2000):
                flash_attn_func(q, k, v, mask=mask_type, width=width, length=width)
            end_time = time.time()

        time_lst.append(((end_time - start_time)/2000) * 1e6)

    return time_lst


def benchmark_length(width, head, mask_type, func_type, dtype=torch.float16):
    time_lst = []
    batch = 2
    length_lst = [64, 128, 256, 382, 512, 768, 1024]
    for i in tqdm(length_lst):
        seq_length = i * width + 512

        # time.perf_counter()
        if func_type == 'org':
            q_sdpa = torch.randn(batch, head, seq_length, 64, dtype=dtype, device='cuda')
            k_sdpa = torch.randn(batch, head, seq_length, 64, dtype=dtype, device='cuda')
            v_sdpa = torch.randn(batch, head, seq_length, 64, dtype=dtype, device='cuda')

            start_time = time.time()
            for i in range(2000):
                F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa)
            end_time = time.time()
        else:
            q = torch.randn(batch, seq_length, head, 64, dtype=dtype, device='cuda')
            k = torch.randn(batch, seq_length, head, 64, dtype=dtype, device='cuda')
            v = torch.randn(batch, seq_length, head, 64, dtype=dtype, device='cuda')

            start_time = time.time()
            for i in range(2000):
                flash_attn_func(q, k, v, mask=mask_type, width=width, length=i)

            end_time = time.time()

        time_lst.append((end_time - start_time) / 2)

    return time_lst




# batch_lst = [1, 1, 1, 2, 2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
# # batch_lst = [2, 2, 2]
# mask = 0
# for seq_length, width in [(4608, 64)]:
#     org_lst = benchmark_batch(seq_length, width, 24, mask, 'org', dtype=torch.bfloat16)
    
#     kernel_lst = benchmark_batch(seq_length, width, 24, mask, 'kernel', dtype=torch.bfloat16)

#     kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]

#     for i in range(len(kernel_org_ratio)):
#         print(f"seq_length: {seq_length} | mask: {mask} | batch: {batch_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f} | kernel_time: {kernel_lst[i]:.5f} | org_time: {org_lst[i]:.5f}")

    
#     print()


# length_lst = [64, 128, 256, 382, 512, 768, 1024]
# for width in [64]:
#     for mask in [0]:
#         head = 24
#         org_lst = benchmark_length(width, head, mask, 'org', dtype=torch.bfloat16)
#         kernel_lst = benchmark_length(width, head, mask, 'kernel', dtype=torch.bfloat16)

#         kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]
#         for i in range(len(kernel_org_ratio)):
#             print(f"seq_length: {length_lst[i]*width} | mask: {mask} | length: {length_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f}")

#         print()




# batch_lst = [2, 3, 5, 7, 10, 12, 15, 17, 20, 22, 25, 27, 30]
# for mask in [0, 1]:
#     for seq_length, width in [(1024, 32), (4096, 64)]:
#         org_lst = benchmark_batch(seq_length, width, mask, 'org')
#         kernel_lst = benchmark_batch(seq_length, width, mask, 'kernel')

#         kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]

#         for i in range(len(kernel_org_ratio)):
#             print(f"seq_length: {seq_length} | mask: {mask} | batch: {batch_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f}")

#         print()


# length_lst = [64, 128, 256, 382, 512, 768, 1024]
# width = 64
# for width in [32, 64]:
#     for mask in [0, 1]:
#         kernel_lst = benchmark_length(width, mask, 'kernel')
#         org_lst = benchmark_length(width, mask, 'org')

#         kernel_org_ratio = [kernel_lst[i] / org_lst[i] for i in range(len(org_lst))]
#         for i in range(len(kernel_org_ratio)):
#             print(f"seq_length: {length_lst[i]*width} | mask: {mask} | length: {length_lst[i]} | kernel/org: {kernel_org_ratio[i]:.5f}")

#         print()




##### Result Check ######
def result_check(seq_length, width, mask_type):
    head = 10

    false_lst = []
    for batch_size in tqdm(range(2, 14)):
        mask = torch.full((batch_size, 10, seq_length, seq_length), float('-inf')).to('cuda')
        if mask_type == 0:
            make_mask_0(seq_length, mask)
        else:
            make_mask_1(seq_length, mask)

        for i in range(1000):
            torch.manual_seed(i)
            q = torch.randn(batch_size, seq_length, head, 64, dtype=torch.float16, device='cuda')
            k = torch.randn(batch_size, seq_length, head, 64, dtype=torch.float16, device='cuda')
            v = torch.randn(batch_size, seq_length, head, 64, dtype=torch.float16, device='cuda')

            out_ref = attention_ref(q, k, v, mask=mask.clone())

            out = flash_attn_func(q, k, v, length=width, width=width, mask=mask_type)[0]

            result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-03)

            if not result:
                print(f"Not passed | seq: {seq_length} | width: {width} | mask: {mask_type} | batch_size: {batch_size}, i: {i}")
                # torch.max(torch.abs(out_ref - out))
                false_lst.append((batch_size, i))



def result_check_flux_mask(seq_length, width, mask_type=0):
    head = 24

    false_lst = []
    for batch_size in tqdm(range(2, 14)):

        if mask_type == 0:
            mask = make_mask_0_flux(seq_length, torch.full((1, 1, seq_length, seq_length), float('-inf'), dtype=torch.bfloat16)).cuda()
        else:
            mask = make_mask_1_flux(seq_length, torch.full((1, 1, seq_length, seq_length), float('-inf'), dtype=torch.bfloat16)).cuda()

        for i in range(1000):
            torch.manual_seed(i)
            q = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            
            q_sdpa = rearrange(q, 'b l h d -> b h l d')
            k_sdpa = rearrange(k, 'b l h d -> b h l d')
            v_sdpa = rearrange(v, 'b l h d -> b h l d')

            out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask)
            out_ref = rearrange(out_ref, 'b h l d -> b l h d')

            out = flash_attn_func(q, k, v, length=width, width=width, mask=mask_type)

            result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-03)

            if not result:
                print(f"Not passed | seq: {seq_length} | width: {width} | mask: {mask_type} | batch_size: {batch_size}, i: {i}")
                # torch.max(torch.abs(out_ref - out))
                false_lst.append((batch_size, i))



def result_check_flux_mask0_8group(seq_length, width):
    head = 24

    false_lst = []
    for batch_size in tqdm(range(2, 14)):
        mask = make_mask_0_8groups_cont_text_chunk(torch.full((1, 1, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()

        for i in range(1000):
            torch.manual_seed(i)
            q = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            
            q_sdpa = rearrange(q, 'b l h d -> b h l d')
            k_sdpa = rearrange(k, 'b l h d -> b h l d')
            v_sdpa = rearrange(v, 'b l h d -> b h l d')

            out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask)
            out_ref = rearrange(out_ref, 'b h l d -> b l h d')

            out = flash_attn_func(q, k, v, length=width, width=width, mask=0)

            result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-03)

            if not result:
                print(f"Not passed | seq: {seq_length} | width: {width} | mask: {0} | batch_size: {batch_size}, i: {i}")
                # torch.max(torch.abs(out_ref - out))
                false_lst.append((batch_size, i))



def result_check_flux_masksw(seq_length, width):
    head = 24

    false_lst = []
    for batch_size in tqdm(range(2, 14)):
        mask = make_mask_0_8groups_cont_text_chunk(torch.full((1, 1, 4608, 4608), float('-inf'), dtype=torch.bfloat16)).cuda()
        make_mask_4_16x16(mask)

        for i in range(1000):
            torch.manual_seed(i)
            q = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            k = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            v = torch.randn(batch_size, seq_length, head, 128, dtype=torch.bfloat16, device='cuda')
            
            q_sdpa = rearrange(q, 'b l h d -> b h l d')
            k_sdpa = rearrange(k, 'b l h d -> b h l d')
            v_sdpa = rearrange(v, 'b l h d -> b h l d')

            out_ref = F.scaled_dot_product_attention(q_sdpa, k_sdpa, v_sdpa, attn_mask=mask)
            out_ref = rearrange(out_ref, 'b h l d -> b l h d')

            out = flash_attn_func(q, k, v, length=width, width=width, mask=4)

            result = torch.allclose(out_ref, out, rtol=1e-02, atol=1e-03)

            if not result:
                print(f"Not passed | seq: {seq_length} | width: {width} | mask: 4 | batch_size: {batch_size}, i: {i}")
                # torch.max(torch.abs(out_ref - out))
                false_lst.append((batch_size, i))




# for seq_length, width in [(1024, 32), (4096, 64)]:
#     for mask in [0, 1]:
#         result_check(seq_length, width, mask)


# result_check_flux_mask(4608, 64, 0)

# result_check_flux_mask(4608, 64, 1)

result_check_flux_mask0_8group(4608, 64)

result_check_flux_masksw(4608, 64)







################## Benchmark Results ##################

#################################### Mask 0 8 Groups ####################################

# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 17.99874 | kernel_time: 27536.91769 | org_time: 1529.93572
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.39188 | kernel_time: 1226.45044 | org_time: 3129.68278
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.38863 | kernel_time: 1224.81871 | org_time: 3151.64566
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.38651 | kernel_time: 1227.39887 | org_time: 3175.61567
# seq_length: 4608 | mask: 0 | batch: 3 | kernel/org: 0.43752 | kernel_time: 1745.78238 | org_time: 3990.16631
# seq_length: 4608 | mask: 0 | batch: 5 | kernel/org: 0.45068 | kernel_time: 2910.82585 | org_time: 6458.69529
# seq_length: 4608 | mask: 0 | batch: 7 | kernel/org: 0.42349 | kernel_time: 4170.31717 | org_time: 9847.49734
# seq_length: 4608 | mask: 0 | batch: 10 | kernel/org: 0.42063 | kernel_time: 5961.10332 | org_time: 14171.99862
# seq_length: 4608 | mask: 0 | batch: 12 | kernel/org: 0.39279 | kernel_time: 7324.20158 | org_time: 18646.63696
# seq_length: 4608 | mask: 0 | batch: 15 | kernel/org: 0.39517 | kernel_time: 9138.02528 | org_time: 23124.00627
# seq_length: 4608 | mask: 0 | batch: 17 | kernel/org: 0.38300 | kernel_time: 10588.46092 | org_time: 27645.76674
# seq_length: 4608 | mask: 0 | batch: 20 | kernel/org: 0.38756 | kernel_time: 12424.73578 | org_time: 32058.56097
# seq_length: 4608 | mask: 0 | batch: 22 | kernel/org: 0.37716 | kernel_time: 13764.69541 | org_time: 36495.50390
# seq_length: 4608 | mask: 0 | batch: 25 | kernel/org: 0.38274 | kernel_time: 15641.91651 | org_time: 40868.50238
# seq_length: 4608 | mask: 0 | batch: 27 | kernel/org: 0.37378 | kernel_time: 16925.93074 | org_time: 45282.82034
# seq_length: 4608 | mask: 0 | batch: 30 | kernel/org: 0.37518 | kernel_time: 18606.48870 | org_time: 49593.66775


# seq_length: 4608 | mask: 0 | batch: 1 | kernel/org: 17.58921 | kernel_time: 26344.69318 | org_time: 1497.77555
# seq_length: 4608 | mask: 0 | batch: 1 | kernel/org: 0.36223 | kernel_time: 1121.15359 | org_time: 3095.10911
# seq_length: 4608 | mask: 0 | batch: 1 | kernel/org: 0.36244 | kernel_time: 1122.10405 | org_time: 3096.01116
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.35905 | kernel_time: 1121.40918 | org_time: 3123.27313
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.35812 | kernel_time: 1403.24688 | org_time: 3918.39278
# seq_length: 4608 | mask: 0 | batch: 3 | kernel/org: 0.35719 | kernel_time: 2259.07516 | org_time: 6324.55575
# seq_length: 4608 | mask: 0 | batch: 5 | kernel/org: 0.35395 | kernel_time: 3406.59392 | org_time: 9624.59123
# seq_length: 4608 | mask: 0 | batch: 7 | kernel/org: 0.35013 | kernel_time: 4846.73285 | org_time: 13842.46981
# seq_length: 4608 | mask: 0 | batch: 10 | kernel/org: 0.34599 | kernel_time: 6298.63405 | org_time: 18204.89931
# seq_length: 4608 | mask: 0 | batch: 12 | kernel/org: 0.34286 | kernel_time: 7734.86221 | org_time: 22559.61156
# seq_length: 4608 | mask: 0 | batch: 15 | kernel/org: 0.34242 | kernel_time: 9216.90536 | org_time: 26916.94748
# seq_length: 4608 | mask: 0 | batch: 17 | kernel/org: 0.34297 | kernel_time: 10707.74484 | org_time: 31220.46816
# seq_length: 4608 | mask: 0 | batch: 20 | kernel/org: 0.34207 | kernel_time: 12139.29391 | org_time: 35487.99062
# seq_length: 4608 | mask: 0 | batch: 22 | kernel/org: 0.34271 | kernel_time: 13618.61384 | org_time: 39737.65540
# seq_length: 4608 | mask: 0 | batch: 25 | kernel/org: 0.34318 | kernel_time: 15093.30285 | org_time: 43980.18539
# seq_length: 4608 | mask: 0 | batch: 27 | kernel/org: 0.34347 | kernel_time: 16530.44474 | org_time: 48127.36046


#################################### Mask 4(Sliding Window) ####################################

# seq_length: 4608 | mask: 0 | batch: 1 | kernel/org: 0.29453 | kernel_time: 925.08864 | org_time: 3140.90085
# seq_length: 4608 | mask: 0 | batch: 1 | kernel/org: 0.29177 | kernel_time: 925.61066 | org_time: 3172.45233
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.28953 | kernel_time: 925.19927 | org_time: 3195.55962
# seq_length: 4608 | mask: 0 | batch: 2 | kernel/org: 0.28860 | kernel_time: 1154.41120 | org_time: 4000.01490
# seq_length: 4608 | mask: 0 | batch: 3 | kernel/org: 0.28817 | kernel_time: 1858.07300 | org_time: 6447.86501
# seq_length: 4608 | mask: 0 | batch: 5 | kernel/org: 0.28603 | kernel_time: 2809.19230 | org_time: 9821.36774
# seq_length: 4608 | mask: 0 | batch: 7 | kernel/org: 0.28392 | kernel_time: 3995.89789 | org_time: 14074.19491
# seq_length: 4608 | mask: 0 | batch: 10 | kernel/org: 0.28187 | kernel_time: 5183.86996 | org_time: 18390.73539
# seq_length: 4608 | mask: 0 | batch: 12 | kernel/org: 0.27944 | kernel_time: 6350.24822 | org_time: 22725.23284
# seq_length: 4608 | mask: 0 | batch: 15 | kernel/org: 0.27986 | kernel_time: 7573.20189 | org_time: 27060.88126
# seq_length: 4608 | mask: 0 | batch: 17 | kernel/org: 0.28046 | kernel_time: 8800.86744 | org_time: 31379.78840
# seq_length: 4608 | mask: 0 | batch: 20 | kernel/org: 0.27977 | kernel_time: 9979.93004 | org_time: 35671.56875
# seq_length: 4608 | mask: 0 | batch: 22 | kernel/org: 0.27974 | kernel_time: 11173.44069 | org_time: 39942.63256
# seq_length: 4608 | mask: 0 | batch: 25 | kernel/org: 0.28018 | kernel_time: 12402.04620 | org_time: 44264.58180
# seq_length: 4608 | mask: 0 | batch: 27 | kernel/org: 0.28018 | kernel_time: 13592.62729 | org_time: 48514.16242
