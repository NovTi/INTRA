"""Check the FLUX training masks (intra_attn/reference.py, via intra_processor.get_mask) against
the loop-based mask builders of the original training code, copied verbatim below. CPU only:

    python experiments/flux/test_masks.py
"""
import torch

from intra_processor import get_mask


# ---- original builders (1024 x 1024 images: 512 text + 64 x 64 image tokens), unmodified
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


scatter = make_mask_0_8groups_cont_text_chunk(torch.full((1, 1, 4608, 4608), float("-inf")))
gather0 = make_mask_4_16x16(scatter.clone())
for pattern, expected in (("scatter", scatter), ("gather0", gather0)):
    allowed = get_mask(pattern, 64, "cpu")[0, 0] == 0
    n_diff = int((allowed != (expected[0, 0] == 0)).sum())
    print(f"{pattern:8s} density={allowed.float().mean().item():.4f} mismatches={n_diff}")
    assert n_diff == 0, pattern
print("masks match the original training masks")
