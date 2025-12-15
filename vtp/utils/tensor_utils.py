import torch


def convert_patch_tokens_to_4d(
    patch_tokens: torch.Tensor,
    patch_size: int,
    img_h: int,
    img_w: int
) -> torch.Tensor:
    """
    Convert 3D patch tokens (B, N, C) to 4D format (B, C, H, W).

    Args:
        patch_tokens: Patch tokens with shape (B, N, C)
        patch_size: Size of each patch
        img_h: Original image height
        img_w: Original image width

    Returns:
        Reshaped tokens with shape (B, C, H, W)
    """
    B, N, C = patch_tokens.shape
    feat_h = img_h // patch_size
    feat_w = img_w // patch_size

    expected_patches = feat_h * feat_w
    if N != expected_patches:
        raise ValueError(
            f"Patch tokens mismatch: got {N}, expected {expected_patches}"
        )

    return patch_tokens.transpose(1, 2).reshape(B, C, feat_h, feat_w)
