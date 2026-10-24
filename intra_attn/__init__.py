try:
    from .interface import intra_image_attn, intra_llm_attn
except ImportError as _err:  # extension not built; intra_attn.reference (pure PyTorch) still works
    def _missing(*args, _err=_err, **kwargs):
        raise ImportError(f"intra_attn_cuda is not built ({_err}); run `pip install --no-build-isolation .`")

    intra_image_attn = intra_llm_attn = _missing

__version__ = "0.1.0"
