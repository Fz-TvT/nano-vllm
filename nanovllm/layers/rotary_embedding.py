from functools import lru_cache
import torch
from torch import nn

try:
    import triton
    from triton import language as tl
    _TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    _TRITON_AVAILABLE = False

def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)

if _TRITON_AVAILABLE:
    @triton.jit
    def rope_kernel(
        x_ptr,
        cos_ptr,
        sin_ptr,
        out_ptr,
        num_rows,
        num_heads,
        head_dim,
        half_dim,
        BLOCK_SIZE: tl.constexpr,
    ):
        pid_row = tl.program_id(0)
        pid_col = tl.program_id(1)
        if pid_row >= num_rows:
            return

        offs = pid_col * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offs < half_dim
        token_row = pid_row // num_heads

        x_row = x_ptr + pid_row * head_dim
        cs_row = token_row * half_dim

        x1 = tl.load(x_row + offs, mask=mask, other=0.0).to(tl.float32)
        x2 = tl.load(x_row + offs + half_dim, mask=mask, other=0.0).to(tl.float32)
        c = tl.load(cos_ptr + cs_row + offs, mask=mask, other=0.0).to(tl.float32)
        s = tl.load(sin_ptr + cs_row + offs, mask=mask, other=0.0).to(tl.float32)

        y1 = x1 * c - x2 * s
        y2 = x2 * c + x1 * s

        out_row = out_ptr + pid_row * head_dim
        tl.store(out_row + offs, y1, mask=mask)
        tl.store(out_row + offs + half_dim, y2, mask=mask)


def apply_rotary_emb_triton(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    if not _TRITON_AVAILABLE:
        return apply_rotary_emb(x, cos, sin)
    if (not x.is_cuda) or (not cos.is_cuda) or (not sin.is_cuda):
        return apply_rotary_emb(x, cos, sin)
    if x.ndim != 3 or x.shape[-1] % 2 != 0:
        return apply_rotary_emb(x, cos, sin)

    num_tokens, num_heads, head_dim = x.shape
    half_dim = head_dim // 2

    if cos.ndim == 3:
        if cos.shape[1] != 1:
            return apply_rotary_emb(x, cos, sin)
        cos2 = cos[:, 0, :]
        sin2 = sin[:, 0, :]
    elif cos.ndim == 2:
        cos2 = cos
        sin2 = sin
    else:
        return apply_rotary_emb(x, cos, sin)

    if cos2.shape != (num_tokens, half_dim) or sin2.shape != (num_tokens, half_dim):
        return apply_rotary_emb(x, cos, sin)

    x2d = x.contiguous().view(num_tokens * num_heads, head_dim)
    cos2 = cos2.contiguous()
    sin2 = sin2.contiguous()
    out2d = torch.empty_like(x2d)

    block_size = 128
    grid = (x2d.shape[0], triton.cdiv(half_dim, block_size))
    rope_kernel[grid](
        x2d,
        cos2,
        sin2,
        out2d,
        x2d.shape[0],
        num_heads,
        head_dim,
        half_dim,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )
    return out2d.view_as(x)

class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb_triton(query, cos, sin)
        key = apply_rotary_emb_triton(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb
