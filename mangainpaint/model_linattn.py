"""
Linear attention (Katharopoulos et al. 2020, "Transformers are RNNs"):
O(N) in sequence length, never forms an NxN attention matrix. `phi(x) =
elu(x)+1` keeps the kernel feature map positive (a valid similarity
kernel), which is what allows the associative KV = phi(K)^T @ V
reordering below.

This is the one building block C2 (`model_lama_slim_fus.py`, Table 1)
reuses for its bottleneck linear-attention pass. The standalone
linear-attention architecture this module originally also defined (a
from-scratch axis replacing dilated convolution with this operator
directly) did not beat the from-scratch baseline and is not reported
anywhere in the paper -- dropped here rather than shipped as unreferenced
code.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearAttention(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        assert ch % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.out = nn.Conv2d(ch, ch, 1)

    @staticmethod
    def _phi(x):
        return F.elu(x) + 1.0

    def forward(self, x, valid_mask=None):
        """`valid_mask`, if given: (B, N) with 1=valid/0=hole (already
        flattened to the same H*W as `x`). Optional and defaulting to None.
        When given, zeroes hole positions out of `k`/`v` before the
        KV/normalizer sums so the global memory bank this op builds is
        only ever pooled from genuinely valid content. Queries still run
        over every position (valid and hole alike): only the *source*
        pool is filtered, not who gets to read from it.
        """
        B, C, H, W = x.shape
        nh, hd = self.num_heads, self.head_dim
        N = H * W

        # Force fp32 for the whole KV/normalizer chain: `z = k.sum(dim=-1)`
        # sums phi(k) (always >= 0, so no cancellation) over N = H*W
        # positions, reaching magnitudes large enough to overflow fp16's
        # ~65504 max after a few stacked blocks under autocast.
        with torch.amp.autocast(x.device.type, enabled=False):
            x32 = x.float()

            def split_heads(t):
                return t.view(B, nh, hd, N)

            q = split_heads(self._phi(self.q(x32)))
            k = split_heads(self._phi(self.k(x32)))
            v = split_heads(self.v(x32).view(B, nh, hd, N))

            if valid_mask is not None:
                vm = valid_mask.to(dtype=torch.float32).view(B, 1, 1, N)  # broadcasts over nh, hd
                k = k * vm
                v = v * vm

            # KV: (B, nh, hd, hd) -- sum over the N (sequence) axis, never an
            # NxN matrix. Z: (B, nh, hd, 1) normalizer.
            kv = torch.einsum('bhdn,bhen->bhde', k, v)
            z = k.sum(dim=-1, keepdim=True)  # (B, nh, hd, 1)

            out = torch.einsum('bhdn,bhde->bhen', q, kv)
            denom = torch.einsum('bhdn,bhdo->bhon', q, z).clamp_min(1e-6)
            out = out / denom

            out = out.reshape(B, C, H, W)
            out = self.out(out)
        return out.to(x.dtype)
