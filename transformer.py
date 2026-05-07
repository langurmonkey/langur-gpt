import torch
import torch.nn as nn
import torch.nn.functional as F
from attention import MultiHeadAttention

class RMSNorm(nn.Module):
    """
    WHAT: Root Mean Square Layer Normalization.
    WHY: Normalizes each token's representation so its magnitude is ~1.0.
         Prevents values from growing/shrinking across deep networks.

         Used in: LLaMA 1/2/3, Mistral, Gemma, Qwen
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        # WHAT: Learnable scale per dimension
        # WHY: After forcing RMS=1, the model can learn to amplify
        #      important dimensions and dampen unimportant ones.
        #      Starts at 1.0 (no change initially).
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps  # WHY: prevents division by zero

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WHAT: Compute 1/sqrt(mean(x²))
        # WHY: rsqrt is 1/sqrt — computed as a single CUDA kernel
        #      for speed. The mean is over the last dimension (d_model).
        #      keepdim=True preserves the dimension for broadcasting.
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

        # WHAT: Normalize then learnable-scale
        return x * rms * self.weight


class SwiGLU(nn.Module):
    """
    WHAT: SwiGLU — gated version of Swish activation.
    WHY: The "gate" (right side of multiplication) learns to
         selectively pass or block information — like a faucet.

         Standard FFN:  output = W2(ReLU(W1(x)))
         SwiGLU FFN:    output = W3(SiLU(W1(x)) * (W2(x)))
                                   ^^^^^^^^      ^^^^^^
                                   values        gate

         The gate multiplies values: if gate ≈ 0, block info.
                                     if gate ≈ 1, pass info.
                                     if gate ≈ 0.5, partial pass.

         This gating mechanism is what makes SwiGLU outperform
         ReLU and GELU — the model learns WHERE to apply non-linearity.

         Paper: "GLU Variants Improve Transformer" (Shazeer, 2020)
         Used in: LLaMA 1/2/3, PaLM, Gemini
    """

    def __init__(self, d_model: int, expansion_factor: int = 4):
        super().__init__()

        # WHAT: Hidden dim is 4x input/output — the "expansion" bottleneck
        # WHY: Expand→process→contract is more expressive than same-size.
        #      784 → 3072 → 784 lets the FFN learn ~4x more complex patterns.
        hidden_dim = expansion_factor * d_model

        self.w1 = nn.Linear(d_model, hidden_dim, bias=False)   # Projects to values
        self.w2 = nn.Linear(d_model, hidden_dim, bias=False)   # Projects to gates
        self.w3 = nn.Linear(hidden_dim, d_model, bias=False)   # Projects back

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # WHAT: SiLU(w1(x)) are the values, w2(x) are the gates
        # WHY: SiLU (also called Swish) = x * sigmoid(x)
        #      It's smooth (unlike ReLU which has a sharp corner at 0),
        #      which makes gradients flow better during training.
        #      Gate multiplies values element-wise, selectively passing info.
        return self.w3(F.silu(self.w1(x)) * self.w2(x))


class TransformerBlock(nn.Module):
    """
    WHAT: One complete Transformer layer (attention + FFN with residuals).
    WHY: Stack N of these to build a deep language model.

         Architecture (Pre-Norm):
         ┌─────────────────────────────────────┐
         │ x = x + Attention(RMSNorm(x), mask) │  ← Mix information BETWEEN tokens
         │ x = x + SwiGLU(RMSNorm(x))          │  ← Process information WITHIN tokens
         └─────────────────────────────────────┘

         Each sublayer: normalize FIRST (pre-norm), then compute,
         then ADD back the original (residual connection).

         Without residuals: deep networks can't train (vanishing gradients)
         Without pre-norm: training is unstable at large depths
         Without FFN: no non-linear processing per token
         Without attention: no information mixing between tokens
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        # WHAT: First normalization — before attention
        # WHY: Pre-norm: clean, well-scaled input → stable attention computation
        self.norm1 = RMSNorm(d_model)

        # WHAT: Multi-head self-attention with RoPE and causal masking
        # WHY: The core mechanism that lets tokens "talk to" each other
        self.attention = MultiHeadAttention(d_model, num_heads, dropout)

        # WHAT: Second normalization — before FFN
        # WHY: FFN expects normalized input for consistent behavior across layers
        self.norm2 = RMSNorm(d_model)

        # WHAT: SwiGLU feed-forward network
        # WHY: Non-linear processing per token. Without this, stacking more
        #      attention layers would be no more powerful than one layer.
        self.ffn = SwiGLU(d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass: norm → sublayer → add residual.
        Executed twice: once for attention, once for FFN.
        """

        # ===== SUB-LAYER 1: Self-Attention with residual =====
        # WHAT: x = x + Attention(Norm(x))
        # WHY: The model learns what CHANGES (the delta) to make to x,
        #      not what to replace x with entirely. This is easier to learn.
        #      If attention can't improve things, it can output near-zero.
        x = x + self.attention(self.norm1(x), mask)

        # ===== SUB-LAYER 2: Feed-Forward with residual =====
        # WHAT: x = x + FFN(Norm(x))
        # WHY: Same residual pattern. After mixing information via attention,
        #      each token "thinks" independently via the FFN.
        #      Attention = group discussion. FFN = private reflection.
        x = x + self.ffn(self.norm2(x))

        return x
