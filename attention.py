import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from rope import RotaryPositionalEmbedding


class MultiHeadAttention(nn.Module):
    """
    WHAT: Multi-Head Self-Attention with RoPE and causal masking.

    WHY: Transformers would be useless without attention. This is the
         mechanism that lets each token "look at" every other token and
         decide how much each matters for understanding the current context.

         Each attention head:
         1. Projects input into Query, Key, Value spaces
         2. Computes Q·K^T / sqrt(d_k) → how well each query matches each key
         3. Applies causal mask → no peeking at future tokens
         4. Softmax → converts scores to a probability distribution
         5. Weighted sum of Values → builds context-aware representation

         Doing this with multiple heads in parallel lets each head
         specialize in different linguistic patterns.
    """

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
        """
        Args:
            d_model:   Total embedding dimension (e.g., 768 for GPT-2 small)
            num_heads: Number of parallel attention heads (e.g., 12)
            dropout:   Probability of randomly zeroing attention weights

        WHY: d_model must be divisible by num_heads because each head
             operates on d_model/num_heads dimensions (64 for GPT-2 small).
             This split-then-concat strategy lets heads specialize while
             keeping total parameter count the same as a single large head.
        """
        super().__init__()

        # WHAT: Validate that heads evenly divide the model dimension
        assert d_model % num_heads == 0, (
            f"d_model ({d_model}) must be divisible by num_heads ({num_heads}). "
            f"This ensures each head has equal dimension."
        )

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads  # 768/12 = 64 dimensions per head
                                               # WHY: 64 is the "sweet spot" —
                                               # enough to capture meaning,
                                               # small enough for efficient compute

        # ===== QKV Projection =====
        # WHAT: One big linear layer that projects input to Q, K, V simultaneously
        # WHY:  3 separate Linear(768→768) layers = 3 matrix multiplies.
        #       One combined Linear(768→2304) = 1 bigger matrix multiply.
        #       On GPU, 1 big operation is much faster than 3 small ones
        #       due to better parallelism and fewer kernel launches.
        #       Shape: [d_model, 3 * d_model] = [768, 2304]
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)

        # ===== Output Projection =====
        # WHAT: Project concatenated head outputs back to d_model
        # WHY:  After concatenation: [batch, seq, d_model] but each head's
        #       output was computed independently. This linear layer MIXES
        #       information across heads, letting them communicate.
        #       Without it, heads would stay isolated — like 12 experts
        #       who never talk to each other.
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # ===== RoPE (Rotary Position Embeddings) =====
        # WHAT: Applies rotation-based position encoding to Q and K only
        # WHY:  RoPE encodes position into the Q and K vectors so that
        #       the dot product Q·K naturally depends on RELATIVE position.
        #       We apply to the head_dim (not d_model) because each head
        #       needs its own position info in its subspace.
        #       V does NOT get RoPE because values carry content, not
        #       position — position is only relevant for deciding
        #       WHICH values to attend to, not the values themselves.
        self.rotary = RotaryPositionalEmbedding(self.head_dim)

        # ===== Dropout =====
        # WHAT: Randomly zero out attention weights during training
        # WHY:  Without dropout, the model can become overconfident —
        #       one token always dominates attention, ignoring other
        #       potentially useful context. Dropout forces the model
        #       to learn redundant attention patterns (backup plans).
        self.attn_dropout = nn.Dropout(dropout)   # Applied to attention weights
        self.resid_dropout = nn.Dropout(dropout)  # Applied to final output

    def forward(self, x: torch.Tensor, mask: torch.Tensor = None) -> torch.Tensor:
        """
        WHAT: Compute multi-head self-attention.

        Input:  x    [batch, seq_len, d_model]  — token embeddings
                mask [batch, 1, seq, seq]       — causal mask (1=visible, 0=masked)

        Output:      [batch, seq_len, d_model]  — context-aware representations

        The forward pass has 8 steps, each critical:
        """
        batch_size, seq_len, _ = x.shape

        # ===== STEP 1: Project input to Q, K, V — all at once =====
        # WHAT: Linearly transform input into query, key, value spaces
        # WHY:  Combined projection is faster on GPU than 3 separate ones.
        #       After this: [batch, seq, 3*d_model] where the last dim
        #       has Q values first, then K values, then V values.
        qkv = self.qkv_proj(x)               # [batch, seq, 3 * d_model]

        # ===== STEP 2: Reshape to expose the head dimension =====
        # WHAT: Split the 3*d_model into separate Q,K,V and separate heads
        # WHY:  We need shape [batch, num_heads, seq, head_dim] for
        #       parallel computation. The reshape + permute does this
        #       in two efficient operations without data copies.
        #
        # Transform: [batch, seq, 3, heads, head_dim]
        # Then permute: [3, batch, heads, seq, head_dim]
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)    # [3, batch, heads, seq, head_dim]

        # WHAT: Unpack the three projections
        q = qkv[0]  # Query:  [batch, heads, seq, head_dim] — "what I'm looking for"
        k = qkv[1]  # Key:    [batch, heads, seq, head_dim] — "what I offer to match"
        v = qkv[2]  # Value:  [batch, heads, seq, head_dim] — "my actual content"

        # ===== STEP 3: Apply Rotary Position Embeddings =====
        # WHAT: Rotate Q and K by position-dependent angles
        # WHY:  After rotation, the dot product q_i · k_j depends on
        #       cos(i-j) and sin(i-j) — the RELATIVE distance between
        #       tokens i and j. This is what we want: attention should
        #       care about "how far apart are these tokens?" not
        #       "what are their absolute positions?"
        q = self.rotary(q, seq_len)
        k = self.rotary(k, seq_len)

        # ===== STEP 4: Compute attention scores (Q · K^T) =====
        # WHAT: For each query token, compute dot product with every key token
        # WHY:  Dot product measures cosine similarity (if vectors normalized).
        #       Higher dot product = query "wants" what key "offers".
        #
        #       Shape: [batch, heads, query_seq, key_seq]
        #       attn_scores[b, h, i, j] = how much token i attends to token j
        #
        #       DIVIDE BY sqrt(head_dim): critical for stable training.
        #       Without this, the variance of dot products grows with d_k,
        #       making softmax too "peaky" → gradients vanish → model dies.
        #       See Part 4 above for the mathematical derivation.
        attn_scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)

        # ===== STEP 5: Apply causal mask — no peeking at future tokens =====
        # WHAT: Set attention scores to future tokens to -infinity
        # WHY:  During training, the model must predict token[i+1] from
        #       tokens[0..i]. If token[i] can see token[i+1], it's like
        #       seeing the answer before the question — cheating.
        #
        #       -infinity → e^(-inf) = 0.0 after softmax = zero attention
        #
        #       The mask is lower-triangular:
        #       Token 0 → sees [0]        (itself only)
        #       Token 1 → sees [0, 1]     (itself + previous)
        #       Token 2 → sees [0, 1, 2]  (itself + all previous)
        #       Token 3 → sees [0, 1, 2, 3]
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))

        # ===== STEP 6: Softmax — scores become attention weights =====
        # WHAT: Convert raw scores to a probability distribution over keys
        # WHY:  softmax(scores)[j] = e^score[j] / sum(e^score[k] for k in all keys)
        #       This makes all weights:
        #       - Positive (e^x > 0 always)
        #       - Sum to 1.0 (proper probability distribution)
        #       - Differentiable (we can compute gradients through it)
        #
        #       The softmax is applied over the LAST dimension (dim=-1),
        #       which is the "key" dimension — so each query gets a
        #       distribution over all keys it can see.
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)

        # ===== STEP 7: Weighted sum of values =====
        # WHAT: Mix the value vectors according to attention weights
        # WHY:  This is WHERE attention actually happens. Each query
        #       token gets a NEW vector that is a weighted blend of
        #       all visible value vectors.
        #
        #       High attention to token j → V_j has large influence
        #       Low attention to token j → V_j has small influence
        #
        #       The result is "context-aware" — each token now "knows"
        #       about the other relevant tokens in the sequence.
        #
        #       [batch, heads, seq, head_dim] @ [batch, heads, seq, head_dim]
        #       → [batch, heads, seq, head_dim]
        attn_output = attn_weights @ v

        # ===== STEP 8: Merge heads and project =====
        # WHAT: Combine all head outputs into one d_model vector per token
        # WHY:  Currently: [batch, heads, seq, head_dim]
        #       Need:       [batch, seq, d_model]
        #
        #       Transpose swaps heads and sequence:
        #       [batch, seq, heads, head_dim]
        #       Reshape flattens heads×head_dim:
        #       [batch, seq, d_model]
        #
        #       The final linear projection lets information flow between
        #       heads — each head's discoveries can now influence the
        #       combined representation.
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, seq_len, self.d_model)

        output = self.out_proj(attn_output)   # Mix across heads
        output = self.resid_dropout(output)   # Regularization

        return output


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """
    WHAT: Create a causal (lower triangular) attention mask.
    WHY:  Prevents tokens from attending to future tokens during training.

    Visual for seq_len=6:
        [[✓, ✗, ✗, ✗, ✗, ✗],     Token 0 (first word)
         [✓, ✓, ✗, ✗, ✗, ✗],     Token 1
         [✓, ✓, ✓, ✗, ✗, ✗],     Token 2
         [✓, ✓, ✓, ✓, ✗, ✗],     Token 3
         [✓, ✓, ✓, ✓, ✓, ✗],     Token 4
         [✓, ✓, ✓, ✓, ✓, ✓]]     Token 5 (last word — sees everything)

    ✓ = position is visible (1.0)
    ✗ = position is masked (0.0, becomes -inf in attention)

    Reshaped to [1, 1, seq_len, seq_len] for broadcasting over:
    - batch dimension (all batches use same mask)
    - head dimension (all heads use same mask — heads CAN'T see future)
    """
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)
