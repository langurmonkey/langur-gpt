from dataclasses import dataclass


@dataclass
class GPTConfig:
    """
    WHAT: All hyperparameters in one place.
    WHY: Changing model size is one line. No hunting through code.
    """

    # ===== Training dataset =====
    max_samples: int = 10000       # WHAT: Max number of samples of the dataset to use for training

    # ===== Architecture =====
    vocab_size: int = 50257        # WHAT: 50,257 unique tokens in GPT-2 vocabulary
    d_model: int = 768             # WHAT: Each token becomes a 768-dim vector
                                   # WHY: Bigger = more nuanced meanings, more compute
    num_heads: int = 12            # WHAT: 12 attention heads (12 × 64 = 768)
    num_layers: int = 12           # WHAT: 12 transformer blocks stacked
                                   # WHY: Deeper = better reasoning, harder to train
    max_seq_len: int = 1024        # WHAT: Max tokens model can process at once

    # ===== Regularization (prevent overfitting) =====
    dropout: float = 0.1           # WHAT: Randomly disable 10% of neurons during training
    embd_dropout: float = 0.1      # WHAT: Dropout applied right after embedding lookup

    # ===== Training =====
    learning_rate: float = 3e-4    # WHAT: Step size for weight updates
    weight_decay: float = 0.1      # WHAT: Penalize large weights (L2 regularization)
    warmup_steps: int = 50         # WHAT: Gradually increase LR for first 2000 steps
    max_steps: int = 500           # WHAT: Total training iterations
    batch_size: int = 8            # WHAT: Sequences processed per GPU step
    grad_accum_steps: int = 4      # WHAT: Accumulate gradient steps (effective batch = 8×4 = 32)
    betas: tuple = (0.9, 0.95)    # WHAT: AdamW momentum coefficients
    eps: float = 1e-8              # WHAT: Small constant preventing division by zero

    def __post_init__(self):
        """Validate configuration consistency."""
        assert self.d_model % self.num_heads == 0, (
            f"d_model ({self.d_model}) must be divisible by "
            f"num_heads ({self.num_heads})"
        )
