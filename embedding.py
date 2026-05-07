import torch
import torch.nn as nn
import math


class Embedding(nn.Module):
    """
    WHAT: Converts token IDs into dense vectors (embeddings).
    WHY: A neural network can't do meaningful math on integer IDs
         like [9246, 6734]. It needs continuous numbers in vectors.

         Think of it as a giant lookup table:
         Row 9246 -> vector of 768 floats (the "meaning" of "cat")
         Row 6734 -> vector of 768 floats (the "meaning" of "sat")

         This table is LEARNED. Initially random, backpropagation
         gradually moves related tokens closer together in the
         768-dimensional space.
    """

    def __init__(self, vocab_size: int, d_model: int):
        """
        WHAT: Create the embedding table (a learnable matrix).

        Args:
            vocab_size: How many unique tokens exist (50,257 for GPT-2)
            d_model:    Size of each embedding vector.

        Examples by model scale:
            GPT-2 small:  vocab=50257, d_model=768   → table is 50257 × 768
            GPT-2 medium: vocab=50257, d_model=1024  → table is 50257 × 1024
            GPT-3 small:  vocab=50257, d_model=4096  → table is 50257 × 4096
            GPT-3 large:  vocab=50257, d_model=12288 → table is 50257 × 12288

        WHY: The embedding dimension determines how much "space"
             each word has to express its meaning. Bigger d_model =
             more nuanced meanings can be captured, at the cost of
             more parameters and slower training.
        """
        super().__init__()

        # WHAT: The actual embedding weights — a [vocab_size, d_model] matrix
        # WHY: nn.Embedding is an optimized lookup table. When you pass
        #      a tensor of token IDs, it returns the corresponding rows.
        #      It's backed by a standard weight matrix, so gradients
        #      flow through it just like any nn.Linear layer.
        #
        #      Internally, nn.Embedding is essentially:
        #      def forward(self, x):
        #          return self.weight[x]  # index into the weight matrix
        self.embed = nn.Embedding(vocab_size, d_model)

        # WHAT: Cache d_model for the scaling step
        self.d_model = d_model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        WHAT: Look up embeddings for each token ID in the input.

        Input shape:  [batch_size, seq_len]    — each cell is a token ID
        Output shape: [batch_size, seq_len, d_model] — each cell is a vector

        Example walkthrough:
            Input:  [[464, 3797]]              # ["The", "cat"]
            Step 1: Look up row 464 → [768 floats] for "The"
                    Look up row 3797 → [768 floats] for "cat"
            Step 2: Scale by sqrt(768) ≈ 27.7
            Output: [[[v0..v767], [v0..v767]]] # 2 vectors of 768 numbers

        WHY each dimension:
            batch_size = how many sequences we process at once (parallelism)
            seq_len    = how many tokens per sequence (context window)
            d_model    = how rich each token's representation is (expressiveness)
        """
        # WHAT: Index into the embedding matrix
        # WHY: For each token ID, return its row. This is an O(1)
        #      lookup operation — very fast, even for 50K+ vocabulary.
        embeddings = self.embed(x)  # [batch, seq_len, d_model]

        # WHAT: Scale by sqrt(d_model)
        # WHY: See explanation above. Without this, positional information
        #      would be dwarfed by the embedding magnitudes after addition.
        #      The factor sqrt(d_model) keeps the variance at roughly d_model
        #      instead of 1.0, making embeddings and positions comparable.
        return embeddings * math.sqrt(self.d_model)
