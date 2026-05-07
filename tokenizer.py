from dataclasses import dataclass
import tiktoken


@dataclass
class TokenizerConfig:
    """
    WHAT: Keeps all tokenizer settings in one place.
    WHY: Like a recipe card — consistent across the whole project.
         Change one value and everything updates automatically.
    """
    name: str = "gpt2"                # WHAT: use GPT-2's pretrained BPE tokenizer
                                       # WHY: same BPE as GPT-3/4 — 50K merges,
                                       #      battle-tested on billions of documents,
                                       #      and already trained (no weeks of work)
    vocab_size: int = 50257           # WHAT: total number of unique tokens
                                       # WHY: 50,257 is the exact GPT-2 vocabulary size
                                       #      (50,000 merges + 256 byte tokens + 1 EOS)
                                       #      This is the "goldilocks" number —
                                       #      big enough for rare subwords,
                                       #      small enough for fast matrix operations


class SimpleTokenizer:
    """
    WHAT: Wraps tiktoken to give us a friendly, consistent interface.
    WHY: tiktoken's raw API is low-level (you need to specify
         allowed_special every call). This wrapper makes encode/decode
         trivial — just call .encode("hello") and get tokens back.
         
         It also handles the EOS token consistently so we never
         accidentally forget to add it during training data prep.
    """

    def __init__(self, config: TokenizerConfig = None):
        """
        WHAT: Initialize the tokenizer with GPT-2's BPE vocabulary.
        WHY: We use a pretrained tokenizer because:
             1. Training a tokenizer from scratch takes weeks of CPU time
             2. GPT-2's tokenizer is open-source, fast, and well-tested
             3. Using the same tokenizer as production models means our
                code works identically to how GPT-3 tokenizes
        """
        self.config = config or TokenizerConfig()

        # WHAT: Load the GPT-2 encoding from tiktoken
        # WHY: tiktoken stores pretrained BPE merge tables.
        #      get_encoding("gpt2") loads the exact 50K merges
        #      that GPT-2 was trained with.
        self.enc = tiktoken.get_encoding(self.config.name)

        # WHAT: Define and encode the End-of-Sequence token
        # WHY: <|endoftext|> is the special token that marks boundaries
        #      between documents. During training, we insert it between
        #      every document so the model learns where one text ends
        #      and another begins.
        self.eos_token = "<|endoftext|>"       # The string representation
        self.eos_token_id = self.enc.encode(    # Convert to its token ID
            self.eos_token,
            allowed_special={self.eos_token}    # WHY: tiktoken blocks special tokens
                                                #      by default for safety. We must
                                                #      explicitly allow EOS encoding.
        )[0]  # [0] because encode() returns a list — we want the single ID

    def encode(self, text: str) -> list[int]:
        """
        WHAT: Turn text into a list of integer token IDs.
        WHY: Neural networks only eat numbers. Raw strings like
             "Hello world" mean nothing to matrix multiplication.

        Example: "Hello world" -> [15496, 995]

        Under the hood: tiktoken splits the text into subword pieces
        using the pretrained BPE merge table, then looks up each
        piece's ID in the vocabulary.
        """
        # WHAT: Use tiktoken's fast C/Rust-based encoder
        # WHY: tiktoken is written in Rust, not Python.
        #      It can tokenize hundreds of MB of text per second.
        #      A pure Python BPE tokenizer would be 100x slower.
        return self.enc.encode(text, allowed_special={self.eos_token})

    def decode(self, ids: list[int]) -> str:
        """
        WHAT: Turn token IDs back into human-readable text.
        WHY: After the model generates a sequence of token IDs
             during inference, we need to convert them back to
             text so humans can read the output.

        Example: [15496, 995] -> "Hello world"
        """
        return self.enc.decode(ids)

    @property
    def vocab_size(self) -> int:
        """
        WHAT: How many unique tokens exist in the vocabulary.
        WHY: This number determines the size of our model's output
             layer — the final Linear layer must have vocab_size
             outputs (one score for each possible next token).
             
             50,257 means the model chooses from 50,257 possibilities
             every time it predicts the next word.
        """
        return self.config.vocab_size


# ===== WHAT: Quick self-test =====
# WHY: Always test each component in isolation before combining.
#      "Does the tokenizer work?" is a 5-second check that saves
#      hours of debugging a misbehaving training loop.
if __name__ == "__main__":
    tokenizer = SimpleTokenizer()

    # Test 1: Basic text
    test_text = "The cat sat on the mat."
    encoded = tokenizer.encode(test_text)
    decoded = tokenizer.decode(encoded)
    print(f"Test 1 — Basic:")
    print(f"  Original: '{test_text}'")
    print(f"  Encoded:  {encoded}")
    print(f"  Decoded:  '{decoded}'")
    print(f"  Match:    {test_text == decoded}")

    # Test 2: EOS token
    eos = tokenizer.encode(tokenizer.eos_token)
    print(f"\nTest 2 — EOS token:")
    print(f"  String: '{tokenizer.eos_token}'")
    print(f"  Token ID: {tokenizer.eos_token_id}")
    print(f"  Encode result: {eos}")

    # Test 3: Rare/unseen word
    rare = tokenizer.encode("antidisestablishmentarianism")
    decoded_rare = tokenizer.decode(rare)
    print(f"\nTest 3 — Rare word:")
    print(f"  Encoded: {rare}")
    print(f"  Pieces:  {[tokenizer.decode([t]) for t in rare]}")
    print(f"  Decoded: '{decoded_rare}'")

    # Test 4: Emoji/Unicode
    emoji = tokenizer.encode("Hello 😊 world")
    print(f"\nTest 4 — Emoji:")
    print(f"  Encoded: {emoji}")
    print(f"  Decoded: '{tokenizer.decode(emoji)}'")

    print(f"\n  Vocab size: {tokenizer.vocab_size:,}")
