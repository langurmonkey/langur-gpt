# Langur GPT

> A modular, from-scratch GPT-style language model implementation — the hands-on code
> behind [how-to-train-your-gpt](https://github.com/raiyanyahya/how-to-train-your-gpt).

This is a standalone, **modular refactor** of the concepts from the tutorial. The chapters
have been split into individually importable files, each representing a real component of a
modern decoder-only Transformer: tokenizer, RoPE, attention, transformer block, training
loop, inference — all wired together in a runnable `main.py`.

## Architecture

The model implements the **publicly-confirmed best practices** used by LLaMA 3, Mistral,
and Qwen 2.5:

| Technique | What it does | Origin |
|---|---|---|
| **RoPE** | Encodes position by rotating Q/K vectors — attention depends on relative distance, not absolute position | LLaMA, Mistral, Qwen |
| **RMSNorm** | Replaces LayerNorm (15% faster, equally stable) | LLaMA, Gemma |
| **SwiGLU** | Gated FFN that learns to selectively pass/block information | PaLM, LLaMA, Gemini |
| **Pre-Norm** | Normalize *before* each sublayer — stable at 100+ layers | All modern Transformers |
| **Weight Tying** | Shared embedding/LM-head matrix — 30% parameter savings | GPT-2/3 |
| **AdamW** | Decoupled weight decay — better generalization | GPT-3+ |
| **Cosine Warmup** | Linear ramp → cosine decay → floor — stable training | GPT-3+ |
| **Mixed Precision** | FP16/BF16 training — 2x speed, half memory | All production LLMs |

## File Map

```
├── main.py           Entry point — train a model, then generate text from prompts
├── gpt.py            GPT model class — ties together embeddings, Nx transformer blocks,
│                     final norm, and the LM head with weight tying
├── gptconfig.py      Dataclass with all hyperparameters in one place
├── transformer.py    TransformerBlock + RMSNorm + SwiGLU
├── attention.py      Multi-head self-attention with RoPE, causal mask, fused QKV
├── rope.py           Rotary Positional Embeddings (precomputed cos/sin cache)
├── embedding.py      Token embedding lookup with sqrt(d_model) scaling
├── tokenizer.py      BPE tokenizer wrapping tiktoken (GPT-2 vocabulary)
├── training.py       Training loop, TextDataset, AdamW optimizer, cosine warmup
│                     scheduler, checkpointing, loss curve plotting
├── polyopt.py        Standalone sine-wave fitting experiment (polynomial via SGD)
├── cudatest.py       Quick GPU availability check
├── pyproject.toml    Project metadata and dependencies
└── uv.lock           Lockfile for uv / pip
```

### What each component does

**`tokenizer.py`** — Wraps OpenAI's `tiktoken` with the GPT-2 BPE vocabulary
(50,257 tokens). Handles encode/decode, automatically inserts `<|endoftext|>` markers
between documents during dataset construction.

**`embedding.py`** — A learnable lookup table mapping each of the 50,257 token IDs to a
dense vector (768 dimensions in the default config). Scales by `sqrt(d_model)` so that
embedding magnitudes don't drown out positional information.

**`rope.py`** — Rotary Position Embeddings. Precomputes cos/sin tables for all positions
up to `max_seq_len`. Instead of adding position numbers, it *rotates* Q and K vectors by
position-dependent angles. Lower dimensions rotate fast (capture local word order),
higher dimensions rotate slow (capture long-range relationships).

**`attention.py`** — Multi-head self-attention with a fused QKV projection (one big
Linear layer → split into Q, K, V — faster on GPU than three separate projections).
Applies RoPE to Q and K only, computes scaled dot-product attention with a causal
(lower-triangular) mask, then concatenates heads and projects back.

**`transformer.py`** — One transformer block: pre-norm → attention (+ residual) →
pre-norm → SwiGLU FFN (+ residual). Includes `RMSNorm` (root-mean-square normalization,
no mean-centering) and `SwiGLU` (gated activation: SiLU(w1(x)) * w2(x)).

**`gpt.py`** — The full model. Stacks N `TransformerBlock` layers, applies final
normalization, projects to vocabulary via the LM head. Weight tying shares the
embedding matrix with the LM head. Includes the inference method (`generate`) with
temperature, top-k, and top-p sampling.

**`gptconfig.py`** — Single dataclass defining model architecture (vocab size, d_model,
layers, heads, max_seq_len) and training hyperparameters (LR, batch size, warmup steps,
weight decay, etc.). Also validates that d_model is divisible by num_heads.

**`training.py`** — The full training pipeline:
- `TextDataset`: concatenates documents with EOS separators, yields shifted
  input/target pairs for next-token prediction (teacher forcing)
- `CosineWarmupScheduler`: three-phase LR schedule (linear warmup → cosine decay → floor)
- `create_optimizer`: AdamW with separate parameter groups (weight decay on weights only,
  none on biases/norms)
- `train`: the main loop — forward/backward/update with gradient accumulation, mixed
  precision (AMP), gradient clipping, periodic logging, and checkpointing
- `plot_loss`: renders a loss curve to `loss_curve.png`

**`main.py`** — Entry point. Loads the Wikitext-103 dataset, creates the model, runs the
training loop, then generates continuations from a few example prompts. Saves the final
model to `checkpoints/model.pt`.

**`polyopt.py`** — A standalone script that fits a 7th-degree polynomial to sin(x) using
AdamW. Not part of the GPT pipeline — it's a minimal PyTorch exercise included for
learning purposes.

**`cudatest.py`** — Prints PyTorch version and CUDA availability. Run this first to
verify your GPU setup.

## Quick Start

```bash
# 1. Create environment (Python 3.12+)
uv venv
source .venv/bin/activate       # or: .venv\Scripts\activate (Windows)

# 2. Install dependencies
uv pip install -r <(uv pip compile pyproject.toml)   # with uv
# or: pip install torch tiktoken datasets numpy matplotlib

# 3. Verify GPU
python cudatest.py

# 4. Train!
python main.py
```

### Two configurations

**Small model (GPT-2 scale)** — default, requires a GPU (~2 hours on an RTX 3090):

```
d_model=768, num_heads=12, num_layers=12, max_seq_len=1024
batch_size=4, grad_accum=8, max_steps=50,000
```

**Tiny model (CPU-friendly)** — swap the config in `main.py` (uncomment the tiny block,
comment the small block):

```
d_model=256, num_heads=4, num_layers=4, max_seq_len=128
batch_size=4, grad_accum=2, max_steps=500
```

### Output

After training, the script:
1. Logs loss every 100 steps with tokens/sec throughput
2. Saves a loss curve to `loss_curve.png`
3. Generates text continuations from three example prompts
4. Saves the trained model to `checkpoints/model.pt`
5. Saves periodic checkpoints to `checkpoints/checkpoint_step_{N}.pt`

Expected loss trajectory (GPT-2 scale):
```
Step    100/50,000 | Loss: 6.23
Step    500/50,000 | Loss: 4.85
Step  5,000/50,000 | Loss: 3.42
Step 50,000/50,000 | Loss: ~2.89
```

## Why

Most ML tutorials show you how to call `model.fit()` or `model.generate()`. This
codebase does the opposite — every line is annotated with **what** it does and **why**
it's there, so you can read the files in any order and understand the full pipeline.
It's the working implementation that results from working through
[how-to-train-your-gpt](https://github.com/raiyanyahya/how-to-train-your-gpt).

## Next Steps / Experiments

- **Bigger model** — increase `num_layers` to 24 or `d_model` to 1024
- **Grouped Query Attention** — add `num_kv_heads` (like Mistral)
- **Flash Attention** — swap the attention module for `flash-attn`
- **LoRA** — add low-rank adapter layers for efficient fine-tuning
- **KV Cache** — implement persistent key-value caching for faster inference
- **Mixture of Experts** — route tokens through different FFNs (like GPT-4)
