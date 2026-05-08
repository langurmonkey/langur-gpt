import time
import os
import math
import glob
import torch
from torch.utils.data import Dataset
from datasets import load_dataset
from tqdm import tqdm

class TextDataset(Dataset):
    """
    WHAT: Prepares text data by splitting into training chunks.
    WHY: The model learns to predict the next token. Each chunk
         provides input-target pairs for next-token prediction.

         Each sample: input[t] and target[t+1] for all positions t.
         This is called "teacher forcing" — we show the correct
         answer for every position during training.
    """

    def __init__(self, texts: list[str], tokenizer, max_seq_len: int = 1024):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # ===== Concatenate all texts with EOS separators =====
        # WHY: EOS prevents the model from learning false connections
        #      between unrelated documents.
        all_tokens = []
        for text in texts:
            tokens = tokenizer.encode(text)
            all_tokens.extend(tokens)
            all_tokens.append(tokenizer.eos_token_id)  # Document boundary marker

        self.tokens = torch.tensor(all_tokens, dtype=torch.long)
        print(f"Total tokens in dataset: {len(self.tokens):,}")

    def __len__(self) -> int:
        """Number of chunks. Each uses max_seq_len+1 tokens."""
        return (len(self.tokens) - 1) // self.max_seq_len

    def __getitem__(self, idx: int) -> tuple:
        """
        Returns (input_ids, target_ids) for one chunk.
        Target is shifted by 1 position:

        tokens:    [The,  cat,  sat,  on,   the,  mat,  EOS,  The,  dog,  ...]
        idx=0:     [The,  cat,  sat,  on,   the]     ← input_ids
                   [cat,  sat,  on,   the,  mat]     ← target_ids (shifted)
        """
        start = idx * self.max_seq_len
        end = start + self.max_seq_len
        input_ids = self.tokens[start:end]
        target_ids = self.tokens[start + 1 : end + 1]
        return input_ids, target_ids


def load_training_data(max_samples: int = None):
    """Download WikiText-103 — clean Wikipedia text."""
    print("Loading dataset: wikitext-103-raw-v1...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")
    texts = [item["text"] for item in dataset if item["text"].strip()]
    if max_samples:
        texts = texts[:max_samples]
    print(f"Loaded {len(texts):,} documents")
    return texts

class CosineWarmupScheduler:
    """
    WHAT: Three-phase learning rate schedule.
    WHY: Warmup prevents early instability. Cosine decay provides
         smooth convergence. Minimum floor prevents zero learning.

    Phase 1 (Warmup):    LR: 0 → max_lr  (linear increase over warmup_steps)
    Phase 2 (Decay):     LR: max_lr → min_lr (cosine curve)
    Phase 3 (Minimum):   LR: min_lr (constant)
    """
    def __init__(self, optimizer, warmup_steps, max_steps, max_lr=3e-4, min_lr=1e-5):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.max_lr = max_lr
        self.min_lr = min_lr
        self.current_step = 0

    def get_lr(self) -> float:
        step = self.current_step
        if step < self.warmup_steps:
            return self.max_lr * step / self.warmup_steps
        if step < self.max_steps:
            progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            return self.min_lr + (self.max_lr - self.min_lr) * cosine_decay
        return self.min_lr

    def step(self):
        lr = self.get_lr()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        self.current_step += 1

    def state_dict(self):
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict):
        self.current_step = state_dict["current_step"]


def create_optimizer(model, config):
    """
    WHAT: AdamW with two parameter groups (with/without weight decay).
    WHY: Norm layers and biases should NOT get weight decay — it
         pushes them toward zero, destroying normalization.

    Group 1 (weight_decay > 0): Linear weights, embeddings
    Group 2 (weight_decay = 0): Biases, RMSNorm, LayerNorm
    """
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.dim() <= 1 or "norm" in name.lower() or "bias" in name:
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return torch.optim.AdamW(
        [
            {"params": decay_params, "weight_decay": config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ],
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
    )


def train(model, train_dataset, config, device, save_dir="checkpoints"):
    """
    WHAT: The main training loop.
    WHY: Iterates: forward → backward → update, logging and saving periodically.
    """
    os.makedirs(save_dir, exist_ok=True)
    model = model.to(device)
    model.train()

    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size,
        shuffle=True, drop_last=True, num_workers=4, pin_memory=True,
    )

    optimizer = create_optimizer(model, config)
    scheduler = CosineWarmupScheduler(
        optimizer, warmup_steps=config.warmup_steps,
        max_steps=config.max_steps, max_lr=config.learning_rate,
    )

    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler('cuda', enabled=use_amp) if use_amp else None

    step = 0
    total_loss = 0.0
    loss_history = []
    best_loss = float("inf")
    start_time = time.time()
    batch_times = []
    max_batch_times = 20  # Keep rolling average of last 20 batches

    # ===== RESUME TRAINING FROM CHECKPOINT =====
    checkpoint_files = sorted(glob.glob(os.path.join(save_dir, "checkpoint_step_*.pt")))
    if checkpoint_files:
        latest_checkpoint_path = checkpoint_files[-1]
        print(f"Found checkpoint: {latest_checkpoint_path}. Resuming training...")
        checkpoint = torch.load(latest_checkpoint_path, map_location=device)
        
        step = checkpoint["step"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        best_loss = checkpoint.get("best_loss", float("inf")) # Load best_loss if available, else default
        
        # Load loss history if saved in checkpoint. This assumes loss_history
        # is always saved in the checkpoint when it's generated.
        if "loss_history" in checkpoint:
            loss_history = checkpoint["loss_history"]
            # We need to re-calculate total_loss for accurate averaging if we are resuming
            # Average loss from the last logged point in loss_history
            if loss_history:
                total_loss = loss_history[-1][1] * 100 # Approximate if step % 100 != 0
        
        print(f"Resumed training from step {step}. Best loss so far: {best_loss:.4f}")
    else:
        print("No checkpoints found. Starting training from scratch.")

    print(f"\n{'='*70}")
    print(f"Training! Params: {model.get_num_params():,} | Device: {device}")
    print(f"Effective batch: {config.batch_size * config.grad_accum_steps}")
    print(f"Max steps: {config.max_steps:,}")
    print(f"{'='*70}\n")

    # Create progress bar
    pbar = tqdm(total=config.max_steps, desc="Training", unit="step", dynamic_ncols=True, initial=step)

    # Store original `step` to properly resume ETA calculation
    original_start_step = step 
    stable_step_start = None # Initialize for ETA tracking
    stable_time_start = None

    while step < config.max_steps:
        for batch_idx, (input_ids, target_ids) in enumerate(dataloader):
            if step >= config.max_steps:
                break

            batch_start = time.time()

            # ===== CUDA GRAPH STEP MARKING (for torch.compile()) =====
            # WHAT: Tell torch.compile() that a new iteration is starting
            # WHY: Prevents CUDA graph conflicts with gradient accumulation
            if device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()

            input_ids = input_ids.to(device, non_blocking=True)
            target_ids = target_ids.to(device, non_blocking=True)

            # ===== FORWARD: Predict next tokens, measure error =====
            with torch.amp.autocast('cuda', enabled=use_amp):
                _, loss = model(input_ids, targets=target_ids)
            loss = loss / config.grad_accum_steps

            # ===== BACKWARD: Calculate how to improve =====
            if use_amp and scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            # Accumulate loss for logging, adjust by grad_accum_steps
            # total_loss += loss.item() * config.grad_accum_steps # This line was correct before

            # Calculate the loss for logging correctly *before* gradient accumulation division
            # Each loss.item() here is already a loss/grad_accum_steps value.
            # So, multiply by grad_accum_steps to get the true per-batch loss.
            if step < original_start_step + 10: # Only consider loss after resuming to re-establish moving average
                total_loss += loss.item() * config.grad_accum_steps


            # ===== UPDATE: Every grad_accum_steps, optimize =====
            if (batch_idx + 1) % config.grad_accum_steps == 0:
                if use_amp and scaler is not None:
                    scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                if use_amp and scaler is not None:
                    scaler.step(optimizer); scaler.update()
                else:
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()
                step += 1

                # ===== ACCURATE ETA CALCULATION =====
                # Skip warmup steps for ETA (they're artificially fast)
                elapsed = time.time() - start_time
                
                if step == warmup_end_step:
                    # Start tracking "stable" speed after warmup
                    stable_step_start = step
                    stable_time_start = elapsed
                    pbar.write(f"💡 Warmup complete, ETA will be accurate from now on")
                
                if stable_step_start is not None:
                    # Use speed from after-warmup period only
                    stable_elapsed = elapsed - stable_time_start
                    stable_steps = step - stable_step_start
                    seconds_per_step = stable_elapsed / stable_steps if stable_steps > 0 else 0
                else:
                    # Still in warmup, show "calculating"
                    seconds_per_step = 0
                
                if seconds_per_step > 0:
                    steps_remaining = config.max_steps - step
                    eta_seconds = steps_remaining * seconds_per_step
                    eta_hours = eta_seconds / 3600
                else:
                    eta_hours = 0
                
                # Tokens per second
                tokens_per_batch = config.batch_size * config.grad_accum_steps * config.max_seq_len
                if seconds_per_step > 0:
                    tps = tokens_per_batch / seconds_per_step / 1000
                else:
                    tps = 0
                
                # Current loss (moving average)
                # Calculate avg_loss based on accumulated total_loss and the current step count
                # Ensure we don't divide by zero if step is 0 (or very early in resume)
                current_log_step = step if step > 0 else 1
                avg_loss = total_loss / (100 if current_log_step > 100 else current_log_step)
                
                # GPU memory (if available)
                if device.type == "cuda":
                    allocated = torch.cuda.memory_allocated(device) / 1024**3
                    postfix_dict = {
                        "Loss": f"{avg_loss:.4f}",
                        "Tk/s": f"{tps:.1f}K",
                        "GPU": f"{allocated:.1f}GB",
                    }
                    if eta_hours > 0:
                        postfix_dict["ETA"] = f"{eta_hours:.1f}h"
                    else:
                        postfix_dict["ETA"] = "computing..."
                else:
                    postfix_dict = {
                        "Loss": f"{avg_loss:.4f}",
                        "Tk/s": f"{tps:.1f}K",
                    }
                    if eta_hours > 0:
                        postfix_dict["ETA"] = f"{eta_hours:.1f}h"
                
                pbar.update(1)
                pbar.set_postfix(postfix_dict)

                # Update loss history
                if step % 100 == 0 or step == 1:
                    loss_history.append((step, avg_loss))
                    total_loss = 0.0

                # Save checkpoint every 5000 steps
                if step % 5000 == 0:
                    checkpoint = {
                        "step": step, "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "loss": avg_loss, "config": config,
                    }
                    torch.save(checkpoint, f"{save_dir}/checkpoint_step_{step}.pt")
                    pbar.write(f"💾 Saved checkpoint at step {step} | Best: {best_loss:.4f}")
                    if avg_loss < best_loss:
                        best_loss = avg_loss
                        torch.save(checkpoint, f"{save_dir}/best_model.pt")
                        pbar.write(f"🎯 New best loss: {best_loss:.4f}")

    pbar.close()

    total_time = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"✅ Done! {total_time/60:.1f} min ({total_time/3600:.2f} hours) | Best loss: {best_loss:.4f}")
    print(f"{'='*70}\n")
    return loss_history


def plot_loss(loss_history, save_path="loss_curve.png"):
    """
    WHAT: Visualize training progress.
    WHY: Loss curves diagnose problems:
         ↘ Steady decrease: training is working
         → Flat line: stalled (higher LR, check data)
         ↗ Increasing: overfitting (more dropout, weight decay)
         ⚡ Spikes: unstable (lower LR, longer warmup)
    """
    import matplotlib.pyplot as plt
    steps, losses = zip(*loss_history)
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses)
    plt.xlabel("Training Step"); plt.ylabel("Loss")
    plt.title("GPT Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout(); plt.savefig(save_path, dpi=150); plt.close()
    print(f"Loss curve saved to {save_path}")
