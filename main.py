import torch
import os
from tokenizer import SimpleTokenizer
from gptconfig import GPTConfig
from training import TextDataset
from training import load_training_data
from training import train
from training import plot_loss
from gpt import GPT

def main():
    print("Langur-GPT\n")

    # TINY MODEL (works on CPU, ~2-5 minutes)
    # config = GPTConfig(
    #     d_model=256, num_heads=4, num_layers=4, max_seq_len=128,
    #     batch_size=4, grad_accum_steps=2, max_steps=500,
    #     warmup_steps=50, learning_rate=3e-4,
    # )

    # SMALL MODEL (GPT-2 scale, needs GPU)
    config = GPTConfig(
        d_model=768, num_heads=12, num_layers=12, max_seq_len=1024,
        batch_size=8, grad_accum_steps=8, max_steps=50000,
        warmup_steps=2000, learning_rate=3e-4,
    )

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    tokenizer = SimpleTokenizer()
    print("Loading training data...")
    texts = load_training_data(max_samples=5000)
    train_dataset = TextDataset(texts, tokenizer, max_seq_len=config.max_seq_len)

    print("Creating model...")
    model = GPT(config)
    model.to(device)
    print(f"Parameters: {model.get_num_params():,}")

    print("\nCompiling model with torch.compile()...")
    model = torch.compile(model, mode="reduce-overhead")
    print("✓ Model compiled!")

    print("\n" + "=" * 50)
    print("TRAINING")
    print("=" * 50)
    loss_history = train(model, train_dataset, config, device)
    plot_loss(loss_history)

    print("\n" + "=" * 50)
    print("GENERATING")
    print("=" * 50 + "\n")

    prompts = [
        "The history of artificial intelligence",
        "In the beginning the universe",
        "The most important scientific discovery",
    ]

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        output_ids = model.generate(input_ids, max_new_tokens=50, temperature=0.8, top_k=50)
        text = tokenizer.decode(output_ids[0].tolist())
        print(f"Prompt: {prompt}")
        print(f"Output: {text}")
        print("-" * 50)
        print()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config,
    }, "checkpoints/model.pt")
    print("Model saved to checkpoints/model.pt")

if __name__ == "__main__":
    main()
