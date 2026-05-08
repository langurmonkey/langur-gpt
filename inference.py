import torch
import os
from tokenizer import SimpleTokenizer
from gpt import GPT


class GPTInference:
    """
    WHAT: Simple wrapper for inference with a trained GPT model
    WHY: Encapsulates loading, device management, and generation
         so you don't have to worry about boilerplate
    """

    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Load a trained model from checkpoint.

        Args:
            checkpoint_path: Path to the saved checkpoint (e.g., "checkpoints/model.pt")
            device: "cuda", "cpu", or None (auto-detect)
        """
        # ===== DEVICE SETUP =====
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        print(f"Using device: {self.device}")

        # ===== LOAD CHECKPOINT =====
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        print(f"Loading checkpoint from {checkpoint_path}...")
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # ===== RESTORE CONFIG & MODEL =====
        config = checkpoint["config"]
        self.model = GPT(config)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()  # Disable dropout
        print(f"✓ Model loaded ({self.model.get_num_params():,} parameters)")

        # ===== LOAD TOKENIZER =====
        self.tokenizer = SimpleTokenizer()
        print("✓ Tokenizer loaded")

        # Cache config for generation
        self.config = config

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = None,
        verbose: bool = False,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Starting text (e.g., "The future of AI")
            max_new_tokens: How many tokens to generate (default 50)
            temperature: Randomness (1.0 = natural, <1.0 = focused, >1.0 = creative)
            top_k: Keep only top K most likely tokens (None = off)
            top_p: Nucleus sampling (0-1, None = off)
            verbose: Print generation progress

        Returns:
            Generated text including the prompt
        """
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"Prompt: {prompt}")
            print(f"{'=' * 60}")

        # Encode prompt to token IDs
        input_ids = self.tokenizer.encode(prompt)
        input_tensor = torch.tensor([input_ids], dtype=torch.long, device=self.device)

        if verbose:
            print(f"Prompt tokens: {len(input_ids)}")

        # Generate using model's generate method
        with torch.no_grad():
            output_ids = self.model.generate(
                input_tensor,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )

        # Decode back to text
        output_text = self.tokenizer.decode(output_ids[0].tolist())

        if verbose:
            print(f"Generated tokens: {max_new_tokens}")
            print(f"Total tokens: {len(output_ids[0])}")

        return output_text

    def generate_batch(
        self,
        prompts: list,
        max_new_tokens: int = 50,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = None,
    ) -> list:
        """
        Generate text for multiple prompts at once (faster than one-by-one).

        Args:
            prompts: List of prompt strings
            max_new_tokens: Tokens to generate per prompt
            temperature: Randomness
            top_k: Top-K sampling
            top_p: Nucleus sampling

        Returns:
            List of generated texts (same order as input prompts)
        """
        results = []
        for prompt in prompts:
            text = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
            )
            results.append(text)
        return results

    def generate_interactive(
        self,
        max_new_tokens: int = 100,
        temperature: float = 0.8,
        top_k: int = 50,
    ):
        """
        Interactive generation: keep prompting until user quits.
        Type 'quit' or 'exit' to stop.
        """
        print(f"\n{'=' * 60}")
        print("Interactive Generation Mode")
        print("Type 'quit' or 'exit' to stop")
        print(f"{'=' * 60}\n")

        while True:
            prompt = input("Prompt: ").strip()

            if prompt.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            if not prompt:
                print("(empty prompt, try again)\n")
                continue

            output = self.generate(
                prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                verbose=False,
            )
            print(f"\nOutput:\n{output}\n")
            print("-" * 60 + "\n")


def main():
    """Example usage of the inference script."""
    import argparse

    parser = argparse.ArgumentParser(description="GPT Inference")
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--mode",
        choices=["single", "batch", "interactive"],
        default="batch",
        help="Inference mode",
    )
    parser.add_argument(
        "--prompt",
        help="Prompt for single mode",
    )
    parser.add_argument(
        "--tokens",
        type=int,
        default=50,
        help="Number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.8,
        help="Temperature for sampling (0.1-2.0)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=50,
        help="Top-K sampling (0 = off)",
    )
    parser.add_argument(
        "--device",
        choices=["cuda", "cpu"],
        help="Device to use (auto-detect if not specified)",
    )

    args = parser.parse_args()

    # ===== LOAD MODEL =====
    print("Loading model...\n")
    inference = GPTInference(args.checkpoint, device=args.device)

    # ===== INFERENCE MODES =====
    if args.mode == "single":
        if not args.prompt:
            print("Error: --prompt required for single mode")
            return

        output = inference.generate(
            args.prompt,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
            verbose=True,
        )
        print(f"\nGenerated Output:\n{output}\n")

    elif args.mode == "batch":
        # Example prompts
        prompts = [
            "The future of artificial intelligence",
            "In the beginning, there was",
            "The most important discovery in science",
            "Once upon a time",
        ]

        print("Generating for multiple prompts...\n")
        results = inference.generate_batch(
            prompts,
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )

        for prompt, output in zip(prompts, results):
            print(f"Prompt: {prompt}")
            print(f"Output: {output}")
            print("-" * 60)
            print()

    elif args.mode == "interactive":
        inference.generate_interactive(
            max_new_tokens=args.tokens,
            temperature=args.temperature,
            top_k=args.top_k if args.top_k > 0 else None,
        )


if __name__ == "__main__":
    main()
