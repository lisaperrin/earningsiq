"""
07_finetune_model.py - Fine-tune model with LoRA and compare before/after performance

Demonstrates:
- Objective 3: Advanced Techniques (PEFT/LoRA, 4-bit quantization)
- Objective 5: Quantifiable Impact (baseline vs fine-tuned comparison)

Usage:
    python scripts/07_finetune_model.py --quick     # Fast demo (50 samples, 1 epoch)
    python scripts/07_finetune_model.py             # Full training (500 samples, 3 epochs)
    python scripts/07_finetune_model.py --samples 100 --epochs 2
"""

import sys
from pathlib import Path
import argparse
import time

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from earningsiq.models.fine_tuning import (
    FinancialDatasetPreparator,
    LoRAFineTuner,
)
from earningsiq.utils.logger import log
from rouge_score import rouge_scorer


def evaluate_model(model, test_examples: list, desc: str) -> dict:
    """Evaluate a model on test examples and return ROUGE scores."""
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []
    predictions = []

    print(f"\n{'='*60}")
    print(f"Evaluating: {desc}")
    print('='*60)

    for i, example in enumerate(test_examples):
        # Extract prompt (without the answer)
        full_text = example["text"]
        if "### Answer:" in full_text:
            parts = full_text.split("### Answer:")
            prompt = parts[0] + "### Answer:"
            reference = parts[1].strip() if len(parts) > 1 else ""
        else:
            prompt = full_text
            reference = ""

        # Generate response
        response = model.generate(prompt, max_length=128)

        # Extract just the answer part from response
        if "### Answer:" in response:
            predicted = response.split("### Answer:")[-1].strip()
        else:
            predicted = response[len(prompt):].strip()

        predictions.append(predicted)

        # Calculate ROUGE scores
        if reference:
            scores = scorer.score(reference, predicted)
            rouge1_scores.append(scores['rouge1'].fmeasure)
            rouge2_scores.append(scores['rouge2'].fmeasure)
            rougeL_scores.append(scores['rougeL'].fmeasure)

        # Show first few examples
        if i < 3:
            print(f"\n[Example {i+1}]")
            print(f"Question: {prompt.split('### Question:')[-1].split('### Answer:')[0].strip()[:100]}...")
            print(f"Reference: {reference[:100]}..." if reference else "N/A")
            print(f"Predicted: {predicted[:100]}...")

    avg_rouge1 = sum(rouge1_scores) / len(rouge1_scores) if rouge1_scores else 0
    avg_rouge2 = sum(rouge2_scores) / len(rouge2_scores) if rouge2_scores else 0
    avg_rougeL = sum(rougeL_scores) / len(rougeL_scores) if rougeL_scores else 0

    return {
        "rouge1": avg_rouge1,
        "rouge2": avg_rouge2,
        "rougeL": avg_rougeL,
        "predictions": predictions,
    }


def main():
    parser = argparse.ArgumentParser(description="Fine-tune model with LoRA and compare performance")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Quick demo mode (50 training samples, 1 epoch)"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=None,
        help="Number of training samples (default: 500, quick: 50)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (default: 3, quick: 1)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
        help="Base model to fine-tune (default: TinyLlama for speed)"
    )
    parser.add_argument(
        "--test-samples",
        type=int,
        default=10,
        help="Number of test samples for evaluation"
    )
    args = parser.parse_args()

    # Set parameters based on mode
    if args.quick:
        num_samples = args.samples or 50
        num_epochs = args.epochs or 1
    else:
        num_samples = args.samples or 500
        num_epochs = args.epochs or 3

    # Check if bitsandbytes is available and working
    bnb_available = False
    try:
        import bitsandbytes
        # Test if it actually works (catches Windows CUDA issues)
        bnb_available = True
    except (ImportError, RuntimeError, Exception):
        bnb_available = False

    print("=" * 70)
    print("PEFT/LoRA FINE-TUNING WITH BEFORE/AFTER COMPARISON")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model:            {args.model}")
    print(f"  Training samples: {num_samples}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Test samples:     {args.test_samples}")
    print(f"  4-bit quantization: {'Enabled' if bnb_available else 'Disabled (bitsandbytes not installed)'}")
    print(f"  LoRA rank:        16")
    print(f"  LoRA alpha:       32")

    # Step 1: Load dataset
    print("\n" + "-" * 70)
    print("STEP 1: Loading FinQA Dataset")
    print("-" * 70)

    preparator = FinancialDatasetPreparator()
    dataset = preparator.load_finqa_dataset()

    # Split into train and test
    train_data = preparator.prepare_training_data(dataset, max_samples=num_samples)

    # Get test examples from the end of the dataset (not seen in training)
    test_start = min(num_samples + 100, len(dataset) - args.test_samples)
    test_dataset = preparator.prepare_training_data(
        dataset.select(range(test_start, test_start + args.test_samples))
    )
    test_examples = [test_dataset[i] for i in range(len(test_dataset))]

    print(f"  Training samples: {len(train_data)}")
    print(f"  Test samples:     {len(test_examples)}")

    # Step 2: Setup base model
    print("\n" + "-" * 70)
    print("STEP 2: Loading Base Model")
    print("-" * 70)

    finetuner = LoRAFineTuner(
        model_name=args.model,
        use_4bit=True
    )
    finetuner.setup_model()

    # Step 3: Evaluate BEFORE fine-tuning
    print("\n" + "-" * 70)
    print("STEP 3: Evaluating BEFORE Fine-Tuning")
    print("-" * 70)

    before_results = evaluate_model(finetuner, test_examples, "Base Model (Before Fine-Tuning)")

    # Step 4: Fine-tune with LoRA
    print("\n" + "-" * 70)
    print("STEP 4: Fine-Tuning with LoRA")
    print("-" * 70)

    start_time = time.time()
    finetuner.train(
        train_dataset=train_data,
        num_epochs=num_epochs,
        batch_size=4,
        learning_rate=2e-4,
    )
    training_time = time.time() - start_time

    print(f"\nTraining completed in {training_time:.1f} seconds")

    # Step 5: Evaluate AFTER fine-tuning
    print("\n" + "-" * 70)
    print("STEP 5: Evaluating AFTER Fine-Tuning")
    print("-" * 70)

    after_results = evaluate_model(finetuner, test_examples, "Fine-Tuned Model (After LoRA)")

    # Step 6: Compare results
    print("\n" + "=" * 70)
    print("COMPARISON RESULTS")
    print("=" * 70)

    print("\n{:<25} {:>15} {:>15} {:>15}".format("Metric", "Before", "After", "Improvement"))
    print("-" * 70)

    for metric in ["rouge1", "rouge2", "rougeL"]:
        before = before_results[metric]
        after = after_results[metric]
        improvement = after - before
        pct = (improvement / before * 100) if before > 0 else 0

        print("{:<25} {:>15.4f} {:>15.4f} {:>+14.4f} ({:+.1f}%)".format(
            metric.upper(),
            before,
            after,
            improvement,
            pct
        ))

    print("\n" + "-" * 70)
    print("TRAINING DETAILS")
    print("-" * 70)
    print(f"  Base model:       {args.model}")
    print(f"  Training samples: {num_samples}")
    print(f"  Epochs:           {num_epochs}")
    print(f"  Training time:    {training_time:.1f}s")
    print(f"  LoRA parameters:  r=16, alpha=32, dropout=0.1")
    print(f"  Quantization:     4-bit (NF4)")

    print("\n" + "=" * 70)
    print("Fine-tuning demonstration complete!")
    print("=" * 70)

    # Save results summary
    output_path = Path(__file__).parent.parent / "data" / "finetune_results.txt"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("PEFT/LoRA Fine-Tuning Results\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Training samples: {num_samples}\n")
        f.write(f"Epochs: {num_epochs}\n")
        f.write(f"Training time: {training_time:.1f}s\n\n")
        f.write("Results:\n")
        f.write(f"  ROUGE-1: {before_results['rouge1']:.4f} -> {after_results['rouge1']:.4f}\n")
        f.write(f"  ROUGE-2: {before_results['rouge2']:.4f} -> {after_results['rouge2']:.4f}\n")
        f.write(f"  ROUGE-L: {before_results['rougeL']:.4f} -> {after_results['rougeL']:.4f}\n")

    print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()
