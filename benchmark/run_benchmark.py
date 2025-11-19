"""
Benchmarking pipeline for Army of Safeguards.

This script evaluates the safeguarding system (via aggregator) against
public benchmark datasets for safety and moderation.

Usage:
    python benchmark/run_benchmark.py --benchmark HarmBench --limit 100
    python benchmark/run_benchmark.py --benchmark WildGuardMix --limit 500
    python benchmark/run_benchmark.py --benchmark JailbreakBench --limit 200
    python benchmark/run_benchmark.py --all --limit 100
"""

import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

# Import aggregator
from aggregator.aggregator import evaluate_text


# Benchmark dataset configurations
BENCHMARKS = {
    "HarmBench": {
        "hf_id": "walledai/HarmBench",
        "config": None,
        "split": "test",
        "text_field": "goal",  # May need adjustment based on actual structure
        "label_field": "is_refusal",  # Harmful if is_refusal is True (model refused = harmful)
        "label_mapping": {
            # If is_refusal=True, model refused = content is harmful (unsafe)
            # If is_refusal=False, model didn't refuse = content is safe
            True: "unsafe",
            False: "safe"
        },
        "note": "Jailbreak & harmful-content robustness benchmark"
    },
    "JailbreakBench": {
        "hf_id": "JailbreakBench/JBB-Behaviors",
        "config": "behaviors",
        "split": None,  # Has 'harmful' and 'benign' splits
        "text_field": "Goal",
        "label_field": None,  # Label determined by split
        "label_mapping": {
            "harmful": "unsafe",
            "benign": "safe"
        },
        "note": "Jailbreak attempt detection benchmark"
    },
    "WildGuardMix": {
        "hf_id": "allenai/wildguardmix",
        "config": None,
        "split": "test",
        "text_field": "prompt",  # May need adjustment
        "label_field": "is_safe",  # May need adjustment
        "label_mapping": {
            True: "safe",
            False: "unsafe"
        },
        "note": "Moderation / guardrail benchmark"
    }
}


def load_benchmark_dataset(benchmark_name: str, limit: Optional[int] = None) -> Tuple[List[str], List[str], List[Dict[str, Any]]]:
    """
    Load a benchmark dataset and extract texts and labels.
    
    Args:
        benchmark_name: Name of the benchmark
        limit: Maximum number of examples to load
        
    Returns:
        Tuple of (texts, labels, metadata)
    """
    config = BENCHMARKS[benchmark_name]
    hf_id = config["hf_id"]
    ds_config = config.get("config")
    split = config.get("split")
    text_field = config["text_field"]
    label_field = config.get("label_field")
    label_mapping = config.get("label_mapping", {})
    
    texts = []
    labels = []
    metadata = []
    
    try:
        # Special handling for JailbreakBench which has multiple splits
        if benchmark_name == "JailbreakBench":
            # Load both harmful and benign splits
            ds_harmful = load_dataset(hf_id, ds_config, split="harmful", trust_remote_code=True)
            ds_benign = load_dataset(hf_id, ds_config, split="benign", trust_remote_code=True)
            
            # Process harmful examples
            for ex in ds_harmful:
                # Try to find text field (case-insensitive)
                text = None
                for key in ex.keys():
                    if key.lower() == text_field.lower():
                        text = ex[key]
                        break
                
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    texts.append(text)
                    labels.append("unsafe")
                    metadata.append({
                        "split": "harmful",
                        "behavior": ex.get("Behavior", ex.get("behavior", "")),
                        "category": ex.get("Category", ex.get("category", "")),
                        **{k: v for k, v in ex.items() if k.lower() != text_field.lower()}
                    })
            
            # Process benign examples
            for ex in ds_benign:
                # Try to find text field (case-insensitive)
                text = None
                for key in ex.keys():
                    if key.lower() == text_field.lower():
                        text = ex[key]
                        break
                
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    texts.append(text)
                    labels.append("safe")
                    metadata.append({
                        "split": "benign",
                        "behavior": ex.get("Behavior", ex.get("behavior", "")),
                        "category": ex.get("Category", ex.get("category", "")),
                        **{k: v for k, v in ex.items() if k.lower() != text_field.lower()}
                    })
        else:
            # Standard dataset loading
            if ds_config:
                ds = load_dataset(hf_id, ds_config, split=split, trust_remote_code=True)
            else:
                ds = load_dataset(hf_id, split=split, trust_remote_code=True)
            
            # Sample if limit is specified
            if limit and len(ds) > limit:
                import random
                indices = random.sample(range(len(ds)), limit)
                ds = ds.select(indices)
            
            # Extract texts and labels
            for ex in ds:
                # Try to find text field (case-insensitive)
                text = None
                for key in ex.keys():
                    if key.lower() == text_field.lower():
                        text = ex[key]
                        break
                
                if not text or not isinstance(text, str) or len(text.strip()) == 0:
                    continue
                
                texts.append(text)
                
                # Extract label
                if label_field:
                    # Try to find label field (case-insensitive)
                    raw_label = None
                    for key in ex.keys():
                        if key.lower() == label_field.lower():
                            raw_label = ex[key]
                            break
                    
                    if raw_label is not None:
                        # Map label
                        if isinstance(raw_label, bool):
                            mapped_label = label_mapping.get(raw_label, "unsafe" if not raw_label else "safe")
                        elif raw_label in label_mapping:
                            mapped_label = label_mapping[raw_label]
                        else:
                            # Default: assume False/0/negative means unsafe
                            mapped_label = "unsafe" if not raw_label else "safe"
                        labels.append(mapped_label)
                    else:
                        # If label field not found, skip this example
                        texts.pop()
                        continue
                else:
                    # No label field specified, skip
                    texts.pop()
                    continue
                
                # Store metadata
                metadata.append({
                    **{k: v for k, v in ex.items() 
                       if k.lower() not in [text_field.lower(), label_field.lower() if label_field else ""]}
                })
        
        # Apply limit if specified (for JailbreakBench or if limit wasn't applied earlier)
        if limit and len(texts) > limit:
            import random
            indices = random.sample(range(len(texts)), limit)
            texts = [texts[i] for i in indices]
            labels = [labels[i] for i in indices]
            metadata = [metadata[i] for i in indices]
        
        if len(texts) == 0:
            raise ValueError(f"No valid examples found in {benchmark_name}. Check field names.")
        
        return texts, labels, metadata
        
    except Exception as e:
        print(f"Error loading {benchmark_name}: {e}")
        import traceback
        traceback.print_exc()
        raise


def evaluate_on_benchmark(
    benchmark_name: str,
    limit: Optional[int] = None,
    threshold: float = 0.7,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Evaluate the safeguarding system on a benchmark dataset.
    
    Args:
        benchmark_name: Name of the benchmark
        limit: Maximum number of examples to evaluate
        threshold: Confidence threshold for flagging
        verbose: Whether to print progress
        
    Returns:
        Dictionary with evaluation results
    """
    if verbose:
        print(f"\n{'='*70}")
        print(f"Evaluating: {benchmark_name}")
        print(f"{BENCHMARKS[benchmark_name].get('note', '')}")
        print(f"{'='*70}")
    
    try:
        # Load dataset
        if verbose:
            print("Loading dataset...")
        texts, ground_truth_labels, metadata = load_benchmark_dataset(benchmark_name, limit)
        
        if len(texts) == 0:
            return {
                "benchmark": benchmark_name,
                "error": "No examples loaded from dataset"
            }
        
        if verbose:
            print(f"Loaded {len(texts)} examples")
            print(f"Running safeguarding system...\n")
        
        # Run predictions
        predictions = []
        confidences = []
        individual_results = []
        
        for text in tqdm(texts, disable=not verbose, desc="Evaluating"):
            result = evaluate_text(text, threshold=threshold)
            
            # Determine if content was flagged as unsafe
            is_flagged = not result['is_safe']
            predictions.append("unsafe" if is_flagged else "safe")
            
            # Store average confidence
            confidences.append(result.get('average_confidence', 0.0))
            
            # Store individual safeguard results
            individual_results.append({
                'flags': result.get('flags', []),
                'individual_results': result.get('individual_results', {})
            })
        
        # Calculate metrics
        accuracy = accuracy_score(ground_truth_labels, predictions)
        precision, recall, f1, support = precision_recall_fscore_support(
            ground_truth_labels,
            predictions,
            average='binary',
            pos_label='unsafe',
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(ground_truth_labels, predictions, labels=['safe', 'unsafe'])
        
        # Per-class metrics
        precision_per_class, recall_per_class, f1_per_class, support_per_class = \
            precision_recall_fscore_support(
                ground_truth_labels,
                predictions,
                labels=['safe', 'unsafe'],
                zero_division=0
            )
        
        # Count predictions
        pred_safe = predictions.count("safe")
        pred_unsafe = predictions.count("unsafe")
        true_safe = ground_truth_labels.count("safe")
        true_unsafe = ground_truth_labels.count("unsafe")
        
        results = {
            "benchmark": benchmark_name,
            "timestamp": datetime.now().isoformat(),
            "threshold": threshold,
            "total_examples": len(texts),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "average_confidence": float(sum(confidences) / len(confidences)) if confidences else 0.0,
            "confusion_matrix": {
                "safe_safe": int(cm[0][0]),
                "safe_unsafe": int(cm[0][1]),
                "unsafe_safe": int(cm[1][0]),
                "unsafe_unsafe": int(cm[1][1])
            },
            "ground_truth_distribution": {
                "safe": true_safe,
                "unsafe": true_unsafe
            },
            "prediction_distribution": {
                "safe": pred_safe,
                "unsafe": pred_unsafe
            },
            "per_class_metrics": {
                "safe": {
                    "precision": float(precision_per_class[0]),
                    "recall": float(recall_per_class[0]),
                    "f1_score": float(f1_per_class[0]),
                    "support": int(support_per_class[0])
                },
                "unsafe": {
                    "precision": float(precision_per_class[1]),
                    "recall": float(recall_per_class[1]),
                    "f1_score": float(f1_per_class[1]),
                    "support": int(support_per_class[1])
                }
            }
        }
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Results for {benchmark_name}:")
            print(f"{'='*70}")
            print(f"  Total examples: {results['total_examples']}")
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Precision: {precision:.2%}")
            print(f"  Recall: {recall:.2%}")
            print(f"  F1-Score: {f1:.2%}")
            print(f"  Average Confidence: {results['average_confidence']:.2%}")
            print(f"\n  Ground Truth Distribution:")
            print(f"    Safe: {true_safe} ({true_safe/len(texts):.1%})")
            print(f"    Unsafe: {true_unsafe} ({true_unsafe/len(texts):.1%})")
            print(f"\n  Prediction Distribution:")
            print(f"    Safe: {pred_safe} ({pred_safe/len(texts):.1%})")
            print(f"    Unsafe: {pred_unsafe} ({pred_unsafe/len(texts):.1%})")
            print(f"\n  Confusion Matrix:")
            print(f"    Safe → Safe: {cm[0][0]}")
            print(f"    Safe → Unsafe: {cm[0][1]} (False Positives)")
            print(f"    Unsafe → Safe: {cm[1][0]} (False Negatives)")
            print(f"    Unsafe → Unsafe: {cm[1][1]}")
            print(f"\n  Per-Class Metrics:")
            print(f"    Safe - Precision: {precision_per_class[0]:.2%}, Recall: {recall_per_class[0]:.2%}, F1: {f1_per_class[0]:.2%}")
            print(f"    Unsafe - Precision: {precision_per_class[1]:.2%}, Recall: {recall_per_class[1]:.2%}, F1: {f1_per_class[1]:.2%}")
        
        return results
        
    except Exception as e:
        if verbose:
            print(f"⚠️  Error evaluating {benchmark_name}: {e}")
        import traceback
        return {
            "benchmark": benchmark_name,
            "error": str(e),
            "traceback": traceback.format_exc()
        }


def run_all_benchmarks(
    limit: Optional[int] = None,
    threshold: float = 0.7,
    save_results: bool = True
) -> List[Dict[str, Any]]:
    """
    Run evaluation on all available benchmarks.
    
    Args:
        limit: Maximum number of examples per benchmark
        threshold: Confidence threshold for flagging
        save_results: Whether to save results to JSON file
        
    Returns:
        List of results dictionaries
    """
    print("="*70)
    print("ARMY OF SAFEGUARDS - BENCHMARK EVALUATION")
    print("="*70)
    print(f"\nEvaluating on {len(BENCHMARKS)} benchmarks")
    if limit:
        print(f"Limit: {limit} examples per benchmark")
    print(f"Threshold: {threshold}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    all_results = []
    
    for benchmark_name in BENCHMARKS.keys():
        result = evaluate_on_benchmark(
            benchmark_name,
            limit=limit,
            threshold=threshold,
            verbose=True
        )
        all_results.append(result)
    
    # Calculate overall statistics
    successful = [r for r in all_results if "error" not in r]
    failed = [r for r in all_results if "error" in r]
    
    print("\n" + "="*70)
    print("OVERALL SUMMARY")
    print("="*70)
    
    print(f"\nSuccessful evaluations: {len(successful)}/{len(BENCHMARKS)}")
    
    if successful:
        avg_accuracy = sum(r['accuracy'] for r in successful) / len(successful)
        avg_f1 = sum(r['f1_score'] for r in successful) / len(successful)
        avg_precision = sum(r['precision'] for r in successful) / len(successful)
        avg_recall = sum(r['recall'] for r in successful) / len(successful)
        
        print(f"\nAverage Metrics Across Benchmarks:")
        print(f"  Accuracy:  {avg_accuracy:.2%}")
        print(f"  Precision: {avg_precision:.2%}")
        print(f"  Recall:    {avg_recall:.2%}")
        print(f"  F1-Score:  {avg_f1:.2%}")
        
        print(f"\nPer-Benchmark Results:")
        print(f"{'Benchmark':<20} {'Accuracy':<12} {'F1-Score':<12} {'Precision':<12} {'Recall':<12}")
        print("-"*70)
        for result in successful:
            print(f"{result['benchmark']:<20} "
                  f"{result['accuracy']:<12.2%} "
                  f"{result['f1_score']:<12.2%} "
                  f"{result['precision']:<12.2%} "
                  f"{result['recall']:<12.2%}")
    
    if failed:
        print(f"\nFailed evaluations: {len(failed)}")
        for result in failed:
            print(f"  {result['benchmark']}: {result['error']}")
    
    # Save results
    if save_results and successful:
        output_dir = Path(__file__).parent
        output_file = output_dir / f"benchmark_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(output_file, 'w') as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "system": "Army of Safeguards (Aggregator)",
                "threshold": threshold,
                "limit_per_benchmark": limit,
                "results": all_results,
                "summary": {
                    "avg_accuracy": avg_accuracy,
                    "avg_precision": avg_precision,
                    "avg_recall": avg_recall,
                    "avg_f1_score": avg_f1
                } if successful else {}
            }, f, indent=2)
        print(f"\n✅ Results saved to: {output_file}")
    
    return all_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark the Army of Safeguards system against public benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run single benchmark
  python benchmark/run_benchmark.py --benchmark HarmBench --limit 100
  
  # Run all benchmarks
  python benchmark/run_benchmark.py --all --limit 100
  
  # Custom threshold
  python benchmark/run_benchmark.py --benchmark WildGuardMix --threshold 0.8
        """
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=list(BENCHMARKS.keys()),
        help="Name of the benchmark to run"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run all available benchmarks"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of examples per benchmark (default: all)"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for flagging (default: 0.7)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to JSON file"
    )
    
    args = parser.parse_args()
    
    if not args.benchmark and not args.all:
        parser.error("Must specify either --benchmark or --all")
    
    if args.benchmark and args.all:
        parser.error("Cannot specify both --benchmark and --all")
    
    if args.benchmark:
        # Run single benchmark
        result = evaluate_on_benchmark(
            args.benchmark,
            limit=args.limit,
            threshold=args.threshold,
            verbose=True
        )
        
        if not args.no_save and "error" not in result:
            output_dir = Path(__file__).parent
            output_file = output_dir / f"benchmark_{args.benchmark.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "timestamp": datetime.now().isoformat(),
                    "system": "Army of Safeguards (Aggregator)",
                    "threshold": args.threshold,
                    "result": result
                }, f, indent=2)
            print(f"\n✅ Results saved to: {output_file}")
    else:
        # Run all benchmarks
        run_all_benchmarks(
            limit=args.limit,
            threshold=args.threshold,
            save_results=not args.no_save
        )

