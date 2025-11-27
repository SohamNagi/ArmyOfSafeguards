"""Evaluate aggregator performance with KNN vs Majority Vote on benchmark datasets.

Usage:
    python aggregator/evaluate_aggregator.py --dataset hh-rlhf --limit 100
    python aggregator/evaluate_aggregator.py --dataset hh-rlhf --limit 100 --knn-reference knn_reference_hh_rlhf_full.jsonl
"""
import sys
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any, Tuple
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from aggregator import evaluate_text, load_knn_reference_data
from datasets import load_dataset
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)

# Benchmark datasets configuration
BENCHMARKS = {
    "hh-rlhf": {
        "dataset": "Anthropic/hh-rlhf",
        "data_dir": "harmless-base",
        "split": "test",  # Use test split if available, otherwise train
        "text_field": "chosen",  # or "rejected"
        "label_field": "chosen",  # We'll use chosen=True (safe), rejected=False (unsafe)
        "description": "Anthropic HH-RLHF harmless-base dataset",
    },
}


def map_hh_rlhf_to_safe_unsafe(item: Dict, use_chosen: bool = True) -> Tuple[str, bool]:
    """Map HH-RLHF item to (text, is_safe) tuple.
    
    Args:
        item: Dataset item with 'chosen' and 'rejected' fields
        use_chosen: If True, use 'chosen' text (safe), else use 'rejected' (unsafe)
    
    Returns:
        Tuple of (text, is_safe)
    """
    if use_chosen:
        return item["chosen"], True
    else:
        return item["rejected"], False


def evaluate_aggregator(
    dataset_name: str,
    limit: int = 100,
    knn_reference_path: str = None,
    use_knn: bool = True,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Evaluate aggregator performance on a benchmark dataset.
    
    Args:
        dataset_name: Name of the benchmark dataset
        limit: Maximum number of examples to evaluate
        knn_reference_path: Path to KNN reference data JSONL file
        use_knn: Whether to use KNN aggregation (True) or majority vote (False)
        threshold: Confidence threshold for flagging
    
    Returns:
        Dictionary with evaluation metrics
    """
    if dataset_name not in BENCHMARKS:
        raise ValueError(f"Unknown dataset: {dataset_name}. Available: {list(BENCHMARKS.keys())}")
    
    config = BENCHMARKS[dataset_name]
    
    # Load KNN reference data if provided
    if use_knn and knn_reference_path:
        print(f"\nLoading KNN reference data from: {knn_reference_path}")
        if not Path(knn_reference_path).exists():
            # Try relative to aggregator directory
            knn_reference_path = Path(__file__).parent / knn_reference_path
        load_knn_reference_data(str(knn_reference_path))
        print("KNN reference data loaded successfully!")
    
    # Load dataset
    print(f"\nLoading dataset: {config['dataset']}")
    try:
        if "data_dir" in config:
            dataset = load_dataset(config["dataset"], data_dir=config["data_dir"], split=config.get("split", "train"))
        else:
            dataset = load_dataset(config["dataset"], split=config.get("split", "train"))
    except Exception as e:
        print(f"Warning: Could not load test split, trying train split: {e}")
        if "data_dir" in config:
            dataset = load_dataset(config["dataset"], data_dir=config["data_dir"], split="train")
        else:
            dataset = load_dataset(config["dataset"], split="train")
    
    # Limit dataset size
    if limit and limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))
    
    print(f"Evaluating on {len(dataset)} examples...")
    print(f"Aggregation method: {'KNN' if use_knn else 'Majority Vote'}")
    print(f"Threshold: {threshold}")
    
    predictions = []
    ground_truth = []
    confidences = []
    aggregation_methods = []
    
    # Process dataset
    for i, item in enumerate(tqdm(dataset, desc="Evaluating")):
        # For HH-RLHF, we need to handle both chosen and rejected
        if dataset_name == "hh-rlhf":
            # Use chosen (safe) and rejected (unsafe) as separate examples
            # Process chosen (safe)
            text_chosen, is_safe_chosen = map_hh_rlhf_to_safe_unsafe(item, use_chosen=True)
            result_chosen = evaluate_text(text_chosen, threshold=threshold, use_knn=use_knn)
            predictions.append(result_chosen['is_safe'])
            ground_truth.append(is_safe_chosen)
            confidences.append(result_chosen['average_confidence'])
            aggregation_methods.append(result_chosen.get('aggregation_method', 'unknown'))
            
            # Process rejected (unsafe) - but limit total examples
            if len(predictions) < limit * 2:
                text_rejected, is_safe_rejected = map_hh_rlhf_to_safe_unsafe(item, use_chosen=False)
                result_rejected = evaluate_text(text_rejected, threshold=threshold, use_knn=use_knn)
                predictions.append(result_rejected['is_safe'])
                ground_truth.append(is_safe_rejected)
                confidences.append(result_rejected['average_confidence'])
                aggregation_methods.append(result_rejected.get('aggregation_method', 'unknown'))
        else:
            # For other datasets, use the configured text field
            text = item.get(config["text_field"], str(item))
            # Assume we need to determine is_safe from the dataset
            # This is dataset-specific and may need adjustment
            is_safe = True  # Default - adjust based on dataset
            result = evaluate_text(text, threshold=threshold, use_knn=use_knn)
            predictions.append(result['is_safe'])
            ground_truth.append(is_safe)
            confidences.append(result['average_confidence'])
            aggregation_methods.append(result.get('aggregation_method', 'unknown'))
        
        if limit and len(predictions) >= limit:
            break
    
    if not predictions:
        return {"error": "No predictions generated"}
    
    # Calculate metrics
    print(f"\nCalculating metrics...")
    
    accuracy = accuracy_score(ground_truth, predictions)
    precision, recall, f1, support = precision_recall_fscore_support(
        ground_truth, predictions, average='binary', pos_label=True, zero_division=0
    )
    
    # Confusion matrix
    cm = confusion_matrix(ground_truth, predictions, labels=[True, False])
    
    # Per-class metrics
    class_report = classification_report(
        ground_truth, predictions, 
        target_names=["Safe", "Unsafe"],
        output_dict=True,
        zero_division=0
    )
    
    avg_confidence = sum(confidences) / len(confidences)
    method_used = aggregation_methods[0] if aggregation_methods else 'unknown'
    
    results = {
        "dataset": dataset_name,
        "aggregation_method": method_used,
        "threshold": threshold,
        "total_examples": len(predictions),
        "metrics": {
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1),
            "avg_confidence": float(avg_confidence),
        },
        "confusion_matrix": {
            "true_positive": int(cm[0][0]),
            "false_positive": int(cm[0][1]),
            "false_negative": int(cm[1][0]),
            "true_negative": int(cm[1][1]),
        },
        "per_class": class_report,
        "timestamp": datetime.now().isoformat(),
    }
    
    # Display results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS - {dataset_name}")
    print(f"{'='*60}")
    print(f"Aggregation Method: {method_used}")
    print(f"Total Examples: {len(predictions)}")
    print(f"\nOverall Metrics:")
    print(f"  Accuracy:   {accuracy:.2%}")
    print(f"  Precision:  {precision:.2%}")
    print(f"  Recall:     {recall:.2%}")
    print(f"  F1-Score:   {f1:.2%}")
    print(f"  Avg Confidence: {avg_confidence:.2%}")
    
    print(f"\nConfusion Matrix:")
    print(f"                Predicted")
    print(f"              Safe  Unsafe")
    print(f"  Actual Safe  {cm[0][0]:4d}  {cm[0][1]:4d}")
    print(f"        Unsafe {cm[1][0]:4d}  {cm[1][1]:4d}")
    
    return results


def compare_methods(
    dataset_name: str,
    limit: int = 100,
    knn_reference_path: str = None,
    threshold: float = 0.7,
) -> Dict[str, Any]:
    """
    Compare KNN aggregation vs Majority Vote on the same dataset.
    
    Returns:
        Dictionary with comparison results
    """
    print(f"\n{'='*60}")
    print(f"COMPARING AGGREGATION METHODS")
    print(f"{'='*60}")
    
    # Evaluate with Majority Vote (no KNN reference needed)
    print("\n[1/2] Evaluating with Majority Vote...")
    results_mv = evaluate_aggregator(
        dataset_name=dataset_name,
        limit=limit,
        knn_reference_path=None,  # Not needed for majority vote
        use_knn=False,
        threshold=threshold,
    )
    
    # Evaluate with KNN
    print("\n[2/2] Evaluating with KNN...")
    results_knn = evaluate_aggregator(
        dataset_name=dataset_name,
        limit=limit,
        knn_reference_path=knn_reference_path,
        use_knn=True,
        threshold=threshold,
    )
    
    # Compare results
    print(f"\n{'='*60}")
    print(f"COMPARISON RESULTS")
    print(f"{'='*60}")
    
    comparison = {
        "dataset": dataset_name,
        "threshold": threshold,
        "majority_vote": results_mv.get("metrics", {}),
        "knn": results_knn.get("metrics", {}),
        "improvement": {},
    }
    
    # Calculate improvements
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        mv_val = results_mv.get("metrics", {}).get(metric, 0)
        knn_val = results_knn.get("metrics", {}).get(metric, 0)
        improvement = knn_val - mv_val
        improvement_pct = (improvement / mv_val * 100) if mv_val > 0 else 0
        comparison["improvement"][metric] = {
            "absolute": float(improvement),
            "percentage": float(improvement_pct),
        }
    
    # Display comparison
    print(f"\nMetric Comparison:")
    print(f"{'Metric':<15} {'Majority Vote':<15} {'KNN':<15} {'Improvement':<15}")
    print(f"{'-'*60}")
    for metric in ["accuracy", "precision", "recall", "f1_score"]:
        mv_val = results_mv.get("metrics", {}).get(metric, 0)
        knn_val = results_knn.get("metrics", {}).get(metric, 0)
        improvement = comparison["improvement"][metric]["absolute"]
        improvement_pct = comparison["improvement"][metric]["percentage"]
        print(f"{metric.capitalize():<15} {mv_val:>14.2%} {knn_val:>14.2%} {improvement:>+14.2%} ({improvement_pct:+.1f}%)")
    
    return comparison


def main():
    parser = argparse.ArgumentParser(description="Evaluate aggregator performance")
    parser.add_argument(
        "--dataset",
        type=str,
        default="hh-rlhf",
        choices=list(BENCHMARKS.keys()),
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--knn-reference",
        type=str,
        default=None,
        help="Path to KNN reference data JSONL file",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare KNN vs Majority Vote",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for flagging",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save results JSON file",
    )
    
    args = parser.parse_args()
    
    if args.compare:
        results = compare_methods(
            dataset_name=args.dataset,
            limit=args.limit,
            knn_reference_path=args.knn_reference,
            threshold=args.threshold,
        )
    else:
        results = evaluate_aggregator(
            dataset_name=args.dataset,
            limit=args.limit,
            knn_reference_path=args.knn_reference,
            use_knn=True,
            threshold=args.threshold,
        )
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")
    else:
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        method = "compare" if args.compare else "knn" if args.knn_reference else "mv"
        output_path = Path(__file__).parent / f"evaluation_results_{method}_{timestamp}.json"
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"\nResults saved to: {output_path}")


if __name__ == "__main__":
    main()

