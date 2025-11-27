"""
Aggregator for Army of Safeguards.

This module imports all individual safeguard critics and provides
a unified interface for running multiple safeguards on input text.
"""
import sys
from pathlib import Path
from typing import Dict, List, Any

# Add parent directory to path so we can import safeguards
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import KNN aggregator
from knn_aggregator import KNNAggregator

# Create global KNN aggregator instance (will be fitted later)
knn_aggregator = KNNAggregator(k=7)  # You can experiment with k=5,7,9,11


def run_all_safeguards(text: str) -> Dict[str, Any]:
    """
    Run all available safeguards on the input text.

    Args:
        text: Input text to evaluate

    Returns:
        Dictionary containing results from all safeguards
    """
    results = {}

    # Import and run factuality safeguard
    try:
        from factuality.safeguard_factuality import predict as fact_check
        results['factuality'] = fact_check(text)
    except ImportError:
        results['factuality'] = {"error": "Factuality safeguard not available"}

    # Import and run toxicity safeguard
    try:
        from toxicity.safeguard_toxicity import predict as toxic_check
        results['toxicity'] = toxic_check(text)
    except ImportError:
        results['toxicity'] = {"error": "Toxicity safeguard not available"}
    
    # Import and run sexual content safeguard
    try:
        from sexual.safeguard_sexual import predict as sex_check
        results['sexual'] = sex_check(text)
    except ImportError:
        results['sexual'] = {"error": "Sexual content safeguard not available"}

    # Import and run jailbreak content safeguard
    try:
        from jailbreak.safeguard_jailbreak import predict as jailbreak_check
        results['jailbreak'] = jailbreak_check(text)
    except ImportError:
        results['jailbreak'] = {"error": "Jailbreak safeguard not available"}

    return results


def load_knn_reference_data(jsonl_path: str = "knn_reference_2000.jsonl"):
    """
    Load reference data for KNN aggregator (only need to run once!).
    
    This function loads all available data from the JSONL file, even if processing
    was incomplete. It gracefully handles malformed lines and uses all valid data.
    
    JSONL format (each line):
    {"text": "...", "conf_fact":0.93, "conf_tox":0.02, "conf_sex":0.88, "conf_jb":0.01, "is_safe": false}
    
    Args:
        jsonl_path: Path to the JSONL file containing reference data
    """
    import json
    from pathlib import Path
    
    jsonl_path = Path(jsonl_path)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Reference data file not found: {jsonl_path}")
    
    data = []
    skipped_lines = 0
    
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # Skip empty lines
                continue
            
            try:
                item = json.loads(line)
                # Validate required fields
                if "conf_fact" not in item or "conf_sex" not in item or "conf_jb" not in item:
                    skipped_lines += 1
                    continue
                
                confs = [
                    float(item["conf_fact"]),
                    float(item.get("conf_tox", 0.0)),     # toxicity may not exist, use 0.0
                    float(item["conf_sex"]),
                    float(item["conf_jb"])
                ]
                
                # Validate is_safe field
                is_safe = item.get("is_safe", True)
                if not isinstance(is_safe, bool):
                    # Try to convert if it's a string
                    if isinstance(is_safe, str):
                        is_safe = is_safe.lower() in ("true", "1", "yes")
                    else:
                        is_safe = bool(is_safe)
                
                data.append({
                    'confidences': confs,
                    'is_safe': is_safe
                })
            except (json.JSONDecodeError, KeyError, ValueError, TypeError) as e:
                skipped_lines += 1
                if skipped_lines <= 5:  # Only show first few errors
                    print(f"[KNN] Warning: Skipping malformed line {line_num}: {e}")
                continue
    
    if not data:
        raise ValueError(f"No valid reference data found in {jsonl_path}. File may be empty or corrupted.")
    
    if skipped_lines > 0:
        print(f"[KNN] Warning: Skipped {skipped_lines} malformed lines, using {len(data)} valid samples")
    
    knn_aggregator.fit(data)
    print(f"[KNN] Loaded {len(data)} reference samples")


def aggregate_results(
    results: Dict[str, Dict[str, Any]], 
    threshold: float = 0.7,
    use_knn: bool = True  # New switch!!!
) -> Dict[str, Any]:
    """
    Aggregate results from multiple safeguards.

    Args:
        results: Dictionary of results from each safeguard
        threshold: Confidence threshold for flagging (default: 0.7)
        use_knn: Whether to use KNN aggregation (default: True)

    Returns:
        Aggregated decision with overall safety assessment
    """
    flags = []
    confidences = []

    for safeguard_name, result in results.items():
        if 'error' in result:
            continue

        confidence = result.get('confidence', 0.0)
        label = result.get('label', '')

        # Check if this safeguard flagged the content
        # For factuality: LABEL_1 = non-factual (unsafe), LABEL_0 = factual (safe)
        # For sexual: LABEL_1 = sensitive/sexual content (unsafe), LABEL_0 = safe
        # For jailbreak: label is boolean (True = jailbreak/unsafe, False = safe)
        is_problematic = False

        if safeguard_name == 'factuality':
            # LABEL_1 means non-factual/misinformation
            is_problematic = (label == 'LABEL_1' and confidence >= threshold)
        elif safeguard_name == 'sexual':
            # LABEL_1 means sensitive/sexual content
            is_problematic = (label == 'LABEL_1' and confidence >= threshold)
        elif safeguard_name == 'jailbreak':
            # Boolean label: True = jailbreak (unsafe)
            is_problematic = (label is True and confidence >= threshold)
        elif safeguard_name == 'toxicity':
            # "unsafe" or LABEL_1 means toxic/racist/hateful
            is_problematic = ((label == 'unsafe' or label ==
                              'LABEL_1') and confidence >= threshold)
        else:
            # For other safeguards, assume higher label numbers = problematic
            # Adjust this logic when adding new safeguards
            is_problematic = (confidence >= threshold and label not in [
                              'LABEL_0', 'safe'])

        if is_problematic:
            flags.append({
                'safeguard': safeguard_name,
                'label': label,
                'confidence': confidence
            })

        confidences.append(confidence)

    # Extract four confidences in fixed order: factuality, toxicity, sexual, jailbreak
    conf_vector = [
        results.get('factuality', {}).get('confidence', 0.0),
        results.get('toxicity', {}).get('confidence', 0.0),
        results.get('sexual', {}).get('confidence', 0.0),
        results.get('jailbreak', {}).get('confidence', 0.0),
    ]

    if use_knn and knn_aggregator.is_fitted:
        knn_result = knn_aggregator.predict(conf_vector)
        return {
            'is_safe': knn_result['is_safe'],
            'flags': flags,  # Still keep which models triggered
            'average_confidence': sum(conf_vector) / len(conf_vector),
            'individual_results': results,
            'aggregation_method': 'knn',
            'knn_detail': knn_result
        }
    else:
        # Original majority vote logic (flag if >= 2 safeguards trigger)
        is_safe = len(flags) < 2  # Original logic: >= 2 flags means unsafe
        return {
            'is_safe': is_safe,
            'flags': flags,
            'average_confidence': sum(conf_vector) / 4,
            'individual_results': results,
            'aggregation_method': 'majority_vote_2'
        }


def evaluate_text(text: str, threshold: float = 0.7, use_knn: bool = True) -> Dict[str, Any]:
    """
    Convenience function to run all safeguards and aggregate results.

    Args:
        text: Input text to evaluate
        threshold: Confidence threshold for flagging
        use_knn: Whether to use KNN aggregation (default: True)

    Returns:
        Aggregated safety assessment
    """
    results = run_all_safeguards(text)
    return aggregate_results(results, threshold, use_knn)


if __name__ == "__main__":
    # Example usage
    import sys

    # You can load KNN reference data here (only need to run once)
    # load_knn_reference_data("data/knn_reference_2000.jsonl")

    if len(sys.argv) > 1:
        test_text = " ".join(sys.argv[1:])
    else:
        test_text = input("Enter text to evaluate: ")

    print("\nRunning all safeguards...")
    print("=" * 60)

    result = evaluate_text(test_text)

    print(
        f"\nOverall Safety: {'✅ SAFE' if result['is_safe'] else '⚠️  FLAGGED'}")
    print(f"Average Confidence: {result['average_confidence']:.2%}")

    if result['flags']:
        print(f"\nFlags ({len(result['flags'])}):")
        for flag in result['flags']:
            print(
                f"  - {flag['safeguard']}: {flag['label']} (confidence: {flag['confidence']:.2%})")

    print("\nIndividual Results:")
    for safeguard, res in result['individual_results'].items():
        if 'error' not in res:
            print(
                f"  {safeguard}: {res.get('label')} (confidence: {res.get('confidence', 0):.2%})")
        else:
            print(f"  {safeguard}: {res['error']}")

    print("=" * 60)
