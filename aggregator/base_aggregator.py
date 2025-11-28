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


def aggregate_results(results: Dict[str, Dict[str, Any]], threshold: float = 0.7) -> Dict[str, Any]:
    """
    Aggregate results from multiple safeguards.

    Args:
        results: Dictionary of results from each safeguard
        threshold: Confidence threshold for flagging (default: 0.7)

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

    # Calculate overall safety score
    avg_confidence = sum(confidences) / \
        len(confidences) if confidences else 0.0
    is_safe = len(flags) == 0

    return {
        'is_safe': is_safe,
        'flags': flags,
        'average_confidence': avg_confidence,
        'individual_results': results
    }


def evaluate_text(text: str, threshold: float = 0.7) -> Dict[str, Any]:
    """
    Convenience function to run all safeguards and aggregate results.

    Args:
        text: Input text to evaluate
        threshold: Confidence threshold for flagging

    Returns:
        Aggregated safety assessment
    """
    results = run_all_safeguards(text)
    return aggregate_results(results, threshold)


if __name__ == "__main__":
    # Example usage
    import sys

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
