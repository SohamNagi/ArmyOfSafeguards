import sys
import math
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from aggregator.aggregator import aggregate_results

def test_nan_handling():
    print("Testing NaN handling in aggregator...")
    
    # Simulate results with NaN confidence
    mock_results = {
        'factuality': {'label': 'LABEL_1', 'confidence': float('nan')},
        'toxicity': {'label': 'safe', 'confidence': 0.99},
        'sexual': {'label': 'LABEL_0', 'confidence': 0.8},
        'jailbreak': {'label': False, 'confidence': 0.95}
    }
    
    aggregated = aggregate_results(mock_results)
    
    print(f"Aggregated results: {aggregated}")
    
    # Check average confidence
    avg_conf = aggregated['average_confidence']
    if math.isnan(avg_conf):
        print("FAIL: Average confidence is NaN")
        sys.exit(1)
    else:
        print(f"PASS: Average confidence is {avg_conf}")
        
    # Check individual results sanitization
    if math.isnan(aggregated['individual_results']['factuality']['confidence']):
        print("FAIL: Individual result confidence is NaN")
        sys.exit(1)
    else:
        print(f"PASS: Individual result confidence is {aggregated['individual_results']['factuality']['confidence']}")

    print("All tests passed!")

if __name__ == "__main__":
    test_nan_handling()
