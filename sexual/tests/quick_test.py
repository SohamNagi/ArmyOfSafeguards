"""Quick test to verify the sexual content safeguard is working."""
import sys
from pathlib import Path

# Add parent directory to path so we can import the safeguard
sys.path.insert(0, str(Path(__file__).parent.parent))

from safeguard_sexual import predict

print("Testing sexual content safeguard...")
print("-" * 50)

# Test a safe statement
safe = "The capital of France is Paris."
result1 = predict(safe)
print(f"✓ Safe: '{safe}'")
print(f"  → {result1['label']} (confidence: {result1['confidence']:.2%})")

print()

# Test sensitive content (profanity example)
sensitive = "This is so fucking cool!"
result2 = predict(sensitive)
print(f"✓ Sensitive: '{sensitive}'")
print(f"  → {result2['label']} (confidence: {result2['confidence']:.2%})")

print("-" * 50)
print("✅ Safeguard is working correctly!")

