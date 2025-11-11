"""Basic test script for the sexual content safeguard."""
import sys
from pathlib import Path

# Add parent directory to path so we can import the safeguard
sys.path.insert(0, str(Path(__file__).parent.parent))

from safeguard_sexual import predict


def main():
    """Run basic tests."""
    print("\n" + "=" * 60)
    print("SEXUAL CONTENT SAFEGUARD - BASIC TESTS")
    print("=" * 60)
    
    # Test safe content
    print("\n1. Safe content:")
    safe_texts = [
        "The sky is blue.",
        "Water boils at 100 degrees Celsius.",
        "This is a normal conversation."
    ]
    for text in safe_texts:
        result = predict(text)
        label_type = "Sensitive" if result['label'] == 'LABEL_1' else "Safe"
        print(f"   '{text}' → {label_type} (confidence: {result['confidence']:.2%})")
    
    # Test sensitive content
    print("\n2. Sensitive content:")
    sensitive_texts = [
        "This is so fucking cool!",
        "That's a damn good idea."
    ]
    for text in sensitive_texts:
        result = predict(text)
        label_type = "Sensitive" if result['label'] == 'LABEL_1' else "Safe"
        print(f"   '{text}' → {label_type} (confidence: {result['confidence']:.2%})")
    
    # Test edge cases
    print("\n3. Edge cases:")
    # Empty string
    try:
        predict("")
        print("   ERROR: Empty string should raise ValueError!")
    except ValueError:
        print("   ✓ Empty string correctly raises ValueError")
    
    # Long text
    long_text = "This is a test. " * 50
    result = predict(long_text)
    print(f"   Long text → {result['label']} (truncated to 128 tokens)")
    
    print("\n" + "=" * 60)
    print("✅ Basic tests completed!")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()

