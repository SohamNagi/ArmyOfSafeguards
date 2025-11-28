import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from toxicity.safeguard_toxicity import predict

text = "I really hate people who look like that, they are disgusting and nasty."
print(f"Testing text: '{text}'")

try:
    result = predict(text)
    print(f"Result: {result}")
except Exception as e:
    print(f"Error: {e}")
