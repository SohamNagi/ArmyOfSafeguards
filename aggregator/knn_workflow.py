#!/usr/bin/env python3
"""KNN Aggregator Complete Workflow Script.

This script automates the complete workflow for KNN aggregator:
1. Generate KNN reference data from HH-RLHF dataset
2. Evaluate aggregator performance (KNN vs Majority Vote)
3. Display comparison results

Usage:
    python aggregator/knn_workflow.py
    python aggregator/knn_workflow.py --skip-generation  # Skip step 1 if data already exists
    python aggregator/knn_workflow.py --limit 200 --threshold 0.7
"""
import sys
import subprocess
import argparse
from pathlib import Path
from typing import Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))


def run_command(cmd: list, description: str, check: bool = True) -> bool:
    """Run a command and handle errors."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print()
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=False)
        return result.returncode == 0
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error: {description} failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print(f"\n❌ Error: Command not found. Make sure Python is in your PATH.")
        return False


def check_reference_data_exists(output_file: Path) -> bool:
    """Check if KNN reference data already exists."""
    return output_file.exists() and output_file.stat().st_size > 0


def generate_reference_data(
    output_file: Optional[str] = None,
    force: bool = False
) -> bool:
    """Generate KNN reference data from HH-RLHF dataset."""
    if output_file is None:
        output_file = Path(__file__).parent / "knn_reference_hh_rlhf_full.jsonl"
    else:
        output_file = Path(output_file)
    
    # Check if file already exists
    if check_reference_data_exists(output_file) and not force:
        print(f"\n⚠️  Reference data already exists: {output_file}")
        response = input("Do you want to regenerate it? (y/N): ").strip().lower()
        if response != 'y':
            print("Skipping reference data generation.")
            return True
    
    # Run generation script
    script_path = Path(__file__).parent / "generate_knn_reference_hh_rlhf_full.py"
    cmd = [sys.executable, str(script_path)]
    
    success = run_command(
        cmd,
        "Step 1: Generating KNN Reference Data from HH-RLHF",
        check=False
    )
    
    if success:
        if check_reference_data_exists(output_file):
            print(f"\n✅ Reference data generated successfully: {output_file}")
            return True
        else:
            print(f"\n⚠️  Warning: Reference data file not found at expected location")
            return False
    else:
        print(f"\n❌ Failed to generate reference data")
        return False


def evaluate_performance(
    reference_file: Optional[str] = None,
    limit: int = 100,
    threshold: float = 0.7,
    compare: bool = True,
) -> bool:
    """Evaluate aggregator performance with KNN vs Majority Vote."""
    if reference_file is None:
        reference_file = Path(__file__).parent / "knn_reference_hh_rlhf_full.jsonl"
    else:
        reference_file = Path(reference_file)
    
    # Check if reference file exists
    if not reference_file.exists():
        print(f"\n❌ Error: Reference data file not found: {reference_file}")
        print("Please run reference data generation first (skip with --skip-generation)")
        return False
    
    # Build evaluation command
    script_path = Path(__file__).parent / "evaluate_aggregator.py"
    cmd = [
        sys.executable,
        str(script_path),
        "--dataset", "hh-rlhf",
        "--limit", str(limit),
        "--knn-reference", str(reference_file),
        "--threshold", str(threshold),
    ]
    
    if compare:
        cmd.append("--compare")
    
    description = "Step 2: Evaluating Aggregator Performance"
    if compare:
        description += " (KNN vs Majority Vote Comparison)"
    
    success = run_command(
        cmd,
        description,
        check=False
    )
    
    return success


def main():
    parser = argparse.ArgumentParser(
        description="KNN Aggregator Complete Workflow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full workflow (generate + evaluate)
  python aggregator/knn_workflow.py
  
  # Skip generation if data exists
  python aggregator/knn_workflow.py --skip-generation
  
  # Custom parameters
  python aggregator/knn_workflow.py --limit 200 --threshold 0.8
  
  # Only evaluate (no comparison)
  python aggregator/knn_workflow.py --skip-generation --no-compare
        """
    )
    
    parser.add_argument(
        "--skip-generation",
        action="store_true",
        help="Skip reference data generation (use existing file)",
    )
    parser.add_argument(
        "--force-generation",
        action="store_true",
        help="Force regeneration of reference data even if it exists",
    )
    parser.add_argument(
        "--reference-file",
        type=str,
        default=None,
        help="Path to KNN reference data file (default: aggregator/knn_reference_hh_rlhf_full.jsonl)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Number of examples to evaluate (default: 100)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Confidence threshold for flagging (default: 0.7)",
    )
    parser.add_argument(
        "--no-compare",
        action="store_true",
        help="Don't compare KNN vs Majority Vote (only test KNN)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save results (default: aggregator/)",
    )
    
    args = parser.parse_args()
    
    print("\n" + "="*60)
    print("KNN Aggregator Workflow")
    print("="*60)
    print(f"Configuration:")
    print(f"  Skip generation: {args.skip_generation}")
    print(f"  Force generation: {args.force_generation}")
    print(f"  Evaluation limit: {args.limit}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Compare methods: {not args.no_compare}")
    print("="*60)
    
    # Step 1: Generate reference data
    if not args.skip_generation:
        success = generate_reference_data(
            output_file=args.reference_file,
            force=args.force_generation
        )
        if not success:
            print("\n❌ Workflow failed at reference data generation step")
            print("You can skip this step with --skip-generation if data already exists")
            sys.exit(1)
    else:
        print("\n⏭️  Skipping reference data generation (--skip-generation)")
        if args.reference_file:
            ref_file = Path(args.reference_file)
        else:
            ref_file = Path(__file__).parent / "knn_reference_hh_rlhf_full.jsonl"
        
        if not ref_file.exists():
            print(f"⚠️  Warning: Reference file not found: {ref_file}")
            print("Consider running without --skip-generation to generate it")
    
    # Step 2: Evaluate performance
    success = evaluate_performance(
        reference_file=args.reference_file,
        limit=args.limit,
        threshold=args.threshold,
        compare=not args.no_compare,
    )
    
    if success:
        print("\n" + "="*60)
        print("✅ Workflow completed successfully!")
        print("="*60)
        print("\nNext steps:")
        print("1. Check the evaluation results in the output JSON file")
        print("2. Review the metrics to see KNN performance improvements")
        print("3. Adjust k value or threshold if needed")
        print("\nTo adjust KNN k value, edit aggregator/aggregator.py:")
        print("  knn_aggregator = KNNAggregator(k=7)  # Try k=5,7,9,11")
    else:
        print("\n" + "="*60)
        print("❌ Workflow failed at evaluation step")
        print("="*60)
        sys.exit(1)


if __name__ == "__main__":
    main()

