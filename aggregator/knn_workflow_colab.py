#!/usr/bin/env python3
"""KNN Aggregator Workflow for Google Colab.

This script is optimized for running in Google Colab environment.
It handles Colab-specific setup like mounting Google Drive and installing dependencies.

Usage in Colab:
    1. Upload this file and the project to Colab
    2. Run: !python aggregator/knn_workflow_colab.py
"""
import sys
import os
import subprocess
from pathlib import Path

# Colab-specific setup
def setup_colab_environment():
    """Setup Colab environment: install dependencies, mount drive, etc."""
    print("="*60)
    print("Setting up Colab environment...")
    print("="*60)
    
    # Check if running in Colab
    try:
        import google.colab
        IN_COLAB = True
        print("✅ Running in Google Colab")
    except ImportError:
        IN_COLAB = False
        print("⚠️  Not running in Colab, skipping Colab-specific setup")
        return IN_COLAB
    
    # Mount Google Drive (optional, for persistent storage)
    try:
        from google.colab import drive
        if not os.path.exists('/content/drive'):
            print("\nMounting Google Drive...")
            drive.mount('/content/drive')
            print("✅ Google Drive mounted")
        else:
            print("✅ Google Drive already mounted")
    except Exception as e:
        print(f"⚠️  Could not mount Google Drive: {e}")
        print("   Continuing without Drive mount...")
    
    # Install/upgrade required packages
    print("\nInstalling/upgrading packages...")
    packages = [
        "transformers>=4.44",
        "torch",
        "scikit-learn",
        "datasets==3.6.0",
        "huggingface_hub",
        "safetensors",
        "tqdm",
        "pandas",
        "numpy",
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print(f"  ✅ {package}")
        except Exception as e:
            print(f"  ⚠️  {package}: {e}")
    
    print("\n✅ Environment setup complete!")
    return IN_COLAB


def clone_or_setup_repo():
    """Clone repository or setup from uploaded files."""
    print("\n" + "="*60)
    print("Setting up project files...")
    print("="*60)
    
    # Check if we're in the project directory
    if Path("aggregator").exists() and Path("aggregator/knn_workflow.py").exists():
        print("✅ Project files already present")
        return True
    
    # Try to clone from git
    repo_url = "https://github.com/SohamNagi/ArmyOfSafeguards.git"  # Update with actual repo URL
    print(f"\nAttempting to clone repository...")
    try:
        subprocess.check_call(["git", "clone", repo_url, "."], 
                            stdout=subprocess.DEVNULL, 
                            stderr=subprocess.DEVNULL)
        print("✅ Repository cloned successfully")
        return True
    except Exception as e:
        print(f"⚠️  Could not clone repository: {e}")
        print("\n📝 Alternative: Upload project files manually:")
        print("   1. Upload the entire project folder to Colab")
        print("   2. Or upload individual files needed for KNN workflow")
        return False


def run_knn_workflow(limit=100, threshold=0.7, skip_generation=False):
    """Run the KNN workflow."""
    print("\n" + "="*60)
    print("Starting KNN Workflow")
    print("="*60)
    
    # Import after setup
    sys.path.insert(0, str(Path.cwd()))
    sys.path.insert(0, str(Path("aggregator")))
    
    try:
        # Import workflow components
        from aggregator.generate_knn_reference_hh_rlhf_full import *
        from aggregator.evaluate_aggregator import compare_methods
        
        # Step 1: Generate reference data (if needed)
        if not skip_generation:
            print("\n" + "="*60)
            print("Step 1: Generating KNN Reference Data")
            print("="*60)
            
            # This will run the generation script
            output_file = Path("aggregator/knn_reference_hh_rlhf_full.jsonl")
            
            # Check if already exists
            if output_file.exists():
                print(f"⚠️  Reference data already exists: {output_file}")
                response = input("Regenerate? (y/N): ").strip().lower()
                if response != 'y':
                    skip_generation = True
        
        if not skip_generation:
            # Run generation (imported functions will be executed)
            print("Generating reference data...")
            # The generation script logic would go here
            # For now, we'll call the actual script
            subprocess.run([sys.executable, "aggregator/generate_knn_reference_hh_rlhf_full.py"])
        
        # Step 2: Evaluate
        print("\n" + "="*60)
        print("Step 2: Evaluating Performance")
        print("="*60)
        
        reference_file = "aggregator/knn_reference_hh_rlhf_full.jsonl"
        results = compare_methods(
            dataset_name="hh-rlhf",
            limit=limit,
            knn_reference_path=reference_file,
            threshold=threshold,
        )
        
        # Save results
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"aggregator/evaluation_results_colab_{timestamp}.json"
        
        import json
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)
        
        print(f"\n✅ Results saved to: {output_file}")
        
        # Download results
        try:
            from google.colab import files
            print("\n📥 Downloading results...")
            files.download(output_file)
            print("✅ Results downloaded!")
        except:
            print("⚠️  Could not auto-download. File saved to:", output_file)
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error running workflow: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point for Colab."""
    print("\n" + "="*60)
    print("KNN Aggregator Workflow - Google Colab Version")
    print("="*60)
    
    # Setup environment
    in_colab = setup_colab_environment()
    
    # Setup project files
    if not clone_or_setup_repo():
        print("\n⚠️  Please ensure project files are available")
        print("   You can:")
        print("   1. Upload the project folder to Colab")
        print("   2. Or clone the repository manually")
        return
    
    # Run workflow
    # You can customize these parameters
    run_knn_workflow(
        limit=100,           # Number of examples to evaluate
        threshold=0.7,       # Confidence threshold
        skip_generation=False  # Set to True if reference data exists
    )
    
    print("\n" + "="*60)
    print("✅ Workflow Complete!")
    print("="*60)


if __name__ == "__main__":
    main()

