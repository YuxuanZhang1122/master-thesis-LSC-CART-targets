"""
Van Galen Random Forest Classifiers

Usage:
    python run_pipeline.py        # Run all steps
    python run_pipeline.py 2      # Run step 2 only
"""

import sys
import os
import step1_prepare_data
import step2_classifier1
import step3_prepare_malignant
import step4_classifier2


STEPS = [
    (1, "Data Preparation", step1_prepare_data),
    (2, "Classifier 1", step2_classifier1),
    (3, "Malignant Cell Preparation", step3_prepare_malignant),
    (4, "Classifier 2", step4_classifier2)
]

# Output directory
RESULTS_DIR = "results"

# Intermediate files to clean up after pipeline completes
INTERMEDIATE_FILES = [
    'results/data_processed_normal.h5ad',
    'results/training_data_21class.h5ad',
    'results/classifier1_final_model.pkl'
]


def setup_results_directory():
    """Create results directory if it doesn't exist"""
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)
        print(f"Created {RESULTS_DIR}/ directory\n")


def cleanup_intermediate_files():
    """Remove intermediate data files, keep final outputs (PNG, CSV)"""
    print("\nCleaning up intermediate files...")
    removed_count = 0
    for filename in INTERMEDIATE_FILES:
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"  Removed: {filename}")
                removed_count += 1
            except Exception as e:
                print(f"  Warning: Could not remove {filename}: {e}")

    if removed_count == 0:
        print("  No intermediate files to remove")
    else:
        print(f"  Removed {removed_count} intermediate file(s)")


def run_step(step_num, step_name, module):
    print(f"\n{'='*60}")
    print(f"STEP {step_num}: {step_name}")
    print(f"{'='*60}\n")
    module.main()
    return True


def main():
    # Setup results directory
    setup_results_directory()

    if len(sys.argv) > 1:
        # Run specific step
        try:
            step_num = int(sys.argv[1])
            if step_num < 1 or step_num > 4:
                print("Error: Step must be 1-4")
                sys.exit(1)

            num, name, module = STEPS[step_num - 1]
            success = run_step(num, name, module)
            if not success:
                sys.exit(1)
        except ValueError:
            print("Error: Invalid step number")
            print("Usage: python run_pipeline.py [1-4]")
            sys.exit(1)
    else:
        # Run all steps
        print("Running complete pipeline (all steps)...")
        for step_num, step_name, module in STEPS:
            success = run_step(step_num, step_name, module)
            if not success:
                sys.exit(1)
        print(f"\n{'='*60}")
        print("Pipeline complete!")
        print(f"{'='*60}\n")

        # Clean up intermediate files after all steps complete
        cleanup_intermediate_files()


if __name__ == "__main__":
    main()