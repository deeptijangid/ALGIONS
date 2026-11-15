"""
Algions AI Labs - E-Commerce Recommendation Engine
Script: Generate synthetic raw catalog, users, and interaction streams.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.data_generator import run_data_generation

def main():
    print("=" * 60)
    print("STEP 1: Generating E-Commerce Raw Dataset")
    print("=" * 60)
    run_data_generation(output_dir="data/raw", random_state=42)
    print("[SUCCESS] Dataset generation complete.")

if __name__ == "__main__":
    main()

