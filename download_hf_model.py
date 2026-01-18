#!/usr/bin/env python3
"""
Script to download model from Hugging Face.

Usage:
    1. First, get your HF token from: https://huggingface.co/settings/tokens
    2. Run: python download_hf_model.py --token YOUR_HF_TOKEN
    
    Or set the HF_TOKEN environment variable:
        export HF_TOKEN=your_token_here
        python download_hf_model.py
"""

import os
import argparse
from huggingface_hub import snapshot_download, login

def main():
    parser = argparse.ArgumentParser(description="Download model from HuggingFace")
    parser.add_argument("--token", type=str, default=None, help="HuggingFace API token")
    parser.add_argument("--output-dir", type=str, default="./", help="Output directory")
    args = parser.parse_args()
    
    # Get token from args or environment
    token = args.token or os.environ.get("HF_TOKEN")
    
    if not token:
        print("=" * 60)
        print("HuggingFace token required!")
        print("=" * 60)
        print("\n1. Go to: https://huggingface.co/settings/tokens")
        print("2. Create a new token (read access is enough)")
        print("3. Run this script with: python download_hf_model.py --token YOUR_TOKEN")
        print("\nOr set environment variable:")
        print("   export HF_TOKEN=your_token_here")
        print("   python download_hf_model.py")
        print("=" * 60)
        
        # Try interactive login
        print("\nAttempting interactive login...")
        try:
            token = input("Enter your HuggingFace token: ").strip()
            if not token:
                print("No token provided. Exiting.")
                return
        except KeyboardInterrupt:
            print("\nCancelled.")
            return
    
    # Login with token
    print(f"\nLogging in to HuggingFace...")
    try:
        login(token=token)
        print("✅ Login successful!")
    except Exception as e:
        print(f"❌ Login failed: {e}")
        return
    
    # Download the specific folder
    repo_id = "LeCar-Lab/BFM-Zero"
    subfolder = "model"
    
    print(f"\nDownloading from {repo_id}...")
    print(f"Subfolder: {subfolder}")
    print(f"Output directory: {args.output_dir}")
    
    try:
        local_dir = snapshot_download(
            repo_id=repo_id,
            local_dir=args.output_dir,
            allow_patterns=f"{subfolder}/**",
            token=token
        )
        print(f"\n✅ Download complete!")
        print(f"Files saved to: {local_dir}")
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        print("\nPossible issues:")
        print("  - Token doesn't have access to this repo")
        print("  - You need to accept the model's license on HuggingFace first")
        print(f"  - Visit: https://huggingface.co/{repo_id} and accept terms")

if __name__ == "__main__":
    main()
