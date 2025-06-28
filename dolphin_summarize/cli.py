"""
Command-line interface for dolphin-summarize
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path
from . import core

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Summarize model architecture from safetensors files")
    parser.add_argument("repo_or_path", nargs="?", type=str, default=".",
                      help="Directory containing safetensors files or Hugging Face repo ID (positional, default: current directory)")
    parser.add_argument("--output", "-o", type=str, help="Path to output file (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    return parser.parse_args()

def is_huggingface_repo(repo_id):
    """Check if the input is a Hugging Face repo ID."""
    # Simple heuristic: if it contains a slash and doesn't exist as a local path
    return "/" in repo_id and not os.path.exists(repo_id)

def process_huggingface_repo(repo_id, verbose=False):
    """Process model from Hugging Face Hub using remote header reading."""
    try:
        if verbose:
            print(f"Processing {repo_id} from Hugging Face Hub (remote header reading)...")
        
        # Use the new remote processing function
        condensed_summary = core.summarize_remote_architecture(repo_id, verbose)
        return condensed_summary
            
    except ImportError:
        print("Error: huggingface_hub and requests packages are required for remote processing.")
        print("Install with: pip install huggingface_hub requests")
        sys.exit(1)
    except Exception as e:
        print(f"Error processing Hugging Face repository: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    try:
        # Determine if input is a local path or a Hugging Face repo
        if is_huggingface_repo(args.repo_or_path):
            # Process from Hugging Face Hub using remote header reading
            condensed_summary = process_huggingface_repo(args.repo_or_path, args.verbose)
        else:
            # Process local directory
            if args.verbose:
                print(f"Processing local directory: {args.repo_or_path}")
            condensed_summary = core.summarize_architecture(args.repo_or_path, args.verbose)
        
        # Print the summary
        for param in condensed_summary:
            print(param)
        
        # Write to output file if specified
        if args.output:
            with open(args.output, 'w') as f:
                for param in condensed_summary:
                    f.write(f"{param}\n")
            print(f"\nSummary written to {args.output}")
            
    except Exception as e:
        print(f"Error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
