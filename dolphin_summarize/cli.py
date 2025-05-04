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

def download_from_huggingface(repo_id, verbose=False):
    """Download model from Hugging Face Hub."""
    try:
        from huggingface_hub import snapshot_download
        if verbose:
            print(f"Downloading {repo_id} from Hugging Face Hub...")
        
        # Create a temporary directory that will be automatically cleaned up
        with tempfile.TemporaryDirectory() as temp_dir:
            download_path = snapshot_download(
                repo_id=repo_id,
                local_dir=temp_dir,
                allow_patterns=["*.json", "*.safetensors", "*.safetensors.index.json"],
            )
            if verbose:
                print(f"Downloaded to temporary location: {download_path}")
                
            # Process the downloaded model
            condensed_summary = core.summarize_architecture(download_path, verbose)
            return condensed_summary
            
    except ImportError:
        print("Error: huggingface_hub package is required to download from Hugging Face Hub.")
        print("Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"Error downloading from Hugging Face Hub: {str(e)}")
        sys.exit(1)

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    try:
        try:
            import safetensors
        except ImportError:
            print("Warning: safetensors package not installed. Shape extraction may be limited.")
            print("Install with: pip install safetensors")
        
        # Determine if input is a local path or a Hugging Face repo
        if is_huggingface_repo(args.repo_or_path):
            # Download and process from Hugging Face Hub
            condensed_summary = download_from_huggingface(args.repo_or_path, args.verbose)
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
