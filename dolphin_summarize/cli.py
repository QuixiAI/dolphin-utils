"""
Command-line interface for dolphin-summarize
"""
import argparse
import sys
from . import core

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Summarize model architecture from safetensors files")
    parser.add_argument("dir_path", nargs="?", type=str, default=".",
                      help="Directory containing safetensors files (positional, default: current directory)")
    parser.add_argument("--output", "-o", type=str, help="Path to output file (optional)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show verbose output")
    return parser.parse_args()

def main():
    """Main entry point for the CLI"""
    args = parse_args()
    
    try:
        if not core.SAFETENSORS_AVAILABLE:
            print("Warning: safetensors package not installed. Shape extraction may be limited.")
            print("Install with: pip install safetensors")
        
        # Get the summary
        condensed_summary = core.summarize_architecture(args.dir_path, args.verbose)
        
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
