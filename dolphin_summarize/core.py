"""
Core functionality for creating condensed summaries of tensor names by identifying
and replacing numeric patterns with range notations.
"""
import json
import re
import os
import glob
import struct
from collections import defaultdict
from typing import Dict, List, Tuple, Optional


def read_header_from_safetensors(file_path: str) -> dict:
    """
    Extract header information from a safetensors file without loading the full file.
    """
    try:
        with open(file_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json)
            return header
    except Exception:
        return {}


def identify_patterns(names: List[str]) -> Dict[str, List[int]]:
    """
    Identify numeric patterns in a list of strings.
    Returns a dictionary mapping patterns to lists of numeric values.
    """
    patterns = defaultdict(set)
    
    # Use regex to find all numeric parts in the strings
    numeric_pattern = re.compile(r'(\D*)(\d+)(\D*)')
    
    for name in names:
        # Use finditer to iterate through all matches in the string
        curr_pos = 0
        matches = list(numeric_pattern.finditer(name))
        
        for i, match in enumerate(matches):
            prefix, number, suffix = match.groups()
            
            # Get the portion of the string from current position to start of match
            if i == 0:
                prefix_context = prefix
            else:
                prefix_context = name[curr_pos:match.start()]
            
            # Update current position to end of match
            curr_pos = match.end()
            
            # If this is the last match, include everything to the end of string
            if i == len(matches) - 1:
                suffix_context = suffix
            else:
                suffix_context = ""
            
            # Create a unique pattern key with prefix and suffix context
            pattern_key = f"{prefix_context}[N]{suffix_context}"
            patterns[pattern_key].add(int(number))
    
    # Convert sets to sorted lists
    return {k: sorted(list(v)) for k, v in patterns.items()}


def generate_range_notation(values: List[int]) -> str:
    """
    Convert a list of integers to a range notation if applicable.
    E.g., [0, 1, 2, 3] becomes "[0-3]"
    """
    if not values:
        return "[]"
    
    if len(values) == 1:
        return f"[{values[0]}]"
    
    # Check if the values form a continuous range
    if values == list(range(min(values), max(values) + 1)):
        return f"[{min(values)}-{max(values)}]"
    
    # For non-continuous ranges, return a comma-separated list
    return f"[{', '.join(map(str, values))}]"


def replace_patterns(names: List[str]) -> List[str]:
    """
    Replace numeric patterns in strings with range notations.
    This is the core function that handles the pattern consolidation.
    """
    # First group strings by their structure (ignoring numeric values)
    structure_map = defaultdict(list)
    
    # Create a regex to match numeric parts
    numeric_re = re.compile(r'\d+')
    
    for name in names:
        # Replace all numeric parts with a placeholder to get the structure
        structure = numeric_re.sub('NUM', name)
        structure_map[structure].append(name)
    
    # Process each structure group separately
    result = []
    
    for structure, items in structure_map.items():
        if len(items) == 1:
            # No consolidation needed for single items
            result.append(items[0])
            continue
        
        # Find all numeric patterns in this structure group
        patterns = identify_patterns(items)
        
        # Start with the first item as a template and replace numeric parts
        template = items[0]
        for pattern, values in patterns.items():
            if len(values) > 1:  # Only replace if there are multiple values
                prefix, suffix = pattern.split('[N]')
                # Create regex pattern to match exactly this occurrence
                pattern_regex = re.escape(prefix) + r'(\d+)' + re.escape(suffix)
                range_notation = generate_range_notation(values)
                template = re.sub(pattern_regex, prefix + range_notation + suffix, template)
        
        result.append(template)
    
    return sorted(result)


def extract_metadata(file_path: str, tensor_name: str) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Extract shape and dtype from a safetensors file for a specific tensor.
    """
    header = read_header_from_safetensors(file_path)
    if tensor_name in header:
        info = header[tensor_name]
        return info.get('shape'), info.get('dtype')
    return None, None


def summarize_architecture(model_dir: str, verbose: bool = False) -> List[str]:
    """
    Generate a condensed summary of tensor names from safetensors files.
    """
    # Look for the index file
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    param_names = []
    param_file_map = {}
    
    if os.path.exists(index_path):
        # Load the JSON file
        with open(index_path, 'r') as f:
            data = json.load(f)
        
        if "weight_map" in data:
            # Extract parameter names and file paths
            param_file_map = data["weight_map"]
            param_names = list(param_file_map.keys())
            
            if verbose:
                print(f"Found {len(param_names)} parameters from index file")
        else:
            if verbose:
                print("The index file does not contain a 'weight_map' field")
    
    # If we don't have parameters yet, try direct safetensors files
    if not param_names:
        safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
        
        if not safetensors_files:
            raise ValueError(f"Could not find any safetensors files in {model_dir}")
        
        # Extract parameter names directly from safetensors headers
        for sf_file in safetensors_files:
            header = read_header_from_safetensors(sf_file)
            file_params = list(header.keys())
            param_names.extend(file_params)
            
            # Map each parameter to its file
            for param in file_params:
                param_file_map[param] = os.path.basename(sf_file)
        
        if verbose:
            print(f"Found {len(param_names)} parameters from direct safetensors inspection")
            
        if not param_names:
            raise ValueError("Could not extract any parameter names from safetensors files")
    
    # Get all safetensors files for metadata extraction
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if verbose:
        print(f"Found {len(safetensors_files)} safetensors files")
    
    # Process the file and get condensed parameter names
    condensed_names = replace_patterns(param_names)
    
    # Add metadata (shape and dtype) to the condensed names
    result = []
    
    # Cache tensor metadata to avoid repeated file reads
    metadata_cache = {}
    
    for condensed_name in condensed_names:
        # Find an original parameter that matches this condensed pattern
        base_pattern = condensed_name
        for range_notation in re.findall(r'\[\d+(?:-\d+)?\]', condensed_name):
            # Replace range notations with regex patterns to match any number in the range
            if '-' in range_notation:
                base_pattern = base_pattern.replace(range_notation, r'\d+')
            else:
                # Single number in brackets like [5]
                num = range_notation.strip('[]')
                base_pattern = base_pattern.replace(range_notation, num)
        
        base_regex = re.compile(f'^{base_pattern}$'.replace(r'\d+', r'(\d+)'))
        
        # Find a matching original parameter
        representative = None
        for param in param_names:
            if base_regex.match(param):
                representative = param
                break
        
        if not representative:
            # If no matching parameter found, just add the condensed name
            result.append(condensed_name)
            continue
        
        # Extract metadata for the representative parameter
        if representative in metadata_cache:
            shape, dtype = metadata_cache[representative]
        else:
            # Find the file containing this parameter
            file_path = os.path.join(model_dir, param_file_map.get(representative, ''))
            if os.path.exists(file_path):
                shape, dtype = extract_metadata(file_path, representative)
                metadata_cache[representative] = (shape, dtype)
            else:
                shape, dtype = None, None
        
        # Format output string with metadata
        output = condensed_name
        if output.endswith('.weight'):
            output = output[:-7]
        if shape:
            output += f",[{','.join(map(str, shape))}]"
        if dtype:
            output += f",{dtype}"
        
        result.append(output)
    
    return sorted(result)
