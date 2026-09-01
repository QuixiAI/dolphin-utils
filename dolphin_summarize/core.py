"""
Core functionality for creating condensed summaries of tensor names by identifying
and replacing numeric patterns with range notations.
"""
import json
import re
import os
import glob
import struct
import requests
import tempfile
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
from huggingface_hub import HfApi, hf_hub_url
from huggingface_hub.utils import get_token


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
    
    # For non-continuous ranges, return a comma-separated list without spaces
    return f"[{','.join(map(str, values))}]"


def _condense_group(items: List[str]) -> str:
    """Collapse one structure-uniform group of names into range notation."""
    if len(items) == 1:
        return items[0]
    patterns = identify_patterns(items)
    template = items[0]
    for pattern, values in patterns.items():
        if len(values) > 1:  # Only replace if there are multiple values
            prefix, suffix = pattern.split('[N]')
            pattern_regex = re.escape(prefix) + r'(\d+)' + re.escape(suffix)
            range_notation = generate_range_notation(values)
            template = re.sub(pattern_regex, prefix + range_notation + suffix, template)
    return template


def replace_patterns(names: List[str]) -> List[str]:
    """
    Replace numeric patterns in strings with range notations.
    This is the core function that handles the pattern consolidation.
    """
    # First group strings by their structure (ignoring numeric values)
    structure_map = defaultdict(list)
    numeric_re = re.compile(r'\d+')
    for name in names:
        # Replace all numeric parts with a placeholder to get the structure
        structure = numeric_re.sub('NUM', name)
        structure_map[structure].append(name)
    return sorted(_condense_group(items) for items in structure_map.values())


def replace_patterns_with_metadata(names, meta_of):
    """Consolidate names into range notation WITHOUT merging tensors whose
    shape or dtype differ.

    Plain name-pattern grouping can collapse e.g. ``layers.[0-45].o_proj``
    over layers where the tensor is [4096,8192] on some layers and
    [4096,16384] on others (hybrid models), and the single reported shape
    is then wrong for part of the range. Grouping here is keyed on
    (structure, shape, dtype), and each condensed name is returned with a
    representative member so callers annotate from a tensor that is
    guaranteed to share the whole group's metadata.

    Returns a list of (condensed_name, representative_name, shape, dtype).
    """
    structure_map = defaultdict(list)
    numeric_re = re.compile(r'\d+')
    for name in names:
        structure = numeric_re.sub('NUM', name)
        shape, dtype = meta_of(name)
        key = (structure, tuple(shape) if shape else None, dtype)
        structure_map[key].append(name)
    result = []
    for (_, shape, dtype), items in structure_map.items():
        result.append((
            _condense_group(items),
            items[0],
            list(shape) if shape else None,
            dtype,
        ))
    result.sort(key=lambda r: r[0])
    return result


def get_repo_safetensors_files(repo_id: str, verbose: bool = False) -> List[str]:
    """
    Get list of all .safetensors files in a Hugging Face repository.
    """
    try:
        # Get authentication token if available
        token = None
        try:
            token = get_token()
            if verbose and token:
                print("Using Hugging Face authentication token for repository access")
        except Exception:
            if verbose:
                print("No Hugging Face authentication token found for repository access")
        
        api = HfApi(token=token)
        repo_info = api.repo_info(repo_id)
        
        safetensors_files = []
        for sibling in repo_info.siblings:
            if sibling.rfilename.endswith('.safetensors'):
                safetensors_files.append(sibling.rfilename)
        
        if verbose:
            print(f"Found {len(safetensors_files)} safetensors files in {repo_id}")
            
        return safetensors_files
    except Exception as e:
        raise ValueError(f"Failed to get repository info for {repo_id}: {str(e)}")


def read_remote_safetensors_header(repo_id: str, filename: str, verbose: bool = False) -> dict:
    """
    Read safetensors header from a remote Hugging Face repository.
    Uses multiple fallback strategies for maximum reliability.
    """
    url = hf_hub_url(repo_id, filename)
    
    # Get authentication headers if available
    auth_headers = {}
    try:
        # Try to get the token from huggingface_hub
        token = get_token()
        if token:
            auth_headers['Authorization'] = f'Bearer {token}'
            if verbose:
                print("Using Hugging Face authentication token")
    except Exception:
        if verbose:
            print("No Hugging Face authentication token found")
    
    # Strategy 1: Try HTTP range request (most efficient)
    try:
        if verbose:
            print(f"Trying range request for {filename}...")
        
        # First, get the header size (first 8 bytes)
        headers = {'Range': 'bytes=0-7'}
        headers.update(auth_headers)
        response = requests.get(url, headers=headers, timeout=30)
        
        if response.status_code == 206:  # Partial content
            header_size = struct.unpack('<Q', response.content)[0]
            
            # Now get the full header
            headers = {'Range': f'bytes=0-{header_size + 7}'}
            headers.update(auth_headers)
            response = requests.get(url, headers=headers, timeout=30)
            
            if response.status_code == 206:
                header_json = response.content[8:8+header_size]
                header = json.loads(header_json)
                if verbose:
                    print(f"Successfully read header via range request ({len(header)} tensors)")
                return header
    except Exception as e:
        if verbose:
            print(f"Range request failed: {str(e)}")
    
    # Strategy 2: Try streaming (read just enough bytes)
    try:
        if verbose:
            print(f"Trying streaming approach for {filename}...")
        
        response = requests.get(url, headers=auth_headers, stream=True, timeout=30)
        response.raise_for_status()
        
        # Read header size
        header_size_bytes = b''
        for chunk in response.iter_content(chunk_size=8):
            header_size_bytes += chunk
            if len(header_size_bytes) >= 8:
                break
        
        header_size = struct.unpack('<Q', header_size_bytes[:8])[0]
        
        # Read the header
        header_bytes = header_size_bytes[8:]
        bytes_needed = header_size - len(header_bytes)
        
        for chunk in response.iter_content(chunk_size=min(bytes_needed, 8192)):
            header_bytes += chunk
            if len(header_bytes) >= header_size:
                break
        
        header = json.loads(header_bytes[:header_size])
        if verbose:
            print(f"Successfully read header via streaming ({len(header)} tensors)")
        return header
        
    except Exception as e:
        if verbose:
            print(f"Streaming failed: {str(e)}")
    
    # Strategy 3: Full download to temporary file (guaranteed to work)
    try:
        if verbose:
            print(f"Falling back to full download for {filename}...")
        
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            response = requests.get(url, headers=auth_headers, timeout=300)  # Longer timeout for full download
            response.raise_for_status()
            
            temp_file.write(response.content)
            temp_file.flush()
            
            # Read header from the temporary file
            header = read_header_from_safetensors(temp_file.name)
            
            # Clean up
            os.unlink(temp_file.name)
            
            if verbose:
                print(f"Successfully read header via full download ({len(header)} tensors)")
            return header
            
    except Exception as e:
        if verbose:
            print(f"Full download failed: {str(e)}")
        raise ValueError(f"Failed to read header from {filename} using all methods: {str(e)}")


def summarize_remote_architecture(repo_id: str, verbose: bool = False) -> List[str]:
    """
    Generate a condensed summary of tensor names from a remote Hugging Face repository.
    """
    # Get list of safetensors files
    safetensors_files = get_repo_safetensors_files(repo_id, verbose)
    
    if not safetensors_files:
        raise ValueError(f"No safetensors files found in repository {repo_id}")
    
    # Read headers from all files, keeping shape/dtype for every tensor so
    # consolidation cannot merge tensors with differing metadata.
    param_names = []
    param_meta = {}
    
    for filename in safetensors_files:
        if verbose:
            print(f"Processing {filename}...")
        
        header = read_remote_safetensors_header(repo_id, filename, verbose)
        for key, info in header.items():
            if key.startswith('__'):
                continue
            param_names.append(key)
            param_meta[key] = (
                info.get('shape') if isinstance(info, dict) else None,
                info.get('dtype') if isinstance(info, dict) else None,
            )
    
    if verbose:
        print(f"Found {len(param_names)} total parameters across {len(safetensors_files)} files")
    
    if not param_names:
        raise ValueError("Could not extract any parameter names from safetensors files")
    
    # Consolidate without merging across differing shape/dtype, then emit.
    result = []
    for condensed, _rep, shape, dtype in replace_patterns_with_metadata(
        param_names, lambda n: param_meta.get(n, (None, None))
    ):
        output = condensed
        if shape:
            output += f",[{','.join(map(str, shape))}]"
        if dtype:
            output += f",{dtype}"
        result.append(output)
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
            # Filter out metadata keys that aren't actual tensor parameters
            file_params = [key for key in header.keys() if not key.startswith('__')]
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
    
    # Consolidate without merging across differing shape/dtype, reading
    # each shard header at most once.
    header_cache = {}

    def _meta_of(name):
        filename = param_file_map.get(name)
        if not filename:
            return (None, None)
        file_path = os.path.join(model_dir, filename)
        if file_path not in header_cache:
            if not os.path.exists(file_path):
                header_cache[file_path] = {}
            else:
                header_cache[file_path] = read_header_from_safetensors(file_path)
        info = header_cache[file_path].get(name)
        if not isinstance(info, dict):
            return (None, None)
        return (info.get('shape'), info.get('dtype'))

    result = []
    for condensed, _rep, shape, dtype in replace_patterns_with_metadata(
        param_names, _meta_of
    ):
        output = condensed
        if shape:
            output += f",[{','.join(map(str, shape))}]"
        if dtype:
            output += f",{dtype}"
        result.append(output)
    return sorted(result)
