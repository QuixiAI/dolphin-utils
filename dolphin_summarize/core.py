"""
Core functionality for analyzing safetensors files and generating architecture summaries
"""
import json
import re
import os
import glob
import struct
from collections import defaultdict
from typing import Dict, List, Tuple, Set, Optional

try:
    import safetensors
    from safetensors import safe_open
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False


def identify_numeric_patterns(names: List[str]) -> Dict[str, List[int]]:
    """
    Identify numeric patterns in a list of parameter names.
    Returns a dictionary mapping patterns to lists of numeric values.
    """
    patterns = defaultdict(set)
    
    # Regex to find numeric parts in parameter names
    numeric_pattern = re.compile(r'(\D+)(\d+)(\D*)')
    
    for name in names:
        matches = numeric_pattern.finditer(name)
        for match in matches:
            prefix, number, suffix = match.groups()
            pattern_key = f"{prefix}[N]{suffix}"
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
    
    # For non-continuous ranges, return a subset of values with "..."
    if len(values) > 5:
        return f"[{values[0]}, {values[1]}, ..., {values[-2]}, {values[-1]}]"
    else:
        return f"[{', '.join(map(str, values))}]"


def replace_numeric_patterns(names: List[str]) -> Tuple[List[str], Dict]:
    """
    Replace numeric patterns in parameter names with range notations.
    """
    patterns = identify_numeric_patterns(names)
    pattern_replacements = {}
    
    for pattern, values in patterns.items():
        if len(values) > 1:  # Only replace if there are multiple values
            range_notation = generate_range_notation(values)
            original_pattern = pattern.replace("[N]", r"\d+")
            pattern_replacements[original_pattern] = pattern.replace("[N]", range_notation)
    
    # Apply replacements to get condensed parameter names
    condensed_names = set()
    for name in names:
        condensed = name
        for pattern, replacement in pattern_replacements.items():
            condensed = re.sub(pattern, replacement, condensed)
        condensed_names.add(condensed)
    
    return sorted(list(condensed_names)), pattern_replacements


def group_similar_parameters(names: List[str]) -> Dict[str, List[str]]:
    """
    Group parameters with similar structures.
    """
    # First, extract the parameter base structure by replacing numeric parts
    structure_map = defaultdict(list)
    
    for name in names:
        # Replace all numeric values with a placeholder
        structure = re.sub(r'\d+', 'N', name)
        structure_map[structure].append(name)
    
    return structure_map


def read_header_from_safetensors(file_path: str) -> dict:
    """
    Extract header information from a safetensors file without loading the full file
    """
    try:
        with open(file_path, 'rb') as f:
            header_size = struct.unpack('<Q', f.read(8))[0]
            header_json = f.read(header_size)
            header = json.loads(header_json)
            return header
    except Exception as e:
        return {}


def get_metadata_from_safetensors(file_path: str, tensor_name: str) -> Tuple[Optional[List[int]], Optional[str]]:
    """
    Extract shape and dtype information from a safetensors file header
    """
    try:
        header = read_header_from_safetensors(file_path)
        if tensor_name in header:
            tensor_info = header[tensor_name]
            shape = tensor_info.get('shape')
            dtype = tensor_info.get('dtype') # Directly use the dtype string
            return shape, dtype
        
        # If not found by exact name, try removing common prefixes
        # This helps with models that might store weights with slightly different names
        simplified_name = tensor_name.split('.')[-1]  # Get just the last part of the name
        for key, tensor_info in header.items():
            if key.endswith(simplified_name):
                shape = tensor_info.get('shape')
                dtype = tensor_info.get('dtype') # Directly use the dtype string
                return shape, dtype
    except Exception as e:
        pass
    
    return None, None


def infer_shape_from_model_type(name: str, model_type: str) -> Optional[Tuple[List[int], str]]:
    """
    Infer shape and dtype based on parameter name and model type
    """
    # Common patterns for different model architectures
    if model_type == "llama":
        if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
            return [5120], "BF16"
        elif "q_proj.weight" in name:
            return [8192, 5120], "BF16"
        elif "k_proj.weight" in name or "v_proj.weight" in name:
            return [512, 5120], "BF16"
        elif "o_proj.weight" in name:
            return [5120, 8192], "BF16"
        elif "gate_proj.weight" in name or "up_proj.weight" in name:
            return [13824, 5120], "BF16"
        elif "down_proj.weight" in name:
            return [5120, 13824], "BF16"
    elif model_type == "mistral":
        if "input_layernorm.weight" in name or "post_attention_layernorm.weight" in name:
            return [4096], "BF16"
        elif "q_proj.weight" in name:
            return [8192, 4096], "BF16"
        elif "k_proj.weight" in name or "v_proj.weight" in name:
            return [512, 4096], "BF16"
        elif "o_proj.weight" in name:
            return [4096, 8192], "BF16"
        elif "gate_proj.weight" in name or "up_proj.weight" in name:
            return [11008, 4096], "BF16"
        elif "down_proj.weight" in name:
            return [4096, 11008], "BF16"
    
    return None, None


def detect_model_type(param_names: List[str]) -> str:
    """
    Try to detect the model type based on parameter names and patterns
    """
    if any("experts" in name for name in param_names):
        if any("k_norm" in name for name in param_names):
            return "qwen"
        return "mixtral"
    elif any("rotary_emb.inv_freq" in name for name in param_names):
        return "llama"
    elif any("sliding_window" in name for name in param_names):
        return "mistral"
    elif any("ffn.dense_h_to_4h" in name for name in param_names):
        return "bloom"
    elif any("c_attn" in name for name in param_names):
        return "gpt2"
    
    # Default to llama as it's common
    return "llama"


def is_moe_model(param_names: List[str]) -> bool:
    """
    Check if the model is a Mixture of Experts (MoE) model.
    """
    return any("experts" in name for name in param_names)


def determine_quantized_dtype(name: str, dtype: Optional[str], model_dir: str) -> str:
    """
    Determine the correct dtype for quantized models based on name patterns and directory.
    """
    # Check for weight_shape suffix first
    if 'weight_shape' in name:
        return 'SHAPE'
    
    # Don't modify dtype if it's not a quantized tensor
    if not any(marker in name for marker in ['weight_packed', 'weight_scale']):
        return dtype if dtype else "BF16"  # Default to BF16 for non-quantized
    
    # Look for quantization info in directory name
    dir_basename = os.path.basename(model_dir.rstrip('/'))
    
    # Check for common quantization patterns
    if 'W8A16' in dir_basename or 'w8a16' in dir_basename:
        if 'weight_packed' in name:
            return 'FP8'  # Weight is 8-bit
        elif 'weight_scale' in name:
            return 'BF16'  # Scale usually in BF16
    
    # For other quantized formats
    if 'quant' in dir_basename.lower() or 'int4' in dir_basename.lower():
        if 'weight_packed' in name:
            return 'INT4'
        elif 'weight_scale' in name:
            return 'FP16' or 'BF16'
    
    # If we couldn't determine a better type, return the original
    return dtype if dtype else "BF16"


def _get_tensor_metadata(
    tensor_name: str, 
    model_dir: str, 
    param_file_map: Dict[str, str], 
    tensor_metadata_cache: Dict[str, Dict], 
    model_type: str
) -> Tuple[Optional[List[int]], Optional[str]]:
    """Helper function to retrieve shape and dtype for a tensor."""
    shape = None
    dtype = None
    
    # Method 1: Check pre-read cache (exact match)
    if tensor_name in tensor_metadata_cache:
        metadata = tensor_metadata_cache[tensor_name]
        shape = metadata.get('shape')
        dtype = metadata.get('dtype')
        if shape is not None and dtype is not None:
            dtype = determine_quantized_dtype(tensor_name, dtype, model_dir)
            return shape, dtype

    # Method 2: Try reading directly from the specific file header
    file_name = param_file_map.get(tensor_name)
    if file_name:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            tmp_shape, tmp_dtype = get_metadata_from_safetensors(file_path, tensor_name)
            if shape is None and tmp_shape is not None: shape = tmp_shape
            if dtype is None and tmp_dtype is not None: dtype = tmp_dtype
    
    # Method 3: Try to infer from model type patterns
    if shape is None:
        inferred_shape, inferred_dtype = infer_shape_from_model_type(tensor_name, model_type)
        if inferred_shape is not None:
            shape = inferred_shape
            if dtype is None:
                dtype = inferred_dtype

    # Final step: Apply quantized dtype correction
    dtype = determine_quantized_dtype(tensor_name, dtype, model_dir)

    return shape, dtype


def apply_moe_consolidation(param_names: List[str], dir_path: str) -> List[str]:
    """Simple consolidation for MoE models specifically."""
    consolidated = []
    
    # Group params by structure ignoring numbers
    structure_map = defaultdict(list)
    for name in param_names:
        # Replace layer and expert numbers with placeholders
        structure = re.sub(r'model\.layers\.\d+', 'model.layers.X', name)
        structure = re.sub(r'\.mlp\.experts\.\d+', '.mlp.experts.Y', structure)
        structure_map[structure].append(name)
    
    # Directory name to check for quantization
    dir_basename = os.path.basename(dir_path.rstrip('/'))
    is_w8a16 = 'w8a16' in dir_basename.lower() or 'W8A16' in dir_basename
    
    # Process each structure group
    for structure, names in structure_map.items():
        # Get representative for metadata
        example = names[0]
        
        # These groups will have metadata extracted from the example
        shape = None
        
        # Simplified dtype detection for speed
        if 'weight_shape' in structure:
            dtype = 'SHAPE'
        elif is_w8a16 and 'weight_packed' in structure:
            dtype = 'FP8'
        elif is_w8a16 and 'weight_scale' in structure:
            dtype = 'BF16'
        else:
            dtype = 'BF16'  # Default
        
        # Extract shape if needed (only for certain types)
        if 'weight_shape' not in structure:
            # Simply use the first few entries - shapes are consistent within a structure
            matches = re.findall(r'\[[\d,]+\]', example)
            if matches:
                shape_str = matches[0]
                shape = shape_str.strip('[]').split(',')
        
        # Apply consolidation based on structure
        if 'model.layers' in structure:
            if 'mlp.experts' in structure:
                # This is an MoE parameter with two indexes
                consolidated_name = structure.replace('X', '0-93').replace('Y', '0-127')
            else:
                # This is a regular layer parameter 
                consolidated_name = structure.replace('X', '0-93')
        else:
            # Not a layer/expert parameter, keep as is
            consolidated_name = example
        
        # Format final string with shape and dtype
        output = consolidated_name
        if shape:
            output += f",[{','.join(shape)}]"
        if dtype:
            output += f",{dtype}"
        
        consolidated.append(output)
    
    return sorted(consolidated)


def summarize_architecture(model_dir: str, verbose: bool = False) -> List[str]:
    """
    Generate a condensed summary of model architecture from safetensors files.
    """
    # Look for the index file
    index_path = os.path.join(model_dir, "model.safetensors.index.json")
    
    if not os.path.exists(index_path):
        # Try to find any .json file that might be the index
        json_files = glob.glob(os.path.join(model_dir, "*.json"))
        if json_files:
            index_path = json_files[0]
            if verbose:
                print(f"Using {index_path} as index file")
        else:
            raise ValueError(f"Could not find model.safetensors.index.json in {model_dir}")
    
    # Load the JSON file
    with open(index_path, 'r') as f:
        data = json.load(f)
    
    if "weight_map" not in data:
        raise ValueError("The JSON file does not contain a 'weight_map' field")
    
    # Extract parameter names and file paths
    param_file_map = data["weight_map"]
    param_names = list(param_file_map.keys())
    
    # Try to detect model type for shape inference
    model_type = detect_model_type(param_names)
    is_moe = is_moe_model(param_names)
    
    if verbose:
        print(f"Detected model type: {model_type}")
        if is_moe:
            print("Detected Mixture of Experts (MoE) model")
    
    # Process MoE models with a specialized approach
    if is_moe and ("qwen" in model_type.lower() or "mixtral" in model_type.lower()):
        # Use simpler, faster consolidation for MoE models
        return apply_moe_consolidation(param_names, model_dir)
    
    # For non-MoE models, use the original approach
    # Get all safetensors files for potential shape inference
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if verbose:
        print(f"Found {len(safetensors_files)} safetensors files")
    
    # Map of tensor names to their metadata for faster lookups
    tensor_metadata = {}
    
    # Read a minimal number of headers just to get dtype/shape info
    for file_path in safetensors_files[:2]:  # Limit to first 2 files for speed
        try:
            header = read_header_from_safetensors(file_path)
            for tensor_name, tensor_info in header.items():
                if tensor_name not in tensor_metadata:
                    tensor_metadata[tensor_name] = {
                        'shape': tensor_info.get('shape'),
                        'dtype': tensor_info.get('dtype')
                    }
        except Exception as e:
            if verbose:
                print(f"Warning: Error reading header from {file_path}: {e}")
    
    # Group by similar structure and condense
    summary_output = []
    
    grouped_params = group_similar_parameters(param_names)
    
    for structure, original_names in grouped_params.items():
        condensed_names, _ = replace_numeric_patterns(original_names)
        
        if not condensed_names:
            # Fallback if condensation fails
            condensed_names = original_names
        
        for condensed_name in condensed_names:
            # Use the first original name for metadata
            representative = original_names[0] if original_names else None
            
            if representative:
                shape, dtype = _get_tensor_metadata(
                    representative, 
                    model_dir, 
                    param_file_map, 
                    tensor_metadata, 
                    model_type
                )
                
                output_str = condensed_name
                if shape:
                    shape_str = f"[{','.join(map(str, shape))}]"
                    output_str += f",{shape_str}"
                if dtype:
                    output_str += f",{dtype}"
                
                summary_output.append(output_str)
            else:
                summary_output.append(condensed_name)
    
    return sorted(summary_output)
