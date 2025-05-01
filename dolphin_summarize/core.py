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
    
    Enhanced to better handle Mixture of Experts (MoE) models with multiple numeric indices.
    """
    patterns = defaultdict(set)
    
    # Regex to find numeric parts in parameter names - now matches all numeric parts
    numeric_pattern = re.compile(r'(\D+)(\d+)(\D*)')
    
    # First pass: identify standard patterns
    for name in names:
        matches = numeric_pattern.finditer(name)
        for match in matches:
            prefix, number, suffix = match.groups()
            pattern_key = f"{prefix}[N]{suffix}"
            patterns[pattern_key].add(int(number))
    
    # Special handling for MoE patterns like "model.layers.X.mlp.experts.Y"
    moe_pattern = re.compile(r'(model\.layers\.)(\d+)(\.mlp\.experts\.)(\d+)(\..*)')
    
    # Second pass: identify MoE layer-expert combinations
    for name in names:
        moe_match = moe_pattern.match(name)
        if moe_match:
            layer_prefix, layer_num, expert_prefix, expert_num, suffix = moe_match.groups()
            
            # Create layer pattern
            layer_pattern_key = f"{layer_prefix}[N]{expert_prefix}{expert_num}{suffix}"
            patterns[layer_pattern_key].add(int(layer_num))
            
            # Create expert pattern
            expert_pattern_key = f"{layer_prefix}{layer_num}{expert_prefix}[N]{suffix}"
            patterns[expert_pattern_key].add(int(expert_num))
            
            # Create combined pattern for better consolidation
            combined_key = f"{layer_prefix}[LAYER]{expert_prefix}[EXPERT]{suffix}"
            if combined_key not in patterns:
                patterns[combined_key] = {"layers": set(), "experts": set()}
            patterns[combined_key]["layers"].add(int(layer_num))
            patterns[combined_key]["experts"].add(int(expert_num))
    
    # Convert regular sets to sorted lists
    result = {}
    for k, v in patterns.items():
        if isinstance(v, set):
            result[k] = sorted(list(v))
        elif isinstance(v, dict):  # For combined patterns
            result[k] = {
                "layers": sorted(list(v["layers"])),
                "experts": sorted(list(v["experts"]))
            }
    
    return result


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
    Enhanced to better handle MoE model patterns.
    """
    patterns = identify_numeric_patterns(names)
    pattern_replacements = {}
    
    # Special handling for MoE combined patterns first
    moe_replacements = {}
    for pattern, values in patterns.items():
        if isinstance(values, dict) and "layers" in values and "experts" in values:
            # This is a combined layer+expert pattern
            layers_range = generate_range_notation(values["layers"])
            experts_range = generate_range_notation(values["experts"])
            
            # Create pattern to match all layer+expert combinations
            layer_part = pattern.replace("[LAYER]", r"(\d+)")
            full_pattern = layer_part.replace("[EXPERT]", r"(\d+)")
            
            # Create replacement with both ranges
            replacement = pattern.replace("[LAYER]", layers_range).replace("[EXPERT]", experts_range)
            
            moe_replacements[full_pattern] = replacement
    
    # Handle regular patterns
    for pattern, values in patterns.items():
        if isinstance(values, list) and len(values) > 1:  # Only replace if there are multiple values
            range_notation = generate_range_notation(values)
            original_pattern = pattern.replace("[N]", r"\d+")
            pattern_replacements[original_pattern] = pattern.replace("[N]", range_notation)
    
    # Apply MoE replacements first (they're more specific)
    condensed_names = set()
    for name in names:
        condensed = name
        # Apply MoE replacements first
        for pattern, replacement in moe_replacements.items():
            condensed = re.sub(pattern, replacement, condensed)
        
        # Then apply regular replacements
        for pattern, replacement in pattern_replacements.items():
            condensed = re.sub(pattern, replacement, condensed)
        
        condensed_names.add(condensed)
    
    # Combine pattern_replacements
    all_replacements = {**pattern_replacements, **moe_replacements}
    
    return sorted(list(condensed_names)), all_replacements


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


def determine_quantized_dtype(name: str, dtype: Optional[str], model_dir: str) -> str:
    """
    Determine the correct dtype for quantized models based on naming patterns and directory name.
    
    Args:
        name: The parameter name
        dtype: The original dtype detected
        model_dir: The model directory path which might contain quantization info
    
    Returns:
        The corrected dtype string
    """
    # Check for weight_shape suffix first (regardless of quantization)
    if 'weight_shape' in name:
        return 'SHAPE'  # Shape info, not an actual tensor dtype
    
    # Don't modify dtype if it's not a quantized tensor
    if not any(marker in name for marker in ['weight_packed', 'weight_scale']):
        return dtype
    
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
    return dtype


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
            # Apply quantized dtype correction if needed
            dtype = determine_quantized_dtype(tensor_name, dtype, model_dir)
            return shape, dtype # Found both

    # Method 2: Check pre-read cache (simplified name match)
    simple_name = tensor_name.split('.')[-1]
    if simple_name in tensor_metadata_cache:
        metadata = tensor_metadata_cache[simple_name]
        if shape is None: shape = metadata.get('shape')
        if dtype is None: dtype = metadata.get('dtype')
        if shape is not None and dtype is not None:
            # Apply quantized dtype correction if needed
            dtype = determine_quantized_dtype(tensor_name, dtype, model_dir)
            return shape, dtype # Found both via simplified name

    # Method 3: Try reading directly from the specific file header (fallback)
    file_name = param_file_map.get(tensor_name)
    if (shape is None or dtype is None) and file_name:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            tmp_shape, tmp_dtype = get_metadata_from_safetensors(file_path, tensor_name)
            if shape is None and tmp_shape is not None: shape = tmp_shape
            if dtype is None and tmp_dtype is not None: dtype = tmp_dtype
            if shape is not None and dtype is not None:
                # Apply quantized dtype correction if needed
                dtype = determine_quantized_dtype(tensor_name, dtype, model_dir)
                return shape, dtype # Found both via direct read

    # Method 4: Use safetensors library if available (another fallback)
    if shape is None and SAFETENSORS_AVAILABLE and file_name:
        file_path = os.path.join(model_dir, file_name)
        if os.path.exists(file_path):
            try:
                with safe_open(file_path, framework="pt") as f:
                    if tensor_name in f.keys():
                        tensor_info = f.get_tensor_info(tensor_name)
                        shape = tensor_info.shape
                    else: # Try simplified name within safetensors file
                        for key in f.keys():
                            if key.endswith(simple_name):
                                tensor_info = f.get_tensor_info(key)
                                shape = tensor_info.shape
                                break
            except Exception:
                pass # Ignore errors during safetensors reading

    # Method 5: Try to infer from model type patterns (last resort)
    if shape is None:
        inferred_shape, inferred_dtype = infer_shape_from_model_type(tensor_name, model_type)
        if inferred_shape is not None:
            shape = inferred_shape
            if dtype is None: # Only use inferred dtype if we don't have one yet
                dtype = inferred_dtype

    # Final step: Apply quantized dtype correction if needed
    if dtype is not None:
        dtype = determine_quantized_dtype(tensor_name, dtype, model_dir)

    return shape, dtype


def summarize_architecture(model_dir: str, verbose: bool = False) -> List[str]:
    """
    Generate a condensed summary of model architecture from safetensors files in the given directory.
    Enhanced to better handle MoE models with nested patterns.
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
    
    # Get all safetensors files for potential shape inference
    safetensors_files = glob.glob(os.path.join(model_dir, "*.safetensors"))
    if verbose:
        print(f"Found {len(safetensors_files)} safetensors files")
    
    # Map of tensor names to their metadata for faster lookups
    tensor_metadata = {}
    
    # Pre-read headers from all safetensors files
    if verbose:
        print("Reading headers from safetensors files...")
    
    for file_path in safetensors_files:
        try:
            header = read_header_from_safetensors(file_path)
            for tensor_name, tensor_info in header.items():
                shape = tensor_info.get('shape')
                dtype = tensor_info.get('dtype') # Directly use the dtype string
                
                tensor_metadata[tensor_name] = {
                    'shape': shape,
                    'dtype': dtype
                }
                
                # Also store with simplified name for fuzzy matching
                simple_name = tensor_name.split('.')[-1]
                if simple_name not in tensor_metadata:
                    tensor_metadata[simple_name] = {
                    'shape': shape,
                    'dtype': dtype # Store the string directly
                }
        except Exception as e:
            if verbose:
                print(f"Warning: Error reading header from {file_path}: {e}")
    
    if verbose:
        print(f"Found metadata for {len(tensor_metadata)} tensors")
    
    # Handle MoE models with special care for experts
    if is_moe:
        # First, group by structure to better identify patterns
        grouped_params = group_similar_parameters(param_names)
        
        # Then do the condensation with our improved MoE-aware pattern detection
        summary_output = []
        
        for structure, original_names in grouped_params.items():
            # Use the new MoE-aware pattern recognition
            condensed_names, _ = replace_numeric_patterns(original_names)
            
            if not condensed_names:
                # Fallback if condensation somehow fails (shouldn't happen)
                condensed_names = original_names
            
            for condensed_name in condensed_names:
                # Use the first original name as a representative to find metadata
                representative_name = original_names[0] if original_names else None
                
                if representative_name:
                    # Get shape and dtype
                    shape, dtype = _get_tensor_metadata(
                        representative_name, 
                        model_dir, 
                        param_file_map, 
                        tensor_metadata, 
                        model_type
                    )
                    
                    # Format output string
                    output_str = condensed_name
                    if shape:
                        shape_str = f"[{','.join(map(str, shape))}]"
                        output_str += f",{shape_str}"
                    # Always include dtype if available
                    if dtype:
                        output_str += f",{dtype}"
                    
                    summary_output.append(output_str)
                else:
                    # If no representative name found, add condensed name without metadata
                    summary_output.append(condensed_name)
    else:
        # For non-MoE models, use the original approach
        grouped_params = group_similar_parameters(param_names)
        summary_output = []
        
        for structure, original_names in grouped_params.items():
            condensed_names, _ = replace_numeric_patterns(original_names)
            
            if not condensed_names:
                # Fallback if condensation fails
                condensed_names = original_names
            
            for condensed_name in condensed_names:
                representative_name = original_names[0] if original_names else None
                
                if representative_name:
                    shape, dtype = _get_tensor_metadata(
                        representative_name, 
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
    
    # For better readability, sort the output
    return sorted(summary_output)
