# dolphin-utils

A collection of utilities for AI/ML model analysis and processing.

## dolphin-summarize

This tool analyzes safetensors model files to generate a condensed summary of the model's architecture. It groups similar parameter names using range notation (e.g., `model.layers.[0-39].mlp.down_proj.weight`) and displays the shape and data type (precision) for each parameter group.

**Key Features:**
- **Remote Processing**: Analyze Hugging Face models without downloading the full model files (downloads only headers - KB instead of GB)
- **Local Processing**: Works with locally stored model directories
- **Efficient**: Uses HTTP range requests and streaming to minimize data transfer
- **Reliable**: Multiple fallback strategies ensure 100% compatibility

## Dependencies

*   Python 3.7+
*   `huggingface_hub` - For accessing Hugging Face repositories
*   `requests` - For HTTP requests and remote file access
*   `safetensors` - For reading safetensors file headers

All dependencies are automatically installed when you install the package:
```bash
pip install dolphin-utils
```

## Usage

After installing the package:
```bash
pip install dolphin-utils
```

You can use the tool in two ways:

**Via CLI command:**
```bash
dolphin-summarize [MODEL_PATH_OR_REPO_ID] [OPTIONS]
```

**Via Python module:**
```bash
python -m dolphin_summarize [MODEL_PATH_OR_REPO_ID] [OPTIONS]
```

**Arguments:**

*   `MODEL_PATH_OR_REPO_ID`: 
    - **Local path**: Directory containing safetensors files (e.g., `~/models/my_llama_model`)
    - **Hugging Face repo ID**: Repository identifier (e.g., `microsoft/DialoGPT-medium`)
    - Defaults to current directory (`.`) if not provided

**Options:**

*   `--output OUTPUT`, `-o OUTPUT`: Path to an output file where the summary will be written (optional).
*   `--verbose`, `-v`: Show verbose output during processing (optional).

## Examples

**Remote Processing (Hugging Face Hub):**
```bash
# Process a model directly from Hugging Face without downloading
python -m dolphin_summarize microsoft/DialoGPT-medium --verbose

# Large models work too - only headers are downloaded
python -m dolphin_summarize meta-llama/Llama-2-70b-hf --verbose
```

**Local Processing:**
```bash
# Process a local model directory
python -m dolphin_summarize ~/models/my_llama_model --verbose

# Process current directory
python -m dolphin_summarize . --verbose
```

## Output Format

The script prints the summary to the console (and optionally to a file). Each line represents a parameter or a group of parameters with a similar structure:

```
parameter_name,[shape],dtype
```

**Example Output Lines:**

```
lm_head.weight,[131072,5120],BF16
model.embed_tokens.weight,[131072,5120],BF16
model.layers.[0-39].input_layernorm.weight,[5120],BF16
model.layers.[0-39].mlp.down_proj.weight,[5120,13824],BF16
model.layers.[0-39].mlp.gate_proj.weight,[13824,5120],BF16
model.layers.[0-39].mlp.up_proj.weight,[13824,5120],BF16
model.layers.[0-39].post_attention_layernorm.weight,[5120],BF16
model.layers.[0-39].self_attn.k_proj.weight,[512,5120],BF16
model.layers.[0-39].self_attn.o_proj.weight,[5120,8192],BF16
model.layers.[0-39].self_attn.q_proj.weight,[8192,5120],BF16
model.layers.[0-39].self_attn.v_proj.weight,[512,5120],BF16
model.norm.weight,[5120],BF16
