# dolphin-summarize

This script analyzes the `model.safetensors.index.json` file within a model directory to generate a condensed summary of the model's architecture. It groups similar parameter names using range notation (e.g., `model.layers.[0-39].mlp.down_proj.weight`) and displays the shape and data type (precision) for each parameter group.

## Dependencies

*   Python 3
*   `safetensors` (Optional, but recommended for full shape extraction capabilities):
    ```bash
    pip install safetensors
    ```

## Usage

```bash
python hf_model_summary.py [MODEL_DIRECTORY_PATH] [OPTIONS]
```

**Arguments:**

*   `MODEL_DIRECTORY_PATH`: (Required) Path to the directory containing the model files (specifically, the `model.safetensors.index.json` or similar index file). Defaults to the current directory (`.`) if not provided.

**Options:**

*   `--output OUTPUT`, `-o OUTPUT`: Path to an output file where the summary will be written (optional).
*   `--verbose`, `-v`: Show verbose output during processing (optional).

## Example

```bash
python hf_model_summary.py ~/models/my_llama_model --verbose
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
