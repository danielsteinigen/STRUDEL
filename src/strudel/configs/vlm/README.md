# VLM Evaluation Configs

These YAML files define STRUDEL evaluation runs for VLM families.

Each config controls:
- `engine`: vLLM model loading arguments
- `sampling`: generation parameters
- `generation`: prompt construction strategy and whether the run uses `generate` or `chat`

Typical usage:

```bash
strudel eval strudel \
    --config src/strudel/configs/vlm/qwen2.5-vl.yaml \
    --output results/qwen2.5-vl
```

You can override the configured model path and tensor parallel size at runtime with `--model-name-or-path` and `--tp-size`.