from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Any

import yaml

DEFAULT_SYSTEM_PROMPT = (
    "You are an AI assistant expert specialized in understanding and interpreting visualizations. "
    "Your task is to analyze the provided structured image and respond to queries with correct answers."
)


def load_eval_config(path: str) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)

    if not isinstance(config, dict):
        raise ValueError(f"Invalid config file: {path}")

    if "engine" not in config or "sampling" not in config or "generation" not in config:
        raise ValueError(f"Config must define 'engine', 'sampling', and 'generation': {path}")

    return config


def filter_none(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: filter_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [filter_none(item) for item in value]
    return value


def resolve_generation_api(config: dict[str, Any]) -> str:
    generation = config["generation"]
    api = generation.get("api")
    if api:
        return api

    builder = generation.get("request_builder", "template_string")
    return "chat" if builder == "chat_messages" else "generate"


def needs_tokenizer(config: dict[str, Any]) -> bool:
    return config["generation"].get("request_builder") == "tokenizer_chat_template"


def load_tokenizer(model_name: str, trust_remote_code: bool = False):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name, trust_remote_code=trust_remote_code)


def ensure_parent_dir(path: str) -> None:
    parent = Path(path).expanduser().resolve().parent
    parent.mkdir(parents=True, exist_ok=True)


def _fill_placeholders(value: Any, *, system_prompt: str, user_prompt: str) -> Any:
    if isinstance(value, str):
        return value.format(system=system_prompt, user=user_prompt)
    if isinstance(value, list):
        return [_fill_placeholders(item, system_prompt=system_prompt, user_prompt=user_prompt) for item in value]
    if isinstance(value, dict):
        return {key: _fill_placeholders(item, system_prompt=system_prompt, user_prompt=user_prompt) for key, item in value.items()}
    return value


def _encode_image(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def _materialize_chat_content(content: Any, image, image_input: str) -> Any:
    if isinstance(content, list):
        return [_materialize_chat_content(item, image, image_input) for item in content]

    if isinstance(content, dict) and content.get("type") == "image":
        if image_input == "image_url":
            return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_encode_image(image)}"}}
        if image_input == "image_pil":
            return {"type": "image_pil", "image_pil": image}
        raise ValueError(f"Unsupported chat image input mode: {image_input}")

    if isinstance(content, dict):
        return {key: _materialize_chat_content(item, image, image_input) for key, item in content.items()}

    return content


def build_request(
    *,
    config: dict[str, Any],
    system_prompt: str,
    user_prompt: str,
    image,
    tokenizer=None,
) -> Any:
    generation = config["generation"]
    builder = generation.get("request_builder", "template_string")

    if builder == "template_string":
        template = generation.get("template")
        if not template:
            raise ValueError("template_string request_builder requires generation.template")

        return {
            "prompt": _fill_placeholders(template, system_prompt=system_prompt, user_prompt=user_prompt),
            "multi_modal_data": {"image": image},
        }

    if builder == "tokenizer_chat_template":
        if tokenizer is None:
            raise ValueError("tokenizer_chat_template request_builder requires a tokenizer")

        messages = _fill_placeholders(generation.get("messages", []), system_prompt=system_prompt, user_prompt=user_prompt)
        return {
            "prompt": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True),
            "multi_modal_data": {"image": image},
        }

    if builder == "chat_messages":
        messages = _fill_placeholders(generation.get("messages", []), system_prompt=system_prompt, user_prompt=user_prompt)
        return _materialize_chat_content(messages, image, generation.get("image_input", "image_pil"))

    raise ValueError(f"Unsupported request_builder: {builder}")