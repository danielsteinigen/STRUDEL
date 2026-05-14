import json
import os
from abc import ABC, abstractmethod

import yaml
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncodingName,
    Message,
    Role,
    SystemContent,
    load_harmony_encoding,
)
from tqdm import tqdm
from vllm import LLM, SamplingParams, TokensPrompt


class DataGenerator(ABC):
    def __init__(
        self,
        model_config: str,
        output_path: str,
        input_path: str = None,
        max_samples: int = None,
        data_batch_size: int = 10000,
        start_index: int = None,
        end_index: int = None,
        use_harmony: bool = False,
    ):
        self.config = self._load_config(model_config)
        self.input_path = input_path
        self.output_path = output_path
        self.max_samples = max_samples
        self.data_batch_size = data_batch_size
        self.start_index = start_index
        self.end_index = end_index
        self.use_harmony = use_harmony

        self._setup_dataset()
        self._setup_engine()
        self._setup_sampling()
        self._tokenizer = self.llm.get_tokenizer()

        os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
        self.output_file = open(self.output_path, "w", encoding="utf-8")

    def _load_config(self, path: str) -> dict:
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def _setup_engine(self):
        if self.use_harmony:
            self.harmony_encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
        engine_args = self.config["engine"]
        self.llm = LLM(**{k: v for k, v in engine_args.items() if v is not None})

    def _setup_sampling(self):
        self.sampling_params = SamplingParams(
            **self.config["sampling"],
            **({"stop_token_ids": self.harmony_encoding.stop_tokens_for_assistant_actions()} if self.use_harmony else {}),
        )

    def _truncate(self, text: str, max_length: int) -> str:
        """Truncates the text according to the number of characters using the tokenizer."""
        return self._tokenizer.decode(
            self._tokenizer.encode(
                text,
                add_special_tokens=False,
                max_length=max_length,
                truncation=True,
            )
        )

    @abstractmethod
    def _setup_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def format_prompt(self, sample: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def get_system_prompt(self, sample: dict) -> str:
        raise NotImplementedError

    @abstractmethod
    def format_output(self, sample: dict, response: str, finish_reason: str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def post_process(self) -> str:
        raise NotImplementedError

    def _process_batch(self, batch):
        if not self.use_harmony:
            if "use_completion" not in self.config or not self.config["use_completion"]:
                if not "system_message" in self.config:
                    messages = [
                        [
                            {"role": "system", "content": self.get_system_prompt(item["original"])},
                            {"role": "user", "content": item["prompt"]},
                        ]
                        for item in batch
                    ]
                elif self.config["system_message"]:
                    messages = [
                        [
                            {"role": "system", "content": self.config["system_message"]},
                            {"role": "user", "content": f"{self.get_system_prompt(item['original'])}\n{item['prompt']}"},
                        ]
                        for item in batch
                    ]
                else:
                    messages = [
                        [{"role": "user", "content": f"{self.get_system_prompt(item['original'])}\n{item['prompt']}"}] for item in batch
                    ]

                results = self.llm.chat(messages=messages, sampling_params=self.sampling_params)
            else:
                prompts = [
                    self.config["system_message"].format(prompt=f"{self.get_system_prompt(item['original'])}\n{item['prompt']}")
                    for item in batch
                ]
                results = self.llm.generate(prompts=prompts, sampling_params=self.sampling_params)

            for item, result in zip(batch, results):
                formatted = self.format_output(item, result.outputs[0].text, result.outputs[0].finish_reason)
                self.output_file.write(json.dumps(formatted, ensure_ascii=False) + "\n")
        else:
            messages = [
                Conversation.from_messages(
                    [
                        Message.from_role_and_content(Role.SYSTEM, SystemContent.new()),
                        Message.from_role_and_content(
                            Role.DEVELOPER, DeveloperContent.new().with_instructions(self.get_system_prompt(item["original"]))
                        ),
                        Message.from_role_and_content(Role.USER, item["prompt"]),
                    ]
                )
                for item in batch
            ]
            results = self.llm.generate(
                [
                    TokensPrompt(prompt_token_ids=self.harmony_encoding.render_conversation_for_completion(conv, Role.ASSISTANT))
                    for conv in messages
                ],
                sampling_params=self.sampling_params,
            )
            for item, result in zip(batch, results):
                output = result.outputs[0]

                try:
                    token_ids = output.token_ids
                    for i in range(len(token_ids) - 1):
                        if token_ids[i] == 200007 and token_ids[i + 1] == 200002:
                            token_ids[i + 1] = 200006
                    token_ids_2 = []
                    prev = None
                    for tok_item in token_ids:
                        if tok_item == 200006 and prev == 200006:
                            continue
                        token_ids_2.append(tok_item)
                        prev = tok_item
                    entries = self.harmony_encoding.parse_messages_from_completion_tokens(token_ids_2, Role.ASSISTANT)
                    final_ent = [ent for ent in entries if ent.channel == "final"]
                    result = final_ent[0].content[0].text if final_ent else ""
                except Exception as e:
                    print(e)
                    result = output.text

                formatted = self.format_output(item, result, output.finish_reason)
                self.output_file.write(json.dumps(formatted, ensure_ascii=False) + "\n")

    def run(self):
        batch, count = [], 0
        if self.start_index and self.end_index and self.end_index > self.start_index:
            self.dataset = self.dataset.skip(self.start_index).take(self.end_index - self.start_index)
        elif self.start_index:
            self.dataset = self.dataset.skip(self.start_index)
        for sample in tqdm(self.dataset, total=self.max_samples if self.max_samples else 1000000, desc="Processing samples"):

            prompt = self.format_prompt(sample)
            if not prompt:
                continue

            batch.append({"prompt": prompt, "original": sample})

            if len(batch) >= self.data_batch_size:
                self._process_batch(batch)
                batch.clear()

            count += 1
            if self.max_samples and count >= self.max_samples:
                break

        if batch:
            self._process_batch(batch)

        self.output_file.close()
        print(f"✅ Done! Output saved to: {self.output_path}")
