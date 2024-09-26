import time
from litellm import batch_completion
from typing import Callable
from tqdm import tqdm


class LitellmModel:

    def __init__(
        self,
        model_id: str,
        parse_generation_fn: Callable[[list[str]], str] = lambda responses: [
            response["choices"][0]["message"]["content"] for response in responses
        ],
        max_send_messages=25,
        generate_kwargs: dict = {"temperature": 0.0},
    ):

        self.model_id = model_id
        self.parse_generation_fn = parse_generation_fn
        self.max_send_messages = max_send_messages
        self.generate_kwargs = generate_kwargs

    @staticmethod
    def format_message(message: str | list[str], system_prompt: str = ""):
        """
        If user input is a string, it is formatted as a single-turn conversation.
        If it is a list, it is formatted as a multi-turn conversation between user and assistant.
        """
        if isinstance(message, str):
            message = [message]
        if system_prompt:
            conv = [{"role": "system", "content": system_prompt}]
        else:
            conv = []
        for i, m in enumerate(message):
            if type(m) == list and len(m) == 1:
                m = m[0]
            conv.append({"role": "user" if i % 2 == 0 else "assistant", "content": m})
        return conv

    def batch_call(
        self, prompts: str | list[str], system_prompt: str = "", **kwargs
    ) -> list[str]:
        generations = []
        if isinstance(prompts, str):
            prompts = [prompts]
        prompts: list[list[dict[str, str]]] = [
            self.format_message(prompt, system_prompt) for prompt in prompts
        ]

        if not system_prompt:
            example_input = prompts[0][0]["content"]
        else:
            example_input = (
                f"SYS:{prompts[0][0]['content']}\nUSER:{prompts[0][1]['content']}"
            )
        print(f"Example model input:\n{example_input}")
        for i in tqdm(
            range(0, len(prompts), self.max_send_messages),
            desc=f"Processing batched requests (max bs={self.max_send_messages})",
        ):
            batch_of_messages = prompts[i : i + self.max_send_messages]
            assert isinstance(batch_of_messages, list) and all(
                isinstance(item, list)
                and all(
                    isinstance(d, dict)
                    and all(
                        isinstance(k, str) and isinstance(v, str) for k, v in d.items()
                    )
                    for d in item
                )
                for item in batch_of_messages
            ), "batch_of_messages must be of type list[list[dict[str, str]]]"

            responses = batch_completion(
                model=self.model_id,
                messages=batch_of_messages,
                **self.generate_kwargs,
                **kwargs,
            )
            # wait 1 second
            time.sleep(1)
            generations.extend(responses)
        return self.parse_generation_fn(generations)
