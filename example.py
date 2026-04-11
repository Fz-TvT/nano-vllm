import os
from nanovllm import LLM, SamplingParams
from transformers import AutoTokenizer


def main():
    path = os.path.expanduser("/home/cfz/nano-vllm/~/huggingface/Qwen1.5-MoE-A2.7B")
    tokenizer = AutoTokenizer.from_pretrained(path)
    llm = LLM(path, enforce_eager=True, tensor_parallel_size=4)
    sampling_params = SamplingParams(temperature=0.01, max_tokens=300)
    prompts = [
        "who are you?",
        "what's the capital of France?",
        "list all prime numbers within 100",
    ]

    # Use a stable instruction prefix and only apply chat template when available.
    if getattr(tokenizer, "chat_template", None):
        prompts = [
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": "You are a helpful assistant. Answer briefly and accurately."},
                    {"role": "user", "content": prompt},
                ],
                tokenize=False,
                add_generation_prompt=False,
            )
            for prompt in prompts
        ]
    outputs = llm.generate(prompts, sampling_params)

    for prompt, output in zip(prompts, outputs):
        print("\n")
        print(f"Prompt: {prompt!r}")
        print(f"Completion: {output['text']!r}")


if __name__ == "__main__":
    main()
