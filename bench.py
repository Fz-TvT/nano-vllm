import os
import json
import time
import torch
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

def main():
    model_path = os.path.expanduser("/home/cfz/nano-vllm/~/huggingface/Qwen1.5-MoE-A2.7B")
    dataset_path = "ShareGPT_V3_unfiltered_cleaned_split.json" 
    num_requests = 500  
    tp_size = 4        

    print(f"Initializing NanoVLLM engine with TP={tp_size}...")
    llm = LLM(model_path, enforce_eager=False, max_model_len=4096, tensor_parallel_size=tp_size)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    print("Loading ShareGPT dataset")
    with open(dataset_path, "r", encoding="utf-8") as f:
        dataset = json.load(f)

    prompt_token_ids = []
    for data in dataset:
        conversations = data.get("conversations", [])
        if len(conversations) > 0:
            text = conversations[0].get("value", "")
            if text:
                token_ids = tokenizer.encode(text)
                # 过滤异常长度：太短没意义，太长可能 OOM
                if 20 < len(token_ids) < 1024:
                    prompt_token_ids.append(token_ids)
        if len(prompt_token_ids) >= num_requests:
            break

    print(f"Successfully loaded {len(prompt_token_ids)} prompts.")

    # 构建采样参数 (注意：这里 ignore_eos=False，模拟真实的提前结束)
    sampling_params = [
        SamplingParams(temperature=0.3, ignore_eos=False, max_tokens=256) 
        for _ in range(len(prompt_token_ids))
    ]


    #预热 (Warmup) - 消除 GPU 唤醒延迟

    print("Warming up GPU...")
    warmup_prompts = prompt_token_ids[:3]
    warmup_params = [SamplingParams(max_tokens=10)] * 3
    llm.generate(warmup_prompts, warmup_params, use_tqdm=False)
    torch.cuda.synchronize() # 强制等待 GPU 真正跑完

    print(f"Starting Benchmark (Generating {num_requests} sequences)...")
    start_time = time.perf_counter()
    
    outputs = llm.generate(prompt_token_ids, sampling_params, use_tqdm=True)
    
    torch.cuda.synchronize() # 确保所有请求都已处理完毕
    end_time = time.perf_counter()
    total_output_tokens = sum(len(out["token_ids"]) for out in outputs)
    elapsed_time = end_time - start_time
    throughput = total_output_tokens / elapsed_time

    print("\n" + "="*50)
    print("🎯 NanoVLLM Benchmark Results")
    print("="*50)
    print(f"Model            : Qwen1.5-MoE-A2.7B")
    print(f"Tensor Parallel  : {tp_size}")
    print(f"Requests         : {num_requests}")
    print(f"Total Time       : {elapsed_time:.2f} s")
    print(f"Total Out Tokens : {total_output_tokens} tok")
    print(f"Throughput       : {throughput:.2f} tokens/s")
    print("="*50)

if __name__ == "__main__":
    main()
