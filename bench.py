import os
import json
import time
import torch
import numpy as np
from transformers import AutoTokenizer
from nanovllm import LLM, SamplingParams

def main():
    model_path = os.path.expanduser("/home/cfz/nano-vllm/~/huggingface/Qwen1.5-MoE-A2.7B")
    dataset_path = "ShareGPT_V3_unfiltered_cleaned_split.json" 
    num_requests = 2000  
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
    
    torch.cuda.synchronize() 
    end_time = time.perf_counter()

    # ==========================================
    # 🌟 新增：精准解析 TTFT 和 TPOT
    # ==========================================
    total_output_tokens = 0
    ttft_list = []
    tpot_list = []

    for out in outputs:
        num_tokens = len(out["token_ids"])
        total_output_tokens += num_tokens

        # 计算 TTFT (首字延迟): first_token_time - arrival_time
        if out.get("first_token_time") is not None and out.get("arrival_time") is not None:
            ttft = out["first_token_time"] - out["arrival_time"]
            ttft_list.append(ttft)

        # 计算 TPOT (每字延迟): (finish_time - first_token_time) / (num_tokens - 1)
        if (
            num_tokens > 1
            and out.get("finish_time") is not None
            and out.get("first_token_time") is not None
        ):
            tpot = (out["finish_time"] - out["first_token_time"]) / (num_tokens - 1)
            tpot_list.append(tpot)

    # 总体吞吐量计算
    elapsed_time = end_time - start_time
    throughput = total_output_tokens / elapsed_time

    # 统计 P90 和 平均值 (转换为毫秒)
    avg_ttft = np.mean(ttft_list) * 1000 if ttft_list else 0
    p90_ttft = np.percentile(ttft_list, 90) * 1000 if ttft_list else 0
    
    avg_tpot = np.mean(tpot_list) * 1000 if tpot_list else 0
    p90_tpot = np.percentile(tpot_list, 90) * 1000 if tpot_list else 0

    print("\n" + "="*50)
    print("🎯 NanoVLLM Benchmark Results")
    print("="*50)
    print(f"Model            : Qwen1.5-MoE-A2.7B")
    print(f"Tensor Parallel  : {tp_size}")
    print(f"Requests         : {num_requests}")
    print(f"Total Time       : {elapsed_time:.2f} s")
    print(f"Total Out Tokens : {total_output_tokens} tok")
    print(f"Throughput       : {throughput:.2f} tokens/s")
    print("-" * 50)
    print(f"🚀 TTFT (首字延迟)  - Avg: {avg_ttft:.2f} ms | P90: {p90_ttft:.2f} ms")
    print(f"⚡ TPOT (每字延迟)  - Avg: {avg_tpot:.2f} ms | P90: {p90_tpot:.2f} ms")
    print("="*50)

if __name__ == "__main__":
    main()