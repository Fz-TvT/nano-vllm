import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open
import re
import torch.distributed as dist

def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)

def load_model(model: nn.Module, path: str):
    tp_rank = dist.get_rank()
    tp_size = dist.get_world_size()
    
    # 1. 计算当前 Rank 负责的专家范围
    num_experts = 60 # Qwen1.5-MoE 的总专家数
    experts_per_rank = num_experts // tp_size
    start_expert = tp_rank * experts_per_rank
    end_expert = start_expert + experts_per_rank
    
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            # 必须遍历文件中的所有权重名字
            for weight_name in f.keys():        
                target_name = weight_name
                
                # ==========================================
                # 核心修复 1：拦截专家权重，做全局到局部的映射
                # ==========================================
                if "mlp.experts." in target_name:
                    # 提取全局专家 ID (例如 mlp.experts.35 -> 35)
                    match = re.search(r"mlp\.experts\.(\d+)\.", target_name)
                    if match:
                        global_idx = int(match.group(1))
                        # 如果该专家不在本卡负责范围内，直接跳过不加载
                        if not (start_expert <= global_idx < end_expert):
                            continue
                        
                        # 翻译为本地索引 (例如 GPU 1 负责 30-59，那么 35 号就是本地的 5 号)
                        local_idx = global_idx - start_expert
                        target_name = target_name.replace(f"mlp.experts.{global_idx}.", f"mlp.experts.{local_idx}.")
                
                # ==========================================
                # 核心修复 2：处理合并权重 (Packed Weights)
                # ==========================================
                matched_packed = False
                for k in packed_modules_mapping:
                    if k in target_name:
                        matched_packed = True
                        v, shard_id = packed_modules_mapping[k]
                        param_name = target_name.replace(k, v)
                        
                        try:
                            param = model.get_parameter(param_name)
                            # 获取这个参数所属的模块，方便下面去模块上找 weight_loader
                            module_name = param_name.rsplit('.', 1)[0]
                            mod = model.get_submodule(module_name)
                        except (AttributeError, KeyError):
                            break
                        
                        # 兼容查找：先在 param 上找，找不到去所在的 module 找
                        weight_loader = getattr(param, "weight_loader", getattr(mod, "weight_loader", None))
                        
                        if weight_loader is not None:
                            weight_loader(param, f.get_tensor(weight_name), shard_id)
                        else:
                            raise NotImplementedError(f"模块 {module_name} 必须实现 weight_loader 方法！")
                        break
                
                # ==========================================
                # 处理普通权重的加载
                # ==========================================
                if not matched_packed:
                    try:
                        param = model.get_parameter(target_name)
                    except (AttributeError, KeyError):
                        continue
                    
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    weight_loader(param, f.get_tensor(weight_name))