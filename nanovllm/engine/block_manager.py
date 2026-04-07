from collections import OrderedDict
import xxhash
import numpy as np

from nanovllm.engine.sequence import Sequence

class Block:
    def __init__(self, block_id):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []

    def update_hash(self, hash: int):
        self.hash = hash

    def reset_hash(self):
        self.hash = -1

    def reset(self):
        # 重置块用于新的分配。
        # 不在这里重置 hash，hash 的清除在 _maybe_evict_cached_block 中完成
        self.ref_count = 1
        self.token_ids = []


class BlockManager:
    def __init__(self, num_blocks: int, block_size: int):
        self.block_size = block_size
        self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        self.hash_to_block_id: dict[int, int] = dict()
        
        # OrderedDict 完美模拟 vLLM 的双向链表 (free_block_queue)：
        # -> 迭代的首个元素 (popitem(last=False)) 是最久未使用的块 (LRU)
        # -> 新插入的元素会默认放在尾部 (MRU)
        self.free_block_ids: OrderedDict[int, None] = OrderedDict((i, None) for i in range(num_blocks))
        self.used_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[int], prefix_hash: int = -1):
        h = xxhash.xxh64()
        if prefix_hash != -1:
            h.update(prefix_hash.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _maybe_evict_cached_block(self, block: Block):
        """vLLM 延迟驱逐 (Lazy Eviction)：真正被重新分配时，才抹除它的前缀记忆"""
        if block.hash != -1:
            # 只有当映射表里的 block_id 还是自己时才删除（防覆盖）
            if self.hash_to_block_id.get(block.hash) == block.block_id:
                del self.hash_to_block_id[block.hash]
            block.reset_hash()

    def _get_new_block(self) -> Block:
        assert self.free_block_ids, "Out of KV Cache Memory!"
        
        # 核心：从 OrderedDict 头部弹出，这里就是排队最久的 LRU 老块
        block_id, _ = self.free_block_ids.popitem(last=False)
        block = self.blocks[block_id]
        assert block.ref_count == 0
        
        # 强制洗脑（驱逐旧哈希），赋予新生
        self._maybe_evict_cached_block(block)
        block.reset()
        
        self.used_block_ids.add(block_id)
        return block

    def _cache_full_block(self, block: Block, h: int):
        block.update_hash(h)
        self.hash_to_block_id[h] = block.block_id

    def _touch(self, block_id: int) -> Block:
        """vLLM 缓存命中 (Touch)：从空闲队列中拯救块，或者增加引用"""
        block = self.blocks[block_id]
        if block.ref_count == 0:
            assert block_id in self.free_block_ids
            del self.free_block_ids[block_id] # 摘除空闲状态
            self.used_block_ids.add(block_id)
        block.ref_count += 1
        return block

    def _free_block(self, block_id: int) -> None:
        """vLLM 释放逻辑：重排到 LRU 队列尾部，保留哈希等待复用"""
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        # 对 OrderedDict 赋值会将该 key 放置到队尾 (MRU)，变为最新鲜的空闲块
        self.free_block_ids[block_id] = None

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def allocate(self, seq: Sequence):
        """Prefill 阶段分配：带断路器的前缀匹配"""
        assert not seq.block_table
        h = -1
        cache_miss = False  # 断路器状态
        
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            is_full = len(token_ids) == self.block_size
            
            # 只要块满都会计算哈希，为了供后续请求使用
            h = self.compute_hash(token_ids, h) if is_full else -1

            if not cache_miss:
                block_id = self.hash_to_block_id.get(h, -1) if h != -1 else -1
                # 必须二次校验 tokens 内容，防止 hash 冲突
                if block_id != -1 and self.blocks[block_id].token_ids == token_ids:
                    seq.num_cached_tokens += len(token_ids)
                    block = self._touch(block_id)
                    seq.block_table.append(block_id)
                    continue
                else:
                    # 一旦某个 Block 未命中，后续所有 Block 必然 Miss
                    cache_miss = True

            # 触发 Miss，走常规新分配流程
            block = self._get_new_block()
            block.token_ids = list(token_ids) # 必须同步 Tokens 数据
            seq.block_table.append(block.block_id)
            
            # 及时缓存：即使刚才 miss 了，当前生成的满块也要立刻注册进缓存
            if h != -1:
                self._cache_full_block(block, h)

    def deallocate(self, seq: Sequence):
        # 极其巧妙的细节：reversed 释放！
        # 倒序释放会让 sequence 的 Suffix（末尾块）先进入 free 队列，
        # Prefix（前缀块）后进入 free 队列。
        # 因此 Prefix 会排在队尾，Suffix 排在队头。
        # 结果：不常复用的 Suffix 被优先驱逐，高频复用的 Prefix 活得最久
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._free_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (1 if len(seq) % self.block_size == 1 else 0)

    def may_append(self, seq: Sequence):
        """Decode 阶段：追加 Token 与 Copy-on-Write 机制"""
        block_table = seq.block_table
        last_block_id = block_table[-1]
        last_block = self.blocks[last_block_id]

        if len(seq) % self.block_size == 1:
            # 需要跨入全新块
            assert last_block.hash != -1 # 上一个块必定已经满了并被缓存了
            new_block = self._get_new_block()
            new_block.token_ids = list(seq.block(seq.num_blocks - 1))
            block_table.append(new_block.block_id)
            
        else:
            # 追加到现有块 -> 必须进行写时复制 (Copy-on-Write) 检查
            if last_block.ref_count > 1:
                # 危险！这个块正被其他 Sequence 共享 (多为 Prefix 阶段遗留的)
                # 分叉 (Fork)：当前请求退出共享，复制一份独立把玩
                last_block.ref_count -= 1 
                new_block = self._get_new_block()
                new_block.token_ids = list(last_block.token_ids)
                block_table[-1] = new_block.block_id
                last_block = new_block # 视角切到新的独立块
                
            # 安全地写入新 token 序列了
            token_ids = seq.block(seq.num_blocks - 1)
            last_block.token_ids = list(token_ids)
            
            # 检查这个块是不是刚刚被塞满
            if len(seq) % self.block_size == 0:
                assert last_block.hash == -1
                prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
                h = self.compute_hash(token_ids, prefix)
                self._cache_full_block(last_block, h)