from collections import deque

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.engine.block_manager import BlockManager


class Scheduler:

    def __init__(self, config: Config):
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        self.mlfq_levels = config.mlfq_levels
        self.mlfq_base_quantum = config.mlfq_base_quantum
        self.block_manager = BlockManager(config.num_kvcache_blocks, config.kvcache_block_size)
        self.waiting: deque[Sequence] = deque()
        self.decode_queues: list[deque[Sequence]] = [deque() for _ in range(self.mlfq_levels)]

    def is_finished(self):
        return not self.waiting and not any(self.decode_queues)

    def add(self, seq: Sequence):
        self.waiting.append(seq)

    def _quantum(self, level: int) -> int:
        return self.mlfq_base_quantum * (2 ** level)

    def _enqueue_decode(self, seq: Sequence):
        self.decode_queues[seq.mlfq_level].append(seq)

    def _remove_decode(self, seq: Sequence):
        for queue in self.decode_queues:
            try:
                queue.remove(seq)
                return
            except ValueError:
                continue

    def _pick_victim_from_mlfq(self, excluded: set[int] | None = None) -> Sequence | None:
        excluded = excluded or set()
        # Preempt from the lowest-priority queue first.
        for level in range(self.mlfq_levels - 1, -1, -1):
            queue = self.decode_queues[level]
            for seq in reversed(queue):
                if seq.seq_id in excluded:
                    continue
                return seq
        return None

    def _ensure_can_append(self, seq: Sequence, excluded: set[int]) -> bool:
        while not self.block_manager.can_append(seq):
            victim = self._pick_victim_from_mlfq(excluded)
            if victim is None:
                self.preempt(seq)
                return False
            excluded.add(victim.seq_id)
            self.preempt(victim)
        return True

    def schedule(self) -> tuple[list[Sequence], bool]:
        # prefill
        scheduled_seqs = []
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            seq.mlfq_level = 0
            seq.decode_steps_in_level = 0
            self.waiting.popleft()
            self._enqueue_decode(seq)
            scheduled_seqs.append(seq)
        if scheduled_seqs:
            return scheduled_seqs, True

        # decode
        excluded: set[int] = set()
        requeue: list[tuple[int, Sequence]] = []
        for level in range(self.mlfq_levels):
            queue = self.decode_queues[level]
            while queue and num_seqs < self.max_num_seqs:
                seq = queue.popleft()
                excluded.add(seq.seq_id)
                if not self._ensure_can_append(seq, excluded):
                    continue
                num_seqs += 1
                self.block_manager.may_append(seq)
                seq.decode_steps_in_level += 1
                next_level = seq.mlfq_level
                if seq.decode_steps_in_level >= self._quantum(level):
                    next_level = min(level + 1, self.mlfq_levels - 1)
                    seq.decode_steps_in_level = 0
                scheduled_seqs.append(seq)
                requeue.append((next_level, seq))
        assert scheduled_seqs
        for level, seq in requeue:
            seq.mlfq_level = level
            self._enqueue_decode(seq)
        return scheduled_seqs, False

    def preempt(self, seq: Sequence):
        self._remove_decode(seq)
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        seq.mlfq_level = 0
        seq.decode_steps_in_level = 0
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self._remove_decode(seq)
