"""One GPU owner, exact-length batching, and bounded shared-prefix snapshots."""

import threading
import time
from collections import OrderedDict, defaultdict
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten
from mlx_lm import load
from mlx_lm.models.cache import make_prompt_cache
from transformers import AutoTokenizer

from .optimizations import SelectedHead, fork_cache, fuse_gate_up
from .protocol import LETTERS, MODEL_ID, MODEL_REVISION, DecisionRequest


@dataclass(frozen=True)
class Row:
    request: int
    key: str
    question: object
    tokens: tuple[int, ...]
    prefix: tuple[int, ...]


def common_prefix(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return min(len(a), len(b))


class Engine:
    def __init__(
        self,
        model=MODEL_ID,
        revision=MODEL_REVISION,
        *,
        max_batch_size=4,
        max_batch_tokens=4096,
        max_prompt_tokens=16384,
        prefill_chunk_size=256,
        prefix_cache_bytes=512 * 1024**2,
        prefix_cache_entries=2,
        selected_head=True,
        prefix_reuse=True,
        fused_gate_up=False,
        rubric_first=False,
    ):
        if min(max_batch_size, max_batch_tokens, max_prompt_tokens, prefill_chunk_size) < 1:
            raise ValueError("batch, prompt, and chunk limits must be positive")
        if prefix_cache_bytes < 0 or prefix_cache_entries < 0:
            raise ValueError("cache limits cannot be negative")
        self.model_id, self.revision = str(model), revision
        if Path(model).is_dir():
            model_path = str(Path(model).resolve())
            # Caller must supply local provenance; don't claim a Hub hash.
            self.revision = None
        else:
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model, revision=revision)
        # Evaluating all checkpoint tensors together can overlap file staging
        # allocations and exhaust a 36 GiB Mac before inference even begins.
        # Materialize the unchanged stored weights one tensor at a time, and
        # do not retain temporary loader allocations in the Metal cache.
        previous_cache_limit = mx.set_cache_limit(0)
        try:
            self.model, _ = load(model_path, lazy=True)
            for _, parameter in tree_flatten(self.model.parameters()):
                mx.eval(parameter)
            mx.clear_cache()
        finally:
            mx.set_cache_limit(previous_cache_limit)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
        self._configure(
            max_batch_size,
            max_batch_tokens,
            max_prompt_tokens,
            prefill_chunk_size,
            prefix_cache_bytes,
            prefix_cache_entries,
            selected_head,
            prefix_reuse,
            rubric_first,
        )
        self.fused_layers = fuse_gate_up(self.model) if fused_gate_up else 0

    def _configure(
        self,
        max_batch_size,
        max_batch_tokens,
        max_prompt_tokens,
        prefill_chunk_size,
        prefix_cache_bytes,
        prefix_cache_entries,
        selected_head,
        prefix_reuse,
        rubric_first=False,
    ):
        if getattr(self.model, "model_type", None) != "qwen3_5":
            raise ValueError("this adapter currently supports dense qwen3_5 text models only")
        self.trunk = self.model.language_model.model
        self.head = self.model.language_model.lm_head
        ids = [self.tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
        if any(len(x) != 1 for x in ids) or len({x[0] for x in ids}) != len(LETTERS):
            raise ValueError("all 52 OpenJEV letters must be distinct single tokens")
        self.letter_ids = [x[0] for x in ids]
        self.selected = SelectedHead(self.head, self.letter_ids)
        self.max_batch_size, self.max_batch_tokens = max_batch_size, max_batch_tokens
        self.max_prompt_tokens, self.prefill_chunk_size = max_prompt_tokens, prefill_chunk_size
        self.prefix_cache_bytes, self.prefix_cache_entries = (
            prefix_cache_bytes,
            prefix_cache_entries,
        )
        self.selected_head, self.prefix_reuse = selected_head, prefix_reuse
        self.rubric_first = rubric_first
        self._prefixes = OrderedDict()
        self._lock = threading.RLock()
        self.fused_layers = 0

    def _tokens(self, content):
        result = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if hasattr(result, "input_ids"):
            result = result.input_ids
        return tuple(result)

    def prepare(self, requests):
        rows = []
        for i, request in enumerate(requests):
            if request.model not in (MODEL_ID, "openjev", self.model_id):
                raise ValueError(f"requested model {request.model!r} is not loaded")
            state = request.state_text()
            state_anchor = None if self.rubric_first else self._tokens(f"State:\n{state}")
            for key, question in request.questions.items():
                anchor = (
                    self._tokens(question.prefix(state, rubric_first=True))
                    if self.rubric_first
                    else state_anchor
                )
                tokens = self._tokens(question.prompt(state, rubric_first=self.rubric_first))
                if len(tokens) > self.max_prompt_tokens:
                    raise ValueError(
                        f"question {key!r} exceeds {self.max_prompt_tokens} prompt tokens"
                    )
                n = min(common_prefix(anchor, tokens), len(tokens) - 1)
                rows.append(Row(i, key, question, tokens, tokens[:n]))
        return rows

    def clear_cache(self):
        with self._lock:
            self._prefixes.clear()
            mx.clear_cache()

    def _prefill(self, tokens, cache):
        for start in range(0, len(tokens), self.prefill_chunk_size):
            hidden = self.trunk(
                mx.array(tokens[start : start + self.prefill_chunk_size])[None], cache=cache
            )
            # Commit recurrent state as well as KV writes, but skip lm_head.
            mx.eval(hidden[:, -1:, :], [layer.state for layer in cache])

    def _prefix(self, tokens):
        if tokens in self._prefixes:
            self._prefixes.move_to_end(tokens)
            return self._prefixes[tokens], True
        cache = make_prompt_cache(self.model)
        self._prefill(tokens, cache)
        size = sum(layer.nbytes for layer in cache)
        if self.prefix_cache_entries and size <= self.prefix_cache_bytes:
            while self._prefixes and (
                len(self._prefixes) >= self.prefix_cache_entries
                or sum(sum(c.nbytes for c in v) for v in self._prefixes.values()) + size
                > self.prefix_cache_bytes
            ):
                self._prefixes.popitem(last=False)
            self._prefixes[tokens] = cache
        return cache, False

    def _score_batch(self, batch, prefix_cache=None):
        start = len(batch[0].prefix) if prefix_cache is not None else 0
        sequences = [r.tokens[start:] for r in batch]
        # Same lengths only: no padding contaminating recurrent state.
        assert len({len(s) for s in sequences}) == 1
        cache = (
            fork_cache(prefix_cache, len(batch))
            if prefix_cache is not None
            else make_prompt_cache(self.model)
        )
        inputs = mx.array(sequences)
        for offset in range(0, inputs.shape[1], self.prefill_chunk_size):
            hidden = self.trunk(inputs[:, offset : offset + self.prefill_chunk_size], cache=cache)
            last = hidden[:, -1, :]
            mx.eval(last, [layer.state for layer in cache])
        scores = (
            self.selected(last) if self.selected_head else self.head(last)[:, self.letter_ids]
        ).astype(mx.float32)
        mx.eval(scores)
        return scores.tolist()

    def raw_scores(self, rows, *, reference=False):
        """Synchronous GPU operation. Returns scores plus observed work counts."""
        with self._lock:
            metrics = {
                "prefix_hits": 0,
                "prefix_misses": 0,
                "prefill_tokens": 0,
                "scored_rows": len(rows),
                "gpu_batches": 0,
                "batch_sizes": [],
            }
            values = {}
            if reference:
                # Full forward on the selected prompt order, not a narrowed-head baseline.
                for row in rows:
                    logits = self.model(mx.array(row.tokens)[None])[0, -1].astype(mx.float32)
                    scores = logits[self.letter_ids]
                    mx.eval(scores)
                    values[(row.request, row.key)] = scores.tolist()
                    metrics["prefill_tokens"] += len(row.tokens)
                    metrics["gpu_batches"] += 1
                    metrics["batch_sizes"].append(1)
                return values, metrics
            groups = defaultdict(list)
            for row in rows:
                groups[row.prefix if self.prefix_reuse else ()].append(row)
            for prefix, group in groups.items():
                cached = None
                if prefix:
                    cached, hit = self._prefix(prefix)
                    metrics["prefix_hits" if hit else "prefix_misses"] += 1
                    if not hit:
                        metrics["prefill_tokens"] += len(prefix)
                buckets = defaultdict(list)
                for row in group:
                    buckets[len(row.tokens) - len(prefix)].append(row)
                for length, bucket in buckets.items():
                    if length > self.max_batch_tokens:
                        capacity = 1  # long prompts are chunked; token budget is a batching target
                    else:
                        capacity = min(self.max_batch_size, max(1, self.max_batch_tokens // length))
                    for offset in range(0, len(bucket), capacity):
                        batch = bucket[offset : offset + capacity]
                        scores = self._score_batch(batch, cached)
                        metrics["prefill_tokens"] += length * len(batch)
                        metrics["gpu_batches"] += 1
                        metrics["batch_sizes"].append(len(batch))
                        for row, score in zip(batch, scores, strict=True):
                            values[(row.request, row.key)] = score
            return values, metrics

    def decide_many(self, requests: list[DecisionRequest]):
        start = time.perf_counter()
        # Tokenizer, lazy arrays and cache all have one owner even for Python callers.
        with self._lock:
            rows = self.prepare(requests)
            raw, metrics = self.raw_scores(rows)
            results = [
                {
                    "model": self.model_id,
                    "revision": self.revision,
                    "prompt_order": "rubric-first" if self.rubric_first else "state-first",
                    "answers": {},
                    "usage": {"input_tokens": 0, "output_tokens": 0},
                }
                for _ in requests
            ]
            for row in rows:
                results[row.request]["answers"][row.key] = row.question.answer(
                    raw[(row.request, row.key)][: len(row.question.options())]
                )
                results[row.request]["usage"]["input_tokens"] += len(row.tokens)
            elapsed = (time.perf_counter() - start) * 1000
            for result in results:
                result["performance"] = {
                    "group_wall_ms": elapsed,
                    "group_requests": len(requests),
                    **metrics,
                }
            return results

    def decide(self, request):
        return self.decide_many([request])[0]
