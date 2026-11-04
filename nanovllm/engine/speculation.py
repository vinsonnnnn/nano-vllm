"""Greedy verification with snapshot/restore/replay of hybrid states.

A drafter-neutral transaction prototype. Real MTP requires trained, compatible
checkpoint weights; a config field is not a usable draft model.
"""
from dataclasses import dataclass
import torch


@dataclass
class VerificationResult:
    accepted: list[int]
    next_token: int
    proposed: int


class GreedySpeculator:
    def verify(self, cache, next_logits, candidates, consume, *, temperature=0,
               max_tokens=None, eos_token_id=None):
        if temperature != 0:
            raise NotImplementedError("speculative prototype supports greedy verification only")
        candidates = list(candidates)
        if max_tokens is not None:
            if max_tokens < 1:
                raise ValueError("max_tokens must be positive")
            candidates = candidates[:max_tokens - 1]
        snapshot = cache.snapshot()
        was_frozen = cache.frozen
        accepted, logits = [], next_logits
        cache.frozen = True
        try:
            for candidate in candidates:
                predicted = int(torch.as_tensor(logits).reshape(-1).argmax())
                if candidate != predicted:
                    cache.restore(snapshot)
                    cache.frozen = True
                    for token in accepted:
                        consume(token, cache)
                    return VerificationResult(accepted, predicted, len(candidates))
                if candidate == eos_token_id:
                    return VerificationResult(accepted, candidate, len(candidates))
                accepted.append(candidate)
                logits = consume(candidate, cache)
            return VerificationResult(accepted, int(torch.as_tensor(logits).reshape(-1).argmax()), len(candidates))
        except Exception:
            cache.restore(snapshot)
            raise
        finally:
            cache.frozen = was_frozen
