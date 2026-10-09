"""JevK5 GGUF readout adapted from allebee/jevk5 v0.3.0.

Upstream commit: 6c6522fe5462a05fdb82bceeb0e8c624c11f1517
Upstream files: jevk5/gguf.py and jevk5/prompt.py
License: Apache-2.0. This version is modified for the redqueen adapter.
"""

from __future__ import annotations

import json
import math
import urllib.request
from collections.abc import Callable, Sequence


LETTERS = "ABCDEFGHIJKLMNOP"
TEMPERATURES = {"knockout": 0.77, "tree": 1.0}
SYSTEM = (
    "Apply the supplied criterion to the supplied evidence. Choose exactly one listed option. "
    "Respond with only its uppercase letter, with no explanation or reasoning."
)
CHAT_TEMPLATE = (
    "<|im_start|>system\n{system}<|im_end|>\n"
    "<|im_start|>user\n{user}<|im_end|>\n"
    "<|im_start|>assistant\n<think>\n\n</think>\n\n"
)
MISSING_MARGIN = 2.0


def prompt_text(state: object, criterion: str, options: list[str]) -> str:
    payload = {
        "evidence": state,
        "criterion": criterion,
        "options": [
            {"letter": LETTERS[index], "description": text}
            for index, text in enumerate(options)
        ],
    }
    return CHAT_TEMPLATE.format(
        system=SYSTEM, user=json.dumps(payload, ensure_ascii=False)
    )


def decision_options(question: dict) -> list[tuple[str, str]]:
    criteria = question.get("criteria")
    if question["type"] == "noul":
        pairs = [
            (key, (criteria or {}).get(key) or f"The proposition is {key}.")
            for key in ("true", "false")
        ]
    elif question["type"] == "choice":
        if isinstance(criteria, list):
            criteria = dict.fromkeys(criteria)
        pairs = [(key, value or key) for key, value in criteria.items()]
    else:
        pairs = [(str(index), level) for index, level in enumerate(criteria)]
    return [(key, f"{key}: {description}") for key, description in pairs]


Reader = Callable[[list[str]], Sequence[float]]


def _groups(count: int, groups: int) -> list[range]:
    base, extra = divmod(count, groups)
    runs, start = [], 0
    for index in range(groups):
        stop = start + base + (index < extra)
        runs.append(range(start, stop))
        start = stop
    return runs


def _combine(read: Reader, texts: list[str]) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    runs = _groups(len(texts), -(-len(texts) // len(LETTERS)))
    inner = [list(read([texts[index] for index in run])) for run in runs]
    inner = [
        [value / sum(probabilities) for value in probabilities]
        for probabilities in inner
    ]
    keep = max(1, len(LETTERS) // len(runs))
    ranked = [
        sorted(range(len(probabilities)), key=lambda j: -probabilities[j])
        for probabilities in inner
    ]
    chosen = {
        (group, option) for group, order in enumerate(ranked) for option in order[:keep]
    }
    rest = sorted(
        (
            (group, option)
            for group, order in enumerate(ranked)
            for option in order[keep:]
        ),
        key=lambda item: -inner[item[0]][item[1]],
    )
    chosen.update(rest[: max(0, len(LETTERS) - len(chosen))])
    tops = [
        sorted(option for group_index, option in chosen if group_index == group)
        for group in range(len(runs))
    ]
    final = _combine(
        read, [texts[run[option]] for run, top in zip(runs, tops) for option in top]
    )
    shares, at = [], 0
    for top in tops:
        shares.append(dict(zip(top, final[at : at + len(top)])))
        at += len(top)
    in_final = sum(
        sum(finalists.values()) * sum(probabilities[index] for index in finalists)
        for probabilities, finalists in zip(inner, shares)
    )
    weights = []
    for probabilities, finalists in zip(inner, shares):
        mass = sum(finalists.values())
        weights.extend(
            finalists[index] * in_final if index in finalists else mass * probability
            for index, probability in enumerate(probabilities)
        )
    total = sum(weights)
    return [value / total for value in weights]


def spread(read: Reader, texts: list[str], temperature: float) -> list[float]:
    if len(texts) <= len(LETTERS):
        return list(read(texts))
    probabilities = _combine(read, texts)
    if temperature != 1.0:
        probabilities = [value ** (1 / temperature) for value in probabilities]
        total = sum(probabilities)
        probabilities = [value / total for value in probabilities]
    return probabilities


def answer(question: dict, probabilities: dict[str, float], tokens: int) -> dict:
    kind = question["type"]
    result = {
        "type": kind,
        "confidence": max(probabilities.values()),
        "input_tokens": tokens,
    }
    if kind == "noul":
        result["noul"] = probabilities["true"]
    elif kind == "choice":
        result.update(
            choice=max(probabilities, key=probabilities.get),
            probabilities=probabilities,
        )
    else:
        result.update(
            score=sum(int(key) * value for key, value in probabilities.items()),
            probabilities=probabilities,
        )
    return result


class JevK5GGUF:
    def __init__(
        self,
        url: str = "http://127.0.0.1:8082",
        temperature: float = 1.22,
        knockout_temperature: float = 0.93,
        top_k: int = 40,
        timeout_s: float = 180.0,
    ) -> None:
        self.url = url.rstrip("/")
        self.temperature = temperature
        self.knockout_temperature = knockout_temperature
        self.top_k = top_k
        self.timeout_s = timeout_s

    def _post(self, path: str, payload: dict) -> dict:
        request = urllib.request.Request(
            self.url + path,
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=self.timeout_s) as response:
            return json.loads(response.read())

    def _logprobs(self, prompt: str) -> tuple[dict[str, float], int]:
        tokens = self._post(
            "/tokenize",
            {"content": prompt, "add_special": False, "parse_special": True},
        )["tokens"]
        output = self._post(
            "/completion",
            {
                "prompt": tokens,
                "n_predict": 1,
                "n_probs": self.top_k,
                "temperature": 0,
                "cache_prompt": False,
            },
        )
        top = output["completion_probabilities"][0]["top_logprobs"]
        return {entry["token"]: entry["logprob"] for entry in top}, output.get(
            "tokens_evaluated", 0
        )

    def decide(self, state: object, question: dict) -> dict:
        options = decision_options(question)
        tokens = 0

        def read(texts: list[str]) -> list[float]:
            nonlocal tokens
            seen, count = self._logprobs(
                prompt_text(state, question["instructions"], texts)
            )
            tokens += count
            floor = min(seen.values(), default=0.0) - MISSING_MARGIN
            logprobs = [seen.get(LETTERS[index], floor) for index in range(len(texts))]
            top = max(logprobs)
            weights = [math.exp((value - top) / self.temperature) for value in logprobs]
            total = sum(weights)
            return [value / total for value in weights]

        values = spread(read, [text for _, text in options], self.knockout_temperature)
        return answer(question, dict(zip((key for key, _ in options), values)), tokens)
