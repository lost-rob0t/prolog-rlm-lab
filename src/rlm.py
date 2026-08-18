from __future__ import annotations

import json
import re
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from providers import Completion, OpenAICompatibleProvider
from tasks import Task


@dataclass
class RunTrace:
    answer: str = ""
    model_calls: int = 0
    prolog_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_ms: float = 0.0
    prolog_latency_ms: float = 0.0
    cost_usd: float = 0.0
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""


class PrologToolbox:
    SAFE_ATOM = re.compile(r"^[a-z][A-Za-z0-9_]*$")
    SAFE_VAR = re.compile(r"^\?[A-Za-z][A-Za-z0-9_]*$")
    SAFE_INT = re.compile(r"^-?\d+$")

    def __init__(self, task: Task):
        self.task = task

    def _term(self, value: Any, variables: dict[str, str]) -> str:
        if isinstance(value, int):
            return str(value)
        text = str(value)
        if self.SAFE_VAR.fullmatch(text):
            return variables.setdefault(text, f"V{len(variables)}")
        if self.SAFE_INT.fullmatch(text):
            return text
        if self.SAFE_ATOM.fullmatch(text):
            return text
        raise ValueError(f"unsafe/unsupported Prolog argument: {text!r}")

    def call(self, predicate: str, args: list[Any]) -> tuple[str, float]:
        expected_arity = self.task.tools.get(predicate)
        if expected_arity is None:
            raise ValueError(f"predicate {predicate!r} is not allowed")
        if len(args) != expected_arity:
            raise ValueError(f"predicate {predicate}/{expected_arity} received {len(args)} args")
        if not self.SAFE_ATOM.fullmatch(predicate):
            raise ValueError("unsafe predicate")

        variables: dict[str, str] = {}
        terms = [self._term(arg, variables) for arg in args]
        call = predicate if not terms else f"{predicate}({','.join(terms)})"
        vars_in_order = list(variables.values())
        row = f"[{','.join(vars_in_order)}]" if vars_in_order else "[]"
        query = f"findall({row},({call}),Rows),write_canonical(Rows),nl"

        program = self.task.prolog_source + "\n:- initialization(main).\nmain:-" + query + ",halt.\n"
        started = time.perf_counter()
        with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False, encoding="utf-8") as handle:
            path = Path(handle.name)
            handle.write(program)
        try:
            proc = subprocess.run(
                ["swipl", "-q", "-s", str(path)],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
            )
        finally:
            path.unlink(missing_ok=True)
        latency = (time.perf_counter() - started) * 1000
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[:3000])
        return proc.stdout.strip(), latency


def normalize_answer(text: str) -> str:
    text = text.strip()
    if not text:
        return ""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and "answer" in obj:
            text = str(obj["answer"])
    except json.JSONDecodeError:
        pass
    upper = text.strip().upper()
    if re.search(r"\bYES\b", upper):
        return "YES"
    if re.search(r"\bNO\b", upper):
        return "NO"
    match = re.search(r"(?<![\w.])-?\d+(?![\w.])", text)
    if match:
        return match.group(0)
    return text.strip().splitlines()[-1][:200]


def correct(answer: str, expected: str) -> bool:
    return normalize_answer(answer) == normalize_answer(expected)


def _record_completion(trace: RunTrace, completion: Completion, role: str) -> None:
    trace.model_calls += 1
    trace.input_tokens += completion.input_tokens
    trace.output_tokens += completion.output_tokens
    trace.model_latency_ms += completion.latency_ms
    trace.cost_usd += completion.cost_usd
    trace.events.append({
        "type": "model",
        "role": role,
        "text": completion.text,
        "input_tokens": completion.input_tokens,
        "output_tokens": completion.output_tokens,
        "latency_ms": completion.latency_ms,
        "cost_usd": completion.cost_usd,
    })


def _parse_action(text: str) -> dict[str, Any]:
    text = text.strip()
    candidates = [text]
    match = re.search(r"\{.*\}", text, flags=re.S)
    if match and match.group(0) != text:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            obj = json.loads(candidate)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    return {"action": "final", "answer": normalize_answer(text)}


def run_condition(
    provider: OpenAICompatibleProvider,
    task: Task,
    condition: str,
    recursion_level: int,
    *,
    max_tokens: int,
    temperature: float,
) -> RunTrace:
    # Explicit names are used in new experiments. Legacy aliases remain accepted
    # so existing raw sessions and presets stay comparable.
    if condition in {"base", "vanilla"}:
        return run_vanilla(provider, task, max_tokens=max_tokens, temperature=temperature)
    if condition in {"agent_plain", "self_recursive"}:
        return run_self_recursive(
            provider, task, recursion_level, max_tokens=max_tokens, temperature=temperature
        )
    if condition in {"rlm_agent", "rlm", "rlm_verified"}:
        return run_rlm(
            provider,
            task,
            recursion_level,
            verify=(condition == "rlm_verified"),
            max_tokens=max_tokens,
            temperature=temperature,
        )
    raise ValueError(condition)


def run_vanilla(
    provider: OpenAICompatibleProvider,
    task: Task,
    *,
    max_tokens: int,
    temperature: float,
) -> RunTrace:
    trace = RunTrace()
    messages = [
        {
            "role": "system",
            "content": "Solve the task. Return only JSON: {\"answer\": \"...\"}. Do not add commentary.",
        },
        {"role": "user", "content": task.prompt},
    ]
    try:
        completion = provider.complete(messages, max_tokens=max_tokens, temperature=temperature)
        _record_completion(trace, completion, "base")
        trace.answer = normalize_answer(completion.text)
    except Exception as exc:
        trace.error = str(exc)
    return trace


def run_self_recursive(
    provider: OpenAICompatibleProvider,
    task: Task,
    recursion_level: int,
    *,
    max_tokens: int,
    temperature: float,
) -> RunTrace:
    """Plain model-agent baseline with the same turn budget but no Prolog tool."""
    trace = RunTrace()
    messages = [
        {
            "role": "system",
            "content": (
                "Act as an iterative reasoning agent. You have no symbolic or Prolog tool. "
                "On every turn return only JSON {\"answer\":\"...\"}. Keep the answer concise."
            ),
        },
        {"role": "user", "content": task.prompt},
    ]
    try:
        for index in range(max(1, recursion_level)):
            completion = provider.complete(messages, max_tokens=max_tokens, temperature=temperature)
            _record_completion(trace, completion, f"agent_plain_{index}")
            trace.answer = normalize_answer(completion.text)
            if index + 1 < recursion_level:
                messages.extend(
                    [
                        {"role": "assistant", "content": completion.text},
                        {
                            "role": "user",
                            "content": "Re-check every constraint and dependency. Revise only if needed, then return the JSON answer again.",
                        },
                    ]
                )
    except Exception as exc:
        trace.error = str(exc)
    return trace


def run_rlm(
    provider: OpenAICompatibleProvider,
    task: Task,
    recursion_level: int,
    *,
    verify: bool,
    max_tokens: int,
    temperature: float,
) -> RunTrace:
    trace = RunTrace()
    toolbox = PrologToolbox(task)
    tool_lines = ", ".join(f"{name}/{arity}" for name, arity in task.tools.items())
    system = f"""You are solving a benchmark task with a symbolic Prolog tool.
Allowed predicates: {tool_lines}.
Return exactly one JSON object per turn.
To query Prolog: {{"action":"prolog","predicate":"NAME","args":["atom","?X"]}}
Variables must be strings beginning with ?. Atoms must be simple lowercase identifiers or integers.
To answer: {{"action":"final","answer":"..."}}
Use the tool when it can replace manual search. Do not emit prose outside JSON."""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": task.prompt}]

    try:
        for index in range(max(1, recursion_level)):
            completion = provider.complete(messages, max_tokens=max_tokens, temperature=temperature)
            _record_completion(trace, completion, f"rlm_{index}")
            action = _parse_action(completion.text)
            kind = str(action.get("action", "final")).lower()

            if kind == "prolog":
                predicate = str(action.get("predicate", ""))
                args = action.get("args", [])
                if not isinstance(args, list):
                    args = []
                try:
                    result, latency = toolbox.call(predicate, args)
                    trace.prolog_calls += 1
                    trace.prolog_latency_ms += latency
                    trace.events.append({
                        "type": "prolog",
                        "predicate": predicate,
                        "args": args,
                        "result": result,
                        "latency_ms": latency,
                    })
                    messages.extend(
                        [
                            {"role": "assistant", "content": completion.text},
                            {"role": "user", "content": f"PROLOG_RESULT: {result}"},
                        ]
                    )
                    continue
                except Exception as exc:
                    messages.extend(
                        [
                            {"role": "assistant", "content": completion.text},
                            {"role": "user", "content": f"PROLOG_ERROR: {exc}"},
                        ]
                    )
                    continue

            answer = normalize_answer(str(action.get("answer", completion.text)))
            trace.answer = answer
            trace.events.append({"type": "candidate_final", "answer": answer})
            if not verify or correct(answer, task.expected):
                return trace
            if index + 1 < recursion_level:
                messages.extend(
                    [
                        {"role": "assistant", "content": completion.text},
                        {
                            "role": "user",
                            "content": "VERIFIER: that final answer is incorrect. Continue reasoning; use Prolog if useful. The correct answer is not disclosed.",
                        },
                    ]
                )
        return trace
    except Exception as exc:
        trace.error = str(exc)
        return trace
