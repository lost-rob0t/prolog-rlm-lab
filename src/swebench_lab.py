from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from bench import load_config
from providers import ModelSpec, OpenAICompatibleProvider
from util import ROOT, append_jsonl, load_dotenv, now_iso, stable_id, write_json


CONDITIONS = ("base", "agent_plain", "rlm_agent")
CODE_SUFFIXES = {
    ".py", ".pyi", ".js", ".jsx", ".ts", ".tsx", ".java", ".go", ".rs",
    ".c", ".h", ".cc", ".cpp", ".hpp", ".rb", ".php", ".sh", ".toml",
    ".yaml", ".yml", ".json", ".md", ".rst", ".txt",
}


def run(cmd: list[str], *, cwd: Path | None = None, timeout: int = 300, check: bool = True) -> subprocess.CompletedProcess[str]:
    proc = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout, check=False)
    if check and proc.returncode != 0:
        raise RuntimeError(f"command failed ({proc.returncode}): {' '.join(cmd)}\n{(proc.stderr or proc.stdout)[-4000:]}")
    return proc


def safe_repo_path(root: Path, relative: str) -> Path:
    root = root.resolve()
    candidate = (root / relative).resolve()
    if candidate != root and root not in candidate.parents:
        raise ValueError(f"path escapes workspace: {relative!r}")
    return candidate


def extract_patch(text: str) -> str:
    marker = text.find("diff --git ")
    if marker < 0:
        return ""
    patch = text[marker:]
    if "```" in patch:
        patch = patch.split("```", 1)[0]
    return patch.strip() + "\n"


def prediction_record(instance_id: str, model_name: str, patch: str) -> dict[str, str]:
    return {
        "instance_id": instance_id,
        "model_name_or_path": model_name,
        "model_patch": patch,
    }


def _parse_action(text: str) -> dict[str, Any]:
    stripped = text.strip()
    candidates = [stripped]
    match = re.search(r"\{.*\}", stripped, flags=re.S)
    if match and match.group(0) != stripped:
        candidates.append(match.group(0))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
    patch = extract_patch(stripped)
    if patch:
        return {"action": "apply_patch", "patch": patch, "raw_diff": True}
    return {"action": "final"}


@dataclass
class AgentTelemetry:
    model_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    model_latency_ms: float = 0.0
    cost_usd: float = 0.0
    tool_calls: int = 0
    prolog_calls: int = 0
    events: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def record_model(self, completion, role: str) -> None:
        self.model_calls += 1
        self.input_tokens += completion.input_tokens
        self.output_tokens += completion.output_tokens
        self.model_latency_ms += completion.latency_ms
        self.cost_usd += completion.cost_usd
        self.events.append({
            "type": "model",
            "role": role,
            "input_tokens": completion.input_tokens,
            "output_tokens": completion.output_tokens,
            "latency_ms": completion.latency_ms,
            "cost_usd": completion.cost_usd,
        })


class Workspace:
    def __init__(self, root: Path):
        self.root = root.resolve()

    def files(self, limit: int = 400) -> list[str]:
        proc = run(["git", "ls-files"], cwd=self.root)
        return [line for line in proc.stdout.splitlines() if line][:limit]

    def read_file(self, path: str, start: int = 1, end: int = 250) -> str:
        target = safe_repo_path(self.root, path)
        if not target.is_file():
            return f"ERROR: file not found: {path}"
        start = max(1, int(start))
        end = min(start + 399, max(start, int(end)))
        text = target.read_text(encoding="utf-8", errors="replace").splitlines()
        rows = text[start - 1:end]
        return "\n".join(f"{index}: {line}" for index, line in enumerate(rows, start=start))

    def grep(self, pattern: str, limit: int = 80) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            return f"ERROR: invalid regex: {exc}"
        hits: list[str] = []
        for relative in self.files(limit=5000):
            path = self.root / relative
            if path.suffix.lower() not in CODE_SUFFIXES or not path.is_file():
                continue
            try:
                if path.stat().st_size > 512_000:
                    continue
                for lineno, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                    if regex.search(line):
                        hits.append(f"{relative}:{lineno}:{line[:300]}")
                        if len(hits) >= limit:
                            return "\n".join(hits)
            except OSError:
                continue
        return "\n".join(hits) if hits else "NO_MATCHES"

    def apply_patch(self, patch: str) -> str:
        if not patch.strip().startswith("diff --git "):
            return "ERROR: patch must be a unified git diff beginning with 'diff --git '"
        with tempfile.NamedTemporaryFile("w", suffix=".patch", delete=False, encoding="utf-8") as handle:
            patch_path = Path(handle.name)
            handle.write(patch)
        try:
            check = run(["git", "apply", "--check", str(patch_path)], cwd=self.root, check=False)
            if check.returncode != 0:
                return "ERROR: git apply --check failed\n" + (check.stderr or check.stdout)[-3000:]
            apply = run(["git", "apply", str(patch_path)], cwd=self.root, check=False)
            if apply.returncode != 0:
                return "ERROR: git apply failed\n" + (apply.stderr or apply.stdout)[-3000:]
            return "PATCH_APPLIED"
        finally:
            patch_path.unlink(missing_ok=True)

    def diff(self) -> str:
        return run(["git", "diff", "--binary"], cwd=self.root).stdout


def _issue_tokens(problem: str) -> list[str]:
    words = re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", problem.lower())
    stop = {"this", "that", "with", "from", "when", "should", "would", "there", "have", "does", "error", "issue"}
    result: list[str] = []
    for word in words:
        if word not in stop and word not in result:
            result.append(word)
    return result[:24]


def build_context_pack(workspace: Workspace, problem: str, max_files: int = 5, max_chars: int = 30000) -> str:
    """Deterministic lexical retrieval shared by base and both agent conditions."""
    tokens = _issue_tokens(problem)
    scored: list[tuple[int, str, str]] = []
    for relative in workspace.files(limit=6000):
        path = workspace.root / relative
        if path.suffix.lower() not in CODE_SUFFIXES or not path.is_file():
            continue
        try:
            if path.stat().st_size > 256_000:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")[:24000]
        except OSError:
            continue
        lower = text.lower()
        path_lower = relative.lower()
        score = sum(12 for token in tokens if token in path_lower) + sum(min(5, lower.count(token)) for token in tokens)
        if score > 0:
            scored.append((score, relative, text))
    scored.sort(key=lambda row: (-row[0], row[1]))
    if not scored:
        fallback = workspace.files(limit=max_files)
        scored = [(0, rel, workspace.read_file(rel, 1, 160)) for rel in fallback]

    parts = ["TRACKED FILES (first 200):\n" + "\n".join(workspace.files(limit=200))]
    used = len(parts[0])
    for _, relative, text in scored[:max_files]:
        block = f"\n\nFILE: {relative}\n{text[:12000]}"
        if used + len(block) > max_chars:
            block = block[: max(0, max_chars - used)]
        if block:
            parts.append(block)
            used += len(block)
        if used >= max_chars:
            break
    return "".join(parts)


class PrologMemory:
    """Small general symbolic memory available only to the RLM code agent."""

    def __init__(self) -> None:
        self.facts: list[tuple[str, str, str]] = []

    def remember(self, kind: str, subject: str, value: str) -> str:
        row = (str(kind)[:80], str(subject)[:200], str(value)[:1000])
        if row not in self.facts:
            self.facts.append(row)
        return f"REMEMBERED {row[0]} {row[1]}"

    @staticmethod
    def _literal(value: str) -> str:
        return json.dumps(value, ensure_ascii=True)

    def query(self, kind: str = "?", subject: str = "?", value: str = "?") -> str:
        if not shutil.which("swipl"):
            return "PROLOG_ERROR: swipl not installed"
        facts = "\n".join(
            f"fact({self._literal(k)},{self._literal(s)},{self._literal(v)})."
            for k, s, v in self.facts
        )
        terms = []
        variables = []
        for name, item in (("K", kind), ("S", subject), ("V", value)):
            if str(item).startswith("?"):
                terms.append(name)
                variables.append(name)
            else:
                terms.append(self._literal(str(item)))
        row = "[K,S,V]"
        goal = f"findall({row},fact({','.join(terms)}),Rows),write_canonical(Rows),nl"
        program = facts + "\n:- initialization(main).\nmain:-" + goal + ",halt.\n"
        with tempfile.NamedTemporaryFile("w", suffix=".pl", delete=False, encoding="utf-8") as handle:
            path = Path(handle.name)
            handle.write(program)
        try:
            proc = run(["swipl", "-q", "-s", str(path)], timeout=30, check=False)
            if proc.returncode != 0:
                return "PROLOG_ERROR: " + (proc.stderr or proc.stdout)[-2000:]
            return proc.stdout.strip()
        finally:
            path.unlink(missing_ok=True)


def _agent_system(use_prolog: bool) -> str:
    actions = [
        '{"action":"read_file","path":"src/file.py","start":1,"end":220}',
        '{"action":"grep","pattern":"regex"}',
        '{"action":"apply_patch","patch":"diff --git ..."}',
        '{"action":"diff"}',
        '{"action":"final"}',
    ]
    if use_prolog:
        actions.extend(
            [
                '{"action":"prolog_remember","kind":"requirement|hypothesis|touches|risk","subject":"...","value":"..."}',
                '{"action":"prolog_query","kind":"?","subject":"?","value":"?"}',
            ]
        )
    return (
        "You are a software engineering agent working on a checked-out SWE-bench repository. "
        "Inspect the repository and implement the issue. Return exactly one JSON object per turn. "
        "Available actions:\n- " + "\n- ".join(actions) +
        "\nDo not claim a patch was applied unless the tool confirms it. Prefer small, targeted edits."
    )


def run_base(provider: OpenAICompatibleProvider, problem: str, context: str, max_tokens: int) -> tuple[str, AgentTelemetry]:
    telemetry = AgentTelemetry()
    messages = [
        {"role": "system", "content": "Solve the software issue in one attempt. Return only a unified git diff beginning with 'diff --git '."},
        {"role": "user", "content": f"ISSUE:\n{problem}\n\nREPOSITORY CONTEXT:\n{context}"},
    ]
    try:
        completion = provider.complete(messages, max_tokens=max_tokens, temperature=0.0)
        telemetry.record_model(completion, "base")
        return extract_patch(completion.text), telemetry
    except Exception as exc:
        telemetry.error = str(exc)
        return "", telemetry


def run_agent(
    provider: OpenAICompatibleProvider,
    workspace: Workspace,
    problem: str,
    context: str,
    *,
    turns: int,
    max_tokens: int,
    use_prolog: bool,
) -> tuple[str, AgentTelemetry]:
    telemetry = AgentTelemetry()
    memory = PrologMemory() if use_prolog else None
    messages = [
        {"role": "system", "content": _agent_system(use_prolog)},
        {"role": "user", "content": f"ISSUE:\n{problem}\n\nINITIAL CONTEXT:\n{context}"},
    ]
    try:
        for turn in range(max(1, turns)):
            completion = provider.complete(messages, max_tokens=max_tokens, temperature=0.0)
            telemetry.record_model(completion, f"agent_{turn}")
            action = _parse_action(completion.text)
            kind = str(action.get("action", "final"))
            result = ""

            if kind == "read_file":
                result = workspace.read_file(str(action.get("path", "")), action.get("start", 1), action.get("end", 250))
            elif kind == "grep":
                result = workspace.grep(str(action.get("pattern", "")))
            elif kind == "apply_patch":
                result = workspace.apply_patch(str(action.get("patch", "")))
            elif kind == "diff":
                result = workspace.diff() or "NO_DIFF"
            elif kind == "prolog_remember" and memory is not None:
                telemetry.prolog_calls += 1
                result = memory.remember(str(action.get("kind", "note")), str(action.get("subject", "")), str(action.get("value", "")))
            elif kind == "prolog_query" and memory is not None:
                telemetry.prolog_calls += 1
                result = memory.query(str(action.get("kind", "?")), str(action.get("subject", "?")), str(action.get("value", "?")))
            elif kind == "final":
                break
            else:
                result = f"ERROR: unavailable action {kind!r}"

            telemetry.tool_calls += 1
            telemetry.events.append({"type": "tool", "action": kind, "result": result[:4000]})
            messages.extend([
                {"role": "assistant", "content": completion.text},
                {"role": "user", "content": f"TOOL_RESULT:\n{result[:12000]}"},
            ])
        return workspace.diff(), telemetry
    except Exception as exc:
        telemetry.error = str(exc)
        return workspace.diff(), telemetry


def load_model(model_id: str) -> ModelSpec:
    _, models = load_config()
    for model in models:
        if model.id == model_id:
            return model
    raise ValueError(f"unknown model id: {model_id}")


def load_instances(dataset_name: str, split: str, limit: int, wanted: list[str]) -> list[dict[str, Any]]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("SWE-bench dependencies missing; run: pip install -r requirements-swebench.txt") from exc
    dataset = load_dataset(dataset_name, split=split)
    rows = [dict(row) for row in dataset]
    if wanted:
        wanted_set = set(wanted)
        rows = [row for row in rows if row.get("instance_id") in wanted_set]
    rows.sort(key=lambda row: str(row.get("instance_id", "")))
    return rows[:limit] if limit > 0 else rows


def prepare_workspace(repo: str, base_commit: str, destination: Path) -> Workspace:
    cache_root = ROOT / ".cache" / "swebench" / "repos"
    cache_root.mkdir(parents=True, exist_ok=True)
    mirror = cache_root / (repo.replace("/", "__") + ".git")
    url = f"https://github.com/{repo}.git"
    if not mirror.exists():
        run(["git", "clone", "--mirror", url, str(mirror)], timeout=1800)
    # Fetch only when the requested commit is not already in the mirror. This
    # keeps reruns deterministic and avoids gratuitous network traffic.
    have = run(["git", "--git-dir", str(mirror), "cat-file", "-e", f"{base_commit}^{{commit}}"], check=False)
    if have.returncode != 0:
        run(["git", "--git-dir", str(mirror), "fetch", "origin", base_commit], timeout=900)
    if destination.exists():
        shutil.rmtree(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    run(["git", "clone", "--shared", "--no-checkout", str(mirror), str(destination)], timeout=900)
    run(["git", "checkout", "--detach", base_commit], cwd=destination, timeout=300)
    return Workspace(destination)


def generate(args: argparse.Namespace) -> int:
    load_dotenv()
    model = load_model(args.model_id)
    probe = OpenAICompatibleProvider(model, timeout_s=180).probe()
    if not probe.ok:
        print(f"provider preflight failed: {probe.detail}")
        return 3
    rows = load_instances(args.dataset_name, args.split, args.limit, args.instance_id)
    if not rows:
        print("no SWE-bench instances selected")
        return 2

    session_id = args.session_id or stable_id("swebench", now_iso(), args.model_id, args.limit)
    root = ROOT / "results" / "swebench"
    prediction_dir = root / "predictions" / session_id
    telemetry_path = root / "telemetry" / f"{session_id}.jsonl"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    telemetry_path.parent.mkdir(parents=True, exist_ok=True)
    for condition in CONDITIONS:
        (prediction_dir / f"{condition}.jsonl").unlink(missing_ok=True)

    for index, row in enumerate(rows, 1):
        instance_id = str(row["instance_id"])
        repo = str(row["repo"])
        base_commit = str(row["base_commit"])
        problem = str(row["problem_statement"])
        context_workspace = prepare_workspace(repo, base_commit, root / "workspaces" / session_id / instance_id / "context")
        context = build_context_pack(context_workspace, problem)

        for condition in CONDITIONS:
            workspace = prepare_workspace(repo, base_commit, root / "workspaces" / session_id / instance_id / condition)
            provider = OpenAICompatibleProvider(model, timeout_s=args.timeout)
            if condition == "base":
                patch, telemetry = run_base(provider, problem, context, args.max_tokens)
            else:
                patch, telemetry = run_agent(
                    provider,
                    workspace,
                    problem,
                    context,
                    turns=args.turns,
                    max_tokens=args.max_tokens,
                    use_prolog=(condition == "rlm_agent"),
                )
            append_jsonl(
                prediction_dir / f"{condition}.jsonl",
                prediction_record(instance_id, f"{model.model}#{condition}", patch),
            )
            append_jsonl(
                telemetry_path,
                {
                    "session_id": session_id,
                    "timestamp": now_iso(),
                    "instance_id": instance_id,
                    "repo": repo,
                    "base_commit": base_commit,
                    "condition": condition,
                    "model_id": model.id,
                    "provider_model": model.model,
                    "model_calls": telemetry.model_calls,
                    "tool_calls": telemetry.tool_calls,
                    "prolog_calls": telemetry.prolog_calls,
                    "input_tokens": telemetry.input_tokens,
                    "output_tokens": telemetry.output_tokens,
                    "latency_ms": round(telemetry.model_latency_ms, 3),
                    "cost_usd": round(telemetry.cost_usd, 8),
                    "patch_bytes": len(patch.encode()),
                    "error": telemetry.error,
                },
            )
            print(
                f"[{index}/{len(rows)}] {instance_id} {condition:<11} "
                f"calls={telemetry.model_calls} tools={telemetry.tool_calls} prolog={telemetry.prolog_calls} patch={len(patch)}B"
            )

    write_json(
        root / "predictions" / session_id / "manifest.json",
        {
            "session_id": session_id,
            "created_at": now_iso(),
            "dataset_name": args.dataset_name,
            "split": args.split,
            "model_id": model.id,
            "provider_model": model.model,
            "conditions": list(CONDITIONS),
            "instances": [str(row["instance_id"]) for row in rows],
            "turns": args.turns,
            "max_tokens": args.max_tokens,
            "telemetry_file": str(telemetry_path.relative_to(ROOT)),
        },
    )
    print(f"session={session_id}")
    print(f"predictions={prediction_dir.relative_to(ROOT)}")
    return 0


def evaluate(args: argparse.Namespace) -> int:
    root = ROOT / "results" / "swebench" / "predictions" / args.session_id
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        print(f"missing SWE-bench session: {manifest_path}")
        return 2
    manifest = json.loads(manifest_path.read_text())
    dataset_name = args.dataset_name or manifest["dataset_name"]
    split = args.split or manifest["split"]
    instance_ids = list(manifest.get("instances", []))
    exit_code = 0
    for condition in CONDITIONS:
        predictions = root / f"{condition}.jsonl"
        run_id = f"{args.session_id}-{condition}"
        command = [
            os.environ.get("PYTHON", "python3"),
            "-m",
            "swebench.harness.run_evaluation",
            "--dataset_name",
            dataset_name,
            "--split",
            split,
            "--predictions_path",
            str(predictions),
            "--max_workers",
            str(args.workers),
            "--run_id",
            run_id,
        ]
        if instance_ids:
            command += ["--instance_ids", *instance_ids]
        print("running official SWE-bench harness:", condition)
        proc = subprocess.run(command, cwd=ROOT, check=False)
        exit_code = max(exit_code, proc.returncode)
    return exit_code


def doctor() -> int:
    missing = [name for name in ("git", "docker", "swipl") if not shutil.which(name)]
    try:
        import swebench  # noqa: F401
        import datasets  # noqa: F401
    except ImportError:
        missing.append("python:swebench/datasets")
    if missing:
        print("SWE-bench doctor FAIL:", ", ".join(missing))
        print("install Python dependencies with: pip install -r requirements-swebench.txt")
        return 2
    docker = run(["docker", "info"], check=False, timeout=30)
    if docker.returncode != 0:
        print("SWE-bench doctor FAIL: Docker daemon unavailable")
        return 2
    print("SWE-bench doctor OK: git, Docker, SWI-Prolog, swebench, datasets")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SWE-bench base vs plain agent vs Prolog-RLM agent")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("doctor")

    generate_parser = sub.add_parser("generate")
    generate_parser.add_argument("--dataset-name", default="princeton-nlp/SWE-bench_Lite")
    generate_parser.add_argument("--split", default="test")
    generate_parser.add_argument("--model-id", default=os.getenv("MODEL_ID", "qwen3-8b-q4-k-m"))
    generate_parser.add_argument("--limit", type=int, default=3)
    generate_parser.add_argument("--instance-id", action="append", default=[])
    generate_parser.add_argument("--turns", type=int, default=8)
    generate_parser.add_argument("--max-tokens", type=int, default=2048)
    generate_parser.add_argument("--timeout", type=int, default=300)
    generate_parser.add_argument("--session-id", default="")

    evaluate_parser = sub.add_parser("evaluate")
    evaluate_parser.add_argument("--session-id", required=True)
    evaluate_parser.add_argument("--dataset-name", default="")
    evaluate_parser.add_argument("--split", default="")
    evaluate_parser.add_argument("--workers", type=int, default=2)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.command == "doctor":
        return doctor()
    if args.command == "generate":
        return generate(args)
    if args.command == "evaluate":
        return evaluate(args)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
