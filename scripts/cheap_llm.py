"""Delegate bulk typing to a free model, keep the judgement here.

What this is for
----------------
Work where the output is long and the thinking is short: tests written to a
finished specification, boilerplate, docstrings, repetitive refactors. The
specification is still written by a human or by the orchestrating model, and
the result is still read and verified before it is used.

What it is not for: anything where being subtly wrong is expensive. Nearly
every defect this project has found — a generator leaking the answer, a
transitive clustering primitive deleting the hard cases, a results table that
did not match what was measured — needed judgement, not typing. Delegating
those costs more in review than it saves in generation.

Key handling
------------
The key is read from ``GROQ_API_KEY``, or from ``.env.local`` in the repo
root, which is git-ignored. It is never passed on the command line, where it
would land in shell history, and never written into a tracked file.

Usage
-----
    python scripts/cheap_llm.py --task "write pytest cases for X" \\
        --context src/module.py --out /tmp/generated.py

    python scripts/cheap_llm.py --task "..." --model qwen/qwen3.8-27b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

GROQ_ENDPOINT = "https://api.groq.com/openai/v1/chat/completions"

# gpt-oss-120b is the largest available on the free tier and the strongest at
# code; qwen3.8-27b is a good second opinion when the first output looks off.
DEFAULT_MODEL = "openai/gpt-oss-120b"

SYSTEM_PROMPT = (
    "You are writing production Python for an existing codebase. "
    "Match the surrounding style: type hints everywhere, `from __future__ "
    "import annotations`, docstrings that say WHY rather than restating the "
    "signature, line length 100. Output only code, no prose, no markdown "
    "fences. If the task is ambiguous, pick the reading that makes the code "
    "simplest and note it in a comment."
)

MAX_CONTEXT_CHARS = 60_000


def load_key(repo_root: Path) -> str:
    """Environment first, then the git-ignored local file."""
    key = os.environ.get("GROQ_API_KEY", "").strip()
    if key:
        return key

    env_file = repo_root / ".env.local"
    if env_file.exists():
        for line in env_file.read_text(encoding="utf-8").splitlines():
            name, _, value = line.partition("=")
            if name.strip() == "GROQ_API_KEY":
                return value.strip()

    raise SystemExit(
        "No GROQ_API_KEY found. Set the environment variable, or put "
        "GROQ_API_KEY=... in .env.local (git-ignored)."
    )


def build_context(paths: list[str]) -> str:
    """Concatenate context files, truncating rather than failing.

    A silently truncated prompt produces plausible wrong code, so the cut is
    announced inside the prompt itself where the model can see it.
    """
    chunks: list[str] = []
    budget = MAX_CONTEXT_CHARS
    for raw in paths:
        path = Path(raw)
        if not path.exists():
            print(f"warning: context file not found, skipping: {path}", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if len(text) > budget:
            text = text[:budget] + f"\n# ...TRUNCATED, {len(text) - budget} chars omitted\n"
        budget -= min(len(text), budget)
        chunks.append(f"# ===== FILE: {path} =====\n{text}")
        if budget <= 0:
            chunks.append("# ...remaining context files omitted, budget exhausted")
            break
    return "\n\n".join(chunks)


def ask(task: str, context: str, model: str, key: str, temperature: float) -> str:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if context:
        messages.append({"role": "user", "content": f"Existing code for context:\n\n{context}"})
    messages.append({"role": "user", "content": task})

    payload = json.dumps(
        {"model": model, "messages": messages, "temperature": temperature}
    ).encode("utf-8")

    parsed = urllib.parse.urlsplit(GROQ_ENDPOINT)
    if parsed.scheme != "https":
        raise SystemExit("refusing to send a key over a non-https endpoint")

    request = urllib.request.Request(
        GROQ_ENDPOINT,
        data=payload,
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            # The default urllib agent is refused at the edge with a bare 403
            # and no body, which reads like an auth failure and is not one.
            "User-Agent": "cheops-ai/1.0",
        },
        method="POST",
    )
    try:
        # nosec B310 - scheme asserted above; the endpoint is a module constant.
        with urllib.request.urlopen(request, timeout=180) as response:  # nosec B310
            body = json.loads(response.read())
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", errors="replace")[:800]
        raise SystemExit(f"HTTP {error.code} from provider: {detail}") from error

    if "error" in body:
        raise SystemExit(f"provider error: {body['error'].get('message')}")

    choice = body["choices"][0]["message"]["content"]
    usage = body.get("usage", {})
    print(
        f"[{model}] prompt {usage.get('prompt_tokens', '?')} tokens, "
        f"completion {usage.get('completion_tokens', '?')} tokens",
        file=sys.stderr,
    )
    return str(choice)


def strip_fences(text: str) -> str:
    """Models add markdown fences however firmly you ask them not to."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip() + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", required=True, help="what to write")
    parser.add_argument("--context", nargs="*", default=[], help="files to read for context")
    parser.add_argument("--out", help="write the result here instead of stdout")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--temperature", type=float, default=0.2)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent.parent
    key = load_key(repo_root)
    context = build_context(args.context)
    result = strip_fences(ask(args.task, context, args.model, key, args.temperature))

    if args.out:
        Path(args.out).write_text(result, encoding="utf-8")
        print(f"written to {args.out} ({len(result)} chars)", file=sys.stderr)
    else:
        print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
