"""Turn the OpenAI Model Spec into a constitution with labeled examples.

The spec (https://github.com/openai/model_spec, CC0) writes each character
trait as a principle plus GOOD/BAD comparisons on real prompts. That is a
constitution in the Constitutional AI sense, and the comparisons are
labeled preference data. This script reads the markdown source and
writes ``constitution.json``: one entry per trait with the principle
text and every comparison that has at least one GOOD and one BAD side.

    python from_model_spec.py                      # fetch main, default traits
    python from_model_spec.py --spec model_spec.md  # a local copy
    python from_model_spec.py --traits be_warm avoid_sycophancy

Trait ids are the spec's own heading anchors (``{#be_warm ...}``), so a
row's ``spec_id`` points back at the sentence it was graded against.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
import urllib.request
from pathlib import Path

RAW_URL = "https://raw.githubusercontent.com/openai/model_spec/main/model_spec.md"

# The style section of the spec plus the honesty traits that read as
# character rather than policy. Each has comparisons with both labels.
DEFAULT_TRAITS = [
    "be_warm",
    "be_clear",
    "avoid_sycophancy",
    "refusal_style",
    "avoid_being_condescending",
    "be_rationally_optimistic",
    "love_humanity",
    "do_not_make_unprompted_personal_comments",
]

_HEADING = re.compile(r"^(#{1,4}) (.+?) \{#([A-Za-z0-9_]+)((?: [^}]*)?)\}\s*$")
_XML_BLOCK = re.compile(r"~~~xml\s*\n(.*?)\n~~~", re.S)
_TAG = re.compile(
    r"<(user|assistant|system|developer|tool)([^>]*)>(.*?)</\1>",
    re.S,
)
_LABEL = re.compile(r"^\s*<!--\s*(GOOD|BAD|OK)\b[:\s]*(.*?)\s*-->", re.S)
_FOOTNOTE = re.compile(r"\[\^[^\]]+\]")
_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def load_spec(path: str | None) -> str:
    if path:
        return Path(path).read_text(encoding="utf-8")
    with urllib.request.urlopen(RAW_URL, timeout=30) as resp:
        return resp.read().decode("utf-8")


def sections(text: str) -> dict[str, dict]:
    """anchor -> {title, level, authority, body}. Body runs to the next heading."""
    out: dict[str, dict] = {}
    current: dict | None = None
    lines: list[str] = []
    for line in text.splitlines():
        m = _HEADING.match(line)
        if m:
            if current is not None:
                current["body"] = "\n".join(lines)
                out[current["id"]] = current
            attrs = m.group(4) or ""
            auth = re.search(r"authority=(\w+)", attrs)
            current = {
                "id": m.group(3),
                "title": m.group(2).strip(),
                "level": len(m.group(1)),
                "authority": auth.group(1) if auth else None,
            }
            lines = []
        elif current is not None:
            lines.append(line)
    if current is not None:
        current["body"] = "\n".join(lines)
        out[current["id"]] = current
    return out


def principle_of(body: str, *, max_chars: int = 1200) -> str:
    """The prose before the first example, footnotes and links stripped."""
    paras: list[str] = []
    buf: list[str] = []
    for line in body.splitlines():
        if line.startswith("~~~") or line.startswith("**Example**") or line.startswith("!!!"):
            break
        if line.strip():
            buf.append(line.strip())
        elif buf:
            paras.append(" ".join(buf))
            buf = []
    if buf:
        paras.append(" ".join(buf))
    text = "\n\n".join(paras)
    text = _FOOTNOTE.sub("", text)
    # "(see [?](#anchor))" and "(see also [?](#anchor) for ...)" are
    # cross-references into the spec; drop the whole parenthetical.
    text = re.sub(r"\s*\([^()]*\[\?\]\([^)]*\)[^()]*\)", "", text)
    text = re.sub(r"\[\?\]\([^)]*\)", "", text)  # bare cross-references
    text = _LINK.sub(r"\1", text)
    text = text.replace(" ,", ",").replace(" .", ".")
    text = re.sub(r"[ \t]{2,}", " ", text).strip()
    return text[:max_chars].rstrip()


def _clean(content: str) -> tuple[str | None, str, str]:
    """(label, note, text) for one tag body; label None outside comparisons."""
    m = _LABEL.match(content)
    label, note = (m.group(1), m.group(2).strip()) if m else (None, "")
    text = content[m.end() :] if m else content
    text = html.unescape(text).strip()
    return label, note, text


def examples_of(body: str) -> list[dict]:
    """Every comparison with a GOOD and a BAD side, with its conversation prefix."""
    out: list[dict] = []
    for block in _XML_BLOCK.findall(body):
        messages: list[dict] = []
        for part in re.split(r"(<comparison>.*?</comparison>)", block, flags=re.S):
            if part.startswith("<comparison>"):
                good: list[str] = []
                bad: list[str] = []
                notes: list[str] = []
                for role, _attrs, content in _TAG.findall(part):
                    if role != "assistant":
                        continue
                    label, note, text = _clean(content)
                    if label == "GOOD":
                        good.append(text)
                    elif label == "BAD":
                        bad.append(text)
                    if note:
                        notes.append(f"{label}: {note}")
                prompt = next(
                    (m["content"] for m in reversed(messages) if m["role"] == "user"), None
                )
                if good and bad and prompt:
                    idx = max(i for i, m in enumerate(messages) if m["role"] == "user")
                    out.append(
                        {
                            "prefix": [m for m in messages[:idx] if m["role"] != "tool"],
                            "prompt": prompt,
                            "good": good,
                            "bad": bad,
                            "notes": notes,
                        }
                    )
                # A GOOD reply may continue the conversation for a later comparison.
                if good:
                    messages.append({"role": "assistant", "content": good[0]})
            else:
                for role, attrs, content in _TAG.findall(part):
                    if "recipient=" in attrs:
                        continue
                    _label, _note, text = _clean(content)
                    messages.append({"role": role, "content": text})
    return out


# The spec illustrates refusal style with one sexually explicit request.
# The GOOD reply is a refusal, but the prompt has no place in a repo
# example; ``--skip ''`` keeps it.
DEFAULT_SKIP = ["fellatio"]


def build(text: str, traits: list[str], *, skip: list[str] | None = None) -> dict:
    secs = sections(text)
    entries: list[dict] = []
    missing: list[str] = []
    skip = [s for s in (skip if skip is not None else DEFAULT_SKIP) if s]
    for trait in traits:
        sec = secs.get(trait)
        if sec is None:
            missing.append(trait)
            continue
        examples = [
            ex
            for ex in examples_of(sec["body"])
            if not any(s in ex["prompt"].lower() for s in skip)
        ]
        entries.append(
            {
                "id": trait,
                "name": sec["title"],
                "authority": sec["authority"],
                "principle": principle_of(sec["body"]),
                "examples": examples,
            }
        )
    return {"traits": entries, "missing": missing}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--spec", help="path to model_spec.md (default: fetch main from GitHub)")
    ap.add_argument("--traits", nargs="*", default=DEFAULT_TRAITS, help="heading anchors to keep")
    ap.add_argument("--commit", default=None, help="spec commit sha to record as provenance")
    ap.add_argument(
        "--skip",
        nargs="*",
        default=DEFAULT_SKIP,
        help="drop examples whose prompt contains any of these",
    )
    ap.add_argument("--out", default=str(Path(__file__).with_name("constitution.json")))
    args = ap.parse_args(argv)

    text = load_spec(args.spec)
    built = build(text, list(args.traits), skip=list(args.skip))
    doc = {
        "source": {
            "repo": "https://github.com/openai/model_spec",
            "file": "model_spec.md",
            "commit": args.commit,
            "license": "CC0-1.0",
        },
        "traits": built["traits"],
    }
    Path(args.out).write_text(
        json.dumps(doc, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    for t in built["traits"]:
        n_good = sum(len(e["good"]) for e in t["examples"])
        n_bad = sum(len(e["bad"]) for e in t["examples"])
        print(f"{t['id']:<42} examples {len(t['examples']):>2}  good {n_good:>2}  bad {n_bad:>2}")
    if built["missing"]:
        print("missing:", ", ".join(built["missing"]), file=sys.stderr)
    print(f"wrote {args.out}")
    return 1 if built["missing"] else 0


if __name__ == "__main__":
    sys.exit(main())
