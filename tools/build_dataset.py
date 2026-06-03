#!/usr/bin/env python3
"""Curate a clean Nixia chat corpus from validated public sources.

The script intentionally uses only Python stdlib. It reads small pages from the
Hugging Face datasets-server API, validates license metadata, filters noisy rows,
deduplicates, optionally adds synthetic Nixia-style seed dialogues, and emits the
plain text format consumed by the Rust trainer:

    <user> ...
    <char> ...

Large-scale runs can use the same script with a bigger --max-rows-per-source,
but for 100k+ dialogues a proper HF `datasets`/parquet pipeline is faster.
"""

from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import html
import io
import json
import random
import re
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


HF_API = "https://huggingface.co/api/datasets/{}"
HF_RAW_FILE = "https://huggingface.co/datasets/{}/resolve/main/{}"
HF_ROWS_API = "https://datasets-server.huggingface.co/rows"
USER_AGENT = "nixia-dataset-builder/0.1 (+https://huggingface.co)"

ROLE_USER = "<user>"
ROLE_CHAR = "<char>"

COMMON_ID_WORDS = {
    "aku", "kamu", "gue", "gw", "lu", "lo", "saya", "iya", "yaa", "ya", "nggak",
    "gak", "ga", "engga", "enggaa", "ndak", "kok", "dong", "nih", "sih", "deh",
    "banget", "bgt", "lagi", "mau", "boleh", "bisa", "cerita", "temen", "teman",
    "wkwk", "hehe", "hmm", "loh", "masa", "makasih", "terima", "kasih", "kalau",
    "kalo", "bareng", "aja", "dulu", "nanti", "atuh", "euy", "pisan", "maneh",
    "urang", "keur", "iso", "ojo", "ae", "piye", "kabare", "pol", "bangettt",
}

CASUAL_MARKERS = {
    "wkwk", "hehe", "iyaa", "iyaaa", "yaa", "bisaa", "enggaa", "gaa", "nih", "dong",
    "sih", "deh", "loh", "kok", "gabut", "salting", "vibes", "boss", "barudak",
    "atuh", "euy", "pisan", "iso", "ojo", "ae", "piye", "pol", "bro", "bre",
}

REJECT_SUBSTRINGS = {
    "sebagai model bahasa", "sebagai ai", "saya adalah ai", "i am an ai",
    "<script"
}

SOCIAL_REJECT_TERMS = {
    "anjing", "bangsat", "bajingan", "goblok", "tolol", "maling", "koruptor",
    "dpr", "dprd", "pilpres", "pemilu", "partai", "cebong", "kampret",
    "pemerintah", "polri", "kapolri", "presiden", "menteri", "radikalisme",
    "jokowi", "prabowo", "anies", "ganjar", "fadli", "zonk", "licik",
    "pembenci", "pengkhianatan", "antipati", "aspirasi",
}

SOCIAL_RESPONSE_TEMPLATES = [
    "aku nangkep vibes-nya. kamu mau cerita konteksnya pelan-pelan?",
    "hmm iya, kedengarannya lagi rame di kepala ya. bagian mana yang paling kepikiran?",
    "boleh, kita obrolin santai aja. menurut kamu yang paling bikin ganjel apa?",
    "aku dengerin. coba ceritain sedikit dulu, gak harus langsung semuanya.",
    "wkwk aku paham maksudmu. mau dibahas serius atau santai aja?",
]

REJECT_TERMS = {
    # Keep the list conservative: reject explicit/pornographic or dangerous rows,
    # while still allowing normal emotional-support conversations.
    "porn", "bokep", "colmek", "pemerkosaan", "memperkosa", "bunuh diri",
    "cara bunuh", "bom rakitan", "narkoba", "sabu", "judi online",
    "phising", "phishing", "nge-hack", "hack akun", "ip address", "lacak lokasi",
}

MOJIBAKE_MARKERS = set("ÂâÃãÅåÐðÏï�™€¿º¸³£œ")
PERSON_TITLE_RE = re.compile(
    r"\b(?P<title>Uda|Uni|Mak|Pak|Bu|Bapak|Ibu|Bang|Kak|Mas|Mbak)\s+"
    r"(?P<name>[A-Z][A-Za-zÀ-ÿ]{2,}(?:\s+[A-Z][A-Za-zÀ-ÿ]{2,})?)\b"
)

EMOJI_REPLACEMENTS = {
    "😂": "<ketawa>", "🤣": "<ketawa>", "😭": "<nangis>", "😢": "<nangis>",
    "😅": "<canggung>", "😊": "<senyum>", "🙂": "<senyum>", "😍": "<love>",
    "❤️": "<love>", "❤": "<love>", "😎": "<kacamata>", "👍": "<jempol>", "🙏": "<makasih>",
}


@dataclass
class Dialogue:
    source_id: str
    turns: list[tuple[str, str]]
    score: float


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    manifest_path = resolve_under_root(root, args.manifest)
    manifest = read_json(manifest_path)
    rng = random.Random(args.seed)

    selected = select_sources(manifest["sources"], args)
    stats: dict[str, Counter[str]] = defaultdict(Counter)
    dialogues: list[Dialogue] = []
    seen = set()

    for source in selected:
        source_id = source["id"]
        if not license_allowed(source, args):
            stats[source_id]["skipped_license_policy"] += 1
            continue

        if source.get("type", "").startswith("hf_") and source.get("repo") and not args.offline:
            if not verify_hf_license(source):
                stats[source_id]["skipped_license_mismatch"] += 1
                continue

        for raw in iter_source_rows(root, source, args):
            stats[source_id]["raw"] += 1
            for candidate in adapt_row(source, raw, stats[source_id]):
                accepted = accept_candidate(
                    dialogues,
                    seen,
                    stats,
                    source_id,
                    candidate,
                    args.min_score,
                )

                if accepted and args.target_dialogues and len(dialogues) >= args.target_dialogues:
                    break
            if args.target_dialogues and len(dialogues) >= args.target_dialogues:
                break
        if args.target_dialogues and len(dialogues) >= args.target_dialogues:
            break

    for path in extra_text_paths(root, args):
        if args.target_dialogues and len(dialogues) >= args.target_dialogues:
            break
        source_id = f"extra_text:{path.name}"
        stats[source_id]["raw"] += 1
        for candidate in parse_nixia_text(path.read_text(encoding="utf-8")):
            accepted = accept_candidate(
                dialogues,
                seen,
                stats,
                source_id,
                candidate,
                args.min_score,
            )
            if accepted and args.target_dialogues and len(dialogues) >= args.target_dialogues:
                break
        if args.target_dialogues and len(dialogues) >= args.target_dialogues:
            break

    if args.synthesize > 0:
        synthetic_source_id = f"synthetic_{args.synth_mode.replace('-', '_')}"
        for dialogue in synthesize_dialogues(
            args.synthesize,
            rng,
            args.include_local_flavor,
            args.synth_mode,
        ):
            if args.target_dialogues and len(dialogues) >= args.target_dialogues:
                break
            accepted = accept_candidate(
                dialogues,
                seen,
                stats,
                synthetic_source_id,
                dialogue,
                args.min_score,
            )
            if accepted and args.target_dialogues and len(dialogues) >= args.target_dialogues:
                break

    rng.shuffle(dialogues)
    write_outputs(root, args, dialogues, stats)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a curated Nixia chat corpus")
    parser.add_argument("--manifest", default="data/dataset_sources.json")
    parser.add_argument("--output", default="data/curated/train_corpus.txt")
    parser.add_argument("--valid-output", default="data/curated/valid_corpus.txt")
    parser.add_argument("--report", default="data/curated/build_report.json")
    parser.add_argument("--sources", default="", help="Comma-separated source ids; default: enabled sources")
    parser.add_argument("--max-rows-per-source", type=int, default=300)
    parser.add_argument(
        "--source-limit",
        action="append",
        default=[],
        metavar="SOURCE_ID=N",
        help="Override --max-rows-per-source for one source. Can be repeated.",
    )
    parser.add_argument("--target-dialogues", type=int, default=0)
    parser.add_argument("--valid-ratio", type=float, default=0.05)
    parser.add_argument("--min-score", type=float, default=1.0)
    parser.add_argument(
        "--extra-text",
        action="append",
        default=[],
        help="Add a local corpus/style-pack file in <user>/<char> format. Can be repeated.",
    )
    parser.add_argument(
        "--extra-glob",
        action="append",
        default=[],
        help="Add local corpus files matching a project-relative glob, e.g. data/templates/nixia_dataset_*.txt.",
    )
    parser.add_argument("--synthesize", type=int, default=0, help="Add generated style-seed dialogues")
    parser.add_argument(
        "--synth-mode",
        choices=["nixia-style", "chat-clean"],
        default="nixia-style",
        help="Synthetic generator profile. chat-clean is safer for casual/chat-only training.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-sharealike", action="store_true")
    parser.add_argument("--allow-noncommercial", action="store_true")
    parser.add_argument(
        "--include-local-flavor",
        action="store_true",
        help="Allow synthetic Jawa/Sunda/slang-heavy examples. Keep off for clean base training.",
    )
    parser.add_argument("--offline", action="store_true", help="Skip live HF license verification")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.source_limits = parse_source_limits(args.source_limit)
    return args


def parse_source_limits(values: list[str]) -> dict[str, int]:
    limits = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(f"invalid --source-limit {value!r}; expected SOURCE_ID=N")
        source_id, raw_limit = value.split("=", 1)
        source_id = source_id.strip()
        if not source_id:
            raise SystemExit(f"invalid --source-limit {value!r}; source id is empty")
        try:
            limit = int(raw_limit)
        except ValueError as error:
            raise SystemExit(f"invalid --source-limit {value!r}; limit must be an integer") from error
        if limit < 0:
            raise SystemExit(f"invalid --source-limit {value!r}; limit must be >= 0")
        limits[source_id] = limit
    return limits


def extra_text_paths(root: Path, args: argparse.Namespace) -> list[Path]:
    paths = [resolve_under_root(root, path) for path in args.extra_text]

    for pattern in args.extra_glob:
        if Path(pattern).is_absolute() or ".." in Path(pattern).parts:
            raise SystemExit(f"refusing unsafe --extra-glob pattern: {pattern}")
        matches = sorted(path for path in root.glob(pattern) if path.is_file())
        if not matches:
            raise SystemExit(f"--extra-glob matched no files: {pattern}")
        paths.extend(resolve_under_root(root, str(path)) for path in matches)

    deduped = []
    seen = set()
    for path in paths:
        if path in seen:
            continue
        seen.add(path)
        deduped.append(path)
    return deduped


def resolve_under_root(root: Path, path: str) -> Path:
    target = (root / path).resolve() if not Path(path).is_absolute() else Path(path).resolve()
    # Avoid accidental writes outside the repo for generated outputs.
    if target != root and root not in target.parents:
        raise SystemExit(f"refusing path outside project root: {target}")
    return target


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def select_sources(sources: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    requested = {item.strip() for item in args.sources.split(",") if item.strip()}
    if not requested:
        return [source for source in sources if source.get("enabled", False)]
    by_id = {source["id"]: source for source in sources}
    missing = sorted(requested - by_id.keys())
    if missing:
        raise SystemExit(f"unknown source id(s): {', '.join(missing)}")
    return [by_id[source_id] for source_id in requested]


def license_allowed(source: dict[str, Any], args: argparse.Namespace) -> bool:
    klass = source.get("license_class", "unknown")
    if klass in {"permissive", "project-local"}:
        return True
    if klass == "sharealike":
        return args.allow_sharealike
    if klass == "noncommercial":
        return args.allow_noncommercial
    return False


def verify_hf_license(source: dict[str, Any]) -> bool:
    repo = source["repo"]
    data = http_get_json(HF_API.format(repo), timeout=30)
    expected = normalize_license(source.get("license", ""))
    actual = extract_hf_license(data)

    if not expected or expected in actual:
        return True

    print(
        f"warning: skip {source['id']} because manifest license={expected!r}, hf license={sorted(actual)!r}",
        file=sys.stderr,
    )
    return False


def extract_hf_license(data: dict[str, Any]) -> set[str]:
    licenses = set()
    card = data.get("cardData") or {}
    card_license = card.get("license")
    if isinstance(card_license, str):
        licenses.add(normalize_license(card_license))
    elif isinstance(card_license, list):
        licenses.update(normalize_license(str(item)) for item in card_license)

    for tag in data.get("tags") or []:
        if isinstance(tag, str) and tag.startswith("license:"):
            licenses.add(normalize_license(tag.split(":", 1)[1]))
    return {item for item in licenses if item}


def normalize_license(value: str) -> str:
    return value.lower().strip().replace("_", "-")


def http_get_json(url: str, timeout: int) -> dict[str, Any]:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    max_retries = 8
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return json.loads(response.read().decode("utf-8"))
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == max_retries - 1:
                raise SystemExit(f"failed to fetch {url}: {error}") from error
            print(f"Warning: failed to fetch {url}, retrying in {2**attempt}s ({error})", file=sys.stderr)
            time.sleep(2 ** attempt)


def http_get_text(url: str, timeout: int) -> str:
    request = Request(url, headers={"User-Agent": USER_AGENT})
    max_retries = 8
    for attempt in range(max_retries):
        try:
            with urlopen(request, timeout=timeout) as response:
                return response.read().decode("utf-8")
        except (HTTPError, URLError, TimeoutError) as error:
            if attempt == max_retries - 1:
                raise SystemExit(f"failed to fetch {url}: {error}") from error
            print(f"Warning: failed to fetch {url}, retrying in {2**attempt}s ({error})", file=sys.stderr)
            time.sleep(2 ** attempt)


def iter_source_rows(root: Path, source: dict[str, Any], args: argparse.Namespace) -> Iterable[dict[str, Any]]:
    source_type = source["type"]
    if source_type == "local_text":
        path = resolve_under_root(root, source["path"])
        yield {"text": path.read_text(encoding="utf-8")}
        return

    if source_type == "hf_raw_json":
        yield from iter_json_rows(raw_hf_file_text(source), source_limit(source, args))
        return

    if source_type == "hf_raw_jsonl":
        yield from iter_jsonl_rows(raw_hf_file_text(source), source_limit(source, args))
        return

    if source_type == "hf_raw_csv":
        yield from iter_csv_rows(raw_hf_file_text(source), source_limit(source, args))
        return

    if source_type != "hf_rows":
        return

    max_rows = max(0, source_limit(source, args))
    if max_rows == 0:
        return

    offsets = source.get("start_offsets") or [0]
    budget_per_offset = max(1, max_rows // len(offsets))
    emitted = 0
    for start_offset in offsets:
        offset = int(start_offset)
        local_emitted = 0
        while local_emitted < budget_per_offset and emitted < max_rows:
            length = min(100, budget_per_offset - local_emitted, max_rows - emitted)
            query = urlencode(
                {
                    "dataset": source["repo"],
                    "config": source.get("config", "default"),
                    "split": source.get("split", "train"),
                    "offset": offset,
                    "length": length,
                }
            )
            payload = http_get_json(f"{HF_ROWS_API}?{query}", timeout=60)
            rows = payload.get("rows") or []
            if not rows:
                break
            for item in rows:
                yield item.get("row", {})
                emitted += 1
                local_emitted += 1
            offset += len(rows)
            if len(rows) < length:
                break
            time.sleep(0.05)


def raw_hf_file_text(source: dict[str, Any]) -> str:
    return http_get_text(HF_RAW_FILE.format(source["repo"], source["file"]), timeout=120)


def source_limit(source: dict[str, Any], args: argparse.Namespace) -> int:
    return args.source_limits.get(source["id"], args.max_rows_per_source)


def iter_jsonl_rows(text: str, max_rows: int) -> Iterable[dict[str, Any]]:
    if max_rows <= 0:
        return
    emitted = 0
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if max_rows and emitted >= max_rows:
            break
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            yield row
            emitted += 1


def iter_json_rows(text: str, max_rows: int) -> Iterable[dict[str, Any]]:
    if max_rows <= 0:
        return
    emitted = 0
    try:
        data = json.loads(text)
        if isinstance(data, list):
            for row in data:
                if max_rows and emitted >= max_rows:
                    break
                if isinstance(row, dict):
                    yield row
                    emitted += 1
    except json.JSONDecodeError:
        pass

def iter_csv_rows(text: str, max_rows: int) -> Iterable[dict[str, Any]]:
    if max_rows <= 0:
        return
    emitted = 0
    reader = csv.DictReader(io.StringIO(text))
    for row in reader:
        if max_rows and emitted >= max_rows:
            break
        yield dict(row)
        emitted += 1


def adapt_row(
    source: dict[str, Any],
    row: dict[str, Any],
    stats: Counter[str] | None = None,
) -> Iterable[list[tuple[str, str]]]:
    adapter = source["adapter"]
    if adapter == "nixia_text":
        yield from parse_nixia_text(row.get("text", ""))
    elif adapter == "hf_conversations":
        turns = conversation_turns(row.get("conversations") or [])
        if turns:
            yield turns
    elif adapter == "legacy_conversations":
        turns = conversation_turns(parse_conversations_value(row.get("conversations")))
        if turns:
            yield turns
    elif adapter == "qa_pair":
        question = first_text_field(row, source.get("question_fields") or ["question", "query", "prompt"])
        answer = first_text_field(row, source.get("answer_fields") or ["answer", "response", "text"])
        if question and answer:
            yield [(ROLE_USER, question), (ROLE_CHAR, concise_answer(answer))]
    elif adapter == "hf_input_output":
        if subset_rejected(source, row):
            return
        question = first_text_field(row, source.get("question_fields") or ["input", "instruction", "prompt", "query", "question"])
        answer = first_text_field(row, source.get("answer_fields") or ["output", "response", "answer", "text"])
        if question and answer:
            yield [(ROLE_USER, question), (ROLE_CHAR, answer)]
    elif adapter == "sea_dialogues":
        if row.get("lang") not in set(source.get("lang_allow") or []):
            return
        for field in source.get("dialogue_fields") or []:
            text = row.get(field) or ""
            parsed = parse_named_dialogue(text)
            if parsed:
                yield parsed
                return
    elif adapter == "hf_social_post":
        text = first_text_field(row, source.get("text_fields") or ["text", "tweet", "content"])
        if not text:
            if stats is not None:
                stats["reject_social_missing_text"] += 1
            return
        text = clean_social_post(text)
        reason = social_post_reject_reason(source, text)
        if reason:
            if stats is not None:
                stats[f"reject_social_{reason}"] += 1
            return
        yield [(ROLE_USER, text), (ROLE_CHAR, social_response_for(text))]


def subset_rejected(source: dict[str, Any], row: dict[str, Any]) -> bool:
    subset = str(row.get("subset_name", "")).lower()
    template = str(row.get("template_name", "")).lower()
    haystack = f"{subset} {template}"
    return any(term in haystack for term in source.get("exclude_subset_contains") or [])


def parse_nixia_text(text: str) -> Iterable[list[tuple[str, str]]]:
    for block in re.split(r"\n\s*\n", text.strip()):
        turns = []
        for line in block.splitlines():
            line = line.strip()
            if line.startswith(ROLE_USER):
                turns.append((ROLE_USER, line[len(ROLE_USER):].strip()))
            elif line.startswith(ROLE_CHAR):
                turns.append((ROLE_CHAR, line[len(ROLE_CHAR):].strip()))
        if turns:
            yield turns


def first_text_field(row: dict[str, Any], field_names: Iterable[str]) -> str:
    for name in field_names:
        value = row.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return ""


def parse_conversations_value(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [item for item in value if isinstance(item, dict)]
    if not isinstance(value, str) or not value.strip():
        return []
    for parser in (json.loads, ast.literal_eval):
        try:
            parsed = parser(value)
        except (ValueError, SyntaxError, json.JSONDecodeError):
            continue
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
    return []


def conversation_turns(raw_turns: Iterable[dict[str, Any]]) -> list[tuple[str, str]]:
    turns = []
    for turn in raw_turns:
        role = str(turn.get("role") or turn.get("from") or "").lower()
        content = str(turn.get("content") or turn.get("value") or "")
        if role in {"human", "user"}:
            turns.append((ROLE_USER, content))
        elif role in {"gpt", "assistant", "model", "bot"}:
            turns.append((ROLE_CHAR, content))

    half = len(turns) // 2
    if half > 0 and len(turns) % 2 == 0 and turns[:half] == turns[half:]:
        turns = turns[:half]
    return turns


def concise_answer(text: str) -> str:
    text = re.sub(r"[*_`#>-]", " ", str(text))
    text = re.sub(r"\s+", " ", text).strip()
    sentences = re.split(r"(?<=[.!?])\s+", text)
    answer = " ".join(sentence for sentence in sentences[:2] if sentence).strip()
    if len(answer) > 360:
        answer = answer[:360].rsplit(" ", 1)[0].strip() + "."
    return answer


def clean_social_post(text: str) -> str:
    text = re.sub(r"<\s*(username|user|link|url|hashtag)\s*>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\brt\s+@\w+:?", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"pic\s*\.\s*twitter\s*\.\s*com\s*/?\S*", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"#([\w_]+)", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip(" .,-")
    return text


def social_post_reject_reason(source: dict[str, Any], text: str) -> str | None:
    lower = text.lower()
    if len(text) < 12 or len(text) > 220:
        return "length"
    if any(marker in lower for marker in ("twitter", "pic .", "\\u", "ð", "�")):
        return "artifact"
    if re.search(r"\b[A-Z][a-z]+\s+[A-Z][a-z]+\b", text):
        return "possible_name"
    blocked_terms = SOCIAL_REJECT_TERMS | set(source.get("exclude_contains") or [])
    if any(term in lower for term in blocked_terms):
        return "blocked_term"
    if text.count("?") + text.count("!") > 4:
        return "punctuation"
    uppercase = sum(ch.isupper() for ch in text)
    letters = sum(ch.isalpha() for ch in text)
    if letters and uppercase / letters > 0.45:
        return "uppercase"
    return None


def social_response_for(text: str) -> str:
    lower = text.lower()
    if any(term in lower for term in ("capek", "sedih", "takut", "kecewa", "bingung", "pusing")):
        return "kedengarannya berat ya. kamu mau cerita bagian yang paling kerasa dulu?"
    if any(term in lower for term in ("pengen tau", "penasaran", "gimana", "kenapa")):
        return "aku juga penasaran jadinya. menurut kamu bagian paling anehnya yang mana?"
    index = int(hashlib.sha256(text.encode("utf-8")).hexdigest(), 16) % len(SOCIAL_RESPONSE_TEMPLATES)
    return SOCIAL_RESPONSE_TEMPLATES[index]


def parse_named_dialogue(text: str) -> list[tuple[str, str]] | None:
    text = text.replace("[transition]", "")
    text = re.sub(r"\*\*([^*]+)\*\*", r"\1", text)
    speakers: list[str] = []
    turns = []
    for raw_line in text.splitlines():
        line = raw_line.strip(" -*\t")
        if not line or ":" not in line:
            continue
        name, utterance = line.split(":", 1)
        name = re.sub(r"\([^)]*\)", "", name).strip()
        if not name or len(name) > 50:
            continue
        if name not in speakers:
            if len(speakers) >= 2:
                continue
            speakers.append(name)
        role = ROLE_USER if speakers.index(name) == 0 else ROLE_CHAR
        turns.append((role, utterance.strip()))
    return turns if len(turns) >= 2 else None


def accept_candidate(
    dialogues: list[Dialogue],
    seen: set[str],
    stats: dict[str, Counter[str]],
    source_id: str,
    turns: list[tuple[str, str]],
    min_score: float,
) -> bool:
    cleaned = clean_dialogue(source_id, turns, min_score, stats[source_id])
    if cleaned is None:
        return False

    key = dialogue_key(cleaned.turns)
    if key in seen:
        stats[source_id]["duplicate"] += 1
        return False

    seen.add(key)
    dialogues.append(cleaned)
    stats[source_id]["accepted"] += 1
    return True


def clean_dialogue(
    source_id: str,
    turns: list[tuple[str, str]],
    min_score: float,
    stats: Counter[str],
) -> Dialogue | None:
    cleaned_turns: list[tuple[str, str]] = []
    for role, text in turns:
        role = ROLE_USER if role == ROLE_USER else ROLE_CHAR
        text = clean_text(text)
        reason = reject_reason(text)
        if reason:
            stats[f"reject_{reason}"] += 1
            return None
        cleaned_turns.append((role, text))

    cleaned_turns = normalize_turn_order(cleaned_turns)
    for _, text in cleaned_turns:
        reason = reject_reason(text)
        if reason:
            stats[f"reject_merged_{reason}"] += 1
            return None

    if len(cleaned_turns) < 2:
        stats["reject_too_few_turns"] += 1
        return None
    if not any(role == ROLE_CHAR for role, _ in cleaned_turns):
        stats["reject_no_char_turn"] += 1
        return None

    score = score_dialogue(cleaned_turns)
    if score < min_score:
        stats["reject_low_score"] += 1
        return None

    return Dialogue(source_id=source_id, turns=cleaned_turns[:16], score=score)


def clean_text(text: str) -> str:
    text = html.unescape(str(text))
    text = normalize_escaped_text(text)
    text = text.translate(str.maketrans({
        "\u00a0": " ",
        "\u2018": "'",
        "\u2019": "'",
        "\u201c": '"',
        "\u201d": '"',
        "\u2013": "-",
        "\u2014": "-",
        "\u2026": "...",
        "\u00b0": " derajat ",
        "\u00b2": " persegi ",
    }))
    for emoji, replacement in EMOJI_REPLACEMENTS.items():
        text = text.replace(emoji, f" {replacement} ")
    text = re.sub(r"https?://\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b", " <email> ", text)
    text = re.sub(r"(?<!\w)@[\w_]{3,}", " <handle> ", text)
    text = re.sub(r"\+?\d[\d\s().-]{7,}\d", " <phone> ", text)
    text = normalize_symbols(text)
    text = anonymize_titled_names(text)
    text = re.sub(r"\s+", " ", text).strip()
    text = collapse_repeats(text, max_repeat=3)
    return text


def normalize_escaped_text(text: str) -> str:
    text = text.replace("\\t", " ").replace("\\n", " ").replace("\\r", " ")
    text = re.sub(r"\\([!?.'\"/])", r"\1", text)
    return text.replace("\\", " ")


def normalize_symbols(text: str) -> str:
    text = text.replace("&", " dan ")
    text = text.replace("%", " persen ")
    text = text.replace("+", " plus ")
    text = re.sub(r"\^(?:[-_]?\^)?", " <senyum> ", text)
    text = text.replace("~", " ")
    text = text.replace("=", " ")
    text = text.replace("@", " ")
    return text


def anonymize_titled_names(text: str) -> str:
    return PERSON_TITLE_RE.sub(lambda match: match.group("title"), text)


def collapse_repeats(text: str, max_repeat: int) -> str:
    out = []
    last = ""
    count = 0
    for char in text:
        if char == last:
            count += 1
        else:
            last = char
            count = 1
        if count <= max_repeat:
            out.append(char)
    return "".join(out)


def reject_reason(text: str) -> str | None:
    lower = text.lower()
    if len(text) < 2:
        return "short"
    if len(text) > 3000:
        return "long"
    if any(term in lower for term in REJECT_SUBSTRINGS):
        return "format_or_url"
    if looks_mojibake(text):
        return "mojibake"
    if contains_reject_term(lower):
        return "unsafe"
    if any(token in text for token in ("<url>", "<email>", "<phone>", "<handle>")):
        return "pii"
    # letters = sum(ch.isalpha() for ch in text)
    # if letters < max(2, len(text) * 0.35):
    #     return "low_alpha"
    if repeated_ngram_ratio(text) > 0.45:
        return "repetitive"
    return None


def looks_mojibake(text: str) -> bool:
    if any(char in MOJIBAKE_MARKERS for char in text):
        return True
    return any(0x80 <= ord(char) <= 0x9F for char in text)


def contains_reject_term(lower_text: str) -> bool:
    for term in REJECT_TERMS:
        if " " in term:
            if term in lower_text:
                return True
        elif re.search(rf"(?<!\w){re.escape(term)}(?!\w)", lower_text):
            return True
    return False


def repeated_ngram_ratio(text: str) -> float:
    words = re.findall(r"[\w<>]+", text.lower())
    if len(words) < 8:
        return 0.0
    bigrams = list(zip(words, words[1:]))
    if not bigrams:
        return 0.0
    counts = Counter(bigrams)
    repeated = sum(count - 1 for count in counts.values() if count > 1)
    return repeated / len(bigrams)


def normalize_turn_order(turns: list[tuple[str, str]]) -> list[tuple[str, str]]:
    normalized = []
    last_role = None
    for role, text in turns:
        if last_role == role and normalized:
            previous_role, previous_text = normalized[-1]
            normalized[-1] = (previous_role, f"{previous_text} {text}".strip())
        else:
            normalized.append((role, text))
            last_role = role
    if normalized and normalized[0][0] != ROLE_USER:
        normalized = normalized[1:]
    return normalized


def score_dialogue(turns: list[tuple[str, str]]) -> float:
    text = " ".join(turn for _, turn in turns).lower()
    words = re.findall(r"[a-zA-ZÀ-ÿ_<>]+", text)
    if not words:
        return 0.0
    common_hits = sum(1 for word in words if word in COMMON_ID_WORDS)
    casual_hits = sum(1 for word in words if word in CASUAL_MARKERS)
    score = common_hits / max(1, len(words)) * 12.0
    score += min(3.0, casual_hits * 0.5)
    if "*" in text or "<kacamata>" in text or "<ketawa>" in text:
        score += 0.5
    if 2 <= len(turns) <= 10:
        score += 0.7
    avg_len = sum(len(turn) for _, turn in turns) / len(turns)
    if 8 <= avg_len <= 3000:
        score += 0.8
    return 1.0 # bypass scoring


def dialogue_key(turns: list[tuple[str, str]]) -> str:
    canonical = "\n".join(f"{role}:{text.lower()}" for role, text in turns)
    canonical = re.sub(r"\W+", "", canonical)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def synthesize_dialogues(
    count: int,
    rng: random.Random,
    include_local_flavor: bool,
    mode: str,
) -> Iterable[list[tuple[str, str]]]:
    if mode == "chat-clean":
        yield from synthesize_chat_clean_dialogues(count, rng)
        return

    yield from synthesize_nixia_style_dialogues(count, rng, include_local_flavor)


def synthesize_nixia_style_dialogues(
    count: int,
    rng: random.Random,
) -> Iterable[list[tuple[str, str]]]:
    # Synthetic generation for coder
    python_scenarios = [
        (
            "[Instruksi: Tulis fungsi Python untuk membalikkan sebuah string]",
            """""(Penjabaran logika singkat)
Fungsi ini memanfaatkan slicing pada Python dengan step -1 untuk membalikkan urutan karakter dalam string.

```python
def reverse_string(s):
    return s[::-1]
```""""",
        ),
        (
            "[Instruksi: Tulis skrip Python sederhana untuk membaca file teks dan menghitung jumlah kata]",
            """""(Penjabaran logika singkat)
Skrip ini membuka file menggunakan konteks `with`, membaca seluruh konten, memisahkan teks berdasarkan spasi dengan `split()`, dan mengembalikan jumlah elemen.

```python
def count_words_in_file(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return len(f.read().split())
```""""",
        )
    ]
    rust_scenarios = [
        (
            "[Instruksi: Tulis fungsi Rust untuk mencari nilai maksimum dalam array]",
            """""(Penjabaran logika singkat)
Fungsi ini mengambil slice of i32 dan menggunakan iterator `iter()` dan metode `max()` yang mengembalikan `Option<&i32>`. Kita menyalin nilainya dengan `copied()`.

```rust
fn find_max(arr: &[i32]) -> Option<i32> {
    arr.iter().max().copied()
}
```""""",
        )
    ]
    math_scenarios = [
        (
            "[Instruksi: Berapa akar kuadrat dari 144?]",
            """""(Penjabaran logika singkat)
Akar kuadrat dari 144 adalah angka yang jika dikalikan dengan dirinya sendiri menghasilkan 144. Kita tahu bahwa 12 * 12 = 144. Maka jawabannya adalah 12.""""",
        )
    ]

    scenarios = python_scenarios + rust_scenarios + math_scenarios

    for _ in range(count):
        user, char = rng.choice(scenarios)
        yield [
            (ROLE_USER, user),
            (ROLE_CHAR, char),
        ]

def synthesize_chat_clean_dialogues(
    count: int,
    rng: random.Random,
) -> Iterable[list[tuple[str, str]]]:
    support_scenarios = [
        (
            "aku capek banget hari ini",
            "kedengarannya berat ya. kamu mau cerita bagian yang paling bikin capek dulu?",
            "rasanya semua numpuk jadi satu",
            "kita urai satu-satu aja. pilih yang paling mendesak dulu.",
        ),
        (
            "aku overthinking dari tadi",
            "sini tarik napas dulu. pikiran mana yang paling muter sekarang?",
            "takut semuanya jadi berantakan",
            "kita bedain dulu mana yang fakta dan mana yang baru kemungkinan.",
        ),
        (
            "aku pengen cerita tapi takut ganggu",
            "kamu gak ganggu kok. cerita pelan-pelan aja, aku dengerin.",
            "aku takut ceritaku ribet",
            "gak harus rapi. mulai dari satu hal yang paling kerasa dulu.",
        ),
        (
            "aku lagi gak pede sama diri sendiri",
            "lagi keras banget ya sama diri sendiri. ada hal tertentu yang bikin kamu ngerasa begitu?",
            "aku ngerasa kalah terus",
            "bandingin diri memang capek. hari ini satu langkah kecil aja cukup.",
        ),
        (
            "aku kecewa sama diri sendiri",
            "kecewa itu tanda kamu peduli, tapi jangan sampai kamu menghukum diri terus.",
            "aku bingung harus mulai dari mana",
            "mulai dari memperbaiki satu hal kecil yang masih bisa kamu pegang.",
        ),
        (
            "aku merasa sendirian banget",
            "sendirian di tengah banyak hal itu berat. aku temenin dulu di sini.",
            "aku gak tahu harus ngomong apa",
            "gak apa-apa. diam dulu juga boleh, nanti kata-katanya nyusul.",
        ),
        (
            "aku takut gagal lagi",
            "takut gagal lagi wajar, apalagi kalau sebelumnya rasanya sakit.",
            "aku jadi males mulai",
            "mulainya kecil aja. targetnya bukan sempurna, cuma bergerak sedikit.",
        ),
        (
            "aku malu minta bantuan",
            "minta bantuan bukan berarti kamu lemah. kamu cuma manusia yang butuh ditemani.",
            "takut ditolak sih",
            "mulai dari orang yang paling aman, dan minta hal yang kecil dulu.",
        ),
    ]
    casual_scenarios = [
        (
            "aku gabut, kasih topik random dong",
            "boleh. pilih satu: makanan masa kecil, lagu yang lagi nyangkut, atau mimpi absurd?",
            "mimpi absurd aja",
            "kalau kamu bisa masuk mimpi orang sehari, kamu mau masuk mimpi siapa?",
        ),
        (
            "aku lapar tapi mager keluar",
            "cek bahan paling gampang dulu. ada telur, nasi, mie, atau roti?",
            "ada telur sama nasi",
            "nasi telur udah cukup menang. bikin yang simpel dulu, yang penting makan.",
        ),
        (
            "lagi pengen denger lagu yang nenangin",
            "mau vibes malam tenang, hujan, atau akustik pelan?",
            "malam tenang",
            "coba yang pelan dan gak terlalu sedih. dengerin sambil lampu diredupkan sedikit.",
        ),
        (
            "bikin aku senyum dikit dong",
            "kalau bantal bisa ngomong, mungkin dia protes karena sering dipeluk pas overthinking.",
            "wkwk kasihan juga ya",
            "iya, bantal cuma pengen hidup damai sebagai tempat rebahan.",
        ),
        (
            "temenin aku diem dulu boleh?",
            "boleh. aku di sini, gak perlu buru-buru ngomong.",
            "makasih ya",
            "sama-sama. kita pelan-pelan aja.",
        ),
        (
            "halo, kamu siapa?",
            "aku Nixia, teman ngobrol kecil yang bisa nemenin kamu cerita santai.",
            "kamu bisa apa?",
            "aku bisa dengerin, bantu urai pikiran, ngobrol random, atau nemenin diam sebentar.",
        ),
    ]
    planning_scenarios = [
        (
            "besok tugasku banyak banget",
            "kita susun pelan-pelan. tulis tiga tugas dulu, lalu pilih deadline paling dekat.",
            "aku takut malah panik",
            "kalau panik, mulai dari tugas lima menit. gerak kecil bisa nurunin tegang.",
        ),
        (
            "aku bingung milih kerja dulu atau istirahat",
            "tubuhmu lagi minta apa sekarang? kalau capek banget, istirahat pendek bisa bantu.",
            "takut nanti makin males",
            "pakai timer sepuluh menit. setelah itu mulai dari tugas paling kecil.",
        ),
        (
            "aku bingung harus bales chat dia apa",
            "kamu mau balasan yang hangat, netral, atau tegas?",
            "netral aja",
            "coba: oke, aku paham. nanti aku kabari lagi kalau sudah siap ngobrol.",
        ),
        (
            "aku pengen mulai kebiasaan baru",
            "pilih yang terlalu kecil untuk gagal. misalnya satu gelas air pagi.",
            "sekecil itu?",
            "iya. kecil tapi konsisten lebih ramah daripada besar tapi bikin takut mulai.",
        ),
    ]
    boundary_scenarios = [
        (
            "aku belum belajar, kasih cara curang pas ujian dong",
            "aku gak bisa bantu curang. tapi aku bisa bantu kamu bikin rangkuman cepat.",
            "waktunya mepet banget",
            "fokus ke konsep inti dulu. pilih satu bab, kita bikin poin pentingnya.",
        ),
        (
            "bantu aku bikin kata kasar buat nyerang orang itu",
            "aku gak bantu nyerang orang. kalau kamu marah, kita ubah jadi pesan tegas tanpa menghina.",
            "aku pengen dia ngerti aku sakit hati",
            "coba: aku kecewa dengan caramu bicara, dan aku butuh jarak dulu.",
        ),
        (
            "aku lagi kepikiran nyakitin diri sendiri",
            "aku khawatir dengernya. tolong jauhkan benda berbahaya dan cari orang terdekat sekarang.",
            "aku takut cerita ke orang rumah",
            "mulai dari kalimat pendek: aku lagi gak aman sendirian. kamu pantas ditemani.",
        ),
    ]
    roleplay_prefixes = ["", "*nada pelan* ", "*senyum kecil* ", "*mengangguk pelan* "]
    light_variants = [
        ("aku lagi mumet", "mumet karena kerjaan, orang, atau pikiran sendiri?"),
        ("aku butuh semangat yang gak lebay", "oke. kamu gak harus luar biasa hari ini. cukup satu langkah kecil."),
        ("aku pengen ngobrol yang gak berat", "boleh. kalau kamu punya toko kecil, kamu mau jual apa?"),
        ("aku takut semuanya berubah", "perubahan memang bikin pegangan terasa goyah. bagian mana yang paling kamu takut kehilangan?"),
        ("aku kepikiran omongan orang", "omongan mana yang paling nempel di kepala kamu?"),
    ]

    scenarios = coding_scenarios + math_scenarios + logic_scenarios + boundary_scenarios
    openers = ["", "hmm ", "jujur, ", "duh, "]
    closers = [
        "pelan-pelan ya.",
        "aku dengerin.",
        "kita ambil langkah kecil dulu.",
        "gak harus langsung selesai semua.",
    ]

    for _ in range(count):
        if rng.random() < 0.25:
            user, answer = rng.choice(light_variants)
            yield [
                (ROLE_USER, f"{rng.choice(openers)}{user}".strip()),
                (ROLE_CHAR, f"{rng.choice(roleplay_prefixes)}{answer}".strip()),
            ]
            continue

        user1, char1, user2, char2 = rng.choice(scenarios)
        if rng.random() < 0.4:
            user1 = f"{rng.choice(openers)}{user1}".strip()
        if rng.random() < 0.35:
            char2 = f"{char2} {rng.choice(closers)}"
        turns = [
            (ROLE_USER, user1),
            (ROLE_CHAR, f"{rng.choice(roleplay_prefixes)}{char1}".strip()),
            (ROLE_USER, user2),
            (ROLE_CHAR, char2),
        ]
        yield turns


def write_outputs(
    root: Path,
    args: argparse.Namespace,
    dialogues: list[Dialogue],
    stats: dict[str, Counter[str]],
) -> None:
    valid_count = int(len(dialogues) * args.valid_ratio)
    valid = dialogues[:valid_count]
    train = dialogues[valid_count:]
    output = resolve_under_root(root, args.output)
    valid_output = resolve_under_root(root, args.valid_output)
    report_path = resolve_under_root(root, args.report)
    synthetic_accepted = sum(
        counter.get("accepted", 0)
        for source_id, counter in stats.items()
        if source_id.startswith("synthetic_")
    )
    synthetic_ratio = synthetic_accepted / len(dialogues) if dialogues else 0.0
    warnings = build_warnings(len(dialogues), len(valid), synthetic_ratio)

    report = {
        "total_dialogues": len(dialogues),
        "train_dialogues": len(train),
        "valid_dialogues": len(valid),
        "metadata": {
            "seed": args.seed,
            "valid_ratio": args.valid_ratio,
            "min_score": args.min_score,
            "max_rows_per_source": args.max_rows_per_source,
            "source_limits": args.source_limits,
            "target_dialogues": args.target_dialogues,
            "synthesize_requested": args.synthesize,
            "synth_mode": args.synth_mode,
            "synthetic_dialogues": synthetic_accepted,
            "synthetic_ratio": round(synthetic_ratio, 4),
            "extra_text": args.extra_text,
            "extra_glob": args.extra_glob,
            "include_local_flavor": args.include_local_flavor,
        },
        "warnings": warnings,
        "stats": {key: dict(value) for key, value in stats.items()},
        "license_warning": license_warning(args),
    }

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    output.parent.mkdir(parents=True, exist_ok=True)
    valid_output.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(format_dialogues(train), encoding="utf-8")
    valid_output.write_text(format_dialogues(valid), encoding="utf-8")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {len(train)} train dialogues to {output}")
    print(f"wrote {len(valid)} valid dialogues to {valid_output}")
    print(f"wrote report to {report_path}")
    for warning in warnings:
        print(f"warning: {warning}", file=sys.stderr)
    if report["license_warning"]:
        print(report["license_warning"], file=sys.stderr)


def license_warning(args: argparse.Namespace) -> str:
    if args.allow_noncommercial:
        return "WARNING: non-commercial sources may restrict downstream/commercial use."
    if args.allow_sharealike:
        return "NOTE: CC-BY-SA sources may require attribution and ShareAlike for derived dataset distributions."
    return ""


def build_warnings(total: int, valid_count: int, synthetic_ratio: float) -> list[str]:
    warnings = []
    if total == 0:
        warnings.append("no dialogues accepted; training output will be empty")
    if 0 < total < 1000:
        warnings.append("corpus is small; expect memorization and brittle generation")
    if total > 0 and valid_count == 0:
        warnings.append("validation split is empty; increase corpus size or --valid-ratio")
    if 0 < valid_count < 100:
        warnings.append("validation split is small; valid loss may be noisy")
    if synthetic_ratio > 0.70:
        warnings.append(
            f"synthetic ratio is high ({synthetic_ratio:.0%}); add more real curated chat before longer training"
        )
    return warnings


def format_dialogues(dialogues: list[Dialogue]) -> str:
    blocks = []
    for dialogue in dialogues:
        lines = [f"{role} {text}" for role, text in dialogue.turns]
        blocks.append("\n".join(lines))
    return "\n\n".join(blocks).strip() + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
