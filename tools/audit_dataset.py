#!/usr/bin/env python3
"""Audit Nixia train/valid corpora before longer training.

The audit is intentionally stdlib-only. It checks formatting, train/valid leakage,
duplicates, unsafe/noisy turns, basic conversation diversity, and build-report risks
such as an over-dominant synthetic slice.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import build_dataset

ROLE_USER = build_dataset.ROLE_USER
ROLE_CHAR = build_dataset.ROLE_CHAR
ROLE_RE = re.compile(r"^(<user>|<char>)\s*(.*)$")
WORD_RE = re.compile(r"[\w<>]+", re.UNICODE)


@dataclass
class Dialogue:
    split: str
    index: int
    turns: list[tuple[str, str]]
    raw: str


@dataclass
class Issue:
    severity: str
    split: str
    index: int
    kind: str
    message: str
    sample: str


def main() -> int:
    args = parse_args()
    root = Path.cwd()
    train_path = resolve_under_root(root, args.train)
    valid_path = resolve_under_root(root, args.valid)

    train_dialogues, train_issues = parse_corpus(train_path, "train")
    valid_dialogues, valid_issues = parse_corpus(valid_path, "valid")
    issues = train_issues + valid_issues

    report = build_audit_report(root, args, train_dialogues, valid_dialogues, issues)
    print_summary(report)

    if args.json_output:
        output = resolve_under_root(root, args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"wrote audit report to {output}")

    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Nixia train/valid corpus quality"
    )
    parser.add_argument("--train", default="data/curated/train_corpus.txt")
    parser.add_argument("--valid", default="data/curated/valid_corpus.txt")
    parser.add_argument("--build-report", default="data/curated/build_report.json")
    parser.add_argument("--json-output", default="data/curated/audit_report.json")
    parser.add_argument("--max-examples", type=int, default=20)
    return parser.parse_args()


def resolve_under_root(root: Path, path: str) -> Path:
    target = (
        (root / path).resolve()
        if not Path(path).is_absolute()
        else Path(path).resolve()
    )
    if target != root and root not in target.parents:
        raise SystemExit(f"refusing path outside project root: {target}")
    return target


def parse_corpus(path: Path, split: str) -> tuple[list[Dialogue], list[Issue]]:
    if not path.is_file():
        raise SystemExit(f"missing {split} corpus: {path}")

    text = path.read_text(encoding="utf-8")
    blocks = [block for block in re.split(r"\n\s*\n", text.strip()) if block.strip()]
    dialogues: list[Dialogue] = []
    issues: list[Issue] = []

    for index, block in enumerate(blocks, start=1):
        turns: list[tuple[str, str]] = []
        last_role = ""
        for line_no, raw_line in enumerate(block.splitlines(), start=1):
            line = raw_line.strip()
            if not line:
                continue
            match = ROLE_RE.match(line)
            if not match:
                issues.append(
                    Issue(
                        "fail",
                        split,
                        index,
                        "malformed_line",
                        f"line {line_no} does not start with <user> or <char>",
                        sample(line),
                    )
                )
                continue

            role, content = match.groups()
            content = content.strip()
            if not content:
                issues.append(
                    Issue(
                        "fail",
                        split,
                        index,
                        "empty_turn",
                        "role marker has no text",
                        sample(line),
                    )
                )
                continue
            if last_role == role:
                issues.append(
                    Issue(
                        "warn",
                        split,
                        index,
                        "consecutive_role",
                        f"two consecutive {role} turns should usually be merged",
                        sample(block),
                    )
                )
            turns.append((role, content))
            last_role = role

        dialogue = Dialogue(split, index, turns, block)
        dialogues.append(dialogue)
        issues.extend(validate_dialogue(dialogue))

    return dialogues, issues


def validate_dialogue(dialogue: Dialogue) -> list[Issue]:
    issues: list[Issue] = []
    roles = [role for role, _ in dialogue.turns]

    if not dialogue.turns:
        return [
            Issue(
                "fail",
                dialogue.split,
                dialogue.index,
                "empty_dialogue",
                "dialogue has no turns",
                "",
            )
        ]
    if len(dialogue.turns) < 2:
        issues.append(
            Issue(
                "fail",
                dialogue.split,
                dialogue.index,
                "too_few_turns",
                "dialogue needs at least 2 turns",
                sample(dialogue.raw),
            )
        )
    if roles[0] != ROLE_USER:
        issues.append(
            Issue(
                "warn",
                dialogue.split,
                dialogue.index,
                "starts_with_char",
                "dialogue should usually start with <user>",
                sample(dialogue.raw),
            )
        )
    if ROLE_USER not in roles:
        issues.append(
            Issue(
                "fail",
                dialogue.split,
                dialogue.index,
                "missing_user",
                "dialogue has no <user> turn",
                sample(dialogue.raw),
            )
        )
    if ROLE_CHAR not in roles:
        issues.append(
            Issue(
                "fail",
                dialogue.split,
                dialogue.index,
                "missing_char",
                "dialogue has no <char> turn",
                sample(dialogue.raw),
            )
        )

    for role, text in dialogue.turns:
        cleaned = build_dataset.clean_text(text)
        reason = build_dataset.reject_reason(cleaned)
        if reason:
            issues.append(
                Issue(
                    "fail",
                    dialogue.split,
                    dialogue.index,
                    f"reject_{reason}",
                    f"{role} turn fails clean-data filter: {reason}",
                    sample(text),
                )
            )
        if len(text) > 280:
            issues.append(
                Issue(
                    "warn",
                    dialogue.split,
                    dialogue.index,
                    "long_turn_for_tiny_lm",
                    "turn is long for a tiny 96-token-context model",
                    sample(text),
                )
            )

    return issues


def build_audit_report(
    root: Path,
    args: argparse.Namespace,
    train: list[Dialogue],
    valid: list[Dialogue],
    issues: list[Issue],
) -> dict[str, Any]:
    train_keys = keys_for(train)
    valid_keys = keys_for(valid)
    duplicate_train = duplicate_count(train_keys)
    duplicate_valid = duplicate_count(valid_keys)
    overlap = sorted(set(train_keys) & set(valid_keys))
    split_stats = {
        "train": split_stats_for(train),
        "valid": split_stats_for(valid),
    }
    build_report = read_build_report(root, args.build_report)
    synthetic_ratio = extract_synthetic_ratio(build_report)

    checks = quality_checks(
        train_count=len(train),
        valid_count=len(valid),
        duplicate_train=duplicate_train,
        duplicate_valid=duplicate_valid,
        overlap_count=len(overlap),
        fail_issues=sum(1 for issue in issues if issue.severity == "fail"),
        synthetic_ratio=synthetic_ratio,
        train_multi_turn_ratio=split_stats["train"]["multi_turn_ratio"],
        valid_multi_turn_ratio=split_stats["valid"]["multi_turn_ratio"],
        top_char_prefix_ratio=max_prefix_ratio(train + valid),
    )
    status = worst_status(check["status"] for check in checks)
    issue_counts = Counter(issue.kind for issue in issues)

    return {
        "status": status,
        "readiness": readiness(status, len(train), len(valid), synthetic_ratio),
        "files": {
            "train": args.train,
            "valid": args.valid,
            "build_report": args.build_report if build_report else None,
        },
        "splits": split_stats,
        "leakage": {
            "train_duplicates": duplicate_train,
            "valid_duplicates": duplicate_valid,
            "train_valid_overlap": len(overlap),
        },
        "build_report_summary": summarize_build_report(build_report),
        "quality_checks": checks,
        "issue_counts": dict(issue_counts),
        "issue_examples": [
            issue_to_dict(issue) for issue in issues[: args.max_examples]
        ],
        "recommendations": recommendations(checks, build_report, synthetic_ratio),
    }


def keys_for(dialogues: list[Dialogue]) -> list[str]:
    return [build_dataset.dialogue_key(dialogue.turns) for dialogue in dialogues]


def duplicate_count(keys: list[str]) -> int:
    counts = Counter(keys)
    return sum(count - 1 for count in counts.values() if count > 1)


def split_stats_for(dialogues: list[Dialogue]) -> dict[str, Any]:
    turn_counts = [len(dialogue.turns) for dialogue in dialogues]
    turn_lengths = [len(text) for dialogue in dialogues for _, text in dialogue.turns]
    char_prefixes = char_response_prefixes(dialogues)
    prefix_counts = Counter(char_prefixes)
    top_prefixes = prefix_counts.most_common(8)

    return {
        "dialogues": len(dialogues),
        "turns": sum(turn_counts),
        "avg_turns": round(mean(turn_counts), 2),
        "multi_turn_dialogues": sum(1 for count in turn_counts if count >= 4),
        "multi_turn_ratio": round(
            ratio(sum(1 for count in turn_counts if count >= 4), len(dialogues)), 4
        ),
        "avg_chars_per_turn": round(mean(turn_lengths), 2),
        "p95_chars_per_turn": percentile(turn_lengths, 95),
        "max_chars_per_turn": max(turn_lengths) if turn_lengths else 0,
        "distinct_user_openers": distinct_prefix_count(dialogues, ROLE_USER),
        "distinct_char_openers": distinct_prefix_count(dialogues, ROLE_CHAR),
        "top_char_prefixes": [
            {"prefix": prefix, "count": count} for prefix, count in top_prefixes
        ],
    }


def char_response_prefixes(dialogues: list[Dialogue]) -> list[str]:
    return [
        turn_prefix(text)
        for dialogue in dialogues
        for role, text in dialogue.turns
        if role == ROLE_CHAR
    ]


def distinct_prefix_count(dialogues: list[Dialogue], role_filter: str) -> int:
    prefixes = {
        turn_prefix(text)
        for dialogue in dialogues
        for role, text in dialogue.turns
        if role == role_filter
    }
    prefixes.discard("")
    return len(prefixes)


def turn_prefix(text: str, words: int = 5) -> str:
    text = re.sub(r"\*[^*]{1,40}\*", " ", text.lower())
    tokens = WORD_RE.findall(text)
    return " ".join(tokens[:words])


def max_prefix_ratio(dialogues: list[Dialogue]) -> float:
    prefixes = char_response_prefixes(dialogues)
    if not prefixes:
        return 0.0
    return max(Counter(prefixes).values()) / len(prefixes)


def read_build_report(root: Path, path: str) -> dict[str, Any] | None:
    report_path = resolve_under_root(root, path)
    if not report_path.is_file():
        return None
    return json.loads(report_path.read_text(encoding="utf-8"))


def extract_synthetic_ratio(report: dict[str, Any] | None) -> float | None:
    if not report:
        return None
    metadata = report.get("metadata") or {}
    if isinstance(metadata.get("synthetic_ratio"), (int, float)):
        return float(metadata["synthetic_ratio"])
    stats = report.get("stats") or {}
    synthetic = (stats.get("synthetic_nixia_style") or {}).get("accepted") or 0
    total = report.get("total_dialogues") or 0
    return synthetic / total if total else None


def summarize_build_report(report: dict[str, Any] | None) -> dict[str, Any] | None:
    if not report:
        return None
    stats = report.get("stats") or {}
    accepted_by_source = {
        source: values.get("accepted", 0)
        for source, values in stats.items()
        if isinstance(values, dict) and values.get("accepted", 0)
    }
    return {
        "total_dialogues": report.get("total_dialogues"),
        "train_dialogues": report.get("train_dialogues"),
        "valid_dialogues": report.get("valid_dialogues"),
        "synthetic_ratio": extract_synthetic_ratio(report),
        "warnings": report.get("warnings") or [],
        "accepted_by_source": accepted_by_source,
    }


def quality_checks(
    *,
    train_count: int,
    valid_count: int,
    duplicate_train: int,
    duplicate_valid: int,
    overlap_count: int,
    fail_issues: int,
    synthetic_ratio: float | None,
    train_multi_turn_ratio: float,
    valid_multi_turn_ratio: float,
    top_char_prefix_ratio: float,
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    add_check(
        checks,
        fail_issues == 0,
        "format_and_content",
        "fail",
        f"{fail_issues} fail-level issue(s)",
    )
    add_check(
        checks,
        duplicate_train == 0,
        "train_duplicates",
        "warn",
        f"{duplicate_train} duplicate train dialogue(s)",
    )
    add_check(
        checks,
        duplicate_valid == 0,
        "valid_duplicates",
        "warn",
        f"{duplicate_valid} duplicate valid dialogue(s)",
    )
    add_check(
        checks,
        overlap_count == 0,
        "train_valid_overlap",
        "fail",
        f"{overlap_count} duplicate dialogue(s) across train/valid",
    )
    add_check(
        checks,
        train_count >= 5_000,
        "train_size_target",
        "warn",
        f"train has {train_count}; target 5k+ curated dialogues",
    )
    add_check(
        checks,
        valid_count >= 500,
        "valid_size_target",
        "warn",
        f"valid has {valid_count}; target 500+ held-out dialogues",
    )
    add_check(
        checks,
        train_multi_turn_ratio >= 0.35,
        "train_multi_turn",
        "warn",
        f"train multi-turn ratio {train_multi_turn_ratio:.1%}; target 35%+",
    )
    add_check(
        checks,
        valid_multi_turn_ratio >= 0.35,
        "valid_multi_turn",
        "warn",
        f"valid multi-turn ratio {valid_multi_turn_ratio:.1%}; target 35%+",
    )
    add_check(
        checks,
        top_char_prefix_ratio <= 0.08,
        "response_template_repetition",
        "warn",
        f"top char prefix ratio {top_char_prefix_ratio:.1%}; target <= 8%",
    )
    if synthetic_ratio is not None:
        add_check(
            checks,
            synthetic_ratio <= 0.30,
            "synthetic_ratio",
            "warn",
            f"synthetic ratio {synthetic_ratio:.1%}; target <= 30%",
        )
    else:
        checks.append(
            {
                "name": "synthetic_ratio",
                "status": "warn",
                "message": "missing build report; cannot measure synthetic ratio",
            }
        )
    return checks


def add_check(
    checks: list[dict[str, Any]], ok: bool, name: str, fail_status: str, message: str
) -> None:
    checks.append(
        {
            "name": name,
            "status": "pass" if ok else fail_status,
            "message": "ok" if ok else message,
        }
    )


def worst_status(statuses: Any) -> str:
    order = {"pass": 0, "warn": 1, "fail": 2}
    return max(statuses, key=lambda status: order.get(status, 0), default="pass")


def readiness(
    status: str, train_count: int, valid_count: int, synthetic_ratio: float | None
) -> str:
    if status == "fail":
        return "fix_required_before_training"
    if train_count < 1_000 or valid_count < 100:
        return "smoke_only"
    if synthetic_ratio is not None and synthetic_ratio > 0.70:
        return "smoke_or_short_finetune_only"
    if train_count < 5_000 or valid_count < 500:
        return "small_finetune_candidate"
    return "longer_training_candidate"


def recommendations(
    checks: list[dict[str, Any]],
    build_report: dict[str, Any] | None,
    synthetic_ratio: float | None,
) -> list[str]:
    recs = []
    failed = {check["name"] for check in checks if check["status"] != "pass"}
    if "format_and_content" in failed:
        recs.append(
            "Fix fail-level corpus rows before training; the model will learn any broken format you leave in."
        )
    if "train_valid_overlap" in failed:
        recs.append(
            "Rebuild the split so validation has no duplicate dialogues from training."
        )
    if "train_size_target" in failed or "valid_size_target" in failed:
        recs.append(
            "Collect more real multi-turn conversations, then rebuild with a 5-10% validation split."
        )
    if synthetic_ratio is None:
        recs.append(
            "Keep build_report.json with the corpus so synthetic/real source mix stays auditable."
        )
    elif synthetic_ratio > 0.30:
        recs.append(
            "Reduce synthetic dominance: keep synthetic rows as style seed, not the main corpus."
        )
    if "response_template_repetition" in failed:
        recs.append(
            "Add varied answers and remove repeated assistant openings/templates."
        )
    if "train_multi_turn" in failed or "valid_multi_turn" in failed:
        recs.append(
            "Add more 4-10 turn dialogues so the model sees context, not only single-turn QA."
        )
    if build_report and (build_report.get("warnings") or []):
        recs.append("Review warnings already emitted by tools/build_dataset.py.")
    return recs


def issue_to_dict(issue: Issue) -> dict[str, Any]:
    return {
        "severity": issue.severity,
        "split": issue.split,
        "index": issue.index,
        "kind": issue.kind,
        "message": issue.message,
        "sample": issue.sample,
    }


def print_summary(report: dict[str, Any]) -> None:
    train = report["splits"]["train"]
    valid = report["splits"]["valid"]
    leakage = report["leakage"]
    print(f"dataset audit: status={report['status']} readiness={report['readiness']}")
    print(
        "split: "
        f"train={train['dialogues']} dialogues, valid={valid['dialogues']} dialogues, "
        f"overlap={leakage['train_valid_overlap']}"
    )
    summary = report.get("build_report_summary")
    if summary:
        print(
            f"source mix: synthetic_ratio={summary['synthetic_ratio']:.1%} accepted_by_source={summary['accepted_by_source']}"
        )
    else:
        print("source mix: missing build report")

    problems = [
        check for check in report["quality_checks"] if check["status"] != "pass"
    ]
    if problems:
        print("checks needing attention:")
        for check in problems:
            print(f"- {check['status']}: {check['name']}: {check['message']}")
    else:
        print("checks: all pass")

    if report["recommendations"]:
        print("recommendations:")
        for rec in report["recommendations"]:
            print(f"- {rec}")


def sample(text: str, max_len: int = 180) -> str:
    text = re.sub(r"\s+", " ", text).strip()
    return text if len(text) <= max_len else f"{text[: max_len - 1]}…"


def mean(values: list[int]) -> float:
    return sum(values) / len(values) if values else 0.0


def ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def percentile(values: list[int], percent: int) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = round((len(ordered) - 1) * percent / 100)
    return ordered[index]


if __name__ == "__main__":
    raise SystemExit(main())
