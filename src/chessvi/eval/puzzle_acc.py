"""Đo accuracy giải puzzle, tách theo Rating bucket và Theme.

    python -m chessvi.eval.puzzle_acc --puzzles data/puzzles/test.parquet \
        --backend gguf --model-path models/chessvi-q4_k_m.gguf --out reports/acc.csv

Chốt chặn sau T8: accuracy phải đạt 30–38%. Dưới 20% là dữ liệu có vấn đề.
Chốt chặn sau T9: RL phải hơn SFT >= 4 điểm.
"""

from __future__ import annotations

import argparse
import csv
import logging
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chess

from chessvi.eval.predictor import Predictor, build_predictor
from chessvi.logging_setup import configure_logging
from chessvi.train.dataset import iter_records
from chessvi.train.grpo import SYSTEM_PROMPT, USER_TEMPLATE
from chessvi.train.reward import extract_move

logger = logging.getLogger(__name__)

__all__ = [
    "AccuracyReport",
    "PuzzleResult",
    "build_puzzle_prompt",
    "evaluate_puzzles",
    "write_csv",
]


@dataclass(frozen=True)
class PuzzleResult:
    """Kết quả một puzzle."""

    puzzle_id: str
    rating_bucket: str
    theme: str
    expected: str
    predicted: str | None
    correct: bool
    legal: bool


@dataclass
class AccuracyReport:
    """Accuracy tổng và tách theo bucket/theme."""

    results: list[PuzzleResult] = field(default_factory=list)

    @property
    def total(self) -> int:
        return len(self.results)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.results if r.correct)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def legal_rate(self) -> float:
        """Tỷ lệ trả về nước hợp lệ, kể cả sai - đo riêng với accuracy."""
        if not self.total:
            return 0.0
        return sum(1 for r in self.results if r.legal) / self.total

    def _group(self, attribute: str) -> dict[str, tuple[int, int]]:
        totals: Counter[str] = Counter()
        hits: Counter[str] = Counter()
        for result in self.results:
            key = getattr(result, attribute)
            totals[key] += 1
            hits[key] += int(result.correct)
        return {key: (hits[key], totals[key]) for key in sorted(totals)}

    def by_bucket(self) -> dict[str, tuple[int, int]]:
        return self._group("rating_bucket")

    def by_theme(self) -> dict[str, tuple[int, int]]:
        return self._group("theme")

    def rows(self, model: str) -> list[dict[str, Any]]:
        """Dòng CSV: một dòng tổng, rồi tách theo bucket và theme."""
        out: list[dict[str, Any]] = [
            {
                "model": model,
                "slice_type": "overall",
                "slice": "all",
                "correct": self.correct,
                "total": self.total,
                "accuracy": round(self.accuracy, 4),
                "legal_rate": round(self.legal_rate, 4),
            }
        ]
        for slice_type, groups in (("rating_bucket", self.by_bucket()), ("theme", self.by_theme())):
            for name, (hits, total) in groups.items():
                out.append(
                    {
                        "model": model,
                        "slice_type": slice_type,
                        "slice": name,
                        "correct": hits,
                        "total": total,
                        "accuracy": round(hits / total, 4) if total else 0.0,
                        "legal_rate": "",
                    }
                )
        return out

    def render(self, model: str) -> str:
        lines = [
            "=" * 56,
            f"Model: {model}",
            f"Accuracy tổng: {self.correct}/{self.total} = {self.accuracy * 100:.2f}%",
            f"Tỷ lệ trả nước hợp lệ: {self.legal_rate * 100:.2f}%",
            "-" * 56,
            "Theo rating bucket:",
        ]
        for name, (hits, total) in self.by_bucket().items():
            lines.append(f"  {name:<16}{hits:>5}/{total:<6}{hits / total * 100:6.2f}%")
        lines.append("Theo theme:")
        for name, (hits, total) in self.by_theme().items():
            lines.append(f"  {name:<24}{hits:>5}/{total:<6}{hits / total * 100:6.2f}%")
        lines.append("=" * 56)
        return "\n".join(lines)


def build_puzzle_prompt(fen: str) -> str:
    """Prompt hỏi nước mạnh nhất — đúng format model đã học ở T9."""
    return f"{SYSTEM_PROMPT}\n\n{USER_TEMPLATE.format(fen=fen)}\n\n[TRẢ LỜI]\n"


def _first_solution_move(board: chess.Board, solution: Any) -> chess.Move | None:
    moves = solution.split() if isinstance(solution, str) else list(solution or [])
    if not moves:
        return None
    try:
        move = chess.Move.from_uci(moves[0])
    except ValueError:
        return None
    return move if move in board.legal_moves else None


def evaluate_puzzles(
    puzzles: Iterable[dict[str, Any]],
    predictor: Predictor,
    *,
    limit: int | None = None,
    batch_size: int = 8,
) -> AccuracyReport:
    """Chạy ``predictor`` trên tập puzzle, trả về báo cáo accuracy."""
    report = AccuracyReport()
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch:
            return
        prompts = [build_puzzle_prompt(item["fen"]) for item in batch]
        answers = predictor.predict_batch(prompts)
        for item, answer in zip(batch, answers, strict=True):
            board = chess.Board(item["fen"])
            expected = _first_solution_move(board, item.get("solution"))
            predicted = extract_move(answer, board)
            report.results.append(
                PuzzleResult(
                    puzzle_id=str(item.get("puzzle_id", "")),
                    rating_bucket=str(item.get("rating_bucket", "?")),
                    theme=str(item.get("primary_theme", "?")),
                    expected=expected.uci() if expected else "",
                    predicted=predicted.uci() if predicted else None,
                    correct=predicted is not None and predicted == expected,
                    legal=predicted is not None,
                )
            )
        batch.clear()

    for item in puzzles:
        if limit is not None and report.total + len(batch) >= limit:
            break
        if not isinstance(item.get("fen"), str):
            logger.warning("Bỏ qua puzzle thiếu FEN: %s", item.get("puzzle_id"))
            continue
        batch.append(item)
        if len(batch) >= batch_size:
            flush()
    flush()
    return report


def write_csv(report: AccuracyReport, path: Path, model: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report.rows(model)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Ghi %d dòng vào %s", len(rows), path)


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.eval.puzzle_acc",
        description="Accuracy giải puzzle, tách theo rating bucket và theme.",
    )
    parser.add_argument("--puzzles", type=Path, required=True, help="test.parquet từ T7")
    parser.add_argument("--backend", default="echo", choices=["echo", "gguf", "hf"])
    parser.add_argument("--model-path", help="File GGUF hoặc id/thư mục model HF")
    parser.add_argument("--adapter", help="Adapter LoRA (backend hf)")
    parser.add_argument("--model-name", help="Tên ghi vào CSV, mặc định theo --model-path")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Trần an toàn cho backend hf. Model dừng sớm ngay khi xuất xong "
        "dòng FINAL_ANSWER, nên trần này chỉ chặn trường hợp nó không bao giờ "
        "kết luận. Hạ xuống ~320 nếu muốn chặn cứng thời gian chạy.",
    )
    parser.add_argument("--out", type=Path, help="File CSV kết quả")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _build(args: argparse.Namespace) -> Predictor:
    if args.backend == "echo":
        return build_predictor("echo")
    if args.backend == "gguf":
        return build_predictor("gguf", model_path=args.model_path)
    return build_predictor(
        "hf",
        model_path=args.model_path,
        adapter=args.adapter,
        max_new_tokens=args.max_new_tokens,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    if args.backend != "echo" and not args.model_path:
        logger.error("Backend %s cần --model-path", args.backend)
        return 2

    model_name = args.model_name or args.model_path or args.backend
    predictor = _build(args)
    try:
        report = evaluate_puzzles(
            iter_records(args.puzzles), predictor, limit=args.limit, batch_size=args.batch_size
        )
    finally:
        predictor.close()

    if report.total == 0:
        logger.error("Không đọc được puzzle nào từ %s", args.puzzles)
        return 1

    logger.info("%s", report.render(model_name))
    if args.out is not None:
        write_csv(report, args.out, model_name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
