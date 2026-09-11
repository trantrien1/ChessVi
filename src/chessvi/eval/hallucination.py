"""Đo tỷ lệ hallucination hoàn toàn tự động.

Với mỗi câu trả lời, trích các khẳng định *kiểm chứng được* rồi đối chiếu với
fact thật từ :mod:`chessvi.engine.facts`:

- nước đi được nhắc tới có hợp lệ không;
- quân nào đang ở ô nào;
- ai đang hơn quân;
- bên đang đi có bị chiếu không;
- đến lượt ai.

Không dùng LLM để chấm LLM: mọi khẳng định đều đối chiếu với python-chess.

    python -m chessvi.eval.hallucination --answers data/answers.jsonl
    python -m chessvi.eval.hallucination --puzzles data/puzzles/test.parquet \
        --backend gguf --model-path models/chessvi-q4_k_m.gguf
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import chess

from chessvi.data.mask import MoveContext, find_move_tokens
from chessvi.engine.facts import PIECE_NAMES_VI, SIDE_VI, PositionFacts, extract_facts
from chessvi.eval.predictor import Predictor, build_predictor
from chessvi.eval.puzzle_acc import build_puzzle_prompt
from chessvi.logging_setup import configure_logging
from chessvi.train.dataset import iter_records

logger = logging.getLogger(__name__)

__all__ = [
    "Claim",
    "ClaimKind",
    "HallucinationReport",
    "check_answer",
    "evaluate_answers",
]

ClaimKind = Literal["move_legal", "piece_square", "material", "in_check", "side_to_move"]

_VI_TO_PIECE: dict[str, chess.PieceType] = {
    name: piece_type for piece_type, name in PIECE_NAMES_VI.items()
}
_PIECE_WORDS = "|".join(sorted(_VI_TO_PIECE, key=len, reverse=True))

#: "mã trắng ở f3", "hậu đen đang ở ô d8", "xe tại a1".
PIECE_SQUARE_RE = re.compile(
    rf"\b({_PIECE_WORDS})\s+(trắng|đen)?\s*(?:đang\s+)?(?:ở|tại|trên)\s+(?:ô\s+)?([a-h][1-8])\b",
    re.IGNORECASE,
)
#: "Trắng hơn 3 điểm quân", "Đen đang hơn quân".
MATERIAL_RE = re.compile(
    r"\b(trắng|đen)\s+(?:đang\s+)?(?:hơn|nhiều hơn)\s+(\d+)?\s*(?:điểm\s*)?quân",
    re.IGNORECASE,
)
MATERIAL_EQUAL_RE = re.compile(r"(?:lực lượng|quân số|cán cân)[^.]{0,20}cân bằng", re.IGNORECASE)
NOT_IN_CHECK_RE = re.compile(r"không\s+(?:đang\s+)?bị\s+chiếu", re.IGNORECASE)
IN_CHECK_RE = re.compile(r"(?<!không )(?:đang\s+)?bị\s+chiếu", re.IGNORECASE)
SIDE_TO_MOVE_RE = re.compile(r"(?:lượt|đến lượt|tới lượt)\s+(?:của\s+)?(trắng|đen)", re.IGNORECASE)
#: "ở b5", "tại ô f3" - đây là *vị trí ô*, không phải nước đi. Không được
#: đếm thành khẳng định "nước đi hợp lệ", nếu không tỷ lệ hallucination bị
#: thổi phồng bởi chính những câu mô tả bàn cờ đúng.
_SQUARE_MENTION_RE = re.compile(r"(?:ô|ở|tại|trên)\s+(?:ô\s+)?[a-h][1-8]", re.IGNORECASE)


@dataclass(frozen=True)
class Claim:
    """Một khẳng định kiểm chứng được, kèm kết luận đúng/sai."""

    kind: ClaimKind
    text: str
    is_true: bool
    detail: str = ""


@dataclass
class HallucinationReport:
    """Thống kê khẳng định sai trên cả tập."""

    answers: int = 0
    claims: list[Claim] = field(default_factory=list)
    #: Số câu trả lời có ít nhất một khẳng định sai.
    answers_with_error: int = 0

    @property
    def total_claims(self) -> int:
        return len(self.claims)

    @property
    def false_claims(self) -> int:
        return sum(1 for claim in self.claims if not claim.is_true)

    @property
    def hallucination_rate(self) -> float:
        """Tỷ lệ khẳng định sai trên tổng số khẳng định kiểm chứng được."""
        return self.false_claims / self.total_claims if self.total_claims else 0.0

    @property
    def answer_error_rate(self) -> float:
        return self.answers_with_error / self.answers if self.answers else 0.0

    def by_kind(self) -> dict[str, tuple[int, int]]:
        totals: Counter[str] = Counter()
        wrong: Counter[str] = Counter()
        for claim in self.claims:
            totals[claim.kind] += 1
            wrong[claim.kind] += int(not claim.is_true)
        return {kind: (wrong[kind], totals[kind]) for kind in sorted(totals)}

    def rows(self, model: str) -> list[dict[str, Any]]:
        out = [
            {
                "model": model,
                "kind": "overall",
                "false_claims": self.false_claims,
                "total_claims": self.total_claims,
                "hallucination_rate": round(self.hallucination_rate, 4),
                "answers": self.answers,
                "answers_with_error": self.answers_with_error,
            }
        ]
        for kind, (wrong, total) in self.by_kind().items():
            out.append(
                {
                    "model": model,
                    "kind": kind,
                    "false_claims": wrong,
                    "total_claims": total,
                    "hallucination_rate": round(wrong / total, 4) if total else 0.0,
                    "answers": "",
                    "answers_with_error": "",
                }
            )
        return out

    def render(self, model: str) -> str:
        lines = [
            "=" * 56,
            f"Model: {model}",
            f"Số câu trả lời: {self.answers}",
            f"Khẳng định kiểm chứng được: {self.total_claims}",
            f"Khẳng định SAI: {self.false_claims} "
            f"({self.hallucination_rate * 100:.2f}%)",
            f"Câu trả lời có ít nhất một lỗi: {self.answers_with_error} "
            f"({self.answer_error_rate * 100:.2f}%)",
            "-" * 56,
        ]
        for kind, (wrong, total) in self.by_kind().items():
            lines.append(f"  {kind:<16}{wrong:>5}/{total:<6}{wrong / total * 100:6.2f}%")
        lines.append("=" * 56)
        return "\n".join(lines)


# -- trích khẳng định -----------------------------------------------------


def _square_mention_spans(text: str) -> list[tuple[int, int]]:
    """Vị trí các cụm chỉ ô cờ, để không nhầm chúng với nước đi."""
    spans = [m.span() for m in _SQUARE_MENTION_RE.finditer(text)]
    spans.extend(m.span() for m in PIECE_SQUARE_RE.finditer(text))
    return spans


def _move_claims(text: str, board: chess.Board) -> list[Claim]:
    spans = _square_mention_spans(text)
    context = MoveContext(board.fen())
    claims: list[Claim] = []
    for token in find_move_tokens(text, require_chess_context=False):
        if any(start <= token.start < end for start, end in spans):
            continue
        ok = context.accepts(token)
        claims.append(
            Claim(
                kind="move_legal",
                text=token.text,
                is_true=ok,
                detail="" if ok else "nước không hợp lệ với thế cờ",
            )
        )
    return claims


def _piece_square_claims(text: str, board: chess.Board) -> list[Claim]:
    claims: list[Claim] = []
    for match in PIECE_SQUARE_RE.finditer(text):
        name, color_word, square_name = match.groups()
        piece_type = _VI_TO_PIECE[name.lower()]
        square = chess.parse_square(square_name.lower())
        actual = board.piece_at(square)

        ok = actual is not None and actual.piece_type == piece_type
        if ok and color_word:
            expected_color = chess.WHITE if color_word.lower() == "trắng" else chess.BLACK
            ok = actual is not None and actual.color == expected_color
        claims.append(
            Claim(
                kind="piece_square",
                text=match.group(0),
                is_true=ok,
                detail="" if ok else f"{square_name} thực tế là {actual or 'ô trống'}",
            )
        )
    return claims


def _material_claims(text: str, facts: PositionFacts) -> list[Claim]:
    claims: list[Claim] = []
    for match in MATERIAL_RE.finditer(text):
        side, amount = match.groups()
        diff = facts.material_diff if side.lower() == "trắng" else -facts.material_diff
        ok = diff > 0 if amount is None else diff == int(amount)
        claims.append(
            Claim(
                kind="material",
                text=match.group(0),
                is_true=ok,
                detail="" if ok else f"chênh lệch thật (theo Trắng) là {facts.material_diff}",
            )
        )
    for match in MATERIAL_EQUAL_RE.finditer(text):
        ok = facts.material_diff == 0
        claims.append(
            Claim(
                kind="material",
                text=match.group(0),
                is_true=ok,
                detail="" if ok else f"chênh lệch thật là {facts.material_diff}",
            )
        )
    return claims


def _check_claims(text: str, facts: PositionFacts) -> list[Claim]:
    negatives = list(NOT_IN_CHECK_RE.finditer(text))
    claims = [
        Claim(
            kind="in_check",
            text=match.group(0),
            is_true=not facts.in_check,
            detail="" if not facts.in_check else "thực tế đang bị chiếu",
        )
        for match in negatives
    ]
    covered = {(m.start(), m.end()) for m in negatives}
    for match in IN_CHECK_RE.finditer(text):
        if any(start <= match.start() < end for start, end in covered):
            continue
        claims.append(
            Claim(
                kind="in_check",
                text=match.group(0),
                is_true=facts.in_check,
                detail="" if facts.in_check else "thực tế không bị chiếu",
            )
        )
    return claims


def _side_claims(text: str, facts: PositionFacts) -> list[Claim]:
    claims: list[Claim] = []
    for match in SIDE_TO_MOVE_RE.finditer(text):
        side = "white" if match.group(1).lower() == "trắng" else "black"
        ok = side == facts.side_to_move
        claims.append(
            Claim(
                kind="side_to_move",
                text=match.group(0),
                is_true=ok,
                detail="" if ok else f"thực tế là lượt {SIDE_VI[facts.side_to_move]}",
            )
        )
    return claims


def check_answer(text: str, fen: str, facts: PositionFacts | None = None) -> list[Claim]:
    """Mọi khẳng định kiểm chứng được trong ``text``, kèm đúng/sai."""
    board = chess.Board(fen)
    resolved = facts if facts is not None else extract_facts(board)
    return [
        *_move_claims(text, board),
        *_piece_square_claims(text, board),
        *_material_claims(text, resolved),
        *_check_claims(text, resolved),
        *_side_claims(text, resolved),
    ]


def evaluate_answers(items: Iterable[dict[str, Any]]) -> HallucinationReport:
    """Chấm một tập ``{fen, answer}`` đã có sẵn câu trả lời."""
    report = HallucinationReport()
    for item in items:
        fen, answer = item.get("fen"), item.get("answer")
        if not isinstance(fen, str) or not isinstance(answer, str):
            logger.warning("Bỏ qua mẫu thiếu fen/answer")
            continue
        try:
            claims = check_answer(answer, fen)
        except ValueError:
            logger.warning("FEN hỏng, bỏ qua: %r", fen)
            continue
        report.answers += 1
        report.claims.extend(claims)
        if any(not claim.is_true for claim in claims):
            report.answers_with_error += 1
    return report


def generate_answers(
    puzzles: Iterable[dict[str, Any]],
    predictor: Predictor,
    *,
    limit: int | None = None,
    batch_size: int = 8,
) -> list[dict[str, Any]]:
    """Chạy model trên puzzle để lấy câu trả lời rồi mới chấm."""
    out: list[dict[str, Any]] = []
    batch: list[dict[str, Any]] = []

    def flush() -> None:
        if not batch:
            return
        answers = predictor.predict_batch([build_puzzle_prompt(i["fen"]) for i in batch])
        out.extend(
            {"fen": item["fen"], "answer": answer}
            for item, answer in zip(batch, answers, strict=True)
        )
        batch.clear()

    for item in puzzles:
        if limit is not None and len(out) + len(batch) >= limit:
            break
        if isinstance(item.get("fen"), str):
            batch.append(item)
        if len(batch) >= batch_size:
            flush()
    flush()
    return out


def write_csv(report: HallucinationReport, path: Path, model: str) -> None:
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
        prog="python -m chessvi.eval.hallucination",
        description="Tỷ lệ khẳng định sai, đối chiếu với fact từ python-chess.",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--answers", type=Path, help="JSONL có sẵn {fen, answer}")
    source.add_argument("--puzzles", type=Path, help="Chạy model trên puzzle rồi chấm")
    parser.add_argument("--backend", default="echo", choices=["echo", "gguf", "hf"])
    parser.add_argument("--model-path")
    parser.add_argument("--adapter")
    parser.add_argument("--model-name")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--out", type=Path, help="File CSV kết quả")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    model_name = args.model_name or args.model_path or args.backend
    if args.answers is not None:
        with args.answers.open(encoding="utf-8") as handle:
            items = [json.loads(line) for line in handle if line.strip()]
    else:
        if args.backend != "echo" and not args.model_path:
            logger.error("Backend %s cần --model-path", args.backend)
            return 2
        kwargs: dict[str, Any] = {}
        if args.backend == "gguf":
            kwargs["model_path"] = args.model_path
        elif args.backend == "hf":
            kwargs.update(model_path=args.model_path, adapter=args.adapter)
        predictor = build_predictor(args.backend, **kwargs)
        try:
            items = generate_answers(
                iter_records(args.puzzles),
                predictor,
                limit=args.limit,
                batch_size=args.batch_size,
            )
        finally:
            predictor.close()

    report = evaluate_answers(items)
    if report.answers == 0:
        logger.error("Không chấm được câu trả lời nào")
        return 1
    logger.info("%s", report.render(model_name))
    if args.out is not None:
        write_csv(report, args.out, model_name)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
