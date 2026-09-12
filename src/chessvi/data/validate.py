"""Kiểm tra dữ liệu đã dịch trước khi đem đi SFT.

Dịch ẩu thì mọi thứ sau đó vô nghĩa mà không ai phát hiện ra cho tới tận T9.
Bước này chạy độc lập, in bảng thống kê và ghi mẫu bị loại ra đĩa để soi tay.

    python -m chessvi.data.validate --input data/translated/sft
    python -m chessvi.data.validate --input out.jsonl --rejected data/rejected.jsonl

Kỳ vọng loại 5-15%. Trên 25% nghĩa là pipeline dịch có vấn đề, không phải dữ
liệu xui.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import chess

from chessvi.config import Paths
from chessvi.data.glossary import glossary_violations
from chessvi.data.mask import PLACEHOLDER_RE, MoveContext, find_move_tokens, parse_move
from chessvi.logging_setup import configure_logging

logger = logging.getLogger(__name__)

__all__ = [
    "ENGLISH_RUN_THRESHOLD",
    "RejectReason",
    "ValidationReport",
    "ValidationResult",
    "longest_english_run",
    "validate_dataset",
    "validate_record",
    "write_clean",
]

DEFAULT_TEXT_FIELDS: tuple[str, ...] = ("question", "answer")
DEFAULT_FEN_FIELD = "fen"
DEFAULT_LABEL_FIELD = "label"

#: Ngưỡng cảnh báo to: trên mức này thì lỗi nằm ở pipeline dịch.
ALARM_REJECT_RATE = 0.25

#: Số từ tiếng Anh liên tiếp đủ để coi là "còn nguyên một khúc chưa dịch".
ENGLISH_RUN_THRESHOLD = 8

#: Dấu thanh/nguyên âm riêng của tiếng Việt - dấu hiệu câu đã được dịch.
_VIETNAMESE_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậđèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữựỳýỷỹỵ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬĐÈÉẺẼẸÊỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰỲÝỶỸỴ"
)

#: Hư từ tiếng Anh. Một khúc dài toàn chữ không dấu mà lại chứa mấy từ này thì
#: gần như chắc chắn là tiếng Anh chưa dịch, không phải tiếng Việt viết không dấu.
_ENGLISH_STOPWORDS = frozenset(
    {
        "the", "and", "is", "of", "to", "with", "that", "this", "for", "are",
        "on", "in", "it", "as", "by", "be", "was", "has", "not", "but", "from",
        "would", "should", "can", "will", "his", "her", "their", "there",
    }
)

_WORD_RE = re.compile(r"[A-Za-zÀ-ỹ]+")


class RejectReason(StrEnum):
    """Lý do loại mẫu. Giá trị dùng luôn làm khoá trong báo cáo và file JSONL."""

    INVALID_MOVE = "invalid_move"
    LABEL_MISMATCH = "label_mismatch"
    BAD_LABEL = "bad_label"
    LEFTOVER_PLACEHOLDER = "leftover_placeholder"
    ENGLISH_CHUNK = "english_chunk"
    GLOSSARY = "glossary"
    MISSING_FEN = "missing_fen"


@dataclass(frozen=True)
class ValidationResult:
    """Kết quả kiểm tra một mẫu."""

    ok: bool
    reasons: tuple[RejectReason, ...] = ()
    details: tuple[str, ...] = ()


@dataclass
class ValidationReport:
    """Thống kê toàn bộ lần chạy."""

    total: int = 0
    passed: int = 0
    by_reason: Counter[str] = field(default_factory=Counter)

    @property
    def rejected(self) -> int:
        return self.total - self.passed

    @property
    def reject_rate(self) -> float:
        return self.rejected / self.total if self.total else 0.0

    def add(self, result: ValidationResult) -> None:
        self.total += 1
        if result.ok:
            self.passed += 1
            return
        for reason in result.reasons:
            self.by_reason[str(reason)] += 1

    def render(self) -> str:
        """Bảng thống kê dạng text, mỗi lý do một dòng kèm tỷ lệ %."""
        lines = [
            "=" * 56,
            f"{'Tổng số mẫu':<34}{self.total:>10}",
            f"{'Pass':<34}{self.passed:>10}{self._pct(self.passed):>12}",
            f"{'Loại':<34}{self.rejected:>10}{self._pct(self.rejected):>12}",
            "-" * 56,
        ]
        if self.by_reason:
            lines.append(f"{'Lý do loại (một mẫu có thể nhiều lý do)':<34}")
            for reason, count in self.by_reason.most_common():
                lines.append(f"  {reason:<32}{count:>10}{self._pct(count):>12}")
        lines.append("=" * 56)
        return "\n".join(lines)

    def _pct(self, count: int) -> str:
        return f"{count / self.total * 100:6.2f}%" if self.total else "     -"


# -- kiểm tra từng tiêu chí -----------------------------------------------


def _texts(record: dict[str, Any], fields: Sequence[str]) -> list[str]:
    return [record[name] for name in fields if isinstance(record.get(name), str)]


def _check_moves(texts: Sequence[str], fen: str) -> list[str]:
    """Nước đi trong output không parse được hoặc không hợp lệ với FEN."""
    bad: list[str] = []
    for text in texts:
        context = MoveContext(fen)
        for token in find_move_tokens(text):
            if not context.accepts(token):
                bad.append(token.text)
    return bad


def _last_move(text: str, fen: str) -> chess.Move | None:
    """Nước đi kết luận = nước hợp lệ cuối cùng xuất hiện trong text."""
    context = MoveContext(fen)
    board = chess.Board(fen)
    found: chess.Move | None = None
    for token in find_move_tokens(text):
        if not context.accepts(token):
            continue
        move = parse_move(board, token.text)
        if move is not None:
            found = move
    return found


def _parse_label(label: str, fen: str) -> chess.Move | None:
    board = chess.Board(fen)
    return parse_move(board, label)


def longest_english_run(text: str) -> int:
    """Độ dài khúc dài nhất gồm toàn từ không dấu *và* có hư từ tiếng Anh.

    Công khai vì :mod:`chessvi.data.translate` dùng lại đúng luật này làm cổng
    chất lượng ngay sau khi sinh: thứ gì T6 sắp vứt thì được thử dịch lại một
    lần trước, thay vì mất trắng cả mẫu.
    """
    best = 0
    run: list[str] = []
    for match in _WORD_RE.finditer(text):
        word = match.group(0)
        if any(char in _VIETNAMESE_CHARS for char in word):
            run = []
            continue
        run.append(word.lower())
        if sum(1 for w in run if w in _ENGLISH_STOPWORDS) >= 2:
            best = max(best, len(run))
    return best


# -- kiểm tra một mẫu -----------------------------------------------------


def validate_record(
    record: dict[str, Any],
    *,
    fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    fen_field: str = DEFAULT_FEN_FIELD,
    label_field: str | None = DEFAULT_LABEL_FIELD,
    english_run_threshold: int = ENGLISH_RUN_THRESHOLD,
) -> ValidationResult:
    """Kiểm tra một mẫu đã dịch theo 5 tiêu chí của T6.

    1. Mọi nước đi trong output parse được và hợp lệ với FEN của mẫu.
    2. Nước đi kết luận khớp với label gốc (bỏ qua nếu mẫu không có label).
    3. Không còn placeholder sót lại (``<M0>`` lọt ra ngoài là lỗi unmask).
    4. Output không còn khúc tiếng Anh dài (heuristic).
    5. Thuật ngữ nhất quán với bảng glossary.
    """
    reasons: list[RejectReason] = []
    details: list[str] = []

    fen = record.get(fen_field)
    if not isinstance(fen, str):
        return ValidationResult(False, (RejectReason.MISSING_FEN,), (f"thiếu {fen_field}",))
    try:
        chess.Board(fen)
    except ValueError:
        return ValidationResult(False, (RejectReason.MISSING_FEN,), (f"FEN hỏng: {fen}",))

    texts = _texts(record, fields)

    bad_moves = _check_moves(texts, fen)
    if bad_moves:
        reasons.append(RejectReason.INVALID_MOVE)
        details.append("nước không hợp lệ: " + ", ".join(sorted(set(bad_moves))))

    label = record.get(label_field) if label_field else None
    if isinstance(label, str) and label.strip():
        expected = _parse_label(label, fen)
        if expected is None:
            reasons.append(RejectReason.BAD_LABEL)
            details.append(f"label gốc không parse được: {label!r}")
        else:
            concluding = _last_move(texts[-1], fen) if texts else None
            if concluding is None or concluding != expected:
                reasons.append(RejectReason.LABEL_MISMATCH)
                got = concluding.uci() if concluding else "không có"
                details.append(f"nước kết luận {got} != label {expected.uci()}")

    leftovers = sorted({m.group(0) for text in texts for m in PLACEHOLDER_RE.finditer(text)})
    if leftovers:
        reasons.append(RejectReason.LEFTOVER_PLACEHOLDER)
        details.append("placeholder còn sót: " + ", ".join(leftovers))

    longest = max((longest_english_run(text) for text in texts), default=0)
    if longest >= english_run_threshold:
        reasons.append(RejectReason.ENGLISH_CHUNK)
        details.append(f"khúc tiếng Anh dài {longest} từ")

    violations = [v for text in texts for v in glossary_violations(text)]
    if violations:
        reasons.append(RejectReason.GLOSSARY)
        details.append(
            "thuật ngữ lệch chuẩn: "
            + ", ".join(sorted({f"{v.found} -> {v.expected}" for v in violations}))
        )

    return ValidationResult(not reasons, tuple(reasons), tuple(details))


def write_clean(records: Sequence[dict[str, Any]], path: Path) -> None:
    """Ghi tập mẫu đã pass ra đĩa để T8 train trên đúng dữ liệu đã validate.

    Đuôi ``.jsonl`` thì ghi JSONL, còn lại coi ``path`` là thư mục và ghi
    ``part-00000.parquet`` bên trong (khớp với cách ``iter_records`` đọc).

    Không mẫu nào pass thì **không ghi gì**. Một parquet rỗng trông y hệt một
    lần chạy thành công: ``ls`` thấy file, T8 chạy rồi mới chết vì 0 example,
    lúc đó nguyên nhân thật đã lùi lại vài bước.
    """
    if not records:
        logger.error("Không mẫu nào pass — KHÔNG ghi %s (file rỗng còn tệ hơn)", path)
        return
    if path.suffix == ".jsonl":
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    else:
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        path.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(list(records)), path / "part-00000.parquet")
    logger.info("Ghi %d mẫu đã pass vào %s", len(records), path)


def validate_dataset(
    records: Iterable[dict[str, Any]],
    *,
    fields: Sequence[str] = DEFAULT_TEXT_FIELDS,
    fen_field: str = DEFAULT_FEN_FIELD,
    label_field: str | None = DEFAULT_LABEL_FIELD,
    rejected_path: Path | None = None,
    clean_path: Path | None = None,
) -> ValidationReport:
    """Kiểm tra cả tập, ghi mẫu bị loại ra ``rejected_path`` (JSONL).

    Truyền ``clean_path`` thì ghi thêm tập đã pass ra đó — đây là đầu vào của
    T8, tránh cảnh "train trên dữ liệu đã validate" mà không có file nào chứa
    dữ liệu đó. Tập pass được giữ trong bộ nhớ tới lúc ghi.
    """
    report = ValidationReport()
    clean: list[dict[str, Any]] = []
    handle = None
    if rejected_path is not None:
        rejected_path.parent.mkdir(parents=True, exist_ok=True)
        handle = rejected_path.open("w", encoding="utf-8")
    try:
        for record in records:
            result = validate_record(
                record, fields=fields, fen_field=fen_field, label_field=label_field
            )
            report.add(result)
            if result.ok:
                if clean_path is not None:
                    clean.append(dict(record))
                continue
            if handle is not None:
                payload = dict(record)
                payload["reason"] = [str(r) for r in result.reasons]
                payload["reason_detail"] = list(result.details)
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    finally:
        if handle is not None:
            handle.close()

    if clean_path is not None:
        write_clean(clean, clean_path)
    return report


# -- nạp dữ liệu ----------------------------------------------------------


def load_records(path: Path) -> Iterator[dict[str, Any]]:
    """Đọc .jsonl, .parquet, hoặc cả thư mục parquet."""
    if path.is_dir():
        for shard in sorted(path.glob("*.parquet")):
            yield from _load_parquet(shard)
        return
    if path.suffix == ".parquet":
        yield from _load_parquet(path)
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_parquet(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    for row in pq.read_table(path).to_pylist():
        yield row


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.data.validate",
        description="Kiểm tra dữ liệu đã dịch, in thống kê, ghi mẫu bị loại.",
    )
    parser.add_argument("--input", type=Path, required=True, help="File/thư mục dữ liệu")
    parser.add_argument("--rejected", type=Path, help="Mặc định: <data_dir>/rejected.jsonl")
    parser.add_argument(
        "--out-clean",
        type=Path,
        help="Ghi tập đã pass ra đây (thư mục parquet, hoặc file .jsonl) — đầu vào của T8",
    )
    parser.add_argument("--fields", default=",".join(DEFAULT_TEXT_FIELDS))
    parser.add_argument("--fen-field", default=DEFAULT_FEN_FIELD)
    parser.add_argument("--label-field", default=DEFAULT_LABEL_FIELD)
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    configure_logging(verbose)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    rejected = args.rejected or (Paths().data_dir / "rejected.jsonl")
    report = validate_dataset(
        load_records(args.input),
        fields=tuple(args.fields.split(",")),
        fen_field=args.fen_field,
        label_field=args.label_field or None,
        rejected_path=rejected,
        clean_path=args.out_clean,
    )
    logger.info("%s", report.render())
    logger.info("Mẫu bị loại đã ghi ra %s", rejected)

    if report.total == 0:
        logger.error("Không đọc được mẫu nào từ %s", args.input)
        return 1
    if report.reject_rate > ALARM_REJECT_RATE:
        logger.warning("!" * 56)
        logger.warning(
            "!! TỶ LỆ LOẠI %.1f%% > %.0f%% — PIPELINE DỊCH CÓ VẤN ĐỀ, QUAY LẠI T5 !!",
            report.reject_rate * 100,
            ALARM_REJECT_RATE * 100,
        )
        logger.warning("!" * 56)
    if args.out_clean is not None and report.passed == 0:
        logger.error("Không có gì để train. Đọc %s xem lý do rồi quay lại T5.", rejected)
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
