"""Chuẩn hoá C1-data về schema mà pipeline T5 → T6 → T8 hiểu.

``UofTCSSLab/C1-data`` (paper *Grounded Chess Reasoning in Language Models via
Master Distillation*, arXiv:2603.20510) là nguồn giải thích tiếng Anh của dự án
này. Schema của nó **không** khớp với thứ pipeline giả định:

==============  ===========================================================
C1-data (sft)   Pipeline cần
==============  ===========================================================
``instruction`` ``question`` + ``fen`` — FEN nằm CHÌM trong câu văn
``input``       (luôn rỗng)
``output``      ``answer`` + ``label`` — nước đi nằm sau ``FINAL_ANSWER:``
==============  ===========================================================

Thiếu cột ``fen`` thì :mod:`chessvi.data.validate` loại 100% mẫu với lý do
``missing_fen``; thiếu ``label`` thì mất luôn phép đối chiếu nước kết luận —
tức mất phần lớn giá trị của T6.

Module này là bước đứng **trước** T5::

    python -m chessvi.data.c1 --split sft --out data/raw/c1_sft.jsonl
    python -m chessvi.data.translate --input-jsonl data/raw/c1_sft.jsonl \\
        --split sft --backend hf --out-dir data/translated

Tách rời thay vì nhét vào ``translate.py`` vì hai lý do: translate không phải
biết gì về một dataset cụ thể, và bước chuẩn hoá này test được mà không cần
mạng.

Chỉ config ``sft`` được hỗ trợ. ``rl`` và ``test`` mang schema khác hẳn
(``prompt`` + ``reward_model.ground_truth`` + ``extra_info.fen``), không có
đoạn giải thích nào để dịch — chúng dành cho RL và eval, không phải T5.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any

import chess

from chessvi.config import Paths
from chessvi.logging_setup import configure_logging

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_DATASET",
    "QUESTION_EN",
    "SUPPORTED_SPLITS",
    "extract_fen",
    "extract_label",
    "normalize_record",
    "normalize_rows",
]

DEFAULT_DATASET = "UofTCSSLab/C1-data"

#: ``--split`` -> (config, split thật bên trong dataset).
SUPPORTED_SPLITS: dict[str, tuple[str, str]] = {"sft": ("sft", "train")}

#: Câu hỏi dùng chung cho mọi mẫu, sẽ được T5 dịch sang tiếng Việt.
#:
#: Vì sao không giữ nguyên ``instruction`` gốc: nó gồm ~95% là **dữ liệu máy
#: sinh** — FEN, danh sách vị trí từng quân, danh sách nước hợp lệ. Ba thứ đó
#: :mod:`chessvi.engine.facts` tự dựng lại bằng tiếng Việt từ FEN lúc phục vụ,
#: nên dịch chúng vừa tốn vô ích vừa vi phạm nguyên tắc 3 trong CLAUDE.md
#: (ký hiệu cờ không đi qua máy dịch). Phần người đọc thật sự cần chỉ là câu
#: hỏi, và nó giống nhau ở mọi mẫu.
QUESTION_EN = (
    "Find the best move for the side to play. "
    "Analyze step by step and explain your reasoning."
)

# FEN đủ 6 trường, bám ngay sau "in FEN:". Bắt đủ 6 trường thay vì "mọi thứ tới
# dấu chấm" vì bản thân FEN kết thúc bằng số và liền sau nó là dấu chấm câu.
_FEN_RE = re.compile(
    r"in FEN:\s*"
    r"(?P<fen>[1-8rnbqkpRNBQKP/]+ [wb] (?:[KQkq]{1,4}|-) (?:[a-h][36]|-) \d+ \d+)"
)

# Nước đi UCI sau FINAL_ANSWER. Lấy khớp CUỐI CÙNG: phần hướng dẫn có thể nhắc
# lại nhãn này như một khuôn mẫu, còn đáp án thật luôn nằm ở cuối.
#
# IGNORECASE vì dữ liệu thật viết thường nhưng không có gì bảo đảm điều đó ở
# mọi mẫu. Nới ra không rủi ro: ký tự ô và chữ số xen kẽ nên không thể khớp
# nhầm một từ tiếng Anh. Kết quả luôn được hạ về chữ thường trước khi parse.
_LABEL_RE = re.compile(
    r"FINAL_ANSWER:\s*(?P<uci>[a-h][1-8][a-h][1-8][qrbn]?)", re.IGNORECASE
)


def extract_fen(text: str) -> str:
    """Rút FEN khỏi câu ``"... in FEN: <fen>. Piece positions: ..."``.

    Ném ``ValueError`` nếu không tìm thấy hoặc FEN không dựng được bàn cờ —
    caller đếm và bỏ mẫu, không bao giờ tạo ra record rác.
    """
    match = _FEN_RE.search(text)
    if match is None:
        raise ValueError("không tìm thấy FEN trong instruction")
    fen = match.group("fen")
    chess.Board(fen)  # ValueError nếu FEN hỏng
    return fen


def extract_label(text: str, fen: str) -> str:
    """Rút nước đi sau ``FINAL_ANSWER:`` và khẳng định nó hợp lệ với ``fen``.

    Nước không hợp lệ nghĩa là mẫu đó hỏng: lời giải thích đang dẫn tới một
    nước không đi được, train vào chỉ dạy model bịa. Ném ``ValueError``.
    """
    matches = _LABEL_RE.findall(text)
    if not matches:
        raise ValueError("không tìm thấy FINAL_ANSWER trong output")
    uci = matches[-1].lower()

    board = chess.Board(fen)
    try:
        move = chess.Move.from_uci(uci)
    except ValueError as error:
        raise ValueError(f"FINAL_ANSWER {uci!r} không phải UCI hợp lệ") from error
    if move not in board.legal_moves:
        raise ValueError(f"FINAL_ANSWER {uci} không hợp lệ với {fen}")
    return move.uci()


def normalize_record(row: dict[str, Any], *, index: int = 0) -> dict[str, Any]:
    """Đưa một row C1-data về ``{id, fen, question, answer, label}``.

    Đây chính là schema của ``tests/fixtures/c1_sample.jsonl`` và là thứ
    ``translate.py`` / ``validate.py`` / ``train/dataset.py`` đang chờ.
    """
    instruction = row.get("instruction")
    output = row.get("output")
    if not isinstance(instruction, str) or not isinstance(output, str):
        raise ValueError(
            "row không có instruction/output dạng chuỗi — có phải config 'sft' không?"
        )

    fen = extract_fen(instruction)
    label = extract_label(output, fen)
    return {
        "id": row.get("id", index),
        "fen": fen,
        "question": QUESTION_EN,
        "answer": output.strip(),
        "label": label,
    }


def normalize_rows(
    rows: Iterable[dict[str, Any]], *, limit: int | None = None
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Chuẩn hoá cả luồng row. Trả ``(records, skipped)`` với lý do bỏ qua."""
    records: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if limit is not None and len(records) >= limit:
            break
        try:
            records.append(normalize_record(row, index=index))
        except ValueError as error:
            skipped[_reason(error)] += 1
            logger.debug("Bỏ row %d: %s", index, error)
    return records, skipped


def _reason(error: ValueError) -> str:
    """Gom lỗi về vài nhãn ngắn để bảng thống kê đọc được."""
    message = str(error)
    if "FEN" in message and "không tìm thấy" in message:
        return "missing_fen"
    if "FINAL_ANSWER" in message and "không tìm thấy" in message:
        return "missing_label"
    if "không hợp lệ" in message:
        return "illegal_label"
    return "other"


# -- nạp và ghi -----------------------------------------------------------


def load_rows(dataset: str, split: str) -> Iterator[dict[str, Any]]:
    """Stream một config của C1-data.

    C1-data dùng **config** chứ không phải split để tách sft/rl/test, nên phải
    truyền tên config làm tham số thứ hai của ``load_dataset`` — gọi
    ``load_dataset(id, split="sft")`` sẽ hỏng.
    """
    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"chỉ hỗ trợ {sorted(SUPPORTED_SPLITS)}; {split!r} mang schema khác "
            f"(prompt + reward_model), dành cho RL/eval chứ không phải T5"
        )
    config, inner_split = SUPPORTED_SPLITS[split]

    from datasets import load_dataset  # noqa: PLC0415

    logger.info("Streaming %s config=%s split=%s", dataset, config, inner_split)
    for row in load_dataset(dataset, config, split=inner_split, streaming=True):
        yield dict(row)


def write_jsonl(records: Sequence[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    logger.info("Ghi %d mẫu vào %s", len(records), path)


def render_report(records: Sequence[dict[str, Any]], skipped: Counter[str]) -> str:
    total = len(records) + sum(skipped.values())
    lines = [
        "=" * 56,
        f"{'Tổng số row đọc':<40}{total:>10}",
        f"{'Chuẩn hoá được':<40}{len(records):>10}",
        f"{'Bỏ qua':<40}{sum(skipped.values()):>10}",
    ]
    if skipped:
        lines.append("-" * 56)
        for reason, count in skipped.most_common():
            lines.append(f"  {reason:<38}{count:>10}")
    lines.append("=" * 56)
    return "\n".join(lines)


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.data.c1",
        description="Chuẩn hoá C1-data thành JSONL cho bước dịch (T5).",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="sft", choices=sorted(SUPPORTED_SPLITS))
    parser.add_argument("--input-jsonl", type=Path, help="File local thay cho HF")
    parser.add_argument("--limit", type=int, help="Chỉ lấy N mẫu đầu (smoke test)")
    parser.add_argument("--out", type=Path, help="Mặc định: <data_dir>/raw/c1_<split>.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in thống kê")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _local_rows(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    rows = (
        _local_rows(args.input_jsonl)
        if args.input_jsonl is not None
        else load_rows(args.dataset, args.split)
    )
    records, skipped = normalize_rows(rows, limit=args.limit)
    logger.info("%s", render_report(records, skipped))

    if not records:
        logger.error("Không chuẩn hoá được mẫu nào")
        return 1
    if args.dry_run:
        logger.info("(dry-run, không ghi file)")
        logger.info("Mẫu đầu tiên: %s", json.dumps(records[0], ensure_ascii=False)[:400])
        return 0

    out = args.out or (Paths().data_dir / "raw" / f"c1_{args.split}.jsonl")
    write_jsonl(records, out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
