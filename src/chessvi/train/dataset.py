"""Nạp và dựng mẫu huấn luyện từ dữ liệu đã qua T6.

Tách riêng khỏi ``sft.py`` để test được mà không cần torch: mọi hàm ở đây
thuần Python.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_SYSTEM_PROMPT",
    "build_example",
    "iter_records",
    "load_examples",
]

DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý cờ vua tiếng Việt. Chỉ trả lời dựa trên sự thật đã xác minh "
    "được cung cấp, không tự suy luận thêm về thế cờ. Dùng đúng thuật ngữ cờ "
    "vua tiếng Việt."
)

DEFAULT_PROMPT_FIELDS: tuple[str, ...] = ("question", "prompt", "instruction")
DEFAULT_COMPLETION_FIELDS: tuple[str, ...] = ("answer", "completion", "response")
DEFAULT_FEN_FIELD = "fen"


def iter_records(path: Path) -> Iterator[dict[str, Any]]:
    """Đọc .jsonl, .parquet hoặc cả thư mục parquet."""
    if path.is_dir():
        shards = sorted(path.glob("*.parquet"))
        if not shards:
            raise FileNotFoundError(f"không có file .parquet nào trong {path}")
        for shard in shards:
            yield from _read_parquet(shard)
        return
    if path.suffix == ".parquet":
        yield from _read_parquet(path)
        return
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _read_parquet(path: Path) -> Iterator[dict[str, Any]]:
    import pyarrow.parquet as pq  # noqa: PLC0415

    yield from pq.read_table(path).to_pylist()


def _first_str(record: dict[str, Any], names: Sequence[str]) -> str | None:
    for name in names:
        value = record.get(name)
        if isinstance(value, str) and value.strip():
            return value
    return None


def build_example(
    record: dict[str, Any],
    *,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
    prompt_fields: Sequence[str] = DEFAULT_PROMPT_FIELDS,
    completion_fields: Sequence[str] = DEFAULT_COMPLETION_FIELDS,
    fen_field: str = DEFAULT_FEN_FIELD,
) -> dict[str, Any] | None:
    """Đổi một mẫu dữ liệu thành hội thoại ``messages`` + ``prompt``/``completion``.

    Trả ``None`` nếu mẫu thiếu câu hỏi hoặc câu trả lời - caller đếm và bỏ qua.
    """
    question = _first_str(record, prompt_fields)
    answer = _first_str(record, completion_fields)
    if question is None or answer is None:
        return None

    fen = record.get(fen_field)
    user = f"FEN: {fen}\n{question}" if isinstance(fen, str) and fen.strip() else question
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user},
            {"role": "assistant", "content": answer},
        ],
        "prompt": user,
        "completion": answer,
    }


def load_examples(
    path: Path,
    *,
    limit: int | None = None,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> list[dict[str, Any]]:
    """Nạp tối đa ``limit`` mẫu đã dựng sẵn hội thoại."""
    examples: list[dict[str, Any]] = []
    skipped = 0
    for record in iter_records(path):
        example = build_example(record, system_prompt=system_prompt)
        if example is None:
            skipped += 1
            continue
        examples.append(example)
        if limit is not None and len(examples) >= limit:
            break
    if skipped:
        logger.warning("Bỏ qua %d mẫu thiếu câu hỏi hoặc câu trả lời", skipped)
    logger.info("Nạp %d mẫu từ %s", len(examples), path)
    return examples
