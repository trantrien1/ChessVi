"""Pipeline dịch C1-data sang tiếng Việt.

Luồng một mẫu: ``mask -> dịch -> unmask -> enforce_glossary``.

Chạy:

    python -m chessvi.data.translate --split sft --limit 20 --dry-run
    python -m chessvi.data.translate --split sft --dataset <hf_id> --backend hf
    python -m chessvi.data.translate --split sft --dataset <hf_id> --resume

Colab/Kaggle hay đứt giữa chừng nên pipeline ghi checkpoint mỗi
``--checkpoint-every`` mẫu; ``--resume`` đọc lại state và chạy tiếp.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from abc import ABC, abstractmethod
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from chessvi.config import Paths
from chessvi.data.glossary import enforce_glossary
from chessvi.data.mask import mask, unmask

logger = logging.getLogger(__name__)

__all__ = [
    "EchoTranslator",
    "HFTranslator",
    "TranslationJob",
    "Translator",
    "build_translator",
    "translate_record",
    "translate_records",
]

#: Các trường text thường gặp trong C1-data. Dùng khi không truyền --fields.
DEFAULT_TEXT_FIELDS: tuple[str, ...] = (
    "question",
    "answer",
    "prompt",
    "completion",
    "reasoning",
    "explanation",
    "text",
)

DEFAULT_FEN_FIELD = "fen"
DEFAULT_CHECKPOINT_EVERY = 500
DEFAULT_BATCH_SIZE = 16
DEFAULT_HF_MODEL = "VietAI/envit5-translation"


# -- backend dịch ---------------------------------------------------------


class Translator(ABC):
    """Giao diện backend dịch. Đổi backend không đụng tới pipeline."""

    name: str = "abstract"

    @abstractmethod
    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        """Dịch cả batch. Trả về đúng số phần tử như đầu vào."""

    def translate(self, text: str) -> str:
        return self.translate_batch([text])[0]


class EchoTranslator(Translator):
    """Không dịch gì, trả nguyên văn.

    Dùng để smoke test pipeline (kể cả trên CI không mạng, không GPU) và để
    kiểm tra riêng phần mask/unmask/glossary mà không lẫn nhiễu từ model.
    """

    name = "echo"

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        return list(texts)


class HFTranslator(Translator):
    """Dịch bằng model seq2seq trên Hugging Face (chạy ở Kaggle/Colab).

    Import ``transformers`` ở trong ``__init__`` để máy local không cài
    transformers vẫn dùng được phần còn lại của module.
    """

    name = "hf"

    def __init__(
        self,
        model_name: str = DEFAULT_HF_MODEL,
        *,
        device: str | None = None,
        max_new_tokens: int = 512,
        prefix: str = "en: ",
    ) -> None:
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer  # noqa: PLC0415

        import torch  # noqa: PLC0415

        self._max_new_tokens = max_new_tokens
        self._prefix = prefix
        resolved = device or ("cuda" if torch.cuda.is_available() else "cpu")
        logger.info("Nạp model dịch %s trên %s", model_name, resolved)
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(resolved)
        self._device = resolved

    def translate_batch(self, texts: Sequence[str]) -> list[str]:
        if not texts:
            return []
        import torch  # noqa: PLC0415

        batch = self._tokenizer(
            [self._prefix + text for text in texts],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_new_tokens,
        ).to(self._device)
        with torch.no_grad():
            generated = self._model.generate(**batch, max_new_tokens=self._max_new_tokens)
        decoded = self._tokenizer.batch_decode(generated, skip_special_tokens=True)
        # envit5 trả về tiền tố "vi: " ở đầu câu.
        return [line.removeprefix("vi: ").strip() for line in decoded]


def build_translator(backend: str, *, model_name: str = DEFAULT_HF_MODEL) -> Translator:
    """Tạo backend theo tên. Thêm backend mới thì khai báo ở đây."""
    if backend == "echo":
        return EchoTranslator()
    if backend == "hf":
        return HFTranslator(model_name)
    raise ValueError(f"Backend không biết: {backend!r} (có: echo, hf)")


# -- dịch ------------------------------------------------------------------


def translate_records(
    records: Sequence[dict[str, Any]],
    translator: Translator,
    fields: Sequence[str],
    *,
    fen_field: str | None = DEFAULT_FEN_FIELD,
) -> list[dict[str, Any]]:
    """Dịch các trường text của cả một batch mẫu, giữ nguyên mọi ký hiệu cờ.

    Toàn bộ text của cả batch đi vào **một** lời gọi ``translate_batch`` — gọi
    từng mẫu một thì GPU ở Kaggle chạy không hết công suất.

    Ký hiệu cờ được mask trước khi gọi model và khôi phục sau đó, nên model
    dịch không bao giờ nhìn thấy ``Nf3`` hay FEN. Mẫu nào có trường FEN thì
    dùng luôn làm ngữ cảnh xác thực nước đi (chỉ mask nước hợp lệ).
    """
    slots: list[tuple[int, str, dict[str, str]]] = []
    masked_texts: list[str] = []
    for index, record in enumerate(records):
        context_fen = None
        if fen_field and isinstance(record.get(fen_field), str):
            context_fen = record[fen_field]
        for name in fields:
            if not isinstance(record.get(name), str):
                continue
            masked_text, mapping = _mask_safely(record[name], context_fen)
            slots.append((index, name, mapping))
            masked_texts.append(masked_text)

    translated = translator.translate_batch(masked_texts) if masked_texts else []
    if len(translated) != len(masked_texts):
        raise RuntimeError(
            f"Backend {translator.name} trả {len(translated)} kết quả cho "
            f"{len(masked_texts)} đầu vào"
        )

    out = [dict(record) for record in records]
    for (index, name, mapping), text in zip(slots, translated, strict=True):
        out[index][name] = enforce_glossary(unmask(text, mapping))
    return out


def translate_record(
    record: dict[str, Any],
    translator: Translator,
    fields: Sequence[str],
    *,
    fen_field: str | None = DEFAULT_FEN_FIELD,
) -> dict[str, Any]:
    """Dịch một mẫu. Bọc mỏng quanh :func:`translate_records`."""
    return translate_records([record], translator, fields, fen_field=fen_field)[0]


def _mask_safely(text: str, context_fen: str | None) -> tuple[str, dict[str, str]]:
    """Mask có ngữ cảnh; FEN hỏng thì lùi về mask không ngữ cảnh."""
    if context_fen is None:
        return mask(text)
    try:
        return mask(text, context_fen)
    except ValueError:
        logger.warning("FEN ngữ cảnh không hợp lệ (%r), mask không ngữ cảnh", context_fen)
        return mask(text)


# -- nạp dữ liệu ----------------------------------------------------------


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _load_hf(dataset: str, split: str) -> Iterator[dict[str, Any]]:
    from datasets import load_dataset  # noqa: PLC0415

    logger.info("Nạp dataset %s (split=%s) từ Hugging Face", dataset, split)
    stream = load_dataset(dataset, split=split, streaming=True)
    for row in stream:
        yield dict(row)


def _detect_fields(record: dict[str, Any], requested: Sequence[str] | None) -> list[str]:
    if requested:
        return list(requested)
    detected = [name for name in DEFAULT_TEXT_FIELDS if isinstance(record.get(name), str)]
    if not detected:
        raise ValueError(
            f"Không tự nhận ra trường text nào trong {sorted(record)}; dùng --fields"
        )
    logger.info("Tự nhận trường text: %s", ", ".join(detected))
    return detected


# -- job ------------------------------------------------------------------


@dataclass
class TranslationJob:
    """Một lần chạy dịch, có checkpoint để ngắt giữa chừng chạy tiếp được."""

    translator: Translator
    out_dir: Path
    split: str
    fields: Sequence[str] | None = None
    fen_field: str | None = DEFAULT_FEN_FIELD
    batch_size: int = DEFAULT_BATCH_SIZE
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    max_retries: int = 3
    retry_backoff: float = 2.0
    _buffer: list[dict[str, Any]] = field(default_factory=list, init=False, repr=False)

    @property
    def state_path(self) -> Path:
        return self.out_dir / self.split / "_state.json"

    def read_state(self) -> dict[str, Any]:
        if not self.state_path.exists():
            return {"done": 0, "shards": 0}
        return json.loads(self.state_path.read_text(encoding="utf-8"))

    def _write_state(self, done: int, shards: int) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(
            json.dumps({"done": done, "shards": shards}, ensure_ascii=False),
            encoding="utf-8",
        )

    def _translate_with_retry(
        self, records: Sequence[dict[str, Any]], fields: Sequence[str]
    ) -> list[dict[str, Any]]:
        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            try:
                return translate_records(
                    records, self.translator, fields, fen_field=self.fen_field
                )
            except Exception as error:  # noqa: BLE001 - backend nào cũng có thể ném
                last_error = error
                wait = self.retry_backoff ** (attempt - 1)
                logger.warning(
                    "Batch lỗi (lần %d/%d): %s — thử lại sau %.1fs",
                    attempt,
                    self.max_retries,
                    error,
                    wait,
                )
                time.sleep(wait)
        raise RuntimeError(f"Batch thất bại sau {self.max_retries} lần") from last_error

    def _flush(self, shard_index: int) -> bool:
        """Ghi buffer ra một shard parquet. Trả về True nếu thật sự có ghi."""
        if not self._buffer:
            return False
        import pyarrow as pa  # noqa: PLC0415
        import pyarrow.parquet as pq  # noqa: PLC0415

        target = self.out_dir / self.split / f"part-{shard_index:05d}.parquet"
        target.parent.mkdir(parents=True, exist_ok=True)
        pq.write_table(pa.Table.from_pylist(self._buffer), target)
        logger.info("Ghi %d mẫu vào %s", len(self._buffer), target)
        self._buffer.clear()
        return True

    def run(
        self,
        records: Iterable[dict[str, Any]],
        *,
        limit: int | None = None,
        resume: bool = False,
        dry_run: bool = False,
        dry_run_samples: int = 20,
    ) -> int:
        """Chạy job, trả về số mẫu đã xử lý trong lần chạy này."""
        state = self.read_state() if resume else {"done": 0, "shards": 0}
        skip: int = state["done"]
        shards: int = state["shards"]
        if skip:
            logger.info("Resume: bỏ qua %d mẫu đã dịch", skip)

        fields: Sequence[str] | None = self.fields
        processed = 0
        taken = 0
        shown = 0
        batch: list[dict[str, Any]] = []

        for index, record in enumerate(records):
            if index < skip:
                continue
            if limit is not None and taken >= limit:
                break
            if fields is None:
                fields = _detect_fields(record, self.fields)
            batch.append(record)
            taken += 1
            if len(batch) < self.batch_size and not (limit is not None and taken >= limit):
                continue
            processed, shown, shards = self._handle_batch(
                batch, fields, processed, shown, shards, skip, dry_run, dry_run_samples
            )
            batch = []

        if batch and fields is not None:
            processed, shown, shards = self._handle_batch(
                batch, fields, processed, shown, shards, skip, dry_run, dry_run_samples
            )

        if not dry_run:
            if self._flush(shards):
                shards += 1
            self._write_state(skip + processed, shards)
        logger.info("Xong %d mẫu%s", processed, " (dry-run, không ghi file)" if dry_run else "")
        return processed

    def _handle_batch(
        self,
        batch: list[dict[str, Any]],
        fields: Sequence[str],
        processed: int,
        shown: int,
        shards: int,
        skip: int,
        dry_run: bool,
        dry_run_samples: int,
    ) -> tuple[int, int, int]:
        translated = self._translate_with_retry(batch, fields)
        if dry_run:
            shown = _show_pairs(batch, translated, fields, shown, dry_run_samples)
        else:
            self._buffer.extend(translated)
        processed += len(batch)
        if not dry_run and len(self._buffer) >= self.checkpoint_every:
            if self._flush(shards):
                shards += 1
            self._write_state(skip + processed, shards)
        return processed, shown, shards


def _show_pairs(
    before: Sequence[dict[str, Any]],
    after: Sequence[dict[str, Any]],
    fields: Sequence[str],
    shown: int,
    limit: int,
) -> int:
    """In cặp before/after ra stdout cho người đọc kiểm tra bằng mắt."""
    for source, result in zip(before, after, strict=True):
        if shown >= limit:
            return shown
        shown += 1
        logger.info("--- mẫu %d ---", shown)
        for name in fields:
            if isinstance(source.get(name), str):
                logger.info("[%s] EN: %s", name, source[name])
                logger.info("[%s] VI: %s", name, result[name])
    return shown


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.data.translate",
        description="Dịch C1-data sang tiếng Việt, giữ nguyên ký hiệu cờ.",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--dataset", help="ID dataset trên Hugging Face (C1-data)")
    source.add_argument("--input-jsonl", type=Path, help="File .jsonl local thay cho HF")
    parser.add_argument("--split", default="sft", help="Split cần dịch (mặc định: sft)")
    parser.add_argument("--limit", type=int, help="Chỉ xử lý N mẫu đầu")
    parser.add_argument("--backend", default="echo", choices=["echo", "hf"])
    parser.add_argument("--model", default=DEFAULT_HF_MODEL, help="Model cho backend hf")
    parser.add_argument("--fields", help="Danh sách trường cần dịch, phân tách bằng dấu phẩy")
    parser.add_argument("--fen-field", default=DEFAULT_FEN_FIELD)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--checkpoint-every", type=int, default=DEFAULT_CHECKPOINT_EVERY)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--out-dir", type=Path, help="Mặc định: <data_dir>/translated")
    parser.add_argument("--resume", action="store_true", help="Chạy tiếp từ checkpoint")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="In 20 cặp before/after ra stdout, không ghi file",
    )
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        handlers=[handler],
        force=True,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    if args.input_jsonl is not None:
        records: Iterable[dict[str, Any]] = _load_jsonl(args.input_jsonl)
    elif args.dataset:
        records = _load_hf(args.dataset, args.split)
    else:
        logger.error("Cần --dataset <hf_id> hoặc --input-jsonl <file>")
        return 2

    out_dir = args.out_dir or (Paths().data_dir / "translated")
    job = TranslationJob(
        translator=build_translator(args.backend, model_name=args.model),
        out_dir=out_dir,
        split=args.split,
        fields=tuple(args.fields.split(",")) if args.fields else None,
        fen_field=args.fen_field or None,
        batch_size=args.batch_size,
        checkpoint_every=args.checkpoint_every,
        max_retries=args.max_retries,
    )
    job.run(records, limit=args.limit, resume=args.resume, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
