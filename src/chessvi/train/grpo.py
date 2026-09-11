"""GRPO (RLVR) trên puzzle set. Chạy trên Colab, KHÔNG chạy local.

Rollout mặc định bằng **vLLM** (bắt buộc trên Colab để rollout không nghẽn).
vLLM sẽ OOM trên 4GB VRAM local — smoke test dùng ``--no-vllm``.

    # smoke test trên CPU
    python -m chessvi.train.grpo --puzzles data/puzzles/train.parquet \
        --base-model Qwen/Qwen3-0.6B --limit 16 --max-steps 5 \
        --no-vllm --no-push --output-dir /tmp/grpo-smoke

    # chạy thật trên Colab
    python -m chessvi.train.grpo --puzzles data/puzzles/train.parquet \
        --adapter outputs/sft --hub-model-id <user>/chessvi-4b-grpo

Hàm reward nằm ở :mod:`chessvi.train.reward` và có unit test riêng không cần
GPU — đó là phần dễ sai nhất của cả pipeline RL.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chessvi.config import hf_token
from chessvi.data.puzzles import balanced_order
from chessvi.train.dataset import iter_records
from chessvi.train.reward import ANSWER_TEMPLATE, puzzle_reward
from chessvi.train.sft import (
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_RANK,
    _bf16_supported,
    _compute_dtype,
)

logger = logging.getLogger(__name__)

__all__ = ["GRPOSettings", "build_dataset_rows", "build_parser", "main", "run_grpo"]

DEFAULT_BASE_MODEL = "Qwen/Qwen3-4B"
DEFAULT_SEED = 20240911

SYSTEM_PROMPT = (
    "Bạn là trợ lý cờ vua tiếng Việt. Hãy tìm nước đi mạnh nhất cho bên đang đi. "
    "Giải thích ngắn gọn rồi kết thúc bằng đúng một dòng theo mẫu:\n"
    + ANSWER_TEMPLATE.format(move="<nước đi>")
)

USER_TEMPLATE = "FEN: {fen}\nBên đang đi cần tìm nước mạnh nhất. Nước đi nào?"


@dataclass(frozen=True)
class GRPOSettings:
    """Tham số một lần chạy GRPO."""

    puzzles: Path
    output_dir: Path
    base_model: str = DEFAULT_BASE_MODEL
    #: Adapter LoRA từ T8 để RL tiếp, None thì RL thẳng từ base model.
    adapter: Path | None = None
    limit: int | None = None
    max_steps: int = -1
    epochs: float = 1.0
    batch_size: int = 4
    grad_accum: int = 4
    num_generations: int = 4
    max_completion_length: int = 512
    learning_rate: float = 1e-6
    beta: float = 0.04
    temperature: float = 1.0
    logging_steps: int = 5
    save_steps: int = 100
    seed: int = DEFAULT_SEED
    use_vllm: bool = True
    push_to_hub: bool = True
    hub_model_id: str | None = None
    load_in_4bit: bool | None = None
    resume: bool = False


# -- dựng dataset ---------------------------------------------------------


def build_dataset_rows(
    puzzles: Sequence[dict[str, Any]],
    *,
    limit: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[dict[str, Any]]:
    """Dựng row cho GRPOTrainer, thứ tự cân bằng theo theme.

    Dùng lại :func:`chessvi.data.puzzles.balanced_order` (logic của T7) nên
    batch RL không bị một theme áp đảo. Cột ``fen`` và ``solution`` đi kèm để
    hàm reward nhận được qua kwargs.
    """
    order = balanced_order(puzzles, size=limit, seed=seed)
    rows: list[dict[str, Any]] = []
    for index in order:
        puzzle = puzzles[index]
        fen = puzzle.get("fen")
        solution = puzzle.get("solution")
        if not isinstance(fen, str) or not solution:
            continue
        rows.append(
            {
                "prompt": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": USER_TEMPLATE.format(fen=fen)},
                ],
                "fen": fen,
                "solution": list(solution),
                "primary_theme": puzzle.get("primary_theme", ""),
                "rating_bucket": puzzle.get("rating_bucket", ""),
            }
        )
    logger.info("Dựng %d mẫu RL từ %d puzzle", len(rows), len(puzzles))
    return rows


# -- chạy -----------------------------------------------------------------


def _has_cuda() -> bool:
    import torch  # noqa: PLC0415

    return bool(torch.cuda.is_available())


def _load_model(settings: GRPOSettings, use_4bit: bool) -> Any:
    import torch  # noqa: PLC0415
    from transformers import AutoModelForCausalLM  # noqa: PLC0415

    kwargs: dict[str, Any] = {"dtype": _compute_dtype() if use_4bit else torch.float32}
    if use_4bit:
        from transformers import BitsAndBytesConfig  # noqa: PLC0415

        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=_compute_dtype(),
        )
        kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(settings.base_model, **kwargs)
    if settings.adapter is not None:
        from peft import PeftModel  # noqa: PLC0415

        logger.info("Nạp adapter SFT từ %s", settings.adapter)
        model = PeftModel.from_pretrained(model, str(settings.adapter), is_trainable=True)
    return model


def run_grpo(settings: GRPOSettings) -> Path:
    """Chạy một lần GRPO. Trả về thư mục output."""
    from datasets import Dataset  # noqa: PLC0415
    from peft import LoraConfig  # noqa: PLC0415
    from transformers import AutoTokenizer  # noqa: PLC0415
    from trl import GRPOConfig, GRPOTrainer  # noqa: PLC0415

    use_4bit = settings.load_in_4bit if settings.load_in_4bit is not None else _has_cuda()
    # bf16 chỉ từ Ampere trở lên; T4 của Colab phải fp16 (xem _bf16_supported).
    use_bf16 = use_4bit and _bf16_supported()
    use_fp16 = use_4bit and not use_bf16

    push = settings.push_to_hub
    if push and (hf_token() is None or settings.hub_model_id is None):
        logger.warning("Thiếu HF_TOKEN hoặc --hub-model-id — tắt push_to_hub")
        push = False

    rows = build_dataset_rows(
        list(iter_records(settings.puzzles)), limit=settings.limit, seed=settings.seed
    )
    if not rows:
        raise ValueError(f"không dựng được mẫu RL nào từ {settings.puzzles}")
    dataset = Dataset.from_list(rows)

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = GRPOConfig(
        output_dir=str(settings.output_dir),
        per_device_train_batch_size=settings.batch_size,
        gradient_accumulation_steps=settings.grad_accum,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        num_generations=settings.num_generations,
        max_completion_length=settings.max_completion_length,
        learning_rate=settings.learning_rate,
        beta=settings.beta,
        temperature=settings.temperature,
        num_train_epochs=settings.epochs,
        max_steps=settings.max_steps,
        bf16=use_bf16,
        fp16=use_fp16,
        logging_steps=settings.logging_steps,
        save_strategy="steps",
        save_steps=settings.save_steps,
        save_total_limit=2,
        report_to=[],
        seed=settings.seed,
        use_vllm=settings.use_vllm,
        push_to_hub=push,
        hub_strategy="checkpoint",
        hub_private_repo=True,
        hub_model_id=settings.hub_model_id,
        hub_token=hf_token(),
    )

    peft_config = (
        None
        if settings.adapter is not None
        else LoraConfig(
            r=LORA_RANK,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules="all-linear",
        )
    )

    trainer = GRPOTrainer(
        model=_load_model(settings, use_4bit),
        reward_funcs=[puzzle_reward],
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
    )
    trainer.train(resume_from_checkpoint=settings.resume or None)
    trainer.save_model(str(settings.output_dir))
    if push:
        trainer.push_to_hub()
    logger.info("Đã lưu model RL vào %s", settings.output_dir)
    return settings.output_dir


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.train.grpo",
        description="GRPO trên puzzle set, reward kiểm chứng bằng python-chess.",
    )
    parser.add_argument("--puzzles", type=Path, required=True, help="train.parquet từ T7")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/grpo"))
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter", type=Path, help="Adapter LoRA từ T8")
    parser.add_argument("--limit", type=int, help="Chỉ dùng N puzzle (smoke test)")
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--num-generations", type=int, default=4)
    parser.add_argument("--max-completion-length", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--beta", type=float, default=0.04)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--hub-model-id")
    parser.add_argument("--no-push", action="store_true")
    parser.add_argument(
        "--no-vllm", action="store_true", help="Tắt vLLM (bắt buộc khi smoke test CPU)"
    )
    quant = parser.add_mutually_exclusive_group()
    quant.add_argument("--4bit", dest="four_bit", action="store_true", default=None)
    quant.add_argument("--no-4bit", dest="four_bit", action="store_false")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def settings_from_args(args: argparse.Namespace) -> GRPOSettings:
    return GRPOSettings(
        puzzles=args.puzzles,
        output_dir=args.output_dir,
        base_model=args.base_model,
        adapter=args.adapter,
        limit=args.limit,
        max_steps=args.max_steps,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        learning_rate=args.learning_rate,
        beta=args.beta,
        save_steps=args.save_steps,
        seed=args.seed,
        use_vllm=not args.no_vllm,
        push_to_hub=not args.no_push,
        hub_model_id=args.hub_model_id,
        load_in_4bit=args.four_bit,
        resume=args.resume,
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, handlers=[handler], force=True
    )
    run_grpo(settings_from_args(args))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
