"""Test phần GRPO chạy được không cần GPU: sampler cân bằng và tham số.

Vòng train thật có acceptance riêng là smoke test ``--max-steps 5`` trên CPU.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

from chessvi.data.puzzles import balanced_order
from chessvi.train.grpo import SYSTEM_PROMPT, build_dataset_rows, build_parser, settings_from_args
from tests.conftest import FEN_START


def _puzzle(theme: str, bucket: str = "1200-1599", **overrides: Any) -> dict[str, Any]:
    base = {
        "fen": FEN_START,
        "solution": ["g1f3", "b8c6"],
        "primary_theme": theme,
        "rating_bucket": bucket,
    }
    base.update(overrides)
    return base


# -- sampler cân bằng theme ----------------------------------------------


def test_sampler_can_bang_theme_khi_du_lieu_lech() -> None:
    """100 puzzle theme fork, 10 theme pin - 20 mẫu đầu phải chia đều."""
    puzzles = [_puzzle("fork") for _ in range(100)] + [_puzzle("pin") for _ in range(10)]
    rows = build_dataset_rows(puzzles, limit=20)
    counts = Counter(row["primary_theme"] for row in rows)
    assert counts == {"fork": 10, "pin": 10}


def test_sampler_can_bang_ca_theo_rating_bucket() -> None:
    puzzles = [
        _puzzle(theme, bucket)
        for theme in ("fork", "pin")
        for bucket in ("800-1199", "1200-1599")
        for _ in range(10)
    ]
    rows = build_dataset_rows(puzzles, limit=8)
    counts = Counter((row["primary_theme"], row["rating_bucket"]) for row in rows)
    assert set(counts.values()) == {2}, "mỗi ô (theme, bucket) đúng 2 mẫu"


def test_sampler_o_thua_khong_chan_o_day() -> None:
    """Ô chỉ có 1 mẫu thì lấy hết rồi tiếp tục rút từ ô còn hàng."""
    puzzles = [_puzzle("fork") for _ in range(10)] + [_puzzle("pin")]
    rows = build_dataset_rows(puzzles, limit=6)
    counts = Counter(row["primary_theme"] for row in rows)
    assert counts["pin"] == 1
    assert counts["fork"] == 5


def test_balanced_order_khong_lap_va_khong_mat_mau() -> None:
    puzzles = [_puzzle(theme) for theme in ("a", "b", "c") for _ in range(4)]
    order = balanced_order(puzzles)
    assert sorted(order) == list(range(12)), "phủ hết, không trùng"


def test_balanced_order_on_dinh_theo_seed() -> None:
    puzzles = [_puzzle(theme) for theme in ("a", "b") for _ in range(6)]
    assert balanced_order(puzzles, seed=3) == balanced_order(puzzles, seed=3)


# -- dựng row -------------------------------------------------------------


def test_row_co_du_cot_cho_ham_reward() -> None:
    rows = build_dataset_rows([_puzzle("fork")])
    assert len(rows) == 1
    row = rows[0]
    assert row["fen"] == FEN_START
    assert row["solution"] == ["g1f3", "b8c6"]
    system, user = row["prompt"]
    assert system == {"role": "system", "content": SYSTEM_PROMPT}
    assert FEN_START in user["content"]


def test_prompt_day_model_ve_dung_format_ket_luan() -> None:
    assert "Nước đi:" in SYSTEM_PROMPT


def test_bo_qua_puzzle_thieu_fen_hoac_loi_giai() -> None:
    puzzles = [_puzzle("fork"), _puzzle("pin", fen=None), _puzzle("skewer", solution=[])]
    assert len(build_dataset_rows(puzzles)) == 1


def test_limit_duoc_ton_trong() -> None:
    assert len(build_dataset_rows([_puzzle("fork") for _ in range(50)], limit=7)) == 7


# -- tham số --------------------------------------------------------------


def test_mac_dinh_bat_vllm_va_push() -> None:
    settings = settings_from_args(build_parser().parse_args(["--puzzles", "p.parquet"]))
    assert settings.use_vllm is True, "rollout mặc định bằng vLLM trên Colab"
    assert settings.push_to_hub is True
    assert settings.num_generations == 4


def test_co_smoke_test_tat_vllm() -> None:
    settings = settings_from_args(
        build_parser().parse_args(
            [
                "--puzzles", "p.parquet",
                "--base-model", "Qwen/Qwen3-0.6B",
                "--limit", "16",
                "--max-steps", "5",
                "--no-vllm",
                "--no-push",
                "--no-4bit",
            ]
        )
    )
    assert settings.use_vllm is False
    assert settings.push_to_hub is False
    assert settings.load_in_4bit is False
    assert settings.max_steps == 5
    assert settings.limit == 16
