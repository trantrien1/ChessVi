"""Cấu hình tập trung. Không hardcode đường dẫn ở bất kỳ module nào khác."""

from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

_ENV_PREFIX = "CHESSVI_"


def _env(name: str, default: str) -> str:
    return os.environ.get(_ENV_PREFIX + name, default)


@dataclass(frozen=True)
class Paths:
    """Đường dẫn tới binary engine và thư mục dữ liệu.

    `stockfish_bin` / `maia_bin` có thể là tên lệnh trên PATH hoặc đường dẫn
    tuyệt đối. Dùng :meth:`resolve_stockfish` / :meth:`resolve_maia` để lấy
    đường dẫn thật, trả về ``None`` nếu không tìm thấy (để test skip gọn thay
    vì fail).
    """

    stockfish_bin: str = field(default_factory=lambda: _env("STOCKFISH_BIN", "stockfish"))
    maia_bin: str = field(default_factory=lambda: _env("MAIA_BIN", "maia3"))
    data_dir: Path = field(default_factory=lambda: Path(_env("DATA_DIR", "data")))
    model_dir: Path = field(default_factory=lambda: Path(_env("MODEL_DIR", "models")))

    @staticmethod
    def _resolve(command: str) -> str | None:
        found = shutil.which(command)
        if found is not None:
            return found
        candidate = Path(command)
        if candidate.is_file():
            return str(candidate)
        return None

    def resolve_stockfish(self) -> str | None:
        """Đường dẫn tuyệt đối tới Stockfish, hoặc ``None`` nếu không có."""
        return self._resolve(self.stockfish_bin)

    def resolve_maia(self) -> str | None:
        """Đường dẫn tuyệt đối tới Maia, hoặc ``None`` nếu không có."""
        return self._resolve(self.maia_bin)

    def ensure_data_dir(self) -> Path:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        return self.data_dir


@dataclass(frozen=True)
class EngineConfig:
    """Tham số mặc định khi gọi engine."""

    analyse_depth: int = 18
    #: Ngưỡng Elo: <= thì dùng Maia (sai giống người), > thì dùng Stockfish.
    maia_elo_ceiling: int = 2200
    min_elo: int = 800
    max_elo: int = 2600
    #: Thời gian tối đa cho một lần gọi engine, giây.
    move_time: float = 0.5


@dataclass(frozen=True)
class ServeConfig:
    """Tham số cho đường inference GGUF. 4GB VRAM thật — Q4, ctx <= 4096.

    Chỉ chi phối backend GGUF. Chatbot đã chuyển sang 14B chạy trên máy có GPU
    (lý do và số đo trong CLAUDE.md), nên các tham số ở đây không còn là trần
    của cả hệ thống — chúng là trần của bản chạy local.
    """

    gguf_path: Path = field(default_factory=lambda: Path(_env("GGUF_PATH", "models/chessvi-q4_k_m.gguf")))
    n_ctx: int = 4096
    n_gpu_layers: int = -1
    cache_type_k: str = "q8_0"
    cache_type_v: str = "q8_0"
    temperature: float = 0.3
    max_tokens: int = 512
    #: Số lần regenerate tối đa khi guard bắt được nước đi không hợp lệ.
    max_regenerations: int = 2


@dataclass(frozen=True)
class Config:
    paths: Paths = field(default_factory=Paths)
    engine: EngineConfig = field(default_factory=EngineConfig)
    serve: ServeConfig = field(default_factory=ServeConfig)


def load_config() -> Config:
    """Đọc cấu hình từ biến môi trường (prefix ``CHESSVI_``)."""
    cfg = Config()
    logger.debug("Đã nạp cấu hình: %s", cfg)
    return cfg


def hf_token() -> str | None:
    """Token Hugging Face từ env. Không bao giờ hardcode."""
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
