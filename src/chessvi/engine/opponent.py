"""Lớp đối thủ: một API duy nhất ``get_move(fen, target_elo)``.

Routing Maia/Stockfish (lý do trong CLAUDE.md): Stockfish yếu đi bằng nước
ngẫu nhiên kỳ quặc — người chơi cảm thấy máy "tự sát" chứ không thấy mình
thắng. Maia sai giống người thật. Nên Elo thấp đi Maia, Elo cao (vượt mức Maia
mô phỏng được) đi Stockfish với Skill Level và depth scale theo Elo.
"""

from __future__ import annotations

import logging
from types import TracebackType
from typing import Literal

import chess

from chessvi.config import EngineConfig, Paths
from chessvi.engine.maia import MaiaEngine
from chessvi.engine.stockfish import StockfishEngine

logger = logging.getLogger(__name__)

__all__ = ["Opponent", "depth_for_elo", "route_engine", "skill_for_elo"]

EngineChoice = Literal["maia", "stockfish"]

#: Biên scale Stockfish cho khoảng Elo trên ngưỡng Maia.
_SKILL_AT_CEILING = 10
_SKILL_AT_MAX = 20
_DEPTH_AT_CEILING = 8
_DEPTH_AT_MAX = 18


def route_engine(elo: int, config: EngineConfig | None = None) -> EngineChoice:
    """Engine nào phụ trách mức Elo này."""
    cfg = config or EngineConfig()
    return "maia" if elo <= cfg.maia_elo_ceiling else "stockfish"


def _interpolate(elo: int, low_value: int, high_value: int, config: EngineConfig) -> float:
    """Nội suy tuyến tính từ ngưỡng Maia tới Elo tối đa."""
    low_elo, high_elo = config.maia_elo_ceiling, config.max_elo
    if high_elo <= low_elo:  # pragma: no cover - cấu hình vô lý
        return float(high_value)
    ratio = (min(max(elo, low_elo), high_elo) - low_elo) / (high_elo - low_elo)
    return low_value + ratio * (high_value - low_value)


def skill_for_elo(elo: int, config: EngineConfig | None = None) -> int:
    """``Skill Level`` UCI cho Stockfish ứng với mức Elo (0–20)."""
    cfg = config or EngineConfig()
    value = round(_interpolate(elo, _SKILL_AT_CEILING, _SKILL_AT_MAX, cfg))
    return max(0, min(20, int(value)))


def depth_for_elo(elo: int, config: EngineConfig | None = None) -> int:
    """Độ sâu tìm kiếm cho Stockfish ứng với mức Elo."""
    cfg = config or EngineConfig()
    return max(1, int(round(_interpolate(elo, _DEPTH_AT_CEILING, _DEPTH_AT_MAX, cfg))))


class Opponent:
    """Đối thủ chơi theo mức Elo mục tiêu.

    Engine được mở lười (chỉ khi mức Elo đó cần tới) và đóng hết trong
    ``__exit__``, nên dùng một ván nhiều nước chỉ tốn một process mỗi loại.

    >>> with Opponent() as bot:                       # doctest: +SKIP
    ...     bot.get_move(chess.STARTING_FEN, 1500)
    """

    def __init__(
        self,
        *,
        paths: Paths | None = None,
        config: EngineConfig | None = None,
        allow_stockfish_fallback: bool = False,
    ) -> None:
        self._paths: Paths = paths or Paths()
        self._config: EngineConfig = config or EngineConfig()
        #: Cho phép dùng Stockfish yếu thay Maia khi máy không có binary Maia.
        #: Mặc định tắt: độ mạnh sẽ *không* giống người, chỉ bật khi chấp nhận.
        self._allow_stockfish_fallback = allow_stockfish_fallback
        self._maia: MaiaEngine | None = None
        self._stockfish: StockfishEngine | None = None
        self._maia_unavailable = False

    # -- vòng đời ---------------------------------------------------------

    def __enter__(self) -> Opponent:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        for engine in (self._maia, self._stockfish):
            if engine is not None:
                engine.close()
        self._maia = None
        self._stockfish = None

    def _get_stockfish(self) -> StockfishEngine:
        if self._stockfish is None:
            binary = self._paths.resolve_stockfish()
            if binary is None:
                raise FileNotFoundError(
                    f"Không tìm thấy Stockfish ({self._paths.stockfish_bin}); "
                    "đặt CHESSVI_STOCKFISH_BIN"
                )
            self._stockfish = StockfishEngine(binary, config=self._config).__enter__()
        return self._stockfish

    def _get_maia(self) -> MaiaEngine | None:
        """Maia đã mở, hoặc None nếu máy không có binary."""
        if self._maia_unavailable:
            return None
        if self._maia is None:
            binary = self._paths.resolve_maia()
            if binary is None:
                self._maia_unavailable = True
                return None
            self._maia = MaiaEngine(binary, config=self._config).__enter__()
        return self._maia

    # -- API --------------------------------------------------------------

    def get_move(self, fen: str, target_elo: int) -> chess.Move:
        """Nước đi của đối thủ ở mức ``target_elo``.

        ``fen`` là str ở biên I/O, trả về ``chess.Move`` để dùng trong code.
        Nước trả về luôn đã được khẳng định nằm trong ``board.legal_moves``.
        """
        board = chess.Board(fen)  # ném ValueError nếu FEN hỏng
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"Ván đã kết thúc, không có nước để đi: {fen}")

        choice = route_engine(target_elo, self._config)
        move = (
            self._maia_move(board, target_elo)
            if choice == "maia"
            else self._stockfish_move(board, target_elo)
        )

        # Nguyên tắc 2: khẳng định trong code, không chỉ trong test.
        assert move in board.legal_moves, f"nước {move.uci()} không hợp lệ với {fen}"
        logger.debug("Elo %d -> %s -> %s", target_elo, choice, move.uci())
        return move

    def _maia_move(self, board: chess.Board, target_elo: int) -> chess.Move:
        maia = self._get_maia()
        if maia is not None:
            return maia.get_move(board, target_elo)

        message = (
            f"Elo {target_elo} cần Maia nhưng không tìm thấy binary "
            f"({self._paths.maia_bin}); đặt CHESSVI_MAIA_BIN"
        )
        if not self._allow_stockfish_fallback:
            raise FileNotFoundError(message)
        logger.warning("%s — tạm dùng Stockfish, độ mạnh sẽ KHÔNG giống người", message)
        return self._weak_stockfish_move(board, target_elo)

    def _stockfish_move(self, board: chess.Board, target_elo: int) -> chess.Move:
        engine = self._get_stockfish()
        engine.configure_skill(skill_for_elo(target_elo, self._config))
        move = engine.play(board, depth=depth_for_elo(target_elo, self._config))
        if move is None:  # pragma: no cover - đã chặn game over ở trên
            raise RuntimeError(f"Stockfish không trả nước cho {board.fen()}")
        return move

    def _weak_stockfish_move(self, board: chess.Board, target_elo: int) -> chess.Move:
        """Chỉ dùng khi fallback: map Elo thấp sang Skill Level thấp."""
        engine = self._get_stockfish()
        span = max(1, self._config.maia_elo_ceiling - self._config.min_elo)
        ratio = (min(max(target_elo, self._config.min_elo), self._config.maia_elo_ceiling)
                 - self._config.min_elo) / span
        engine.configure_skill(max(0, min(20, round(ratio * _SKILL_AT_CEILING))))
        move = engine.play(board, depth=max(1, round(1 + ratio * 6)))
        if move is None:  # pragma: no cover
            raise RuntimeError(f"Stockfish không trả nước cho {board.fen()}")
        return move
