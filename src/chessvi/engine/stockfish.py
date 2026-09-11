"""Wrapper Stockfish qua giao thức UCI.

Stockfish là *oracle* của dự án: mọi con số eval mà LLM được phép nhắc tới đều
phải đi ra từ đây, không bao giờ do model tự nghĩ.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from types import TracebackType

import chess
import chess.engine

from chessvi.config import EngineConfig, Paths

logger = logging.getLogger(__name__)

__all__ = ["EngineAnalysis", "EngineNotOpenError", "StockfishEngine"]


class EngineNotOpenError(RuntimeError):
    """Gọi engine khi chưa vào context manager."""


@dataclass(frozen=True)
class EngineAnalysis:
    """Kết quả phân tích một thế cờ.

    Quy ước điểm: **luôn theo góc nhìn bên đang đi** (side-to-move POV).
    ``cp > 0`` nghĩa là bên đang đi đang hơn, bất kể đó là trắng hay đen.

    - Thế còn đánh được, không có mate: ``cp`` là số centipawn, ``mate`` là None.
    - Thế có chiếu hết cưỡng bức: ``cp`` là None, ``mate`` là số nước tới mate
      (dương = bên đang đi chiếu hết, âm = bên đang đi bị chiếu hết).
    - Thế đã kết thúc: ``best_move`` là None, ``cp`` là None, ``pv`` rỗng.
      Chiếu hết thì ``mate == 0``; hòa thì ``mate`` là None.
    """

    best_move: chess.Move | None
    cp: int | None
    mate: int | None
    pv: list[chess.Move] = field(default_factory=list)

    @property
    def is_mate(self) -> bool:
        return self.mate is not None

    @property
    def is_terminal(self) -> bool:
        return self.best_move is None


def _terminal_analysis(board: chess.Board) -> EngineAnalysis:
    """Kết quả cho thế đã kết thúc — không gọi engine."""
    mate = 0 if board.is_checkmate() else None
    return EngineAnalysis(best_move=None, cp=None, mate=mate, pv=[])


class StockfishEngine:
    """Context manager quanh một process Stockfish.

    >>> with StockfishEngine() as engine:            # doctest: +SKIP
    ...     engine.analyse(chess.Board()).best_move

    Process chỉ được mở trong ``__enter__`` và luôn bị đóng trong ``__exit__``,
    kể cả khi thân ``with`` ném exception.
    """

    def __init__(
        self,
        binary: str | None = None,
        *,
        config: EngineConfig | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        paths = Paths()
        self._binary: str = binary or paths.stockfish_bin
        self._config: EngineConfig = config or EngineConfig()
        self._pending_options: dict[str, object] = dict(options or {})
        self._engine: chess.engine.SimpleEngine | None = None

    # -- vòng đời ---------------------------------------------------------

    def __enter__(self) -> StockfishEngine:
        logger.debug("Mở Stockfish: %s", self._binary)
        self._engine = chess.engine.SimpleEngine.popen_uci(self._binary)
        if self._pending_options:
            self._engine.configure(self._pending_options)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def close(self) -> None:
        """Đóng process. Gọi nhiều lần vẫn an toàn."""
        if self._engine is None:
            return
        engine, self._engine = self._engine, None
        try:
            engine.quit()
        except chess.engine.EngineError:  # pragma: no cover - engine đã chết
            logger.warning("Stockfish không thoát sạch, kill cứng")
            engine.close()

    @property
    def is_open(self) -> bool:
        return self._engine is not None

    def _require(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            raise EngineNotOpenError("StockfishEngine phải dùng trong `with`")
        return self._engine

    # -- cấu hình ---------------------------------------------------------

    def configure(self, options: dict[str, object]) -> None:
        """Đặt option UCI. Nhớ lại nếu engine chưa mở."""
        self._pending_options.update(options)
        if self._engine is not None:
            self._engine.configure(options)

    def configure_skill(self, level: int) -> None:
        """Đặt option UCI ``Skill Level`` (0–20).

        Cảnh báo: Stockfish yếu đi bằng nước ngẫu nhiên kỳ quặc, không giống
        người. Với Elo <= 2200 hãy dùng Maia (xem ``engine/opponent.py``).
        """
        if not 0 <= level <= 20:
            raise ValueError(f"Skill Level phải trong [0, 20], nhận {level}")
        self.configure({"Skill Level": level})

    # -- phân tích --------------------------------------------------------

    def analyse(self, board: chess.Board, depth: int | None = None) -> EngineAnalysis:
        """Phân tích ``board`` tới độ sâu ``depth``.

        Điểm trả về theo góc nhìn bên đang đi (xem :class:`EngineAnalysis`).
        Thế đã kết thúc thì trả kết quả rỗng mà không gọi engine.
        """
        if board.is_game_over(claim_draw=True):
            logger.debug("Thế đã kết thúc, bỏ qua engine: %s", board.fen())
            return _terminal_analysis(board)

        engine = self._require()
        limit = chess.engine.Limit(depth=depth if depth is not None else self._config.analyse_depth)
        info = engine.analyse(board, limit)

        pv: list[chess.Move] = list(info.get("pv") or [])
        score = info.get("score")
        cp: int | None = None
        mate: int | None = None
        if score is not None:
            relative = score.relative  # POV bên đang đi
            cp = relative.score()
            mate = relative.mate()

        best_move = pv[0] if pv else None
        if best_move is not None and best_move not in board.legal_moves:
            # Nguyên tắc 2: không tin chuỗi thô, kể cả từ engine.
            raise chess.engine.EngineError(
                f"Stockfish trả nước không hợp lệ {best_move.uci()} cho {board.fen()}"
            )
        return EngineAnalysis(best_move=best_move, cp=cp, mate=mate, pv=pv)

    def play(
        self,
        board: chess.Board,
        *,
        depth: int | None = None,
        move_time: float | None = None,
    ) -> chess.Move | None:
        """Chọn một nước để đi thật (tôn trọng ``Skill Level``).

        Khác ``analyse``: đi qua lệnh UCI ``go`` + ``bestmove`` nên chịu ảnh
        hưởng của Skill Level, dùng cho lớp đối thủ. Trả ``None`` nếu ván đã hết.
        """
        if board.is_game_over(claim_draw=True):
            return None
        engine = self._require()
        limit = chess.engine.Limit(
            depth=depth if depth is not None else self._config.analyse_depth,
            time=move_time,
        )
        result = engine.play(board, limit)
        move = result.move
        if move is None:
            raise chess.engine.EngineError(f"Stockfish không trả nước nào cho {board.fen()}")
        if move not in board.legal_moves:
            raise chess.engine.EngineError(
                f"Stockfish trả nước không hợp lệ {move.uci()} cho {board.fen()}"
            )
        return move
