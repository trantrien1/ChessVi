"""Wrapper Maia-3 qua giao thức UCI.

Maia sai *giống người thật* ở từng mức Elo, khác Stockfish yếu đi bằng nước
ngẫu nhiên kỳ quặc. Đây là lý do lớp đối thủ route Elo thấp sang Maia.
"""

from __future__ import annotations

import logging
from types import TracebackType

import chess
import chess.engine

from chessvi.config import EngineConfig, Paths

logger = logging.getLogger(__name__)

__all__ = ["MaiaEngine", "MaiaEloUnsupportedError"]

#: Tên option UCI để ép mức Elo — khác nhau giữa các bản build của Maia.
#: Dò theo thứ tự này, so khớp không phân biệt hoa thường.
ELO_OPTION_CANDIDATES: tuple[str, ...] = (
    "Elo",
    "elo_self",
    "UCI_Elo",
    "Maia Elo",
    "Rating",
)


class MaiaEloUnsupportedError(chess.engine.EngineError):
    """Binary Maia không có option nào để ép mức Elo."""


class MaiaEngine:
    """Context manager quanh một process Maia-3.

    Cùng interface vòng đời với :class:`~chessvi.engine.stockfish.StockfishEngine`.
    """

    def __init__(
        self,
        binary: str | None = None,
        *,
        config: EngineConfig | None = None,
        elo_option: str | None = None,
        options: dict[str, object] | None = None,
    ) -> None:
        paths = Paths()
        self._binary: str = binary or paths.maia_bin
        self._config: EngineConfig = config or EngineConfig()
        #: Nếu None thì tự dò trong ELO_OPTION_CANDIDATES khi mở engine.
        self._elo_option: str | None = elo_option
        self._explicit_elo_option: bool = elo_option is not None
        self._pending_options: dict[str, object] = dict(options or {})
        self._engine: chess.engine.SimpleEngine | None = None

    # -- vòng đời ---------------------------------------------------------

    def __enter__(self) -> MaiaEngine:
        logger.debug("Mở Maia: %s", self._binary)
        self._engine = chess.engine.SimpleEngine.popen_uci(self._binary)
        if not self._explicit_elo_option:
            self._elo_option = self._detect_elo_option(self._engine)
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
            logger.warning("Maia không thoát sạch, kill cứng")
            engine.close()

    @property
    def is_open(self) -> bool:
        return self._engine is not None

    @property
    def elo_option(self) -> str | None:
        """Tên option UCI đang dùng để ép Elo (biết sau khi mở engine)."""
        return self._elo_option

    def _require(self) -> chess.engine.SimpleEngine:
        if self._engine is None:
            raise chess.engine.EngineError("MaiaEngine phải dùng trong `with`")
        return self._engine

    @staticmethod
    def _detect_elo_option(engine: chess.engine.SimpleEngine) -> str | None:
        available = {name.lower(): name for name in engine.options}
        for candidate in ELO_OPTION_CANDIDATES:
            found = available.get(candidate.lower())
            if found is not None:
                logger.debug("Maia dùng option Elo %r", found)
                return found
        logger.warning(
            "Maia không có option Elo nào trong %s; option sẵn có: %s",
            ELO_OPTION_CANDIDATES,
            sorted(available.values()),
        )
        return None

    # -- chơi -------------------------------------------------------------

    def clamp_elo(self, elo: int) -> int:
        """Kẹp Elo vào khoảng engine hỗ trợ, ghi log nếu phải kẹp."""
        low, high = self._config.min_elo, self._config.max_elo
        clamped = max(low, min(high, elo))
        if clamped != elo:
            logger.warning("Elo %d ngoài khoảng [%d, %d], kẹp về %d", elo, low, high, clamped)
        return clamped

    def get_move(self, board: chess.Board, elo: int) -> chess.Move:
        """Nước đi của Maia ở mức ``elo``.

        Raise nếu ván đã hết, nếu không ép được Elo, hoặc nếu Maia trả nước
        không hợp lệ — không bao giờ im lặng nuốt lỗi.
        """
        if board.is_game_over(claim_draw=True):
            raise ValueError(f"Ván đã kết thúc, không có nước để đi: {board.fen()}")

        engine = self._require()
        if self._elo_option is None:
            raise MaiaEloUnsupportedError(
                "Maia không có option ép Elo; truyền elo_option=... khi khởi tạo MaiaEngine"
            )
        engine.configure({self._elo_option: self.clamp_elo(elo)})

        result = engine.play(board, chess.engine.Limit(time=self._config.move_time))
        move = result.move
        if move is None:
            raise chess.engine.EngineError(f"Maia không trả nước nào cho {board.fen()}")
        if move not in board.legal_moves:
            raise chess.engine.EngineError(
                f"Maia trả nước không hợp lệ {move.uci()} cho {board.fen()}"
            )
        return move
