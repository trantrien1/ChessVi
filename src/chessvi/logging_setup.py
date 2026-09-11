"""Cấu hình logging dùng chung cho mọi CLI.

Lý do tồn tại: ``logging.basicConfig(level=INFO)`` đặt level cho **root** logger,
nên httpx / huggingface_hub / fsspec cũng đổ INFO ra màn hình theo. Một lần chạy
``chessvi.data.puzzles`` in ra hàng trăm dòng ``HTTP Request: GET ...`` kèm URL
ký sẵn dài ngoằng, nhấn chìm bảng phân phối — thứ người dùng thật sự cần đọc.

Ở đây root giữ mức WARNING, chỉ log của ta được hạ xuống INFO/DEBUG.
Cần soi tầng mạng thì ``-v`` mở lại toàn bộ.
"""

from __future__ import annotations

import logging
import sys

__all__ = [
    "DEFAULT_FORMAT",
    "NOISY_LOGGERS",
    "OUR_LOGGERS",
    "TIMESTAMPED_FORMAT",
    "configure_logging",
]

DEFAULT_FORMAT = "%(message)s"
#: Dùng cho job train chạy hàng giờ — mốc thời gian đáng giá hơn dòng ngắn.
TIMESTAMPED_FORMAT = "%(asctime)s %(levelname)s %(message)s"

#: Log của ta. ``__main__`` là bắt buộc: chạy ``python -m chessvi.data.puzzles``
#: thì ``__name__`` của chính module đó là ``"__main__"``, nên logger của nó tên
#: ``__main__`` chứ không nằm dưới cây ``chessvi`` — bỏ sót là toàn bộ output CLI
#: biến mất, vì root đang ở WARNING.
OUR_LOGGERS: tuple[str, ...] = ("chessvi", "__main__")

#: Thư viện bên thứ ba nói quá nhiều ở mức INFO.
NOISY_LOGGERS: tuple[str, ...] = (
    "datasets",
    "filelock",
    "fsspec",
    "httpcore",
    "httpx",
    "huggingface_hub",
    "urllib3",
)


def configure_logging(verbose: bool = False, *, fmt: str = DEFAULT_FORMAT) -> None:
    """Log của chessvi ra stdout ở INFO (``verbose`` thì DEBUG), thư viện ngoài im.

    ``force=True`` là cố ý: CLI có thể được gọi nhiều lần trong một tiến trình và
    ta muốn cấu hình sau ghi đè cấu hình trước.
    """
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(fmt))
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=logging.WARNING, handlers=[handler], force=True)
    for name in OUR_LOGGERS:
        logging.getLogger(name).setLevel(level)
    for name in NOISY_LOGGERS:
        logging.getLogger(name).setLevel(level if verbose else logging.WARNING)
