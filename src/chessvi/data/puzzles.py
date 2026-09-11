"""Chuẩn bị puzzle set từ Lichess/chess-puzzles.

**Cạm bẫy kinh điển của dataset này**: trường ``Moves`` là chuỗi UCI, nhưng
nước **ĐẦU TIÊN là nước của đối thủ dẫn vào thế puzzle**, không phải lời giải.
Phải đẩy nước đó lên bàn trước rồi phần còn lại mới là lời giải. Đọc kỹ
dataset card. Ở đây :func:`prepare_puzzle` làm đúng việc đó và khẳng định
từng nước đều hợp lệ.

    python -m chessvi.data.puzzles --train-size 50000 --test-size 2000
    python -m chessvi.data.puzzles --input-jsonl mau.jsonl --out-dir data/puzzles
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import chess

from chessvi.config import Paths
from chessvi.logging_setup import configure_logging

logger = logging.getLogger(__name__)

__all__ = [
    "UNKNOWN_THEME",
    "Puzzle",
    "PuzzleSplit",
    "balanced_order",
    "balanced_sample",
    "prepare_puzzle",
    "rating_bucket",
    "render_distribution",
    "write_parquet",
]

DEFAULT_DATASET = "Lichess/chess-puzzles"
DEFAULT_BUCKET_WIDTH = 400
DEFAULT_TRAIN_SIZE = 50_000
DEFAULT_TEST_SIZE = 2_000
DEFAULT_SEED = 20240911
UNKNOWN_THEME = "untagged"


@dataclass(frozen=True)
class Puzzle:
    """Một puzzle đã được đưa về đúng thế cờ cần giải."""

    puzzle_id: str
    #: FEN gốc trong dataset, TRƯỚC nước dẫn của đối thủ.
    setup_fen: str
    #: Nước của đối thủ dẫn vào thế puzzle (UCI).
    opponent_move: str
    #: FEN của thế người chơi thật sự phải giải.
    fen: str
    #: Lời giải, UCI, tính từ ``fen``.
    solution: list[str]
    rating: int
    rating_bucket: str
    themes: list[str]
    #: Theme đại diện dùng để cân bằng mẫu.
    primary_theme: str

    @property
    def first_move(self) -> str:
        """Nước đầu tiên của lời giải - cái mà model phải đoán đúng."""
        return self.solution[0]


@dataclass
class PuzzleSplit:
    """Hai tập train/test rời nhau."""

    train: list[Puzzle]
    test: list[Puzzle]

    def assert_disjoint(self) -> None:
        train_ids = {p.puzzle_id for p in self.train}
        test_ids = {p.puzzle_id for p in self.test}
        overlap = train_ids & test_ids
        if overlap:
            raise AssertionError(f"rò rỉ {len(overlap)} puzzle giữa train và test")


# -- chuẩn hoá row --------------------------------------------------------


def _get(row: dict[str, Any], name: str) -> Any:
    """Lấy cột không phân biệt hoa thường (bản HF và bản CSV đặt tên khác nhau)."""
    if name in row:
        return row[name]
    lowered = {key.lower(): value for key, value in row.items()}
    return lowered.get(name.lower())


def _themes(row: dict[str, Any]) -> list[str]:
    raw = _get(row, "Themes")
    if raw is None:
        return []
    if isinstance(raw, str):
        return sorted(filter(None, raw.split()))
    return sorted(filter(None, (str(theme) for theme in raw)))


def _moves(row: dict[str, Any]) -> list[str]:
    raw = _get(row, "Moves")
    if raw is None:
        return []
    if isinstance(raw, str):
        return raw.split()
    return [str(move) for move in raw]


def rating_bucket(rating: int, width: int = DEFAULT_BUCKET_WIDTH) -> str:
    """Nhãn bucket rating, ví dụ ``1200-1599`` với width 400."""
    if width <= 0:
        raise ValueError(f"width phải dương, nhận {width}")
    low = (rating // width) * width
    return f"{low}-{low + width - 1}"


# -- chuẩn bị một puzzle --------------------------------------------------


def prepare_puzzle(
    row: dict[str, Any],
    *,
    bucket_width: int = DEFAULT_BUCKET_WIDTH,
) -> Puzzle:
    """Đưa một row của dataset về :class:`Puzzle`.

    Đẩy nước dẫn của đối thủ lên bàn, phần còn lại là lời giải. Mọi nước đều
    phải hợp lệ - nếu không thì ném ``ValueError`` để caller đếm và bỏ qua,
    không bao giờ tạo ra puzzle rác.
    """
    fen = _get(row, "FEN")
    if not isinstance(fen, str):
        raise ValueError("thiếu trường FEN")
    moves = _moves(row)
    if len(moves) < 2:
        raise ValueError(f"cần ít nhất 2 nước (1 dẫn + 1 giải), có {len(moves)}")

    board = chess.Board(fen)  # ValueError nếu FEN hỏng
    opponent_uci, *solution_uci = moves

    opponent_move = chess.Move.from_uci(opponent_uci)
    if opponent_move not in board.legal_moves:
        raise ValueError(f"nước dẫn {opponent_uci} không hợp lệ với {fen}")
    board.push(opponent_move)
    puzzle_fen = board.fen()

    # Khẳng định cả lời giải chạy được, rồi mới trả về.
    replay = chess.Board(puzzle_fen)
    for uci in solution_uci:
        move = chess.Move.from_uci(uci)
        if move not in replay.legal_moves:
            raise ValueError(f"nước lời giải {uci} không hợp lệ tại {replay.fen()}")
        replay.push(move)

    rating_raw = _get(row, "Rating")
    rating = int(rating_raw) if rating_raw is not None else 0
    themes = _themes(row)
    return Puzzle(
        puzzle_id=str(_get(row, "PuzzleId") or _get(row, "id") or puzzle_fen),
        setup_fen=fen,
        opponent_move=opponent_uci,
        fen=puzzle_fen,
        solution=solution_uci,
        rating=rating,
        rating_bucket=rating_bucket(rating, bucket_width),
        themes=themes,
        primary_theme=themes[0] if themes else UNKNOWN_THEME,
    )


# -- lấy mẫu cân bằng -----------------------------------------------------


def _pick_primary(themes: Sequence[str], bucket: str, counts: Counter[tuple[str, str]]) -> str:
    """Theme đại diện = theme đang ít mẫu nhất trong bucket đó.

    Một puzzle mang nhiều theme; gán về đúng một theme để tập mẫu không bị
    đếm trùng và để train/test chắc chắn rời nhau.
    """
    if not themes:
        return UNKNOWN_THEME
    return min(themes, key=lambda theme: (counts[(theme, bucket)], theme))


def balanced_sample(
    rows: Iterable[dict[str, Any]],
    *,
    train_size: int = DEFAULT_TRAIN_SIZE,
    test_size: int = DEFAULT_TEST_SIZE,
    bucket_width: int = DEFAULT_BUCKET_WIDTH,
    per_cell: int | None = None,
    max_scan: int | None = None,
    seed: int = DEFAULT_SEED,
) -> tuple[PuzzleSplit, Counter[str]]:
    """Lấy mẫu cân bằng theo ``(theme, rating bucket)``.

    Trả về ``(split, skipped)`` với ``skipped`` đếm lý do bỏ qua row.

    Hai pha: gom tối đa ``per_cell`` mẫu cho mỗi ô ``(theme, bucket)``, rồi
    chia round-robin qua các ô cho tới khi đủ chỉ tiêu, nên ô thưa không kéo
    tụt ô dày và phân phối vẫn phẳng.
    """
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[Puzzle]] = defaultdict(list)
    counts: Counter[tuple[str, str]] = Counter()
    skipped: Counter[str] = Counter()
    seen_ids: set[str] = set()
    cap = per_cell if per_cell is not None else max(1, (train_size + test_size) // 100)

    for index, row in enumerate(rows):
        if max_scan is not None and index >= max_scan:
            break
        try:
            puzzle = prepare_puzzle(row, bucket_width=bucket_width)
        except (ValueError, KeyError) as error:
            skipped[type(error).__name__] += 1
            logger.debug("Bỏ qua row %d: %s", index, error)
            continue
        if puzzle.puzzle_id in seen_ids:
            skipped["duplicate"] += 1
            continue

        primary = _pick_primary(puzzle.themes, puzzle.rating_bucket, counts)
        cell = (primary, puzzle.rating_bucket)
        if len(cells[cell]) >= cap:
            skipped["cell_full"] += 1
            continue
        seen_ids.add(puzzle.puzzle_id)
        counts[cell] += 1
        # primary_theme là frozen dataclass field nên tạo lại bản có theme đúng.
        cells[cell].append(
            puzzle if puzzle.primary_theme == primary else _with_primary(puzzle, primary)
        )

    for bucket_cells in cells.values():
        rng.shuffle(bucket_cells)

    test = _round_robin(cells, test_size)
    taken = {id(p) for p in test}
    remaining = {
        cell: [p for p in items if id(p) not in taken] for cell, items in cells.items()
    }
    train = _round_robin(remaining, train_size)

    split = PuzzleSplit(train=train, test=test)
    split.assert_disjoint()
    return split, skipped


def _with_primary(puzzle: Puzzle, primary: str) -> Puzzle:
    data = asdict(puzzle)
    data["primary_theme"] = primary
    return Puzzle(**data)


def balanced_order(
    items: Sequence[dict[str, Any]],
    *,
    size: int | None = None,
    theme_key: str = "primary_theme",
    bucket_key: str = "rating_bucket",
    seed: int = DEFAULT_SEED,
) -> list[int]:
    """Thứ tự chỉ số cân bằng theo ``(theme, bucket)``, dùng lại logic T7.

    ``grpo.py`` gọi hàm này để dựng sampler: mỗi vòng round-robin lấy một mẫu
    từ mỗi ô nên batch RL không bị một theme áp đảo. Trả về danh sách chỉ số
    vào ``items``, dài tối đa ``size`` (mặc định: hết).
    """
    rng = random.Random(seed)
    cells: dict[tuple[str, str], list[int]] = defaultdict(list)
    for index, item in enumerate(items):
        theme = str(item.get(theme_key) or UNKNOWN_THEME)
        bucket = str(item.get(bucket_key) or "?")
        cells[(theme, bucket)].append(index)
    for indices in cells.values():
        rng.shuffle(indices)

    target = len(items) if size is None else min(size, len(items))
    picked: list[int] = []
    cursors = dict.fromkeys(cells, 0)
    order = sorted(cells)
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for cell in order:
            if len(picked) >= target:
                break
            cursor = cursors[cell]
            if cursor < len(cells[cell]):
                picked.append(cells[cell][cursor])
                cursors[cell] = cursor + 1
                progressed = True
    return picked


def _round_robin(cells: dict[tuple[str, str], list[Puzzle]], target: int) -> list[Puzzle]:
    """Lấy lần lượt một mẫu mỗi ô cho tới khi đủ ``target``."""
    if target <= 0:
        return []
    cursors = dict.fromkeys(cells, 0)
    picked: list[Puzzle] = []
    order = sorted(cells)
    progressed = True
    while len(picked) < target and progressed:
        progressed = False
        for cell in order:
            if len(picked) >= target:
                break
            index = cursors[cell]
            items = cells[cell]
            if index < len(items):
                picked.append(items[index])
                cursors[cell] = index + 1
                progressed = True
    return picked


# -- báo cáo phân phối ----------------------------------------------------


def render_distribution(puzzles: Sequence[Puzzle], title: str) -> str:
    """Bảng phân phối theo bucket rating và theo theme."""
    by_bucket: Counter[str] = Counter(p.rating_bucket for p in puzzles)
    by_theme: Counter[str] = Counter(p.primary_theme for p in puzzles)
    total = len(puzzles)
    lines = [f"=== {title}: {total} puzzle ===", "Theo rating bucket:"]
    for bucket, count in sorted(by_bucket.items(), key=lambda kv: int(kv[0].split("-")[0])):
        lines.append(f"  {bucket:<14}{count:>8}{_pct(count, total):>10}")
    lines.append(f"Theo theme ({len(by_theme)} theme):")
    for theme, count in by_theme.most_common():
        lines.append(f"  {theme:<24}{count:>8}{_pct(count, total):>10}")
    return "\n".join(lines)


def _pct(count: int, total: int) -> str:
    return f"{count / total * 100:6.2f}%" if total else "     -"


# -- nạp và ghi -----------------------------------------------------------


def load_rows(dataset: str | None, input_jsonl: Path | None, split: str) -> Iterator[dict[str, Any]]:
    if input_jsonl is not None:
        with input_jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    from datasets import load_dataset  # noqa: PLC0415

    logger.info("Streaming %s (split=%s)", dataset, split)
    for row in load_dataset(dataset, split=split, streaming=True):
        yield dict(row)


def write_parquet(puzzles: Sequence[Puzzle], path: Path) -> None:
    import pyarrow as pa  # noqa: PLC0415
    import pyarrow.parquet as pq  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist([asdict(p) for p in puzzles]), path)
    logger.info("Ghi %d puzzle vào %s", len(puzzles), path)


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.data.puzzles",
        description="Sample puzzle cân bằng theo theme và rating bucket.",
    )
    parser.add_argument("--dataset", default=DEFAULT_DATASET)
    parser.add_argument("--split", default="train", help="Split của dataset nguồn")
    parser.add_argument("--input-jsonl", type=Path, help="File local thay cho HF")
    parser.add_argument("--train-size", type=int, default=DEFAULT_TRAIN_SIZE)
    parser.add_argument("--test-size", type=int, default=DEFAULT_TEST_SIZE)
    parser.add_argument("--bucket-width", type=int, default=DEFAULT_BUCKET_WIDTH)
    parser.add_argument("--per-cell", type=int, help="Trần mẫu cho mỗi (theme, bucket)")
    parser.add_argument("--max-scan", type=int, help="Chỉ quét N row đầu của nguồn")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--out-dir", type=Path, help="Mặc định: <data_dir>/puzzles")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in phân phối")
    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def _configure_logging(verbose: bool) -> None:
    configure_logging(verbose)


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _configure_logging(args.verbose)

    rows = load_rows(args.dataset, args.input_jsonl, args.split)
    split, skipped = balanced_sample(
        rows,
        train_size=args.train_size,
        test_size=args.test_size,
        bucket_width=args.bucket_width,
        per_cell=args.per_cell,
        max_scan=args.max_scan,
        seed=args.seed,
    )

    logger.info("%s", render_distribution(split.train, "TRAIN"))
    logger.info("%s", render_distribution(split.test, "TEST"))
    if skipped:
        logger.info("Bỏ qua: %s", dict(skipped))
    if not split.train:
        logger.error("Không lấy được puzzle nào")
        return 1

    if args.dry_run:
        logger.info("(dry-run, không ghi file)")
        return 0

    out_dir = args.out_dir or (Paths().data_dir / "puzzles")
    write_parquet(split.train, out_dir / "train.parquet")
    write_parquet(split.test, out_dir / "test.parquet")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
