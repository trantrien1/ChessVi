"""Trục eval thứ ba: người chấm tay.

Hai trục chấm, thang 1-5, chấm độc lập nhau:

- **độ tự nhiên tiếng Việt** - câu có đọc như người Việt viết không;
- **độ đúng thuật ngữ cờ** - có dùng đúng bảng thuật ngữ trong CLAUDE.md không.

Máy không đo được hai thứ này: ``hallucination.py`` chỉ bắt được khẳng định
sai, không bắt được câu đúng mà đọc như máy dịch.

    # xuất 100 mẫu ngẫu nhiên ra CSV để nhập vào Google Form
    python -m chessvi.eval.human_eval export --input data/answers.jsonl \
        --out reports/human_eval.csv

    # gom kết quả của nhiều người chấm và tính độ đồng thuận
    python -m chessvi.eval.human_eval collect \
        --ratings reports/rater_a.csv reports/rater_b.csv --out reports/agreement.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from chessvi.logging_setup import configure_logging
from chessvi.train.dataset import iter_records

logger = logging.getLogger(__name__)

__all__ = [
    "AXES",
    "FORM_FIELDS",
    "AgreementReport",
    "collect_ratings",
    "export_samples",
    "quadratic_weighted_kappa",
]

#: Hai trục chấm và mô tả hiện trên form.
AXES: dict[str, str] = {
    "naturalness": "Độ tự nhiên tiếng Việt (1 = như máy dịch, 5 = như người Việt viết)",
    "terminology": "Độ đúng thuật ngữ cờ (1 = sai/tự chế, 5 = đúng bảng thuật ngữ)",
}

MIN_SCORE = 1
MAX_SCORE = 5
DEFAULT_SAMPLE_SIZE = 100
DEFAULT_SEED = 20240911

FORM_FIELDS: tuple[str, ...] = (
    "sample_id",
    "fen",
    "question",
    "answer",
    "naturalness",
    "terminology",
    "notes",
)


# -- xuất form ------------------------------------------------------------


def export_samples(
    records: Iterable[dict[str, Any]],
    *,
    size: int = DEFAULT_SAMPLE_SIZE,
    seed: int = DEFAULT_SEED,
    answer_field: str = "answer",
    question_field: str = "question",
) -> list[dict[str, Any]]:
    """Chọn ngẫu nhiên ``size`` mẫu và dựng dòng cho form chấm.

    Hai cột điểm để trống - người chấm điền. ``sample_id`` giữ nguyên qua các
    bản CSV của từng người để :func:`collect_ratings` ghép lại được.
    """
    pool = [record for record in records if isinstance(record.get(answer_field), str)]
    if not pool:
        raise ValueError("không có mẫu nào có câu trả lời để chấm")

    rng = random.Random(seed)
    chosen = pool if len(pool) <= size else rng.sample(pool, size)
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(chosen):
        rows.append(
            {
                "sample_id": str(record.get("puzzle_id") or record.get("id") or index),
                "fen": record.get("fen", ""),
                "question": record.get(question_field, ""),
                "answer": record[answer_field],
                "naturalness": "",
                "terminology": "",
                "notes": "",
            }
        )
    logger.info("Chọn %d/%d mẫu để chấm tay", len(rows), len(pool))
    return rows


def write_form_csv(rows: Sequence[dict[str, Any]], path: Path) -> None:
    """Ghi CSV nhập thẳng được vào Google Form/Sheets."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(FORM_FIELDS))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Ghi %d mẫu vào %s", len(rows), path)
    for axis, description in AXES.items():
        logger.info("  cột %-12s %s", axis, description)


# -- gom kết quả ----------------------------------------------------------


def quadratic_weighted_kappa(
    first: Sequence[int],
    second: Sequence[int],
    *,
    min_score: int = MIN_SCORE,
    max_score: int = MAX_SCORE,
) -> float:
    """Cohen's kappa có trọng số bậc hai cho thang điểm thứ bậc.

    Trọng số bậc hai hợp với thang 1-5: lệch 1 điểm nhẹ hơn lệch 4 điểm nhiều.
    Trả về 1.0 khi hai người chấm giống hệt, 0.0 khi chỉ bằng mức ngẫu nhiên,
    âm khi tệ hơn ngẫu nhiên.
    """
    if len(first) != len(second):
        raise ValueError(f"số lượng lệch nhau: {len(first)} vs {len(second)}")
    if not first:
        raise ValueError("không có điểm nào để so")

    size = max_score - min_score + 1
    observed = [[0.0] * size for _ in range(size)]
    for a, b in zip(first, second, strict=True):
        observed[a - min_score][b - min_score] += 1

    total = float(len(first))
    hist_a = [sum(row) for row in observed]
    hist_b = [sum(observed[i][j] for i in range(size)) for j in range(size)]

    denominator = (size - 1) ** 2
    numerator_sum = 0.0
    denominator_sum = 0.0
    for i in range(size):
        for j in range(size):
            weight = ((i - j) ** 2) / denominator
            expected = hist_a[i] * hist_b[j] / total
            numerator_sum += weight * observed[i][j]
            denominator_sum += weight * expected
    if denominator_sum == 0:
        return 1.0
    return 1.0 - numerator_sum / denominator_sum


@dataclass
class AgreementReport:
    """Điểm trung bình từng trục và độ đồng thuận giữa những người chấm."""

    raters: list[str]
    #: trục -> tên người chấm -> điểm trung bình.
    means: dict[str, dict[str, float]]
    #: trục -> kappa trung bình của mọi cặp người chấm.
    kappa: dict[str, float]
    #: trục -> tỷ lệ hai người chấm lệch nhau không quá 1 điểm.
    within_one: dict[str, float]
    shared_samples: int

    def rows(self) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for axis in AXES:
            for rater, mean in self.means.get(axis, {}).items():
                out.append(
                    {
                        "axis": axis,
                        "metric": "mean",
                        "who": rater,
                        "value": round(mean, 3),
                        "n": self.shared_samples,
                    }
                )
            out.append(
                {
                    "axis": axis,
                    "metric": "quadratic_weighted_kappa",
                    "who": "all_pairs",
                    "value": round(self.kappa.get(axis, 0.0), 3),
                    "n": self.shared_samples,
                }
            )
            out.append(
                {
                    "axis": axis,
                    "metric": "within_one_point",
                    "who": "all_pairs",
                    "value": round(self.within_one.get(axis, 0.0), 3),
                    "n": self.shared_samples,
                }
            )
        return out

    def render(self) -> str:
        lines = [
            "=" * 56,
            f"Người chấm: {', '.join(self.raters)}",
            f"Số mẫu cả nhóm cùng chấm: {self.shared_samples}",
            "-" * 56,
        ]
        for axis, description in AXES.items():
            lines.append(f"{axis} — {description}")
            for rater, mean in self.means.get(axis, {}).items():
                lines.append(f"  trung bình {rater:<16}{mean:6.2f}")
            lines.append(f"  kappa (trọng số bậc 2){self.kappa.get(axis, 0.0):>12.3f}")
            lines.append(f"  lệch <= 1 điểm{self.within_one.get(axis, 0.0) * 100:>18.1f}%")
        lines.append("=" * 56)
        return "\n".join(lines)


def _read_ratings(path: Path) -> dict[str, dict[str, int]]:
    """``sample_id -> {trục: điểm}``, bỏ qua dòng chưa chấm."""
    scores: dict[str, dict[str, int]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            sample_id = (row.get("sample_id") or "").strip()
            if not sample_id:
                continue
            parsed: dict[str, int] = {}
            for axis in AXES:
                raw = (row.get(axis) or "").strip()
                if not raw:
                    continue
                try:
                    value = int(float(raw))
                except ValueError:
                    logger.warning("%s: điểm %s không đọc được ở mẫu %s", path.name, raw, sample_id)
                    continue
                if not MIN_SCORE <= value <= MAX_SCORE:
                    logger.warning("%s: điểm %d ngoài thang 1-5 ở mẫu %s", path.name, value, sample_id)
                    continue
                parsed[axis] = value
            if parsed:
                scores[sample_id] = parsed
    return scores


def collect_ratings(paths: Sequence[Path]) -> AgreementReport:
    """Gom CSV của từng người chấm, tính điểm trung bình và độ đồng thuận."""
    if len(paths) < 2:
        raise ValueError("cần ít nhất 2 người chấm để tính độ đồng thuận")

    by_rater = {path.stem: _read_ratings(path) for path in paths}
    raters = list(by_rater)

    means: dict[str, dict[str, float]] = {}
    kappa: dict[str, float] = {}
    within_one: dict[str, float] = {}
    shared_total = 0

    for axis in AXES:
        common = set.intersection(
            *(
                {sid for sid, scores in ratings.items() if axis in scores}
                for ratings in by_rater.values()
            )
        )
        ordered = sorted(common)
        shared_total = max(shared_total, len(ordered))
        if not ordered:
            logger.warning("Không có mẫu nào cả nhóm cùng chấm ở trục %s", axis)
            continue

        means[axis] = {
            rater: sum(by_rater[rater][sid][axis] for sid in ordered) / len(ordered)
            for rater in raters
        }
        kappas: list[float] = []
        closes: list[float] = []
        for left, right in itertools.combinations(raters, 2):
            a = [by_rater[left][sid][axis] for sid in ordered]
            b = [by_rater[right][sid][axis] for sid in ordered]
            kappas.append(quadratic_weighted_kappa(a, b))
            closes.append(
                sum(1 for x, y in zip(a, b, strict=True) if abs(x - y) <= 1) / len(ordered)
            )
        kappa[axis] = sum(kappas) / len(kappas)
        within_one[axis] = sum(closes) / len(closes)

    return AgreementReport(
        raters=raters,
        means=means,
        kappa=kappa,
        within_one=within_one,
        shared_samples=shared_total,
    )


def write_agreement_csv(report: AgreementReport, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report.rows()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Ghi %d dòng vào %s", len(rows), path)


# -- CLI ------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m chessvi.eval.human_eval",
        description="Xuất mẫu cho người chấm tay và gom kết quả.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    export = sub.add_parser("export", help="Xuất N mẫu ngẫu nhiên ra CSV")
    export.add_argument("--input", type=Path, required=True, help="JSONL/parquet có câu trả lời")
    export.add_argument("--out", type=Path, required=True)
    export.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE)
    export.add_argument("--seed", type=int, default=DEFAULT_SEED)

    collect = sub.add_parser("collect", help="Gom CSV đã chấm, tính đồng thuận")
    collect.add_argument("--ratings", type=Path, nargs="+", required=True)
    collect.add_argument("--out", type=Path)

    parser.add_argument("-v", "--verbose", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(args.verbose)

    if args.command == "export":
        rows = export_samples(iter_records(args.input), size=args.n, seed=args.seed)
        write_form_csv(rows, args.out)
        return 0

    report = collect_ratings(args.ratings)
    logger.info("%s", report.render())
    if args.out is not None:
        write_agreement_csv(report, args.out)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
