"""
Потоковый парсер PGN-дампов Lichess -> CSV.

Читает .pgn.zst (или обычный .pgn) построчно, не загружая файл в память,
и пишет по одной строке на партию: заголовки + несколько вычисленных полей.
Имена игроков намеренно не сохраняются: для задач проекта они не нужны.

Пример:
    python scripts/parse_pgn.py data/raw/lichess_2013-01.pgn.zst
    python scripts/parse_pgn.py data/raw/lichess_2024-01.pgn.zst --max-games 2000000
"""

import argparse
import csv
import io
import logging
import re
from contextlib import contextmanager
from pathlib import Path

import zstandard

log = logging.getLogger("parse_pgn")

HEADER_RE = re.compile(r'^\[(\w+) "(.*)"\]\s*$')
COMMENT_RE = re.compile(r"\{[^}]*\}")
MOVE_NUM_RE = re.compile(r"\d+\.(\.\.)?")
NAG_RE = re.compile(r"\$\d+")
RESULTS = {"1-0", "0-1", "1/2-1/2", "*"}

COLUMNS = [
    "game_id",
    "speed",
    "event",
    "result",
    "utc_date",
    "utc_time",
    "white_elo",
    "black_elo",
    "white_diff",
    "black_diff",
    "eco",
    "opening",
    "time_control",
    "base_sec",
    "inc_sec",
    "termination",
    "n_plies",
    "has_eval",
    "has_clock",
    "moves_6",
]


@contextmanager
def open_text(path: Path):
    """Открывает .zst как поток текста, остальное - как обычный файл.

    Все файлы закрываются при выходе из `with`.
    max_window_size разрешает декодеру кадры с длинным окном сжатия
    (по умолчанию лимит 128 МБ); память выделяется только если кадр её требует.
    """
    if path.suffix == ".zst":
        decompressor = zstandard.ZstdDecompressor(max_window_size=2**31)
        with open(path, "rb") as raw, decompressor.stream_reader(raw) as reader:
            yield io.TextIOWrapper(reader, encoding="utf-8", errors="replace")
    else:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            yield f


def to_int(value):
    """'1500' -> 1500, '+12' -> 12, '?' / '' / None -> None."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_speed(event: str) -> str:
    m = re.search(r"Rated (\w+)", event or "")
    return m.group(1).lower() if m else ""


def parse_time_control(tc: str):
    """'300+3' -> (300, 3); '-' (корреспонденция) -> (None, None)."""
    m = re.fullmatch(r"(\d+)\+(\d+)", tc or "")
    if not m:
        return None, None
    return int(m.group(1)), int(m.group(2))


def analyze_movetext(text: str):
    """Возвращает (число полуходов, первые 6 полуходов строкой)."""
    clean = COMMENT_RE.sub(" ", text)
    clean = NAG_RE.sub(" ", clean)
    clean = MOVE_NUM_RE.sub(" ", clean)
    moves = [t for t in clean.split() if t not in RESULTS]
    return len(moves), " ".join(moves[:6])


def build_row(headers: dict, movetext: str):
    """Строка CSV из заголовков и текста ходов. None, если запись непригодна."""
    site = headers.get("Site", "")
    game_id = site.rstrip("/").rsplit("/", 1)[-1] if site else ""
    result = headers.get("Result", "")
    if not game_id or result not in RESULTS:
        return None
    base, inc = parse_time_control(headers.get("TimeControl", ""))
    n_plies, moves_6 = analyze_movetext(movetext)
    return {
        "game_id": game_id,
        "speed": parse_speed(headers.get("Event", "")),
        "event": headers.get("Event", ""),
        "result": result,
        "utc_date": headers.get("UTCDate", "").replace(".", "-"),
        "utc_time": headers.get("UTCTime", ""),
        "white_elo": to_int(headers.get("WhiteElo")),
        "black_elo": to_int(headers.get("BlackElo")),
        "white_diff": to_int(headers.get("WhiteRatingDiff")),
        "black_diff": to_int(headers.get("BlackRatingDiff")),
        "eco": headers.get("ECO", ""),
        "opening": headers.get("Opening", ""),
        "time_control": headers.get("TimeControl", ""),
        "base_sec": base,
        "inc_sec": inc,
        "termination": headers.get("Termination", ""),
        "n_plies": n_plies,
        "has_eval": int("[%eval" in movetext),
        "has_clock": int("[%clk" in movetext),
        "moves_6": moves_6,
    }


def iter_games(lines):
    """Генератор (заголовки, текст ходов) по потоку строк PGN."""
    headers, moves = {}, []
    for line in lines:
        line = line.rstrip("\n")
        if line.startswith("[Event ") and (headers or moves):
            yield headers, " ".join(moves)
            headers, moves = {}, []
        m = HEADER_RE.match(line)
        if m:
            headers[m.group(1)] = m.group(2)
        elif line.strip():
            moves.append(line.strip())
    if headers or moves:
        yield headers, " ".join(moves)


def parse_file(src: Path, dst: Path, max_games: int | None = None):
    dst.parent.mkdir(parents=True, exist_ok=True)
    total = written = rejected = 0
    with open_text(src) as fin, open(dst, "w", newline="", encoding="utf-8") as fout:
        writer = csv.DictWriter(fout, fieldnames=COLUMNS)
        writer.writeheader()
        for headers, movetext in iter_games(fin):
            total += 1
            row = build_row(headers, movetext)
            if row is None:
                rejected += 1
            else:
                writer.writerow(row)
                written += 1
            if total % 100_000 == 0:
                log.info(
                    "прочитано %d партий, записано %d, отброшено %d",
                    total,
                    written,
                    rejected,
                )
            if max_games and total >= max_games:
                log.info("достигнут лимит --max-games=%d, останавливаюсь", max_games)
                break
    log.info(
        "готово: всего %d, записано %d, отброшено %d -> %s",
        total,
        written,
        rejected,
        dst,
    )
    return total, written, rejected


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("src", type=Path, help="входной .pgn.zst или .pgn")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="выходной CSV (по умолчанию data/processed/games_<имя>.csv)",
    )
    ap.add_argument(
        "--max-games",
        type=int,
        default=None,
        help="остановиться после N партий (читается только начало файла)",
    )
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    dst = args.out
    if dst is None:
        stem = args.src.name.replace(".pgn.zst", "").replace(".pgn", "")
        dst = Path("data/processed") / f"games_{stem.replace('lichess_', '')}.csv"
    parse_file(args.src, dst, args.max_games)


if __name__ == "__main__":
    main()
