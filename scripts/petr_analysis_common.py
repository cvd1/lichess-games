"""Общие функции анализов Петра: потоковое чтение, качество данных, графики.

Используются только стандартная библиотека и зависимости requirements.txt.
"""

import argparse
from collections import Counter
from contextlib import closing
import glob
import json
from pathlib import Path
import sqlite3
import tempfile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPEEDS = ("blitz", "rapid", "classical")
ALL_SPEEDS = ("ultrabullet", "bullet", "blitz", "rapid", "classical", "correspondence")
LABELS = {"ultrabullet": "Ультрапуля", "bullet": "Пуля", "blitz": "Блиц",
          "rapid": "Рапид", "classical": "Классика", "correspondence": "Переписка"}
COLORS = {"ultrabullet": "#8172b3", "bullet": "#c44e52", "blitz": "#4c72b0",
          "rapid": "#dd8452", "classical": "#55a868", "correspondence": "#937860"}
RESULTS = ("1-0", "1/2-1/2", "0-1")
BASE_COLUMNS = {
    "game_id", "speed", "result", "utc_date", "base_sec", "inc_sec",
    "white_elo", "black_elo", "n_plies", "has_eval",
}
QUALITY_LABELS = {
    "input_rows": "Прочитано строк",
    "missing_id": "Исключено строк без game_id",
    "unfinished_or_invalid_result": "Исключено незавершённых/неизвестных результатов",
    "duplicate_games": "Исключено повторов game_id (среди завершённых)",
    "unique_completed": "Уникальных завершённых партий",
    "speed_mismatches": "Расхождений метки speed и расчёта по времени",
    "unknown_time_control": "Партий без корректного контроля времени",
    "outside_selected_speeds": "Исключено вне выбранных контролей",
    "selected_games": "Партий в анализе",
    "unknown_date": "Партий в анализе с неизвестной датой",
    "invalid_n_plies": "Партий с отсутствующей/некорректной длиной",
    "invalid_white_elo": "Партий с отсутствующим/некорректным рейтингом белых",
    "invalid_black_elo": "Партий с отсутствующим/некорректным рейтингом чёрных",
    "games_with_eval": "Партий с has_eval = 1",
    "valid_player_games": "Наблюдений игрок–партия с пригодными метриками и рейтингом",
    "invalid_eval_player_games": "Наблюдений с has_eval = 1, но непригодными метриками",
    "unrated_eval_player_games": "Наблюдений с метриками, но без корректного рейтинга",
}


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("значение должно быть положительным")
    return number


def argument_parser(description, out):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("csv", nargs="+", help="CSV от parse_pgn.py или шаблон в кавычках")
    parser.add_argument("--out", type=Path, default=Path(out), help="каталог результатов")
    parser.add_argument("--chunksize", type=positive_int, default=100_000,
                        help="строк в блоке (по умолчанию 100000)")
    parser.add_argument("--speed-source", choices=("time_control", "event"),
                        default="time_control", help="контроль из времени (по умолчанию) или speed из PGN")
    return parser


def resolve_inputs(patterns):
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        if not matches:
            raise ValueError(f"Не найден входной файл: {pattern}")
        for match in matches:
            path = Path(match).resolve()
            if not path.is_file():
                raise ValueError(f"Ожидался CSV-файл: {match}")
            if path not in paths:
                paths.append(path)
    return paths


def nonnegative_integer(series, positive=False):
    values = pd.to_numeric(series, errors="coerce")
    valid = np.isfinite(values) & values.ge(1 if positive else 0) & values.eq(values.round())
    return values.where(valid)


def classify_speed(base, increment):
    """Границы https://lichess.org/faq#time-controls; секунды на одного игрока."""
    base = nonnegative_integer(base)
    increment = nonnegative_integer(increment)
    estimated = base + 40 * increment
    valid = estimated.notna() & estimated.gt(0)
    result = pd.Series("unknown", index=base.index, dtype="str")
    for lower, upper, name in (
        (0, 30, "ultrabullet"), (30, 180, "bullet"), (180, 480, "blitz"),
        (480, 1500, "rapid"), (1500, np.inf, "classical"),
    ):
        result.loc[valid & estimated.ge(lower) & estimated.lt(upper)] = name
    return result


def iter_chunks(paths, extra_columns, args, quality):
    """Читает только нужные поля; точная дедупликация между файлами на диске.

    Из завершённых строк с одинаковым game_id используется первая. Порядок
    файлов — порядок аргументов, внутри шаблона — лексикографический.
    """
    columns = sorted(BASE_COLUMNS | set(extra_columns))
    for path in paths:
        actual = set(pd.read_csv(path, nrows=0).columns)
        missing = set(columns) - actual
        if missing:
            raise ValueError(f"{path.name}: нет колонок {', '.join(sorted(missing))}. "
                             "Получите CSV текущим scripts/parse_pgn.py.")
    with tempfile.TemporaryDirectory(prefix="lichess-dedup-") as tmp:
        with closing(sqlite3.connect(str(Path(tmp) / "seen.sqlite"))) as conn:
            conn.execute("CREATE TABLE seen (id TEXT PRIMARY KEY) WITHOUT ROWID")
            for path in paths:
                for frame in pd.read_csv(path, usecols=columns, dtype=str,
                                         keep_default_na=False, chunksize=args.chunksize):
                    quality["input_rows"] += len(frame)
                    ids = frame["game_id"].str.strip()
                    quality["missing_id"] += int(ids.eq("").sum())
                    frame = frame.loc[ids.ne("")].copy()
                    frame["game_id"] = ids.loc[frame.index]
                    completed = frame["result"].isin(RESULTS)
                    quality["unfinished_or_invalid_result"] += int((~completed).sum())
                    frame = frame.loc[completed].copy()
                    unique = np.fromiter(
                        (conn.execute("INSERT OR IGNORE INTO seen VALUES (?)", (game_id,)).rowcount == 1
                         for game_id in frame["game_id"]), dtype=bool, count=len(frame))
                    conn.commit()
                    quality["duplicate_games"] += int((~unique).sum())
                    frame = frame.loc[unique].copy()
                    quality["unique_completed"] += len(frame)
                    computed = classify_speed(frame["base_sec"], frame["inc_sec"])
                    declared = frame["speed"].str.strip().str.lower()
                    computed = computed.mask(computed.eq("unknown") & declared.eq("correspondence"), "correspondence")
                    known = computed.ne("unknown") & declared.ne("")
                    quality["speed_mismatches"] += int((known & computed.ne(declared)).sum())
                    quality["unknown_time_control"] += int(computed.eq("unknown").sum())
                    frame["speed"] = computed if args.speed_source == "time_control" else declared
                    selected = frame["speed"].isin(getattr(args, "speeds", SPEEDS))
                    quality["outside_selected_speeds"] += int((~selected).sum())
                    frame = frame.loc[selected].copy()
                    quality["selected_games"] += len(frame)
                    if frame.empty:
                        continue
                    dates = pd.to_datetime(frame["utc_date"], format="%Y-%m-%d", errors="coerce")
                    frame["month"] = dates.dt.strftime("%Y-%m").fillna("unknown")
                    quality["unknown_date"] += int(dates.isna().sum())
                    for column in ("n_plies", "white_elo", "black_elo"):
                        frame[column] = nonnegative_integer(frame[column], positive=column != "n_plies")
                        quality[f"invalid_{column}"] += int(frame[column].isna().sum())
                    frame["has_eval"] = pd.to_numeric(frame["has_eval"], errors="coerce").eq(1)
                    quality["games_with_eval"] += int(frame["has_eval"].sum())
                    yield frame


def add_grouped(target, frame, keys, values):
    """Добавляет суммы по небольшому набору групп без хранения отдельных партий."""
    for key, group in frame.groupby(keys, observed=True, sort=False, dropna=False):
        key = key if isinstance(key, tuple) else (key,)
        current = target.setdefault(key, Counter())
        current.update({column: float(group[column].sum()) for column in values})


def grouped_frame(target, keys, values):
    rows = [dict(zip(keys, key), **{name: counts[name] for name in values})
            for key, counts in sorted(target.items())]
    frame = pd.DataFrame(rows, columns=list(keys) + list(values))
    # Явный тип нужен и для пустого набора: графики ожидают числа, не object.
    frame[list(values)] = frame[list(values)].astype(float)
    return frame


def ratio(numerator, denominator):
    return numerator / denominator.replace(0, np.nan)


def md_table(frame):
    if frame.empty:
        return "Нет подходящих наблюдений."
    def cell(value):
        if pd.isna(value):
            return "—"
        if isinstance(value, (float, np.floating)):
            value = f"{value:.3f}".rstrip("0").rstrip(".")
        return str(value).replace("|", "\\|").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, frame.columns)) + " |",
            "| " + " | ".join("---" for _ in frame.columns) + " |"]
    rows += ["| " + " | ".join(map(cell, row)) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join(rows)


def report_preamble(title, paths, args, quality):
    classification = ("base_sec + 40 × inc_sec; единые современные границы Lichess"
                      if args.speed_source == "time_control" else "исходная метка speed из PGN")
    return [f"# {title}", "", "## Данные и метод", "",
            "Входные файлы: " + ", ".join(f"`{path.name}`" for path in paths) + ".",
            f"Классификация контроля: {classification}.",
            "Выбранные контроли: " + ", ".join(LABELS[s] for s in getattr(args, "speeds", SPEEDS)) + ". "
            "Анализируются завершённые партии. Повторы game_id "
            "удалены между всеми файлами; сохранена первая завершённая запись.", "",
            "Месяц определяется по `utc_date`, а не имени дампа: в месячном файле "
            "могут встречаться партии с датой соседнего месяца.", "",
            "## Качество данных", "",
            md_table(pd.DataFrame([(QUALITY_LABELS.get(k, k), v) for k, v in quality.items()],
                                  columns=["Показатель", "Число"])), ""]


def write_metadata(out, paths, args, quality, status="ok"):
    metadata = {
        "status": status, "inputs": [{"name": p.name, "bytes": p.stat().st_size} for p in paths],
        "parameters": {k: str(v) if isinstance(v, Path) else v
                       for k, v in vars(args).items() if k != "csv"},
        "quality": dict(quality),
        "versions": {"numpy": np.__version__, "pandas": pd.__version__,
                     "matplotlib": matplotlib.__version__},
    }
    (out / "run_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
                                            encoding="utf-8")


def configure_plots():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 14, "axes.titleweight": "bold",
                         "figure.facecolor": "white", "axes.facecolor": "white",
                         "savefig.facecolor": "white"})


def savefig(fig, path):
    fig.tight_layout(pad=1.5)
    fig.savefig(path, dpi=160)
    plt.close(fig)


def empty_plot(ax, message="Недостаточно данных"):
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def save_tables(tables, out):
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(out / f"{name}.csv", index=False, float_format="%.6f", encoding="utf-8")
