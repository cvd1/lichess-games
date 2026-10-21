"""
Первичная простая аналитика по CSV, который делает parse_pgn.py.

Считает по всей выборке и сохраняет в reports/:
  - summary.md              сводка с числами (для отчёта)
  - result_share.png        доли результатов
  - top_openings.png        самые частые дебюты
  - white_score_by_diff.png результат белых в зависимости от разницы рейтингов
  - speed_stats.png         доли контролей времени
  - rating_hist.png         распределение рейтингов
  - games_per_day.png       партий по дням
Покрывает простые версии задач 1, 2, 3 и проверку данных для задачи 4.

Пример:
    python scripts/analyze_basic.py data/processed/games_2013-01.csv
"""

import argparse
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.switch_backend("Agg")  # рисуем только в файлы, без окон

RESULT_LABELS = {"1-0": "победа белых", "0-1": "победа чёрных", "1/2-1/2": "ничья"}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    """DataFrame -> markdown-таблица без внешних зависимостей."""
    cols = [str(df.index.name or "")] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, *row in df.itertuples(name=None):  # itertuples сохраняет типы колонок
        cells = [str(idx)]
        for v in row:
            cells.append(
                floatfmt.format(v) if isinstance(v, (float, np.floating)) else str(v)
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, keep_default_na=False, na_values=[""])
    df["utc_date"] = pd.to_datetime(df["utc_date"], errors="coerce")
    # счёт белых: победа = 1, ничья = 0.5, поражение = 0 (незавершённые "*" исключаем)
    df = df[df["result"].isin(RESULT_LABELS)].copy()
    df["white_score"] = df["result"].map({"1-0": 1.0, "1/2-1/2": 0.5, "0-1": 0.0})
    df["opening_family"] = df["opening"].fillna("").str.split(":").str[0].str.strip()
    df["rating_diff"] = df["white_elo"] - df["black_elo"]
    return df


def savefig(fig, path: Path):
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def overview(df, out, lines):
    n = len(df)
    lines.append("## Общая сводка\n")
    lines.append(f"- партий: **{n:,}**".replace(",", " "))
    lines.append(
        f"- период: {df['utc_date'].min():%Y-%m-%d} – {df['utc_date'].max():%Y-%m-%d}"
    )
    lines.append(
        f"- партий с известными рейтингами обоих игроков: "
        f"{df['rating_diff'].notna().sum():,}".replace(",", " ")
    )
    lines.append(f"- средняя длина партии: {df['n_plies'].mean():.1f} полуходов")
    lines.append("")

    share = df["result"].value_counts(normalize=True).rename(index=RESULT_LABELS)
    lines.append("### Результаты партий\n")
    share_df = share.to_frame("доля")
    share_df.index.name = "результат"
    lines.append(md_table(share_df))
    lines.append("")
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.bar(
        share.index, share.values, color=["#4c72b0", "#dd8452", "#8c8c8c"][: len(share)]
    )
    ax.set_ylabel("Доля партий")
    ax.set_title("Результаты партий")
    savefig(fig, out / "result_share.png")


def openings(df, out, lines, top=15):
    g = df.groupby("opening_family")["white_score"].agg(["count", "mean"])
    g.columns = ["партий", "счёт белых"]
    g = g[g.index != ""].sort_values("партий", ascending=False).head(top)
    g.insert(1, "доля партий", g["партий"] / len(df))
    g.index.name = "дебют"
    lines.append("## Задача 1 (простая версия): популярные дебюты\n")
    lines.append(
        "Счёт белых – средний результат белых (победа 1, ничья 0.5, поражение 0).\n"
    )
    lines.append(md_table(g))
    lines.append("")
    fig, ax = plt.subplots(figsize=(8, 6))
    g["партий"][::-1].plot.barh(ax=ax, color="#4c72b0")
    ax.set_xlabel("Число партий")
    ax.set_ylabel("")
    ax.set_title(f"Топ-{top} дебютов по числу партий")
    savefig(fig, out / "top_openings.png")


def rating_diff(df, out, lines, step=50, limit=400, min_n=50):
    d = df.dropna(subset=["rating_diff"]).copy()
    d = d[d["rating_diff"].abs() <= limit]
    bins = np.arange(-limit, limit + step, step).tolist()
    d["bin"] = pd.cut(d["rating_diff"], bins=bins)
    g = d.groupby("bin", observed=True)["white_score"].agg(["count", "mean"])
    g = g[g["count"] >= min_n]
    centers = np.array([iv.mid for iv in g.index])
    lines.append("## Задача 2 (простая версия): результат белых от разницы рейтингов\n")
    lines.append(
        f"Разница = рейтинг белых − рейтинг чёрных, корзины по {step}, "
        f"показаны корзины от {min_n} партий.\n"
    )
    t = g.copy()
    t.index = pd.Index(
        [f"{int(iv.left)}…{int(iv.right)}" for iv in g.index], name="разница"
    )
    t.columns = ["партий", "счёт белых"]
    lines.append(md_table(t))
    lines.append("")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(centers, g["mean"].values, marker="o")
    ax.axhline(0.5, color="grey", ls="--", lw=1)
    ax.set_xlabel("Рейтинг белых − рейтинг чёрных")
    ax.set_ylabel("Средний счёт белых")
    ax.set_title("Результат белых от разницы рейтингов")
    savefig(fig, out / "white_score_by_diff.png")


def speeds(df, out, lines):
    g = (
        df.groupby("speed")
        .agg(
            n=("game_id", "count"),
            draws=("result", lambda s: (s == "1/2-1/2").mean()),
            score=("white_score", "mean"),
            plies=("n_plies", "mean"),
        )
        .sort_values("n", ascending=False)
    )
    g.insert(1, "share", g["n"] / len(df))
    g.columns = [
        "партий",
        "доля партий",
        "доля ничьих",
        "счёт белых",
        "средняя длина, полуходов",
    ]
    g.index.name = "контроль"
    lines.append("## Задача 3 (простая версия): контроли времени\n")
    lines.append(md_table(g))
    lines.append("")
    fig, ax = plt.subplots(figsize=(6, 4))
    g["доля партий"].plot.bar(ax=ax, color="#4c72b0")
    ax.set_ylabel("Доля партий")
    ax.set_xlabel("")
    ax.set_title("Партии по контролю времени")
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    savefig(fig, out / "speed_stats.png")

    term = df["termination"].value_counts(normalize=True).to_frame("доля")
    term.index.name = "завершение"
    lines.append("### Способ завершения партий\n")
    lines.append(md_table(term))
    lines.append("")


def ratings(df, out, lines):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(
        pd.concat([df["white_elo"], df["black_elo"]]).dropna(), bins=60, color="#4c72b0"
    )
    ax.set_xlabel("Рейтинг")
    ax.set_ylabel("Число игроков в партиях")
    ax.set_title("Распределение рейтингов")
    savefig(fig, out / "rating_hist.png")

    per_day = df.groupby(df["utc_date"].dt.normalize()).size()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(per_day.index, per_day.values, marker="o", ms=3)
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )  # короткие подписи дат
    ax.set_xlabel("Дата")
    ax.set_ylabel("Партий в день")
    ax.set_title("Активность по дням")
    savefig(fig, out / "games_per_day.png")


def data_quality(df, lines):
    lines.append("## Проверка данных (важно для задачи 4 и планирования)\n")
    ev = df["has_eval"].mean()
    clk = df["has_clock"].mean()
    lines.append(f"- партий с оценками Stockfish (`%eval`): **{ev:.1%}**")
    lines.append(f"- партий с часами (`%clk`): **{clk:.1%}**")
    if df["has_eval"].sum() == 0:
        lines.append(
            "- ⚠ В этой выборке нет ни одной партии с оценками движка: "
            "задачу 4 (грубые ошибки) на ней сделать нельзя, "
            "нужны более поздние месяцы."
        )
    lines.append(
        f"- пропущен рейтинг хотя бы у одного игрока: "
        f"{df['rating_diff'].isna().mean():.2%}"
    )
    lines.append(f"- дублей game_id: {df['game_id'].duplicated().sum()}")
    lines.append("")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("csv", type=Path, help="CSV от parse_pgn.py")
    ap.add_argument("--out", type=Path, default=Path("reports"))
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    df = load(args.csv)
    lines = [f"# Первичная аналитика: {args.csv.name}\n"]
    overview(df, args.out, lines)
    openings(df, args.out, lines)
    rating_diff(df, args.out, lines)
    speeds(df, args.out, lines)
    ratings(df, args.out, lines)
    data_quality(df, lines)
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Готово: {args.out}/summary.md и графики ({len(df)} партий)")


if __name__ == "__main__":
    main()
