"""Задача 3: сравнение блица, рапида и классики на одном или нескольких CSV.

    python scripts/analyze_speed.py "data/processed/games_*.csv"
"""

from collections import Counter

import numpy as np
import pandas as pd

if __package__:
    from . import petr_analysis_common as common
else:
    import petr_analysis_common as common

plt = common.plt
SUMS = (
    "games", "white_wins", "draws", "black_wins", "length_games", "plies_sum",
    "white_rating_games", "white_rating_sum", "black_rating_games", "black_rating_sum",
    "paired_rating_games", "rating_gap_sum",
)


def histogram_quantile(histogram, q):
    """Точный линейный квантиль целочисленных значений из частотной таблицы."""
    if not histogram:
        return np.nan
    values = np.array(sorted(histogram), dtype=float)
    cumulative = np.cumsum([histogram[value] for value in values])
    position = (cumulative[-1] - 1) * q
    lower, upper = int(np.floor(position)), int(np.ceil(position))
    a, b = values[np.searchsorted(cumulative, [lower + 1, upper + 1])]
    return float(a + (b - a) * (position - lower))


def derive_stats(frame):
    frame = frame.copy()
    for count, name in (("white_wins", "white_win_share"), ("draws", "draw_share"),
                        ("black_wins", "black_win_share")):
        frame[name] = common.ratio(frame[count], frame["games"])
    frame["decisive_share"] = common.ratio(frame["white_wins"] + frame["black_wins"], frame["games"])
    frame["white_score"] = common.ratio(frame["white_wins"] + 0.5 * frame["draws"], frame["games"])
    frame["mean_plies"] = common.ratio(frame["plies_sum"], frame["length_games"])
    for side in ("white", "black"):
        frame[f"{side}_mean_rating"] = common.ratio(frame[f"{side}_rating_sum"], frame[f"{side}_rating_games"])
    frame["mean_abs_rating_gap"] = common.ratio(frame["rating_gap_sum"], frame["paired_rating_games"])
    return frame


def analyze(paths, args):
    quality = Counter()
    by_speed, by_month, terminations = {}, {}, {}
    lengths, ratings = {}, {}
    for frame in common.iter_chunks(paths, {"termination"}, args, quality):
        frame["games"] = 1
        for result, column in zip(common.RESULTS, ("white_wins", "draws", "black_wins")):
            frame[column] = frame["result"].eq(result).astype(int)
        frame["length_games"] = frame["n_plies"].notna().astype(int)
        frame["plies_sum"] = frame["n_plies"].fillna(0)
        for side in ("white", "black"):
            frame[f"{side}_rating_games"] = frame[f"{side}_elo"].notna().astype(int)
            frame[f"{side}_rating_sum"] = frame[f"{side}_elo"].fillna(0)
        gap = (frame["white_elo"] - frame["black_elo"]).abs()
        frame["paired_rating_games"] = gap.notna().astype(int)
        frame["rating_gap_sum"] = gap.fillna(0)
        frame["termination"] = frame["termination"].str.strip().replace("", "Unknown")
        common.add_grouped(by_speed, frame, ["speed"], SUMS)
        common.add_grouped(by_month, frame, ["month", "speed"], SUMS)
        common.add_grouped(terminations, frame, ["speed", "termination"], ["games"])
        for speed, group in frame.groupby("speed", sort=False):
            lengths.setdefault(speed, Counter()).update(group["n_plies"].dropna().astype(int).value_counts().to_dict())
            for side in ("white", "black"):
                ratings.setdefault((speed, side), Counter()).update(
                    group[f"{side}_elo"].dropna().astype(int).value_counts().to_dict())
    summary = common.grouped_frame(by_speed, ["speed"], SUMS).set_index("speed")
    summary = summary.reindex(common.SPEEDS, fill_value=0).reset_index()
    summary = derive_stats(summary)
    summary["share_selected_games"] = summary["games"] / max(1, summary["games"].sum())
    for name, q in (("q25", .25), ("median", .5), ("q75", .75)):
        summary[f"{name}_plies"] = [histogram_quantile(lengths.get(speed, {}), q) for speed in summary["speed"]]
        for side in ("white", "black"):
            summary[f"{side}_{name}_rating"] = [
                histogram_quantile(ratings.get((speed, side), {}), q) for speed in summary["speed"]]
    monthly = derive_stats(common.grouped_frame(by_month, ["month", "speed"], SUMS))
    terms = common.grouped_frame(terminations, ["speed", "termination"], ["games"])
    terms["share_within_speed"] = common.ratio(terms["games"], terms.groupby("speed")["games"].transform("sum"))
    length_hist = pd.DataFrame([(speed, plies, n) for speed in common.SPEEDS
                               for plies, n in sorted(lengths.get(speed, {}).items())],
                              columns=["speed", "n_plies", "games"])
    rating_hist = pd.DataFrame([(speed, side, rating, n) for speed in common.SPEEDS
                               for side in ("white", "black")
                               for rating, n in sorted(ratings.get((speed, side), {}).items())],
                              columns=["speed", "side", "rating", "player_games"])
    return {"speed_summary": summary, "speed_by_month": monthly, "terminations": terms,
            "length_distribution": length_hist, "rating_distribution": rating_hist}, quality


def plots(tables, out):
    common.configure_plots()
    summary = tables["speed_summary"]
    x = np.arange(3)
    labels = [common.LABELS[s] for s in common.SPEEDS]
    colors = [common.COLORS[s] for s in common.SPEEDS]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    bars = ax.bar(labels, summary["games"], color=colors, width=.6)
    ax.bar_label(bars, labels=[f"{int(n):,}".replace(",", " ") for n in summary["games"]], padding=5)
    ax.set_ylim(0, max(1, summary["games"].max()) * 1.15)
    ax.set_ylabel("Число партий")
    ax.set_title("Объём выборки по контролям времени")
    common.savefig(fig, out / "games_by_speed.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    bottom = np.zeros(3)
    for column, label, color in (("white_win_share", "Победа белых", "#4c72b0"),
                                 ("draw_share", "Ничья", "#9ba3ad"),
                                 ("black_win_share", "Победа чёрных", "#dd8452")):
        values = summary[column].fillna(0).to_numpy() * 100
        ax.bar(labels, values, bottom=bottom, label=label, color=color, width=.6)
        for i, value in enumerate(values):
            if value >= 4:
                ax.text(i, bottom[i] + value / 2, f"{value:.1f}%", ha="center", va="center", color="white", fontsize=10)
        bottom += values
    for i, n in enumerate(summary["games"]):
        if n == 0:
            ax.text(i, 5, "Нет данных", ha="center")
    for i, row in summary.iterrows():
        if row["games"] > 0:
            ax.text(i, 103, f"Ничьи: {row['draw_share']:.2%}", ha="center", fontsize=10)
    ax.set_ylim(0, 111)
    ax.set_yticks(np.arange(0, 101, 20))
    ax.set_ylabel("Доля завершённых партий, %")
    ax.set_title("Результаты партий")
    ax.legend(loc="upper center", bbox_to_anchor=(.5, -.1), ncol=3, frameon=False)
    common.savefig(fig, out / "outcomes_by_speed.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for i, row in summary.iterrows():
        if pd.notna(row["median_plies"]):
            ax.errorbar(i, row["median_plies"],
                        yerr=[[row["median_plies"] - row["q25_plies"]], [row["q75_plies"] - row["median_plies"]]],
                        fmt="o", color=colors[i], capsize=10, markersize=8)
            ax.scatter(i, row["mean_plies"], marker="D", color=colors[i], s=30, alpha=.6)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Длина партии, полуходы")
    ax.set_title("Длина партии: медиана и средние 50%")
    ax.text(.02, .97, "Круг — медиана; ромб — среднее; отрезок — Q25–Q75", transform=ax.transAxes,
            va="top", fontsize=9)
    highest = max(summary["q75_plies"].max(), summary["mean_plies"].max())
    ax.set_ylim(0, highest * 1.2 if pd.notna(highest) and highest > 0 else 1)
    if not summary["length_games"].sum():
        common.empty_plot(ax)
    common.savefig(fig, out / "length_by_speed.png")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    for side, shift, color, label in (("white", -.12, "#4c72b0", "Белые"), ("black", .12, "#dd8452", "Чёрные")):
        mid = summary[f"{side}_median_rating"].to_numpy(dtype=float)
        lower = summary[f"{side}_q25_rating"].to_numpy(dtype=float)
        upper = summary[f"{side}_q75_rating"].to_numpy(dtype=float)
        ax.errorbar(x + shift, mid, yerr=[mid - lower, upper - mid], fmt="o", capsize=7, label=label, color=color)
    ax.set_xticks(x, labels)
    ax.set_ylabel("Рейтинг игрока в данной партии")
    ax.set_title("Рейтинги: медиана и межквартильный размах")
    ax.legend(frameon=False)
    common.savefig(fig, out / "ratings_by_speed.png")

    fig, ax = plt.subplots(figsize=(9, 5))
    terms = tables["terminations"]
    if terms.empty:
        common.empty_plot(ax)
    else:
        pivot = terms.pivot(index="speed", columns="termination", values="share_within_speed")
        pivot = pivot.reindex(common.SPEEDS).fillna(0) * 100
        bottom = np.zeros(3)
        palette = plt.get_cmap("tab20")
        for i, column in enumerate(pivot.columns):
            ax.bar(labels, pivot[column], bottom=bottom, label=column, color=palette(i % 20), width=.6)
            bottom += pivot[column].to_numpy()
        ax.legend(loc="upper left", bbox_to_anchor=(1, 1), frameon=False, fontsize=9)
        ax.set_ylim(0, 100)
    ax.set_ylabel("Доля партий внутри контроля, %")
    ax.set_title("Способы завершения по тегу Termination")
    common.savefig(fig, out / "termination_by_speed.png")


def write_report(tables, paths, args, quality):
    summary = tables["speed_summary"]
    lines = common.report_preamble("Задача 3 — контроли времени", paths, args, quality)
    brief = summary[["speed", "games", "draw_share", "mean_plies", "median_plies",
                     "white_mean_rating", "black_mean_rating"]].copy()
    brief["speed"] = brief["speed"].map(common.LABELS)
    brief["draw_share"] *= 100
    brief.columns = ["Контроль", "Партий", "Ничьих, %", "Среднее, полуходы", "Медиана, полуходы",
                     "Рейтинг белых", "Рейтинг чёрных"]
    lines += ["## Сравнение", "", common.md_table(brief), "", "## Выводы по этой выборке", ""]
    available = summary[summary["games"].gt(0)]
    if available.empty:
        lines += ["Подходящих партий нет. Проверьте входные файлы и классификацию контроля."]
    else:
        most = available.loc[available["games"].idxmax()]
        lines += [f"- Больше всего партий: {common.LABELS[most['speed']]} — {int(most['games']):,}.".replace(",", " ")]
        if len(available) >= 2:
            draw = available.loc[available["draw_share"].idxmax()]
            lines += [f"- Наибольшая наблюдаемая доля ничьих: {common.LABELS[draw['speed']]} — {draw['draw_share']:.2%}."]
        for row in available.itertuples():
            if pd.notna(row.median_plies):
                lines += [f"- {common.LABELS[row.speed]}: медиана {row.median_plies:g} полуходов, "
                          f"средние 50% партий — от {row.q25_plies:g} до {row.q75_plies:g} полуходов."]
    lines += ["", "## Ограничения и определения", "",
              "- Длина — `n_plies`, число отдельных ходов сторон. Это не время в минутах. "
              "В текущем CSV нет показаний часов, только флаг `has_clock`.",
              "- Доли результатов считаются от всех завершённых партий соответствующего контроля. "
              "Результативность — доля побед (1 − доля ничьих); `white_score` учитывает ничью как 0.5.",
              "- `Normal` в Termination не позволяет разделить мат, сдачу и другие нормальные окончания. "
              "Пустой тег показан как `Unknown`; более подробные причины не выдумываются.",
              "- Классы времени вычислены по единым границам, если не задан `--speed-source event`. "
              "Это полезно для старых PGN, но не превращает исторические рейтинги в рейтинги современного пула.",
              "- Рейтинги описывают участия игроков в партиях, а не уникальных людей. "
              "Средние используют только известные значения; их знаменатели сохранены в CSV.",
              "- Сравнение описательное: рейтинг, год, состав игроков и отбор месяцев могут влиять на результат. "
              "Статистическая значимость и причинный эффект контроля времени здесь не утверждаются.",
              "", "Источник границ: [Lichess FAQ](https://lichess.org/faq#time-controls).", "",
              "## Графики", ""]
    for name, label in (("games_by_speed", "Объём выборки"), ("outcomes_by_speed", "Исходы"),
                        ("length_by_speed", "Длина партий"), ("ratings_by_speed", "Рейтинги"),
                        ("termination_by_speed", "Завершение")):
        lines += [f"![{label}]({name}.png)", ""]
    lines += ["## Таблицы", "", "- `speed_summary.csv`: сводка и знаменатели метрик.",
              "- `speed_by_month.csv`: те же средние и доли по месяцам.",
              "- `terminations.csv`: число и доля каждого окончания внутри контроля.",
              "- `length_distribution.csv`, `rating_distribution.csv`: точные частотные распределения.",
              "- `run_metadata.json`: входы, параметры, версии библиотек и качество данных.", ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = common.argument_parser(__doc__, "reports/speed")
    args = parser.parse_args()
    try:
        paths = common.resolve_inputs(args.csv)
        tables, quality = analyze(paths, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    common.save_tables(tables, args.out)
    plots(tables, args.out)
    write_report(tables, paths, args, quality)
    common.write_metadata(args.out, paths, args, quality, "ok" if quality["selected_games"] else "no_data")
    print(f"Готово: {args.out}/summary.md; партий: {quality['selected_games']:,}")


if __name__ == "__main__":
    main()
