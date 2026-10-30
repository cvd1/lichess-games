"""Задача 4: грубые ошибки и ACPL по рейтингу, контролю времени и месяцу.

    python scripts/analyze_blunders.py "data/processed/games_*.csv"
"""

from collections import Counter

import numpy as np
import pandas as pd

if __package__:
    from . import petr_analysis_common as common
else:
    import petr_analysis_common as common

plt = common.plt
EXTRA_COLUMNS = {f"{side}_{metric}" for side in ("white", "black")
                 for metric in ("acpl", "evaluated", "blunders")}
SUMS = ("player_games", "evaluated_moves", "blunders", "acpl_times_moves", "acpl_sum",
        "player_games_with_blunder", "possible_moves")


def player_rows(frame):
    """Одна строка = один игрок в одной партии; не уникальный человек."""
    parts = []
    for side in ("white", "black"):
        part = frame[["speed", "month", "has_eval"]].copy()
        part["side"] = side
        part["rating"] = frame[f"{side}_elo"]
        part["acpl"] = pd.to_numeric(frame[f"{side}_acpl"], errors="coerce")
        part["evaluated_moves"] = common.nonnegative_integer(frame[f"{side}_evaluated"], positive=True)
        part["blunders"] = common.nonnegative_integer(frame[f"{side}_blunders"])
        # У белых первый ход не оценивается: нет оценки исходной позиции.
        part["possible_moves"] = (np.ceil(frame["n_plies"] / 2) - 1).clip(lower=0) if side == "white" else np.floor(frame["n_plies"] / 2)
        part["valid_metrics"] = (
            part["has_eval"] & np.isfinite(part["acpl"]) & part["acpl"].between(0, 2000)
            & part["evaluated_moves"].notna() & part["blunders"].notna()
            & part["blunders"].le(part["evaluated_moves"])
            & part["evaluated_moves"].le(part["possible_moves"])
        )
        parts.append(part)
    return pd.concat(parts, ignore_index=True)


def derive_stats(frame, args):
    frame = frame.copy()
    frame["blunders_per_100_moves"] = 100 * common.ratio(frame["blunders"], frame["evaluated_moves"])
    frame["weighted_acpl"] = common.ratio(frame["acpl_times_moves"], frame["evaluated_moves"])
    frame["mean_blunders_per_player_game"] = common.ratio(frame["blunders"], frame["player_games"])
    frame["mean_player_game_acpl"] = common.ratio(frame["acpl_sum"], frame["player_games"])
    frame["player_games_with_blunder_share"] = common.ratio(frame["player_games_with_blunder"], frame["player_games"])
    frame["evaluated_move_coverage"] = common.ratio(frame["evaluated_moves"], frame["possible_moves"])
    frame["plot_eligible"] = frame["player_games"].ge(args.min_player_games) & frame["evaluated_moves"].ge(args.min_evaluated)
    if "rating_lower" in frame:
        frame["rating_upper_exclusive"] = frame["rating_lower"] + args.rating_bin
        frame["rating_center"] = frame["rating_lower"] + args.rating_bin / 2
    return frame


def analyze(paths, args):
    quality = Counter()
    groups = {"blunders_by_rating": (["rating_lower"], {}),
              "blunders_by_speed_rating": (["speed", "rating_lower"], {}),
              "blunders_by_month_speed_rating": (["month", "speed", "rating_lower"], {}),
              "blunders_by_color_rating": (["speed", "side", "rating_lower"], {})}
    coverage, monthly_coverage = {}, {}
    for frame in common.iter_chunks(paths, EXTRA_COLUMNS, args, quality):
        players = player_rows(frame)
        quality["invalid_eval_player_games"] += int((players["has_eval"] & ~players["valid_metrics"]).sum())
        quality["unrated_eval_player_games"] += int((players["valid_metrics"] & players["rating"].isna()).sum())
        players = players.loc[players["rating"].notna()].copy()
        players["rating_lower"] = (players["rating"] // args.rating_bin).astype(int) * args.rating_bin
        players["all_player_games"] = 1
        players["analyzed_player_games"] = players["valid_metrics"].astype(int)
        common.add_grouped(coverage, players, ["speed", "rating_lower"], ["all_player_games", "analyzed_player_games"])
        common.add_grouped(monthly_coverage, players, ["month", "speed"], ["all_player_games", "analyzed_player_games"])
        valid = players.loc[players["valid_metrics"]].copy()
        quality["valid_player_games"] += len(valid)
        if valid.empty:
            continue
        valid["player_games"] = 1
        valid["acpl_times_moves"] = valid["acpl"] * valid["evaluated_moves"]
        valid["acpl_sum"] = valid["acpl"]
        valid["player_games_with_blunder"] = valid["blunders"].gt(0).astype(int)
        for keys, target in groups.values():
            common.add_grouped(target, valid, keys, SUMS)
    tables = {name: derive_stats(common.grouped_frame(target, keys, SUMS), args)
              for name, (keys, target) in groups.items()}
    for name, keys, target in (("analysis_coverage", ["speed", "rating_lower"], coverage),
                               ("coverage_by_month", ["month", "speed"], monthly_coverage)):
        table = common.grouped_frame(target, keys, ["all_player_games", "analyzed_player_games"])
        table["analyzed_share"] = common.ratio(table["analyzed_player_games"], table["all_player_games"])
        if "rating_lower" in table:
            table["rating_upper_exclusive"] = table["rating_lower"] + args.rating_bin
            table["rating_center"] = table["rating_lower"] + args.rating_bin / 2
        tables[name] = table
    return tables, quality


def metric_plot(table, metric, ylabel, title, path, args):
    fig, ax = plt.subplots(figsize=(9, 5.2))
    plotted = False
    for speed in args.speeds:
        group = table.loc[table["speed"].eq(speed)].sort_values("rating_lower")
        if group.empty or not group["plot_eligible"].any():
            continue
        # Пустые и малые корзины разрывают линию, а не соединяют далёкие точки.
        lower = np.arange(int(group.rating_lower.min()), int(group.rating_lower.max()) + args.rating_bin, args.rating_bin)
        group = group.set_index("rating_lower").reindex(lower)
        y = group[metric].where(group["plot_eligible"].eq(True))
        ax.plot(lower + args.rating_bin / 2, y, marker="o", markersize=5, linewidth=1.7,
                color=common.COLORS[speed], label=common.LABELS[speed])
        plotted = True
    ax.set_title(title)
    ax.set_xlabel(f"Рейтинг: центры групп шириной {args.rating_bin}")
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    if plotted:
        ax.legend(frameon=False, ncol=min(3, len(args.speeds)), fontsize=10)
        ax.grid(axis="y", alpha=.2)
    else:
        common.empty_plot(ax, "Недостаточно оценённых наблюдений\nдля заданных порогов")
    fig.text(.5, .01, f"Группа: ≥{args.min_player_games} наблюдений игрок–партия и ≥{args.min_evaluated} оценённых ходов",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, .04, 1, 1))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plots(tables, args):
    common.configure_plots()
    table = tables["blunders_by_speed_rating"]
    for metric, ylabel, title, name in (
        ("blunders_per_100_moves", "Ошибок на 100 оценённых ходов", "Грубые ошибки и рейтинг", "blunders_by_rating"),
        ("weighted_acpl", "Средняя потеря оценки, сотые пешки", "ACPL с весом по числу оценённых ходов", "acpl_by_rating"),
        ("mean_blunders_per_player_game", "Грубых ошибок на участие в партии", "Число ошибок на игрока в партии", "blunders_per_game"),
    ):
        metric_plot(table, metric, ylabel, title, args.out / f"{name}.png", args)
    coverage = tables["analysis_coverage"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for speed in args.speeds:
        group = coverage.loc[coverage["speed"].eq(speed)].sort_values("rating_lower")
        if group.empty:
            continue
        axes[0].plot(group["rating_center"], group["analyzed_share"] * 100, marker=".",
                     label=common.LABELS[speed], color=common.COLORS[speed])
        axes[1].plot(group["rating_center"], group["analyzed_player_games"], marker=".", color=common.COLORS[speed])
    axes[0].set_title("Покрытие выборки пригодными оценками Stockfish")
    axes[0].set_ylabel("Участий с метриками, %")
    axes[1].set_ylabel("Участий с метриками, число")
    axes[1].set_xlabel(f"Рейтинг: центры групп шириной {args.rating_bin}")
    if coverage.empty:
        for ax in axes:
            common.empty_plot(ax)
    else:
        axes[0].legend(frameon=False, ncol=3, fontsize=10)
        axes[1].axhline(args.min_player_games, color="grey", ls="--", linewidth=1)
    for ax in axes:
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", alpha=.2)
    common.savefig(fig, args.out / "analysis_coverage.png")


def write_report(tables, paths, args, quality):
    by_rating = tables["blunders_by_rating"]
    by_speed = tables["blunders_by_speed_rating"]
    lines = common.report_preamble("Задача 4 — ошибки, ACPL и рейтинг", paths, args, quality)
    lines += ["## Определения", "",
              "- Единица наблюдения — игрок в партии: белые и чёрные учитываются отдельно со своим рейтингом.",
              "- Грубая ошибка — потеря ≥200 сотых пешки по принятому парсером определению. "
              "Используются готовые `*_blunders`; новые оценки движком не рассчитываются.",
              "- Основная частота = `100 × sum(blunders) / sum(evaluated)`. "
              "Нельзя делить на все ходы партии или заменять отсутствующие оценки нулём.",
              "- ACPL по ходам = `sum(acpl × evaluated) / sum(evaluated)`. "
              "Это взвешенное среднее; исходные ACPL округлены парсером до 0.1, поэтому результат приближённый.",
              "- Дополнительно сохранены среднее число ошибок на участие в партии, "
              "средний ACPL по участиям и доля участий хотя бы с одной грубой ошибкой.",
              f"- Рейтинговые группы: [нижняя граница, нижняя + {args.rating_bin}). "
              f"Графики основных метрик показывают группы с ≥{args.min_player_games} участиями "
              f"и ≥{args.min_evaluated} оценёнными ходами. CSV содержит и меньшие группы.", "",
              "## Сводка по рейтингам, все выбранные контроли", "",
              "Объединённая таблица описывает состав выборки. Для сравнения рейтингов используйте также разрез по контролям и месяцам.", ""]
    brief = by_rating[["rating_lower", "rating_upper_exclusive", "player_games", "evaluated_moves",
                       "blunders", "blunders_per_100_moves", "weighted_acpl"]].copy()
    brief.columns = ["Рейтинг от", "Рейтинг до, не включая", "Участий", "Оценённых ходов",
                     "Грубых ошибок", "Ошибок на 100 ходов", "ACPL по ходам"]
    lines += [common.md_table(brief), "", "## Выводы по этой выборке", ""]
    if by_rating.empty:
        lines += ["**Пригодных оценённых наблюдений нет.** Анализ ошибок выполнить нельзя; "
                  "нулевая частота ошибок не утверждается. Нужен CSV с `has_eval=1`, "
                  "корректными рейтингами, ACPL и положительными `*_evaluated`."]
    else:
        moves = by_rating["evaluated_moves"].sum()
        blunders = by_rating["blunders"].sum()
        acpl = by_rating["acpl_times_moves"].sum() / moves
        coverage = tables["analysis_coverage"]
        total = coverage["all_player_games"].sum()
        included = by_rating["player_games"].sum()
        lines += [f"- Пригодных участий: {int(included):,} из {int(total):,} с известным рейтингом "
                  f"({included / total:.2%}).".replace(",", " "),
                  f"- На {int(moves):,} оценённых ходов приходится {int(blunders):,} грубых ошибок: "
                  f"{100 * blunders / moves:.2f} на 100 ходов; ACPL по ходам — {acpl:.2f}.".replace(",", " ")]
        for speed in args.speeds:
            group = by_speed.loc[by_speed["speed"].eq(speed) & by_speed["plot_eligible"]].sort_values("rating_lower")
            if len(group) < 2:
                lines += [f"- {common.LABELS[speed]}: меньше двух групп достигают порогов; "
                          "для описания связи рейтинга и ошибок данных недостаточно."]
                continue
            low, high = group.iloc[0], group.iloc[-1]
            lines += [f"- {common.LABELS[speed]}: в нижней доступной группе "
                      f"[{int(low.rating_lower)}, {int(low.rating_upper_exclusive)}) — "
                      f"{low.blunders_per_100_moves:.2f} ошибки на 100 ходов; в верхней "
                      f"[{int(high.rating_lower)}, {int(high.rating_upper_exclusive)}) — "
                      f"{high.blunders_per_100_moves:.2f}. Это сравнение крайних пригодных групп, "
                      "а не доказательство монотонной зависимости."]
    lines += ["", "## Ограничения", "",
              "- Проанализированные Stockfish партии — отобранная часть базы. "
              "График покрытия показывает различия отбора по рейтингу; результаты нельзя автоматически переносить на все партии.",
              "- Оценки обрезаны парсером до ±1000 cp; мат заменён на ±1000. "
              "Первый ход белых и ходы без двух соседних оценок не учитываются. "
              "Это учебная метрика проекта, она не обязана совпадать с метками ошибок на сайте Lichess.",
              "- В выборку метрик входит участие с `has_eval=1`, корректным рейтингом, "
              "конечным ACPL от 0 до 2000, целыми `evaluated>0` и `0≤blunders≤evaluated`. "
              "Количество оценённых ходов не может превышать доступное по `n_plies`.",
              "- Частота на оценённый ход — основная метрика. Число ошибок на партию зависит от её длины "
              "и полноты оценок; считать его самостоятельной мерой силы игрока нельзя.",
              "- Рейтинги относятся к разным пулам и периодам; единая классификация времени не пересчитывает рейтинги. "
              "Контроли и месяцы сохранены раздельно для проверки смешения выборок.",
              "- Идентификаторов игроков в CSV нет: участия повторных игроков не независимы. "
              "Пороги на графиках — фильтр малых групп, не проверка статистической значимости. "
              "Причинные выводы и доверительные интервалы по независимым ходам не строятся.", "",
              "## Графики", ""]
    for name, label in (("blunders_by_rating", "Частота ошибок"), ("acpl_by_rating", "ACPL"),
                        ("blunders_per_game", "Ошибки на участие"), ("analysis_coverage", "Покрытие")):
        lines += [f"![{label}]({name}.png)", ""]
    lines += ["## Таблицы", "",
              "- `blunders_by_rating.csv`: объединённые рейтинговые группы.",
              "- `blunders_by_speed_rating.csv`: основные сравнения внутри контролей.",
              "- `blunders_by_month_speed_rating.csv`: контроль периода и состава выборки.",
              "- `blunders_by_color_rating.csv`: белые и чёрные раздельно внутри контроля.",
              "- `analysis_coverage.csv`, `coverage_by_month.csv`: доля пригодных участий среди всех с известным рейтингом.",
              "- `run_metadata.json`: входы, параметры, версии библиотек и качество данных.", ""]
    (args.out / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = common.argument_parser(__doc__, "reports/blunders")
    parser.add_argument("--rating-bin", type=common.positive_int, default=200,
                        help="ширина рейтинговой группы (по умолчанию 200)")
    parser.add_argument("--min-player-games", type=common.positive_int, default=30,
                        help="минимум участий игрок–партия в группе на графике (30)")
    parser.add_argument("--min-evaluated", type=common.positive_int, default=100,
                        help="минимум оценённых ходов в группе на графике (100)")
    parser.add_argument("--speeds", nargs="+", choices=common.ALL_SPEEDS, default=list(common.ALL_SPEEDS),
                        help="контроли для задачи 4 (по умолчанию все известные)")
    args = parser.parse_args()
    args.speeds = list(dict.fromkeys(args.speeds))
    try:
        paths = common.resolve_inputs(args.csv)
        tables, quality = analyze(paths, args)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    common.save_tables(tables, args.out)
    plots(tables, args)
    write_report(tables, paths, args, quality)
    common.write_metadata(args.out, paths, args, quality, "ok" if quality["valid_player_games"] else "no_evaluations")
    print(f"Готово: {args.out}/summary.md; пригодных участий: {quality['valid_player_games']:,}")


if __name__ == "__main__":
    main()
