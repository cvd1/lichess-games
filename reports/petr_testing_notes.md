# Проверка задач 3–4

**Юдин Пётр · 29 октября 2026 года**

Рабочие версии `scripts/analyze_speed.py` и `scripts/analyze_blunders.py`
проверены на официальном дампе Lichess за январь 2013 года. Дамп скачан
отдельно скриптом `download_data.py` и преобразован в CSV через `parse_pgn.py`.
Оба исходных скрипта использованы без изменений.

Источник: [lichess_db_standard_rated_2013-01.pgn.zst](https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst).
Полученный файл: `data/processed/games_2013-01.csv`.

- В CSV — 121 332 партии, из них 518 с оценками Stockfish.
- В сравнение блица, рапида и классики вошли 88 080 партий.
- Для анализа ошибок учтены 1 036 участий игроков: белые и чёрные отдельно.
- Оба скрипта завершились без ошибок; таблицы, графики и выводы сохранены
  в `reports/speed/` и `reports/blunders/`.
- 29 октября повторно пройдены 3 проверки парсера и 10 тестов анализа.

Команды проверки из корня проекта с активированным окружением:

```bash
python -m tests.test_parse_pgn
python -m unittest tests.test_petr_analysis -v
python scripts/analyze_speed.py data/processed/games_2013-01.csv
python scripts/analyze_blunders.py data/processed/games_2013-01.csv
```

Общий многомесячный датасет команды на момент проверки не получен.
На 29 октября ссылки на него в репозитории нет. Текущие отчёты относятся
к январскому дампу, указанному выше.

После получения общего набора CSV нужно сохранить его в отдельную папку
`data/processed/full/` и повторить анализ:

```bash
python scripts/analyze_speed.py "data/processed/full/*.csv"
python scripts/analyze_blunders.py "data/processed/full/*.csv"
```

В эту папку следует поместить только файлы общего датасета. Список прочитанных
CSV и число обработанных строк записываются в `run_metadata.json` в каждом
каталоге отчётов. После запуска на общем датасете нужно обновить эту заметку.
