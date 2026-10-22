"""Проверка parse_pgn.py на маленьких примерах.

Запуск из корня проекта: python -m tests.test_parse_pgn
"""

import scripts.parse_pgn as pp

ANALYZED = """[Event "Rated Blitz game"]
[Site "https://lichess.org/abcd1234"]
[White "a"]
[Black "b"]
[Result "1-0"]
[UTCDate "2024.01.05"]
[UTCTime "10:00:00"]
[WhiteElo "1800"]
[BlackElo "1700"]
[ECO "C20"]
[Opening "King's Pawn Game"]
[TimeControl "300+0"]
[Termination "Normal"]

1. e4 { [%eval 0.20] [%clk 0:05:00] } 1... e5 { [%eval 0.10] [%clk 0:05:00] } 2. Nf3 { [%eval 0.15] } 2... Nc6 { [%eval 0.90] } 3. Bb5 { [%eval 0.70] } 3... a6 { [%eval #3] } 4. Ba4 { [%eval -2.50] } 1-0
"""

PLAIN = """[Event "Rated Classical game"]
[Site "https://lichess.org/plain0001"]
[Result "1/2-1/2"]
[WhiteElo "?"]
[BlackElo "1500"]

1. d4 d5 2. c4 e6 1/2-1/2
"""


def rows(text):
    result = []
    for headers, movetext in pp.iter_games(text.splitlines(keepends=True)):
        row = pp.build_row(headers, movetext)
        assert row is not None, "партия отброшена парсером"
        result.append(row)
    return result


def test_parse_eval():
    assert pp.parse_eval("[%eval 0.17]") == 17
    assert pp.parse_eval("[%eval -2.5] [%clk 0:00:01]") == -250
    assert pp.parse_eval("[%eval #3]") == 1000
    assert pp.parse_eval("[%eval #-3]") == -1000
    assert pp.parse_eval("[%eval 25.0]") == 1000  # обрезка
    assert pp.parse_eval("[%clk 0:05:00]") is None


def test_analyzed_game():
    (r,) = rows(ANALYZED)
    assert r["n_plies"] == 7 and r["has_eval"] == 1 and r["has_clock"] == 1
    assert r["moves_6"] == "e4 e5 Nf3 Nc6 Bb5 a6"
    assert r["evals"] == "20,10,15,90,70,1000,-250"
    # потери за ход (оценки 20,10,15,90,70,1000,-250; ход 1 не считается):
    # ход 2 (чёрные): 10-20 -> оценка ушла в пользу чёрных, потеря 0
    # ход 3 (белые): 10-15 -> рост, потеря 0
    # ход 4 (чёрные): 90-15 = 75 -> неточность
    # ход 5 (белые): 90-70 = 20
    # ход 6 (чёрные): 1000-70 = 930 -> грубая ошибка
    # ход 7 (белые): 1000-(-250) = 1250 -> грубая ошибка
    assert (r["white_evaluated"], r["black_evaluated"]) == (3, 3)
    assert (r["white_inaccuracies"], r["white_mistakes"], r["white_blunders"]) == (
        0,
        0,
        1,
    )
    assert (r["black_inaccuracies"], r["black_mistakes"], r["black_blunders"]) == (
        1,
        0,
        1,
    )
    assert r["white_acpl"] == round((0 + 20 + 1250) / 3, 1)
    assert r["black_acpl"] == round((0 + 75 + 930) / 3, 1)


def test_plain_game():
    (r,) = rows(PLAIN)
    assert r["has_eval"] == 0 and r["evals"] == ""
    assert r["white_elo"] is None and r["black_elo"] == 1500
    assert r["white_acpl"] is None and r["white_blunders"] is None
    assert r["speed"] == "classical" and r["n_plies"] == 4


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
