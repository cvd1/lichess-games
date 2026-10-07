"""
Скачивает один месячный дамп партий Lichess (сырые данные) для курсового проекта.
Источник: https://database.lichess.org/
"""
import requests
from pathlib import Path

# Можно взять любой месяц из списка на https://database.lichess.org/
# Для "первичных данных" лучше взять дамп поменьше (старые годы — меньше партий)
YEAR_MONTH = "2013-01"  # ранние месяцы весят в разы меньше свежих
URL = f"https://database.lichess.org/standard/lichess_db_standard_rated_{YEAR_MONTH}.pgn.zst"

OUT_DIR = Path("data/raw")
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / f"lichess_{YEAR_MONTH}.pgn.zst"


def download(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{done / total:.1%}", end="")
    print(f"\nСохранено: {out_path} ({out_path.stat().st_size / 1e6:.1f} МБ)")


if __name__ == "__main__":
    download(URL, OUT_PATH)
