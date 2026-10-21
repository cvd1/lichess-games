"""
Скачивает месячные дампы партий Lichess (сырые данные) для курсового проекта.
Источник: https://database.lichess.org/

Примеры:
    python scripts/download_data.py                 # 2013-01 (маленький, ~18 МБ)
    python scripts/download_data.py 2013-01 2014-01 # несколько месяцев

Файл сначала пишется во временный .part и переименовывается только после
полной загрузки, поэтому оборванная загрузка не оставит «готовый» битый файл.
Уже скачанные месяцы пропускаются.
"""

import argparse
from pathlib import Path

import requests

BASE_URL = "https://database.lichess.org/standard"
OUT_DIR = Path("data/raw")


def download(url: str, out_path: Path, chunk_size: int = 1024 * 1024) -> None:
    part = out_path.with_name(out_path.name + ".part")
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        done = 0
        with open(part, "wb") as f:
            for chunk in r.iter_content(chunk_size=chunk_size):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r{out_path.name}: {done / total:.1%}", end="")
    if total and done != total:
        raise OSError(f"Загрузка неполная: получено {done} из {total} байт ({part})")
    part.replace(out_path)
    print(f"\nСохранено: {out_path} ({out_path.stat().st_size / 1e6:.1f} МБ)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "months",
        nargs="*",
        default=["2013-01"],
        help="месяцы в формате ГГГГ-ММ (по умолчанию 2013-01)",
    )
    ap.add_argument("--base-url", default=BASE_URL, help=argparse.SUPPRESS)
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for month in args.months:
        out_path = OUT_DIR / f"lichess_{month}.pgn.zst"
        if out_path.exists():
            print(f"Пропускаю {month}: {out_path} уже есть")
            continue
        url = f"{args.base_url}/lichess_db_standard_rated_{month}.pgn.zst"
        download(url, out_path)


if __name__ == "__main__":
    main()
