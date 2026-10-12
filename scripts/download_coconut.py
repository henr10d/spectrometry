"""Download COCONUT CSV Lite (CC0) for candidate pool expansion."""
import argparse
import zipfile
from pathlib import Path
import urllib.request

LITE_URL = "https://coconut.s3.uni-jena.de/prod/downloads/2026-10/coconut_csv_lite-10-2026.zip"
ZIP_NAME = "coconut_csv_lite-10-2026.zip"
CSV_NAME = "coconut_csv_lite-10-2026.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="external")
    ap.add_argument("--url", default=LITE_URL)
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    zpath = out / ZIP_NAME
    csv_path = out / CSV_NAME
    if csv_path.is_file():
        print("Already extracted:", csv_path)
        return
    if not zpath.is_file():
        print("Downloading", args.url, "->", zpath)
        urllib.request.urlretrieve(args.url, zpath)
    with zipfile.ZipFile(zpath) as zf:
        names = [n for n in zf.namelist() if n.endswith(".csv")]
        if not names:
            raise SystemExit("No CSV in zip")
        target = names[0]
        print("Extracting", target, "->", csv_path)
        with zf.open(target) as src, csv_path.open("wb") as dst:
            dst.write(src.read())
    print("Ready:", csv_path, csv_path.stat().st_size // 1_000_000, "MB")


if __name__ == "__main__":
    main()
