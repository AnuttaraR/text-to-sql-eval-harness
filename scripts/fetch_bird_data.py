"""
Downloads the BIRD Mini-Dev dataset (500 SQLite instances, CC BY-SA 4.0) from the
official Aliyun OSS distribution linked in https://github.com/bird-bench/mini_dev,
and extracts it into data/minidev/MINIDEV/ (that nested casing is how the official
zip is laid out - not renamed here, so paths match the upstream documentation).

The HuggingFace dataset (birdsql/bird_mini_dev) only ships the question/gold-SQL
JSON, not the actual per-database SQLite files needed to execute queries - so this
script uses the official zip instead, which contains both.

Usage: python scripts/fetch_bird_data.py
"""
import os
import sys
import zipfile

import requests
from tqdm import tqdm

DATASET_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/minidev.zip"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
ZIP_PATH = os.path.join(DATA_DIR, "minidev.zip")
EXTRACTED_DIR = os.path.join(DATA_DIR, "minidev", "MINIDEV")
EXTRACT_MARKER = os.path.join(DATA_DIR, ".fetched")


def download(url: str, dest: str, max_retries: int = 15) -> None:
    """
    Resumable download via HTTP Range requests - the OSS host is a China-region
    bucket and connections from here drop/stall every 1-3MB, so a single
    unresumed GET never completes. Each retry picks up from the current file size.
    """
    head = requests.head(url, timeout=30)
    total = int(head.headers.get("content-length", 0))

    attempt = 0
    while True:
        existing = os.path.getsize(dest) if os.path.exists(dest) else 0
        if total and existing >= total:
            return

        attempt += 1
        if attempt > max_retries:
            raise RuntimeError(
                f"Gave up after {max_retries} retries at {existing}/{total} bytes."
            )

        headers = {"Range": f"bytes={existing}-"} if existing else {}
        mode = "ab" if existing else "wb"
        try:
            resp = requests.get(url, headers=headers, stream=True, timeout=(10, 20))
            resp.raise_for_status()
            with open(dest, mode) as f, tqdm(
                total=total, initial=existing, unit="B", unit_scale=True,
                desc=f"Downloading BIRD Mini-Dev (attempt {attempt})",
            ) as bar:
                for chunk in resp.iter_content(chunk_size=256 * 1024):
                    if chunk:
                        f.write(chunk)
                        bar.update(len(chunk))
        except (requests.exceptions.RequestException, ConnectionError) as e:
            print(f"\nAttempt {attempt} interrupted ({e}); resuming from "
                  f"{os.path.getsize(dest) if os.path.exists(dest) else 0} bytes...")
            continue


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)

    if os.path.exists(EXTRACT_MARKER):
        print(f"Already fetched - see {EXTRACTED_DIR}")
        print("Delete data/.fetched to force a re-download.")
        return

    if not os.path.exists(ZIP_PATH):
        print(f"Downloading {DATASET_URL} (~800MB)...")
        download(DATASET_URL, ZIP_PATH)
    else:
        print("Zip already downloaded, skipping fetch.")

    print("Extracting...")
    with zipfile.ZipFile(ZIP_PATH) as zf:
        zf.extractall(DATA_DIR)

    with open(EXTRACT_MARKER, "w") as f:
        f.write("ok\n")

    if os.path.exists(ZIP_PATH):
        os.remove(ZIP_PATH)
        print("Removed minidev.zip after extraction (dataset itself is ~3.3GB).")

    print(f"Done. Dataset extracted under {EXTRACTED_DIR}")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.RequestException as e:
        print(f"Download failed: {e}", file=sys.stderr)
        sys.exit(1)
