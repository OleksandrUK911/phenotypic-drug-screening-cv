"""Download BBBC021 metadata and (a subset of) plate image archives.

BBBC021 publishes per-plate ZIP archives and CSV metadata directly from its
download page rather than a stable, enumerable URL scheme, so this module
scrapes the page for the current links instead of hardcoding filenames that
would silently go stale. See TODO.md section 19 re: deliberately scoping
down to a small plate subset instead of the full multi-terabyte dataset.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

BBBC021_PAGE = "https://bbbc.broadinstitute.org/BBBC021"
CHUNK_SIZE = 1 << 20

logger = logging.getLogger(__name__)


def list_remote_files() -> dict[str, list[str]]:
    """Return {'csv': [...], 'zip': [...]} absolute URLs found on the BBBC021 page."""
    resp = requests.get(BBBC021_PAGE, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls: dict[str, list[str]] = {"csv": [], "zip": []}
    for a in soup.find_all("a", href=True):
        href = urljoin(BBBC021_PAGE, a["href"])
        if href.lower().endswith(".csv"):
            urls["csv"].append(href)
        elif href.lower().endswith(".zip"):
            urls["zip"].append(href)
    return urls


def _download_file(url: str, dest: Path, sha256: str | None = None) -> Path:
    if dest.exists() and sha256 is None:
        logger.info("Skipping existing file: %s", dest)
        return dest

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with requests.get(url, stream=True, timeout=60) as resp:
        resp.raise_for_status()
        digest = hashlib.sha256()
        with tmp.open("wb") as f:
            for chunk in resp.iter_content(CHUNK_SIZE):
                f.write(chunk)
                digest.update(chunk)

    if sha256 is not None and digest.hexdigest() != sha256:
        tmp.unlink(missing_ok=True)
        raise ValueError(f"Checksum mismatch for {url}: expected {sha256}, got {digest.hexdigest()}")

    tmp.rename(dest)
    return dest


def download_metadata(dest_dir: Path) -> list[Path]:
    urls = list_remote_files()
    return [_download_file(url, dest_dir / Path(url).name) for url in urls["csv"]]


def download_plate_subset(dest_dir: Path, n_plates: int) -> list[Path]:
    """Download only the first `n_plates` plate archives — see module docstring on scoping."""
    urls = list_remote_files()
    selected = sorted(urls["zip"])[:n_plates]
    if not selected:
        raise RuntimeError("No plate ZIP archives found on BBBC021 page — site layout may have changed.")
    return [_download_file(url, dest_dir / Path(url).name) for url in selected]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--n-plates",
        type=int,
        default=2,
        help="Number of plate ZIP archives to download (full dataset is multi-terabyte; default keeps this tractable).",
    )
    parser.add_argument("--metadata-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    metadata_paths = download_metadata(args.dest)
    logger.info("Downloaded metadata: %s", metadata_paths)

    if not args.metadata_only:
        plate_paths = download_plate_subset(args.dest, args.n_plates)
        logger.info("Downloaded %d plate archive(s): %s", len(plate_paths), plate_paths)


if __name__ == "__main__":
    main()
