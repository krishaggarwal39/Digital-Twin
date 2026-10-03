"""Download and extract the ShanghaiT1DM / ShanghaiT2DM real-world dataset.

Source (open for research use):
  Zhao et al., "Chinese diabetes datasets for data-driven machine learning",
  Scientific Data 10, 35 (2023). https://doi.org/10.1038/s41597-023-01940-7
  figshare article 20444397.

Usage:
  python scripts/download_shanghai.py

Produces:
  data/real_shanghai/Shanghai_T2DM/*.xls[x]      (109 patient CGM files)
  data/real_shanghai/Shanghai_T1DM/*.xls[x]      (19 patient CGM files)
  data/real_shanghai/Shanghai_T2DM_Summary.xlsx  (static EHR)
  data/real_shanghai/Shanghai_T1DM_Summary.xlsx
"""

import io
import os
import re
import ssl
import sys
import urllib.request
import zipfile

FIGSHARE_URL = "https://ndownloader.figshare.com/files/38259264"
OUT_DIR = os.path.join("data", "real_shanghai")


def _urlopen(url: str, timeout: int = 120):
    """Open a URL with a certifi CA bundle when available (robust on macOS)."""
    try:
        import certifi  # usually present via pip's dependencies
        ctx = ssl.create_default_context(cafile=certifi.where())
    except Exception:  # noqa: BLE001
        ctx = ssl.create_default_context()
    return urllib.request.urlopen(url, timeout=timeout, context=ctx)


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"Downloading ShanghaiT1DM/T2DM dataset (~3.4 MB) from figshare ...")
    try:
        with _urlopen(FIGSHARE_URL, timeout=120) as resp:
            blob = resp.read()
    except Exception as e:  # noqa: BLE001
        print(f"ERROR: download failed: {e}")
        print("Manually download data.zip from https://doi.org/10.6084/m9.figshare.20444397 "
              f"and extract its Shanghai_* folders into {OUT_DIR}/")
        sys.exit(1)

    z = zipfile.ZipFile(io.BytesIO(blob))
    extracted = 0
    for n in z.namelist():
        if n.endswith("/"):
            continue
        m = re.search(r"(Shanghai_T[12]DM(?:_Summary)?.*)", n)
        if not m:
            continue
        dest = os.path.join(OUT_DIR, m.group(1))
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        with z.open(n) as src, open(dest, "wb") as out:
            out.write(src.read())
        extracted += 1

    t2 = os.path.join(OUT_DIR, "Shanghai_T2DM")
    n_t2 = len(os.listdir(t2)) if os.path.isdir(t2) else 0
    print(f"Extracted {extracted} files. ShanghaiT2DM patient files: {n_t2}")
    print(f"Done. Next: python -m src.validation.real_data_adapter")


if __name__ == "__main__":
    main()
