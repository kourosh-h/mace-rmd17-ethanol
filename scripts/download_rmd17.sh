#!/usr/bin/env bash
# Download the revised MD17 dataset (Christensen and von Lilienfeld, 2020; CC0) from the Materials Cloud
# Archive, check its md5 sum and extract the ethanol data and the index files of the splits.
#
# The archive (1.0 GB) holds all ten molecules; only rmd17_ethanol.npz (about 20 MB) and splits/ are kept.
# Output: work/download/rmd17/npz_data/rmd17_ethanol.npz, work/download/rmd17/splits/index_*.csv
#
# Run:  bash scripts/download_rmd17.sh
set -euo pipefail
cd "$(dirname "$0")/.."

RECORD="https://archive.materialscloud.org/records/pfffs-fff86/files"
MD5_ARCHIVE="cb1a927628d96f2e966025da4fb63d18"     # md5 published on the record page
DOWNLOAD_DIR="work/download"

mkdir -p "$DOWNLOAD_DIR"
cd "$DOWNLOAD_DIR"

if [ ! -f rmd17.tar.bz2 ]; then
    echo "downloading rmd17.tar.bz2 (1.0 GB)"
    curl -L --fail -o rmd17.tar.bz2 "$RECORD/rmd17.tar.bz2?download=1"
    curl -L --fail -o readme.txt "$RECORD/readme.txt?download=1"
fi

# md5 on macOS, md5sum on Linux
if command -v md5 > /dev/null; then
    checksum=$(md5 -q rmd17.tar.bz2)
else
    checksum=$(md5sum rmd17.tar.bz2 | cut -d ' ' -f 1)
fi
if [ "$checksum" != "$MD5_ARCHIVE" ]; then
    echo "md5 sum does not match: $checksum (expected $MD5_ARCHIVE)"
    exit 1
fi
echo "md5 sum checked"

tar -xjf rmd17.tar.bz2 rmd17/npz_data/rmd17_ethanol.npz rmd17/splits
ls -l rmd17/npz_data rmd17/splits
