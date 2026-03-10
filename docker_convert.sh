#!/usr/bin/env bash
set -euo pipefail

# docker_convert.sh
# 在容器内将 PPTX 转为 PDF 并放到 outputs/ 下

PPT="${1:-}"
OUTDIR="outputs"

cd "$(dirname "$0")" || exit 1

if [ -z "$PPT" ]; then
  mapfile -t matches < <(find . -maxdepth 1 -type f -name '*.pptx' | sort)
  if [ "${#matches[@]}" -eq 1 ]; then
    PPT="${matches[0]#./}"
  else
    echo "Error: expected exactly one PPTX in $(pwd), or pass the file name as the first argument"
    exit 2
  fi
fi

if [ ! -f "$PPT" ]; then
  echo "Error: $PPT not found in $(pwd)"
  exit 3
fi

mkdir -p "$OUTDIR"

echo "Converting $PPT -> PDF (headless LibreOffice)..."
libreoffice --headless --convert-to pdf --outdir "$OUTDIR" "$PPT"

OUTPDF="$OUTDIR/${PPT%.pptx}.pdf"
if [ -f "$OUTPDF" ]; then
  echo "Conversion successful: $OUTPDF"
else
  echo "Conversion failed"
  exit 4
fi
