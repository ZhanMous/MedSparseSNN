#!/usr/bin/env bash
# Build FINAL_RESEARCH_PAPER.md to PDF using pandoc + xelatex
set -euo pipefail

MD_FILE="FINAL_RESEARCH_PAPER.md"
OUT_DIR="outputs"
OUT_PDF="$OUT_DIR/FINAL_RESEARCH_PAPER.pdf"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc 未安装。请先安装 pandoc（例如: sudo apt install pandoc）并确保已安装 TeX 发行版（xelatex）。"
  exit 1
fi

mkdir -p "$OUT_DIR"

echo "开始使用 Pandoc + xelatex 生成 PDF..."
pandoc "$MD_FILE" \
  -o "$OUT_PDF" \
  --pdf-engine=xelatex \
  -V geometry:margin=1in \
  -V fontsize=11pt \
  --toc \
  --number-sections

echo "生成完成： $OUT_PDF"
echo "提示：如果需要自定义模板或引用 BibTeX，可修改本脚本以传递 --template 或 --bibliography 参数。"
