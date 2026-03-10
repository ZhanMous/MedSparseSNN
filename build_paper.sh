#!/usr/bin/env bash
# Build FINAL_RESEARCH_PAPER.md to PDF using pandoc + xelatex
set -euo pipefail

MD_FILE="${1:-FINAL_RESEARCH_PAPER.md}"
OUT_DIR="outputs/paper-build"
DEFAULT_NAME="$(basename "$MD_FILE" .md)"
OUT_PDF="${2:-$OUT_DIR/${DEFAULT_NAME}.pdf}"
TEMPLATE_FILE="paper/pandoc-conference-template.tex"

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc 未安装。请先安装 pandoc（例如: sudo apt install pandoc）并确保已安装 TeX 发行版（xelatex）。"
  exit 1
fi

mkdir -p "$OUT_DIR"

if [[ ! -f "$TEMPLATE_FILE" ]]; then
  echo "缺少模板文件：$TEMPLATE_FILE"
  exit 1
fi

if [[ ! -f "$MD_FILE" ]]; then
  echo "缺少输入文件：$MD_FILE"
  exit 1
fi

echo "开始使用 Pandoc + xelatex 生成会议风格 PDF..."
pandoc "$MD_FILE" \
  -o "$OUT_PDF" \
  --standalone \
  --pdf-engine=xelatex \
  --template="$TEMPLATE_FILE" \
  --resource-path=. \
  --number-sections \
  --top-level-division=section \
  -V colorlinks=false \
  -V linkcolor=black \
  --number-sections

echo "生成完成： $OUT_PDF"
echo "提示：当前输出采用自定义会议风格模板；如需进一步接近特定顶会，可继续替换为对应官方 LaTeX class。"
