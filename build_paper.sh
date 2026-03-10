#!/usr/bin/env bash
# Build FINAL_RESEARCH_PAPER.md to PDF using pandoc + xelatex
set -euo pipefail

MD_FILE="${1:-FINAL_RESEARCH_PAPER.md}"
DEFAULT_NAME="$(basename "$MD_FILE" .md)"
DEFAULT_OUT_DIR="outputs/paper-build"
OUT_PDF="${2:-$DEFAULT_OUT_DIR/${DEFAULT_NAME}.pdf}"
OUT_DIR="$(dirname "$OUT_PDF")"
OUT_TEX="${OUT_PDF%.pdf}.tex"

LANG_CODE="$(python - <<'PY' "$MD_FILE"
from pathlib import Path
import sys

path = Path(sys.argv[1])
lang = ""
text = path.read_text(encoding='utf-8')
lines = text.splitlines()
if lines and lines[0].strip() == "---":
    for line in lines[1:]:
        stripped = line.strip()
        if stripped in {"---", "..."}:
            break
        if stripped.startswith("lang:"):
            lang = stripped.split(":", 1)[1].strip().strip('"\'')
            break
print(lang)
PY
)"

if [[ "$LANG_CODE" == zh* || "$LANG_CODE" == cn* ]]; then
  TEMPLATE_FILE="paper/pandoc-cn-journal-template.tex"
  TEMPLATE_LABEL="中文期刊模板"
else
  TEMPLATE_FILE="paper/pandoc-conference-template.tex"
  TEMPLATE_LABEL="NeurIPS 风格模板"
fi

if ! command -v pandoc >/dev/null 2>&1; then
  echo "pandoc 未安装。请先安装 pandoc（例如: sudo apt install pandoc）并确保已安装 TeX 发行版（xelatex）。"
  exit 1
fi

if ! command -v xelatex >/dev/null 2>&1; then
  echo "xelatex 未安装。请先安装 TeX 发行版（例如: sudo apt install texlive-xetex texlive-lang-chinese）。"
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

echo "开始生成 LaTeX 源文件并编译 PDF（${TEMPLATE_LABEL}）..."
pandoc "$MD_FILE" \
  -o "$OUT_TEX" \
  -t latex \
  --standalone \
  --template="$TEMPLATE_FILE" \
  --resource-path=. \
  --number-sections \
  --top-level-division=section \
  -V colorlinks=false \
  -V linkcolor=black \
  --number-sections

python - <<'PY' "$OUT_TEX" "$LANG_CODE"
from pathlib import Path
import sys

tex_path = Path(sys.argv[1])
lang_code = sys.argv[2].lower()
content = tex_path.read_text(encoding='utf-8')

reference_anchors = [
  "\\addcontentsline{toc}{section}{参考文献}",
  "\\addcontentsline{toc}{section}{References}",
]
closing_anchors = [
  "\\section*{伦理声明}",
  "\\section*{致谢}",
  "\\section*{Ethics Statement}",
  "\\section*{Acknowledgements}",
  "\\section*{Acknowledgments}",
]

for reference_anchor in reference_anchors:
  if reference_anchor in content and "\\everypar{\\hangindent=1.5em\\hangafter=1}" not in content:
    reference_format = (
      f"{reference_anchor}\n"
      "\\begingroup\n"
      "\\setlength{\\parindent}{0pt}\n"
      "\\setlength{\\parskip}{0.3em}\n"
      "\\everypar{\\hangindent=1.5em\\hangafter=1}"
    )
    content = content.replace(reference_anchor, reference_format, 1)
    break

if "\\endgroup" not in content:
  for closing_anchor in closing_anchors:
    if closing_anchor in content:
      content = content.replace(closing_anchor, "\\endgroup\n\n" + closing_anchor, 1)
      break

if lang_code.startswith("zh") or lang_code.startswith("cn"):
  content = content.replace("\\begin{table*}", "\\begin{table}")
  content = content.replace("\\end{table*}", "\\end{table}")

tex_path.write_text(content, encoding='utf-8')
PY

xelatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" "$OUT_TEX" >/dev/null
xelatex -interaction=nonstopmode -halt-on-error -output-directory="$OUT_DIR" "$OUT_TEX" >/dev/null

echo "生成完成： $OUT_PDF"
echo "提示：当前输出采用 ${TEMPLATE_LABEL}；如需继续细调，可针对对应模板进一步优化。"
