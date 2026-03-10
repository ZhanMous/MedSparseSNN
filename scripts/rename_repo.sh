#!/usr/bin/env bash
set -euo pipefail

CURRENT_NAME="HemoSparse"
TARGET_NAME="MedSparseSNN"
CURRENT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
PARENT_DIR="$(dirname "$CURRENT_DIR")"
CURRENT_BASENAME="$(basename "$CURRENT_DIR")"

if [[ "$CURRENT_BASENAME" != "$CURRENT_NAME" ]]; then
  echo "当前目录名为 $CURRENT_BASENAME，不是 $CURRENT_NAME；如需继续，请手动检查脚本。"
  exit 1
fi

if [[ -e "$PARENT_DIR/$TARGET_NAME" ]]; then
  echo "目标目录已存在：$PARENT_DIR/$TARGET_NAME"
  exit 1
fi

echo "即将把仓库目录从 $CURRENT_DIR 重命名为 $PARENT_DIR/$TARGET_NAME"
echo "请先关闭当前 VS Code 工作区或在新的 shell 会话中执行此脚本。"
mv "$CURRENT_DIR" "$PARENT_DIR/$TARGET_NAME"
echo "完成。新的仓库目录：$PARENT_DIR/$TARGET_NAME"