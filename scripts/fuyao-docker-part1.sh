#!/bin/bash
# Part1 构建：必须在项目根目录打包，否则 COPY setup.py/conftopo 会失败（context 仅 2B）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# fuyao 需要 git upstream（无公司 remote 时用本地 bare 仓库）
if ! git rev-parse '@{u}' >/dev/null 2>&1; then
  ORIGIN_DIR="${CONFTOPO_GIT_ORIGIN:-/tmp/conftopo-fuyao-origin.git}"
  if [ ! -d "$ORIGIN_DIR" ]; then
    git init --bare "$ORIGIN_DIR"
  fi
  if ! git remote get-url origin >/dev/null 2>&1; then
    git remote add origin "$ORIGIN_DIR"
  fi
  git push -u origin "$(git branch --show-current)" 2>/dev/null || true
fi

DOCKERFILE="${1:-docker/Dockerfile.part1}"
SITE="${FUYAO_SITE:-fuyao_hk}"

echo "Build context: $ROOT"
echo "Dockerfile:    $DOCKERFILE"
echo "Site:          $SITE"
echo ""

# fuyao 危险目录提示从 TTY 读入，管道 echo y 无效，用 expect 自动确认
expect <<EOF
set timeout -1
spawn fuyao docker --push --dockerfile=$DOCKERFILE --site=$SITE
expect {
  "Do you want to continue?" {
    send "y\r"
    exp_continue
  }
  eof
}
EOF
