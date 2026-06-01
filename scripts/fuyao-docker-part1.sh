#!/bin/bash
# Part1 构建：必须在项目根目录打包，否则 COPY setup.py/conftopo 会失败（context 仅 2B）
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

DOCKERFILE="${1:-docker/Dockerfile.part1}"
SITE="${FUYAO_SITE:-fuyao_hk}"

echo "Build context: $ROOT"
echo "Dockerfile:    $DOCKERFILE"
echo "Site:          $SITE"
echo ""

printf 'y\n' | fuyao docker --push \
  --dockerfile="$DOCKERFILE" \
  --site="$SITE" \
  "${@:2}"
