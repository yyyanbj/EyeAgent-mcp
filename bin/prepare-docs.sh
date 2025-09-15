#!/usr/bin/env bash
# Aggregate per-package docs folders into root-level _docs_build for MkDocs.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${ROOT_DIR}/_docs_build"
ROOT_DOCS="${ROOT_DIR}/docs"

rm -rf "${BUILD_DIR}"
mkdir -p "${BUILD_DIR}"

if [ -d "${ROOT_DOCS}" ]; then
  rsync -a "${ROOT_DOCS}/" "${BUILD_DIR}/"
fi

# 遍历第一层子目录，聚合其中的 docs 子目录
for pkg_docs in "${ROOT_DIR}"/*/docs ; do
  [ -d "${pkg_docs}" ] || continue
  pkg_dir="$(basename "$(dirname "${pkg_docs}")")"
  # 排除根 docs 目录本身（通常不会匹配，如果用户命名冲突则跳过）
  if [ "${pkg_dir}" = "docs" ]; then
    continue
  fi
  echo "[prepare-docs] collecting ${pkg_dir}/docs"
  mkdir -p "${BUILD_DIR}/${pkg_dir}"
  rsync -a "${pkg_docs}/" "${BUILD_DIR}/${pkg_dir}/"
done

echo "[prepare-docs] aggregation complete -> ${BUILD_DIR}"
