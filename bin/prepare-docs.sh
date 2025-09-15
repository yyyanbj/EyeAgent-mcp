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

# Traverse first-level subdirectories and aggregate their docs folders
for pkg_docs in "${ROOT_DIR}"/*/docs ; do
  [ -d "${pkg_docs}" ] || continue
  pkg_dir="$(basename "$(dirname "${pkg_docs}")")"
  # Skip root docs directory itself (name collision safeguard)
  if [ "${pkg_dir}" = "docs" ]; then
    continue
  fi
  echo "[prepare-docs] collecting ${pkg_dir}/docs"
  mkdir -p "${BUILD_DIR}/${pkg_dir}"
  rsync -a "${pkg_docs}/" "${BUILD_DIR}/${pkg_dir}/"
done

echo "[prepare-docs] aggregation complete -> ${BUILD_DIR}"
