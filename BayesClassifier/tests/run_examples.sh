#!/usr/bin/env bash
set -euo pipefail

root_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="${root_dir}/build"
exe="${build_dir}/naive_bayes_cli"

if [[ ! -x "${exe}" ]]; then
  echo "Expected executable not found: ${exe}" >&2
  echo "Build it with: cmake --build ${build_dir}" >&2
  exit 1
fi

"${exe}" "${root_dir}/config/classifier/inference.single.config.example.json"
"${exe}" "${root_dir}/config/classifier/inference.batch_json.example.json"
"${exe}" "${root_dir}/config/classifier/inference.batch_text.example.json"
