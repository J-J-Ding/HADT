#!/usr/bin/env bash
# Build the supplied ACM manuscript with a pinned, user-local TeX engine.
set -euo pipefail

PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
TECTONIC_BIN="${TECTONIC_BIN:-$PROJECT_ROOT/.tools/tectonic/tectonic}"
INPUT="${1:-$PROJECT_ROOT/HADT_NQ_ICCRD2027_LaTeX/main.tex}"
if [[ ! -x "$TECTONIC_BIN" ]]; then
  echo "Tectonic is missing: $TECTONIC_BIN" >&2
  echo "See results/BUILD_ENVIRONMENT.md for the pinned local installation command." >&2
  exit 1
fi
INPUT="$(realpath -- "$INPUT")"
OUTPUT_DIR="${2:-$(dirname -- "$INPUT")}"
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(realpath -- "$OUTPUT_DIR")"
OPTIONS=(--bundle https://relay.fullyjustified.net/default_bundle_v33.tar --keep-logs --keep-intermediates --outdir "$OUTPUT_DIR")
if [[ "${TECTONIC_OFFLINE:-0}" == 1 ]]; then
  OPTIONS+=(--only-cached)
fi
cd -- "$(dirname -- "$INPUT")"
"$TECTONIC_BIN" "${OPTIONS[@]}" "$(basename -- "$INPUT")"
printf 'PDF: %s/%s.pdf\n' "$OUTPUT_DIR" "$(basename -- "$INPUT" .tex)"
