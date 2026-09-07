#!/usr/bin/env bash
# AE-Primate Multi-Format Build Script
# Compiles both camera-ready (paper/main.pdf) and double-blind anonymous (paper/main_anonymous.pdf).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

TECTONIC_BIN="/opt/homebrew/bin/tectonic"
if ! command -v "$TECTONIC_BIN" &> /dev/null; then
    TECTONIC_BIN="tectonic"
fi

echo "============================================================"
echo "AE-Primate: Compiling Publication Manuscripts"
echo "============================================================"

# 1. Camera-Ready IEEE / CVPR 2-Column Manuscript
echo "[1/2] Compiling main.tex (Camera-Ready)..."
"$TECTONIC_BIN" main.tex
cp main.pdf ../main.pdf
PAGES_MAIN=$(pdfinfo main.pdf | grep "Pages:" | awk '{print $2}')
echo "  -> Built: paper/main.pdf (${PAGES_MAIN} pages)"
if [ "$PAGES_MAIN" -ne 10 ]; then
    echo "ERROR: main.pdf page count is $PAGES_MAIN (expected strictly 10 pages)!"
    exit 1
fi

# 2. Double-Blind Anonymous Manuscript
echo "[2/2] Compiling main_anonymous.tex (Double-Blind Anonymous)..."
cp main.tex main_anonymous.tex
sed -i '' 's/\\anonymousfalse/\\anonymoustrue/' main_anonymous.tex
"$TECTONIC_BIN" main_anonymous.tex
PAGES_ANON=$(pdfinfo main_anonymous.pdf | grep "Pages:" | awk '{print $2}')
echo "  -> Built: paper/main_anonymous.pdf (${PAGES_ANON} pages)"
if [ "$PAGES_ANON" -ne 10 ]; then
    echo "ERROR: main_anonymous.pdf page count is $PAGES_ANON (expected strictly 10 pages)!"
    exit 1
fi

echo "============================================================"
echo "SUCCESS: Both camera-ready and double-blind manuscripts compiled cleanly (10 pages each)!"
echo "============================================================"
