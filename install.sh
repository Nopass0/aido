#!/usr/bin/env sh
# AIDO one-line installer for Linux / macOS:
#   curl -fsSL https://raw.githubusercontent.com/Nopass0/aido/main/install.sh | sh
#
# Installs Rust if needed, builds AIDO optimised for this CPU and puts
# `aido` on PATH (~/.cargo/bin; `aido install` also links ~/.local/bin).
set -e
REPO="${AIDO_REPO:-https://github.com/Nopass0/aido}"

if ! command -v cargo >/dev/null 2>&1; then
    echo "[aido] Rust not found - installing rustup (stable)..."
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal
    . "$HOME/.cargo/env"
fi

echo "[aido] building from $REPO for this CPU (takes a few minutes)..."
RUSTFLAGS="-C target-cpu=native" cargo install --git "$REPO" --locked --force aido
"$HOME/.cargo/bin/aido" install
echo "[aido] done. Open a new shell and run:  aido info"
