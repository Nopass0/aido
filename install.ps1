# AIDO one-line installer for Windows (PowerShell 5.1+):
#   irm https://raw.githubusercontent.com/Nopass0/aido/main/install.ps1 | iex
#
# 1. installs Rust (rustup, stable, MSVC toolchain) if `cargo` is missing;
# 2. builds AIDO from GitHub optimised for THIS CPU (-C target-cpu=native: AVX2/FMA/AVX-512 kernels);
# 3. the binary lands in %USERPROFILE%\.cargo\bin, which rustup puts on PATH,
#    and `aido install` additionally registers it for cmd / PowerShell / Terminal.
$ErrorActionPreference = 'Stop'
$repo = if ($env:AIDO_REPO) { $env:AIDO_REPO } else { 'https://github.com/Nopass0/aido' }

function Have($cmd) { $null -ne (Get-Command $cmd -ErrorAction SilentlyContinue) }

if (-not (Have 'cargo')) {
    Write-Host '[aido] Rust not found - installing rustup (stable)...' -ForegroundColor Cyan
    $init = Join-Path $env:TEMP 'rustup-init.exe'
    Invoke-WebRequest 'https://static.rust-lang.org/rustup/dist/x86_64-pc-windows-msvc/rustup-init.exe' -OutFile $init -UseBasicParsing
    & $init -y --default-toolchain stable --profile minimal | Out-Host
    $env:Path = "$env:USERPROFILE\.cargo\bin;$env:Path"
    if (-not (Have 'cargo')) { throw 'cargo still not found; open a new terminal and re-run the installer' }
}
if (-not (Have 'git')) { Write-Warning 'git not found: cargo will use its built-in git client.' }

Write-Host "[aido] building from $repo for this CPU (takes a few minutes)..." -ForegroundColor Cyan
$env:RUSTFLAGS = '-C target-cpu=native'
cargo install --git $repo --locked --force aido
if ($LASTEXITCODE -ne 0) {
    throw 'build failed. On Windows the MSVC linker is required: install "Visual Studio Build Tools" (C++ workload) and re-run.'
}
& "$env:USERPROFILE\.cargo\bin\aido.exe" install
Write-Host '[aido] done. Open a new terminal and run:  aido info' -ForegroundColor Green
