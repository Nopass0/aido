# Установка

## Одной строкой

**Windows (PowerShell):**
```powershell
irm https://raw.githubusercontent.com/Nopass0/aido/main/install.ps1 | iex
```

**Linux / macOS:**
```bash
curl -fsSL https://raw.githubusercontent.com/Nopass0/aido/main/install.sh | sh
```

Скрипт:
1. ставит Rust (rustup, stable), если его нет;
2. собирает AIDO из GitHub **под ваш процессор** (`-C target-cpu=native`:
   включаются AVX2/FMA/AVX-512-ядра, которые есть именно у вас);
3. выполняет `aido install` — команда `aido` становится доступна глобально
   в cmd, PowerShell, Windows Terminal, bash, zsh.

На Windows для сборки нужен линкер MSVC: «Visual Studio Build Tools» с
нагрузкой «Разработка классических приложений на C++». GPU-ускорение
работает через Vulkan/DX12 (Windows, Linux) или Metal (macOS) — CUDA не нужна.

## Из исходников

```bash
git clone https://github.com/Nopass0/aido && cd aido
cargo build --release          # .cargo/config.toml уже включает target-cpu=native
./target/release/aido install  # добавить в PATH
```

`aido install --uninstall` убирает команду из PATH.

## Модели

```bash
aido pull hf:Qwen/Qwen3-8B-GGUF                          # скачать (выбирается Q4_K_M)
aido pull hf:bartowski/Llama-3.2-1B-Instruct-GGUF:*Q8_0* # конкретный файл по шаблону
aido run -m hf:Qwen/Qwen3-8B-GGUF -p "Привет"            # hf:-путь работает в любой команде
```

Файлы кладутся в `models/<org>--<repo>/`, докачка прерванных загрузок
поддерживается. Приватные репозитории: переменная `HF_TOKEN`.

## Проверка

```bash
aido info        # процессор, кэши, пропускная способность RAM, GPU
cargo test --release            # юнит-, интеграционные и doc-тесты
python tests/api_test.py http://127.0.0.1:8080   # при запущенном `aido serve`
```
