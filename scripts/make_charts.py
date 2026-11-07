"""Generates the SVG charts used by README.md / docs (no dependencies).

    python scripts/make_charts.py

Numbers are the measurements quoted in docs/RESEARCH.md, docs/LONG_CONTEXT.md
and docs/UPSCALE.md; edit DATA below and re-run after new benchmarks.
"""
import os

OUT = os.path.join(os.path.dirname(__file__), "..", "docs", "img")
os.makedirs(OUT, exist_ok=True)

STYLE = """<style>
 text{font-family:Segoe UI,Helvetica,Arial,sans-serif;fill:#24292f}
 .t{font-size:17px;font-weight:600}.s{font-size:12px;fill:#57606a}.l{font-size:13px}.v{font-size:12px;font-weight:600}
 @media (prefers-color-scheme:dark){text{fill:#e6edf3}.s{fill:#8b949e}}
</style>"""


def hbar(name, title, subtitle, rows, unit, maxv=None, colors=None, width=760):
    """rows: [(label, value, note)] -> horizontal bar chart."""
    maxv = maxv or max(r[1] for r in rows) * 1.15
    left, top, bh, gap = 230, 64, 26, 12
    h = top + len(rows) * (bh + gap) + 24
    w_plot = width - left - 120
    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{h}" viewBox="0 0 {width} {h}">', STYLE,
           f'<text x="16" y="28" class="t">{title}</text><text x="16" y="48" class="s">{subtitle}</text>']
    for i, (label, v, note) in enumerate(rows):
        y = top + i * (bh + gap)
        c = (colors or {}).get(label, "#2f81f7")
        bw = max(2, w_plot * v / maxv)
        out.append(f'<text x="{left - 10}" y="{y + bh * 0.68}" text-anchor="end" class="l">{label}</text>')
        out.append(f'<rect x="{left}" y="{y}" width="{bw:.1f}" height="{bh}" rx="4" fill="{c}"/>')
        out.append(f'<text x="{left + bw + 8}" y="{y + bh * 0.68}" class="v">{v:g} {unit}{(" · " + note) if note else ""}</text>')
    out.append("</svg>")
    open(os.path.join(OUT, name), "w", encoding="utf-8").write("\n".join(out))


GREY, BLUE, GREEN, ORANGE, RED = "#8c959f", "#2f81f7", "#2da44e", "#d4a72c", "#cf222e"

# KV cache memory for a 256K-token context of an 8B GQA model (36 layers x 8 KV heads x 128).
per_tok = {"f32": 36 * 2 * 1024 * 4, "f16": 36 * 2 * 1024 * 2,
           "Q8 (AIDO)": 36 * (2 * (1024 + 64) + 2 * 1024 * 4 // 64),
           "Q4 (AIDO, K8+V4)": 36 * ((1024 + 64) + (512 + 64) + 2 * 1024 * 4 // 64)}
hbar("kv_memory.svg", "KV-кэш для 256K токенов (8B-модель, GQA 8×128, 36 слоёв)",
     "ГБ оперативной памяти; на машине 32 ГБ, из них ~5 ГБ занимают веса Q4_K_M",
     [(k, round(v * 262144 / 1e9, 1), "") for k, v in per_tok.items()], "ГБ",
     colors={"f32": RED, "f16": ORANGE, "Q8 (AIDO)": BLUE, "Q4 (AIDO, K8+V4)": GREEN})

# Positions read per generated token at long context (sparse attention budget 8192).
hbar("kv_reads.svg", "Позиций KV, читаемых на каждый токен генерации",
     "полное внимание растёт с контекстом; блочно-разреженное — ограничено бюджетом (8192)",
     [("8K, полное", 8192, ""), ("32K, полное", 32768, ""), ("128K, полное", 131072, ""), ("256K, полное", 262144, ""),
      ("128K, AIDO sparse", 8192, "×16 меньше"), ("256K, AIDO sparse", 8192, "×32 меньше")], "",
     colors={"128K, AIDO sparse": GREEN, "256K, AIDO sparse": GREEN, "8K, полное": GREY, "32K, полное": ORANGE,
             "128K, полное": RED, "256K, полное": RED})

# LLM decode vs memory roofline (docs/RESEARCH.md §2).
hbar("decode_roofline.svg", "Генерация на CPU против теоретического предела памяти",
     "ток/с; предел = пропускная способность RAM (13.9 ГБ/с) / байт весов на токен",
     [("Qwen3-0.6B Q8 · предел", 22, ""), ("Qwen3-0.6B Q8 · AIDO", 18.5, "85 %"),
      ("Llama-3.2-1B Q4_K_M · предел", 17, ""), ("Llama-3.2-1B Q4_K_M · AIDO", 15.6, "90 %"),
      ("Qwen3.6-35B-A3B · предел", 5.3, ""), ("Qwen3.6-35B-A3B · AIDO", 3.7, "CPU")], "ток/с",
     colors={k: (GREY if "предел" in k else BLUE) for k in ["Qwen3-0.6B Q8 · предел", "Llama-3.2-1B Q4_K_M · предел", "Qwen3.6-35B-A3B · предел"]})

# Whole-model GPU decoding (src/gpu/llama.rs) vs CPU, same prompt, greedy, 128 tokens.
hbar("gpu_decode.svg", "Генерация: весь граф модели на GPU против CPU",
     "ток/с, RTX 3070 (Vulkan) против Xeon E5-2670 v3 c одноканальной DDR4; один и тот же текст на выходе",
     [("Qwen2.5-0.5B Q4_0 · CPU", 20.2, ""), ("Qwen2.5-0.5B Q4_0 · GPU", 165.1, "×8.2"),
      ("Qwen3-0.6B Q8_0 · CPU", 13.7, ""), ("Qwen3-0.6B Q8_0 · GPU", 146.7, "×10.7"),
      ("Llama-3.2-1B Q4_K_M · CPU", 11.7, ""), ("Llama-3.2-1B Q4_K_M · GPU", 118.3, "×10.1")], "ток/с",
     colors={k: (GREY if "CPU" in k else GREEN) for k in ["Qwen2.5-0.5B Q4_0 · CPU", "Qwen3-0.6B Q8_0 · CPU", "Llama-3.2-1B Q4_K_M · CPU"]} | {k: GREEN for k in ["Qwen2.5-0.5B Q4_0 · GPU", "Qwen3-0.6B Q8_0 · GPU", "Llama-3.2-1B Q4_K_M · GPU"]})

# Upscaler quality (docs/UPSCALE.md).
hbar("upscale_psnr.svg", "Апскейл ×2 аниме (Baikal LoopSR) — качество",
     "PSNR, дБ, относительно оригинала 128×128 (выше — лучше)",
     [("бикубик", 35.5, ""), ("LoopSR, 1 цикл", 46.4, ""), ("LoopSR, 2 цикла", 48.1, ""), ("LoopSR, 4 цикла", 48.7, "")], "дБ",
     maxv=55, colors={"бикубик": GREY})

# API test results (tests/api_test.py).
hbar("api_tests.svg", "Сквозные тесты API (tests/api_test.py)",
     "число проверок по группам — 49 из 49 проходят",
     [("LLM: chat/completions/SSE/JSON", 13, ""), ("Инструменты, размышления, n", 6, ""), ("Кэш токенов, память, суммаризация", 4, ""),
      ("Decision: текст/фото/аудио", 11, ""), ("Upscale: 5 форматов", 6, ""), ("Diffusion: форматы, SSE", 5, ""),
      ("Своя SDK-модель /v1/predict", 3, ""), ("Реестр моделей", 1, "")], "",
     colors={})
print("charts written to", os.path.abspath(OUT))
