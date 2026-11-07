# AIDO HTTP API — полный справочник

Сервер совместим с OpenAI API (официальные клиенты `openai` для Python/JS
работают, если указать `base_url`), плюс эндпоинты для моделей решений и
апскейла. Все модели любых типов регистрируются при запуске и выбираются
полем `model`, как в OpenAI.

```bash
aido serve \
  -m qwen=models/Qwen3-0.6B-Q8_0.gguf \          # LLM (GGUF / .aido / MoE-папка)
  -m llama=hf:bartowski/Llama-3.2-1B-Instruct-GGUF \ # скачивается с Hugging Face при первом запуске
  -m d1=models/d1-omni \                         # модель решений (текст + фото + аудио)
  -m x2=models/baikal/Baikal_LoopSR_x2_v2.safetensors \  # апскейлер
  --sd models/sd-turbo --taesd models/taesd \    # генерация изображений
  --port 8080 --ram-gb 12 --keep-alive 600
```

| Флаг | Значение |
|---|---|
| `-m, --model name=path` | регистрирует модель (повторяемый). Без `name=` имя = имя файла. Тип определяется автоматически |
| `--sd DIR`, `--taesd DIR` | диффузионная модель (имя `sd`) и быстрый декодер |
| `--ram-gb N` | бюджет ОЗУ на одновременно загруженные модели; сверх него выгружаются давно не использованные (LRU) |
| `--vram-gb N` | бюджет видеопамяти на модель |
| `--keep-alive S` | выгружать модель после S секунд простоя |
| `--preload` | загрузить все модели при старте (иначе — при первом запросе) |
| `--ctx N`, `-t N`, `--device cpu\|gpu\|auto`, `--host`, `--port` | контекст LLM, потоки, устройство диффузии, адрес |

### Как несколько моделей делят слабый ПК

* **Ленивая загрузка**: модель грузится при первом обращении.
* **Общие ресурсы**: один пул потоков, привязанный к физическим ядрам, и
  одно GPU-устройство на все модели.
* **Общие веса**: веса отображаются в память (`mmap`) — ОС держит одну
  копию страниц файла для всех имён и процессов и сбрасывает чистые
  страницы без свопа под давлением памяти; один и тот же файл под двумя
  именами загружается один раз.
* **LRU в бюджете `--ram-gb`** и выгрузка по простою.
* Запросы к одной модели выполняются по очереди (два параллельных прохода
  по одним весам только делят пропускную способность памяти); к разным
  моделям — параллельно.

## Общее

* Тело запроса и ответа — JSON (`Content-Type: application/json`), лимит тела 256 МБ.
* Ошибки: HTTP 400/500 и `{"error": {"message": "...", "type": "invalid_request_error"}}`.
* Стриминг — Server-Sent Events: строки `data: {json}`, конец — `data: [DONE]`.
* Пустое или отсутствующее `model` → первая зарегистрированная модель нужного типа.
* Медиа на входе принимаются как `data` (base64), `url` (`data:`-URL или
  локальный путь) или `path`. Изображения: PNG, JPEG, WebP, BMP, GIF, TIFF,
  ICO, PNM, TGA, QOI, HDR. Аудио: WAV (PCM 8/16/24/32 бит, float, любая
  частота — ресемплинг в 16 кГц).

## `GET /health`
`{"status": "ok"}`

## `GET /v1/models`
```json
{"object": "list", "data": [
  {"id": "qwen", "object": "model", "owned_by": "aido", "kind": "llm", "path": "models/Qwen3-0.6B-Q8_0.gguf", "loaded": true, "bytes": 639446688},
  {"id": "d1", "kind": "decision", ...}, {"id": "x2", "kind": "upscale", ...}, {"id": "sd", "kind": "diffusion", ...}]}
```
`kind`: `llm` | `decision` | `upscale` | `diffusion` | `custom` (SDK-модель, плюс поле `io` с сигнатурой); `loaded` — в памяти ли сейчас.

## `POST /v1/chat/completions`

| Поле | Тип | По умолчанию | |
|---|---|---|---|
| `model` | string | первая LLM | |
| `messages` | `[{role, content}]` | — | `content` — строка или массив частей `[{"type":"text","text":...}]` |
| `max_tokens` / `max_completion_tokens` | int | 512 | |
| `temperature` | float | 0.7 | 0 = жадный выбор |
| `top_p`, `top_k` | float, int | 0.95, 40 | |
| `seed` | int | время | |
| `stop` | string \| string[] | — | |
| `frequency_penalty` | float | 0 | → штраф повторов |
| `stream` | bool | false | SSE-чанки `chat.completion.chunk` |
| `response_format` | `{"type": "json_object"}` | — | JSON-режим: модель просится ответить JSON, из ответа извлекается первый JSON-объект |
| `enable_thinking` / `reasoning_effort` | bool / string | false | режим рассуждений; текст рассуждений → `message.reasoning_content` |
| `tools`, `tool_choice` | OpenAI | — | вызов функций: ответ `message.tool_calls`, `finish_reason: "tool_calls"` ([CHAT.md](CHAT.md#инструменты-function-calling)) |
| `n` | int | 1 | число вариантов ответа (до 8) |
| `user` | string | — | id пользователя для долгосрочной памяти |
| `memory` | bool | true | использовать память для этого запроса |
| `auto_summarize` | bool | true | сворачивать старую историю, если не помещается в контекст |

Ответ:
```json
{"id": "chatcmpl-…", "object": "chat.completion", "created": 1762560000, "model": "qwen",
 "choices": [{"index": 0, "message": {"role": "assistant", "content": "Paris"}, "finish_reason": "stop"}],
 "usage": {"prompt_tokens": 18, "completion_tokens": 2, "total_tokens": 20,
   "aido": {"tokens_per_s": 13.1, "prefill_tokens_per_s": 183.3, "cached_prompt_tokens": 3, "drafted": 1, "accepted": 1}}}
```
`usage.prompt_tokens_details.cached_tokens` — сколько токенов промпта взято
из кэша префикса (не пересчитывались). `usage.aido`: скорость генерации и
prefill, статистика спекулятивного декодирования, `summarized_messages`
(свёрнуто в изложение), `memories_used` (подставлено из памяти).
Системное сообщение, история, инструменты, размышления, память — подробно
в [CHAT.md](CHAT.md).

Стриминг (`"stream": true`):
```
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":"assistant","content":""},"finish_reason":null}],...}
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":"Par"},"finish_reason":null}],...}
data: {"object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{...}}
data: [DONE]
```

```python
from openai import OpenAI
c = OpenAI(base_url="http://127.0.0.1:8080/v1", api_key="none")
for ch in c.chat.completions.create(model="qwen", messages=[{"role": "user", "content": "Привет!"}], stream=True):
    print(ch.choices[0].delta.content or "", end="", flush=True)
```

## `GET|POST|DELETE /v1/memory` — долгосрочная память

* `GET /v1/memory?user=ID` — все записи; `&query=текст&k=5` — поиск (векторы + BM25).
* `POST /v1/memory {"user", "text"}` — добавить запись → `{"id", "created", "text"}`.
* `DELETE /v1/memory {"user", "id"?}` — удалить одну или все → `{"deleted": n}`.

Каталог хранения: `aido serve --memory-dir DIR` (по умолчанию `data/memory`, `none` — выключить).

## `POST /v1/predict` — свои модели (SDK)

Для моделей с сигнатурой входа/выхода ([SDK.md](SDK.md)):
```json
{"model": "seq", "input": {"sequence": [1, 1, 2, 3, 5, 8]}}
```
→ `{"object": "prediction", "model": "seq", "output": {"next": 13}, "ms": 4.1}`.
Вход проверяется по типам сигнатуры (ошибка 400 называет поле).
`GET /v1/models` отдаёт сигнатуру custom-модели в поле `io`.

## `POST /v1/completions`
Как выше, но вместо `messages` — `prompt` (строка, без шаблона чата).
Ответ `object: "text_completion"`, текст в `choices[0].text`; при `stream`
— чанки с `choices[0].text`.

## `POST /v1/decide` — модели решений (d1-подобные)

Одним проходом модели отвечает сразу на несколько типизированных вопросов
о «состоянии» (тексте) и приложенных картинках/аудио. Ответ генерацией не
строится: читаются вероятности в слотах ответа, поэтому это быстро и
калибровано.

```json
{
  "model": "d1",
  "state": "Customer: My package is 5 days late and I want my money back NOW!",
  "questions": {
    "refund": {"type": "noul", "instructions": "Is the customer asking for a refund?"},
    "team":   {"type": "choice", "instructions": "Which team should handle this?",
               "criteria": {"billing": "Charges, refunds", "shipping": "Delivery", "technical": "App faults"}},
    "topics": {"type": "choice", "multiple": true, "instructions": "Which topics are raised?", "criteria": ["delivery", "payment", "quality"]},
    "anger":  {"type": "score", "instructions": "How angry is the customer?", "criteria": ["calm", "annoyed", "angry", "furious"]},
    "days":   {"type": "number", "instructions": "How many days late?", "range": [0, 30], "step": 1, "unit": "days"},
    "where":  {"type": "box", "instructions": "Where is the damaged item?"}
  },
  "media": [
    {"type": "image", "data": "<base64>"},
    {"type": "image", "url": "data:image/webp;base64,..."},
    {"type": "audio", "path": "C:/calls/call1.wav"}
  ]
}
```

| `type` | Ответ |
|---|---|
| `noul` | `{"type":"noul","noul":0.947}` — вероятность «да» |
| `choice` | `{"type":"choice","choice":"refund","confidence":0.95,"probabilities":{...}}`; при `multiple: true` — `{"selected":[...],"probabilities":{...}}` (независимые вероятности) |
| `score` | `{"type":"score","score":2.06,"legend":[...]}` — ожидаемый уровень по шкале `criteria` |
| `number` | `{"type":"number","number":5.0,"interval":[4,17],"unit":"days"}` — оценка и интервал |
| `box` | `{"type":"box","present":0.74,"box":{"left":0.03,"top":0.05,"right":0.98,"bottom":0.98}}` — координаты в долях кадра |

Ответ также содержит `model`, `tokens` (длина промпта) и `timing_ms`.
`state` может быть строкой или любым JSON (будет сериализован).

## `POST /v1/images/upscale`

```json
{"model": "x2", "image": "<base64 или data URL>", "loops": 2,
 "output_format": "png", "response_format": "b64_json", "quality": 92}
```

| Поле | |
|---|---|
| `image` (или `url` / `path`) | входное изображение любого поддерживаемого формата |
| `loops` | циклов общего тела LoopSR: 1 — быстрее всего, 4 — лучшее качество (по умолчанию — значение из модели) |
| `output_format` | `png` (по умолчанию) \| `jpeg` \| `webp` (без потерь) \| `bmp` \| `raw` (RGB24, `width·height·3` байт) |
| `response_format` | `b64_json` (по умолчанию) \| `url` (`data:`-URL) |
| `quality` | качество JPEG 1–100 |

Ответ:
```json
{"created": 1762560000, "model": "x2", "width": 256, "height": 256, "input": {"width": 128, "height": 128},
 "output_format": "png", "mime": "image/png", "ms": 585.1, "data": [{"b64_json": "..."}]}
```

Для видео удобнее CLI-поток без HTTP: `aido upscale --stream WxH` читает
сырые кадры RGB24 из stdin и пишет кадры ×2 в stdout (см. [UPSCALE.md](UPSCALE.md)).

## `POST /v1/images/generations`

| Поле | По умолчанию | |
|---|---|---|
| `model` | первая диффузионная | |
| `prompt` | — | |
| `negative_prompt` | `""` | используется при `guidance_scale > 1` |
| `size` | `512x512` | кратно 64 |
| `n` | 1 | до 8 |
| `steps` | 1 | SD-Turbo: 1–4 |
| `guidance_scale` | 1.0 | |
| `seed` | время | изображение `i` получает `seed + i` |
| `output_format` | `png` | `png` \| `jpeg` \| `webp` \| `bmp` |
| `response_format` | `b64_json` | или `url` (`data:`-URL) |
| `stream` | false | SSE с превью каждого шага |

Ответ: `{"created", "model", "output_format", "ms", "timings", "data": [{"b64_json"}]}`.

Со `"stream": true` после каждого шага денойзинга приходит превью —
текущая оценка чистого изображения `x₀ = x − σ·ε`, декодированная VAE:
```
data: {"type":"preview","index":0,"step":1,"steps":4,"width":512,"height":512,"b64_json":"<jpeg>"}
...
data: {"type":"final","result":{...полный ответ...}}
data: [DONE]
```

## Проверка

`python tests/api_test.py http://127.0.0.1:8080` — сквозные проверки всех
эндпоинтов и форматов: текст, стриминг, JSON-режим, системное сообщение,
инструменты (вызов, ответ по результату, стриминг), размышления, `n`,
кэш токенов, долгосрочная память, автосуммаризация, ошибки; решения по
тексту, фото в 6 форматах, data URL, аудио и рамкам; апскейл в 5
форматов; генерация в 3 формата, SSE-превью, `n>1`; `/v1/predict` своей
модели и проверка ввода. Результаты пишутся в `data/api_test_results.json`.
Ожидаемые модели: `qwen` (Qwen3-0.6B), `llama`, `d1`, `x2`, `sd`, `seq` —
см. команду запуска в начале файла теста.
