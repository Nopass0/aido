# SDK: свои модели — сборка, обучение, сигнатура входа/выхода, компиляция, запуск

## Цикл жизни модели

```
определение (Rust, sdk::autograd / sdk::models)
      │  aido train-* / свой цикл обучения с AdamW и чекпоинтами
      ▼
checkpoint .aido  (веса f32 + состояние оптимизатора + конфиг)
      │  + сигнатура входа/выхода (aido.io)
      ▼
aido compile -m model.aido -o model-f16.aido --quant f16     # инференс-версия
      │
      ├── aido model-info model-f16.aido                      # что на входе/выходе
      ├── aido predict -m model-f16.aido -i '{"…": …}'        # запуск с JSON
      ├── aido serve -m name=model-f16.aido  →  POST /v1/predict
      └── Models::predict("name", &json!({…}))                # из своего приложения
```

## Сигнатура: что на входе, что на выходе

Сигнатура лежит в метаданных `.aido` (ключ `aido.io`) и делает файл
самоописываемым:

```json
{
  "task": "sequence-next",
  "description": "Finds the rule of an integer sequence … and predicts the next element",
  "inputs":  [{"name": "sequence", "type": "int[]", "description": "5–8 integers"}],
  "outputs": [{"name": "next", "type": "int", "description": "the next element"}],
  "codec": {"kind": "char-lm", "vocab": ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", " ", "-", "|", "\n"],
            "bos": 14, "stop": 14, "template": "{sequence}|", "join": ", ", "max_new": 12},
  "examples": [{"input": {"sequence": [2, 4, 6, 8, 10]}, "output": {"next": 12}}]
}
```

* **Поля** (`inputs` / `outputs`): `int`, `float`, `text`, `bool`, `int[]`,
  `float[]`, `text[]`, `label` (с `labels`), `image` (`shape`), `json`.
  Вход проверяется по типам до запуска модели, ошибка называет поле.
* **Кодек** описывает, как значения превращаются в токены модели и
  обратно: `char-lm` — посимвольный словарь (шаблон промпта с `{поле}`,
  массивы склеиваются через `join`, генерация до `stop`, разбор выхода по
  типу поля), `byte-lm` — байтовая модель.

Из Rust:
```rust
use aido::sdk::io::{Signature, Field, Codec, attach};
let sig = Signature {
    task: "sentiment".into(), description: "Тональность отзыва".into(),
    inputs: vec![Field::new("text", "text", "отзыв")],
    outputs: vec![Field::new("label", "text", "positive | negative")],
    codec: Codec::ByteLm { template: "{text}\n=> ".into(), max_new: 10 },
    examples: vec![],
};
attach("models/my.aido", "models/my.aido", &sig)?;
```

## Пример: модель поиска закономерностей `seq`

```bash
aido train-seq --steps 1500 -o models/seq.aido        # обучение (сигнатура прикрепляется автоматически)
aido model-info models/seq.aido                        # входы/выходы/кодек/примеры
aido compile -m models/seq.aido -o models/seq-c.aido --quant f16   # без состояния оптимизатора, веса f16
aido predict -m models/seq-c.aido -i '{"sequence": [3, 6, 9, 12, 15]}'
#  { "next": 18 }
aido serve -m seq=models/seq-c.aido --port 8080
curl -s localhost:8080/v1/predict -H "Content-Type: application/json" \
     -d '{"model": "seq", "input": {"sequence": [1, 1, 2, 3, 5, 8]}}'
#  {"object":"prediction","model":"seq","output":{"next":13},"ms":4.1}
```

Для старого чекпоинта без сигнатуры: `aido model-info models/seq.aido --attach sequence-next`.
`GET /v1/models` показывает сигнатуру каждой custom-модели в поле `io`,
так что клиент может построить форму ввода автоматически.

## Ресурсы

SDK-модели в сервере и библиотеке используют общий пул потоков движка
(привязанный к физическим ядрам) и тот же реестр с LRU-выгрузкой и
бюджетом ОЗУ, что и остальные модели. Обучение (`sdk::autograd`) идёт на
CPU; инференс маленьких моделей — тоже на CPU, где для них нет задержки
пересылки в GPU.

## Строительные блоки

`sdk::autograd::Graph` — define-by-run автоград: `linear` (+LoRA), `add`,
`mul`, `scale`, `relu`, `silu`, `tanh`, `gelu`, `swiglu`, `rmsnorm`,
`embed`, `rope`, `reshape`, `causal_attention`, `concat_cols`,
`cross_entropy`, `mse`, `conv3x3` (шаг 1/2), `upsample2x`, `bce_logits`,
`gather_rows`, `ctc`. `sdk::AdamW` (warmup + cosine, клиппинг),
`sdk::save_checkpoint` / `load_checkpoint`, `sdk::data` (`.aidata`),
готовые модели в `sdk::models` (`LmModel`, `Classifier`, `Ddpm`), а также
`sdk::pixelart`, `sdk::textdet`, `sdk::snake`, `sdk::bpe`.
