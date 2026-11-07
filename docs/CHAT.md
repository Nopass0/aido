# Диалог: системное сообщение, история, инструменты, размышления, память

Всё ниже работает одинаково в HTTP API (`/v1/chat/completions`) и во
встраиваемой библиотеке (`Models::chat_full`).

## Сообщения и история

`messages` — вся история диалога: `system` (инструкции модели), `user`,
`assistant` (в том числе с `tool_calls`), `tool` (результат вызова
инструмента, `tool_call_id`). Применяется собственный шаблон чата модели
из GGUF (Jinja), поэтому форматирование совпадает с обучением модели.

**Кэш токенов.** Движок держит до 6 сессий с KV-кэшем. Новый запрос
продолжает ту сессию, у которой самый длинный общий префикс токенов, и
пересчитывается только новый хвост. В длинном диалоге каждый следующий
ход стоит столько, сколько новые сообщения, а не вся история. Сколько
токенов взято из кэша, видно в `usage.prompt_tokens_details.cached_tokens`.
KV хранится постранично и сжатым (Q8), поэтому слоты дешёвые.

**Автосуммаризация.** Если история не помещается в контекст (`--ctx`)
вместе с `max_tokens`, самые старые сообщения (кроме системных и
последнего) сворачиваются в краткое изложение — его пишет та же модель
(факты, имена, числа, решения, предпочтения), — и оно подставляется
системным сообщением «Summary of the earlier conversation». Сколько
сообщений свёрнуто — `usage.aido.summarized_messages`. Отключается
`"auto_summarize": false` (тогда при переполнении вернётся ошибка).

## Инструменты (function calling)

Формат OpenAI:
```json
{"model": "qwen", "messages": [{"role": "user", "content": "Какая погода в Казани?"}],
 "tools": [{"type": "function", "function": {"name": "get_weather", "description": "Погода в городе",
            "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}],
 "tool_choice": "auto"}
```
Ответ:
```json
{"choices": [{"message": {"role": "assistant", "content": null,
   "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "get_weather", "arguments": "{\"city\":\"Казань\"}"}}]},
   "finish_reason": "tool_calls"}]}
```
Дальше клиент выполняет функцию и присылает историю с сообщением
`{"role": "tool", "tool_call_id": "call_0", "content": "{\"temp\": 3}"}`, и
модель отвечает с учётом результата. Поддерживаются форматы вызовов Qwen
2.5/3 и Hermes (`<tool_call>…</tool_call>`), Llama 3.1+ (JSON с
`parameters`, `<|python_tag|>`), несколько вызовов за ход. Если шаблон
модели не знает про инструменты, их описание добавляется в системное
сообщение в формате Hermes. `tool_choice: "none"` отключает инструменты.
При стриминге текст идёт обычными чанками, а вызовы приходят структурно
(`delta.tool_calls`) в конце хода: разметку вызова движок в поток не
выпускает.

## Размышления (thinking)

`"enable_thinking": true` (или `"reasoning_effort": "low|medium|high"`)
включает режим рассуждений у моделей, которые его поддерживают (Qwen3,
DeepSeek-R1-дистилляты…). Текст внутри `<think>…</think>` отделяется от
ответа:
```json
{"message": {"role": "assistant", "content": "Ответ: 42", "reasoning_content": "Сначала посчитаю…"}}
```
При стриминге рассуждения идут в `delta.reasoning_content`, ответ — в
`delta.content`. Разметка тегов не попадает в поток, даже если тег пришёл
по частям в разных токенах.

## Несколько вариантов ответа

`"n": 3` — три ответа (`choices[0..2]`), каждый со своим seed. Промпт
считается один раз: остальные варианты берут его из кэша префикса. При
стриминге чанки помечены `choices[].index`.

## Долгосрочная память (векторная база)

Сервер хранит память по пользователю (поле `user` запроса, как в OpenAI)
в `--memory-dir` (по умолчанию `data/memory`; `none` — выключить):

* перед ходом в системное сообщение подставляются до 5 самых релевантных
  записей о пользователе;
* модель получает встроенные инструменты `memory_search(query)` и
  `memory_save(text)`, вызывает их сама сколько нужно (до 4 раундов за
  ход), а движок выполняет их и продолжает генерацию. Так модель
  «вспоминает» нужное по ходу рассуждения, а не держит всё в контексте;
* после хода обмен «пользователь сказал / ассистент ответил» сохраняется.

Поиск гибридный: `0.6·косинус(эмбеддинги) + 0.4·BM25`. Векторы считает
встроенный многоязычный эмбеддер `aido-hash-512` (хеширование слов, пар
слов и символьных 3–5-грамм любой письменности). Ему не нужна отдельная
модель, он устойчив к словоформам («кошка/кошку») и опечаткам. BM25
отвечает за точные имена и числа. Векторы лежат рядом с фактами
(`<user>.vec`), поиск по 100 тыс. записей занимает около 10 мс. Через
трейт `aido::vector::Embedder` можно подключить нейросетевой эмбеддер.

Управление памятью:
```bash
curl "http://127.0.0.1:8080/v1/memory?user=alice"                      # список
curl "http://127.0.0.1:8080/v1/memory?user=alice&query=кошка&k=3"      # поиск
curl -X POST   http://127.0.0.1:8080/v1/memory -d '{"user":"alice","text":"Алиса — веган"}' -H "Content-Type: application/json"
curl -X DELETE http://127.0.0.1:8080/v1/memory -d '{"user":"alice","id":3}' -H "Content-Type: application/json"   # без id — всё
```
Полностью отключить для запроса: `"memory": false`.

## Библиотека

```rust
use aido::{Models, ModelsConfig, models::ChatOptions};
use aido::llm::{chat::Message, conversation::Piece};

let models = Models::new(ModelsConfig { ctx: 8192, memory_dir: Some("data/memory".into()), ..Default::default() })?;
models.add("qwen", "models/Qwen3-8B-Q4_K_M.gguf")?;
let r = models.chat_full("qwen",
    &[Message::new("system", "Ты лаконичный помощник."), Message::new("user", "Как меня зовут?")],
    &ChatOptions { user: Some("alice".into()), think: true, n: 1, ..Default::default() },
    &mut |i, piece| { match piece { Piece::Content(t) => print!("{t}"), Piece::Reasoning(t) => eprint!("{t}") }; true })?;
println!("\n{:?} {:?}", r.choices[0].tool_calls, r.choices[0].reasoning);
```
