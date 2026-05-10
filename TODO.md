# AgentPinBoard TODO

Phase 1, 2, 3 завершены — 172 теста, lint clean, оба sanity-скрипта
зелёные. Ниже — список известных ограничений и желаемых улучшений,
который накопился в процессе.

## Приоритет 1 — `LangfuseHook` переделать

**Проблема.** Текущая реализация технически работает, но неудобна в
реальной Langfuse-работе. При открытии Langfuse UI после агент-рана:

1. **Spans оторваны от agent-trace.** `client.start_observation(...)`
   без `trace_context` создаёт **новый top-level trace на каждый
   ingest**. Правильно — вкладывать span в текущий trace (у которого
   есть LLM-call на верхнем уровне), чтобы была иерархия:

   ```
   trace: "agent invoke"
     ├─ LLM: gpt-4o
     ├─ tool: fetch_cloudtrail
     │   └─ agent_pinboard.ingest    ← должно быть ЗДЕСЬ
     └─ LLM: next step
   ```

   Сейчас agent-pinboard-spans рождаются рядом как disconnected traces —
   корреляция с LLM-reasoning'ом, породившим ingest, теряется.

2. **Два span'a на каждый @pin — шум.** `agent_pinboard.ingest` +
   `agent_pinboard.graph_snapshot` стреляют всегда парой (снимок дёргается
   из `on_graph_changed`, который зовётся сразу после
   `on_ingest_complete`). 10 тул-вызовов → 20 лишних spans в trace.
   Правильнее один span на ingest.

3. **Mermaid в `metadata` скорее всего не рендерится диаграммой.**
   Langfuse-UI рендерит Markdown (включая ```mermaid fenced blocks)
   в **output** полях, не в произвольной `metadata`. Текущий код
   кладёт Mermaid в metadata → получаем просто текст вместо визуала.

4. **Нет `session_id` / `user_id` / `tags`.** Langfuse использует эти
   поля для группировки traces. `thread_id` естественно ложится на
   `session_id`, но хук это не прокидывает.

5. **Нет авто-flush.** Langfuse-клиент буферизует; без
   `client.flush()` spans могут не уйти к моменту завершения агента.
   Пользователь должен сам догадаться.

**Желаемая реализация.**

- **A. Один span `agent_pinboard.ingest` на вызов**:
  - `input` — canonical args summary
  - `output` — Markdown с дельтой + Mermaid в ```mermaid fenced
    block (рендерится в UI)
  - `metadata` — structured `IngestResult`
  - Параметр `include_graph_in_output: bool = True`

- **B. Nest под текущий trace** через
  `client.start_as_current_observation(...)` как context manager
  (Langfuse v3+ nativeовый OTel-нэстинг). Если активного trace нет —
  создаём disconnected, как сейчас (graceful degradation).

- **C. Опциональные `session_id` / `user_id` / `tags`** в
  `LangfuseHook(...)` конструкторе.

- **D. `client.flush()` после каждого ingest** (Langfuse батчит,
  дополнительных round-trips не будет).

- **E. Убрать `on_graph_changed` из хука** — всё переезжает в
  `on_ingest_complete`. Если больше никто не использует
  `on_graph_changed(graph)` extended-сигнатуру — можно её упростить
  обратно до `on_graph_changed()` (пометить deprecated для
  pre-1.0 upgrade-path'a).

Объём: ~200 LOC рефактор, тесты остаются те же (за исключением
`test_on_graph_changed_emits_mermaid` → переезжает в
`test_on_ingest_complete_emits_mermaid_in_output`).

## Приоритет 2 — quality-of-life

- [ ] **Flush-helper для `LoggingHook`/`LangfuseHook`/`WebSocketHook`.**
  Единый вызов `hooks.shutdown()` перед выходом агента — закрывает
  соединения, ждёт буферы.
- [ ] **`FactGraph.to_mermaid()` как первоклассный метод.** Сейчас
  `render_mermaid` живёт внутри `integrations/langfuse_hook.py`; логичнее
  держать его на `FactGraph` (а в LangfuseHook импортировать). Также
  нужен для WebSocketHook (сейчас не использует Mermaid).
- [ ] **`@pin(store_raw=True)` + `get_evidence` — async-тестов нет.**
  Async-покрытие есть в других местах; для этих путей проверить
  симметрию.

## Приоритет 3 — возможные расширения

- [ ] **`find_path` с взвешенными рёбрами.** Сейчас все рёбра считаются
  равноправными. Можно взвешивать по `len(source_events)` ноды на
  другом конце (частые facts — короче путь) или по типу связи. Будет
  различаться только с `top>1`.
- [ ] **`search_nodes(include_tools=True)`.** Фильтр «только ноды,
  которые пришли из этого тула». Удобно для debug'a когда интересно
  «что конкретно мы получили из VT».
- [ ] **Typed `hooks` protocol.** `AgentPinBoardHooks` сейчас — обычный
  класс с no-op методами. Можно перевести на `typing.Protocol`, чтобы
  duck-typing без наследования был валиден для typechecker'a.
- [ ] **`configure(ingest_queue_maxsize=...)`** для `WebSocketHook`
  (сейчас задаётся в конструкторе, возможно имеет смысл сделать
  глобальным).
- [ ] **Примеры `agent-pinboard-aws` / `agent-pinboard-enrichment-vt` пакетов-плагинов** —
  демонстрация паттерна, как расширять AgentPinBoard для конкретных
  источников.

## Known limitations (документированные, не баги)

- Multi-process не поддерживается — in-memory кеш + `RLock` работают
  только в одном процессе. Для worker-based deployment нужен
  PostgreSQL-Store + distributed lock (сейчас вне scope).
- Mermaid рендеринг в графе с 500+ нодами превышает лимит UI;
  `max_facts_in_snapshot=30` подобрано эмпирически.
- `skip_events=True` в `explore`/`find_path` делает `direction`
  параметр безразличным для FactNode (inbound-only топология звезды).
