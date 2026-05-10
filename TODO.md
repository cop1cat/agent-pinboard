# AgentPinBoard TODO

Phase 1, 2, 3 завершены. PR #1 (rename `pinboard` → `agent_pinboard`,
`@fact` → `@pin`), PR #2 (mergeable FactNode storage, no process-local
cache), PR #3 (LangChain-callbacks observability) тоже смёрджены.
Ниже — список известных ограничений и желаемых улучшений.

## Резолвированное в PR #3

- **`LangfuseHook` переделан** под LangChain `BaseCallbackHandler`.
  Span теперь nest'ятся под текущий tool-span (не disconnected
  trace), потому что dispatch идёт через стандартный
  `dispatch_custom_event` — родительский run-context передаётся
  автоматически.
- **`hooks=` параметр у `@pin` удалён.** `agent_pinboard.hooks`
  модуль удалён вместе с ним. Observability — только через
  `config={"callbacks": [...]}` на `agent.invoke`.
- **`WebSocketHook` тоже стал `BaseCallbackHandler`** и подписывается
  на тот же `agent_pinboard:ingest` custom-event.

Открытые вопросы по интеграциям, не вошедшие в PR #3:

- [ ] **Mermaid в `output` вместо `metadata`.** Langfuse UI рендерит
  Markdown (включая ```mermaid fenced blocks) в `output`-полях;
  сейчас Mermaid-снимок кладётся в `metadata` отдельного span'а и
  отображается как plain text. Перенести Mermaid в `output` основного
  `agent_pinboard.ingest` span'а — будет один span с визуалом, не два.
- [ ] **Опциональные `session_id` / `user_id` / `tags`** в
  `LangfuseHook(...)` конструкторе для группировки traces.
- [ ] **Авто `client.flush()`** после каждого ingest (Langfuse батчит,
  без flush span может не уйти к моменту завершения агента).

## Приоритет 2 — quality-of-life

- [ ] **`FactGraph.to_mermaid()` как первоклассный метод.** Сейчас
  `render_mermaid` живёт внутри `integrations/langfuse_hook.py`;
  логичнее держать его на `FactGraph` (а в LangfuseHook импортировать).
  Также нужен для WebSocketHook (сейчас не использует Mermaid).
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
- [ ] **`configure(ingest_queue_maxsize=...)`** для `WebSocketHook`
  (сейчас задаётся в конструкторе, возможно имеет смысл сделать
  глобальным).
- [ ] **Примеры `agent-pinboard-aws` / `agent-pinboard-enrichment-vt`
  пакетов-плагинов** — демонстрация паттерна, как расширять
  AgentPinBoard для конкретных источников.

## Known limitations (документированные, не баги)

- Mermaid рендеринг в графе с 500+ нодами превышает лимит UI;
  `max_facts_in_snapshot=30` подобрано эмпирически.
- `skip_events=True` в `explore`/`find_path` делает `direction`
  параметр безразличным для FactNode (inbound-only топология звезды).
