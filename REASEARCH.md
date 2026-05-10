# LLM-агентская рабочая память как граф фактов: prior art и ландшафт решений

**PinBoard занимает уникальную и пока незанятую нишу.** Ни одна из существующих библиотек не реализует комбинацию декларативных Fact-аннотаций на полях Pydantic-моделей, side-effect декоратора `@fact` для тулов и session-scoped графа без LLM-извлечения. Ближайшие конкуренты — Graphiti и Cognee — ориентированы на персистентную долговременную память с LLM-extraction, что делает их принципиально другими по архитектуре, стоимости и паттернам использования. Однако зрелые системы предлагают ряд возможностей (bi-temporal модель, confidence scoring, обработка противоречий), отсутствие которых в PinBoard составляет основной дизайн-риск.

---

## A. Прямые конкуренты и аналоги

### Graphiti (Zep AI) — мощный, но тяжёлый и LLM-зависимый

Graphiti реализует трёхуровневый граф знаний в Neo4j/FalkorDB/Neptune: **эпизодический подграф** (сырые данные), **семантический подграф** (извлечённые сущности и факты на рёбрах) и **community-подграф** (кластеры сущностей с суммаризациями). Архитектура напоминает модель человеческой памяти с разделением на эпизодическую и семантическую. Каждая единица информации — «эпизод» (message, text или json) — обрабатывается через 6-шаговый LLM-пайплайн: извлечение сущностей → дедупликация сущностей → извлечение фактов → дедупликация рёбер → темпоральное извлечение → инвалидация старых фактов.

**Ключевое отличие от PinBoard**: Graphiti полагается на LLM для всех этапов извлечения — это даёт гибкость, но приводит к **600K+ токенов на диалог** по некоторым бенчмаркам и **1000+ API-запросов на 10K символов текста**. PinBoard полностью исключает LLM из извлечения: факты определяются декларативно через аннотации `Annotated[T, Fact(...)]`, что делает процесс детерминированным и бесплатным. Graphiti реализует **bi-temporal модель** (valid_at/invalid_at + created_at/expired_at) — зрелый подход, которого у PinBoard нет. Entity resolution в Graphiti прошёл эволюцию от чистого LLM-matching до MinHash+LSH с LLM-fallback, что существенно снизило стоимость. Graphiti имеет официальную интеграцию с LangGraph, но спроектирован для **персистентной cross-session памяти** — использование для session-scoped сценариев возможно через `group_id`, но не является основным паттерном. Проблемы на практике: event loop конфликты с async-фреймворками, сбои structured output с не-OpenAI моделями (issues #912, #485), hallucinations при entity resolution (#760).

GitHub: https://github.com/getzep/graphiti | arXiv: 2501.13956

### Mem0 — простой API, но граф вторичен

Mem0 — гибридная система: **vector store (основной) + опциональный graph store**. При вызове `memory.add()` LLM извлекает факты и решает, что делать: ADD, UPDATE, DELETE или NOOP. Граф в Mem0 хранит только тонкие триплеты (source → relationship → destination) без натурально-языковых фактов на рёбрах — в отличие от Graphiti, где рёбра несут полные описания фактов с эмбеддингами.

**Ключевое отличие от PinBoard**: Mem0 — это персистентная долговременная память для персонализации (user preferences), а не session-scoped рабочая память для расследований. Граф и вектор-стор **не разделяют ID и работают независимо**, что приводит к рассинхронизации. Нет temporal validity на фактах. На LongMemEval набирает только **49%**, проваливаясь на multi-hop запросах. Токен-эффективнее Graphiti (~1,764 токена/диалог), но за счёт потери структурной информации.

GitHub: https://github.com/mem0ai/mem0

### Letta/MemGPT — другая парадигма, не конкурент

Letta реализует **OS-inspired иерархическую память**: core memory (in-context блоки, которые агент редактирует через tool calls), recall memory (поиск по истории), archival memory (vector store). Это принципиально другой подход — **агент сам управляет своей памятью** через self-editing, без автоматического извлечения сущностей. Нет knowledge graph, нет entity resolution, нет структурированных фактов.

**Ключевое отличие от PinBoard**: Letta решает задачу **context window management** (как дать агенту иллюзию бесконечной памяти), а не задачу структурированного представления знаний. Концепция memory blocks (self-editable текстовые блоки) ортогональна графу фактов и может быть комплементарна — PinBoard мог бы использовать подход memory blocks для summary-представления графа в контексте.

GitHub: https://github.com/letta-ai/letta

### Cognee — ближайший по духу, но document-oriented

Cognee строит **полноценный knowledge graph** через ECL-пайплайн (Extract → Cognify → Load). Использует Pydantic-модели DataPoint как основу для нод и рёбер — **архитектурно ближайший аналог PinBoard** в части декларативного определения типов. Поддерживает **онтологии (OWL/RDF)** для каноникализации извлечённых сущностей — зрелая функция, отсутствующая у PinBoard. Уникальная функция `memify()` — самосовершенствование графа: pruning стейл-нод, усиление частых связей, добавление производных фактов.

**Ключевое отличие от PinBoard**: Cognee ориентирован на **document ingestion** (PDF, CSV, 38+ форматов), а не на парсинг ответов API-тулов. Извлечение LLM-driven. Заявляет поддержку session memory через `session_id`, но основной фокус — персистентный граф знаний. Нет side-effect декоратора для тулов, нет автолинковки по canonical_value.

GitHub: https://github.com/topoteretes/cognee

### LlamaIndex PropertyGraphIndex — инфраструктура без agent lifecycle

PropertyGraphIndex предоставляет **зрелую graph-инфраструктуру**: labeled property graph с типизированными нодами (EntityNode, ChunkNode), рёбрами с properties, несколькими стратегиями извлечения (SchemaLLMPathExtractor с Pydantic-валидацией, ImplicitPathExtractor без LLM). API `upsert_nodes()` / `upsert_relations()` позволяет программно добавлять ноды/рёбра в runtime.

**Ключевое отличие от PinBoard**: PropertyGraphIndex спроектирован для **document RAG**, не для agent working memory. Нет session scoping, нет TTL, нет интеграции с agent lifecycle. Теоретически можно использовать `SimplePropertyGraphStore` (in-memory) как хранилище для session-графа, но потребуется полностью custom обвязка. KnowledgeGraphIndex **официально deprecated** в пользу PropertyGraphIndex.

Docs: https://docs.llamaindex.ai/en/stable/module_guides/indexing/lpg_index_guide/

### LangGraph Store и checkpoints — фундамент без графовой семантики

LangGraph реализует **двухуровневую память**: checkpoints (thread-scoped состояние агента между шагами) и Store (cross-thread key-value хранилище с namespace-изоляцией и vector search). `InMemoryStore` поддерживает semantic search по эмбеддингам. Namespace-паттерн (`("user_123", "memories")`) обеспечивает мульти-тенантную изоляцию.

**Ключевое отличие от PinBoard**: Store — это **плоский key-value store**, не граф. Нет концепции нод, рёбер, graph traversal, entity resolution. PinBoard фактически строит граф **поверх** InMemoryStore, добавляя графовую семантику, которой LangGraph не предоставляет. Библиотека `langmem` добавляет automated fact extraction, но хранит факты как отдельные документы без графовой структуры. Старые LangChain memory-классы (ConversationEntityMemory и др.) **deprecated**.

Docs: https://docs.langchain.com/oss/python/langgraph/persistence

---

## B. Академические работы (2023–2025)

| # | Работа | Ключевая идея |
|---|--------|---------------|
| 1 | **A-MEM** (Xu et al., 2025, arXiv:2502.12110, NeurIPS 2025) | Zettelkasten-inspired: структурированные «заметки» с ключевыми словами, тегами и LLM-generated связями между ними; memory evolution — новые факты обновляют контекст старых |
| 2 | **GraphRAG** (Edge et al., 2024, arXiv:2404.16130) | LLM-извлечение entity/relationship → Leiden community detection → иерархические суммаризации сообществ; local vs global search |
| 3 | **HippoRAG** (Gutierrez et al., 2024, arXiv:2405.14831, NeurIPS 2024) | Schemaless KG как hippocampal index + Personalized PageRank для single-step multi-hop retrieval; **20% лучше** SOTA на multi-hop QA |
| 4 | **MemWalker** (Chen et al., 2023, arXiv:2310.05029) | Иерархическое дерево суммаризаций + LLM-навигация по дереву с backtracking; working memory при обходе |
| 5 | **MemoryBank** (Zhong et al., 2024, AAAI) | Ebbinghaus forgetting curve для decay релевантности; трёхуровневое хранение (диалоги → события → профили) |
| 6 | **AriGraph** (Anokhin et al., 2024, arXiv:2407.04363, IJCAI 2025) | Инкрементальное построение KG из наблюдений агента в интерактивной среде; dual semantic+episodic memory в одном графе — **ближайший аналог session-scoped fact graph** |
| 7 | **MAGMA** (2026, arXiv:2601.03236) | Четыре ортогональных графовых слоя + dual-stream write (fast path + async deep); SOTA на LongMemEval и LoCoMo |
| 8 | **Graph-based Agent Memory Survey** (2026, arXiv:2602.05665) | Таксономия: KG, temporal graphs, hypergraphs, hierarchical trees, hybrid; lifecycle: extraction → storage → retrieval → evolution |
| 9 | **BoostER** (Li et al., 2024, WWW '24, arXiv:2401.03426) | Cost-efficient entity resolution через uncertainty-based active selection LLM-запросов для matching |
| 10 | **Knowledge Graph Prompting** (Wang et al., 2024, AAAI) | KG-навигация LLM-агентом по multi-type графу для выбора контекста; graph-to-prompt сериализация |

Дополнительно релевантны: **Reasoning on Graphs (RoG)** (ICLR 2024) — faithful KG-reasoning через relation path planning; **Think-on-Graph (ToG)** (ICLR 2024) — beam search по KG с чередованием exploration/reasoning шагов; **LogAct** (2025) — write-ahead log как shared state machine для агентов с semantic recovery.

---

## C. Инженерные best practices

### Компактное представление подграфов для LLM

Исследования показывают, что **формат сериализации критически влияет на качество**. Adjacency list устойчивее к перетасовке порядка элементов, чем edge list (arxiv:2511.10234). RDF Turtle и Markdown дают LLM лучшие результаты, чем prose или CSV, потому что «синтаксис сам несёт семантику о том, что является нодой, а что свойством» (TrustGraph). Формат TOON заявляет **60% сокращение токенов** vs JSON за счёт однократного объявления схемы. Практические границы: **1–2 хопа** для стандартных запросов, **3 хопа максимум** до over-smoothing (InstructGLM); **5–35 нод** — типичный размер подграфа для reasoning; 200 нод превышает **32K токенов**. Стратегия выбора salient nodes: PageRank/community detection для центральности, embedding-similarity для релевантности запросу, tiered фильтрация лёгкими моделями перед дорогим LLM-вызовом — даёт **10x сокращение токенов**.

### Discovery-тулы для агентов с неизвестной структурой

Паттерн MCP-Zero (arxiv:2506.01056) решает проблему: агент **сам запрашивает недостающие инструменты** через hierarchical semantic routing вместо инъекции всех схем в промпт. Maltego задаёт золотой стандарт UX: Transform (entity → related entities) = `explore(entity_id)`, entity type inheritance (transforms на родительском типе работают на subtypes), Machines (цепочки transforms). Минимальный набор discovery-тулов: `list_types()`, `explore(entity)`, `search(query, pattern)`, `get_schema()`, `expand(entity, rel_type)`. Архитектурный принцип: **не давать агенту все инструменты сразу**, а позволять обнаруживать их по мере необходимости.

### Контроль каскадного обогащения без budget-механизмов

Семь паттернов из практики: **(1) Hard guardrails** — абсолютные лимиты на количество шагов и время выполнения (MAX_TURNS=25, MAX_TIME=300s); **(2) Repetition detection** — мониторинг повторных tool calls с одинаковыми параметрами; **(3) Depth limits** — ReDel (arxiv:2408.02248) показывает, что цепочки 3+ делегаций с 0–1 детьми = undercommitment и бесконечные циклы; **(4) FSM + explicit Exit tool** — агент работает в конечном автомате с явным инструментом завершения; **(5) Goal reminders** — периодическая реинъекция исходной цели в контекст; **(6) LogAct write-ahead log** — действия записываются до выполнения, pluggable voters могут заблокировать; **(7) Relevance decay** — implicit budget через снижение relevance score при удалении от исходного запроса. Для PinBoard рекомендуется комбинация: depth limit на explore + repetition detection на уровне графа + explicit Exit.

### Паттерны логов действий при сжатии контекста

Каноническая трёхуровневая модель: short-term (in-context window), episodic (лог событий с TTL), semantic (знания). **Structured tool call log** должен фиксировать: tool name, input params (sanitized), output, timestamp, latency, status, error, **rationale** (почему агент выбрал этот тул) и **interpretation** (как агент интерпретировал результат). При сжатии контекста: **rolling window** последних N tool calls в контексте + суммаризация старших записей через LLM + off-context retrieval из vector store по запросу. LogAct-паттерн (arxiv:2604.07988) предлагает write-ahead log с semantic recovery: после crash агент анализирует свой лог и генерирует компенсирующие действия. PinBoard's `what_have_i_done` реализует именно этот паттерн — лог вызовов как инструмент для агента.

---

## D. Доменно-специфичные платформы

### Что переносимо из i2, Maltego и Palantir в LLM-агентскую память

IBM i2 Analyst's Notebook задаёт методологию **ELP (Entity-Link-Property)** — фундаментальную модель данных, совпадающую с PinBoard. Ключевой инсайт: **не все связи равноценны** — i2 и Sentinel Visualizer взвешивают рёбра по типу отношения и надёжности источника (source reliability + credibility). Timeline analysis — хронологическое отображение связей для аномалий — прямо переносим в `timeline`-тул PinBoard. Maltego's Transform-паттерн (entity → related entities) — точный аналог PinBoard's `explore`. **Entity merging по type + key value** в Maltego — это ровно автолинковка PinBoard по `(node_type, canonical_value)`. Maltego Machines (макросы из цепочек transforms с автоочисткой orphan-нод) подсказывают паттерн для automated enrichment pipelines. Palantir Gotham подтверждает жизнеспособность модели «множество источников → единый граф», что соответствует PinBoard's use case связывания сущностей между разными API.

### OCSF и STIX/TAXII как словари для node_type

**STIX 2.1** — готовый к использованию граф-ориентированный стандарт. SDO (Attack Pattern, Threat Actor, Malware, Campaign, Vulnerability и др.) и SCO (IPv4-Addr, Domain-Name, Email-Addr, File, Process, User-Account и др.) **идеально подходят как значения node_type** в PinBoard. Relationship types (uses, targets, indicates, attributed-to) задают словарь для типов рёбер. Python-библиотека `stix2` (PyPI) — зрелая, с immutable объектами и валидацией. **OCSF** лучше подходит для **нормализации событий** (Authentication, DNS Activity, Network Activity), а не для типизации сущностей. Объекты OCSF (user, device, process) могут служить node_type, но event classes — скорее шаблоны для фактов/рёбер. Python-библиотека `ocsf-lib-py` существует, но менее зрелая. **Рекомендация**: STIX SDO/SCO для entity types, OCSF event classes для event normalization — два стандарта комплементарны.

---

## E. Итоговая оценка

### Уникальность ниши PinBoard подтверждена

Ни одна из исследованных библиотек не реализует комбинацию трёх ключевых архитектурных решений PinBoard:

1. **Декларативные Fact-аннотации** (`Annotated[T, Fact(node_type="...", role="...", normalizer=...)]`) — ни Graphiti, ни Mem0, ни Cognee не используют аннотации на полях Pydantic-моделей для определения, какое поле становится нодой. Cognee использует Pydantic DataPoint, но для LLM-extraction, не для декларативной разметки существующих моделей.

2. **Side-effect декоратор `@fact`** — во всех конкурентах memory API вызывается явно (`graphiti.add_episode()`, `memory.add()`, `cognee.add()`). PinBoard's подход — прозрачное извлечение фактов при вызове тула без изменения return value — не имеет аналогов.

3. **Session-scoped граф без LLM-extraction** — все конкуренты либо персистентны (Graphiti, Mem0, Cognee), либо используют LLM для извлечения. PinBoard делает extraction детерминированным и привязанным к сессии.

### Десять рисков дизайна, которые необходимо адресовать

**1. Temporal invalidation.** Graphiti реализует bi-temporal модель с valid_at/invalid_at — критически важно для security use cases, где IP может быть скомпрометирован в определённый период. PinBoard не имеет механизма инвалидации фактов при получении противоречащей информации.

**2. Обработка противоречивых фактов.** Когда два тула возвращают конфликтующую информацию об одной сущности (например, разные GeoIP-данные), граф молча содержит оба факта. Нужна хотя бы стратегия last-write-wins или confidence scoring.

**3. Confidence scoring и source reliability.** i2 и Maltego реализуют взвешивание рёбер по надёжности источника — для security расследований это не опция, а необходимость. Факт из VirusTotal и факт из WHOIS имеют разный вес.

**4. Рост графа и context budget.** При **200 нодах граф превышает 32K токенов** в текстовом представлении. Без стратегии salient node selection и summarization агент быстро потеряет способность работать с графом. Нужны PageRank-like метрики или relevance decay.

**5. Entity resolution beyond exact match.** Автолинковка по `(node_type, canonical_value)` работает для IP-адресов, но не для person names (John Smith vs J. Smith), company names (Google LLC vs Alphabet Inc) или email-вариаций. Литература (BoostER, Peeters & Bizer 2025) показывает, что LLM-based matching даёт **40–68% improvement** на cross-domain entity matching.

**6. Каскадное обогащение.** Без budget-механизмов агент может войти в бесконечный цикл explore → enrich → explore. Минимально необходимы: depth limit, repetition detection, explicit completion signal.

**7. Schema discovery.** Типы нод user-defined, ядро их не знает. Агенту нужны discovery-тулы (`list_types`, `get_schema`) для ориентации в незнакомом графе. MCP-Zero показывает, что hierarchical tool discovery драматически сокращает token overhead.

**8. Offline/async enrichment.** Letta's sleep-time compute и MAGMA's dual-stream write показывают, что async обогащение (глубокая обработка в фоне, пока агент работает) — зрелый паттерн, повышающий responsiveness.

**9. Multi-hop query support.** HippoRAG демонстрирует, что Personalized PageRank для multi-hop retrieval **на 20% лучше** чистого vector search. `find_path` и `explore` в PinBoard покрывают basic traversal, но нужен механизм ранжирования путей по relevance.

**10. Provenance и аудит.** Graphiti хранит двунаправленные ссылки episode ↔ fact. PinBoard's `get_evidence` идёт в правильном направлении, но для compliance-сценариев (due diligence, incident response) нужна полная цепочка: tool call → raw response → extracted fact → linked node, с возможностью воспроизведения.

---

## Заключение: стратегические выводы для проектирования PinBoard

PinBoard не конкурирует с Graphiti, Mem0 или Cognee напрямую — это **другой architectural tradeoff**: детерминизм и дешевизна extraction за счёт отказа от LLM-гибкости, session scope за счёт отказа от персистентности. Этот tradeoff обоснован для целевых use cases (security investigations, due diligence), где структура данных **известна заранее** (API возвращают Pydantic-модели с фиксированными полями) и LLM-извлечение избыточно.

Три приоритета на основе анализа prior art: **(1)** Добавить minimal temporal model — хотя бы `created_at` + `source` на каждый факт для аудита; **(2)** Реализовать strategy для context budget — PageRank-based salient node selection + configurable hop depth в explore/graph_summary; **(3)** Расширить entity resolution normalizer-ами для доменных типов (IP normalization, email canonicalization, fuzzy company name matching) — это differentiator, которого нет ни в одном конкуренте в таком декларативном виде.

Ближайшая академическая работа по духу — **AriGraph** (IJCAI 2025): инкрементальное построение KG из наблюдений агента с dual semantic+episodic memory. Ближайший инструментальный аналог — **Maltego**: entity merging по type+value, Transform-паттерн как аналог explore, entity type inheritance. STIX 2.1 SDO/SCO — готовый словарь node_type для security use cases.
