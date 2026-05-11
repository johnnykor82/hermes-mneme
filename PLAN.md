# Custom Router Context Engine — план доработки до полной спеки

**Создан**: 2026-05-07
**Расположение**: `~/.hermes/hermes-agent/plugins/context_engine/custom_router/PLAN.md`
**Статус плагина на старте**: B1-B13 + Q1-Q4 + 4 spec-override v1.1 применены и работают в production. Все «функциональные дыры» из спеки v1.0 ниже.

## Как пользоваться этим документом

При обрыве сессии — укажи Claude путь к этому файлу. Он:
1. Прочитает раздел **«Текущий этап»** ниже.
2. Прочитает раздел **«Лог прогресса»** (что уже сделано).
3. Возобновит работу с текущего этапа без переспрашивания.

Claude **обязан** обновлять разделы «Текущий этап» и «Лог прогресса» после каждого завершённого этапа.

---

## Текущий этап

**Этап**: Stage B (memory navigation) — DONE 2026-05-09. Версия 0.3.0.
**Статус**: следующая работа не запланирована. Возможные направления —
Этап 6 (question-about-output), Этап 7 (dependency propagation), либо
опциональный шаг "индекс упоминаний по сущностям" — оставлен на случай
если шаги 1–3 не закроют боль.
**Следующее действие**: ждём отзыва от пользователя на работающую
сборку 0.3.0 (длинная сессия, проверка memory-навигации в `agent.log`).

---

## Этапы

Все пути относительно `~/.hermes/hermes-agent/plugins/context_engine/custom_router/`.

### Этап 1 — Tool output compression
**Спека**: Component 1, конфиг `tool_output_compress_threshold_tokens=500`, `tool_output_summary_tokens=100`.
**Объём**: 2-3 часа.
**Зависимости**: нет.
**Риск**: низкий.

**Что сделать**:
- В `engine.py::compress()` (или в `parser.py`) при создании события `tool_output`: если `token_estimate > tool_output_compress_threshold_tokens` — сохранить полный текст в `events.content`, но дополнительно сгенерировать summary (head + tail, ~`tool_output_summary_tokens`) и индексировать **summary**, а не оригинал.
- Альтернатива: добавить колонку `events.content_summary` и индексировать summary, при `fetch_event` отдавать оригинал.
- Решение по архитектуре зафиксировать в комментарии перед реализацией.

**Smoke test**:
- 1-2 turn-а с tool, возвращающим >2000 токенов (например, `Read` большого файла).
- Проверить размер `embedding_index.embedding` BLOB и что в trace.jsonl попадает summary.

**Файлы**: `engine.py`, возможно `parser.py`, `store.py`.

---

### Этап 2 — Retrieval mode detection
**Спека**: Component 6, `MODE_WEIGHTS` (general/reasoning/factual/debugging).
**Объём**: 3-5 часов.
**Зависимости**: нет.
**Риск**: низкий.

**Что сделать**:
- В `router.py::_get_retrieval_mode()` — заменить `return 'general'` на эвристику:
  - `debugging` если `last_tool` ∈ {Read, Bash, Grep} И в сообщении есть ключевые слова (error, fail, traceback, не работает, ошибка, debug, не запускается);
  - `factual` если intent = `INTENT_QUESTION` (классификатор должен быть расширен — см. ниже);
  - `reasoning` если сообщение длинное (>500 символов) и содержит вопросительные конструкции «почему / зачем / как лучше / что если»;
  - default: `general`.
- Расширить `classifier.py` если нужно — добавить `INTENT_QUESTION`.

**Smoke test**:
- 5-10 turn-ов разного типа.
- В `trace.jsonl` поле `signals` или новое поле `mode` должно меняться.

**Файлы**: `router.py`, `classifier.py`.

---

### Этап 3 — Embedding drift score + drift weights composition
**Спека**: Component 4, `drift_threshold=0.35`, `drift_weights=[0.4, 0.3, 0.3]`.
**Объём**: 4-6 часов.
**Зависимости**: нет.
**Риск**: средний (можно случайно поломать сегментацию — нужны тесты сегмент-границ).

**Что сделать**:
- В `segmenter.py`: вычислять центроид сегмента (среднее эмбеддингов всех событий сегмента), кэшировать в памяти.
- На каждое user-сообщение считать `cosine(message_embedding, centroid)` → `1 - cosine` = drift.
- Композиция трёх сигналов: `weighted = w0*embedding_drift + w1*entity_drift + w2*explicit_switch`. Сейчас `entity_drift` — заглушка (см. Этап 5), временно использовать `0.0` или `classifier.classify_entity_contradiction`.
- Прокинуть `embedding_drift` в `router.py::_get_embedding_drift_score()`.

**Smoke test**:
- Сессия с резкой сменой темы (5 сообщений про Python → 5 про кулинарию).
- В `trace.jsonl` `signals.embedding_drift` должен быть >0 на границе.
- В DB `SELECT DISTINCT segment_id FROM events WHERE session_id=...` — должно быть 2+ сегмента.

**Файлы**: `segmenter.py`, `router.py`.

---

### Этап 4 — Reindex on model change
**Спека**: конфиг `reindex_on_model_change`.
**Объём**: 2 часа.
**Зависимости**: нет.
**Риск**: низкий.

**Что сделать**:
- В `engine.py::__init__` или `on_session_start`: `SELECT DISTINCT embedding_model_id FROM embedding_index`.
- Если в БД есть другая модель И `reindex_on_model_change=True`:
  - Дропнуть `embedding_index` и `vec_items`/`vec_items_meta` для старой модели.
  - Перебрать все `events` с непустым `content`, переэмбеддить под новой моделью.
  - Прогресс логировать каждые 100 событий.
- Если `reindex_on_model_change=False` — только warning «embedding model changed, retrieval may degrade».

**Smoke test**:
- Поменять `embedding_model` в `config.yaml` → restart → в логах должно быть `Reindexing N events under new model X`.

**Файлы**: `engine.py`, возможно `index.py`.

---

### Этап 5 — LLM enrichment + open_loops/decision_stack/active_entities
**Спека**: Component 3, флаги `llm_enrichment_enabled`, `delta_extraction_enabled`.
**Объём**: 1-1.5 дня. Разбить на саб-этапы.
**Зависимости**: нет (но Этап 6 ждёт его).
**Риск**: высокий — LLM-промпт нужно итерировать.

**Саб-этапы**:

**5.1 — Базовый enrichment skeleton (3-4ч)**
- Новый модуль `enrichment.py` с классом `LLMEnricher`.
- Метод `enrich(state, recent_events) -> EnrichmentResult` с полями `intent_label`, `topic_tags`, `decision_summary`.
- Локальный LLM endpoint (тот же что и Jina? нужно уточнить — скорее всего отдельный LiteLLM).
- Промпт-template на extract JSON из последних 10 turn-ов.
- Парсинг JSON с fallback при невалидном ответе.
- Кеширование: вызывать только при сегмент-границе ИЛИ раз в N=5 turn-ов.

**5.2 — open_loops + decision_stack (2-3ч)**
- Расширить промпт: `open_loops: [строки с незавершёнными действиями]`, `decisions: [{decision, rationale}]`.
- Запись в `state["open_loops"]`, `state["decision_stack"]` (FIFO, max 5 элементов).
- В сегментаторе: `decision_stack` сбрасывается на новый сегмент, `open_loops` переносятся.

**5.3 — active_entities (NER) (3-4ч)**
- Промпт: `entities: [{name, type, last_mentioned_event_id}]`.
- Запись в `state["active_entities"]` (max 10, LRU).
- Хранить `last_assistant_entities` отдельно (для Этапа 6).

**Smoke test всего Этапа 5**:
- 2-3 длинные сессии (50+ turn-ов).
- Subjective review: `state.enrichment.intent_label` адекватен реальной задаче?
- `decision_stack` содержит реальные решения?
- `active_entities` отражают текущие сущности?

**Файлы**: новый `enrichment.py`, `engine.py`, `segmenter.py`.

---

### Этап 6 — Question-about-output classifier
**Спека**: Component 6, `classify_question_about_output`.
**Объём**: 3-5 часов.
**Зависимости**: Этап 5 (нужен `last_assistant_entities`).
**Риск**: низкий.

**Что сделать**:
- В `classifier.py::classify_question_about_output`: реальный код вместо упрощённого.
- Эвристика: если в user-message есть pronouns («он», «это», «то», «which one», «that») И есть пересечение с `last_assistant_entities` → True.
- Прокинуть `last_assistant_entities` через `RouterInput` (новое поле).

**Smoke test**:
- Сессия: "перечисли пакеты" → assistant отвечает → "а почему второй так называется?"
- В `trace.jsonl` `signals.question_about_output` должен быть True.

**Файлы**: `classifier.py`, `router.py`, `engine.py`.

---

### Этап 7 — Dependency propagation
**Спека**: Component 5+10, Phase 10.
**Объём**: 1 день.
**Зависимости**: нет (но рискованно делать первым).
**Риск**: высокий — может неожиданно поменять ranking, нужны A/B сравнения.

**Что сделать**:
- В `router.py::_calculate_dependency_score`: убрать `return 0.0`.
- BFS по `execution_graph` от seed-events (events с высоким similarity).
- Score propagation: `dep_score(neighbor) = base_score * decay^depth`, `decay=0.7`, max depth 3.
- Веса по типам рёбер: `tool_output: 1.0`, `decision: 0.8`.
- Восстановить веса в `MODE_WEIGHTS`: вернуть `dependency: 0.1-0.2` (как в спеке v1.0), уменьшив `similarity` обратно на эту дельту.

**Smoke test (тяжёлый)**:
- Снять snapshot trace.jsonl до изменений.
- Применить.
- Прогнать 20-30 turn-ов на той же реальной задаче.
- Сравнить top-5 кандидатов до/после: должно быть улучшение в кейсах «вспомни tool-вывод после которого ты решил X».
- Если retrieval становится хаотичным — откатить или уменьшить веса.

**Файлы**: `router.py`, возможно `graph.py`.

---

## Лог прогресса

Формат: `[YYYY-MM-DD HH:MM] Этап N — статус — заметка`

```
[2026-05-07] Создан план. Старт ожидается.
[2026-05-07] Этап 1 — DONE. Создан compressor.py (head+tail summary, deterministic, без LLM). engine.py: при индексации tool_output > tool_output_compress_threshold_tokens применяется summarize_for_embedding. Оригинал в events.content не трогается. Smoke-тест OK: 4000 chars → 430 chars summary с маркером "truncated". Кеш почищен, gateway пользователь рестартует сам перед боевой проверкой.
[2026-05-07] Этап 2 — DONE. router._get_retrieval_mode реализована эвристика: debugging (error/traceback/ошибк/last_tool в Read/Bash/Grep+негатив), reasoning (почему/why/что если или len>500), factual (what is/что такое/перечисли), default general. RouterResult.mode прокинут в observability.log_turn (новое поле "mode" в trace.jsonl). _score_candidates принимает mode явно. Smoke-тест 7 кейсов — все OK.
[2026-05-07] Этап 3 — DONE. segmenter.py переписан: get_centroid с LRU-кешем (max 100 segments, инвалидация по event_count); calculate_embedding_drift = (1-cosine)/2 ∈ [0,1]; cold-start (<3 событий) → 0.0; SEGMENT_HARD_CAP=200; композиция w0*emb + w1*entity + w2*explicit. router._get_embedding_drift_score вызывает segmenter; engine передаёт segmenter в router. Smoke-тест: близкое drift=0.0, далёкое 0.73, cold start 0.0, cache hit OK.
[2026-05-07] Этап 3 — ENHANCEMENT. Hard-cap (200 событий → принудительный cut) удалён — терял id темы. Заменён на скользящее окно центроида: centroid считается по последним N эмбеддингам сегмента (config: centroid_window=200, 0=без окна). Segment_id стабилен → семантика возвращает всю тему, drift следует за активной частью. LRU-кеш вынесен в config (centroid_cache_size=100, динамический resize). Fingerprint кеша: (total_event_count, window) — новое событие или смена окна инвалидирует. SEGMENT_LARGE_WARN=300: одноразовое предупреждение в лог. Smoke-тест 5 кейсов OK.
[2026-05-07] Этап 4 — DONE. engine.__init__ вызывает _check_and_reindex_embeddings: SELECT DISTINCT embedding_model_id FROM embedding_index. Если есть устаревшая модель И reindex_on_model_change=True → _reindex_under_model(): удаляет vec_items/vec_items_meta/embedding_index по stale-моделям, переэмбеддит все events с непустым content под текущей моделью, прогресс каждые 100 событий. tool_output применяет compressor.summarize_for_embedding. Если flag=False — только warning. Smoke-тест 3 кейса (warn-only / reindex / no-op) OK.
[2026-05-08] Stage A (semantic tail) — DONE. После первой компрессии retrieval возвращал [] несколько ходов, т.к. index.search фильтровал только по текущему session_id. Починка в 5 файлов: store.get_latest_compressed_session, index.search принимает список session_id и OR-фильтрует KNN, router._route резолвит lineage раз в ход, engine префетчит lineage на старте, tools._context_search использует тот же lineage при session='current'. Подробности — docs/DIVERGENCE.md (Stage A). Версия 0.2.0.
[2026-05-09] Stage B (memory navigation) — DONE. Граф остаётся в БД, агент ходит в него инструментами. Сделано: (1) таблица state_history + append после save_state; (2) новые tools list_segments и get_goal_history; (3) expand_context.mode='segment' + store.get_segment_skeleton; (4) fetch_event с truncate-по-умолчанию + флаг full; (5) prompt_builder инжектит [MEMORY ACCESS] / [GOAL TRAIL] / [EXECUTION STATE] / [CHECKPOINT]; (6) loop-guard: счётчик _consecutive_memory_calls, при N подряд memory-вызовах подмешивается [CHECKPOINT] на следующий ход; (7) 4 новых config-флага. Подробности — docs/DIVERGENCE.md (Stage B). Версия 0.3.0.
```

---

## Восстановление контекста после обрыва сессии

Если сессия Claude оборвалась:

1. **Открой этот файл** — `cat ~/.hermes/hermes-agent/plugins/context_engine/custom_router/PLAN.md`
2. **Скажи Claude**: «Продолжаем с PLAN.md» (или укажи путь явно).
3. Claude должен:
   - Прочитать раздел «Текущий этап».
   - Прочитать «Лог прогресса» — посмотреть что уже сделано.
   - Прочитать описание текущего этапа.
   - Если этап начат частично — проверить состояние файлов через `git diff` (если репо инициализирован) или прочитать ключевые файлы этапа.
   - Возобновить работу.

## Если что-то идёт не так

- **БД испорчена**: `rm -f db/plugin.db*` → restart gateway → схема пересоздастся.
- **Кеш**: `find ~/.hermes -name __pycache__ -path "*custom_router*" -exec rm -rf {} +`.
- **Откат этапа**: каждый этап коммитить отдельно (если репо есть) — `git revert <commit>`.
- **Логи**: `~/.hermes/logs/agent.log`, `~/.hermes/hermes-agent/plugins/context_engine/custom_router/trace.jsonl`.

## Что НЕ делаем в рамках этого плана

- Не трогаем Hermes core.
- Не меняем embedding model по своему усмотрению (только если пользователь явно просит).
- Не публикуем на GitHub (отдельная задача после завершения всех этапов).
- Не добавляем новых компонентов вне 7 этапов выше.
