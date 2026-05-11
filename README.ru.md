# Hermes-Mneme

> 🇬🇧 Read in English: [README.md](README.md)

> Retrieval-based context engine для [Hermes Agent](https://github.com/NousResearch/hermes-agent) — замена дефолтного компрессора, осведомлённая о графе исполнения.

**Mneme** (Μνήμη — древнегреческая муза памяти) — это drop-in плагин-context-engine для Hermes. Он заменяет дефолтный lossy-компрессор на state-aware слой памяти, который сохраняет каждое событие, индексирует его для семантического поиска, отслеживает граф исполнения для причинно-следственных лукапов и собирает каждый промпт из микса протоколированных недавних ходов, найденного по retrieval контекста и состояния исполнения — всё в рамках токен-бюджета.

Естественно сочетается с [Mnemosyne](https://github.com/johnnykor82/mnemosyne) — своей мифологической матерью и отдельным плагином Hermes для кросс-сессионной памяти.

## Зачем

Дефолтный компрессор теряет детали, когда контекстное окно заполняется. LCM хранит всё, но восстанавливает контекст эвристически. Mneme хранит всё **и** использует retrieval + граф исполнения, чтобы собрать *минимально достаточный* контекст для каждого хода.

## Возможности

- **SQLite event store** с детерминированными `event_id` — повторная индексация идемпотентна между компрессиями и возобновлениями сессий.
- **Embedding index** через [sqlite-vec](https://github.com/asg017/sqlite-vec) (KNN) с Python-фоллбэком. Поддерживает любой OpenAI-совместимый эндпоинт эмбеддингов (по умолчанию локальный Jina-MLX, также Ollama, OpenAI и т.д.).
- **Session segmenter** — авто-определение границ тем через дрифт эмбеддингов; скользящий центроид для эволюционирующих тем.
- **Intent classifier** (CONTINUATION / SWITCH / NEW_TASK / CLARIFICATION) — детерминированный, без LLM в hot path.
- **Граф исполнения** — отслеживает рёбра `tool_call → tool_output → decision`; питает scoring через распространение зависимостей (Stage 7).
- **Опциональный reranker** — второй этап ранжирования через Cohere/Jina/BGE эндпоинты (LiteLLM работает).
- **Опциональное LLM-обогащение** — извлекает `open_loops`, `decisions`, `active_entities` каждые N ходов.
- **Memory tools для агента** — `context_search` (с кросс-сессионным режимом), `fetch_event`, `expand_context`, `get_execution_state`.
- **Observability** — per-turn JSONL trace + in-memory метрики (hit rate, dependency usage, fallback rate, segmentation count).

## Установка

```bash
git clone https://github.com/johnnykor82/hermes-mneme.git \
  ~/.hermes/plugins/hermes-mneme
cd ~/.hermes/plugins/hermes-mneme
./install.sh
hermes gateway restart
```

Установщик находит Hermes venv (по умолчанию `~/.hermes/hermes-agent/venv`), ставит Python-зависимости из `requirements.txt` и проверяет их. Переопределить путь к venv: `HERMES_VENV=...`.

Работает на **macOS** и **Linux** (Python 3.10+).

После рестарта посмотрите активацию:
```bash
tail -f ~/.hermes/logs/agent.log | grep -i mneme
```

Должно появиться: `Hermes-Mneme context engine loaded.`

## Конфигурация

У всех настроек разумные дефолты. Переопределить можно через:
- env-переменные: `HERMES_CTX_<KEY>` (например, `HERMES_CTX_PROTECTED_TAIL_TURNS=12`)
- `config.yaml` в директории плагина

Ключевые параметры:
- `active_window_tokens` / `context_window_usage_percent` — общий бюджет токенов
- `protected_tail_turns` — последние N ходов всегда включаются дословно
- `state_budget_ratio` / `retrieved_budget_ratio` — деление бюджета
- `dependency_max_depth` / `dependency_decay` — распространение по графу исполнения
- `reranker_enabled` + `reranker_endpoint` — второй этап ранжирования
- `llm_enrichment_enabled` — асинхронное обогащение состояния

Полная схема с inline RU+EN комментариями: [`config.py`](config.py).

## Архитектура

Компонентные deep-dive'ы — в [`docs/`](docs/):
- `store.py` — SQLite event store (идемпотентный re-ingest, session lineage)
- `index.py` — embedding index (sqlite-vec + фоллбэк)
- `segmenter.py` — drift-based сегментация
- `classifier.py` — intent signals (без LLM)
- `router.py` — конструирование запросов, retrieval, scoring (Stages 6–7)
- `prompt_builder.py` — соблюдение токен-бюджета
- `engine.py` — основной lifecycle (compress, on_session_start, …)
- `graph.py` — граф исполнения + распространение зависимостей
- `tools.py` — memory tools для агента

## Обновление

Когда на `main` появляются новые коммиты:

```bash
cd ~/.hermes/plugins/hermes-mneme
git pull
./install.sh              # переустановит зависимости, если они изменились
hermes gateway restart
```

Ваши runtime-данные (`db/plugin.db`, `trace.jsonl`) лежат в `.gitignore` и переживут обновление.

## Тесты

```bash
~/.hermes/hermes-agent/venv/bin/pytest tests/unit -q
```

## Вклад в разработку

Issues и pull requests приветствуются. Стандартный GitHub-flow:

1. **Issues** — открывайте issue с описанием проблемы или идеи фичи.
2. **Pull requests** — fork, branch, commit, push, открываете PR против `main`.

Перед PR:

- Проверьте, что изменение работает на **macOS** и **Linux**, если оно затрагивает `install.sh` или пути в файловой системе.
- Прогоните `pytest tests/unit -q`.
- Запустите `ruff check` для базовой проверки стиля.
- Держите коммиты сфокусированными — одна тема на коммит.

## Лицензия

[Apache-2.0](LICENSE)
