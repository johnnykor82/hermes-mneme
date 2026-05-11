import os
import yaml
from typing import Any, Dict, Optional

# Default configuration
DEFAULT_CONFIG = {
    # Context window budget — EITHER absolute OR percentage, never both.
    # Absolute (priority): set active_window_tokens to a positive integer (e.g. 60000).
    #   If set, this value is used directly as the total context budget.
    # Percentage (fallback): set active_window_tokens to null/0 and use context_window_usage_percent.
    #   Budget = model_context_length * context_window_usage_percent.
    #   Keeps (100 - percent)% as headroom for tool results and agent replies.
    # ------------------------------------------------------------------
    # RU: Бюджет окна контекста — ЛИБО абсолютное число, ЛИБО процент, не оба.
    #   Абсолютный (приоритет): active_window_tokens = положительное число (напр. 60000) →
    #   используется напрямую как полный бюджет.
    #   Процентный (если absolute=0/null): бюджет = model_context_length * context_window_usage_percent.
    #   Оставшиеся (100-percent)% — запас под ответ модели и tool-результаты.
    "active_window_tokens": 0,               # 0 means "use percentage mode" / 0 = режим процента
    "context_window_usage_percent": 0.70,    # 70% of model context window / 70% от окна модели
    "protected_tail_turns": 64,              # Last N turns always included / последние N turn-ов всегда в промпте

    # Budget ratios (must sum to <= 1.0; remainder is headroom)
    # RU: Доли бюджета (сумма ≤ 1.0; остаток — запас).
    "state_budget_ratio": 0.05,              # 5% for execution state / 5% под execution state (goal, last_tool, open_loops)
    "retrieved_budget_ratio": 0.30,          # 30% for retrieved context / 30% под семантически-найденные чанки
    "protected_tail_ratio": 0.55,            # 55% for protected tail / 55% под последние N turn-ов
    # Headroom = 1.0 - 0.05 - 0.30 - 0.55 = 0.10 (запас под current_msg
    # и неточность tokenizer'а; spec, "Dynamic system prompt deduction").
    # RU: Headroom 10% — под current_user_message и tokenizer-погрешность.

    # Tokenizer
    # RU: Токенайзер для подсчёта длины сообщений.
    "token_counter": "tiktoken",
    "tokenizer_model": "cl100k_base",

    # Embedding
    # RU: Эмбеддинг-провайдер. Локальный Jina-совместимый сервер на 8000.
    "embedding_provider": "jina_compatible",
    "embedding_model": "jina-embeddings-v5-text-small-retrieval-mlx",
    "embedding_endpoint": "http://127.0.0.1:8000",
    "embedding_api_key": "1234",

    # Segmentation
    # RU: Сегментация сессии — детектит смену темы и режет сессию на сегменты.
    "segmentation_enabled": True,
    "drift_threshold": 0.35,                 # Порог дрейфа эмбеддинга для границы сегмента
    "drift_weights": [0.4, 0.3, 0.3],        # Веса сигналов: similarity / entity / explicit-switch
    # Centroid cache: how many segment centroids to keep in memory (LRU).
    # RU: Кеш центроидов: сколько центроидов сегментов держать в памяти (LRU).
    #   Один центроид ≈ 4 KB. 100 → 400 KB, 1000 → 4 MB. Поднимай если у тебя
    #   очень много активных сегментов и часто прыгаешь между ними.
    "centroid_cache_size": 100,
    # Sliding-window centroid: compute centroid over the last N embeddings of
    # the segment instead of all events. Lets the centroid drift as the topic
    # naturally evolves, while segment_id stays stable so semantic search
    # returns the WHOLE topic, not just recent slice.
    # 0 = use all segment events (no window).
    # RU: Скользящее окно центроида: считаем центроид по последним N эмбеддингам
    #   сегмента, а не по всем. Центроид плавно «скользит» вместе с темой,
    #   но segment_id не меняется — семантический поиск находит ВСЮ тему.
    #   Резкая смена темы всё равно ловится drift-порогом и режет сегмент.
    #   0 = по всем событиям сегмента (окна нет).
    "centroid_window": 200,

    # Execution state
    # RU: Обогащение state через LLM. Когда включено — раз в N turn-ов и/или
    #   на сегмент-границе вызывается LLM, чтобы извлечь intent_label,
    #   topic_tags, decisions, open_loops, active_entities из последних N turn-ов.
    #   Если endpoint/model/api_key пустые — плагин использует параметры
    #   текущей LLM-модели Hermes (то что Hermes передал в update_model).
    "llm_enrichment_enabled": True,
    "delta_extraction_enabled": False,
    "enricher_endpoint": "",                # пусто → берём от Hermes
    "enricher_model": "",                   # пусто → берём от Hermes
    "enricher_api_key": "",                 # пусто → берём от Hermes
    "enricher_every_n_turns": 5,            # вызывать каждые N turn-ов
    "enricher_on_segment_boundary": True,   # дополнительно при смене сегмента
    "enricher_max_history_turns": 10,       # сколько последних turn-ов отдавать LLM
    "enricher_timeout_seconds": 30,         # таймаут запроса; при отвале — fallback на старое значение
    # Cap on LLM completion tokens for the enrichment JSON response.
    # RU: Лимит токенов на ответ enricher-LLM. Слишком маленький → ответ
    #     обрезается, в логи летят "recovered partial JSON" сообщения; tier-3
    #     парсер восстанавливает intent_label / topic_tags, но decisions
    #     теряются. Подними если видишь частые recovery-логи.
    "enricher_max_tokens": 1500,
    # Initial estimate of the per-turn prompt overhead — bytes that the
    # plugin's content-only sum doesn't see (system prompt + tool schemas +
    # tool_call/tool_output JSON wrappers + reasoning). The pass-through
    # guard uses this so it doesn't tell Hermes "you can pass-through" when
    # the real prompt would be 30-40k tokens larger than `content_tokens`.
    # Recalibrated upwards on every update_from_response with the actually
    # observed delta; this default just protects the first compress() call.
    # RU: Стартовая оценка накладных расходов prompt-а (system prompt + tool
    #     schemas + JSON-обёртки). Pass-through guard прибавляет её к
    #     content_tokens, чтобы не пропускать буфер, который реально вылезет
    #     за бюджет. После первого ответа LLM значение перевычисляется по
    #     реальному prompt_tokens.
    "pass_through_overhead_initial": 16000,

    # Compression (for tool outputs)
    # RU: Сжатие выводов tool-call-ов: если больше threshold токенов — суммаризировать до summary_tokens.
    "tool_output_compress_threshold_tokens": 500,
    "tool_output_summary_tokens": 100,
    "reindex_on_model_change": False,        # Переиндексировать всё при смене модели эмбеддинга

    # Retrieval
    # router_top_k: max candidates returned by KNN before scoring/budget cut.
    #   0 (or missing) = unlimited — return everything semantically close,
    #   prompt_builder will trim by retrieved_budget anyway.
    #   Set to a positive integer to cap (e.g. 100, 200).
    # RU: router_top_k — макс. количество кандидатов после KNN до скоринга и бюджет-cut.
    #   0 / отсутствует = без ограничения (вернёт всё семантически близкое,
    #   prompt_builder всё равно обрежет по retrieved_budget).
    #   Положительное число = жёсткий cap (напр. 100, 200).
    "router_top_k": 0,
    # Cross-segment fallback threshold: if the per-segment KNN returns fewer
    # than this many candidates (cold start, drift-shredded sessions, post-
    # RESUME), the router does a second pass with segment_id="all" within
    # the current session and unions the results. Same threshold drives the
    # post-dedup fallback in engine.compress(): if dedup against
    # protected_tail leaves fewer than N candidates, we widen the search.
    # 0 disables the fallback entirely.
    # RU: Порог для cross-segment фолбэка. Если по текущему сегменту KNN
    #   нашёл меньше N кандидатов (или dedup-против protected_tail оставил
    #   меньше N), плагин делает второй проход по всей сессии. 0 = выкл.
    "router_min_candidates": 12,
    # Reranker (optional, off by default).
    #   When enabled, scored candidates are passed through reranker_endpoint
    #   for second-stage ranking. Endpoint must accept POST {query, documents:[...]}
    #   and return {scores: [...]} aligned with documents order.
    # RU: Реранкер (опционально). При enabled=true кандидаты после скоринга идут
    #   во второй этап ранжирования через reranker_endpoint. Поддерживаются Cohere/BGE
    #   ({results:[{index, relevance_score}]}) и Jina ({scores:[...]}) форматы.
    "reranker_enabled": True,
    "reranker_endpoint": "http://127.0.0.1:4000/v1/rerank",
    "reranker_model": "rerank",
    "reranker_api_key": "sk-local-litellm",
    "reranker_top_k": 0,                     # 0 = keep all after rerank / 0 = оставить все после rerank

    # Dependency propagation (Stage 7).
    # Боним кандидатам, лежащим в графе выполнения близко к точке отсчёта
    # (последнему tool_output / assistant_message). Чем ближе по графу — тем
    # выше бонус: bonus = decay ** depth. depth=0 → 1.0, decay=0.6, depth=2 → 0.36.
    # max_depth = 0 отключает propagation.
    # RU: Распространение по графу зависимостей. Cобытия в причинно-следственной
    #   цепочке текущего сообщения получают бонус к dependency-score.
    "dependency_max_depth": 4,
    "dependency_decay": 0.6,

    # Observability
    # RU: Логирование трасс ретривала в trace.jsonl. Авто-ротация при превышении.
    "debug_mode": False,
    "trace_log_max_mb": 10,                  # Размер файла трассы до ротации (МБ)
    "trace_log_max_rotations": 3,            # Сколько ротированных файлов хранить

    # Memory navigation hints (Stage 8: agent-side memory tools).
    # RU: Подсказки агенту о наличии БД и графа. См. plan sunny-forging-haven.
    "memory_access_hint_enabled": True,      # инжектить [MEMORY ACCESS] в system msg
    "goal_trail_size": 3,                    # сколько последних уникальных целей в [GOAL TRAIL]
    "checkpoint_after_n_memory_calls": 5,    # 0 = выкл; иначе подмешивать [CHECKPOINT]
    "memory_tool_names": [
        "context_search", "fetch_event", "expand_context",
        "list_segments", "get_goal_history",
    ],
}

PLUGIN_DIR = os.path.dirname(os.path.abspath(__file__))

class PluginConfig:
    """Configuration manager for the context engine plugin."""

    def __init__(self, config_dict: Optional[Dict] = None):
        self._config = DEFAULT_CONFIG.copy()
        if config_dict:
            self._config.update(config_dict)
        self._load_from_env()
        self._load_from_yaml()

    def _load_from_env(self):
        """Load config from HERMES_CTX_* environment variables."""
        prefix = "HERMES_CTX_"
        for key in self._config.keys():
            env_key = prefix + key.upper()
            value = os.environ.get(env_key)
            if value is not None:
                # Convert type
                if isinstance(self._config[key], bool):
                    self._config[key] = value.lower() in ("1", "true", "yes", "on")
                elif isinstance(self._config[key], int):
                    self._config[key] = int(value)
                elif isinstance(self._config[key], float):
                    self._config[key] = float(value)
                elif isinstance(self._config[key], list):
                    # Simple CSV parse
                    self._config[key] = [float(x.strip()) for x in value.split(",")]
                else:
                    self._config[key] = value

    def _load_from_yaml(self):
        """Load config from config.yaml in the plugin directory."""
        yaml_path = os.path.join(PLUGIN_DIR, "config.yaml")
        if os.path.exists(yaml_path):
            try:
                with open(yaml_path, "r") as f:
                    user_config = yaml.safe_load(f)
                    if user_config:
                        self._config.update(user_config)
            except Exception as e:
                logging.getLogger(__name__).error(f"Failed to load YAML config: {e}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._config.get(key, default)

    def __getattr__(self, name: str) -> Any:
        if name in self._config:
            return self._config[name]
        raise AttributeError(f"'PluginConfig' object has no attribute '{name}'")

    @property
    def as_dict(self) -> Dict:
        return self._config.copy()
