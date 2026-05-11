"""Hermes-Mneme — retrieval-based context engine.

Replaces the default Hermes ContextCompressor with a state-aware
memory layer: persistent SQLite store + embedding index +
execution graph + intent classifier.
"""

import logging

logger = logging.getLogger(__name__)


def register(ctx):
    """Plugin entry point — register the Mneme context engine."""
    from .engine import CustomRouterContextEngine
    from . import config as config_module

    plugin_config = config_module.PluginConfig()
    engine = CustomRouterContextEngine(config=plugin_config)
    ctx.register_context_engine(engine)
    logger.info("Hermes-Mneme context engine loaded.")
