from __future__ import annotations

from collections import OrderedDict
import importlib.util
import os
from pathlib import Path

from .base import PluginAction, PluginContext, PluginResult, SubmergerPlugin
from .dictionary import DictionaryPlugin
from .explanation import PhraseExplanationPlugin
from .sentence_diagram import SentenceDiagramPlugin
from submerger.settings import LLMEndpointSettings


class PluginRegistry:
    def __init__(self, plugins: list[SubmergerPlugin] | None = None) -> None:
        self._plugins: OrderedDict[str, SubmergerPlugin] = OrderedDict()
        for plugin in plugins or []:
            self.register(plugin)

    def register(self, plugin: SubmergerPlugin) -> None:
        self._plugins[plugin.plugin_id] = plugin

    def actions_for_event(self, kind: str) -> list[PluginAction]:
        return [action for plugin in self._plugins.values() for action in plugin.actions if kind in action.event_kinds]

    def run(self, plugin_id: str, action: str, context: PluginContext) -> PluginResult:
        return self._plugins[plugin_id].run(action, context)


def create_default_registry(*, include_external: bool = True, llm_settings: LLMEndpointSettings | None = None) -> PluginRegistry:
    registry = PluginRegistry([
        DictionaryPlugin(),
        PhraseExplanationPlugin(settings=llm_settings),
        SentenceDiagramPlugin(settings=llm_settings),
    ])
    if include_external:
        for plugin in load_plugins_from_directory(default_plugin_dir()):
            registry.register(plugin)
    return registry


def default_plugin_dir() -> Path:
    configured = os.environ.get("SUBMERGER_PLUGIN_DIR")
    if configured:
        return Path(configured).expanduser()
    return Path.home() / ".local" / "share" / "submerger" / "plugins"


def load_plugins_from_directory(path: str | Path) -> list[SubmergerPlugin]:
    directory = Path(path).expanduser()
    if not directory.exists():
        return []

    plugins: list[SubmergerPlugin] = []
    for file_path in sorted(directory.glob("*.py")):
        plugin = load_plugin_file(file_path)
        if plugin is not None:
            plugins.append(plugin)
    return plugins


def load_plugin_file(path: Path) -> SubmergerPlugin | None:
    spec = importlib.util.spec_from_file_location(f"submerger_user_plugin_{path.stem}", path)
    if spec is None or spec.loader is None:
        return None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "create_plugin"):
        return module.create_plugin()
    return getattr(module, "plugin", None)
