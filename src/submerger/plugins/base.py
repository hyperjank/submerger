from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from submerger.interaction import SubtitleInteraction


@dataclass(frozen=True)
class PluginResult:
    title: str
    body: str
    source: str
    content_type: str = "text"


@dataclass(frozen=True)
class PluginContext:
    interaction: SubtitleInteraction
    primary_text: str
    secondary_text: str
    timestamp: float | None = None


@dataclass(frozen=True)
class PluginAction:
    plugin_id: str
    label: str
    event_kinds: tuple[str, ...]


class SubmergerPlugin(Protocol):
    plugin_id: str
    name: str
    actions: tuple[PluginAction, ...]

    def run(self, action: str, context: PluginContext) -> PluginResult:
        ...
