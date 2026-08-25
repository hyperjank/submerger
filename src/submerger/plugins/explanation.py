from __future__ import annotations

from submerger.interaction import LMStudioExplanationClient
from submerger.settings import LLMEndpointSettings

from .base import PluginAction, PluginContext, PluginResult


class PhraseExplanationPlugin:
    plugin_id = "phrase_explanation"
    name = "Phrase Explanation"
    actions = (
        PluginAction(plugin_id, "Explain Phrase", ("selection",)),
    )

    def __init__(
        self,
        client: LMStudioExplanationClient | None = None,
        settings: LLMEndpointSettings | None = None,
    ) -> None:
        self.client = client or LMStudioExplanationClient(settings=settings)

    def run(self, action: str, context: PluginContext) -> PluginResult:
        result = self.client.explain(context.interaction)
        return PluginResult(result.title, result.body, result.source)
