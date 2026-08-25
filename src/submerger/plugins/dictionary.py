from __future__ import annotations

from submerger.interaction import normalize_lookup_text

from .base import PluginAction, PluginContext, PluginResult


class DictionaryPlugin:
    plugin_id = "dictionary"
    name = "Dictionary"
    actions = (
        PluginAction(plugin_id, "Dictionary Lookup", ("hover", "click")),
    )

    def __init__(self) -> None:
        self.entries = {
            "hockey": "A team sport played on ice with sticks and a puck.",
            "flyer": "A member of the Philadelphia Flyers hockey team in this context.",
            "coming": "Present participle of come; often part of a phrasal verb or future construction.",
            "back": "Return to a previous place, topic, or state.",
        }

    def run(self, action: str, context: PluginContext) -> PluginResult:
        key = normalize_lookup_text(context.interaction.text)
        body = self.entries.get(key, f"No local dictionary entry yet for '{context.interaction.text}'.")
        if context.secondary_text:
            body += f"\n\nPaired subtitle:\n{context.secondary_text}"
        return PluginResult(f"Dictionary: {context.interaction.text}", body, "plugin:dictionary")
