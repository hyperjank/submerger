import tempfile
import textwrap
import unittest
from pathlib import Path

from submerger.interaction import SubtitleInteraction
from submerger.plugins import PluginContext, create_default_registry
from submerger.plugins.registry import load_plugins_from_directory
from submerger.settings import LLMEndpointSettings
from submerger.plugins.sentence_diagram import (
    LLMSentenceDiagramClient,
    diagram_response_format,
    parse_diagram_response,
    render_diagram_html,
)


class PluginTests(unittest.TestCase):
    def test_default_registry_exposes_selection_plugins(self) -> None:
        registry = create_default_registry(include_external=False)

        labels = [action.label for action in registry.actions_for_event("selection")]

        self.assertIn("Explain Phrase", labels)
        self.assertIn("Diagram Segment", labels)

    def test_default_registry_exposes_real_dictionary_for_words(self) -> None:
        registry = create_default_registry(include_external=False)

        labels = [action.label for action in registry.actions_for_event("hover")]

        self.assertEqual(labels, ["Dictionary Lookup"])

    def test_sentence_diagram_plugin_renders_both_languages(self) -> None:
        def transport(_body):
            return """
            {
              "summary": "A phrasal verb maps to a result complement.",
              "primary": {
                "text": "I got through the call.",
                "units": [
                  {"id": "p1", "text": "I", "role": "subject", "gloss": "speaker"},
                  {"id": "p2", "text": "got through", "role": "predicate", "gloss": "successfully connected"},
                  {"id": "p3", "text": "the call", "role": "object", "gloss": "phone call"}
                ],
                "relations": [{"from": "p1", "to": "p2", "label": "does"}]
              },
              "secondary": {
                "text": "我打通了电话。",
                "units": [
                  {"id": "s1", "text": "我", "role": "subject", "gloss": "I"},
                  {"id": "s2", "text": "打通了", "role": "predicate", "gloss": "got through"},
                  {"id": "s3", "text": "电话", "role": "object", "gloss": "call"}
                ],
                "relations": [{"from": "s1", "to": "s2", "label": "does"}]
              },
              "links": [{"from": "p2", "to": "s2", "label": "same action"}]
            }
            """

        client = LLMSentenceDiagramClient(
            settings=LLMEndpointSettings(provider="lmstudio"),
            model="diagram-model",
            transport=transport,
        )
        registry = create_default_registry(include_external=False)
        registry._plugins["sentence_diagram"].client = client
        context = PluginContext(
            interaction=SubtitleInteraction(kind="selection", text="I got through", language="primary"),
            primary_text="I got through the call.",
            secondary_text="我打通了电话。",
            timestamp=12.0,
        )

        result = registry.run("sentence_diagram", "Diagram Segment", context)

        self.assertEqual(result.content_type, "html")
        self.assertIn("Primary", result.body)
        self.assertIn("Secondary", result.body)
        self.assertIn("Meaning Links", result.body)
        self.assertEqual(result.source, "lmstudio:diagram-model")

    def test_parse_and_render_diagram_response(self) -> None:
        context = PluginContext(
            interaction=SubtitleInteraction(kind="selection", text="make it", language="primary"),
            primary_text="If you make it, you win two VIP passes.",
            secondary_text="如果你投进，就赢两张贵宾票。",
            timestamp=8.0,
        )
        diagram = parse_diagram_response(
            """
            <think>hidden</think>
            {"summary":"Condition plus result.","primary":{"text":"If you make it, you win two VIP passes.","units":[{"id":"p1","text":"If you make it","role":"condition","gloss":"condition"},{"id":"p2","text":"you win two VIP passes","role":"result","gloss":"result"}],"relations":[{"from":"p1","to":"p2","label":"enables"}]},"secondary":{"text":"如果你投进，就赢两张贵宾票。","units":[{"id":"s1","text":"如果你投进","role":"condition","gloss":"if you make it"},{"id":"s2","text":"就赢两张贵宾票","role":"result","gloss":"win two VIP passes"}],"relations":[{"from":"s1","to":"s2","label":"enables"}]},"links":[{"from":"p2","to":"s2","label":"same result"}]}
            """,
            context,
        )
        html = render_diagram_html(diagram)

        self.assertIn("Condition plus result.", html)
        self.assertIn("role-condition", html)
        self.assertIn("same result", html)

    def test_diagram_response_format_requires_structured_output(self) -> None:
        schema = diagram_response_format()["json_schema"]["schema"]

        self.assertEqual(schema["required"], ["summary", "primary", "secondary", "links"])

    def test_load_plugins_from_user_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            plugin_file = Path(tmp) / "sample_plugin.py"
            plugin_file.write_text(
                textwrap.dedent(
                    """
                    from submerger.plugins.base import PluginAction, PluginResult

                    class SamplePlugin:
                        plugin_id = "sample"
                        name = "Sample"
                        actions = (PluginAction(plugin_id, "Sample Action", ("selection",)),)

                        def run(self, action, context):
                            return PluginResult("Sample", context.primary_text, "sample")

                    def create_plugin():
                        return SamplePlugin()
                    """
                ),
                encoding="utf-8",
            )

            plugins = load_plugins_from_directory(tmp)

            self.assertEqual(len(plugins), 1)
            self.assertEqual(plugins[0].plugin_id, "sample")


if __name__ == "__main__":
    unittest.main()
