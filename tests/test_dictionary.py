import tempfile
import unittest
from pathlib import Path

from submerger.interaction import SubtitleInteraction
from submerger.plugins.base import PluginContext
from submerger.plugins.dictionary import (
    DictionaryCache,
    DictionaryPlugin,
    WiktionaryClient,
    html_to_text,
    lookup_language,
    parse_wiktionary_entry,
)


ENGLISH_PAYLOAD = {
    "en": [
        {
            "language": "English",
            "partOfSpeech": "Noun",
            "definitions": [
                {
                    "definition": 'A team <a href="/wiki/sport">sport</a> played with sticks.',
                    "examples": ["They played <b>hockey</b> after school."],
                },
                {"definition": "A family of related games."},
            ],
        }
    ]
}

CHINESE_PAYLOAD = {
    "zh": [
        {
            "language": "Chinese",
            "partOfSpeech": "Interjection",
            "definitions": [{"definition": '<a href="/wiki/hello">hello</a>; hi'}],
        }
    ]
}


def context_for(term: str, *, role: str = "primary", paired: str = "冰球") -> PluginContext:
    return PluginContext(
        interaction=SubtitleInteraction("click", term, role, paired_text=paired),
        primary_text=term if role == "primary" else paired,
        secondary_text=paired if role == "primary" else term,
        primary_language="en",
        secondary_language="zh",
    )


class DictionaryTests(unittest.TestCase):
    def test_parser_strips_wiktionary_markup_and_keeps_examples(self) -> None:
        entry = parse_wiktionary_entry("hockey", "en", ENGLISH_PAYLOAD)

        self.assertEqual(entry.sections[0].language, "English")
        self.assertEqual(entry.sections[0].part_of_speech, "Noun")
        self.assertEqual(entry.sections[0].senses[0].definition, "A team sport played with sticks.")
        self.assertEqual(entry.sections[0].senses[0].example, "They played hockey after school.")

    def test_chinese_lookup_uses_chinese_section(self) -> None:
        entry = parse_wiktionary_entry("你好", "zh", CHINESE_PAYLOAD)

        self.assertEqual(entry.sections[0].language, "Chinese")
        self.assertEqual(entry.sections[0].senses[0].definition, "hello; hi")
        self.assertEqual(lookup_language(context_for("你好", role="secondary"), "你好"), "zh")

    def test_plugin_caches_lookup_and_renders_paired_subtitle(self) -> None:
        calls = []

        def transport(term, language):
            calls.append((term, language))
            return ENGLISH_PAYLOAD

        with tempfile.TemporaryDirectory() as tmp:
            cache = DictionaryCache(Path(tmp) / "dictionary.json")
            plugin = DictionaryPlugin(
                client=WiktionaryClient(transport=transport),
                cache=cache,
            )

            first = plugin.run("Dictionary Lookup", context_for("Hockey!"))
            second = plugin.run("Dictionary Lookup", context_for("hockey"))

        self.assertEqual(calls, [("hockey", "en")])
        self.assertEqual(first.content_type, "html")
        self.assertIn("team sport", html_to_text(first.body))
        self.assertIn("冰球", html_to_text(first.body))
        self.assertEqual(first.body, second.body)

    def test_plugin_returns_friendly_offline_result(self) -> None:
        def unavailable(_term, _language):
            raise RuntimeError("offline")

        with tempfile.TemporaryDirectory() as tmp:
            plugin = DictionaryPlugin(
                client=WiktionaryClient(transport=unavailable),
                cache=DictionaryCache(Path(tmp) / "dictionary.json"),
            )
            result = plugin.run("Dictionary Lookup", context_for("hockey"))

        self.assertIn("offline", result.body)
        self.assertIn("cached entries", result.body)

    def test_plugin_still_returns_lookup_when_cache_write_fails(self) -> None:
        class UnwritableCache:
            def get(self, _language, _term):
                return None

            def put(self, _entry):
                raise OSError("read-only filesystem")

        plugin = DictionaryPlugin(
            client=WiktionaryClient(transport=lambda _term, _language: ENGLISH_PAYLOAD),
            cache=UnwritableCache(),
        )

        result = plugin.run("Dictionary Lookup", context_for("hockey"))

        self.assertIn("team sport", html_to_text(result.body))


if __name__ == "__main__":
    unittest.main()
