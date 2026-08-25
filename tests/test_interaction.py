import unittest

from submerger.interaction import (
    DictionaryPlugin,
    LMStudioExplanationClient,
    SubtitleInteraction,
    clean_llm_explanation,
    explanation_response_format,
    explanation_user_prompt,
    normalize_lookup_text,
    tokenize_subtitle,
)
from submerger.settings import LLMEndpointSettings


class InteractionTests(unittest.TestCase):
    def test_tokenize_subtitle_keeps_words_and_cjk_tokens(self) -> None:
        tokens = tokenize_subtitle("Where's the H key? 没有H键", "primary", "line")

        self.assertEqual([token.text for token in tokens], ["Where's", "the", "H", "key", "没有H键"])

    def test_normalize_lookup_text_strips_punctuation(self) -> None:
        self.assertEqual(normalize_lookup_text('"Hockey!"'), "hockey")

    def test_dictionary_plugin_includes_paired_subtitle(self) -> None:
        result = DictionaryPlugin().lookup(
            SubtitleInteraction(
                kind="click",
                text="hockey",
                language="primary",
                paired_text="冰球",
            )
        )

        self.assertIn("team sport", result.body)
        self.assertIn("冰球", result.body)

    def test_explanation_prompt_includes_context(self) -> None:
        prompt = explanation_user_prompt(
            SubtitleInteraction(
                kind="selection",
                text="get through",
                language="primary",
                timestamp=31.2,
                paired_text="我们会打通的",
            )
        )

        self.assertIn("31.20s", prompt)
        self.assertIn("get through", prompt)
        self.assertIn("我们会打通的", prompt)

    def test_lmstudio_explanation_client_uses_transport(self) -> None:
        captured = {}

        def transport(body):
            captured.update(body)
            return "It means to successfully connect or succeed after difficulty."

        client = LMStudioExplanationClient(
            settings=LLMEndpointSettings(provider="lmstudio"),
            model="local-model",
            transport=transport,
        )
        result = client.explain(
            SubtitleInteraction(
                kind="selection",
                text="get through",
                language="primary",
                paired_text="我们会打通的",
            )
        )

        self.assertEqual(captured["model"], "local-model")
        self.assertEqual(captured["response_format"]["type"], "json_schema")
        self.assertIn("successfully connect", result.body)
        self.assertEqual(result.source, "lmstudio:local-model")

    def test_clean_llm_explanation_removes_thinking_wrappers(self) -> None:
        self.assertEqual(
            clean_llm_explanation("<think>private notes</think>\nAnswer: Public explanation."),
            "Public explanation.",
        )

    def test_clean_llm_explanation_extracts_qwen_drafting_content(self) -> None:
        self.assertEqual(
            clean_llm_explanation('Thinking Process:\n\n5. **Drafting Content:**\n"Get through means succeed."\n\n6. **Word Count Check:**'),
            "Get through means succeed.",
        )

    def test_clean_llm_explanation_extracts_unquoted_qwen_drafting_content(self) -> None:
        self.assertEqual(
            clean_llm_explanation("Thinking Process:\n\n5. **Drafting Content:**\nPhrase: get through\nMeaning: succeed.\n\n6. **Review:**"),
            "Phrase: get through Meaning: succeed.",
        )

    def test_clean_llm_explanation_extracts_qwen_draft_one(self) -> None:
        self.assertEqual(
            clean_llm_explanation("Thinking Process:\n* Draft 1: Get through means succeed.\n* Word count check: ok"),
            "Get through means succeed.",
        )

    def test_explanation_response_format_requires_explanation(self) -> None:
        schema = explanation_response_format()["json_schema"]["schema"]

        self.assertEqual(schema["required"], ["explanation"])


if __name__ == "__main__":
    unittest.main()
