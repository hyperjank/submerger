import os
import tempfile
import unittest
from pathlib import Path

from submerger.settings import (
    LLMEndpointSettings,
    load_llm_settings,
    model_supports_custom_temperature,
    save_llm_settings,
    settings_from_provider,
)


class SettingsTests(unittest.TestCase):
    def test_provider_presets_include_requested_endpoints(self) -> None:
        self.assertEqual(settings_from_provider("lmstudio").base_url, "http://192.168.86.113:1234/v1")
        self.assertEqual(settings_from_provider("openai").base_url, "https://api.openai.com/v1")
        self.assertEqual(settings_from_provider("deepseek").base_url, "https://api.deepseek.com/v1")

    def test_save_and_load_llm_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            settings = LLMEndpointSettings(
                provider="custom",
                model="local-a",
                base_url="http://localhost:1234/v1",
                api_key="key",
                timeout=45,
                max_tokens=900,
            )

            save_llm_settings(settings, path)
            loaded = load_llm_settings(path)

        self.assertEqual(loaded, settings)

    def test_environment_overrides_saved_settings(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "settings.json"
            save_llm_settings(LLMEndpointSettings(model="saved"), path)
            old = os.environ.get("SUBMERGER_LLM_MODEL")
            os.environ["SUBMERGER_LLM_MODEL"] = "env-model"
            try:
                loaded = load_llm_settings(path)
            finally:
                if old is None:
                    os.environ.pop("SUBMERGER_LLM_MODEL", None)
                else:
                    os.environ["SUBMERGER_LLM_MODEL"] = old

        self.assertEqual(loaded.model, "env-model")

    def test_reasoning_models_omit_custom_temperature(self) -> None:
        self.assertFalse(model_supports_custom_temperature("gpt-5.6-luna"))
        self.assertFalse(model_supports_custom_temperature("o3"))
        self.assertTrue(model_supports_custom_temperature("gpt-4.1-mini"))
        self.assertTrue(model_supports_custom_temperature("qwen3.5-4b"))


if __name__ == "__main__":
    unittest.main()
