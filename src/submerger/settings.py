from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path


@dataclass(frozen=True)
class LLMEndpointSettings:
    provider: str = "lmstudio"
    model: str = "qwen3.5-4b"
    base_url: str = "http://192.168.86.113:1234/v1"
    api_key: str = "lm-studio"
    timeout: float = 180.0
    max_tokens: int = 1200

    @property
    def normalized_base_url(self) -> str:
        return self.base_url.rstrip("/")

    @property
    def is_openai_compatible(self) -> bool:
        return self.provider in {"lmstudio", "openai", "deepseek", "custom"}


PROVIDER_PRESETS: dict[str, LLMEndpointSettings] = {
    "lmstudio": LLMEndpointSettings(),
    "openai": LLMEndpointSettings(
        provider="openai",
        model=os.environ.get("OPENAI_MODEL", "gpt-4.1-mini"),
        base_url="https://api.openai.com/v1",
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        timeout=120.0,
    ),
    "deepseek": LLMEndpointSettings(
        provider="deepseek",
        model=os.environ.get("DEEPSEEK_MODEL", "deepseek-chat"),
        base_url="https://api.deepseek.com/v1",
        api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
        timeout=120.0,
    ),
    "custom": LLMEndpointSettings(provider="custom"),
}


def model_supports_custom_temperature(model: str) -> bool:
    """Return whether an OpenAI-compatible model accepts sampling temperature."""
    normalized = model.casefold().strip()
    return not normalized.startswith(("gpt-5", "o1", "o3", "o4"))


def default_settings_path() -> Path:
    configured = os.environ.get("SUBMERGER_SETTINGS_PATH")
    if configured:
        return Path(configured).expanduser()
    config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")).expanduser()
    return config_home / "submerger" / "llm_settings.json"


def settings_from_provider(provider: str) -> LLMEndpointSettings:
    return PROVIDER_PRESETS.get(provider, PROVIDER_PRESETS["custom"])


def load_llm_settings(path: str | Path | None = None) -> LLMEndpointSettings:
    settings_path = Path(path).expanduser() if path is not None else default_settings_path()
    base = settings_from_provider(os.environ.get("SUBMERGER_LLM_PROVIDER", "lmstudio"))
    if settings_path.exists():
        try:
            data = json.loads(settings_path.read_text(encoding="utf-8"))
            base = merge_settings(base, data)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    return merge_settings(
        base,
        {
            "provider": os.environ.get("SUBMERGER_LLM_PROVIDER"),
            "model": os.environ.get("SUBMERGER_LLM_MODEL") or os.environ.get("SUBMERGER_EXPLAIN_MODEL"),
            "base_url": os.environ.get("SUBMERGER_LLM_BASE_URL") or os.environ.get("SUBMERGER_EXPLAIN_BASE_URL"),
            "api_key": os.environ.get("SUBMERGER_LLM_API_KEY") or os.environ.get("SUBMERGER_EXPLAIN_API_KEY"),
            "timeout": os.environ.get("SUBMERGER_LLM_TIMEOUT"),
            "max_tokens": os.environ.get("SUBMERGER_LLM_MAX_TOKENS"),
        },
    )


def save_llm_settings(settings: LLMEndpointSettings, path: str | Path | None = None) -> None:
    settings_path = Path(path).expanduser() if path is not None else default_settings_path()
    settings_path.parent.mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(asdict(settings), indent=2, sort_keys=True), encoding="utf-8")


def merge_settings(base: LLMEndpointSettings, updates: dict) -> LLMEndpointSettings:
    values = asdict(base)
    for key in values:
        value = updates.get(key)
        if value in (None, ""):
            continue
        if key == "timeout":
            value = float(value)
        elif key == "max_tokens":
            value = int(value)
        values[key] = value
    provider = values.get("provider", base.provider)
    if provider != base.provider and provider in PROVIDER_PRESETS and not any(
        updates.get(key) for key in ("model", "base_url", "api_key")
    ):
        preset = asdict(PROVIDER_PRESETS[provider])
        preset.update({key: values[key] for key in ("timeout", "max_tokens")})
        values = preset
    return LLMEndpointSettings(**values)
