from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
import urllib.error
import urllib.request

from .settings import LLMEndpointSettings, load_llm_settings, model_supports_custom_temperature


TOKEN_RE = re.compile(r"[\w\u3400-\u9fff]+(?:['’-][\w\u3400-\u9fff]+)?|[^\s\w]", re.UNICODE)


@dataclass(frozen=True)
class SubtitleToken:
    text: str
    index: int
    language: str
    line: str

    @property
    def normalized(self) -> str:
        return normalize_lookup_text(self.text)


@dataclass(frozen=True)
class SubtitleInteraction:
    kind: str
    text: str
    language: str
    timestamp: float | None = None
    paired_text: str = ""
    token_index: int | None = None


@dataclass(frozen=True)
class PluginResult:
    title: str
    body: str
    source: str
    content_type: str = "text"


def tokenize_subtitle(text: str, language: str, line: str) -> list[SubtitleToken]:
    tokens: list[SubtitleToken] = []
    for index, match in enumerate(TOKEN_RE.finditer(text)):
        token_text = match.group(0).strip()
        if token_text and any(char.isalnum() for char in token_text):
            tokens.append(SubtitleToken(token_text, index, language, line))
    return tokens


def normalize_lookup_text(text: str) -> str:
    return text.strip().lower().strip(".,!?;:\"'()[]{}“”‘’")


class LMStudioExplanationClient:
    def __init__(
        self,
        *,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 180.0,
        max_tokens: int = 1000,
        settings: LLMEndpointSettings | None = None,
        transport=None,
    ) -> None:
        resolved = settings or load_llm_settings()
        self.provider = resolved.provider
        self.model = model or os.environ.get("SUBMERGER_EXPLAIN_MODEL") or resolved.model
        self.base_url = (base_url or os.environ.get("SUBMERGER_EXPLAIN_BASE_URL") or resolved.base_url).rstrip("/")
        self.api_key = api_key or os.environ.get("SUBMERGER_EXPLAIN_API_KEY") or resolved.api_key
        self.timeout = timeout if settings is None else resolved.timeout
        self.max_tokens = max_tokens if settings is None else resolved.max_tokens
        self.transport = transport

    def explain(self, interaction: SubtitleInteraction) -> PluginResult:
        body = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "response_format": explanation_response_format(),
            "messages": [
                {"role": "system", "content": EXPLANATION_SYSTEM_PROMPT},
                {"role": "user", "content": explanation_user_prompt(interaction)},
            ],
        }
        if model_supports_custom_temperature(self.model):
            body["temperature"] = 0.2
        if self.transport is not None:
            content = self.transport(body)
        else:
            content = self._post(body)
        return PluginResult(title=f"Explanation: {interaction.text}", body=clean_llm_explanation(content), source=f"{self.provider}:{self.model}")

    def _post(self, body: dict) -> str:
        request = urllib.request.Request(
            f"{self.base_url}/chat/completions",
            data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM explanation request failed: {exc.code} {detail}") from exc
        content = payload["choices"][0]["message"]["content"]
        try:
            parsed = json.loads(content)
            if "explanation" in parsed:
                return str(parsed["explanation"])
        except json.JSONDecodeError:
            pass
        return content


EXPLANATION_SYSTEM_PROMPT = """You are a concise language-learning assistant inside a subtitle-based video player.
Explain the selected phrase using the subtitle context. Focus on meaning, grammar, idiom/slang, and how it maps to the paired subtitle.
Keep the answer under 120 words. Do not invent context that is not present.
Answer directly. Do not show hidden reasoning, chain-of-thought, planning, analysis, or a thinking process."""


def explanation_user_prompt(interaction: SubtitleInteraction) -> str:
    timestamp = "unknown" if interaction.timestamp is None else f"{interaction.timestamp:.2f}s"
    paired = interaction.paired_text or "(none)"
    return (
        "/no_think\n"
        f"Timestamp: {timestamp}\n"
        f"Selected language: {interaction.language}\n"
        f"Selected phrase: {interaction.text}\n"
        f"Paired subtitle: {paired}\n\n"
        "Explain this phrase for a language learner. Include a compact translation/gloss if useful."
    )


def clean_llm_explanation(content: str) -> str:
    text = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    drafting_match = re.search(r"Drafting Content:?\**\s*\n\s*[\"“](.+?)(?:[\"”]\s*(?:\n\d+\.|\Z))", text, flags=re.DOTALL)
    if drafting_match:
        return re.sub(r"\s+", " ", drafting_match.group(1)).strip()
    drafting_section_match = re.search(r"Drafting Content:?\**\s*\n(.+?)(?:\n\s*\d+\.\s|\Z)", text, flags=re.DOTALL)
    if drafting_section_match:
        return re.sub(r"\s+", " ", drafting_section_match.group(1)).strip()
    draft_one_match = re.search(r"Draft 1:\s*(.+?)(?:\n\s*\*\s*Word count|\n\s*\d+\.|\Z)", text, flags=re.DOTALL)
    if draft_one_match:
        return re.sub(r"\s+", " ", draft_one_match.group(1)).strip()
    for marker in ("Final Answer:", "Final answer:", "Answer:", "Learner-facing answer:"):
        if marker in text:
            text = text.split(marker, 1)[1].strip()
    if text.startswith("Thinking Process:"):
        parts = re.split(r"\n\s*(?:Final|Answer|Explanation)\s*:?\s*", text, maxsplit=1)
        if len(parts) == 2:
            text = parts[1].strip()
    return text.strip()


def explanation_response_format() -> dict:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "phrase_explanation_response",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "explanation": {
                        "type": "string",
                        "description": "A concise learner-facing explanation under 120 words with no hidden reasoning.",
                    }
                },
                "required": ["explanation"],
            },
        },
    }
