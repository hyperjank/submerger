from __future__ import annotations

from dataclasses import dataclass
import html
import json
import re
import urllib.error
import urllib.request

from submerger.settings import LLMEndpointSettings, load_llm_settings

from .base import PluginAction, PluginContext, PluginResult


@dataclass(frozen=True)
class DiagramUnit:
    unit_id: str
    text: str
    role: str
    gloss: str = ""


@dataclass(frozen=True)
class DiagramRelation:
    source: str
    target: str
    label: str


@dataclass(frozen=True)
class LanguageDiagram:
    label: str
    text: str
    units: list[DiagramUnit]
    relations: list[DiagramRelation]


@dataclass(frozen=True)
class SentenceDiagram:
    summary: str
    primary: LanguageDiagram
    secondary: LanguageDiagram
    links: list[DiagramRelation]


class LLMSentenceDiagramClient:
    def __init__(
        self,
        *,
        settings: LLMEndpointSettings | None = None,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float | None = None,
        max_tokens: int | None = None,
        transport=None,
    ) -> None:
        resolved = settings or load_llm_settings()
        self.provider = resolved.provider
        self.model = model or resolved.model
        self.base_url = (base_url or resolved.base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else resolved.api_key
        self.timeout = timeout if timeout is not None else resolved.timeout
        self.max_tokens = max_tokens if max_tokens is not None else resolved.max_tokens
        self.transport = transport

    def diagram(self, context: PluginContext) -> SentenceDiagram:
        body = {
            "model": self.model,
            "temperature": 0.1,
            "max_tokens": self.max_tokens,
            "response_format": diagram_response_format(),
            "messages": [
                {"role": "system", "content": DIAGRAM_SYSTEM_PROMPT},
                {"role": "user", "content": diagram_user_prompt(context)},
            ],
        }
        if self.transport is not None:
            content = self.transport(body)
        else:
            content = self._post(body)
        return parse_diagram_response(content, context)

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
            raise RuntimeError(f"LLM sentence diagram request failed: {exc.code} {detail}") from exc
        return payload["choices"][0]["message"]["content"]


class SentenceDiagramPlugin:
    plugin_id = "sentence_diagram"
    name = "Sentence Diagram"
    actions = (
        PluginAction(plugin_id, "Diagram Segment", ("selection", "segment")),
    )

    def __init__(self, client: LLMSentenceDiagramClient | None = None, settings: LLMEndpointSettings | None = None) -> None:
        self.client = client or LLMSentenceDiagramClient(settings=settings)

    def run(self, action: str, context: PluginContext) -> PluginResult:
        if not context.primary_text.strip() and not context.secondary_text.strip():
            return PluginResult("Sentence Diagram", "No subtitle text is available for diagramming.", "plugin:sentence-diagram")
        diagram = self.client.diagram(context)
        return PluginResult(
            "Sentence Diagram",
            render_diagram_html(diagram),
            f"{self.client.provider}:{self.client.model}",
            content_type="html",
        )


DIAGRAM_SYSTEM_PROMPT = """You analyze bilingual subtitle segments for language learners.
Return only valid JSON matching the provided schema. Do not include hidden reasoning, chain-of-thought, markdown, or prose outside JSON.
Split each language into meaningful units: subjects, predicates, objects, complements, modifiers, discourse markers, clauses, particles, or idiomatic chunks.
Use short role labels. Preserve the original wording. Relations should explain how units connect semantically or grammatically.
Create cross-language links where units correspond in meaning, even when word order differs."""


def diagram_user_prompt(context: PluginContext) -> str:
    timestamp = "unknown" if context.timestamp is None else f"{context.timestamp:.2f}s"
    return (
        "/no_think\n"
        f"Timestamp: {timestamp}\n"
        f"Primary subtitle: {context.primary_text or '(none)'}\n"
        f"Secondary subtitle: {context.secondary_text or '(none)'}\n\n"
        "Build a responsive language-learning diagram model for these subtitles."
    )


def diagram_response_format() -> dict:
    unit_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "id": {"type": "string"},
            "text": {"type": "string"},
            "role": {"type": "string"},
            "gloss": {"type": "string"},
        },
        "required": ["id", "text", "role", "gloss"],
    }
    relation_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "from": {"type": "string"},
            "to": {"type": "string"},
            "label": {"type": "string"},
        },
        "required": ["from", "to", "label"],
    }
    language_schema = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "text": {"type": "string"},
            "units": {"type": "array", "items": unit_schema},
            "relations": {"type": "array", "items": relation_schema},
        },
        "required": ["text", "units", "relations"],
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "sentence_diagram_response",
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "summary": {"type": "string"},
                    "primary": language_schema,
                    "secondary": language_schema,
                    "links": {"type": "array", "items": relation_schema},
                },
                "required": ["summary", "primary", "secondary", "links"],
            },
        },
    }


def parse_diagram_response(content: str, context: PluginContext) -> SentenceDiagram:
    data = extract_json_object(content)
    primary = parse_language_diagram("Primary", data.get("primary", {}), context.primary_text)
    secondary = parse_language_diagram("Secondary", data.get("secondary", {}), context.secondary_text)
    return SentenceDiagram(
        summary=str(data.get("summary", "")).strip(),
        primary=primary,
        secondary=secondary,
        links=parse_relations(data.get("links", [])),
    )


def parse_language_diagram(label: str, data: dict, fallback_text: str) -> LanguageDiagram:
    return LanguageDiagram(
        label=label,
        text=str(data.get("text") or fallback_text).strip(),
        units=parse_units(data.get("units", [])),
        relations=parse_relations(data.get("relations", [])),
    )


def parse_units(values: list) -> list[DiagramUnit]:
    units: list[DiagramUnit] = []
    for index, value in enumerate(values):
        if not isinstance(value, dict):
            continue
        text = str(value.get("text", "")).strip()
        if not text:
            continue
        unit_id = str(value.get("id") or f"u{index + 1}").strip()
        units.append(
            DiagramUnit(
                unit_id=safe_id(unit_id),
                text=text,
                role=str(value.get("role") or "unit").strip(),
                gloss=str(value.get("gloss") or "").strip(),
            )
        )
    return units


def parse_relations(values: list) -> list[DiagramRelation]:
    relations: list[DiagramRelation] = []
    for value in values:
        if not isinstance(value, dict):
            continue
        source = str(value.get("from", "")).strip()
        target = str(value.get("to", "")).strip()
        label = str(value.get("label", "")).strip()
        if source and target and label:
            relations.append(DiagramRelation(safe_id(source), safe_id(target), label))
    return relations


def extract_json_object(content: str) -> dict:
    cleaned = re.sub(r"<think>.*?</think>", "", content, flags=re.IGNORECASE | re.DOTALL).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if match is None:
            raise ValueError("LLM did not return a JSON diagram.")
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM diagram response was not a JSON object.")
    return value


def render_diagram_html(diagram: SentenceDiagram) -> str:
    sections = [
        '<div class="diagram-root">',
        "<style>",
        DIAGRAM_CSS,
        "</style>",
        '<h2>Sentence Diagram</h2>',
    ]
    if diagram.summary:
        sections.append(f'<p class="summary">{escape(diagram.summary)}</p>')
    sections.append(render_language_section(diagram.primary))
    sections.append(render_language_section(diagram.secondary))
    if diagram.links:
        sections.append('<section class="links"><h3>Meaning Links</h3>')
        for relation in diagram.links:
            sections.append(render_relation(relation))
        sections.append("</section>")
    sections.append("</div>")
    return "\n".join(sections)


def render_language_section(language: LanguageDiagram) -> str:
    if not language.text and not language.units:
        return ""
    parts = [f'<section class="language"><h3>{escape(language.label)}</h3>']
    if language.text:
        parts.append(f'<p class="subtitle-text">{escape(language.text)}</p>')
    parts.append('<div class="unit-cloud">')
    for unit in language.units:
        parts.append(
            f'<span class="unit role-{role_class(unit.role)}" title="{escape(unit.gloss)}">'
            f'<span class="unit-text">{escape(unit.text)}</span>'
            f'<span class="unit-role">{escape(unit.role)}</span>'
            "</span>"
        )
    parts.append("</div>")
    if language.relations:
        parts.append('<div class="relations">')
        for relation in language.relations:
            parts.append(render_relation(relation))
        parts.append("</div>")
    parts.append("</section>")
    return "\n".join(parts)


def render_relation(relation: DiagramRelation) -> str:
    return (
        '<div class="relation">'
        f'<span class="endpoint">{escape(relation.source)}</span>'
        f'<span class="relation-label">{escape(relation.label)}</span>'
        f'<span class="endpoint">{escape(relation.target)}</span>'
        "</div>"
    )


DIAGRAM_CSS = """
.diagram-root { color: #e5e7eb; background: #0f172a; font-family: Inter, Arial, sans-serif; line-height: 1.35; }
h2 { font-size: 18px; margin: 0 0 10px; }
h3 { font-size: 13px; margin: 16px 0 6px; color: #bfdbfe; text-transform: uppercase; }
.summary, .subtitle-text { margin: 0 0 8px; color: #e2e8f0; }
.language { border-top: 1px solid #334155; padding-top: 10px; }
.unit-cloud { margin: 6px 0 10px; }
.unit { display: inline-block; vertical-align: top; max-width: 96%; margin: 3px; padding: 6px 8px; border-radius: 6px; border: 1px solid #475569; background: #172033; }
.unit-text { display: block; font-weight: 700; color: #f8fafc; }
.unit-role { display: block; margin-top: 2px; font-size: 10px; color: #cbd5e1; text-transform: uppercase; }
.role-predicate, .role-verb { border-color: #38bdf8; background: #113244; }
.role-subject { border-color: #a7f3d0; background: #123329; }
.role-object, .role-complement { border-color: #facc15; background: #332d12; }
.role-modifier, .role-adverbial { border-color: #c4b5fd; background: #29213d; }
.relation { margin: 4px 0; padding: 5px 7px; border-left: 2px solid #64748b; background: #111827; }
.endpoint { color: #f8fafc; font-weight: 700; }
.relation-label { color: #93c5fd; margin: 0 8px; }
"""


def safe_id(value: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_-]+", "-", value.strip())
    return cleaned.strip("-") or "unit"


def role_class(value: str) -> str:
    return safe_id(value.lower())


def escape(value: str) -> str:
    return html.escape(value, quote=True)
