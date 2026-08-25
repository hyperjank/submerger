from __future__ import annotations

from dataclasses import asdict, dataclass
from html import escape
from html.parser import HTMLParser
import json
import os
from pathlib import Path
import re
import threading
import urllib.error
import urllib.parse
import urllib.request

from submerger.interaction import normalize_lookup_text

from .base import PluginAction, PluginContext, PluginResult


WIKTIONARY_BASE_URL = "https://en.wiktionary.org/api/rest_v1/page/definition"
WIKTIONARY_PAGE_URL = "https://en.wiktionary.org/wiki"
USER_AGENT = "Submerger/0.1 (https://github.com/hyperjank/submerger)"
LANGUAGE_CODES = {
    "eng": "en",
    "english": "en",
    "zho": "zh",
    "chi": "zh",
    "cmn": "zh",
    "chinese": "zh",
    "spa": "es",
    "spanish": "es",
    "fra": "fr",
    "fre": "fr",
    "french": "fr",
    "deu": "de",
    "ger": "de",
    "german": "de",
    "jpn": "ja",
    "japanese": "ja",
    "kor": "ko",
    "korean": "ko",
}


@dataclass(frozen=True)
class DictionarySense:
    definition: str
    example: str = ""


@dataclass(frozen=True)
class DictionarySection:
    language: str
    part_of_speech: str
    senses: tuple[DictionarySense, ...]


@dataclass(frozen=True)
class DictionaryEntry:
    term: str
    language_code: str
    sections: tuple[DictionarySection, ...]

    @property
    def url(self) -> str:
        return f"{WIKTIONARY_PAGE_URL}/{urllib.parse.quote(self.term, safe='')}"


class WiktionaryClient:
    def __init__(
        self,
        *,
        base_url: str | None = None,
        timeout: float = 8.0,
        transport=None,
    ) -> None:
        self.base_url = (base_url or os.environ.get("SUBMERGER_DICTIONARY_BASE_URL") or WIKTIONARY_BASE_URL).rstrip("/")
        self.timeout = timeout
        self.transport = transport

    def lookup(self, term: str, language_code: str) -> DictionaryEntry:
        url = f"{self.base_url}/{urllib.parse.quote(term, safe='')}"
        if self.transport is not None:
            payload = self.transport(term, language_code)
        else:
            request = urllib.request.Request(
                url,
                headers={"Accept": "application/json", "User-Agent": USER_AGENT},
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                if exc.code == 404:
                    payload = {}
                else:
                    raise RuntimeError(f"Wiktionary request failed: HTTP {exc.code}") from exc
            except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
                raise RuntimeError(f"Wiktionary is unavailable: {exc}") from exc
        return parse_wiktionary_entry(term, language_code, payload)


class DictionaryCache:
    def __init__(self, path: str | Path | None = None, *, limit: int = 2000) -> None:
        self.path = Path(path).expanduser() if path is not None else default_dictionary_cache_path()
        self.limit = limit
        self._lock = threading.Lock()

    def get(self, language_code: str, term: str) -> DictionaryEntry | None:
        with self._lock:
            entries = self._read()
        value = entries.get(cache_key(language_code, term))
        return dictionary_entry_from_dict(value) if isinstance(value, dict) else None

    def put(self, entry: DictionaryEntry) -> None:
        with self._lock:
            entries = self._read()
            key = cache_key(entry.language_code, entry.term)
            entries.pop(key, None)
            entries[key] = asdict(entry)
            while len(entries) > self.limit:
                entries.pop(next(iter(entries)))
            self.path.parent.mkdir(parents=True, exist_ok=True)
            temporary = self.path.with_name(f".{self.path.name}.tmp")
            temporary.write_text(
                json.dumps({"version": 1, "entries": entries}, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
            temporary.replace(self.path)

    def _read(self) -> dict[str, dict]:
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            return {}
        entries = payload.get("entries") if isinstance(payload, dict) and payload.get("version") == 1 else None
        return entries if isinstance(entries, dict) else {}


class DictionaryPlugin:
    plugin_id = "dictionary"
    name = "Dictionary"
    actions = (
        PluginAction(plugin_id, "Dictionary Lookup", ("hover", "click")),
    )

    def __init__(
        self,
        *,
        client: WiktionaryClient | None = None,
        cache: DictionaryCache | None = None,
    ) -> None:
        self.client = client or WiktionaryClient()
        self.cache = cache or DictionaryCache()

    def run(self, action: str, context: PluginContext) -> PluginResult:
        term = normalize_lookup_text(context.interaction.text)
        language_code = lookup_language(context, term)
        entry = self.cache.get(language_code, term)
        if entry is None:
            try:
                entry = self.client.lookup(term, language_code)
            except RuntimeError as exc:
                return dictionary_error_result(term, str(exc), context)
            try:
                self.cache.put(entry)
            except OSError:
                pass
        return render_dictionary_result(entry, context)


def default_dictionary_cache_path() -> Path:
    configured = os.environ.get("SUBMERGER_DICTIONARY_CACHE")
    if configured:
        return Path(configured).expanduser()
    cache_home = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()
    return cache_home / "submerger" / "dictionary.json"


def lookup_language(context: PluginContext, term: str) -> str:
    if re.search(r"[\u3400-\u9fff]", term):
        return "zh"
    configured = context.primary_language if context.interaction.language == "primary" else context.secondary_language
    normalized = configured.casefold().strip()
    return LANGUAGE_CODES.get(normalized, normalized or "en")


def parse_wiktionary_entry(term: str, language_code: str, payload: object) -> DictionaryEntry:
    if not isinstance(payload, dict):
        return DictionaryEntry(term, language_code, ())
    raw_sections = payload.get(language_code)
    if not isinstance(raw_sections, list):
        raw_sections = next(
            (
                value
                for value in payload.values()
                if isinstance(value, list)
                and value
                and isinstance(value[0], dict)
                and str(value[0].get("language", "")).casefold() == language_name(language_code).casefold()
            ),
            [],
        )
    grouped: dict[tuple[str, str], list[DictionarySense]] = {}
    for raw_section in raw_sections:
        if not isinstance(raw_section, dict):
            continue
        language = str(raw_section.get("language") or language_name(language_code))
        part_of_speech = str(raw_section.get("partOfSpeech") or "Definition")
        key = (language, part_of_speech)
        senses = grouped.setdefault(key, [])
        definitions = raw_section.get("definitions")
        if not isinstance(definitions, list):
            continue
        for raw_sense in definitions:
            if not isinstance(raw_sense, dict):
                continue
            definition = html_to_text(str(raw_sense.get("definition") or ""))
            if not definition:
                continue
            example = first_example(raw_sense)
            sense = DictionarySense(definition, example)
            if sense not in senses:
                senses.append(sense)
    sections = tuple(
        DictionarySection(language, part_of_speech, tuple(senses[:5]))
        for (language, part_of_speech), senses in grouped.items()
        if senses
    )
    return DictionaryEntry(term, language_code, sections[:6])


def render_dictionary_result(entry: DictionaryEntry, context: PluginContext) -> PluginResult:
    parts = [
        "<style>",
        ".dict-section{margin:0 0 14px}.dict-meta{color:#93c5fd;font-weight:600}",
        ".dict-example{color:#94a3b8;font-style:italic;margin-top:3px}",
        ".dict-paired{border-top:1px solid #334155;margin-top:14px;padding-top:10px}",
        "ol{padding-left:22px}li{margin:5px 0}a{color:#7dd3fc}",
        "</style>",
        f"<h2>{escape(entry.term)}</h2>",
    ]
    if not entry.sections:
        parts.append(f"<p>No Wiktionary entry was found for <b>{escape(entry.term)}</b>.</p>")
    for section in entry.sections:
        parts.append('<div class="dict-section">')
        parts.append(
            f'<div class="dict-meta">{escape(section.language)} · '
            f"{escape(section.part_of_speech)}</div><ol>"
        )
        for sense in section.senses:
            parts.append(f"<li>{escape(sense.definition)}")
            if sense.example:
                parts.append(f'<div class="dict-example">{escape(sense.example)}</div>')
            parts.append("</li>")
        parts.append("</ol></div>")
    if context.interaction.paired_text:
        parts.append(
            '<div class="dict-paired"><b>Paired subtitle</b><br>'
            f"{escape(context.interaction.paired_text)}</div>"
        )
    parts.append(
        f'<p><a href="{escape(entry.url)}">Open in Wiktionary</a> · '
        'definitions available under CC BY-SA</p>'
    )
    return PluginResult(
        f"Dictionary: {entry.term}",
        "".join(parts),
        f"Wiktionary · {entry.url}",
        "html",
    )


def dictionary_error_result(term: str, message: str, context: PluginContext) -> PluginResult:
    paired = (
        f'<div class="dict-paired"><b>Paired subtitle</b><br>{escape(context.interaction.paired_text)}</div>'
        if context.interaction.paired_text
        else ""
    )
    return PluginResult(
        f"Dictionary: {term}",
        f"<p>{escape(message)}</p>{paired}<p>Previously cached entries remain available offline.</p>",
        "Wiktionary unavailable",
        "html",
    )


def dictionary_entry_from_dict(value: dict) -> DictionaryEntry | None:
    try:
        return DictionaryEntry(
            term=str(value["term"]),
            language_code=str(value["language_code"]),
            sections=tuple(
                DictionarySection(
                    language=str(section["language"]),
                    part_of_speech=str(section["part_of_speech"]),
                    senses=tuple(
                        DictionarySense(str(sense["definition"]), str(sense.get("example") or ""))
                        for sense in section["senses"]
                    ),
                )
                for section in value["sections"]
            ),
        )
    except (KeyError, TypeError, ValueError):
        return None


def cache_key(language_code: str, term: str) -> str:
    return f"{language_code.casefold()}:{term.casefold()}"


def first_example(raw_sense: dict) -> str:
    examples = raw_sense.get("examples")
    if isinstance(examples, list):
        for value in examples:
            example = html_to_text(str(value))
            if example:
                return example
    parsed = raw_sense.get("parsedExamples")
    if isinstance(parsed, list):
        for value in parsed:
            if not isinstance(value, dict):
                continue
            example = html_to_text(str(value.get("example") or value.get("translation") or ""))
            if example:
                return example
    return ""


def language_name(language_code: str) -> str:
    return {
        "en": "English",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "ja": "Japanese",
        "ko": "Korean",
    }.get(language_code, language_code)


def html_to_text(value: str) -> str:
    parser = PlainTextHTMLParser()
    parser.feed(value)
    return re.sub(r"\s+", " ", "".join(parser.parts)).strip()


class PlainTextHTMLParser(HTMLParser):
    BLOCK_TAGS = {"br", "div", "li", "ol", "p", "tr"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_starttag(self, tag: str, attrs) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_endtag(self, tag: str) -> None:
        if tag in self.BLOCK_TAGS:
            self.parts.append(" ")

    def handle_data(self, data: str) -> None:
        self.parts.append(data)
