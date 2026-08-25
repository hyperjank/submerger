from __future__ import annotations

import argparse
from pathlib import Path
import sys

from .alignment import (
    HeuristicAlignmentClient,
    OpenAICompatibleAlignmentClient,
    align_subtitles,
    alignment_artifact_path,
    write_alignment_outputs,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Clean and align two subtitle tracks for Submerger.")
    parser.add_argument("primary", help="Primary SRT subtitle path.")
    parser.add_argument("secondary", help="Secondary SRT subtitle path.")
    parser.add_argument("-o", "--output-prefix", default="aligned", help="Output prefix for cleaned/aligned files.")
    parser.add_argument("--primary-language", default="primary")
    parser.add_argument("--secondary-language", default="secondary")
    parser.add_argument("--provider", choices=("heuristic", "openai"), default="heuristic")
    parser.add_argument("--model", default=None, help="LLM model for --provider openai.")
    parser.add_argument("--base-url", default=None, help="OpenAI-compatible API base URL.")
    parser.add_argument("--api-key", default=None, help="API key. LM Studio can use any non-empty value.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Per-request timeout in seconds.")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--pad-seconds", type=float, default=3.0)
    parser.add_argument("--quiet", action="store_true", help="Do not print per-batch progress.")
    parser.add_argument("--keep-non-dialogue", action="store_true", help="Keep SDH-only, ad, and positioned annotation cues.")
    parser.add_argument("--no-cache", action="store_true", help="Disable per-batch alignment cache/resume.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    client = HeuristicAlignmentClient()
    if args.provider == "openai":
        client = OpenAICompatibleAlignmentClient(
            model=args.model,
            api_key=args.api_key,
            base_url=args.base_url,
            timeout=args.timeout,
        )

    output_prefix = Path(args.output_prefix)
    package = align_subtitles(
        args.primary,
        args.secondary,
        primary_language=args.primary_language,
        secondary_language=args.secondary_language,
        client=client,
        batch_size=args.batch_size,
        pad_seconds=args.pad_seconds,
        progress=None if args.quiet else print_progress,
        drop_non_dialogue=not args.keep_non_dialogue,
        cache_path=None if args.no_cache else alignment_artifact_path(output_prefix, ".alignment-cache.json"),
    )
    paths = write_alignment_outputs(package, output_prefix)
    print(f"Wrote {paths[2]}")
    print(f"Exported {paths[0]}")
    print(f"Exported {paths[1]}")
    print(f"Segments: {len(package.segments)}")
    print(f"Needs review: {len(package.issues)}")
    return 0


def print_progress(batch_number: int, total_batches: int) -> None:
    print(f"Aligning batch {batch_number}/{total_batches}...", file=sys.stderr, flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
