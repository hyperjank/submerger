import tempfile
import unittest
from pathlib import Path

from submerger.alignment import (
    AlignmentResult,
    AlignmentPackage,
    AlignmentValidator,
    CandidateWindow,
    GeneratedSubtitle,
    carry_forward_human_reviews,
    HeuristicAlignmentClient,
    PairedSegment,
    PrimarySegment,
    PrimarySegmenter,
    SecondaryWindowBuilder,
    SubtitleDocument,
    align_subtitles,
    build_alignment_blocks,
    filter_dialogue_document,
    has_terminal_sentence_end,
    load_alignment_cache,
    load_alignment_package,
    context_alignment_response_format,
    effective_secondary_text,
    alignment_response_format,
    review_alignment_segment,
    save_alignment_cache,
    tracks_from_alignment_package,
    SubtitleRepairResult,
    write_aligned_srt_exports,
    write_alignment_sidecar,
    write_alignment_outputs,
)
from submerger.subtitles import SubtitleCue


PRIMARY_SRT = """1
00:00:01,000 --> 00:00:02,000
I thought you said

2
00:00:02,100 --> 00:00:03,000
you weren't coming back.

3
00:00:04,000 --> 00:00:05,000
But here you are.
"""


SECONDARY_SRT = """1
00:00:00,900 --> 00:00:01,800
Pensé que dijiste

2
00:00:01,900 --> 00:00:03,200
que no volverías.

3
00:00:04,100 --> 00:00:05,100
Pero aquí estás.
"""


class AlignmentTests(unittest.TestCase):
    def test_primary_segmenter_merges_until_sentence_end(self) -> None:
        document = SubtitleDocument("en", [
            SubtitleCue(1, 2, "I thought you said", "1"),
            SubtitleCue(2.1, 3, "you weren't coming back.", "2"),
            SubtitleCue(4, 5, "But here you are.", "3"),
        ])

        segments = PrimarySegmenter().segment(document)

        self.assertEqual(len(segments), 2)
        self.assertEqual(segments[0].text, "I thought you said you weren't coming back.")
        self.assertEqual(segments[0].source_cue_ids, ["1", "2"])

    def test_primary_segmenter_does_not_split_on_ellipsis(self) -> None:
        document = SubtitleDocument("en", [
            SubtitleCue(1, 2, "The first listener who answers...", "1"),
            SubtitleCue(2.1, 3, "is gonna win.", "2"),
        ])

        segments = PrimarySegmenter().segment(document)

        self.assertEqual(len(segments), 1)
        self.assertEqual(segments[0].source_cue_ids, ["1", "2"])
        self.assertFalse(has_terminal_sentence_end("answers..."))
        self.assertTrue(has_terminal_sentence_end("is gonna win."))

    def test_window_builder_collects_fuzzy_secondary_candidates(self) -> None:
        primary = PrimarySegmenter().segment(SubtitleDocument("en", [SubtitleCue(10, 11, "Hello.", "1")]))
        secondary = SubtitleDocument("es", [SubtitleCue(7.5, 8.2, "No", "1"), SubtitleCue(8.5, 9.2, "Hola", "2")])

        windows = SecondaryWindowBuilder(pad_seconds=1, max_pad_seconds=4).build(primary, secondary)

        self.assertEqual([cue.cue_id for cue in windows[0].secondary_cues], ["2"])

    def test_filter_dialogue_document_drops_annotations_and_ads(self) -> None:
        document = SubtitleDocument("zh", [
            SubtitleCue(1, 2, "上午9点30分", "1", "{\\an8}上午9点30分"),
            SubtitleCue(2, 3, "你好。", "2", "你好。"),
            SubtitleCue(3, 4, "[Screams]", "3", "[Screams]"),
            SubtitleCue(4, 5, "Advertise your product or brand here", "4", "Advertise your product or brand here"),
        ])

        filtered = filter_dialogue_document(document)

        self.assertEqual([cue.cue_id for cue in filtered.cues], ["2"])

    def test_validator_allows_reuse_and_repairs_internal_order(self) -> None:
        cue_1 = SubtitleCue(5, 6, "uno", "1")
        cue_2 = SubtitleCue(7, 8, "dos", "2")
        windows = [
            CandidateWindow(PrimarySegment("p_1", 5, 6, "one", ["1"]), [cue_1]),
            CandidateWindow(PrimarySegment("p_2", 7, 8, "two", ["2"]), [cue_1, cue_2]),
        ]
        results = [
            AlignmentResult("p_1", ["1"], 0.9),
            AlignmentResult("p_2", ["2", "1"], 0.8),
        ]

        validated = AlignmentValidator().validate(results, windows)

        self.assertEqual(validated[0].status, "ok")
        self.assertEqual(validated[1].status, "repaired")
        self.assertIn("non-monotonic secondary order", validated[1].problems)
        self.assertNotIn("reused secondary cue ids: 1", validated[1].problems)
        self.assertEqual(validated[1].accepted_secondary_cue_ids, ["1", "2"])

    def test_shared_secondary_cue_builds_one_many_to_many_block(self) -> None:
        cue_1 = SubtitleCue(5, 6, "共同字幕", "1")
        cue_2 = SubtitleCue(6, 8, "第二部分", "2")
        windows = [
            CandidateWindow(PrimarySegment("p_1", 5, 6, "First.", ["p1"]), [cue_1]),
            CandidateWindow(PrimarySegment("p_2", 6, 8, "Second.", ["p2"]), [cue_1, cue_2]),
        ]
        validated = AlignmentValidator().validate(
            [
                AlignmentResult("p_1", ["1"], 0.9),
                AlignmentResult("p_2", ["1", "2"], 0.9),
            ],
            windows,
        )

        blocks = build_alignment_blocks(windows, validated)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].segment_id, "p_1")
        self.assertEqual(blocks[0].primary_segment_ids, ["p_1", "p_2"])
        self.assertEqual(blocks[0].primary_cue_ids, ["p1", "p2"])
        self.assertEqual(blocks[0].secondary_cue_ids, ["1", "2"])
        self.assertEqual(blocks[0].primary_text, "First. Second.")
        self.assertEqual(blocks[0].secondary_text, "共同字幕 第二部分")
        self.assertEqual(blocks[0].status, "ok")

    def test_monotonic_disjoint_mappings_remain_separate_blocks(self) -> None:
        cue_1 = SubtitleCue(5, 6, "一", "1")
        cue_2 = SubtitleCue(7, 8, "二", "2")
        windows = [
            CandidateWindow(PrimarySegment("p_1", 5, 6, "One.", ["p1"]), [cue_1]),
            CandidateWindow(PrimarySegment("p_2", 7, 8, "Two.", ["p2"]), [cue_2]),
        ]
        validated = AlignmentValidator().validate(
            [
                AlignmentResult("p_1", ["1"], 0.9),
                AlignmentResult("p_2", ["2"], 0.9),
            ],
            windows,
        )

        blocks = build_alignment_blocks(windows, validated)

        self.assertEqual([block.segment_id for block in blocks], ["p_1", "p_2"])

    def test_crossing_mappings_become_one_repaired_monotonic_block(self) -> None:
        cue_1 = SubtitleCue(5, 6, "一", "1")
        cue_2 = SubtitleCue(7, 8, "二", "2")
        windows = [
            CandidateWindow(PrimarySegment("p_1", 5, 6, "One.", ["p1"]), [cue_1, cue_2]),
            CandidateWindow(PrimarySegment("p_2", 7, 8, "Two.", ["p2"]), [cue_1, cue_2]),
        ]
        validated = AlignmentValidator().validate(
            [
                AlignmentResult("p_1", ["2"], 0.9),
                AlignmentResult("p_2", ["1"], 0.9),
            ],
            windows,
        )

        blocks = build_alignment_blocks(windows, validated)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].secondary_cue_ids, ["1", "2"])
        self.assertEqual(blocks[0].status, "repaired")
        self.assertIn("crossing mappings merged", blocks[0].problems[0])

    def test_low_confidence_shared_boundary_is_promoted_into_block(self) -> None:
        cue = SubtitleCue(5, 8, "共同字幕", "1")
        windows = [
            CandidateWindow(PrimarySegment("p_1", 5, 6, "Setup.", ["p1"]), [cue]),
            CandidateWindow(PrimarySegment("p_2", 6, 8, "Punchline.", ["p2"]), [cue]),
        ]
        validated = AlignmentValidator().validate(
            [
                AlignmentResult("p_1", ["1"], 0.9),
                AlignmentResult("p_2", ["1"], 0.4),
            ],
            windows,
        )

        blocks = build_alignment_blocks(windows, validated)

        self.assertEqual(len(blocks), 1)
        self.assertEqual(blocks[0].primary_segment_ids, ["p_1", "p_2"])
        self.assertEqual(blocks[0].secondary_cue_ids, ["1"])
        self.assertEqual(blocks[0].status, "repaired")
        self.assertIn("shared cue promoted", blocks[0].problems[0])

    def test_validator_merges_duplicates_and_reports_every_missing_or_unknown_result(self) -> None:
        cue_1 = SubtitleCue(1, 2, "uno", "1")
        cue_2 = SubtitleCue(2, 3, "dos", "2")
        windows = [
            CandidateWindow(PrimarySegment("p_1", 1, 2, "one", ["1"]), [cue_1, cue_2]),
            CandidateWindow(PrimarySegment("p_2", 3, 4, "two", ["2"]), [cue_2]),
        ]
        results = [
            AlignmentResult("p_1", ["1"], 0.9),
            AlignmentResult("p_1", ["2"], 0.8),
            AlignmentResult("p_unknown", ["1"], 1.0),
        ]

        validated = AlignmentValidator().validate(results, windows)

        self.assertEqual(validated[0].accepted_secondary_cue_ids, ["1", "2"])
        self.assertEqual(validated[0].status, "repaired")
        self.assertIn("duplicate primary result entries: 2 (merged)", validated[0].problems)
        self.assertEqual(validated[1].status, "needs_review")
        self.assertIn("missing alignment result", validated[1].problems)
        self.assertEqual(validated[2].result.primary_id, "p_unknown")
        self.assertIn("unknown primary id", validated[2].problems[0])

    def test_align_subtitles_writes_clean_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            primary_path = tmp_path / "primary.srt"
            secondary_path = tmp_path / "secondary.srt"
            primary_path.write_text(PRIMARY_SRT, encoding="utf-8")
            secondary_path.write_text(SECONDARY_SRT, encoding="utf-8")

            package = align_subtitles(
                primary_path,
                secondary_path,
                primary_language="en",
                secondary_language="es",
                client=HeuristicAlignmentClient(),
            )
            primary_out, secondary_out, json_out = write_alignment_outputs(package, tmp_path / "episode")

            self.assertEqual(len(package.segments), 2)
            self.assertTrue(primary_out.exists())
            self.assertTrue(secondary_out.exists())
            self.assertTrue(json_out.exists())
            self.assertIn("I thought you said you weren't coming back.", primary_out.read_text(encoding="utf-8"))
            self.assertIn("Pensé que dijiste que no volverías.", secondary_out.read_text(encoding="utf-8"))

    def test_align_subtitles_merges_shared_cues_without_creating_review_work(self) -> None:
        class SharedCueClient:
            def align_batch(self, windows):
                return [
                    AlignmentResult(
                        window.primary.segment_id,
                        ["1"],
                        0.9 if index == 0 else 0.4,
                    )
                    for index, window in enumerate(windows)
                ]

        primary_srt = """1
00:00:01,000 --> 00:00:02,000
First sentence.

2
00:00:02,100 --> 00:00:03,000
Second sentence.
"""
        secondary_srt = """1
00:00:00,900 --> 00:00:03,100
Una traducción compartida.
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            primary_path = tmp_path / "primary.srt"
            secondary_path = tmp_path / "secondary.srt"
            primary_path.write_text(primary_srt, encoding="utf-8")
            secondary_path.write_text(secondary_srt, encoding="utf-8")

            package = align_subtitles(
                primary_path,
                secondary_path,
                client=SharedCueClient(),
            )

        self.assertEqual(len(package.segments), 1)
        self.assertEqual(package.segments[0].primary_segment_ids, ["p_00001", "p_00002"])
        self.assertEqual(package.segments[0].status, "repaired")
        self.assertEqual(package.issues, [])

    def test_context_retry_uses_wider_candidates_neighbors_anchors_and_cache(self) -> None:
        class ContextRetryClient:
            def __init__(self) -> None:
                self.initial_calls = 0
                self.retry_calls = 0

            def align_batch(self, windows):
                self.initial_calls += 1
                return [
                    AlignmentResult(windows[0].primary.segment_id, ["1"], 0.95, "anchor before"),
                    AlignmentResult(windows[1].primary.segment_id, [], 0.1, "no nearby match"),
                    AlignmentResult(windows[2].primary.segment_id, ["4"], 0.95, "anchor after"),
                ]

            def align_batch_with_context(self, regions):
                self.retry_calls += 1
                self.assert_retry_region(regions[0])
                return [
                    AlignmentResult(
                        regions[0]["target"]["primary"]["segment_id"],
                        ["3"],
                        0.92,
                        "resolved from wider context",
                    )
                ]

            @staticmethod
            def assert_retry_region(region):
                candidate_ids = {
                    cue["cue_id"] for cue in region["target"]["secondary_candidates"]
                }
                neighbor_ids = {
                    primary["segment_id"] for primary in region["neighboring_primary"]
                }
                anchor_ids = {
                    anchor["primary_id"] for anchor in region["accepted_neighbor_mappings"]
                }
                assert candidate_ids == {"2", "3"}
                assert neighbor_ids == {"p_00001", "p_00003"}
                assert anchor_ids == {"p_00001", "p_00003"}
                assert region["first_attempt"]["secondary_cue_ids"] == []

        primary_srt = """1
00:00:10,000 --> 00:00:11,000
Before.

2
00:00:20,000 --> 00:00:21,000
Target.

3
00:00:30,000 --> 00:00:31,000
After.
"""
        secondary_srt = """1
00:00:10,000 --> 00:00:11,000
Antes.

2
00:00:22,000 --> 00:00:22,500
Unrelated.

3
00:00:25,000 --> 00:00:26,000
Objetivo.

4
00:00:30,000 --> 00:00:31,000
Después.
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            primary_path = tmp_path / "primary.srt"
            secondary_path = tmp_path / "secondary.srt"
            cache_path = tmp_path / "alignment-cache.json"
            primary_path.write_text(primary_srt, encoding="utf-8")
            secondary_path.write_text(secondary_srt, encoding="utf-8")
            client = ContextRetryClient()

            package = align_subtitles(
                primary_path,
                secondary_path,
                client=client,
                cache_path=cache_path,
            )
            cached_package = align_subtitles(
                primary_path,
                secondary_path,
                client=client,
                cache_path=cache_path,
            )
            sidecar_path = write_alignment_sidecar(package, tmp_path / "episode")
            reloaded_package = load_alignment_package(sidecar_path)

        target = next(
            segment for segment in package.segments if segment.segment_id == "p_00002"
        )
        self.assertEqual(target.secondary_cue_ids, ["3"])
        self.assertEqual(target.alignment_stage, "context_retry")
        self.assertEqual(target.alignment_notes, "resolved from wider context")
        self.assertEqual(package.issues, [])
        self.assertEqual(cached_package.segments, package.segments)
        reloaded_target = next(
            segment
            for segment in reloaded_package.segments
            if segment.segment_id == "p_00002"
        )
        self.assertEqual(reloaded_target.alignment_stage, "context_retry")
        self.assertEqual(reloaded_target.alignment_notes, "resolved from wider context")
        self.assertEqual(reloaded_package.schema_version, 5)
        self.assertEqual(client.initial_calls, 1)
        self.assertEqual(client.retry_calls, 1)

    def test_context_retry_cannot_overturn_semantic_rejection_without_structure(self) -> None:
        class OvereagerRetryClient:
            def align_batch(self, windows):
                return [
                    AlignmentResult(
                        windows[0].primary.segment_id,
                        [],
                        0.1,
                        "candidate does not match",
                    )
                ]

            def align_batch_with_context(self, regions):
                return [
                    AlignmentResult(
                        regions[0]["target"]["primary"]["segment_id"],
                        ["1"],
                        0.99,
                        "plausible localization",
                    )
                ]

        primary_srt = """1
00:00:20,000 --> 00:00:21,000
Center ice shot, baby!
"""
        secondary_srt = """1
00:00:20,000 --> 00:00:21,000
News flash, babies!
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            primary_path = tmp_path / "primary.srt"
            secondary_path = tmp_path / "secondary.srt"
            primary_path.write_text(primary_srt, encoding="utf-8")
            secondary_path.write_text(secondary_srt, encoding="utf-8")

            package = align_subtitles(
                primary_path,
                secondary_path,
                client=OvereagerRetryClient(),
            )

        segment = package.segments[0]
        self.assertEqual(segment.secondary_cue_ids, [])
        self.assertEqual(segment.alignment_stage, "context_retry")
        self.assertIn("kept the first pass", segment.alignment_notes)
        self.assertEqual(len(package.issues), 1)

    def test_alignment_cache_round_trips_batch_results(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "cache.json"
            identity = {"primary_sha256": "one", "model": "local"}
            save_alignment_cache(
                path,
                {"0": [AlignmentResult("p_1", ["2"], 0.9, "ok")]},
                identity=identity,
            )

            loaded = load_alignment_cache(path, expected_identity=identity)

            self.assertEqual(loaded["0"][0], AlignmentResult("p_1", ["2"], 0.9, "ok"))
            self.assertEqual(
                load_alignment_cache(path, expected_identity={"primary_sha256": "different"}),
                {},
            )

    def test_sidecar_round_trip_review_and_runtime_tracks(self) -> None:
        candidate = SubtitleCue(1, 2, "Hola.", "s1", "Hola.")
        package = AlignmentPackage(
            primary_language="en",
            secondary_language="es",
            segments=[
                PairedSegment(
                    "p_1",
                    1,
                    2,
                    "Hello.",
                    "",
                    ["p1"],
                    [],
                    0,
                    "needs_review",
                    ["missing alignment result"],
                    [candidate],
                )
            ],
            issues=[],
        )
        with tempfile.TemporaryDirectory() as tmp:
            dotted_prefix = Path(tmp) / "Show.S01E01"
            sidecar = write_alignment_sidecar(package, dotted_prefix)
            self.assertEqual(sidecar.name, "Show.S01E01.alignment.json")
            loaded = load_alignment_package(sidecar)
            reviewed = review_alignment_segment(loaded, "p_1", ["s1"])
            write_alignment_sidecar(reviewed, sidecar)
            reloaded = load_alignment_package(sidecar)

        self.assertEqual(reloaded.segments[0].status, "reviewed")
        self.assertEqual(reloaded.segments[0].alignment_stage, "human_review")
        self.assertEqual(reloaded.segments[0].primary_segment_ids, ["p_1"])
        self.assertEqual(reloaded.segments[0].secondary_text, "Hola.")
        primary, secondary = tracks_from_alignment_package(reloaded)
        self.assertEqual(primary.active_text(1.5), "Hello.")
        self.assertEqual(secondary.active_text(1.5), "Hola.")

    def test_human_reviews_survive_a_matching_rerun(self) -> None:
        cue = SubtitleCue(1, 2, "Hola.", "s1", "Hola.")
        identity = {"primary_sha256": "primary", "secondary_sha256": "secondary"}
        generated = AlignmentPackage(
            "en",
            "es",
            [
                PairedSegment(
                    "p_1", 1, 2, "Hello.", "", ["p1"], [], 0,
                    "needs_review", ["missing alignment result"], [cue],
                )
            ],
            [],
            cache_identity=identity,
        )
        reviewed = review_alignment_segment(generated, "p_1", ["s1"])

        merged = carry_forward_human_reviews(generated, reviewed)
        changed_source = carry_forward_human_reviews(
            AlignmentPackage(
                "en", "es", generated.segments, [],
                cache_identity={"primary_sha256": "changed", "secondary_sha256": "secondary"},
            ),
            reviewed,
        )

        self.assertEqual(merged.segments[0].status, "reviewed")
        self.assertEqual(merged.segments[0].secondary_text, "Hola.")
        self.assertEqual(changed_source.segments[0].status, "needs_review")

    def test_context_dispositions_require_high_confidence_and_context_stage(self) -> None:
        window = CandidateWindow(
            PrimarySegment("p_1", 1, 2, "[door closes]", ["p1"]),
            [],
        )
        validator = AlignmentValidator()

        accepted = validator.validate([
            AlignmentResult(
                "p_1", [], 0.96, "sound only", "context_retry", "non_dialogue"
            )
        ], [window])[0]
        low_confidence = validator.validate([
            AlignmentResult(
                "p_1", [], 0.75, "maybe omitted", "context_retry", "omitted_dialogue"
            )
        ], [window])[0]
        wrong_stage = validator.validate([
            AlignmentResult(
                "p_1", [], 0.99, "premature", "initial", "non_dialogue"
            )
        ], [window])[0]
        calibrated_terminal = validator.validate([
            AlignmentResult(
                "p_1", [], 0.87, "confident omission", "context_retry", "omitted_dialogue"
            )
        ], [window])[0]

        self.assertEqual(accepted.status, "ignored")
        self.assertEqual(accepted.result.disposition, "non_dialogue")
        self.assertEqual(low_confidence.status, "needs_review")
        self.assertIn("low-confidence terminal disposition", low_confidence.problems)
        self.assertEqual(wrong_stage.status, "needs_review")
        self.assertIn("terminal disposition requires context retry", wrong_stage.problems)
        self.assertEqual(calibrated_terminal.status, "needs_repair")

    def test_repair_is_opt_in_cached_and_preserves_imported_sources(self) -> None:
        class RepairClient:
            def __init__(self) -> None:
                self.initial_calls = 0
                self.retry_calls = 0
                self.repair_calls = 0

            def cache_identity(self):
                return {"provider": "test", "model": "translator-v1"}

            def align_batch(self, windows):
                self.initial_calls += 1
                return [
                    AlignmentResult(window.primary.segment_id, [], 0.1, "not translated")
                    for window in windows
                ]

            def align_batch_with_context(self, regions):
                self.retry_calls += 1
                return [
                    AlignmentResult(
                        region["target"]["primary"]["segment_id"],
                        [],
                        0.98,
                        "meaningful dialogue is omitted",
                        disposition="omitted_dialogue",
                    )
                    for region in regions
                ]

            def repair_batch(self, regions, *, source_language, target_language):
                self.repair_calls += 1
                assert source_language == "en"
                assert target_language == "es"
                return [
                    SubtitleRepairResult(
                        region["repair_target"]["primary_id"],
                        "No volveré.",
                        target_language,
                        0.97,
                        "faithful concise translation",
                    )
                    for region in regions
                ]

        primary_srt = """1
00:00:01,000 --> 00:00:02,000
I will not return.
"""
        secondary_srt = """1
00:00:01,000 --> 00:00:02,000
Hasta luego.
"""
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            primary_path = tmp_path / "primary.srt"
            secondary_path = tmp_path / "secondary.srt"
            alignment_cache = tmp_path / "alignment-cache.json"
            repair_cache = tmp_path / "repair-cache.json"
            primary_path.write_text(primary_srt, encoding="utf-8")
            secondary_path.write_text(secondary_srt, encoding="utf-8")
            original_primary = primary_path.read_bytes()
            original_secondary = secondary_path.read_bytes()
            client = RepairClient()

            unrepaired = align_subtitles(
                primary_path,
                secondary_path,
                primary_language="en",
                secondary_language="es",
                client=client,
            )
            repaired = align_subtitles(
                primary_path,
                secondary_path,
                primary_language="en",
                secondary_language="es",
                client=client,
                cache_path=alignment_cache,
                repair_cache_path=repair_cache,
                repair_target_dialogue=True,
            )
            cached = align_subtitles(
                primary_path,
                secondary_path,
                primary_language="en",
                secondary_language="es",
                client=client,
                cache_path=alignment_cache,
                repair_cache_path=repair_cache,
                repair_target_dialogue=True,
            )
            _, secondary_export = write_aligned_srt_exports(
                repaired, tmp_path / "episode"
            )
            secondary_export_text = secondary_export.read_text(encoding="utf-8")

            self.assertEqual(primary_path.read_bytes(), original_primary)
            self.assertEqual(secondary_path.read_bytes(), original_secondary)

        self.assertEqual(unrepaired.segments[0].status, "needs_repair")
        self.assertIsNone(unrepaired.segments[0].generated_secondary)
        segment = repaired.segments[0]
        self.assertEqual(segment.status, "generated")
        self.assertEqual(segment.disposition, "omitted_dialogue")
        self.assertEqual(segment.secondary_text, "")
        self.assertEqual(segment.secondary_cue_ids, [])
        self.assertEqual(effective_secondary_text(segment), "No volveré.")
        self.assertEqual(segment.generated_secondary.candidate_secondary_cue_ids, ["1"])
        self.assertEqual(segment.generated_secondary.provider, "test")
        self.assertEqual(repaired.issues, [])
        self.assertEqual(cached.segments, repaired.segments)
        self.assertEqual(client.repair_calls, 1)
        self.assertIn("No volveré.", secondary_export_text)

    def test_generated_secondary_round_trips_and_human_review_supersedes_it(self) -> None:
        candidate = SubtitleCue(1, 2, "Original.", "s1", "Original.")
        generated = GeneratedSubtitle(
            text="Generated.",
            target_language="es",
            confidence=0.98,
            reason="omitted",
            provider="test",
            model="model",
            prompt_version="v1",
            source_primary_segment_ids=["p_1"],
            source_primary_cue_ids=["p1"],
            candidate_secondary_cue_ids=["s1"],
            source_primary_sha256="digest",
        )
        package = AlignmentPackage(
            "en",
            "es",
            [
                PairedSegment(
                    "p_1", 1, 2, "Hello.", "", ["p1"], [], 0.98,
                    "generated", [], [candidate], ["p_1"], "omitted",
                    "generated_repair", "omitted_dialogue", generated,
                )
            ],
            [],
            media_language="en",
            repair_enabled=True,
        )
        with tempfile.TemporaryDirectory() as tmp:
            sidecar = write_alignment_sidecar(package, Path(tmp) / "episode")
            loaded = load_alignment_package(sidecar)

        self.assertEqual(effective_secondary_text(loaded.segments[0]), "Generated.")
        accepted_generated = review_alignment_segment(loaded, "p_1")
        self.assertEqual(accepted_generated.segments[0].status, "reviewed")
        self.assertIsNotNone(accepted_generated.segments[0].generated_secondary)
        self.assertEqual(
            effective_secondary_text(accepted_generated.segments[0]),
            "Generated.",
        )
        reviewed = review_alignment_segment(loaded, "p_1", ["s1"])
        self.assertIsNone(reviewed.segments[0].generated_secondary)
        self.assertEqual(reviewed.segments[0].secondary_text, "Original.")
        self.assertEqual(reviewed.segments[0].disposition, "matched")

    def test_repair_requires_primary_to_be_media_language(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "primary.srt"
            secondary = Path(tmp) / "secondary.srt"
            primary.write_text(PRIMARY_SRT, encoding="utf-8")
            secondary.write_text(SECONDARY_SRT, encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "swap the subtitle inputs"):
                align_subtitles(
                    primary,
                    secondary,
                    primary_language="es",
                    secondary_language="en",
                    media_language="en",
                    repair_target_dialogue=True,
                )

    def test_invalid_repair_output_remains_reviewable(self) -> None:
        class InvalidRepairClient:
            def cache_identity(self):
                return {"provider": "test", "model": "bad-translator"}

            def align_batch(self, windows):
                return [AlignmentResult(window.primary.segment_id, [], 0.1) for window in windows]

            def align_batch_with_context(self, regions):
                return [
                    AlignmentResult(
                        region["target"]["primary"]["segment_id"],
                        [],
                        0.99,
                        "omitted",
                        disposition="omitted_dialogue",
                    )
                    for region in regions
                ]

            def repair_batch(self, regions, *, source_language, target_language):
                return [
                    SubtitleRepairResult(
                        region["repair_target"]["primary_id"],
                        "wrong language",
                        source_language,
                        0.99,
                        "invalid target language",
                    )
                    for region in regions
                ]

        with tempfile.TemporaryDirectory() as tmp:
            primary = Path(tmp) / "primary.srt"
            secondary = Path(tmp) / "secondary.srt"
            primary.write_text(PRIMARY_SRT, encoding="utf-8")
            secondary.write_text(SECONDARY_SRT, encoding="utf-8")
            package = align_subtitles(
                primary,
                secondary,
                primary_language="en",
                secondary_language="es",
                client=InvalidRepairClient(),
                repair_target_dialogue=True,
            )

        self.assertTrue(package.issues)
        self.assertTrue(all(segment.status == "needs_repair" for segment in package.segments))
        self.assertTrue(all(segment.generated_secondary is None for segment in package.segments))

    def test_alignment_response_schemas_only_classify_context_retries(self) -> None:
        initial_item = alignment_response_format()["json_schema"]["schema"]["properties"]["alignments"]["items"]
        context_item = context_alignment_response_format()["json_schema"]["schema"]["properties"]["alignments"]["items"]

        self.assertNotIn("disposition", initial_item["properties"])
        self.assertIn("disposition", context_item["properties"])
        self.assertIn("disposition", context_item["required"])

    def test_legacy_sidecar_infers_uncertain_for_empty_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "legacy.alignment.json"
            path.write_text(
                '{"schema_version":4,"primary_language":"en",'
                '"secondary_language":"es","segments":[{'
                '"segment_id":"p_1","primary_text":"Hello",'
                '"secondary_text":"","secondary_cue_ids":[],"status":"needs_review"}]}',
                encoding="utf-8",
            )
            loaded = load_alignment_package(path)

        self.assertEqual(loaded.schema_version, 4)
        self.assertEqual(loaded.segments[0].disposition, "uncertain")
        self.assertIsNone(loaded.segments[0].generated_secondary)


if __name__ == "__main__":
    unittest.main()
