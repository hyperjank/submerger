import tempfile
import unittest
from pathlib import Path

from submerger.alignment import (
    AlignmentResult,
    AlignmentPackage,
    AlignmentValidator,
    CandidateWindow,
    carry_forward_human_reviews,
    HeuristicAlignmentClient,
    PairedSegment,
    PrimarySegment,
    PrimarySegmenter,
    SecondaryWindowBuilder,
    SubtitleDocument,
    align_subtitles,
    filter_dialogue_document,
    has_terminal_sentence_end,
    load_alignment_cache,
    load_alignment_package,
    review_alignment_segment,
    save_alignment_cache,
    tracks_from_alignment_package,
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

    def test_validator_flags_reuse_low_confidence_and_order(self) -> None:
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
        self.assertIn("reused secondary cue ids: 1", validated[1].problems)
        self.assertIn("non-monotonic secondary order", validated[1].problems)
        self.assertEqual(validated[1].accepted_secondary_cue_ids, ["2"])

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


if __name__ == "__main__":
    unittest.main()
