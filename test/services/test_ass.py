import os
import tempfile
import unittest
from unittest.mock import Mock, patch

from app.services import ass
from app.utils import utils


def _font_path() -> str:
    return os.path.join(utils.font_dir(), "STHeitiMedium.ttc")


class TestTokenize(unittest.TestCase):
    def test_latin_text_is_split_on_whitespace(self):
        words = ass.tokenize("hello brave new world")
        self.assertEqual(
            ["hello", "brave", "new", "world"], [word.text for word in words]
        )
        self.assertEqual(["", " ", " ", " "], [word.separator for word in words])

    def test_cjk_text_is_split_per_character(self):
        words = ass.tokenize("人工智能")
        self.assertEqual(["人", "工", "智", "能"], [word.text for word in words])

    def test_mixed_text_keeps_latin_words_intact(self):
        words = ass.tokenize("聊聊 AI 技术")
        self.assertEqual(["聊", "聊", "AI", "技", "术"], [word.text for word in words])

    def test_punctuation_stays_attached_to_its_word(self):
        words = ass.tokenize("wait, really?")
        self.assertEqual(["wait,", "really?"], [word.text for word in words])


class TestWordTimes(unittest.TestCase):
    def test_times_are_distributed_and_cover_the_whole_cue(self):
        words = ass.tokenize("one two three")
        ass.distribute_word_times(words, 1.0, 4.0)
        self.assertEqual(1.0, words[0].start)
        self.assertEqual(4.0, words[-1].end)
        for previous, current in zip(words, words[1:]):
            self.assertAlmostEqual(previous.end, current.start)

    def test_longer_words_get_more_time(self):
        words = ass.tokenize("a considerably")
        ass.distribute_word_times(words, 0.0, 10.0)
        self.assertLess(words[0].end - words[0].start, words[1].end - words[1].start)

    def test_empty_word_list_is_ignored(self):
        ass.distribute_word_times([], 0.0, 1.0)


class TestWordTimestamps(unittest.TestCase):
    def _cue(self):
        cue = ass.Cue(start=0.0, end=2.0, text="one two")
        cue.words = ass.tokenize(cue.text)
        return cue

    def test_matching_timestamps_are_applied(self):
        cue = self._cue()
        applied = ass.apply_word_timestamps(
            cue,
            [
                {"text": "one", "start": 0.1, "end": 0.6},
                {"text": "two", "start": 0.9, "end": 1.8},
            ],
        )
        self.assertTrue(applied)
        self.assertAlmostEqual(0.1, cue.words[0].start)
        # 词间静音要被吸收掉，否则高亮块会闪断。
        self.assertAlmostEqual(0.9, cue.words[0].end)
        self.assertAlmostEqual(2.0, cue.words[1].end)

    def test_timestamps_outside_the_cue_window_are_ignored(self):
        cue = self._cue()
        self.assertFalse(
            ass.apply_word_timestamps(
                cue,
                [
                    {"text": "one", "start": 5.0, "end": 5.4},
                    {"text": "two", "start": 5.4, "end": 5.9},
                ],
            )
        )

    def test_word_count_mismatch_falls_back(self):
        cue = self._cue()
        self.assertFalse(
            ass.apply_word_timestamps(cue, [{"text": "one", "start": 0.1, "end": 0.6}])
        )


class TestColorAndTime(unittest.TestCase):
    def test_hex_color_is_converted_to_bgr(self):
        self.assertEqual("&H000000FF", ass.to_ass_color("#FF0000"))
        self.assertEqual("&H00FF0000", ass.to_ass_color("#0000FF"))

    def test_short_hex_is_expanded(self):
        self.assertEqual("&H000000FF", ass.to_ass_color("#F00"))

    def test_invalid_color_falls_back_to_default(self):
        self.assertEqual("&H00FFFFFF", ass.to_ass_color("not-a-color", "#FFFFFF"))

    def test_time_round_trip(self):
        self.assertEqual("0:00:01.50", ass.format_ass_time(1.5))
        self.assertEqual("1:01:01.00", ass.format_ass_time(3661.0))
        self.assertAlmostEqual(3661.5, ass.parse_srt_time("01:01:01,500"))

    def test_rounding_up_does_not_produce_100_centiseconds(self):
        self.assertEqual("0:00:02.00", ass.format_ass_time(1.999))


class TestEscaping(unittest.TestCase):
    def test_braces_and_backslashes_are_neutralized(self):
        self.assertEqual("(a) /b", ass.escape_ass_text("{a} \\b"))


class TestBuildCues(unittest.TestCase):
    def test_invalid_entries_are_skipped(self):
        cues = ass.build_cues(
            [
                (1, "00:00:00,000 --> 00:00:01,000", "hello"),
                (2, "00:00:01,000 --> 00:00:01,000", "zero length"),
                (3, "not a time range", "broken"),
                (4, "00:00:02,000 --> 00:00:03,000", "   "),
            ]
        )
        self.assertEqual(1, len(cues))
        self.assertEqual("hello", cues[0].text)


class TestBuildKaraokeAss(unittest.TestCase):
    def setUp(self):
        self.font = _font_path()
        self.items = [
            (1, "00:00:00,000 --> 00:00:02,000", "hello brave world"),
            (2, "00:00:02,000 --> 00:00:04,000", "人工智能"),
        ]

    def _build(self, tmp_path, **kwargs):
        output = os.path.join(tmp_path, "subs.ass")
        return ass.build_karaoke_ass(
            subtitle_items=self.items,
            output_path=output,
            font_path=self.font,
            font_size=60,
            video_width=1080,
            video_height=1920,
            **kwargs,
        )

    def test_one_event_per_word(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build(tmp, highlight_color="#E11D2E")
            content = open(path, encoding="utf-8").read()
            events = [
                line for line in content.splitlines() if line.startswith("Dialogue:")
            ]
            # 3 个英文词 + 4 个汉字
            self.assertEqual(7, len(events))
            # 每条事件都渲染整句，只有当前词切到高亮样式
            self.assertIn("{\\rHighlight}hello{\\r}", events[0])
            self.assertIn("brave world", events[0])

    def test_highlight_style_uses_an_opaque_box(self):
        """矩形色块靠 BorderStyle=3 实现，粗描边只能得到圆润色团。"""
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build(tmp, highlight_color="#E11D2E")
            style = [
                line
                for line in open(path, encoding="utf-8").read().splitlines()
                if line.startswith("Style: Highlight,")
            ][0]
            fields = style.split(",")
            # BorderStyle 字段：3 表示不透明色块
            self.assertEqual("3", fields[15])
            # 该模式下 OutlineColour 就是色块颜色；#E11D2E 的 BGR 写法是 2E1DE1
            self.assertEqual("&H002E1DE1", fields[5])
            # 色块内边距必须为正，否则文字会贴着色块边缘
            self.assertGreater(float(fields[16]), 0)

    def test_style_positions_follow_subtitle_position(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build(tmp, subtitle_position="top")
            style = [
                line
                for line in open(path, encoding="utf-8").read().splitlines()
                if line.startswith("Style:")
            ][0]
            # Alignment 字段：8 表示顶部居中
            self.assertEqual("8", style.split(",")[18])

    def test_wrapped_lines_are_separated_by_a_spacer_line(self):
        """
        折行字幕靠一行小字号硬空格拉开行距，直接用 `\\N` 两行会贴在一起。
        """
        items = [(1, "00:00:00,000 --> 00:00:02,000", "word " * 40)]
        with tempfile.TemporaryDirectory() as tmp:
            path = ass.build_karaoke_ass(
                subtitle_items=items,
                output_path=os.path.join(tmp, "subs.ass"),
                font_path=self.font,
                font_size=60,
                video_width=1080,
                video_height=1920,
            )
            event = [
                line
                for line in open(path, encoding="utf-8").read().splitlines()
                if line.startswith("Dialogue:")
            ][0]

        self.assertIn("\\N{\\fs24}\\h\\N{\\r}", event)

    def test_single_line_cues_get_no_spacer(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build(tmp)
            events = [
                line
                for line in open(path, encoding="utf-8").read().splitlines()
                if line.startswith("Dialogue:")
            ]

        self.assertNotIn("\\N", events[0])

    def test_no_usable_cues_returns_empty_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(
                "",
                ass.build_karaoke_ass(
                    subtitle_items=[],
                    output_path=os.path.join(tmp, "subs.ass"),
                    font_path=self.font,
                    font_size=60,
                    video_width=1080,
                    video_height=1920,
                ),
            )

    def test_word_timestamps_override_estimated_times(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = self._build(
                tmp,
                word_timestamps=[
                    {"text": "hello", "start": 0.0, "end": 1.5},
                    {"text": "brave", "start": 1.5, "end": 1.7},
                    {"text": "world", "start": 1.7, "end": 2.0},
                ],
            )
            first_event = [
                line
                for line in open(path, encoding="utf-8").read().splitlines()
                if line.startswith("Dialogue:")
            ][0]
            self.assertIn("0:00:00.00,0:00:01.50", first_event)


class TestFfmpegCapability(unittest.TestCase):
    def setUp(self):
        ass.resolve_ffmpeg_binary.cache_clear()
        ass._ffmpeg_has_ass_filter.cache_clear()

    def tearDown(self):
        ass.resolve_ffmpeg_binary.cache_clear()
        ass._ffmpeg_has_ass_filter.cache_clear()

    def test_filter_list_is_parsed(self):
        completed = Mock(
            returncode=0,
            stdout=(
                " ... ass               V->V       Render ASS subtitles.\n"
                " ... assdraw           V->V       Something else.\n"
            ),
        )
        with patch("app.services.ass.subprocess.run", return_value=completed):
            self.assertTrue(ass._ffmpeg_has_ass_filter("ffmpeg"))

    def test_missing_filter_is_detected(self):
        completed = Mock(
            returncode=0, stdout=" ... scale             V->V       Scale.\n"
        )
        with patch("app.services.ass.subprocess.run", return_value=completed):
            self.assertFalse(ass._ffmpeg_has_ass_filter("ffmpeg"))

    def test_is_supported_is_false_when_no_binary_has_the_filter(self):
        with patch("app.services.ass._ffmpeg_has_ass_filter", return_value=False):
            self.assertFalse(ass.is_supported())

    def test_burn_requires_a_capable_binary(self):
        with patch("app.services.ass.resolve_ffmpeg_binary", return_value=""):
            with self.assertRaises(RuntimeError):
                ass.burn_subtitles("in.mp4", "subs.ass", "out.mp4")


class TestAssFilePath(unittest.TestCase):
    def test_path_is_derived_from_the_output_video(self):
        self.assertTrue(
            ass.ass_file_path(os.path.join("dir", "video-1.mp4")).endswith(
                os.path.join("dir", "video-1.karaoke.ass")
            )
        )

    def test_parallel_outputs_do_not_collide(self):
        self.assertNotEqual(
            ass.ass_file_path("video-1.mp4"), ass.ass_file_path("video-2.mp4")
        )


if __name__ == "__main__":
    unittest.main()
