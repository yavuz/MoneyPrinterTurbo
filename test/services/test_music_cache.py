import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services import music_cache


class TestMusicCache(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        patcher = patch(
            "app.services.music_cache.utils.storage_dir",
            side_effect=self._fake_storage_dir,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

    def _fake_storage_dir(self, sub_dir="", create=False):
        d = os.path.join(self._tmp.name, sub_dir) if sub_dir else self._tmp.name
        if create and not os.path.exists(d):
            os.makedirs(d)
        return d

    def _write(self, path, data=b"MUSIC"):
        with open(path, "wb") as f:
            f.write(data)

    def test_store_then_get_hits_same_key(self):
        src = os.path.join(self._tmp.name, "src.mp3")
        self._write(src, b"LYRIA-AUDIO")

        dest = music_cache.store_music(src, "lyria", "lyria-3-pro-preview", "calm", 30)
        self.assertTrue(os.path.exists(dest))

        hit = music_cache.get_cached_music("lyria", "lyria-3-pro-preview", "calm", 30)
        self.assertEqual(hit, dest)
        with open(hit, "rb") as f:
            self.assertEqual(f.read(), b"LYRIA-AUDIO")

    def test_key_is_sensitive_to_prompt_model_and_duration(self):
        src = os.path.join(self._tmp.name, "src.mp3")
        self._write(src)
        music_cache.store_music(src, "lyria", "lyria-3-pro-preview", "calm", 30)

        # 不同提示词 / 模型 / 时长都应视为不同缓存条目。
        self.assertIsNone(
            music_cache.get_cached_music("lyria", "lyria-3-pro-preview", "epic", 30)
        )
        self.assertIsNone(
            music_cache.get_cached_music("lyria", "lyria-3-clip-preview", "calm", 30)
        )
        self.assertIsNone(
            music_cache.get_cached_music("lyria", "lyria-3-pro-preview", "calm", 45)
        )

    def test_duration_rounds_to_seconds(self):
        src = os.path.join(self._tmp.name, "src.mp3")
        self._write(src)
        music_cache.store_music(src, "lyria", "m", "calm", 30)
        # 30.4 与 30 归一化到同一秒数，应命中同一条目。
        self.assertIsNotNone(music_cache.get_cached_music("lyria", "m", "calm", 30.4))

    def test_empty_source_not_treated_as_hit(self):
        src = os.path.join(self._tmp.name, "empty.mp3")
        self._write(src, b"")
        dest = music_cache.store_music(src, "lyria", "m", "calm", 30)
        # 空文件不应被后续运行误判为有效缓存。
        self.assertIsNone(music_cache.get_cached_music("lyria", "m", "calm", 30))
        self.assertTrue(os.path.exists(dest))

    def test_resolve_auto_prompt_caches_and_reuses(self):
        calls = {"n": 0}

        def generator():
            calls["n"] += 1
            return "soft ambient piano, slow tempo, peaceful"

        first = music_cache.resolve_auto_prompt("lyria", "calm morning", "wake up", generator)
        second = music_cache.resolve_auto_prompt("lyria", "calm morning", "wake up", generator)

        self.assertEqual(first, "soft ambient piano, slow tempo, peaceful")
        self.assertEqual(second, first)
        # 第二次命中缓存，不应再次调用生成器。
        self.assertEqual(calls["n"], 1)

    def test_resolve_auto_prompt_does_not_cache_empty_or_failure(self):
        # 生成器返回空：不缓存，下次仍会重试。
        self.assertEqual(
            music_cache.resolve_auto_prompt("lyria", "s", "c", lambda: ""), ""
        )

        def boom():
            raise RuntimeError("llm down")

        self.assertEqual(music_cache.resolve_auto_prompt("lyria", "s", "c", boom), "")

        calls = {"n": 0}

        def generator():
            calls["n"] += 1
            return "epic score"

        # 之前的空/失败都没写缓存，这次应真正调用生成器。
        self.assertEqual(music_cache.resolve_auto_prompt("lyria", "s", "c", generator), "epic score")
        self.assertEqual(calls["n"], 1)

    def test_resolve_auto_prompt_empty_inputs_skip_generator(self):
        with patch.object(music_cache, "logger"):
            called = {"n": 0}
            music_cache.resolve_auto_prompt("lyria", "  ", "", lambda: called.update(n=1) or "x")
            self.assertEqual(called["n"], 0)


if __name__ == "__main__":
    unittest.main()
