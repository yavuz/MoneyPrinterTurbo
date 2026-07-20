import base64
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config
from app.services import lyria


def _fake_response(payload=None, status_code=200, text=""):
    return SimpleNamespace(
        status_code=status_code,
        ok=200 <= status_code < 300,
        reason="OK" if 200 <= status_code < 300 else "error",
        text=text,
        json=lambda: (payload if payload is not None else {}),
    )


def _audio_payload(raw: bytes, mime="audio/mpeg"):
    return {
        "steps": [
            {
                "type": "model_output",
                "content": [
                    {"type": "text", "text": "some structure"},
                    {
                        "type": "audio",
                        "data": base64.b64encode(raw).decode(),
                        "mime_type": mime,
                    },
                ],
            }
        ]
    }


class TestLyriaConfig(unittest.TestCase):
    def setUp(self):
        self._app = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self._app)

    def test_api_key_prefers_config_and_falls_back_to_env(self):
        config.app["gemini_api_key"] = "config-key"
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            self.assertEqual(lyria.get_api_key(), "config-key")

        config.app["gemini_api_key"] = ""
        with patch.dict(os.environ, {"GEMINI_API_KEY": "env-key"}):
            self.assertEqual(lyria.get_api_key(), "env-key")

    def test_model_and_timeout_reject_invalid_configuration(self):
        cases = [
            ({"lyria_model_id": "lyria-3-clip-preview"}, "lyria-3-clip-preview", (15, 600)),
            ({"lyria_model_id": "unknown", "lyria_timeout": 0.2}, "lyria-3-pro-preview", (15, 1)),
            ({"lyria_model_id": "", "lyria_timeout": float("inf")}, "lyria-3-pro-preview", (15, 600)),
            ({"lyria_timeout": 2000}, "lyria-3-pro-preview", (15, 1800)),
        ]
        for configured, expected_model, expected_timeout in cases:
            with self.subTest(configured=configured):
                config.app.clear()
                config.app.update(configured)
                self.assertEqual(lyria._model_id(), expected_model)
                self.assertEqual(lyria._request_timeout(), expected_timeout)


class TestLyriaExtractAudio(unittest.TestCase):
    def test_extracts_first_audio_block(self):
        data, mime = lyria._extract_audio(_audio_payload(b"SONG"))
        self.assertEqual(data, b"SONG")
        self.assertEqual(mime, "audio/mpeg")

    def test_no_audio_block_raises(self):
        payload = {"steps": [{"content": [{"type": "text", "text": "no audio"}]}]}
        with self.assertRaises(lyria.LyriaError):
            lyria._extract_audio(payload)

    def test_undecodable_audio_raises(self):
        payload = {"steps": [{"content": [{"type": "audio", "data": "!!not-base64!!"}]}]}
        with self.assertRaises(lyria.LyriaError):
            lyria._extract_audio(payload)


class TestLyriaGenerateBgm(unittest.TestCase):
    def setUp(self):
        self._app = dict(config.app)
        self._proxy = dict(config.proxy)
        config.proxy.clear()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)

    def tearDown(self):
        config.app.clear()
        config.app.update(self._app)
        config.proxy.clear()
        config.proxy.update(self._proxy)

    def test_missing_api_key_raises_auth_error(self):
        config.app["gemini_api_key"] = ""
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(lyria.LyriaAuthenticationError):
                lyria.generate_bgm("in.mp4", "out.mp3", 30.0, prompt="calm")

    def test_invalid_duration_raises(self):
        config.app["gemini_api_key"] = "k"
        with self.assertRaises(lyria.LyriaError):
            lyria.generate_bgm("in.mp4", "out.mp3", 0, prompt="calm")

    def test_success_requests_decodes_and_fits_duration(self):
        config.app["gemini_api_key"] = "k"
        output_path = os.path.join(self._tmp.name, "lyria-bgm.mp3")
        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            captured["headers"] = kwargs.get("headers")
            return _fake_response(_audio_payload(b"RAWAUDIO"))

        def fake_fit(src_path, out_path, duration):
            captured["fit_duration"] = duration
            captured["raw_existed"] = os.path.exists(src_path)
            with open(out_path, "wb") as f:
                f.write(b"FITTED")
            return out_path

        with patch("app.services.lyria.requests.post", side_effect=fake_post), \
             patch("app.services.lyria.bgm_service.validate_audio_file"), \
             patch("app.services.lyria._fit_to_duration", side_effect=fake_fit):
            result = lyria.generate_bgm("in.mp4", output_path, 42.0, prompt="calm piano")

        self.assertEqual(result, output_path)
        self.assertTrue(os.path.exists(output_path))
        # 请求命中 Interactions 端点，模型与提示词正确透传。
        self.assertIn("/v1beta/interactions", captured["url"])
        self.assertEqual(captured["json"]["model"], "lyria-3-pro-preview")
        self.assertEqual(captured["json"]["input"], "calm piano")
        self.assertEqual(captured["headers"]["x-goog-api-key"], "k")
        # 时长适配拿到目标时长，且解码后的原始音频文件确实落盘供 FFmpeg 处理。
        self.assertEqual(captured["fit_duration"], 42.0)
        self.assertTrue(captured["raw_existed"])
        # 中间原始文件在返回前已清理，只留下适配后的成品。
        leftovers = [n for n in os.listdir(self._tmp.name) if n.startswith(".lyria-music-")]
        self.assertEqual(leftovers, [])

    def test_empty_prompt_uses_default(self):
        config.app["gemini_api_key"] = "k"
        output_path = os.path.join(self._tmp.name, "lyria-bgm.mp3")
        captured = {}

        def fake_post(url, **kwargs):
            captured["json"] = kwargs.get("json")
            return _fake_response(_audio_payload(b"RAWAUDIO"))

        with patch("app.services.lyria.requests.post", side_effect=fake_post), \
             patch("app.services.lyria.bgm_service.validate_audio_file"), \
             patch("app.services.lyria._fit_to_duration",
                   side_effect=lambda s, o, d: (open(o, "wb").write(b"x"), o)[1]):
            lyria.generate_bgm("in.mp4", output_path, 20.0, prompt="   ")

        self.assertEqual(captured["json"]["input"], lyria.DEFAULT_PROMPT)

    def test_http_error_raises_lyria_error(self):
        config.app["gemini_api_key"] = "k"

        def fake_post(url, **kwargs):
            return _fake_response(status_code=500, text="server error")

        with patch("app.services.lyria.requests.post", side_effect=fake_post):
            with self.assertRaises(lyria.LyriaError):
                lyria.generate_bgm("in.mp4", "out.mp3", 20.0, prompt="calm")

    def test_401_raises_auth_error(self):
        config.app["gemini_api_key"] = "k"

        def fake_post(url, **kwargs):
            return _fake_response(status_code=401, text="unauthorized")

        with patch("app.services.lyria.requests.post", side_effect=fake_post):
            with self.assertRaises(lyria.LyriaAuthenticationError):
                lyria.generate_bgm("in.mp4", "out.mp3", 20.0, prompt="calm")

    def test_content_block_raises_specific_error(self):
        config.app["gemini_api_key"] = "k"

        def fake_post(url, **kwargs):
            return _fake_response(
                status_code=400,
                text='{"error":{"message":"blocked","code":"content_blocked"}}',
            )

        with patch("app.services.lyria.requests.post", side_effect=fake_post):
            with self.assertRaises(lyria.LyriaContentBlockedError):
                lyria._request_audio("some risky prompt")

    def test_blocked_prompt_falls_back_to_default(self):
        """被内容策略拦截时用中性默认提示词重试，用户仍能拿到配乐。"""
        config.app["gemini_api_key"] = "k"
        output_path = os.path.join(self._tmp.name, "lyria-bgm.mp3")
        sent_prompts = []

        def fake_post(url, **kwargs):
            prompt = kwargs["json"]["input"]
            sent_prompts.append(prompt)
            # 第一个（用户/自动）提示词被拦截，中性默认提示词通过。
            if prompt != lyria.DEFAULT_PROMPT:
                return _fake_response(
                    status_code=400,
                    text='{"error":{"code":"content_blocked"}}',
                )
            return _fake_response(_audio_payload(b"RAWAUDIO"))

        with patch("app.services.lyria.requests.post", side_effect=fake_post), \
             patch("app.services.lyria.bgm_service.validate_audio_file"), \
             patch("app.services.lyria._fit_to_duration",
                   side_effect=lambda s, o, d: (open(o, "wb").write(b"x"), o)[1]):
            result = lyria.generate_bgm(
                "in.mp4", output_path, 20.0, prompt="in the style of some artist"
            )

        self.assertEqual(result, output_path)
        # 先试原始提示词、被拦截后再试默认提示词。
        self.assertEqual(sent_prompts[0], "in the style of some artist")
        self.assertEqual(sent_prompts[1], lyria.DEFAULT_PROMPT)

    def test_default_prompt_block_is_not_retried(self):
        """连中性默认提示词都被拦截时不再无谓重试，直接抛出。"""
        config.app["gemini_api_key"] = "k"
        calls = {"n": 0}

        def fake_post(url, **kwargs):
            calls["n"] += 1
            return _fake_response(
                status_code=400, text='{"error":{"code":"content_blocked"}}'
            )

        with patch("app.services.lyria.requests.post", side_effect=fake_post):
            with self.assertRaises(lyria.LyriaContentBlockedError):
                lyria.generate_bgm(
                    "in.mp4", "out.mp3", 20.0, prompt=lyria.DEFAULT_PROMPT
                )
        self.assertEqual(calls["n"], 1)


if __name__ == "__main__":
    unittest.main()
