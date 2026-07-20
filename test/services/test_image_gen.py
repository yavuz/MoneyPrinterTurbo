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
from app.services import image_gen


def _fake_response(json_data=None, content=b""):
    return SimpleNamespace(
        raise_for_status=lambda: None,
        json=lambda: (json_data or {}),
        content=content,
    )


class TestImageGenBase(unittest.TestCase):
    def setUp(self):
        self.original_app_config = dict(config.app)
        self.original_proxy_config = dict(config.proxy)
        self._tmp = tempfile.TemporaryDirectory()
        self.cache_dir = os.path.join(self._tmp.name, "cache_images")

        # 把内容寻址缓存目录指向临时目录，避免污染真实 storage 目录。
        patcher = patch(
            "app.services.image_gen.utils.storage_dir",
            side_effect=self._fake_storage_dir,
        )
        self.addCleanup(patcher.stop)
        patcher.start()

        config.proxy.clear()

    def _fake_storage_dir(self, sub_dir="", create=False):
        d = os.path.join(self._tmp.name, sub_dir) if sub_dir else self._tmp.name
        if create and not os.path.exists(d):
            os.makedirs(d)
        return d

    def tearDown(self):
        config.app.clear()
        config.app.update(self.original_app_config)
        config.proxy.clear()
        config.proxy.update(self.original_proxy_config)
        self._tmp.cleanup()

    def _cache_path_for(self, prompt, provider="fal",
                        model="fal-ai/flux/schnell", resolution="1080x1920"):
        return image_gen._cache_path(
            self.cache_dir, provider, model, resolution, prompt
        )


class TestGenerateImagesOrchestration(TestImageGenBase):
    def test_resume_skips_cached_images(self):
        """已缓存的图片不应重新生成——这正是断点续跑的核心。"""
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        os.makedirs(self.cache_dir, exist_ok=True)

        prompts = ["a cached scene", "a fresh scene"]
        # 预先写入第一个提示词对应的缓存文件。
        with open(self._cache_path_for(prompts[0]), "wb") as f:
            f.write(b"cached-png")

        with patch(
            "app.services.image_gen._fal_generate", return_value=b"new-png"
        ) as fal:
            paths = image_gen.generate_images("t1", prompts)

        # 只为未缓存的第二个提示词调用一次生成。
        self.assertEqual(fal.call_count, 1)
        self.assertEqual(len(paths), 2)
        for p in paths:
            self.assertTrue(os.path.exists(p))

    def test_partial_failure_raises_and_preserves_successful_cache(self):
        """部分失败时抛错，但成功的图片必须留在缓存中以便重跑续接。"""
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        config.app["image_gen_max_concurrency"] = 1
        config.app["image_gen_max_retries"] = 0

        prompts = ["good scene", "bad scene"]

        def fake_fal(prompt, *args, **kwargs):
            if "good" in prompt:
                return b"good-png"
            raise RuntimeError("provider exploded")

        with patch("app.services.image_gen._fal_generate", side_effect=fake_fal):
            with self.assertRaises(image_gen.ImageGenError):
                image_gen.generate_images("t2", prompts)

        # 成功的图片已落盘缓存，失败的没有。
        self.assertTrue(os.path.exists(self._cache_path_for("good scene")))
        self.assertFalse(os.path.exists(self._cache_path_for("bad scene")))

    def test_missing_api_key_raises_config_error(self):
        config.app["image_provider"] = "fal"
        config.app.pop("fal_api_key", None)

        with self.assertRaises(image_gen.ImageGenConfigError):
            image_gen.generate_images("t3", ["some scene"])

    def test_unsupported_provider_raises_config_error(self):
        config.app["image_provider"] = "midjourney"
        with self.assertRaises(image_gen.ImageGenConfigError):
            image_gen.generate_images("t4", ["some scene"])

    def test_empty_prompts_raises(self):
        with self.assertRaises(image_gen.ImageGenError):
            image_gen.generate_images("t5", [])


class TestCacheKey(TestImageGenBase):
    def test_cache_path_is_stable_and_provider_sensitive(self):
        a = self._cache_path_for("same prompt", provider="fal")
        b = self._cache_path_for("same prompt", provider="fal")
        c = self._cache_path_for("same prompt", provider="replicate",
                                 model="black-forest-labs/flux-schnell")
        d = self._cache_path_for("different prompt", provider="fal")

        self.assertEqual(a, b)  # 相同输入 → 稳定路径（缓存可命中）
        self.assertNotEqual(a, c)  # provider 不同 → 不同缓存
        self.assertNotEqual(a, d)  # prompt 不同 → 不同缓存


class TestRegenerateAndPrepare(TestImageGenBase):
    def test_image_cache_path_matches_generate_location(self):
        config.app["image_provider"] = "fal"
        self.assertEqual(
            image_gen.image_cache_path("a scene"),
            self._cache_path_for("a scene"),
        )

    def test_regenerate_one_overwrites_existing_cache(self):
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        os.makedirs(self.cache_dir, exist_ok=True)
        dest = image_gen.image_cache_path("a scene")
        with open(dest, "wb") as f:
            f.write(b"OLD")

        with patch("app.services.image_gen._fal_generate", return_value=b"NEW"):
            out = image_gen.regenerate_one("a scene")

        self.assertEqual(out, dest)
        with open(dest, "rb") as f:
            self.assertEqual(f.read(), b"NEW")

    def test_prepare_images_returns_aligned_list_tolerating_failures(self):
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        config.app["image_gen_max_retries"] = 0

        def fake_fal(prompt, *args, **kwargs):
            if "good" in prompt:
                return b"ok-png"
            raise RuntimeError("provider exploded")

        with patch("app.services.image_gen._fal_generate", side_effect=fake_fal):
            result = image_gen.prepare_images(["good scene", "bad scene"])

        # 与提示词等长、对齐；失败项为 None，且不抛异常（供逐张重试）。
        self.assertEqual(len(result), 2)
        self.assertIsNotNone(result[0])
        self.assertIsNone(result[1])

    def test_prepare_images_empty_returns_empty(self):
        self.assertEqual(image_gen.prepare_images([]), [])


class TestFalHttpFlow(TestImageGenBase):
    def test_fal_queue_flow_downloads_and_caches(self):
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        config.app["image_gen_poll_interval"] = 0

        def fake_post(url, **kwargs):
            return _fake_response(
                {
                    "status_url": "https://queue.fal.run/status/1",
                    "response_url": "https://queue.fal.run/response/1",
                }
            )

        def fake_get(url, **kwargs):
            if "status" in url:
                return _fake_response({"status": "COMPLETED"})
            if "response" in url:
                return _fake_response({"images": [{"url": "https://img/x.png"}]})
            return _fake_response(content=b"PNGDATA")

        with patch("app.services.image_gen.requests.post", side_effect=fake_post), \
             patch("app.services.image_gen.requests.get", side_effect=fake_get), \
             patch("app.services.image_gen.time.sleep"):
            paths = image_gen.generate_images("t6", ["a scene"])

        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as f:
            self.assertEqual(f.read(), b"PNGDATA")

    def test_fal_timeout_raises(self):
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        config.app["image_gen_timeout"] = 0  # 立即超预算
        config.app["image_gen_max_retries"] = 0
        config.app["image_gen_poll_interval"] = 0

        def fake_post(url, **kwargs):
            return _fake_response(
                {
                    "status_url": "https://queue.fal.run/status/1",
                    "response_url": "https://queue.fal.run/response/1",
                }
            )

        def fake_get(url, **kwargs):
            # 永远返回未完成，配合 timeout=0 触发超时保护。
            return _fake_response({"status": "IN_PROGRESS"})

        with patch("app.services.image_gen.requests.post", side_effect=fake_post), \
             patch("app.services.image_gen.requests.get", side_effect=fake_get), \
             patch("app.services.image_gen.time.sleep"):
            with self.assertRaises(image_gen.ImageGenError):
                image_gen.generate_images("t7", ["a scene"])


class TestReplicateHttpFlow(TestImageGenBase):
    def test_replicate_synchronous_success(self):
        config.app["image_provider"] = "replicate"
        config.app["replicate_api_token"] = "fake-token"

        def fake_post(url, **kwargs):
            # Prefer: wait 命中，直接返回成功结果。
            return _fake_response(
                {
                    "status": "succeeded",
                    "output": ["https://img/r.png"],
                    "urls": {"get": "https://api.replicate.com/get/1"},
                }
            )

        def fake_get(url, **kwargs):
            return _fake_response(content=b"REPLICATE-PNG")

        with patch("app.services.image_gen.requests.post", side_effect=fake_post), \
             patch("app.services.image_gen.requests.get", side_effect=fake_get):
            paths = image_gen.generate_images("t8", ["a scene"])

        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as f:
            self.assertEqual(f.read(), b"REPLICATE-PNG")

    def test_replicate_failed_status_raises(self):
        config.app["image_provider"] = "replicate"
        config.app["replicate_api_token"] = "fake-token"
        config.app["image_gen_max_retries"] = 0

        def fake_post(url, **kwargs):
            return _fake_response({"status": "failed", "urls": {"get": "x"}})

        with patch("app.services.image_gen.requests.post", side_effect=fake_post):
            with self.assertRaises(image_gen.ImageGenError):
                image_gen.generate_images("t9", ["a scene"])


class TestGeminiHttpFlow(TestImageGenBase):
    def test_gemini_synchronous_success_decodes_inline_image(self):
        config.app["image_provider"] = "gemini"
        config.app["gemini_api_key"] = "fake-key"
        # 显式清掉可能来自本机 config.toml 的模型覆盖，确保断言的是默认模型。
        config.app.pop("gemini_image_model_name", None)
        raw = b"NANO-BANANA-PNG"

        captured = {}

        def fake_post(url, **kwargs):
            captured["url"] = url
            captured["json"] = kwargs.get("json")
            return _fake_response(
                {
                    "candidates": [
                        {
                            "content": {
                                "parts": [
                                    {
                                        "inlineData": {
                                            "mimeType": "image/png",
                                            "data": base64.b64encode(raw).decode(),
                                        }
                                    }
                                ]
                            }
                        }
                    ]
                }
            )

        with patch("app.services.image_gen.requests.post", side_effect=fake_post):
            paths = image_gen.generate_images("t10", ["a scene"])

        self.assertEqual(len(paths), 1)
        with open(paths[0], "rb") as f:
            self.assertEqual(f.read(), raw)
        # 默认走 Nano Banana Pro 模型，并把画幅比例透传给 imageConfig。
        self.assertIn("gemini-3-pro-image", captured["url"])
        self.assertEqual(
            captured["json"]["generationConfig"]["imageConfig"]["aspectRatio"],
            "9:16",
        )

    def test_gemini_missing_api_key_raises_config_error(self):
        config.app["image_provider"] = "gemini"
        config.app.pop("gemini_api_key", None)

        with self.assertRaises(image_gen.ImageGenConfigError):
            image_gen.generate_images("t11", ["a scene"])

    def test_gemini_no_inline_image_raises(self):
        config.app["image_provider"] = "gemini"
        config.app["gemini_api_key"] = "fake-key"
        config.app["image_gen_max_retries"] = 0

        def fake_post(url, **kwargs):
            # 只有文字、没有图片的响应应当被判为失败。
            return _fake_response(
                {"candidates": [{"content": {"parts": [{"text": "no image"}]}}]}
            )

        with patch("app.services.image_gen.requests.post", side_effect=fake_post):
            with self.assertRaises(image_gen.ImageGenError):
                image_gen.generate_images("t12", ["a scene"])


if __name__ == "__main__":
    unittest.main()
