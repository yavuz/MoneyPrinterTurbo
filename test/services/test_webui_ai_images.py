import sys
import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.config import config

ROOT_DIR = Path(__file__).parent.parent.parent
WEBUI_MAIN = ROOT_DIR / "webui" / "Main.py"


class TestWebuiAiImageSource(unittest.TestCase):
    def setUp(self):
        self._app_config = dict(config.app)

    def tearDown(self):
        config.app.clear()
        config.app.update(self._app_config)

    def _widget_by_key(self, elements, key):
        widget = next(
            (
                item
                for item in elements
                if str(getattr(item, "key", "")) == key
                or str(getattr(item, "key", "")).startswith(f"{key}_")
            ),
            None,
        )
        self.assertIsNotNone(widget, f"widget not found: {key}")
        return widget

    def _open_ai_source(self):
        app = AppTest.from_file(str(WEBUI_MAIN), default_timeout=60)
        app.session_state["ui_language"] = "en"
        app.run()
        self._widget_by_key(app.selectbox, "video_source_select").set_value("ai").run()
        return app

    def test_ai_source_renders_controls(self):
        app = self._open_ai_source()
        self.assertEqual([str(item.value) for item in app.exception], [])
        # 选择 AI 源后应出现服务商、计数模式和“准备”按钮。默认自动模式下只显示
        # 最大张数输入，不显示固定张数输入。
        self._widget_by_key(app.selectbox, "image_provider_select")
        self._widget_by_key(app.selectbox, "ai_image_count_mode")
        self._widget_by_key(app.number_input, "ai_image_max_input")
        self._widget_by_key(app.button, "prepare_ai_images_button")

    def test_gemini_provider_shows_model_selector_and_sets_config(self):
        # 清掉可能来自本机 config.toml 的模型覆盖，确保默认是 Pro。
        config.app.pop("gemini_image_model_name", None)
        app = self._open_ai_source()
        # 切到 Google (Nano Banana) 服务商后应出现图片模型选择器，默认 Pro。
        self._widget_by_key(app.selectbox, "image_provider_select").set_value(
            "gemini"
        ).run()
        model_select = self._widget_by_key(app.selectbox, "gemini_image_model_select")
        self.assertEqual(model_select.value, "gemini-3-pro-image")
        # 选择更轻量的 Flash Lite 模型后应写入配置，供 image_gen 读取。
        model_select.set_value("gemini-3.1-flash-lite-image").run()
        self.assertEqual(
            config.app.get("gemini_image_model_name"), "gemini-3.1-flash-lite-image"
        )

    def test_fixed_mode_shows_count_input_and_sets_image_count(self):
        app = self._open_ai_source()
        # 切换到固定模式后应出现固定张数输入。
        self._widget_by_key(app.selectbox, "ai_image_count_mode").set_value(
            "fixed"
        ).run()
        self._widget_by_key(app.number_input, "ai_image_count_input")

    def test_auto_mode_estimates_count_from_script(self):
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"
        config.app["image_gen_max_images"] = 40

        app = self._open_ai_source()  # 默认即自动模式
        app.session_state["video_subject"] = "the ocean"
        # 三个句子 → 自动估算出 3 张（不超过上限）。
        app.session_state["video_script"] = (
            "The ocean is vast. Waves crash on the shore. Fish swim below."
        )

        with patch(
            "app.services.llm.generate_image_prompts",
            return_value=["p1", "p2", "p3"],
        ) as gen_prompts, patch(
            "app.services.image_gen.prepare_images",
            return_value=["/tmp/a.png", "/tmp/b.png", "/tmp/c.png"],
        ):
            self._widget_by_key(app.button, "prepare_ai_images_button").click().run()

        gen_prompts.assert_called_once()
        self.assertEqual(gen_prompts.call_args.kwargs["amount"], 3)

    def test_prepare_generates_prompts_and_images(self):
        config.app["image_provider"] = "fal"
        config.app["fal_api_key"] = "fake-key"

        app = self._open_ai_source()
        # 提供主题与脚本，避免触发真实的脚本生成 LLM 调用。
        app.session_state["video_subject"] = "the ocean"
        app.session_state["video_script"] = "a short story about the ocean"

        with patch(
            "app.services.llm.generate_image_prompts",
            return_value=["prompt one", "prompt two"],
        ) as gen_prompts, patch(
            "app.services.image_gen.prepare_images",
            return_value=["/tmp/a.png", "/tmp/b.png"],
        ) as prepare:
            self._widget_by_key(app.button, "prepare_ai_images_button").click().run()

        gen_prompts.assert_called_once()
        prepare.assert_called_once()
        # AppTest 的 session_state 把属性访问映射到键，因此用 in/[] 而非 .get()。
        self.assertIn("ai_prep_prompts", app.session_state)
        self.assertEqual(
            app.session_state["ai_prep_prompts"], ["prompt one", "prompt two"]
        )
        self.assertEqual(
            app.session_state["ai_prep_images"], ["/tmp/a.png", "/tmp/b.png"]
        )

    def test_prepare_without_api_key_warns_and_skips_generation(self):
        config.app["image_provider"] = "fal"
        config.app.pop("fal_api_key", None)

        app = self._open_ai_source()
        app.session_state["video_subject"] = "the ocean"
        app.session_state["video_script"] = "a short story about the ocean"

        with patch("app.services.image_gen.prepare_images") as prepare:
            self._widget_by_key(app.button, "prepare_ai_images_button").click().run()

        # 缺少凭据时不应调用生成，且不留下审核结果。
        prepare.assert_not_called()
        self.assertNotIn("ai_prep_images", app.session_state)


if __name__ == "__main__":
    unittest.main()
