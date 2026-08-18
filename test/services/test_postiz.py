import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import requests

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from app.services.postiz import PostizService


_CONFIG_BASE = {
    "postiz_enabled": True,
    "postiz_api_key": "test-key",
    "postiz_api_base": "https://postiz.example.com/public/v1",
    "postiz_channels": ["tiktok", "youtube"],
    "postiz_post_type": "draft",
    "postiz_schedule_delay_minutes": 10,
    "postiz_auto_upload": True,
    "postiz_channel_settings": {},
}

_INTEGRATIONS = [
    {"id": "chan-tiktok", "identifier": "tiktok", "name": "TikTok", "disabled": False},
    {"id": "chan-yt", "identifier": "youtube", "name": "YouTube", "disabled": False},
    {"id": "chan-x", "identifier": "x", "name": "X", "disabled": False},
    {"id": "chan-old", "identifier": "tiktok", "name": "Old", "disabled": True},
]


def _json_response(payload):
    response = MagicMock()
    response.json.return_value = payload
    response.raise_for_status = MagicMock()
    return response


def _media_payload():
    return {
        "id": "media-1",
        "name": "video.mp4",
        "path": "https://uploads.postiz.com/video.mp4",
    }


class TestConfiguration(unittest.TestCase):
    @patch("app.services.postiz.config.app", {**_CONFIG_BASE, "postiz_enabled": False})
    def test_disabled_service_is_not_configured(self):
        self.assertFalse(PostizService().is_configured())

    @patch("app.services.postiz.config.app", {**_CONFIG_BASE, "postiz_api_key": ""})
    def test_missing_api_key_is_not_configured(self):
        self.assertFalse(PostizService().is_configured())

    @patch("app.services.postiz.config.app", {})
    def test_defaults_point_at_the_cloud_api(self):
        service = PostizService()
        self.assertEqual("https://api.postiz.com/public/v1", service.api_base)
        self.assertEqual("draft", service.post_type)

    @patch(
        "app.services.postiz.config.app",
        {**_CONFIG_BASE, "postiz_api_base": "https://self.hosted/public/v1/"},
    )
    def test_trailing_slash_is_stripped_from_base_url(self):
        self.assertEqual(
            "https://self.hosted/public/v1", PostizService().api_base
        )

    @patch(
        "app.services.postiz.config.app",
        {**_CONFIG_BASE, "postiz_post_type": "publish-now"},
    )
    def test_unknown_post_type_falls_back_to_draft(self):
        self.assertEqual("draft", PostizService().post_type)


class TestChannelResolution(unittest.TestCase):
    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    def test_platform_identifier_matches_connected_channels(self):
        matched, missing = PostizService().resolve_channels(
            ["tiktok"], _INTEGRATIONS
        )
        self.assertEqual(["chan-tiktok"], [item["id"] for item in matched])
        self.assertEqual([], missing)

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    def test_channel_id_can_be_used_directly(self):
        matched, missing = PostizService().resolve_channels(["chan-x"], _INTEGRATIONS)
        self.assertEqual(["chan-x"], [item["id"] for item in matched])
        self.assertEqual([], missing)

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    def test_disabled_channels_are_never_selected(self):
        matched, _ = PostizService().resolve_channels(["chan-old"], _INTEGRATIONS)
        self.assertEqual([], matched)

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    def test_unknown_selectors_are_reported(self):
        matched, missing = PostizService().resolve_channels(
            ["tiktok", "mastodon"], _INTEGRATIONS
        )
        self.assertEqual(["chan-tiktok"], [item["id"] for item in matched])
        self.assertEqual(["mastodon"], missing)

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    def test_overlapping_selectors_do_not_duplicate_channels(self):
        matched, _ = PostizService().resolve_channels(
            ["tiktok", "chan-tiktok"], _INTEGRATIONS
        )
        self.assertEqual(1, len(matched))


class TestUploadVideoGuards(unittest.TestCase):
    @patch("app.services.postiz.config.app", {**_CONFIG_BASE, "postiz_enabled": False})
    @patch("app.services.postiz.requests.post")
    def test_unconfigured_service_skips_request(self, mock_post):
        """功能未启用时不能意外上传文件或消耗第三方 API 配额。"""
        result = PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("not configured", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.postiz.config.app", {**_CONFIG_BASE, "postiz_channels": []})
    @patch("app.services.postiz.requests.post")
    def test_no_channels_skips_request(self, mock_post):
        result = PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("No Postiz channels", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.os.path.exists", return_value=False)
    @patch("app.services.postiz.requests.post")
    def test_missing_video_skips_request(self, mock_post, _exists):
        """本地成片不存在时应在发起网络请求前返回明确错误。"""
        result = PostizService().upload_video("/missing/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("Video file not found", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.os.path.exists", return_value=True)
    @patch("app.services.postiz.os.path.getsize", return_value=80 * 1024 * 1024)
    @patch("app.services.postiz.requests.post")
    @patch("app.services.postiz.requests.get")
    def test_oversized_video_fails_before_uploading(
        self, mock_get, mock_post, _size, _exists
    ):
        """Postiz 上传上限是 50MB，提前拦截好过传几分钟再失败。"""
        mock_get.return_value = _json_response(_INTEGRATIONS)

        result = PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("50 MB", result["error"])
        mock_post.assert_not_called()

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.os.path.exists", return_value=True)
    @patch("app.services.postiz.requests.get")
    def test_no_matching_channel_fails_before_uploading(self, mock_get, _exists):
        mock_get.return_value = _json_response(
            [{"id": "chan-x", "identifier": "x", "disabled": False}]
        )

        result = PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("No matching Postiz channels", result["error"])

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.os.path.exists", return_value=True)
    @patch("app.services.postiz.requests.get")
    def test_request_error_returns_failure(self, mock_get, _exists):
        """网络异常需要转换为稳定结果，不能让发布失败中断视频生成任务。"""
        mock_get.side_effect = requests.exceptions.Timeout("integrations timed out")

        result = PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("timed out", result["error"])

    @patch("app.services.postiz.config.app", _CONFIG_BASE)
    @patch("app.services.postiz.os.path.exists", return_value=True)
    @patch("app.services.postiz.os.path.getsize", return_value=1024)
    @patch("builtins.open", mock_open(read_data=b"fake"))
    @patch("app.services.postiz.requests.post")
    @patch("app.services.postiz.requests.get")
    def test_validation_error_body_is_surfaced(
        self, mock_get, mock_post, _size, _exists
    ):
        """Postiz 把字段校验错误放在响应体里，只报异常会丢掉定位信息。"""
        mock_get.return_value = _json_response(_INTEGRATIONS)
        response = MagicMock()
        response.text = '{"message":"settings.privacy_level is required"}'
        mock_post.side_effect = requests.exceptions.HTTPError(
            "400 Client Error", response=response
        )

        result = PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertFalse(result["success"])
        self.assertIn("privacy_level is required", result["error"])


class TestUploadVideoSuccess(unittest.TestCase):
    def _run(self, config_overrides=None, title="My video", youtube_extra=None):
        app_config = {**_CONFIG_BASE, **(config_overrides or {})}
        with (
            patch("app.services.postiz.config.app", app_config),
            patch("app.services.postiz.os.path.exists", return_value=True),
            patch("app.services.postiz.os.path.getsize", return_value=1024),
            patch("builtins.open", mock_open(read_data=b"fake")),
            patch("app.services.postiz.requests.get") as mock_get,
            patch("app.services.postiz.requests.post") as mock_post,
        ):
            mock_get.return_value = _json_response(_INTEGRATIONS)
            mock_post.side_effect = [
                _json_response(_media_payload()),
                _json_response([{"postId": "post-1", "integration": "chan-tiktok"}]),
            ]
            result = PostizService().upload_video(
                "/fake/v.mp4", title, youtube_extra=youtube_extra
            )
            create_payload = mock_post.call_args_list[1].kwargs["json"]
        return result, create_payload

    def test_successful_publish_reports_channels_and_posts(self):
        result, _ = self._run()

        self.assertTrue(result["success"])
        self.assertEqual("postiz", result["provider"])
        self.assertEqual(["post-1"], result["post_ids"])
        self.assertEqual(["tiktok", "youtube"], result["channels"])

    def test_media_is_uploaded_once_and_shared_by_all_channels(self):
        _, payload = self._run()

        self.assertEqual(2, len(payload["posts"]))
        for post in payload["posts"]:
            self.assertEqual(
                [{"id": "media-1", "path": "https://uploads.postiz.com/video.mp4"}],
                post["value"][0]["image"],
            )

    def test_settings_carry_the_platform_type(self):
        _, payload = self._run()

        self.assertEqual(
            ["tiktok", "youtube"],
            [post["settings"]["__type"] for post in payload["posts"]],
        )

    def test_configured_platform_settings_are_merged_in(self):
        _, payload = self._run(
            {
                "postiz_channel_settings": {
                    "tiktok": {"privacy_level": "PUBLIC_TO_EVERYONE"}
                }
            }
        )

        tiktok = payload["posts"][0]["settings"]
        self.assertEqual("tiktok", tiktok["__type"])
        self.assertEqual("PUBLIC_TO_EVERYONE", tiktok["privacy_level"])
        # 没有配置的频道不能被别的平台设置污染
        self.assertEqual({"__type": "youtube"}, payload["posts"][1]["settings"])

    def test_draft_is_the_default_post_type(self):
        _, payload = self._run()

        self.assertEqual("draft", payload["type"])
        self.assertTrue(payload["date"].endswith("Z"))

    def test_now_post_type_is_passed_through(self):
        _, payload = self._run({"postiz_post_type": "now"})

        self.assertEqual("now", payload["type"])

    def test_youtube_metadata_is_used_for_youtube_channels_only(self):
        _, payload = self._run(
            youtube_extra={
                "youtube_title": "YT title",
                "youtube_description": "YT description",
                "tags": ["shorts"],
            }
        )

        self.assertEqual("My video", payload["posts"][0]["value"][0]["content"])
        youtube_content = payload["posts"][1]["value"][0]["content"]
        self.assertIn("YT title", youtube_content)
        self.assertIn("YT description", youtube_content)
        self.assertIn("#shorts", youtube_content)

    def test_api_key_is_sent_without_a_bearer_prefix(self):
        with (
            patch("app.services.postiz.config.app", _CONFIG_BASE),
            patch("app.services.postiz.os.path.exists", return_value=True),
            patch("app.services.postiz.os.path.getsize", return_value=1024),
            patch("builtins.open", mock_open(read_data=b"fake")),
            patch("app.services.postiz.requests.get") as mock_get,
            patch("app.services.postiz.requests.post") as mock_post,
        ):
            mock_get.return_value = _json_response(_INTEGRATIONS)
            mock_post.side_effect = [
                _json_response(_media_payload()),
                _json_response([]),
            ]
            PostizService().upload_video("/fake/v.mp4", "Title")

        self.assertEqual(
            "test-key", mock_get.call_args.kwargs["headers"]["Authorization"]
        )

    def test_requests_target_the_configured_base_url(self):
        with (
            patch("app.services.postiz.config.app", _CONFIG_BASE),
            patch("app.services.postiz.os.path.exists", return_value=True),
            patch("app.services.postiz.os.path.getsize", return_value=1024),
            patch("builtins.open", mock_open(read_data=b"fake")),
            patch("app.services.postiz.requests.get") as mock_get,
            patch("app.services.postiz.requests.post") as mock_post,
        ):
            mock_get.return_value = _json_response(_INTEGRATIONS)
            mock_post.side_effect = [
                _json_response(_media_payload()),
                _json_response([]),
            ]
            PostizService().upload_video("/fake/v.mp4", "Title")

        base = "https://postiz.example.com/public/v1"
        self.assertEqual(f"{base}/integrations", mock_get.call_args.args[0])
        self.assertEqual(f"{base}/upload", mock_post.call_args_list[0].args[0])
        self.assertEqual(f"{base}/posts", mock_post.call_args_list[1].args[0])


class TestProviderSelection(unittest.TestCase):
    """task.py 依赖两个提供商模块暴露同样的接口。"""

    def test_both_providers_expose_the_same_entry_points(self):
        from app.services import postiz, upload_post

        for module in (postiz, upload_post):
            service = module.get_service()
            for name in ("is_configured", "upload_video"):
                self.assertTrue(callable(getattr(service, name)), f"{module}.{name}")
            for name in ("platforms", "auto_upload", "youtube_privacy_status"):
                self.assertTrue(hasattr(service, name), f"{module}.{name}")
            self.assertTrue(callable(module.cross_post_video))

    def test_configured_provider_is_selected(self):
        from app.services import postiz, task, upload_post

        with patch.dict(task.config.app, {"cross_post_provider": "postiz"}):
            self.assertIs(postiz, task.cross_post_module())
        with patch.dict(task.config.app, {"cross_post_provider": "upload_post"}):
            self.assertIs(upload_post, task.cross_post_module())

    def test_unknown_provider_falls_back_to_upload_post(self):
        from app.services import task, upload_post

        with patch.dict(task.config.app, {"cross_post_provider": "buffer"}):
            self.assertIs(upload_post, task.cross_post_module())


if __name__ == "__main__":
    unittest.main()
