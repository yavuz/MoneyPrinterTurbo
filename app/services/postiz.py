"""
Postiz 跨平台发布集成。

Postiz（https://github.com/gitroomhq/postiz-app）是一个开源的社交媒体排期工具，
可以自托管。相比项目原有的 Upload-Post，它的优势是各平台的 OAuth 授权由 Postiz
自己维护：用户在 Postiz 界面里连接一次 TikTok/YouTube/Instagram，本项目只需要
按频道 id 投递内容，不必自己处理各平台的授权与刷新。

发布流程分两步（Public API v1）：
1. `POST /upload`：multipart 上传成片，返回 `{id, name, path, ...}`；
2. `POST /posts`：按频道创建投稿，媒体通过第 1 步返回的 `id`/`path` 引用。

本模块刻意与 `upload_post.UploadPostService` 保持同样的对外接口
（`is_configured` / `platforms` / `auto_upload` / `upload_video`），这样
`task.py` 里已经写好的发布队列、状态机和中断恢复逻辑可以原样复用。
"""

import os
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.parse import urlparse

import requests
from loguru import logger

from app.config import config

# Postiz 云服务的公开 API 地址；自托管时改成 `https://<后端域名>/public/v1`。
DEFAULT_API_BASE = "https://api.postiz.com/public/v1"

# Public API 的单文件上传上限。超过这个体积 Postiz 会直接拒绝，
# 与其等上传几分钟后失败，不如提前给出可读的报错。
MAX_UPLOAD_BYTES = 50 * 1024 * 1024

# `draft` 会把内容放进 Postiz 待办里等人工确认，`now` 立即发布，
# `schedule` 按给定时间排期。
VALID_POST_TYPES = ("draft", "schedule", "now")

_UPLOAD_TIMEOUT_SECONDS = 300
_REQUEST_TIMEOUT_SECONDS = 60

# 私有地址上的媒体链接各平台都拉不到（TikTok 等要求公网 HTTPS）。
# 自托管 Postiz 时这是最常见的踩坑点，检测到就明确警告。
_PRIVATE_HOSTS = ("localhost", "127.0.0.1", "0.0.0.0", "::1")


class PostizService:
    def __init__(self):
        self.api_key = str(config.app.get("postiz_api_key", "") or "")
        self.api_base = str(
            config.app.get("postiz_api_base", "") or DEFAULT_API_BASE
        ).rstrip("/")
        self.enabled = bool(config.app.get("postiz_enabled", False))
        # 与 upload_post 的 `platforms` 同名，方便 task.py 用同一套调度代码。
        self.platforms = list(config.app.get("postiz_channels", []) or [])
        self.auto_upload = bool(config.app.get("postiz_auto_upload", False))
        self.post_type = str(config.app.get("postiz_post_type", "draft") or "draft")
        self.schedule_delay_minutes = int(
            config.app.get("postiz_schedule_delay_minutes", 10) or 0
        )
        # 各平台的 settings 字段差异很大，且会随平台政策变化。这里不硬编码，
        # 由用户按 identifier 配置，运行时合并到 `__type` 之上。
        self.channel_settings = dict(config.app.get("postiz_channel_settings", {}) or {})
        # Postiz 不使用该字段，仅为与 upload_post 保持接口一致。
        self.youtube_privacy_status = config.app.get(
            "postiz_youtube_privacy_status", "public"
        )

        if self.post_type not in VALID_POST_TYPES:
            logger.warning(
                f"unsupported postiz_post_type: {self.post_type}, fallback to draft"
            )
            self.post_type = "draft"

    def is_configured(self) -> bool:
        return bool(self.api_key and self.enabled)

    def _headers(self) -> dict:
        # Postiz 的 Authorization 直接放裸 API Key，没有 Bearer 前缀。
        return {"Authorization": self.api_key}

    def list_integrations(self) -> list:
        """拉取 Postiz 里已连接的频道（Postiz 界面中称为 channel）。"""
        response = requests.get(
            f"{self.api_base}/integrations",
            headers=self._headers(),
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        payload = response.json()
        if not isinstance(payload, list):
            raise RuntimeError("Postiz returned an unexpected integrations payload")
        return payload

    def resolve_channels(self, selectors: list, integrations: list) -> tuple[list, list]:
        """
        把配置里的频道选择器解析成具体的 Postiz 频道。

        选择器既可以是频道 id，也可以是平台标识（`identifier`，例如 tiktok）。
        后者更好写，但一个平台可能连了多个账号，因此同一标识会展开成多个频道。
        返回 (命中的频道列表, 没有匹配到的选择器列表)。
        """
        available = [item for item in integrations if not item.get("disabled")]
        matched = []
        missing = []
        for selector in selectors:
            key = str(selector or "").strip()
            if not key:
                continue
            hits = [
                item
                for item in available
                if str(item.get("id", "")) == key
                or str(item.get("identifier", "")).lower() == key.lower()
            ]
            if not hits:
                missing.append(key)
                continue
            for hit in hits:
                if hit not in matched:
                    matched.append(hit)
        return matched, missing

    def upload_media(self, video_path: str) -> dict:
        """上传成片，返回 Postiz 的媒体对象。"""
        file_size = os.path.getsize(video_path)
        if file_size > MAX_UPLOAD_BYTES:
            raise RuntimeError(
                f"video is {file_size / 1024 / 1024:.1f} MB, which exceeds the "
                f"{MAX_UPLOAD_BYTES // 1024 // 1024} MB Postiz upload limit"
            )

        with open(video_path, "rb") as video_file:
            response = requests.post(
                f"{self.api_base}/upload",
                headers=self._headers(),
                files={"file": (os.path.basename(video_path), video_file, "video/mp4")},
                timeout=_UPLOAD_TIMEOUT_SECONDS,
            )
        response.raise_for_status()
        media = response.json()
        if not isinstance(media, dict) or not media.get("id"):
            raise RuntimeError("Postiz returned an unexpected upload payload")

        _warn_on_private_media_url(media.get("path", ""))
        return media

    def _post_date(self) -> str:
        moment = datetime.now(timezone.utc)
        if self.post_type == "schedule":
            moment += timedelta(minutes=max(0, self.schedule_delay_minutes))
        # Postiz 要求 UTC ISO 格式，即使 type 是 now 也必须带上 date。
        return moment.isoformat(timespec="milliseconds").replace("+00:00", "Z")

    def _channel_content(
        self, channel: dict, title: str, youtube_extra: Optional[dict]
    ) -> str:
        identifier = str(channel.get("identifier", "")).lower()
        if youtube_extra and identifier.startswith("youtube"):
            # YouTube 的标题和正文是分开的，这里把 LLM 生成的描述一并带上，
            # 比只发一个标题更接近可直接发布的状态。
            parts = [
                str(youtube_extra.get("youtube_title") or title or "").strip(),
                str(youtube_extra.get("youtube_description") or "").strip(),
            ]
            hashtags = youtube_extra.get("hashtags") or youtube_extra.get("tags") or []
            if hashtags:
                parts.append(" ".join(f"#{tag.lstrip('#')}" for tag in hashtags))
            content = "\n\n".join(part for part in parts if part)
            if content:
                return content
        return title

    def _channel_settings(self, channel: dict) -> dict:
        identifier = str(channel.get("identifier", ""))
        settings = {"__type": identifier}
        # 平台专属字段（TikTok 的 privacy_level、content_posting_method 等）由用户
        # 配置提供。Postiz 官方提供了一个可视化向导来生成这些 JSON，硬编码在这里
        # 只会随平台政策变化而过期。
        extra = self.channel_settings.get(identifier)
        if isinstance(extra, dict):
            settings.update(extra)
        return settings

    def create_post(self, channels: list, media: dict, title: str, youtube_extra=None):
        image_ref = {"id": media.get("id"), "path": media.get("path")}
        payload = {
            "type": self.post_type,
            "date": self._post_date(),
            "shortLink": False,
            "tags": [],
            "posts": [
                {
                    "integration": {"id": channel.get("id")},
                    "value": [
                        {
                            "content": self._channel_content(
                                channel, title, youtube_extra
                            ),
                            "image": [image_ref],
                        }
                    ],
                    "settings": self._channel_settings(channel),
                }
                for channel in channels
            ],
        }

        response = requests.post(
            f"{self.api_base}/posts",
            headers={**self._headers(), "Content-Type": "application/json"},
            json=payload,
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        return response.json()

    def upload_video(
        self,
        video_path: str,
        title: str,
        platforms: Optional[list] = None,
        privacy_level: str = "PUBLIC_TO_EVERYONE",
        youtube_extra: Optional[dict] = None,
    ) -> dict:
        """
        发布单个成片，返回结构与 `upload_post.UploadPostService.upload_video`
        一致，便于 task.py 用同一套结果判定逻辑。

        `privacy_level` 仅为保持接口一致：Postiz 的隐私设置属于各平台的
        `settings`，由 `postiz_channel_settings` 配置。
        """
        if not self.is_configured():
            logger.warning("Postiz is not configured. Skipping cross-post.")
            return {"success": False, "error": "Postiz not configured"}

        selectors = list(platforms) if platforms is not None else list(self.platforms)
        if not selectors:
            return {"success": False, "error": "No Postiz channels configured"}

        if not os.path.exists(video_path):
            logger.error(f"Video file not found: {video_path}")
            return {"success": False, "error": f"Video file not found: {video_path}"}

        try:
            integrations = self.list_integrations()
            channels, missing = self.resolve_channels(selectors, integrations)
            if missing:
                # 频道没连上或名字写错是最常见的配置问题，明确告诉用户是哪个。
                logger.warning(f"Postiz channels not found: {', '.join(missing)}")
            if not channels:
                return {
                    "success": False,
                    "error": f"No matching Postiz channels: {', '.join(selectors)}",
                }

            channel_names = [
                str(channel.get("identifier") or channel.get("name") or channel.get("id"))
                for channel in channels
            ]
            logger.info(
                f"Cross-posting video to {', '.join(channel_names)} via Postiz "
                f"({self.post_type})..."
            )

            media = self.upload_media(video_path)
            result = self.create_post(channels, media, title, youtube_extra)

            post_ids = [
                item.get("postId")
                for item in (result if isinstance(result, list) else [])
                if isinstance(item, dict) and item.get("postId")
            ]
            logger.success(
                f"✅ Video sent to Postiz as {self.post_type}. "
                f"channels: {len(channels)}, posts: {len(post_ids)}"
            )
            return {
                "success": True,
                "provider": "postiz",
                "post_type": self.post_type,
                "channels": channel_names,
                "post_ids": post_ids,
                "media_id": media.get("id"),
                "missing_channels": missing,
            }

        except requests.exceptions.RequestException as e:
            # Postiz 的校验错误（例如平台必填 settings 缺失）会放在响应体里，
            # 只打印异常本身会丢掉最关键的定位信息。
            detail = _response_detail(e)
            logger.error(f"Failed to cross-post video via Postiz: {detail}")
            return {"success": False, "error": detail}
        except (OSError, RuntimeError, ValueError) as e:
            logger.error(f"Failed to cross-post video via Postiz: {str(e)}")
            return {"success": False, "error": str(e)}


def _response_detail(exc: requests.exceptions.RequestException) -> str:
    response = getattr(exc, "response", None)
    body = ""
    if response is not None:
        body = (response.text or "").strip()[:500]
    return f"{str(exc)}: {body}" if body else str(exc)


def _warn_on_private_media_url(path: str) -> None:
    host = (urlparse(str(path or "")).hostname or "").lower()
    if not host:
        return
    if host in _PRIVATE_HOSTS or host.endswith(".local"):
        logger.warning(
            f"Postiz returned a media URL on a private host ({host}). "
            "TikTok, Instagram and YouTube fetch media over the public internet "
            "and will reject it. Configure a public storage/domain in Postiz."
        )


# Singleton instance
postiz_service = PostizService()


def get_service() -> PostizService:
    """跨平台发布提供商的统一入口，见 task.cross_post_module()。"""
    return postiz_service


def cross_post_video(
    video_path: str,
    title: str,
    platforms: Optional[list] = None,
    youtube_extra: Optional[dict] = None,
) -> dict:
    return postiz_service.upload_video(
        video_path, title, platforms, youtube_extra=youtube_extra
    )
