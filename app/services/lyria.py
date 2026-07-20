"""Google Lyria 3 background music generation for ``bgm_type = "lyria"``.

与 Sonilo / ElevenLabs 的 *video-to-music*（把成片画面送去生成配乐）不同，
Lyria 是 *text-to-music*：它只根据文字提示词生成音乐，不分析视频。因此本模块：

  1. 只需要提示词和目标时长，不上传视频代理，省去一次 FFmpeg 转码和上传流量。
  2. Lyria 生成的时长只能通过提示词粗略引导（Pro 版约几分钟、Clip 版固定 30s），
     无法精确等于旁白长度。为遵守任务层“override 文件即已完成时长适配”的约定
     （见 ``video.py`` 混音逻辑：override 文件不再循环铺满），本模块在返回前用
     FFmpeg 把音频循环/裁剪到恰好 ``video_duration`` 秒，交付一条时长对齐的音轨。
  3. 复用 LLM/图片侧已有的 ``gemini_api_key``，用户配置一次即可多处生效。

接口与其它视频配乐供应商保持一致（``is_enabled`` + ``generate_bgm``），从而直接
接入 ``task.py`` 的 ``_VIDEO_MUSIC_PROVIDERS`` 统一编排、0 音量短路与失败降级。
"""

import base64
import binascii
import math
import os
import subprocess
import tempfile
from typing import Any

import requests
from loguru import logger

from app.config import config
from app.services import bgm as bgm_service
from app.utils import utils


DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
INTERACTIONS_PATH = "/v1beta/interactions"
DEFAULT_MODEL_ID = "lyria-3-pro-preview"
SUPPORTED_MODEL_IDS = frozenset({"lyria-3-pro-preview", "lyria-3-clip-preview"})
# 提示词为空时的安全兜底：Lyria 没有视频输入，空提示会得到随机风格，这里给一个
# 通用的无人声背景乐提示，保证纯净可用的配乐。
DEFAULT_PROMPT = "gentle instrumental background music, no vocals"
MAX_PROMPT_LENGTH = 1000
# 循环铺满时长的合理上限，避免异常的超长旁白让 FFmpeg 反复拼接失控。
MAX_VIDEO_DURATION_SECONDS = 3600
MAX_GENERATED_AUDIO_BYTES = 50 * 1024 * 1024


class LyriaError(RuntimeError):
    """表示 Lyria 生成请求、返回音频解码或时长适配失败。"""


class LyriaAuthenticationError(LyriaError):
    """表示 Gemini API Key 缺失或被服务端拒绝。"""


class LyriaContentBlockedError(LyriaError):
    """表示提示词被 Lyria 内容策略拦截（HTTP 400 content_blocked）。

    与网络/服务错误不同，这是提示词内容问题：换一个更中性的提示词即可通过，
    因此调用方可据此回退到通用默认提示词，而不是整体判定配乐失败。
    """


def get_api_key() -> str:
    """读取 Lyria 使用的 API Key。

    与 LLM / AI 图片侧共用 ``gemini_api_key``；环境变量仅作为本机配置未填写时的
    后备来源，避免用户为同一个 Google 账号在 WebUI 维护多份 Key。
    """
    configured_key = str(config.app.get("gemini_api_key", "") or "").strip()
    return configured_key or os.getenv("GEMINI_API_KEY", "").strip()


def is_enabled() -> bool:
    return bool(get_api_key())


def _base_url() -> str:
    return str(
        config.app.get("lyria_base_url", DEFAULT_BASE_URL) or DEFAULT_BASE_URL
    ).rstrip("/")


def _model_id() -> str:
    """只允许 Lyria 3 当前公开的两个模型，错误配置时安全回退到 Pro 版。"""
    model_id = str(
        config.app.get("lyria_model_id", DEFAULT_MODEL_ID) or DEFAULT_MODEL_ID
    ).strip()
    return model_id if model_id in SUPPORTED_MODEL_IDS else DEFAULT_MODEL_ID


def model_tag() -> str:
    """供全局音乐缓存计算稳定缓存键：同一模型的相同提示词/时长可复用结果。"""
    return _model_id()


def _request_timeout() -> tuple[int, int]:
    """限制读取超时，兼顾 Lyria Pro 生成整首歌的耗时与错误配置的可恢复性。"""
    raw_timeout = config.app.get("lyria_timeout", 600)
    try:
        read_timeout = float(raw_timeout)
    except (TypeError, ValueError):
        read_timeout = 600
    if not math.isfinite(read_timeout) or read_timeout <= 0:
        read_timeout = 600
    return 15, max(1, math.ceil(min(read_timeout, 1800)))


def _safe_response_error(response: requests.Response) -> str:
    """截断第三方错误正文，保留定位信息但不让长页面污染任务日志。"""
    body = (response.text or "").strip().replace("\n", " ")[:500]
    return body or response.reason or "request failed"


def _remove_file(file_path: str) -> None:
    """尽力清理中间文件，不覆盖调用方正在处理的原始异常。"""
    if not file_path or not os.path.exists(file_path):
        return
    try:
        os.remove(file_path)
    except OSError as exc:
        logger.warning(
            f"failed to remove Lyria temporary file: path={file_path}, error={exc}"
        )


def _suffix_for_mime(mime_type: str) -> str:
    """按返回的 MIME 猜测临时文件后缀，帮助 FFmpeg 正确识别输入容器。"""
    mime = (mime_type or "").split(";")[0].strip().lower()
    return {
        "audio/mpeg": ".mp3",
        "audio/mp3": ".mp3",
        "audio/wav": ".wav",
        "audio/x-wav": ".wav",
        "audio/wave": ".wav",
    }.get(mime, ".mp3")


def _extract_audio(payload: Any) -> tuple[bytes, str]:
    """从 Interactions 响应里取出第一段音频并解码为 ``(字节, mime)``。

    文档结构为 ``steps[].content[]``，音频块形如
    ``{"type": "audio", "data": "<base64>", "mime_type": "audio/mpeg"}``。
    这里做防御式解析：只要块里带 ``data`` 且 MIME 是音频即可命中，容忍缺省
    ``type`` 字段或字段命名的小差异。
    """
    if not isinstance(payload, dict):
        raise LyriaError("Lyria returned an unexpected response")
    steps = payload.get("steps")
    if not isinstance(steps, list):
        raise LyriaError("Lyria response contained no steps")
    for step in steps:
        content = step.get("content") if isinstance(step, dict) else None
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            data = block.get("data")
            mime = str(block.get("mime_type") or block.get("mimeType") or "")
            block_type = str(block.get("type") or "")
            if not data:
                continue
            if block_type != "audio" and not mime.lower().startswith("audio"):
                continue
            try:
                content_bytes = base64.b64decode(data)
            except (ValueError, binascii.Error) as exc:
                raise LyriaError(
                    f"Lyria returned undecodable audio data: {exc}"
                ) from exc
            if not content_bytes:
                raise LyriaError("Lyria returned empty audio data")
            if len(content_bytes) > MAX_GENERATED_AUDIO_BYTES:
                raise LyriaError("Lyria audio exceeds the 50 MB limit")
            return content_bytes, mime or "audio/mpeg"
    raise LyriaError("Lyria response contained no audio block")


def _request_audio(prompt: str) -> tuple[bytes, str]:
    """向 Interactions API 请求一段音乐，返回 ``(音频字节, mime)``。"""
    model_id = _model_id()
    logger.info(
        f"requesting Lyria background music: model={model_id}, "
        f"prompt_provided={bool(prompt)}"
    )
    request_body = {
        "model": model_id,
        "input": prompt,
        "response_format": {"type": "audio"},
    }
    try:
        response = requests.post(
            f"{_base_url()}{INTERACTIONS_PATH}",
            headers={
                "x-goog-api-key": get_api_key(),
                "Content-Type": "application/json",
            },
            json=request_body,
            proxies=config.proxy,
            timeout=_request_timeout(),
        )
    except requests.RequestException as exc:
        raise LyriaError(f"failed to request Lyria music: {exc}") from exc
    if response.status_code == 401:
        raise LyriaAuthenticationError(
            f"Lyria API key was rejected (401): {_safe_response_error(response)}"
        )
    if not response.ok:
        detail = _safe_response_error(response)
        # 400 content_blocked 属于提示词内容问题，单独抛出以便上层换用中性提示词
        # 重试，而不是把它当成不可恢复的服务错误。
        if response.status_code == 400 and "content_blocked" in detail:
            raise LyriaContentBlockedError(
                f"Lyria blocked the prompt by content policy: {detail}"
            )
        raise LyriaError(
            f"Lyria generation failed ({response.status_code}): {detail}"
        )
    try:
        payload = response.json()
    except ValueError as exc:
        raise LyriaError("Lyria returned an invalid JSON response") from exc
    return _extract_audio(payload)


def _fit_to_duration(src_path: str, output_path: str, duration: float) -> str:
    """用 FFmpeg 把音频循环/裁剪到恰好 ``duration`` 秒并重编码为 MP3。

    Lyria 生成的时长与旁白无关，因此这里统一交付时长对齐的音轨，让任务层可以
    把它当作已适配时长的 override 文件传给混音（不再二次循环），与 Sonilo /
    ElevenLabs 的行为一致。
    """
    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    command = [
        utils.get_ffmpeg_binary(),
        "-nostdin",
        "-v",
        "error",
        "-y",
        # -stream_loop -1 在到达目标时长前无限循环输入；-t 精确裁剪长度。
        # 输入比目标长时同样正确（首遍即被裁断）。
        "-stream_loop",
        "-1",
        "-i",
        src_path,
        "-t",
        f"{duration:.3f}",
        "-vn",
        "-c:a",
        "libmp3lame",
        "-b:a",
        "192k",
        output_path,
    ]
    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        _remove_file(output_path)
        raise LyriaError("Lyria audio duration fitting timed out") from exc
    except OSError as exc:
        _remove_file(output_path)
        raise LyriaError("failed to run FFmpeg for Lyria audio") from exc
    if result.returncode != 0:
        _remove_file(output_path)
        detail = (result.stderr or "").strip().replace("\n", " ")[-500:]
        raise LyriaError(f"failed to fit Lyria audio to duration: {detail}")
    if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 0:
        _remove_file(output_path)
        raise LyriaError("Lyria produced an empty audio file")
    return output_path


def generate_bgm(
    video_path: str,
    output_path: str,
    video_duration: float,
    prompt: str = "",
) -> str:
    """为一条视频生成时长对齐的 Lyria 背景音乐。

    ``video_path`` 仅为与其它视频配乐供应商保持统一签名而保留，Lyria 是纯
    text-to-music，不读取视频内容。
    """
    if not get_api_key():
        raise LyriaAuthenticationError("Lyria API key is required")
    try:
        duration = float(video_duration)
    except (TypeError, ValueError) as exc:
        raise LyriaError("Lyria video duration is invalid") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise LyriaError("Lyria video duration is invalid")
    if duration > MAX_VIDEO_DURATION_SECONDS:
        raise LyriaError(
            f"Lyria supports videos up to {MAX_VIDEO_DURATION_SECONDS} seconds"
        )

    prompt = str(prompt or "").strip() or DEFAULT_PROMPT
    if len(prompt) > MAX_PROMPT_LENGTH:
        raise LyriaError("Lyria music prompt exceeds 1000 characters")

    try:
        audio_bytes, mime = _request_audio(prompt)
    except LyriaContentBlockedError:
        # 提示词被内容策略拦截：用户/自动生成的提示词里可能带了敏感词、真实
        # 艺人或版权引用。回退到中性默认提示词再试一次，让用户至少拿到一段
        # 背景乐，而不是整条视频没有配乐。仍被拦截才向上抛出。
        if prompt == DEFAULT_PROMPT:
            raise
        logger.warning(
            "Lyria blocked the music prompt by content policy; retrying with a "
            f"neutral default prompt. blocked_prompt={prompt[:200]!r}"
        )
        audio_bytes, mime = _request_audio(DEFAULT_PROMPT)

    output_dir = os.path.dirname(os.path.abspath(output_path))
    os.makedirs(output_dir, exist_ok=True)
    descriptor, raw_path = tempfile.mkstemp(
        prefix=".lyria-music-",
        suffix=_suffix_for_mime(mime),
        dir=output_dir,
    )
    try:
        with os.fdopen(descriptor, "wb") as raw_file:
            raw_file.write(audio_bytes)
            raw_file.flush()
            os.fsync(raw_file.fileno())
        try:
            bgm_service.validate_audio_file(raw_path, timeout_seconds=120)
        except (bgm_service.BgmUploadError, bgm_service.BgmServiceError) as exc:
            raise LyriaError(
                "Lyria returned audio that FFmpeg cannot decode"
            ) from exc
        _fit_to_duration(raw_path, output_path, duration)
        logger.info(
            f"Lyria background music generated: output={output_path}, "
            f"target_duration={duration:.3f}s"
        )
        return output_path
    except OSError as exc:
        raise LyriaError(f"Lyria local file operation failed: {exc}") from exc
    finally:
        _remove_file(raw_path)
