"""AI image generation for the ``video_source = "ai"`` material path.

不同于 Pexels/Pixabay 等库存视频源，这个模块把 LLM 生成的分镜提示词发送给
文生图服务（fal.ai / Replicate），生成静态图片，随后由 ``video.py`` 转成带
Ken Burns 效果的短视频片段进入合成流水线。

设计要点：
  1. 生成任务基于队列且耗时较长，因此每次外部调用都必须带连接/读取超时和
     轮询上限。没有超时的请求会让任务线程无限挂起，长期占用并发名额。
  2. 图片按内容寻址缓存在 ``storage/cache_images/`` 下，缓存键由
     ``provider + model + 分辨率 + prompt`` 计算得到。已生成的图片不会重复
     付费生成，这也让“中途失败后再次运行从断点继续”天然成立。
  3. 部分提示词失败时，成功的图片已经落盘缓存；本模块抛出
     ``ImageGenError`` 让 materials 阶段失败，但绝不删除已完成的缓存，
     这样重新运行时只补齐缺失的图片。
"""

import base64
import binascii
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from loguru import logger

from app.config import config
from app.models.schema import VideoAspect
from app.utils import utils

_DEFAULT_FAL_MODEL = "fal-ai/flux/schnell"
_DEFAULT_REPLICATE_MODEL = "black-forest-labs/flux-schnell"
# Google Gemini 文生图，默认使用 Nano Banana Pro（Gemini 3 Pro Image）。
_DEFAULT_GEMINI_MODEL = "gemini-3-pro-image"
_DEFAULT_GEMINI_BASE_URL = "https://generativelanguage.googleapis.com"

_SUPPORTED_PROVIDERS = ("fal", "replicate", "gemini")

# 单张图片的默认生成预算（秒）。文生图队列冷启动时可能偏慢，但超过这个上限
# 就应视为失败并重试，避免任务线程被单张图片无限拖住。
_DEFAULT_TIMEOUT = 120
_DEFAULT_POLL_INTERVAL = 3.0
_DEFAULT_MAX_RETRIES = 2
_DEFAULT_MAX_CONCURRENCY = 4

# HTTP 层的连接/读取超时。轮询和提交都很快返回，真正的等待通过上面的总预算
# ``deadline`` 控制，而不是靠单次请求的读取超时。
_SUBMIT_TIMEOUT = (10, 60)
_POLL_TIMEOUT = (10, 30)
_DOWNLOAD_TIMEOUT = (10, 120)


class ImageGenError(Exception):
    """图片无法在重试次数内生成时抛出。"""


class ImageGenConfigError(ImageGenError):
    """所选 provider 缺少必要配置（如 API Key）时抛出，属于快速失败。"""


def _tls_verify() -> bool:
    tls_verify = config.app.get("tls_verify", True)
    if isinstance(tls_verify, str):
        tls_verify = tls_verify.strip().lower() not in ("0", "false", "no", "off")
    return bool(tls_verify)


def _cfg_int(key: str, default: int) -> int:
    try:
        return int(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


def _cfg_float(key: str, default: float) -> float:
    try:
        return float(config.app.get(key, default))
    except (TypeError, ValueError):
        return default


def get_provider() -> str:
    return (config.app.get("image_provider") or "fal").strip().lower()


def _model_tag(provider: str) -> str:
    if provider == "replicate":
        return (
            config.app.get("replicate_model_name") or _DEFAULT_REPLICATE_MODEL
        ).strip()
    if provider == "gemini":
        return (
            config.app.get("gemini_image_model_name") or _DEFAULT_GEMINI_MODEL
        ).strip()
    return (config.app.get("fal_model_name") or _DEFAULT_FAL_MODEL).strip()


def _download_image(url: str, deadline: float, verify: bool) -> bytes:
    remaining = max(1.0, deadline - time.monotonic())
    r = requests.get(
        url,
        proxies=config.proxy,
        verify=verify,
        timeout=(_DOWNLOAD_TIMEOUT[0], min(_DOWNLOAD_TIMEOUT[1], remaining)),
    )
    r.raise_for_status()
    content = r.content
    if not content:
        raise ImageGenError("downloaded image is empty")
    return content


def _extract_fal_images(payload: dict) -> list:
    images = payload.get("images") if isinstance(payload, dict) else None
    if not isinstance(images, list):
        return []
    urls = []
    for item in images:
        if isinstance(item, dict) and item.get("url"):
            urls.append(item["url"])
        elif isinstance(item, str):
            urls.append(item)
    return urls


def _fal_generate(
    prompt: str,
    width: int,
    height: int,
    deadline: float,
    poll_interval: float,
    verify: bool,
) -> bytes:
    api_key = (config.app.get("fal_api_key") or "").strip()
    if not api_key:
        raise ImageGenConfigError(
            "fal_api_key is not set; configure it in config.toml under [app]"
        )
    model = _model_tag("fal")
    headers = {
        "Authorization": f"Key {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "prompt": prompt,
        "image_size": {"width": width, "height": height},
        "num_images": 1,
    }
    submit_url = f"https://queue.fal.run/{model}"
    r = requests.post(
        submit_url,
        json=payload,
        headers=headers,
        proxies=config.proxy,
        verify=verify,
        timeout=_SUBMIT_TIMEOUT,
    )
    r.raise_for_status()
    data = r.json()

    # 有些同步模型会直接在提交响应里返回图片，无需轮询。
    direct = _extract_fal_images(data)
    if direct:
        return _download_image(direct[0], deadline, verify)

    status_url = data.get("status_url")
    response_url = data.get("response_url")
    if not status_url or not response_url:
        raise ImageGenError("fal submit response missing status/response url")

    while True:
        if time.monotonic() > deadline:
            raise ImageGenError("fal image generation timed out")
        sr = requests.get(
            status_url,
            headers=headers,
            proxies=config.proxy,
            verify=verify,
            timeout=_POLL_TIMEOUT,
        )
        sr.raise_for_status()
        status = (sr.json() or {}).get("status")
        if status == "COMPLETED":
            break
        if status in ("FAILED", "ERROR"):
            raise ImageGenError("fal reported a failed generation")
        time.sleep(poll_interval)

    rr = requests.get(
        response_url,
        headers=headers,
        proxies=config.proxy,
        verify=verify,
        timeout=_POLL_TIMEOUT,
    )
    rr.raise_for_status()
    images = _extract_fal_images(rr.json())
    if not images:
        raise ImageGenError("fal returned no images")
    return _download_image(images[0], deadline, verify)


def _replicate_generate(
    prompt: str,
    aspect_ratio: str,
    deadline: float,
    poll_interval: float,
    verify: bool,
) -> bytes:
    token = (config.app.get("replicate_api_token") or "").strip()
    if not token:
        raise ImageGenConfigError(
            "replicate_api_token is not set; configure it in config.toml under [app]"
        )
    model = _model_tag("replicate")
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
        # Prefer: wait 让 Replicate 官方模型在一次请求内最多阻塞 60s 直接返回
        # 结果，命中时可跳过轮询；未命中再走下面的 GET 轮询。
        "Prefer": "wait",
    }
    url = f"https://api.replicate.com/v1/models/{model}/predictions"
    payload = {
        "input": {
            "prompt": prompt,
            "aspect_ratio": aspect_ratio,
            "num_outputs": 1,
            "output_format": "png",
        }
    }
    r = requests.post(
        url,
        json=payload,
        headers=headers,
        proxies=config.proxy,
        verify=verify,
        timeout=(_SUBMIT_TIMEOUT[0], 70),
    )
    r.raise_for_status()
    pred = r.json()

    terminal = {"succeeded", "failed", "canceled"}
    while (pred or {}).get("status") not in terminal:
        if time.monotonic() > deadline:
            raise ImageGenError("replicate image generation timed out")
        get_url = (pred.get("urls") or {}).get("get")
        if not get_url:
            raise ImageGenError("replicate prediction missing poll url")
        time.sleep(poll_interval)
        gr = requests.get(
            get_url,
            headers=headers,
            proxies=config.proxy,
            verify=verify,
            timeout=_POLL_TIMEOUT,
        )
        gr.raise_for_status()
        pred = gr.json()

    if pred.get("status") != "succeeded":
        raise ImageGenError("replicate reported a failed generation")

    output = pred.get("output")
    image_url = output[0] if isinstance(output, list) and output else output
    if not isinstance(image_url, str) or not image_url:
        raise ImageGenError("replicate returned no image output")
    return _download_image(image_url, deadline, verify)


# Gemini 的 generateContent 主要由文字提示词驱动构图，光靠 imageConfig 有时不足以
# 保证竖屏/横屏取景。这里把画幅方向也写进提示词，让模型按目标画幅安排构图。
_ASPECT_ORIENTATION = {
    "9:16": "vertical portrait",
    "16:9": "horizontal landscape",
    "1:1": "square",
}


def _aspect_prompt_hint(aspect_ratio: str) -> str:
    orientation = _ASPECT_ORIENTATION.get(aspect_ratio)
    if orientation:
        return (
            f"Aspect ratio {aspect_ratio}: frame the full scene for a "
            f"{orientation} composition."
        )
    return f"Aspect ratio {aspect_ratio}."


def _extract_gemini_image(payload: dict) -> bytes:
    """从 Gemini generateContent 响应里取出第一张内联图片并解码为字节。"""
    candidates = payload.get("candidates") if isinstance(payload, dict) else None
    if not isinstance(candidates, list) or not candidates:
        raise ImageGenError("gemini returned no candidates")
    parts = ((candidates[0] or {}).get("content") or {}).get("parts") or []
    for part in parts:
        if not isinstance(part, dict):
            continue
        # REST 用 camelCase(inlineData)，SDK 风格用 snake_case(inline_data)，都兼容。
        inline = part.get("inlineData") or part.get("inline_data")
        data = inline.get("data") if isinstance(inline, dict) else None
        if not data:
            continue
        try:
            content = base64.b64decode(data)
        except (ValueError, binascii.Error) as e:
            raise ImageGenError(f"gemini returned undecodable image data: {e}")
        if not content:
            raise ImageGenError("gemini returned empty image data")
        return content
    raise ImageGenError("gemini response contained no inline image")


def _gemini_generate(
    prompt: str,
    aspect_ratio: str,
    deadline: float,
    verify: bool,
) -> bytes:
    api_key = (config.app.get("gemini_api_key") or "").strip()
    if not api_key:
        raise ImageGenConfigError(
            "gemini_api_key is not set; configure it in config.toml under [app]"
        )
    model = _model_tag("gemini")
    base_url = (
        config.app.get("gemini_base_url") or _DEFAULT_GEMINI_BASE_URL
    ).strip().rstrip("/")
    url = f"{base_url}/v1beta/models/{model}:generateContent"
    # 明确记录真正请求的模型，方便用户核对是否在用 Nano Banana Pro
    # （gemini-3-pro-image）还是更轻量的 Flash / Flash Lite。
    logger.info(f"requesting Gemini image: model={model}, aspect={aspect_ratio}")
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    # Nano Banana Pro 通过 generateContent 同步返回内联图片，无需队列轮询。
    # imageConfig.aspectRatio 直接接受 "9:16" 这类比例，与 VideoAspect 对齐；
    # 同时把方向写进文字提示词，双保险地引导竖屏/横屏取景。
    full_prompt = f"{prompt}\n\n{_aspect_prompt_hint(aspect_ratio)}"
    payload = {
        "contents": [{"parts": [{"text": full_prompt}]}],
        "generationConfig": {
            "responseModalities": ["Image"],
            "imageConfig": {"aspectRatio": aspect_ratio},
        },
    }
    # 单次同步请求，读取超时挂靠在总预算 deadline 上，避免慢生成拖死任务线程。
    remaining = max(1.0, deadline - time.monotonic())
    r = requests.post(
        url,
        json=payload,
        headers=headers,
        proxies=config.proxy,
        verify=verify,
        timeout=(_SUBMIT_TIMEOUT[0], min(_DEFAULT_TIMEOUT, remaining)),
    )
    r.raise_for_status()
    return _extract_gemini_image(r.json())


def _generate_one(
    provider: str,
    prompt: str,
    width: int,
    height: int,
    aspect_ratio: str,
    timeout: int,
    poll_interval: float,
    max_retries: int,
    verify: bool,
    dest: str,
) -> str:
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        deadline = time.monotonic() + timeout
        try:
            if provider == "replicate":
                data = _replicate_generate(
                    prompt, aspect_ratio, deadline, poll_interval, verify
                )
            elif provider == "gemini":
                data = _gemini_generate(prompt, aspect_ratio, deadline, verify)
            else:
                data = _fal_generate(
                    prompt, width, height, deadline, poll_interval, verify
                )
            # 先写临时文件再原子替换，避免并发/中断时留下半张损坏图片被后续
            # 运行误判为“已完成缓存”。
            tmp = f"{dest}.{os.getpid()}.tmp"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dest)
            if os.path.getsize(dest) <= 0:
                raise ImageGenError("generated image file is empty")
            logger.success(f"image generated: {dest}")
            return dest
        except ImageGenConfigError:
            # 配置缺失对所有提示词都成立，没必要重试，直接向上冒泡快速失败。
            raise
        except Exception as e:  # noqa: BLE001 - 网络/服务错误统一进入重试
            last_error = e
            logger.warning(
                f"image generation attempt {attempt + 1}/{max_retries + 1} "
                f"failed via {provider}: {type(e).__name__}: {e}"
            )
    raise ImageGenError(
        f"image generation failed after {max_retries + 1} attempts: {last_error}"
    )


def _cache_path(cache_dir: str, provider: str, model_tag: str, resolution: str, prompt: str) -> str:
    key = f"{provider}|{model_tag}|{resolution}|{prompt}"
    return os.path.join(cache_dir, f"img-{utils.md5(key)}.png")


class _Runtime:
    """把一次生成会话依赖的配置解析集中到一处，供批量生成、单张重试和
    路径查询复用，避免三处各自读取配置导致行为漂移。"""

    def __init__(self, video_aspect: VideoAspect):
        provider = get_provider()
        if provider not in _SUPPORTED_PROVIDERS:
            raise ImageGenConfigError(
                f"unsupported image_provider '{provider}', "
                f"expected one of {', '.join(_SUPPORTED_PROVIDERS)}"
            )
        aspect = VideoAspect(video_aspect)
        self.provider = provider
        self.width, self.height = aspect.to_resolution()
        self.aspect_ratio = aspect.value  # 例如 "9:16"，Replicate 直接接受该格式
        self.resolution = f"{self.width}x{self.height}"
        self.timeout = _cfg_int("image_gen_timeout", _DEFAULT_TIMEOUT)
        self.poll_interval = _cfg_float(
            "image_gen_poll_interval", _DEFAULT_POLL_INTERVAL
        )
        self.max_retries = max(0, _cfg_int("image_gen_max_retries", _DEFAULT_MAX_RETRIES))
        self.max_workers = max(
            1, _cfg_int("image_gen_max_concurrency", _DEFAULT_MAX_CONCURRENCY)
        )
        self.verify = _tls_verify()
        self.model_tag = _model_tag(provider)
        self.cache_dir = utils.storage_dir("cache_images", create=True)

    def cache_path(self, prompt: str) -> str:
        return _cache_path(
            self.cache_dir, self.provider, self.model_tag, self.resolution, prompt
        )

    def generate_one(self, prompt: str, dest: str) -> str:
        return _generate_one(
            self.provider,
            prompt,
            self.width,
            self.height,
            self.aspect_ratio,
            self.timeout,
            self.poll_interval,
            self.max_retries,
            self.verify,
            dest,
        )


def image_cache_path(
    prompt: str, video_aspect: VideoAspect = VideoAspect.portrait
) -> str:
    """返回某个提示词在当前 provider/model/分辨率下的缓存图片路径。

    供 WebUI 判断某张图片是否已生成、以及该显示哪个文件，不触发任何生成。
    """
    return _Runtime(video_aspect).cache_path(prompt)


def regenerate_one(
    prompt: str, video_aspect: VideoAspect = VideoAspect.portrait
) -> str:
    """强制重新生成单张图片并覆盖其缓存条目，返回图片路径。

    面向 WebUI 的“重试”按钮：即使该提示词已有缓存，也先删除旧文件再重新
    生成，确保用户拿到一张新图片而不是命中旧缓存。
    """
    runtime = _Runtime(video_aspect)
    dest = runtime.cache_path(prompt)
    try:
        if os.path.exists(dest):
            os.remove(dest)
    except OSError as e:
        logger.warning(f"failed to remove cached image before retry: {e}")
    return runtime.generate_one(prompt, dest)


def _generate_batch(runtime: "_Runtime", image_prompts: list) -> tuple:
    """并行生成一批图片，返回 ``(与提示词等长的结果列表, 失败下标列表)``。

    结果列表中，成功项是图片路径，失败项是 ``None``；命中缓存的项直接复用。
    配置错误（如缺少 API Key）对所有提示词都成立，直接向上冒泡而不计入
    单项失败。
    """
    total = len(image_prompts)
    results: list = [None] * total
    pending = []
    for idx, prompt in enumerate(image_prompts):
        dest = runtime.cache_path(prompt)
        if os.path.exists(dest) and os.path.getsize(dest) > 0:
            logger.info(f"image cache hit [{idx + 1}/{total}]: {dest}")
            results[idx] = dest
        else:
            pending.append((idx, prompt, dest))

    errors: list = []
    if pending:
        logger.info(
            f"generating {len(pending)}/{total} images via {runtime.provider} "
            f"[model={runtime.model_tag}] ({runtime.resolution})"
        )
        with ThreadPoolExecutor(
            max_workers=runtime.max_workers, thread_name_prefix="mpt-image-gen"
        ) as executor:
            future_map = {
                executor.submit(runtime.generate_one, prompt, dest): idx
                for idx, prompt, dest in pending
            }
            for future in as_completed(future_map):
                idx = future_map[future]
                try:
                    results[idx] = future.result()
                except ImageGenConfigError:
                    raise
                except Exception as e:  # noqa: BLE001
                    logger.error(
                        f"failed to generate image [{idx + 1}/{total}]: "
                        f"{type(e).__name__}: {e}"
                    )
                    errors.append(idx)

    return results, errors


def generate_images(
    task_id: str,
    image_prompts: list,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list:
    """按提示词顺序返回本地图片路径列表；失败时抛出 ``ImageGenError``。

    已生成的图片按内容缓存，缺失的图片并行生成。任一提示词最终失败时抛出
    异常，但已成功的图片保留在缓存中，便于重新运行从断点续跑。
    """
    if not image_prompts:
        raise ImageGenError("no image prompts were provided")

    results, errors = _generate_batch(_Runtime(video_aspect), image_prompts)
    if errors:
        missing = sorted(i + 1 for i in errors)
        raise ImageGenError(
            f"failed to generate {len(missing)} image(s) at positions "
            f"{missing}; successful images are cached, re-run to resume"
        )
    return [path for path in results if path]


def prepare_images(
    image_prompts: list,
    video_aspect: VideoAspect = VideoAspect.portrait,
) -> list:
    """面向 WebUI 审核的容错批量生成：返回与提示词等长的列表，成功项为图片
    路径，失败项为 ``None``，单项失败不抛异常（供逐张“重试”）。配置错误仍抛出。
    """
    if not image_prompts:
        return []
    results, _errors = _generate_batch(_Runtime(video_aspect), image_prompts)
    return results
