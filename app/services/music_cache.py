"""Global content-addressed cache for generated background music and the
auto-generated prompts that key it.

与 AI 图片缓存（``storage/cache_images``）同思路：把昂贵的第三方配乐结果按内容
寻址缓存在 ``storage/cache_music`` 下，缓存键由 ``provider + model + duration +
prompt`` 计算。这样即使每次生成都是新的 ``task_id``，只要输入相同就能命中已生成
的音乐，避免失败重试时重复付费生成。

适用范围：*text-to-music*（如 Lyria）的输出只取决于提示词与时长，因此这种缓存
既安全又有效；*video-to-music*（Sonilo / ElevenLabs）的输出依赖具体画面，不走
本缓存（由任务层的 provider 标记控制）。

自动生成的配乐提示词同样按 ``provider + subject + script`` 内容寻址缓存：否则每次
重试都会让 LLM 生成不同的提示词，进而让音乐缓存键漂移、永远无法命中。
"""

import os

from loguru import logger

from app.utils import utils


def _cache_dir() -> str:
    return utils.storage_dir("cache_music", create=True)


def _music_key(provider: str, model_tag: str, prompt: str, duration) -> str:
    try:
        seconds = int(round(float(duration)))
    except (TypeError, ValueError):
        seconds = 0
    return f"{provider}|{model_tag}|{seconds}|{prompt}"


def music_cache_path(provider: str, model_tag: str, prompt: str, duration) -> str:
    key = _music_key(provider, model_tag, prompt, duration)
    return os.path.join(_cache_dir(), f"music-{utils.md5(key)}.mp3")


def get_cached_music(
    provider: str, model_tag: str, prompt: str, duration
) -> str | None:
    """返回命中的缓存音乐路径；未命中或文件为空时返回 ``None``。"""
    path = music_cache_path(provider, model_tag, prompt, duration)
    if os.path.exists(path) and os.path.getsize(path) > 0:
        return path
    return None


def store_music(
    src_path: str, provider: str, model_tag: str, prompt: str, duration
) -> str:
    """把生成好的音乐复制进全局缓存并返回缓存路径。

    先写临时文件再原子替换，避免并发/中断留下半个文件被后续运行误判为命中。
    缓存写入失败不影响主流程：记录警告并回退到原始文件路径。
    """
    dest = music_cache_path(provider, model_tag, prompt, duration)
    if os.path.abspath(src_path) == os.path.abspath(dest):
        return dest
    tmp = f"{dest}.{os.getpid()}.tmp"
    try:
        with open(src_path, "rb") as src, open(tmp, "wb") as out:
            out.write(src.read())
        os.replace(tmp, dest)
    except OSError as exc:
        logger.warning(f"failed to cache generated music: {exc}")
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except OSError:
            pass
        return src_path
    return dest


def _prompt_cache_path(provider: str, subject: str, script: str) -> str:
    key = f"{provider}|{subject}|{script}"
    return os.path.join(_cache_dir(), f"prompt-{utils.md5(key)}.txt")


def resolve_auto_prompt(provider: str, subject: str, script: str, generator) -> str:
    """返回稳定的自动配乐提示词：命中缓存则复用，否则调用 ``generator`` 生成并缓存。

    ``generator`` 失败或返回空时返回空字符串，且不写入缓存（留待下次重试再生成），
    保证配乐提示词生成失败绝不中断整条视频任务。
    """
    subject = str(subject or "").strip()
    script = str(script or "").strip()
    if not subject and not script:
        return ""

    path = _prompt_cache_path(provider, subject, script)
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                cached = f.read().strip()
            if cached:
                logger.info("reusing cached auto-generated music prompt")
                return cached
    except OSError as exc:
        logger.warning(f"failed to read cached music prompt: {exc}")

    try:
        prompt = (generator() or "").strip()
    except Exception as exc:  # noqa: BLE001 - 配乐提示词是可选增强，失败即回退
        logger.warning(f"failed to generate music prompt: {exc}")
        return ""

    if prompt:
        try:
            tmp = f"{path}.{os.getpid()}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                f.write(prompt)
            os.replace(tmp, path)
        except OSError as exc:
            logger.warning(f"failed to cache music prompt: {exc}")
    return prompt
