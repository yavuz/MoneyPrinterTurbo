"""
逐词高亮（卡拉OK）字幕：把 SRT 转成 ASS，再交给 FFmpeg/libass 一次性烧录。

为什么不复用 MoviePy 的字幕合成：
`video.generate_video` 里的字幕是逐条 `TextClip` 通过 `CompositeVideoClip`
叠加的，合成开销在 Python/NumPy 层，和字幕条数成正比。逐词高亮会把字幕条数
从「每句一条」放大到「每个词一条」，60 秒短视频就是 150~200 条，合成时间会
成倍上涨。改成生成一份 ASS、由 libass 在 FFmpeg 里烧录，渲染成本几乎与词数
无关，只多付一次视频编码。

样式实现说明：
高亮色块用 `BorderStyle=3`（不透明色块）实现，画出来是真正的矩形。它是样式
级属性，没有对应的行内 override 标签，所以这里额外定义一个 `Highlight` 样式，
再用 `{\\rHighlight}词{\\r}` 在一行中间切换样式——`\\r` 后面带样式名时会把后续
文字整体重置到该样式，包括 BorderStyle。

另一种常见做法是给高亮词加超粗描边（`\\bord` + `\\3c`）伪造色块，但那样边缘会
沿着字形起伏，得到的是圆润的色团而不是矩形。
"""

import os
import re
import shutil
import subprocess
from dataclasses import dataclass, field
from functools import lru_cache
from typing import List, Optional, Sequence

from loguru import logger
from PIL import ImageFont

from app.utils import utils

# 中日韩字符没有空格分词，按单字切分才能做出逐字高亮效果。
_CJK_PATTERN = re.compile(
    r"[　-〿぀-ヿㇰ-ㇿ㐀-䶿"
    r"一-鿿豈-﫿가-힯＀-￯]"
)


# 折行字幕的行间距，相对字号的比例。调大调小只需要改这一个数。
_LINE_GAP_RATIO = 0.4


@dataclass
class Word:
    """一个高亮单位：西文按词、中日韩按字。"""

    text: str
    # 该词与前一个词之间的原始分隔符，重新拼行时要原样写回。
    separator: str = ""
    start: float = 0.0
    end: float = 0.0


@dataclass
class Cue:
    start: float
    end: float
    text: str
    words: List[Word] = field(default_factory=list)


def _is_cjk(char: str) -> bool:
    return bool(_CJK_PATTERN.match(char))


def tokenize(text: str) -> List[Word]:
    """
    把一句字幕切成高亮单位，并记录每个单位前面的空白，保证还原时排版不变。
    """
    words: List[Word] = []
    pending_separator = ""
    buffer = ""

    def flush():
        nonlocal buffer, pending_separator
        if buffer:
            words.append(Word(text=buffer, separator=pending_separator))
            buffer = ""
            pending_separator = ""

    for char in text.replace("\n", " "):
        if char.isspace():
            flush()
            pending_separator += char
            continue
        if _is_cjk(char):
            flush()
            words.append(Word(text=char, separator=pending_separator))
            pending_separator = ""
            continue
        buffer += char

    flush()
    return words


def _word_weight(word: Word) -> float:
    # 中日韩单字发音时长明显长于一个拉丁字母，按 2 个字符计权更接近真实语速。
    return float(sum(2 if _is_cjk(char) else 1 for char in word.text)) or 1.0


def distribute_word_times(words: Sequence[Word], start: float, end: float):
    """
    没有真实词级时间戳时，按字符权重把整句时长均分给每个词。

    这是所有 TTS/字幕来源都能用的兜底方案：一条字幕通常只有几个词，
    按权重线性分配的误差在观感上可以接受。
    """
    if not words:
        return
    duration = max(0.0, end - start)
    total_weight = sum(_word_weight(word) for word in words)
    cursor = start
    for word in words:
        span = duration * (_word_weight(word) / total_weight) if total_weight else 0.0
        word.start = cursor
        cursor = min(end, cursor + span)
        word.end = cursor
    # 浮点累积误差会让最后一个词提前几毫秒结束，这里对齐到整句结尾。
    words[-1].end = end


def apply_word_timestamps(cue: Cue, timestamps: Sequence[dict]) -> bool:
    """
    用识别结果里的真实词级时间戳覆盖某条字幕的分配结果。

    按「词中点落在字幕时间窗内」来匹配，而不是按下标匹配：
    `subtitle.correct()` 会合并或改写字幕条目，下标对应关系并不可靠。
    匹配到的词数必须与切分结果一致，否则宁可退回按权重分配，避免错位高亮。
    """
    matched = [
        item
        for item in timestamps
        if cue.start <= (float(item["start"]) + float(item["end"])) / 2 <= cue.end
    ]
    if len(matched) != len(cue.words):
        return False

    for word, item in zip(cue.words, matched):
        word.start = max(cue.start, float(item["start"]))
        word.end = min(cue.end, float(item["end"]))
    # 词与词之间的静音会让高亮块闪断，这里把每个词延长到下一个词的开始。
    for index in range(len(cue.words) - 1):
        cue.words[index].end = cue.words[index + 1].start
    cue.words[-1].end = cue.end
    return True


def parse_srt_time(value: str) -> float:
    hours, minutes, rest = value.strip().split(":")
    seconds, milliseconds = rest.replace(".", ",").split(",")
    return (
        int(hours) * 3600 + int(minutes) * 60 + int(seconds) + int(milliseconds) / 1000
    )


def format_ass_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    whole_seconds = int(seconds % 60)
    centiseconds = int(round((seconds - int(seconds)) * 100))
    if centiseconds == 100:
        centiseconds = 0
        whole_seconds += 1
    return f"{hours:d}:{minutes:02d}:{whole_seconds:02d}.{centiseconds:02d}"


def to_ass_color(color: Optional[str], default: str = "#FFFFFF") -> str:
    """
    把 `#RRGGBB` 转成 ASS 的 `&HAABBGGRR`（BGR 顺序，AA 为透明度且 0 表示不透明）。

    样式行里颜色不带结尾的 `&`，override 标签里需要补上，由调用方处理。
    """
    value = str(color or default).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) != 6 or not re.fullmatch(r"[0-9a-fA-F]{6}", value):
        value = default.lstrip("#")
    red, green, blue = value[0:2], value[2:4], value[4:6]
    return f"&H00{blue}{green}{red}".upper()


def escape_ass_text(text: str) -> str:
    # `{}` 是 ASS 的override块定界符，`\` 是转义符，正文里出现必须替换掉。
    return (
        text.replace("\\", "/")
        .replace("{", "(")
        .replace("}", ")")
    )


def resolve_font_family(font_path: str, fallback: str = "Arial") -> str:
    """
    ASS 的 Style 只认字体族名，不认文件路径，这里从字体文件里读出族名。
    """
    try:
        family, _ = ImageFont.truetype(font_path, 40).getname()
        return family or fallback
    except Exception as exc:
        logger.warning(f"failed to read font family from {font_path}: {str(exc)}")
        return fallback


def wrap_words(
    words: Sequence[Word], font_path: str, font_size: int, max_width: int
) -> List[List[int]]:
    """
    按像素宽度把词切成多行，返回每行包含的词下标。

    这里只做近似测量：PIL 与 libass 的排版细节并不完全一致，但字幕最大宽度
    取的是视频宽度的 90%，留出的余量足以吸收这点差异。
    """
    try:
        font = ImageFont.truetype(font_path, font_size)
    except Exception as exc:
        logger.warning(f"failed to load font for wrapping: {str(exc)}")
        return [list(range(len(words)))]

    def measure(text: str) -> int:
        if not text:
            return 0
        box = font.getbbox(text)
        return int(box[2] - box[0])

    lines: List[List[int]] = []
    current: List[int] = []
    current_text = ""
    for index, word in enumerate(words):
        candidate = current_text + (word.separator if current else "") + word.text
        if current and measure(candidate) > max_width:
            lines.append(current)
            current = [index]
            current_text = word.text
            continue
        current.append(index)
        current_text = candidate
    if current:
        lines.append(current)
    return lines


def _alignment_and_margin(
    subtitle_position: str, custom_position: float, video_height: int
) -> tuple[int, int]:
    """
    把项目里的字幕位置参数翻译成 ASS 的 Alignment + MarginV。

    数值刻意与 `video.create_text_clip` 的定位保持一致，切换渲染方式时
    字幕不会跳位。
    """
    if subtitle_position == "top":
        return 8, int(video_height * 0.05)
    if subtitle_position == "center":
        return 5, 0
    if subtitle_position == "custom":
        ratio = max(0.0, min(100.0, float(custom_position))) / 100
        return 8, int(video_height * ratio)
    return 2, int(video_height * 0.05)


def build_cues(
    subtitle_items: Sequence[tuple], word_timestamps: Optional[Sequence[dict]] = None
) -> List[Cue]:
    """
    把 `subtitle.file_to_subtitles()` 的结果转成带词级时间的字幕条目。
    """
    cues: List[Cue] = []
    for item in subtitle_items:
        time_range = item[1]
        text = str(item[2] or "").strip()
        if not text or "-->" not in time_range:
            continue
        start_text, end_text = time_range.split("-->")
        cue = Cue(
            start=parse_srt_time(start_text),
            end=parse_srt_time(end_text),
            text=text,
        )
        if cue.end <= cue.start:
            continue
        cue.words = tokenize(text)
        if not cue.words:
            continue
        if not (word_timestamps and apply_word_timestamps(cue, word_timestamps)):
            distribute_word_times(cue.words, cue.start, cue.end)
        cues.append(cue)
    return cues


def render_ass(
    cues: Sequence[Cue],
    font_path: str,
    font_size: int,
    video_width: int,
    video_height: int,
    text_color: str,
    stroke_color: str,
    stroke_width: float,
    highlight_color: str,
    highlight_text_color: Optional[str],
    subtitle_position: str,
    custom_position: float,
) -> str:
    font_family = resolve_font_family(font_path)
    alignment, margin_v = _alignment_and_margin(
        subtitle_position, custom_position, video_height
    )
    side_margin = int(video_width * 0.05)
    max_text_width = int(video_width * 0.9)
    # ASS 没有行间距字段，libass 的行高完全由该行的字号决定。这里在两行之间
    # 插入一行「只有一个硬空格、字号很小」的内容，用它的行高充当行间距。
    # `\h` 是不会被折叠的硬空格，所以这一行不会被当成空行丢掉。
    line_gap = max(1, int(font_size * _LINE_GAP_RATIO))
    line_separator = f"\\N{{\\fs{line_gap}}}\\h\\N{{\\r}}"
    # BorderStyle=3 下 Outline 字段表示色块相对文字的内边距，而不是描边宽度。
    highlight_padding = max(3, int(font_size * 0.14))

    primary = to_ass_color(text_color, "#FFFFFF")
    outline = to_ass_color(stroke_color, "#000000")
    highlight = to_ass_color(highlight_color, "#E11D2E")
    highlight_primary = to_ass_color(highlight_text_color or text_color, "#FFFFFF")

    header = "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            f"PlayResX: {video_width}",
            f"PlayResY: {video_height}",
            # WrapStyle 2 = 完全不自动换行，换行位置由本模块自己算好。
            "WrapStyle: 2",
            "ScaledBorderAndShadow: yes",
            "",
            "[V4+ Styles]",
            "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
            "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, "
            "ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, "
            "Alignment, MarginL, MarginR, MarginV, Encoding",
            f"Style: Default,{font_family},{font_size},{primary},{primary},"
            f"{outline},&H00000000,0,0,0,0,100,100,0,0,1,{stroke_width:g},0,"
            f"{alignment},{side_margin},{side_margin},{margin_v},1",
            # 高亮词单独一个样式：BorderStyle=3 是 ASS 原生的“不透明色块”，
            # 画出来的是真正的矩形，比用超粗描边伪造色块干净得多。此模式下
            # OutlineColour 表示色块颜色，Outline 表示色块的内边距。
            f"Style: Highlight,{font_family},{font_size},{highlight_primary},"
            f"{highlight_primary},{highlight},&H00000000,0,0,0,0,100,100,0,0,3,"
            f"{highlight_padding},0,{alignment},{side_margin},{side_margin},"
            f"{margin_v},1",
            "",
            "[Events]",
            "Format: Layer, Start, End, Style, Name, MarginL, MarginR, "
            "MarginV, Effect, Text",
        ]
    )

    events: List[str] = []
    for cue in cues:
        lines = wrap_words(cue.words, font_path, font_size, max_text_width)
        for active_index, active_word in enumerate(cue.words):
            if active_word.end <= active_word.start:
                continue
            rendered_lines = []
            for line in lines:
                parts = []
                for position, word_index in enumerate(line):
                    word = cue.words[word_index]
                    separator = word.separator if position else ""
                    body = escape_ass_text(word.text)
                    if word_index == active_index:
                        # `\r<样式名>` 会把后续文字整体重置到该样式，包括
                        # BorderStyle；结尾的 `\r` 再还原成事件自身的样式。
                        body = f"{{\\rHighlight}}{body}{{\\r}}"
                    parts.append(escape_ass_text(separator) + body)
                rendered_lines.append("".join(parts))
            text = line_separator.join(rendered_lines)
            events.append(
                f"Dialogue: 0,{format_ass_time(active_word.start)},"
                f"{format_ass_time(active_word.end)},Default,,0,0,0,,{text}"
            )

    return header + "\n" + "\n".join(events) + "\n"


def build_karaoke_ass(
    subtitle_items: Sequence[tuple],
    output_path: str,
    font_path: str,
    font_size: int,
    video_width: int,
    video_height: int,
    text_color: str = "#FFFFFF",
    stroke_color: str = "#000000",
    stroke_width: float = 1.5,
    highlight_color: str = "#E11D2E",
    highlight_text_color: Optional[str] = None,
    subtitle_position: str = "bottom",
    custom_position: float = 70.0,
    word_timestamps: Optional[Sequence[dict]] = None,
) -> str:
    """
    生成逐词高亮的 ASS 文件，返回文件路径；没有可用字幕时返回空字符串。
    """
    cues = build_cues(subtitle_items, word_timestamps)
    if not cues:
        logger.warning("no usable subtitle cues for karaoke subtitle")
        return ""

    content = render_ass(
        cues=cues,
        font_path=font_path,
        font_size=int(font_size),
        video_width=video_width,
        video_height=video_height,
        text_color=text_color,
        stroke_color=stroke_color,
        stroke_width=stroke_width,
        highlight_color=highlight_color,
        highlight_text_color=highlight_text_color,
        subtitle_position=subtitle_position,
        custom_position=custom_position,
    )
    with open(output_path, "w", encoding="utf-8") as fp:
        fp.write(content)
    logger.info(f"karaoke subtitle created: {output_path}")
    return output_path


def burn_subtitles(
    video_path: str,
    ass_path: str,
    output_file: str,
    font_path: str = "",
    codec: str = "libx264",
    threads: int = 2,
) -> None:
    """
    用 libass 把 ASS 烧录进视频，只重编码视频轨，音轨直接复制。

    FFmpeg 的滤镜参数里 `:`、`\\`、`'` 都需要多层转义，Windows 盘符路径尤其
    容易踩坑。这里改用「切到 ASS 所在目录 + 只传文件名」的方式，并把字体也
    复制过去，从根本上避开转义问题。
    """
    work_dir = os.path.dirname(os.path.abspath(ass_path)) or "."
    ass_name = os.path.basename(ass_path)

    if font_path and os.path.isfile(font_path):
        target_font = os.path.join(work_dir, os.path.basename(font_path))
        if os.path.abspath(target_font) != os.path.abspath(font_path):
            try:
                shutil.copyfile(font_path, target_font)
            except OSError as exc:
                # 复制失败不致命：libass 仍会尝试用系统已安装的同名字体。
                logger.warning(f"failed to stage subtitle font: {str(exc)}")

    ffmpeg_binary = resolve_ffmpeg_binary()
    if not ffmpeg_binary:
        raise RuntimeError("no ffmpeg build with the libass 'ass' filter is available")

    command = [
        ffmpeg_binary,
        "-y",
        "-i",
        os.path.abspath(video_path),
        "-vf",
        f"ass=filename={ass_name}:fontsdir=.",
        "-c:v",
        codec,
        "-c:a",
        "copy",
        "-pix_fmt",
        "yuv420p",
        "-threads",
        str(threads or 2),
        os.path.abspath(output_file),
    ]
    result = subprocess.run(
        command,
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_message = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(error_message or "ffmpeg failed to burn ass subtitles")


@lru_cache(maxsize=8)
def _ffmpeg_has_ass_filter(ffmpeg_binary: str) -> bool:
    try:
        result = subprocess.run(
            [ffmpeg_binary, "-hide_banner", "-filters"],
            capture_output=True,
            text=True,
            check=False,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning(f"failed to inspect ffmpeg filters: {str(exc)}")
        return False
    if result.returncode != 0:
        return False
    return any(
        line.split()[1:2] == ["ass"] for line in result.stdout.splitlines() if line.strip()
    )


@lru_cache(maxsize=1)
def resolve_ffmpeg_binary() -> str:
    """
    找一个带 libass 的 FFmpeg，找不到返回空字符串。

    项目默认用 `utils.get_ffmpeg_binary()`，但发行版编译选项差异很大：
    Homebrew 等常见构建可能没有 `--enable-libass`，此时 `ass` 滤镜根本不存在。
    imageio-ffmpeg 附带的二进制一定包含 libass，可以作为备选，避免仅仅因为
    系统 FFmpeg 缺一个滤镜就让逐词字幕整体不可用。
    """
    candidates = [utils.get_ffmpeg_binary()]
    try:
        import imageio_ffmpeg

        candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception as exc:
        logger.warning(f"failed to resolve bundled ffmpeg binary: {str(exc)}")

    for candidate in candidates:
        if candidate and _ffmpeg_has_ass_filter(candidate):
            return candidate
    return ""


def is_supported() -> bool:
    """当前环境是否具备烧录 ASS 字幕的能力。"""
    if resolve_ffmpeg_binary():
        return True
    logger.warning(
        "no ffmpeg build with the libass 'ass' filter was found, "
        "word-level subtitles are unavailable"
    )
    return False


def ass_file_path(output_file: str) -> str:
    """
    按成片文件名派生 ASS 路径。

    同一个任务目录里会并行产出 video-1.mp4 / video-2.mp4，用固定文件名会让
    多个视频互相覆盖字幕。
    """
    root, _ = os.path.splitext(os.path.abspath(output_file))
    return f"{root}.karaoke.ass"
