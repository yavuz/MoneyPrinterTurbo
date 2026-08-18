import json
import random
import re
import time
from typing import List
from loguru import logger

from app.services import llm
from app.services.channel_manager import ChannelProfile

ANGLES_EN = [
    "weirdest behaviors and mysteries",
    "mind-blowing scientific discoveries",
    "unbelievable survival instincts",
    "dark secrets and terrifying facts",
    "hidden superpowers and skills",
    "bizarre myths debunked",
    "ancient evolutionary secrets",
    "smartest tactics and habits",
    "deadly strategies and weapons",
    "unusual emotional intelligence",
]

ANGLES_TR = [
    "en tuhaf davranışlar ve gizemler",
    "zihin dudak uçuklatan bilimsel gerçekler",
    "inanılmaz hayatta kalma içgüdüleri",
    "korkutucu ve karanlık gerçekler",
    "gizli süper güçler ve yetenekler",
    "yanlış bilinen en büyük efsaneler",
    "en zeki taktikler ve alışkanlıklar",
    "en tehlikeli savunma mekanizmaları",
]


def fetch_niche_trending_topics(
    niche: str,
    language: str = "en",
    count: int = 5,
) -> List[str]:
    """
    Generate viral, high-engagement short video topics for a specific niche and language.
    """
    is_tr = language.lower() in ("tr", "turkish")
    default_niche = "İlginç Bilgiler" if is_tr else "Animals & Wildlife Mysteries"
    niche_clean = niche.strip() if niche else default_niche

    angle = random.choice(ANGLES_TR if is_tr else ANGLES_EN)
    random_seed = int(time.time() * 1000) % 10000

    if is_tr:
        prompt = f"""
Sen sosyal medya için yüksek izlenme alan viral içerik fikirleri üreten uzman bir kreatif direktörsün.
Özellikle şu açıya odaklan: "{angle}". (Rastgele İstem Kodu: {random_seed})

Aşağıda verilen niş/kategori için TikTok, YouTube Shorts ve Instagram Reels ortamında yüksek merak uyandıracak {count} adet YENİ ve ÖZGÜN video konusu/başlığı öner.

Niş / Kategori: {niche_clean}
Hedef Dil: Türkçe

Kurallar:
1. Sadece her satırda 1 adet konu başlığı olacak şekilde {count} satır yaz.
2. Başında numara, tire veya madde imleri olmasın.
3. Başlıklar Türkçe olsun, son derece sürükleyici ve merak uyandırıcı olsun.
4. Başka hiçbir açıklama yapma.
""".strip()
    else:
        prompt = f"""
You are an expert creative director generating viral, high-CTR short video topics for TikTok, YouTube Shorts, and Instagram Reels.
Focus specifically on the angle: "{angle}". (Random Seed: {random_seed})

Generate {count} FRESH, UNIQUE, and intriguing video topics/titles for the following niche strictly in ENGLISH.

Niche / Category: {niche_clean}
Target Language: English

Rules:
1. Output exactly {count} lines, with 1 topic title per line.
2. DO NOT include line numbers, bullet points, or dashes.
3. The titles MUST be strictly in ENGLISH.
4. Make them unique, catchy, mysterious, and high-CTR.
5. Do not include any extra text or conversational chatter.
""".strip()

    try:
        response_text = llm._call_llm(
            prompt=prompt,
            system_prompt="You are a viral social media video topic generator. Output strictly in the requested target language without repeating previous ideas.",
        )
        topics = []
        lines = response_text.splitlines()
        for line in lines:
            cleaned = re.sub(r"^\d+[\.\)]\s*", "", line.strip()).strip()
            cleaned = cleaned.lstrip("-*• ").strip()
            if cleaned and len(cleaned) > 3:
                topics.append(cleaned)

        if topics:
            return topics[:count]
    except Exception as e:
        logger.error(f"Error fetching trending topics for niche '{niche}': {e}")

    # Dynamic randomized fallback pool if LLM is offline or unconfigured
    if is_tr:
        fallback_templates = [
            f"Neden {niche_clean} Hakkında Bildiğiniz Her Şey Yanlış?",
            f"{niche_clean}: Asla Duymadığınız 5 Şok Edici Gerçek",
            f"{niche_clean} Arasındaki En İlginç Gizli İletişim Yolları",
            f"Vahşi Doğada {niche_clean} Nasıl Hayatta Kalıyor?",
            f"{niche_clean} Hakkındaki En Büyük Bilimsel Gizem",
            f"Korkunç Ama Gerçek: {niche_clean} Savunma Taktikleri",
            f"İnsanları Şaşırtan {niche_clean} Zekası",
            f"Tarihteki En Sıradışı {niche_clean} Olayı",
        ]
    else:
        fallback_templates = [
            f"Why Everything You Knew About {niche_clean} Is Wrong",
            f"5 Shocking Facts About {niche_clean} You Never Heard Before",
            f"The Dark Secret Behind {niche_clean} Revealed",
            f"How {niche_clean} Outsmart Their Enemies in the Wild",
            f"The Biggest Scientific Mystery Involving {niche_clean}",
            f"Terrifying Truths About {niche_clean} Survival Instincts",
            f"The Unbelievable Intelligence of {niche_clean}",
            f"10 Seconds That Will Change How You View {niche_clean}",
        ]
    
    random.shuffle(fallback_templates)
    return fallback_templates[:count]


def generate_script_for_channel(
    channel: ChannelProfile,
    topic: str,
    paragraph_number: int = 2,
) -> str:
    """
    Generate a short video script tailored to a channel's system prompt and language.
    """
    system_prompt = channel.system_prompt.strip()
    if not system_prompt:
        system_prompt = (
            f"You are a short video content creator for the channel '{channel.name}' in the niche '{channel.niche}'. "
            "Write engaging, punchy, narrative-driven short video scripts."
        )
        
    script = llm.generate_script(
        video_subject=topic,
        language=channel.video_language,
        paragraph_number=paragraph_number,
        custom_system_prompt=system_prompt,
    )
    return script
