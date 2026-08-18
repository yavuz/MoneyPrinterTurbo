import os
import uuid
import streamlit as st
from loguru import logger

from app.models.schema import VideoAspect, VideoParams
from app.services import webui_task, voice
from app.services.channel_manager import ChannelProfile, channel_manager
from app.services import trend_service
from app.utils import utils
from app.config import config


def get_available_bgm_songs():
    song_dir = os.path.join(utils.root_dir(), "resource", "songs")
    songs = []
    if os.path.exists(song_dir):
        for root, dirs, files in os.walk(song_dir):
            for file in files:
                if file.endswith((".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg")):
                    songs.append(file)
    return sorted(songs)


def get_available_voices(language="en"):
    try:
        filter_prefix = ["tr-TR"] if language.lower() in ("tr", "turkish") else ["en-US"]
        all_v = voice.get_all_azure_voices(filter_locals=filter_prefix)
        if not all_v:
            all_v = voice.get_all_azure_voices()
        return all_v
    except Exception:
        return [
            "en-US-ChristopherNeural-Male",
            "en-US-GuyNeural-Male",
            "en-US-JennyNeural-Female",
            "en-US-AriaNeural-Female",
            "tr-TR-AhmetNeural-Male",
            "tr-TR-EmelNeural-Female",
        ]


def get_available_fonts():
    font_dir = os.path.join(utils.root_dir(), "resource", "fonts")
    fonts = []
    if os.path.exists(font_dir):
        for file in os.listdir(font_dir):
            if file.endswith((".ttf", ".ttc", ".otf")):
                fonts.append(file)
    return sorted(fonts) if fonts else ["MicrosoftYaHeiBold.ttc", "STHeitiMedium.ttc", "BeVietnamPro-Bold.ttf"]


def render_channel_hub(tr_func=None):
    """
    Renders the Channel Management Hub UI in Streamlit.
    """
    def tr(text):
        if tr_func:
            return tr_func(text)
        return text

    st.title("📺 Kanal Yönetim Merkezi (Channel Management Hub)")
    st.caption("Farklı nişlerde kanallar tanımlayın, canlı trendleri yakalayın ve kanallarınıza özel videolar üretin.")

    # Fetch all channels
    channels = channel_manager.get_all_channels()
    available_songs = get_available_bgm_songs()

    # Layout tabs: 1. Kanal Listesi & Düzenleme, 2. Trend & Senaryo Stüdyosu
    tab_studio, tab_channels = st.tabs([
        "⚡ Trend & Senaryo Stüdyosu",
        "⚙️ Kanal Profilleri Tanımla (" + str(len(channels)) + ")"
    ])

    # ----------------------------------------------------
    # TAB 1: Trend & Senaryo Stüdyosu (Content Generator)
    # ----------------------------------------------------
    with tab_studio:
        if not channels:
            st.info("Henüz tanımlanmış bir kanalınız yok. Lütfen '⚙️ Kanal Profilleri Tanımla' sekmesinden yeni bir kanal ekleyin.")
        else:
            channel_options = {ch.id: f"{ch.name} ({ch.niche or 'Genel'})" for ch in channels}
            selected_ch_id = st.selectbox(
                "Aktif Kanal Seçin:",
                options=list(channel_options.keys()),
                format_func=lambda x: channel_options[x],
                key="active_channel_selector"
            )
            
            active_channel = channel_manager.get_channel(selected_ch_id)
            if active_channel:
                st.divider()
                st.markdown(f"### 🎯 **Kanal:** {active_channel.name}")
                col_info1, col_info2, col_info3 = st.columns(3)
                with col_info1:
                    st.markdown(f"**Niş:** `{active_channel.niche or 'Belirtilmedi'}`")
                with col_info2:
                    st.markdown(f"**Ses:** `{active_channel.voice_name}`")
                with col_info3:
                    st.markdown(f"**Format:** `{active_channel.video_aspect}` | **Dil:** `{active_channel.video_language}`")
                    st.markdown(f"**Müzik:** `{active_channel.bgm_type}` ({active_channel.bgm_name})")

                st.subheader("1. Trend & Konu Radarı")
                col_t1, col_t2 = st.columns([3, 1])
                with col_t2:
                    if st.button("🔥 Trend Başlıkları Getir", use_container_width=True, type="secondary"):
                        with st.spinner("Trend konular taranıyor..."):
                            topics = trend_service.fetch_niche_trending_topics(
                                niche=active_channel.niche or active_channel.name,
                                language=active_channel.video_language,
                                count=5
                            )
                            st.session_state["fetched_trends"] = topics
                            st.success(f"{len(topics)} adet trend konu getirildi.")

                fetched_topics = st.session_state.get("fetched_trends", [])
                if fetched_topics:
                    st.write("**Canlı Trend Fikirleri (Seçmek için üzerine tıklayın):**")

                    def _on_trend_select():
                        st.session_state["channel_hub_subject_input"] = st.session_state.get("selected_trend_radio", "")

                    selected_trend_topic = st.radio(
                        "Trend Seçimi:",
                        options=fetched_topics,
                        label_visibility="collapsed",
                        key="selected_trend_radio",
                        on_change=_on_trend_select,
                    )
                else:
                    selected_trend_topic = ""

                # Subject Input
                video_subject = st.text_input(
                    "Video Konusu / Başlığı:",
                    value=selected_trend_topic if selected_trend_topic else "",
                    placeholder="Örn: Yapay Zekanın Gelecekte Değiştireceği 5 Meslek",
                    key="channel_hub_subject_input"
                )

                # Determine effective subject from text input or radio state
                effective_subject = (
                    video_subject.strip()
                    or st.session_state.get("selected_trend_radio", "").strip()
                )

                st.subheader("2. Kanala Özel Senaryo Üretimi")
                col_s1, col_s2 = st.columns([1, 3])
                with col_s1:
                    para_count = st.slider("Paragraf Sayısı:", min_value=1, max_value=5, value=2, key="ch_para_count")
                    if st.button("✨ Senaryo Oluştur", use_container_width=True, type="primary"):
                        if not effective_subject:
                            st.warning("Lütfen önce bir video konusu girin veya trend başlığı seçin.")
                        else:
                            with st.spinner("Kanala özel senaryo yazılıyor..."):
                                try:
                                    generated_script = trend_service.generate_script_for_channel(
                                        channel=active_channel,
                                        topic=effective_subject,
                                        paragraph_number=para_count
                                    )
                                    if generated_script and generated_script.strip():
                                        st.session_state["channel_script_content"] = generated_script
                                        st.session_state["channel_script_area"] = generated_script
                                        st.success("Senaryo başarıyla oluşturuldu!")
                                    else:
                                        st.error("LLM boş senaryo döndürdü. Lütfen Ayarlar (Settings) menüsünden LLM API Key ve Model ayarlarınızı kontrol edin.")
                                except Exception as e:
                                    st.error(f"Senaryo üretilirken hata oluştu: {e}")

                with col_s2:
                    current_script = st.text_area(
                        "Üretilen Senaryo Metni (Düzenleyebilirsiniz):",
                        value=st.session_state.get("channel_script_content", ""),
                        height=160,
                        key="channel_script_area"
                    )

                st.subheader("3. Otomatik Video Üretim Kuyruğu")
                if st.button("🚀 Videoyu Üret ve Sıraya Al", type="primary", use_container_width=True, key="submit_ch_video"):
                    if not effective_subject:
                        st.error("Lütfen bir video konusu girin.")
                    else:
                        task_id = str(uuid.uuid4())
                        # If video source is AI image, set global config parameters
                        if active_channel.video_source == "ai":
                            config.app["image_provider"] = active_channel.image_provider or "gemini"
                            config.app["gemini_image_model_name"] = active_channel.gemini_image_model_name or "gemini-3-pro-image"
                            if active_channel.image_gen_max_images:
                                config.app["image_gen_max_images"] = active_channel.image_gen_max_images

                        parsed_voice = voice.parse_voice_name(active_channel.voice_name)
                        params = VideoParams(
                            video_subject=effective_subject,
                            video_script=current_script.strip() if current_script else "",
                            video_aspect=VideoAspect.portrait.value if active_channel.video_aspect == "9:16" else VideoAspect.landscape.value,
                            voice_name=parsed_voice,
                            voice_volume=active_channel.voice_volume,
                            video_language=active_channel.video_language,
                            video_source=active_channel.video_source,
                            video_clip_duration=active_channel.video_clip_duration,
                            image_count=active_channel.image_count if active_channel.video_source == "ai" else 0,
                            bgm_type=active_channel.bgm_type,
                            bgm_name=active_channel.bgm_name,
                            bgm_volume=active_channel.bgm_volume,
                            subtitle_enabled=active_channel.subtitle_enabled,
                            subtitle_position=active_channel.subtitle_position,
                            custom_position=active_channel.custom_position,
                            font_name=active_channel.font_name,
                            font_size=active_channel.font_size,
                            text_fore_color=active_channel.text_fore_color,
                            stroke_color=active_channel.stroke_color,
                            stroke_width=active_channel.stroke_width,
                            text_background_color=active_channel.subtitle_background_color if active_channel.subtitle_background_enabled else False,
                            rounded_subtitle_background=active_channel.rounded_subtitle_background if active_channel.subtitle_background_enabled else False,
                        )
                        try:
                            webui_task.submit_generation(
                                task_id=task_id,
                                params=params,
                                capture_logs=True
                            )
                            st.balloons()
                            st.success(f"🎉 '{active_channel.name}' kanalı için video üretimi başlatıldı! Görev ID: `{task_id}`")
                            st.info("Üretim durumunu sağ üstteki 'Görev Yöneticisi' (Task Manager) panelinden takip edebilirsiniz.")
                        except Exception as e:
                            st.error(f"Video üretimi başlatılırken hata oluştu: {e}")

    # ----------------------------------------------------
    # TAB 2: Kanal Profilleri Yönetimi (Management)
    # ----------------------------------------------------
    with tab_channels:
        st.subheader("Kanal Tanımlama & Düzenleme")
        
        with st.expander("➕ Yeni Kanal Ekle", expanded=len(channels) == 0):
            with st.form("new_channel_form", clear_on_submit=True):
                new_name = st.text_input("Kanal Adı *", placeholder="Örn: Animals Explained HQ")
                new_niche = st.text_input("Niş / Kategori *", placeholder="Örn: Animal Wildlife & Nature Mysteries")
                new_desc = st.text_area("Açıklama", placeholder="Kanalın konsepti hakkında kısa bilgi")
                new_system_prompt = st.text_area(
                    "Kanala Özel Senaryo İstemi (System Prompt)",
                    value="You are an expert wildlife documentary scriptwriter...",
                    help="AI senaryo yazarken bu üslup ve kurallara uyacaktır."
                )
                
                col1, col2 = st.columns(2)
                with col1:
                    voice_options = get_available_voices("en")
                    new_voice = st.selectbox(
                        "Ses Modeli (Voice Name)",
                        options=voice_options,
                        index=voice_options.index("en-US-ChristopherNeural-Male") if "en-US-ChristopherNeural-Male" in voice_options else 0,
                        key="new_voice_select"
                    )
                    new_aspect = st.selectbox("Video Formatı", options=["9:16", "16:9", "1:1"])
                    new_lang = st.text_input("Video Dili", value="en")
                with col2:
                    source_options = ["pexels", "pixabay", "coverr", "local", "ai"]
                    source_labels = {
                        "pexels": "Pexels",
                        "pixabay": "Pixabay",
                        "coverr": "Coverr",
                        "local": "Yerel dosya",
                        "ai": "AI Görsel",
                    }
                    new_source = st.selectbox(
                        "Video Kaynağı",
                        options=source_options,
                        format_func=lambda x: source_labels.get(x, x),
                        key="new_video_source_select"
                    )
                    new_clip_dur = st.number_input("Klip Süresi (sn)", min_value=2, max_value=15, value=4)
                    
                st.markdown("#### 🎨 AI Görsel Ayarları")
                ai_col1, ai_col2 = st.columns(2)
                with ai_col1:
                    ai_provider_opts = ["gemini", "fal", "replicate"]
                    ai_provider_labels = {
                        "gemini": "Google (Nano Banana Pro)",
                        "fal": "fal.ai",
                        "replicate": "Replicate",
                    }
                    new_ai_provider = st.selectbox(
                        "AI Görsel Sağlayıcı",
                        options=ai_provider_opts,
                        format_func=lambda x: ai_provider_labels.get(x, x),
                        key="new_ai_provider_select"
                    )
                    
                    gemini_model_opts = ["gemini-3-pro-image", "gemini-3.1-flash-image", "gemini-3.1-flash-lite-image"]
                    gemini_model_labels = {
                        "gemini-3-pro-image": "Nano Banana Pro — Gemini 3 Pro Image",
                        "gemini-3.1-flash-image": "Nano Banana — Gemini 3.1 Flash Image",
                        "gemini-3.1-flash-lite-image": "Nano Banana Lite — Gemini 3.1 Flash Lite Image",
                    }
                    new_ai_model = st.selectbox(
                        "Görsel Modeli",
                        options=gemini_model_opts,
                        format_func=lambda x: gemini_model_labels.get(x, x),
                        key="new_ai_model_select"
                    )
                with ai_col2:
                    new_gemini_key = st.text_input(
                        "Google Gemini API Anahtarı",
                        value="",
                        type="password",
                        key="new_gemini_key_input"
                    )
                    count_mode_opts = ["auto", "fixed"]
                    count_mode_labels = {"auto": "Otomatik", "fixed": "Sabit"}
                    new_ai_count_mode = st.selectbox(
                        "Görsel Sayısı",
                        options=count_mode_opts,
                        format_func=lambda x: count_mode_labels.get(x, x),
                        key="new_ai_count_mode_select"
                    )
                    new_ai_max_count = st.number_input(
                        "Maksimum Görsel",
                        min_value=1,
                        max_value=100,
                        value=14,
                        key="new_ai_max_count_input"
                    )

                st.markdown("#### ✍️ Altyazı Ayarları")
                new_sub_enabled = st.checkbox("Altyazıları Etkinleştir", value=True, key="new_sub_enabled_cb")
                font_options = get_available_fonts()
                new_font_name = st.selectbox("Altyazı Fontu", options=font_options, index=0, key="new_font_name_select")

                pos_opts = ["top", "center", "bottom", "custom"]
                pos_labels = {"top": "Üst", "center": "Orta", "bottom": "Alt", "custom": "Özel"}
                new_sub_pos = st.selectbox("Altyazı Konumu", options=pos_opts, index=3, format_func=lambda x: pos_labels.get(x, x), key="new_sub_pos_select")
                new_custom_pos = st.number_input("Özel Konum (üstten %)", min_value=0.0, max_value=100.0, value=70.0, step=1.0, key="new_custom_pos_input")

                c_f1, c_f2 = st.columns([0.42, 0.58])
                with c_f1:
                    new_text_fore_color = st.color_picker("Metin", value="#FFFFFF", key="new_font_color_picker")
                with c_f2:
                    new_font_size = st.slider("Font Boyutu", min_value=30, max_value=100, value=60, step=1, key="new_font_size_slider")

                c_s1, c_s2 = st.columns([0.42, 0.58])
                with c_s1:
                    new_stroke_color = st.color_picker("Çerçeve", value="#000000", key="new_stroke_color_picker")
                with c_s2:
                    new_stroke_width = st.slider("Çerçeve Kalınlığı", min_value=0.0, max_value=10.0, value=1.50, step=0.1, key="new_stroke_width_slider")

                c_bg1, c_bg2 = st.columns([0.55, 0.45])
                with c_bg1:
                    new_sub_bg_enabled = st.checkbox("Arka Plan", value=False, key="new_sub_bg_cb")
                with c_bg2:
                    new_sub_bg_color = st.color_picker("Renk", value="#FFA500", key="new_sub_bg_color_picker", disabled=not new_sub_bg_enabled)

                new_rounded_bg = st.checkbox("Yuvarlak Arka Plan", value=False, key="new_rounded_bg_cb", disabled=not new_sub_bg_enabled)

                st.markdown("#### 🎵 Arka Plan Müziği Ayarları")
                bgm_col1, bgm_col2 = st.columns(2)
                with bgm_col1:
                    bgm_type_options = ["none", "random", "custom", "sonilo", "elevenlabs", "lyria"]
                    bgm_type_labels = {
                        "none": "Arka Plan Müziği Yok",
                        "random": "Rastgele Arka Plan Müziği",
                        "custom": "Özel Arka Plan Müziği",
                        "sonilo": "Videoya uyumlu otomatik müzik (Sonilo AI)",
                        "elevenlabs": "Videoya uyumlu otomatik müzik (ElevenLabs AI)",
                        "lyria": "Prompttan AI müzik (Google Lyria)",
                    }
                    new_bgm_type = st.selectbox(
                        "Arka Plan Müziği Kaynağı",
                        options=bgm_type_options,
                        index=1,
                        format_func=lambda x: bgm_type_labels.get(x, x),
                        key="new_bgm_type_select"
                    )
                with bgm_col2:
                    vol_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                    new_bgm_volume = st.selectbox(
                        "Arka Plan Müziği Seviyesi",
                        options=vol_options,
                        index=2,
                        format_func=lambda v: f"{int(v * 100)}%",
                        key="new_bgm_vol_select"
                    )
                
                submitted = st.form_submit_button("💾 Kanalı Kaydet", type="primary", use_container_width=True)
                if submitted:
                    if not new_name.strip() or not new_niche.strip():
                        st.error("Lütfen Kanal Adı ve Niş alanlarını doldurun.")
                    else:
                        new_channel = ChannelProfile(
                            name=new_name.strip(),
                            niche=new_niche.strip(),
                            description=new_desc.strip(),
                            system_prompt=new_system_prompt.strip(),
                            voice_name=new_voice.strip(),
                            video_aspect=new_aspect,
                            video_language=new_lang.strip(),
                            video_source=new_source,
                            video_clip_duration=new_clip_dur,
                            image_provider=new_ai_provider,
                            gemini_image_model_name=new_ai_model,
                            ai_image_count_mode=new_ai_count_mode,
                            image_count=0 if new_ai_count_mode == "auto" else new_ai_max_count,
                            image_gen_max_images=new_ai_max_count,
                            font_name=new_font_name,
                            font_size=new_font_size,
                            text_fore_color=new_text_fore_color,
                            stroke_color=new_stroke_color,
                            stroke_width=new_stroke_width,
                            subtitle_enabled=new_sub_enabled,
                            subtitle_position=new_sub_pos,
                            custom_position=new_custom_pos,
                            subtitle_background_enabled=new_sub_bg_enabled,
                            subtitle_background_color=new_sub_bg_color,
                            rounded_subtitle_background=new_rounded_bg,
                            bgm_type=new_bgm_type,
                            bgm_name="random",
                            bgm_volume=new_bgm_volume,
                        )
                        channel_manager.save_channel(new_channel)
                        st.success(f"'{new_channel.name}' kanalı başarıyla eklendi!")
                        st.rerun()

        # Existing Channels List
        if channels:
            st.markdown("### Mevcut Kanallar")
            for ch in channels:
                with st.container(border=True):
                    c1, c2, c3 = st.columns([3, 2, 1])
                    with c1:
                        st.markdown(f"#### 📺 {ch.name}")
                        st.caption(f"**Niş:** {ch.niche} | {ch.description}")
                        st.text(f"Prompt: {ch.system_prompt[:100]}..." if len(ch.system_prompt) > 100 else f"Prompt: {ch.system_prompt}")
                    with c2:
                        st.markdown(f"**Ses:** `{ch.voice_name}`")
                        st.markdown(f"**Format:** `{ch.video_aspect}` | **Dil:** `{ch.video_language}`")
                        source_label = source_labels.get(ch.video_source, ch.video_source)
                        st.markdown(f"**Kaynak:** `{source_label}` | **Font:** `{ch.font_name}`")
                        bgm_display = bgm_type_labels.get(ch.bgm_type, ch.bgm_type)
                        st.markdown(f"**Müzik:** `{bgm_display}` - Seviye: `{int((ch.bgm_volume or 0.2)*100)}%`")
                    with c3:
                        if st.button("🗑️ Sil", key=f"del_ch_{ch.id}", type="secondary", use_container_width=True):
                            channel_manager.delete_channel(ch.id)
                            st.success(f"'{ch.name}' kanalı silindi.")
                            st.rerun()

                    with st.expander(f"✏️ '{ch.name}' Kanalını Düzenle"):
                        with st.form(f"edit_channel_form_{ch.id}"):
                            edit_name = st.text_input("Kanal Adı *", value=ch.name, key=f"edit_name_{ch.id}")
                            edit_niche = st.text_input("Niş / Kategori *", value=ch.niche, key=f"edit_niche_{ch.id}")
                            edit_desc = st.text_area("Açıklama", value=ch.description, key=f"edit_desc_{ch.id}")
                            edit_system_prompt = st.text_area(
                                "Kanala Özel Senaryo İstemi (System Prompt)",
                                value=ch.system_prompt,
                                height=150,
                                key=f"edit_sys_prompt_{ch.id}"
                            )
                            
                            ec1, ec2 = st.columns(2)
                            with ec1:
                                edit_voice_options = get_available_voices(ch.video_language)
                                if ch.voice_name and ch.voice_name not in edit_voice_options:
                                    edit_voice_options = [ch.voice_name] + edit_voice_options
                                edit_voice_idx = edit_voice_options.index(ch.voice_name) if ch.voice_name in edit_voice_options else 0
                                edit_voice = st.selectbox(
                                    "Ses Modeli (Voice Name)",
                                    options=edit_voice_options,
                                    index=edit_voice_idx,
                                    key=f"edit_voice_{ch.id}"
                                )
                                aspect_opts = ["9:16", "16:9", "1:1"]
                                edit_aspect = st.selectbox(
                                    "Video Formatı",
                                    options=aspect_opts,
                                    index=aspect_opts.index(ch.video_aspect) if ch.video_aspect in aspect_opts else 0,
                                    key=f"edit_aspect_{ch.id}"
                                )
                                edit_lang = st.text_input("Video Dili", value=ch.video_language, key=f"edit_lang_{ch.id}")
                            with ec2:
                                edit_source = st.selectbox(
                                    "Video Kaynağı",
                                    options=source_options,
                                    index=source_options.index(ch.video_source) if ch.video_source in source_options else 0,
                                    format_func=lambda x: source_labels.get(x, x),
                                    key=f"edit_source_{ch.id}"
                                )
                                edit_clip_dur = st.number_input("Klip Süresi (sn)", min_value=2, max_value=15, value=int(ch.video_clip_duration or 5), key=f"edit_clip_dur_{ch.id}")
                            
                            st.markdown("#### 🎨 AI Görsel Ayarları")
                            e_ai_col1, e_ai_col2 = st.columns(2)
                            with e_ai_col1:
                                edit_ai_provider = st.selectbox(
                                    "AI Görsel Sağlayıcı",
                                    options=ai_provider_opts,
                                    index=ai_provider_opts.index(ch.image_provider) if ch.image_provider in ai_provider_opts else 0,
                                    format_func=lambda x: ai_provider_labels.get(x, x),
                                    key=f"edit_ai_provider_select_{ch.id}"
                                )
                                edit_ai_model = st.selectbox(
                                    "Görsel Modeli",
                                    options=gemini_model_opts,
                                    index=gemini_model_opts.index(ch.gemini_image_model_name) if ch.gemini_image_model_name in gemini_model_opts else 0,
                                    format_func=lambda x: gemini_model_labels.get(x, x),
                                    key=f"edit_ai_model_select_{ch.id}"
                                )
                            with e_ai_col2:
                                edit_gemini_key = st.text_input(
                                    "Google Gemini API Anahtarı",
                                    value="",
                                    type="password",
                                    key=f"edit_gemini_key_input_{ch.id}"
                                )
                                edit_ai_count_mode = st.selectbox(
                                    "Görsel Sayısı",
                                    options=count_mode_opts,
                                    index=count_mode_opts.index(ch.ai_image_count_mode) if ch.ai_image_count_mode in count_mode_opts else 0,
                                    format_func=lambda x: count_mode_labels.get(x, x),
                                    key=f"edit_ai_count_mode_select_{ch.id}"
                                )
                                edit_ai_max_count = st.number_input(
                                    "Maksimum Görsel",
                                    min_value=1,
                                    max_value=100,
                                    value=int(ch.image_gen_max_images or 14),
                                    key=f"edit_ai_max_count_input_{ch.id}"
                                )

                            st.markdown("#### ✍️ Altyazı Ayarları")
                            edit_sub_enabled = st.checkbox("Altyazıları Etkinleştir", value=bool(ch.subtitle_enabled), key=f"edit_sub_enabled_cb_{ch.id}")
                            
                            font_options = get_available_fonts()
                            edit_font_idx = font_options.index(ch.font_name) if ch.font_name in font_options else 0
                            edit_font_name = st.selectbox("Altyazı Fontu", options=font_options, index=edit_font_idx, key=f"edit_font_name_select_{ch.id}")

                            pos_opts = ["top", "center", "bottom", "custom"]
                            pos_labels = {"top": "Üst", "center": "Orta", "bottom": "Alt", "custom": "Özel"}
                            edit_pos_idx = pos_opts.index(ch.subtitle_position) if ch.subtitle_position in pos_opts else 3
                            edit_sub_pos = st.selectbox("Altyazı Konumu", options=pos_opts, index=edit_pos_idx, format_func=lambda x: pos_labels.get(x, x), key=f"edit_sub_pos_select_{ch.id}")
                            edit_custom_pos = st.number_input("Özel Konum (üstten %)", min_value=0.0, max_value=100.0, value=float(ch.custom_position or 70.0), step=1.0, key=f"edit_custom_pos_input_{ch.id}")

                            ec_f1, ec_f2 = st.columns([0.42, 0.58])
                            with ec_f1:
                                edit_text_fore_color = st.color_picker("Metin", value=ch.text_fore_color if ch.text_fore_color else "#FFFFFF", key=f"edit_font_color_picker_{ch.id}")
                            with ec_f2:
                                edit_font_size = st.slider("Font Boyutu", min_value=30, max_value=100, value=int(ch.font_size or 60), step=1, key=f"edit_font_size_slider_{ch.id}")

                            ec_s1, ec_s2 = st.columns([0.42, 0.58])
                            with ec_s1:
                                edit_stroke_color = st.color_picker("Çerçeve", value=ch.stroke_color if ch.stroke_color else "#000000", key=f"edit_stroke_color_picker_{ch.id}")
                            with ec_s2:
                                edit_stroke_width = st.slider("Çerçeve Kalınlığı", min_value=0.0, max_value=10.0, value=float(ch.stroke_width if ch.stroke_width is not None else 1.50), step=0.1, key=f"edit_stroke_width_slider_{ch.id}")

                            ec_bg1, ec_bg2 = st.columns([0.55, 0.45])
                            with ec_bg1:
                                edit_sub_bg_enabled = st.checkbox("Arka Plan", value=bool(ch.subtitle_background_enabled), key=f"edit_sub_bg_cb_{ch.id}")
                            with ec_bg2:
                                edit_sub_bg_color = st.color_picker("Renk", value=ch.subtitle_background_color if ch.subtitle_background_color else "#FFA500", key=f"edit_sub_bg_color_picker_{ch.id}", disabled=not edit_sub_bg_enabled)

                            edit_rounded_bg = st.checkbox("Yuvarlak Arka Plan", value=bool(ch.rounded_subtitle_background), key=f"edit_rounded_bg_cb_{ch.id}", disabled=not edit_sub_bg_enabled)

                            st.markdown("#### 🎵 Arka Plan Müziği Ayarları")
                            eb1, eb2 = st.columns(2)
                            with eb1:
                                bgm_type_options = ["none", "random", "custom", "sonilo", "elevenlabs", "lyria"]
                                bgm_type_labels = {
                                    "none": "Arka Plan Müziği Yok",
                                    "random": "Rastgele Arka Plan Müziği",
                                    "custom": "Özel Arka Plan Müziği",
                                    "sonilo": "Videoya uyumlu otomatik müzik (Sonilo AI)",
                                    "elevenlabs": "Videoya uyumlu otomatik müzik (ElevenLabs AI)",
                                    "lyria": "Prompttan AI müzik (Google Lyria)",
                                }
                                edit_bgm_type = st.selectbox(
                                    "Arka Plan Müziği Kaynağı",
                                    options=bgm_type_options,
                                    index=bgm_type_options.index(ch.bgm_type) if ch.bgm_type in bgm_type_options else 1,
                                    format_func=lambda x: bgm_type_labels.get(x, x),
                                    key=f"edit_bgm_type_select_{ch.id}"
                                )
                            with eb2:
                                vol_options = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
                                curr_vol = float(ch.bgm_volume if ch.bgm_volume is not None else 0.2)
                                curr_vol_idx = int(round(curr_vol * 10)) if 0 <= curr_vol <= 1.0 else 2
                                edit_bgm_volume = st.selectbox(
                                    "Arka Plan Müziği Seviyesi",
                                    options=vol_options,
                                    index=curr_vol_idx,
                                    format_func=lambda v: f"{int(v * 100)}%",
                                    key=f"edit_bgm_vol_select_{ch.id}"
                                )
                            
                            save_edited = st.form_submit_button("💾 Güncellemeleri Kaydet", type="primary", use_container_width=True)
                            if save_edited:
                                if not edit_name.strip() or not edit_niche.strip():
                                    st.error("Kanal Adı ve Niş alanları boş bırakılamaz.")
                                else:
                                    ch.name = edit_name.strip()
                                    ch.niche = edit_niche.strip()
                                    ch.description = edit_desc.strip()
                                    ch.system_prompt = edit_system_prompt.strip()
                                    ch.voice_name = edit_voice.strip()
                                    ch.video_aspect = edit_aspect
                                    ch.video_language = edit_lang.strip()
                                    ch.video_source = edit_source
                                    ch.video_clip_duration = edit_clip_dur
                                    ch.image_provider = edit_ai_provider
                                    ch.gemini_image_model_name = edit_ai_model
                                    ch.ai_image_count_mode = edit_ai_count_mode
                                    ch.image_count = 0 if edit_ai_count_mode == "auto" else edit_ai_max_count
                                    ch.image_gen_max_images = edit_ai_max_count
                                    ch.font_name = edit_font_name
                                    ch.font_size = edit_font_size
                                    ch.text_fore_color = edit_text_fore_color
                                    ch.stroke_color = edit_stroke_color
                                    ch.stroke_width = edit_stroke_width
                                    ch.subtitle_enabled = edit_sub_enabled
                                    ch.subtitle_position = edit_sub_pos
                                    ch.custom_position = edit_custom_pos
                                    ch.subtitle_background_enabled = edit_sub_bg_enabled
                                    ch.subtitle_background_color = edit_sub_bg_color
                                    ch.rounded_subtitle_background = edit_rounded_bg
                                    ch.bgm_type = edit_bgm_type
                                    ch.bgm_volume = edit_bgm_volume
                                    
                                    channel_manager.save_channel(ch)
                                    st.success(f"'{ch.name}' kanalı başarıyla güncellendi!")
                                    st.rerun()
