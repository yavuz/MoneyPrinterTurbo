import json
import os
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional
from loguru import logger
from pydantic import BaseModel, Field

from app.utils import utils

CHANNELS_FILE_PATH = os.path.join(utils.root_dir(), "storage", "channels.json")


class ChannelProfile(BaseModel):
    """
    Channel profile containing metadata, prompt rules, and preset video settings.
    """
    id: str = Field(default_factory=lambda: f"ch_{uuid.uuid4().hex[:8]}")
    name: str
    niche: str = ""
    description: str = ""
    system_prompt: str = ""
    voice_name: str = "tr-TR-AhmetNeural"
    voice_volume: float = 1.0
    video_aspect: str = "9:16"
    video_language: str = "tr"
    video_source: str = "pexels"
    video_concat_mode: str = "random"
    video_transition_mode: str = "none"
    video_clip_duration: int = 5
    bgm_type: str = "random"
    bgm_name: str = "random"
    bgm_volume: float = 0.2
    image_provider: str = "gemini"
    gemini_image_model_name: str = "gemini-3-pro-image"
    ai_image_count_mode: str = "auto"
    image_count: int = 0
    image_gen_max_images: int = 14
    font_name: str = "MicrosoftYaHeiBold.ttc"
    font_size: int = 60
    text_fore_color: str = "#FFFFFF"
    stroke_color: str = "#000000"
    stroke_width: float = 1.5
    subtitle_enabled: bool = True
    subtitle_position: str = "custom"
    custom_position: float = 70.0
    subtitle_background_enabled: bool = False
    subtitle_background_color: str = "#FFA500"
    rounded_subtitle_background: bool = False
    target_platforms: List[str] = Field(default_factory=lambda: ["YouTube Shorts", "TikTok", "Instagram Reels"])
    created_at: str = Field(default_factory=lambda: datetime.now().isoformat())
    updated_at: str = Field(default_factory=lambda: datetime.now().isoformat())


class ChannelManager:
    """
    Manager class to handle CRUD operations for Channel Profiles.
    """

    def __init__(self, storage_path: str = CHANNELS_FILE_PATH):
        self.storage_path = storage_path
        self._ensure_storage_exists()

    def _ensure_storage_exists(self) -> None:
        """
        Ensure the storage directory and json file exist.
        """
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        if not os.path.exists(self.storage_path):
            self._save_all([])

    def _load_raw(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load channels from {self.storage_path}: {e}")
            return []

    def _save_all(self, channels: List[Dict[str, Any]]) -> bool:
        try:
            with open(self.storage_path, "w", encoding="utf-8") as f:
                json.dump(channels, f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save channels to {self.storage_path}: {e}")
            return False

    def get_all_channels(self) -> List[ChannelProfile]:
        """
        Get all channel profiles.
        """
        raw_list = self._load_raw()
        channels = []
        for item in raw_list:
            try:
                channels.append(ChannelProfile(**item))
            except Exception as e:
                logger.warning(f"Error parsing channel profile {item.get('id')}: {e}")
        return channels

    def get_channel(self, channel_id: str) -> Optional[ChannelProfile]:
        """
        Get a specific channel profile by ID.
        """
        for ch in self.get_all_channels():
            if ch.id == channel_id:
                return ch
        return None

    def save_channel(self, channel: ChannelProfile) -> ChannelProfile:
        """
        Create or update a channel profile.
        """
        channels = self.get_all_channels()
        channel.updated_at = datetime.now().isoformat()
        
        found = False
        for i, ch in enumerate(channels):
            if ch.id == channel.id:
                channels[i] = channel
                found = True
                break
        
        if not found:
            channels.append(channel)
            
        self._save_all([ch.model_dump() for ch in channels])
        logger.info(f"Saved channel profile: {channel.name} ({channel.id})")
        return channel

    def delete_channel(self, channel_id: str) -> bool:
        """
        Delete a channel profile by ID.
        """
        channels = self.get_all_channels()
        initial_len = len(channels)
        channels = [ch for ch in channels if ch.id != channel_id]
        if len(channels) < initial_len:
            self._save_all([ch.model_dump() for ch in channels])
            logger.info(f"Deleted channel profile ID: {channel_id}")
            return True
        return False


# Singleton instance
channel_manager = ChannelManager()
