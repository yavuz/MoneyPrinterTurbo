import os
import shutil
import tempfile
import unittest

from app.services.channel_manager import ChannelManager, ChannelProfile
from app.services import trend_service


class TestChannelHub(unittest.TestCase):

    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.json_path = os.path.join(self.test_dir, "test_channels.json")
        self.manager = ChannelManager(storage_path=self.json_path)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_channel_crud(self):
        # 1. Create
        ch = ChannelProfile(
            name="Tarih Parkı",
            niche="Tarih",
            description="Tarihi olaylar ve anlatılar",
            system_prompt="Sen gizemli bir tarih anlatıcısısın.",
            voice_name="tr-TR-AhmetNeural",
            video_aspect="9:16",
            video_language="tr"
        )
        saved = self.manager.save_channel(ch)
        self.assertEqual(saved.name, "Tarih Parkı")

        # 2. Read All
        all_channels = self.manager.get_all_channels()
        self.assertEqual(len(all_channels), 1)
        self.assertEqual(all_channels[0].id, ch.id)

        # 3. Read Single
        fetched = self.manager.get_channel(ch.id)
        self.assertIsNotNone(fetched)
        self.assertEqual(fetched.niche, "Tarih")

        # 4. Update
        ch.name = "Tarih & Gizem Parkı"
        self.manager.save_channel(ch)
        updated = self.manager.get_channel(ch.id)
        self.assertEqual(updated.name, "Tarih & Gizem Parkı")

        # 5. Delete
        deleted = self.manager.delete_channel(ch.id)
        self.assertTrue(deleted)
        self.assertEqual(len(self.manager.get_all_channels()), 0)


if __name__ == "__main__":
    unittest.main()
