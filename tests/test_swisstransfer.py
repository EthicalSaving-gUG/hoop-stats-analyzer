import json
import unittest
from unittest.mock import patch, MagicMock
from hoop_stats.swisstransfer import resolve_swisstransfer_url


class TestSwissTransferResolver(unittest.TestCase):
    def test_invalid_url_returns_none(self):
        url = "https://example.com/other-link"
        stream_url, meta = resolve_swisstransfer_url(url)
        self.assertIsNone(stream_url)
        self.assertIsNone(meta)

    @patch("urllib.request.urlopen")
    def test_successful_resolution(self, mock_urlopen):
        # 1. Mock page HTML response with Inertia props
        fake_html = """
        <html>
        <script data-page="app" type="application/json">
        {
            "props": {
                "transfer": {
                    "id": "trans-123",
                    "expires_at": 1800000000,
                    "files": [
                        {
                            "id": "file-456",
                            "path": "test_game.mp4",
                            "size": 1073741824,
                            "mime_type": "video/mp4"
                        }
                    ]
                }
            }
        }
        </script>
        </html>
        """.encode('utf-8')

        # 2. Mock API response
        fake_api_json = json.dumps({
            "result": "success",
            "data": {
                "url": "https://s3.infomaniak.cloud/bucket/trans-123/file-456?signature=abc"
            }
        }).encode('utf-8')

        # Mock urlopen context managers
        mock_resp1 = MagicMock()
        mock_resp1.read.return_value = fake_html
        mock_resp1.__enter__.return_value = mock_resp1

        mock_resp2 = MagicMock()
        mock_resp2.read.return_value = fake_api_json
        mock_resp2.__enter__.return_value = mock_resp2

        mock_urlopen.side_effect = [mock_resp1, mock_resp2]

        url = "https://www.swisstransfer.com/dl/01a1dbe6-1d8b-73f7-a03c-05fe50a667e3"
        stream_url, meta = resolve_swisstransfer_url(url)

        self.assertIsNotNone(stream_url)
        self.assertTrue(stream_url.startswith("https://s3.infomaniak.cloud"))
        self.assertIsNotNone(meta)
        self.assertEqual(meta["link_id"], "01a1dbe6-1d8b-73f7-a03c-05fe50a667e3")
        self.assertEqual(meta["filename"], "test_game.mp4")
        self.assertEqual(meta["size_gb"], 1.0)
        self.assertEqual(meta["mime_type"], "video/mp4")


if __name__ == "__main__":
    unittest.main()
