import io
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import acquire


class AcquisitionTests(unittest.TestCase):
    def archive(self):
        buffer = io.BytesIO()
        name = "JTubeSpeech-ASV/train/AAA/AAA_example.mp3"
        with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
            archive.writestr(name, b"example audio bytes")
        with zipfile.ZipFile(buffer) as archive:
            info = archive.getinfo(name)
        return buffer.getvalue(), {
            "path": name,
            "offset": info.header_offset,
            "compressed_size": info.compress_size,
            "size": info.file_size,
            "crc": info.CRC,
            "split": "train",
            "speaker_id": "jtube_AAA",
        }

    def fetch(self, payload, row, status=206):
        response = io.BytesIO(payload)
        response.status = status
        response.headers = {
            "Content-Range": f"bytes 0-{len(payload) - 1}/{len(payload)}"
        }
        with patch.object(acquire.urllib.request, "urlopen", return_value=response):
            return acquire.fetch_member(
                {"source_url": "https://example.com/data.zip", "run_directory": "run"},
                {"archive_size": len(payload), "archive_etag": '"fixed"'},
                row,
            )

    def test_member_round_trip_checks_crc_and_saved_digest(self):
        payload, row = self.archive()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(acquire, "ROOT", Path(directory)),
        ):
            result = self.fetch(payload, row)
            self.assertEqual(
                (Path(directory) / result["local_path"]).read_bytes(),
                b"example audio bytes",
            )
            self.assertEqual(
                result["sha256"], acquire.sha(Path(directory) / result["local_path"])
            )

    def test_corrupt_crc_is_rejected(self):
        payload, row = self.archive()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(acquire, "ROOT", Path(directory)),
            self.assertRaises(AssertionError),
        ):
            self.fetch(payload, {**row, "crc": 0})

    def test_full_response_instead_of_range_is_rejected(self):
        payload, row = self.archive()
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.object(acquire, "ROOT", Path(directory)),
            self.assertRaises(AssertionError),
        ):
            self.fetch(payload, row, status=200)


if __name__ == "__main__":
    unittest.main()
