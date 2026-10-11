"""Read official ZIP metadata without downloading its full audio payload."""

from __future__ import annotations

import io
import urllib.request
import zipfile

from acquire import BASE, ROOT, read, write


class RemoteZip(io.RawIOBase):
    def __init__(self, url):
        self.url = url
        with urllib.request.urlopen(
            urllib.request.Request(url, method="HEAD"), timeout=30
        ) as response:
            self.size = int(response.headers["Content-Length"])
            self.etag = response.headers["ETag"]
        self.position = 0

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = (
            offset
            if whence == 0
            else self.position + offset
            if whence == 1
            else self.size + offset
        )
        return self.position

    def tell(self):
        return self.position

    def read(self, count=-1):
        count = (
            self.size - self.position
            if count < 0
            else min(count, self.size - self.position)
        )
        if count == 0:
            return b""
        if not 0 < count <= 32 * 1024 * 1024:
            raise ValueError("ZIP metadata range exceeds the inspection limit")
        start = self.position
        request = urllib.request.Request(
            self.url,
            headers={
                "Range": f"bytes={start}-{start + count - 1}",
                "If-Match": self.etag,
            },
        )
        with urllib.request.urlopen(request, timeout=60) as response:
            assert response.status == 206
            assert response.headers["Content-Range"].startswith(f"bytes {start}-")
            data = response.read(count + 1)
        assert len(data) == count
        self.position += count
        return data


def main():
    config = read(BASE / "config.json")
    output = ROOT / config["source_directory"]
    remote = RemoteZip(config["source_url"])
    with zipfile.ZipFile(remote) as archive:
        entries = [
            {
                "path": v.filename,
                "size": v.file_size,
                "compressed_size": v.compress_size,
                "crc": v.CRC,
                "offset": v.header_offset,
            }
            for v in archive.infolist()
        ]
        index = {
            "url": config["source_url"],
            "size": remote.size,
            "etag": remote.etag,
            "files": entries,
        }
        path = output / "jtube-archive-index.json"
        if path.exists():
            assert read(path) == index, (
                "Archive metadata changed; use a new source directory"
            )
        else:
            write(path, index)
        for name in (
            "README.md",
            "meta/Japanese.lst",
            "meta/Original.lst",
            "meta/test_2sec.lst",
            "meta/test_5sec.lst",
        ):
            info = archive.getinfo("JTubeSpeech-ASV/" + name)
            assert info.file_size < 3_000_000
            data = archive.read(info)
            path = output / name.split("/")[-1]
            if path.exists():
                assert path.read_bytes() == data, "Source metadata changed"
            else:
                path.write_bytes(data)
        print(f"Verified {len(entries)} archive entries and five metadata files")


if __name__ == "__main__":
    main()
