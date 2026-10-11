import base64
import http.client
import json
import sys
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import server
import test_pilot


class HTTPTests(unittest.TestCase):
    def setUp(self):
        test_pilot.PilotTests.setUp(self)
        self.http = ThreadingHTTPServer(
            ("127.0.0.1", 0), server.handler(self.db, "secret", 0)
        )
        self.http.RequestHandlerClass = server.handler(
            self.db, "secret", self.http.server_port
        )
        self.thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        self.thread.start()

    def tearDown(self):
        self.http.shutdown()
        self.http.server_close()
        self.thread.join()
        test_pilot.PilotTests.tearDown(self)

    def request_http(self, path, body=None, **headers):
        client = http.client.HTTPConnection(
            "127.0.0.1", self.http.server_port, timeout=3
        )
        client.request(
            "GET" if body is None else "POST",
            path,
            body=None if body is None else json.dumps(body),
            headers=headers,
        )
        response = client.getresponse()
        status = response.status
        content = response.read()
        client.close()
        return status, content

    def test_http_write_and_read(self):
        status, body = self.request_http(
            "/api/episode", self.request, **{"X-Phonia-Token": "secret"}
        )
        self.assertEqual(status, 200)
        episode = json.loads(body)
        status, body = self.request_http(
            "/api/clip",
            {
                "episode_id": episode["id"],
                "prompt_id": episode["prompt"]["id"],
                "origin": "microphone",
                "wav_base64": base64.b64encode(test_pilot.wav()).decode(),
            },
            **{"X-Phonia-Token": "secret"},
        )
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["status"], "complete")
        status, body = self.request_http("/api/episode?id=" + episode["id"])
        self.assertEqual(status, 200)
        self.assertIsNone(json.loads(body)["threshold"])

    def test_host_origin_token_and_asset_boundary(self):
        self.assertEqual(self.request_http("/api/info", Host="evil.example")[0], 403)
        self.assertEqual(self.request_http("/api/episode", self.request)[0], 403)
        self.assertEqual(
            self.request_http(
                "/api/episode",
                self.request,
                **{"X-Phonia-Token": "secret", "Origin": "https://evil.example"},
            )[0],
            403,
        )
        for path in ("/", "/app.js", "/audio.js", "/capture.js", "/style.css"):
            self.assertEqual(self.request_http(path)[0], 200)
        self.assertEqual(self.request_http("/../protocol.json")[0], 404)


if __name__ == "__main__":
    unittest.main()
